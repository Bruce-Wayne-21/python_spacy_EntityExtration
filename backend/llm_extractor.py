"""
LLM-based extraction (prototype).
=================================

An alternative to `nlp_extractor` (spaCy) that uses an OpenAI chat model to pull
structured booking filters out of a raw natural-language query.

Entities extracted:
    min_amount / max_amount   capacity range  -> Qdrant `capacity` field
    min_price  / max_price    cost range      -> (NOT yet filtered; see note)
    date                      requested date
    time                      requested time

The model is instructed (via a system prompt) to return strict JSON so we can
parse it deterministically. Reuses the OPENAI_API_KEY already used for
embeddings — no new dependency.

NOTE on price: the Qdrant payload stores `price` as a formatted STRING
("$50 to $200"), which Qdrant cannot range-filter. So min_price / max_price are
extracted and returned, but NOT applied as a hard filter — same limitation the
spaCy `/api/search` flow already has.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENAI_EXTRACT_MODEL", "gpt-4o-mini")

# The system prompt: defines the entities and forces strict JSON output.
SYSTEM_PROMPT = """\
You are an entity-extraction engine for a venue-booking search platform.

Given a user's natural-language query, extract these fields and return them as a
single strict JSON object. Use null for any field that is not present.

Fields:
  min_amount  (integer): minimum capacity / number of people the venue must
                         accommodate. From phrases like "at least 50 people",
                         "50+ guests", "for 50 people".
  max_amount  (integer): maximum capacity. From "up to 200 people",
                         "no more than 100 guests".
  min_price   (number):  minimum price / budget floor. From "over $100",
                         "at least 50 dollars".
  max_price   (number):  maximum price / budget ceiling. From "under $200",
                         "below 150", "cheaper than $300", "up to $200".
  date        (string):  the requested date, kept as the user wrote it
                         (e.g. "next Friday", "December 5", "2025-06-01").
  time        (string):  the requested time, as written (e.g. "2pm", "morning").
  clean_query (string):  the semantic core of the request with the numeric
                         filters, dates, times and prices removed. This is what
                         we embed for vector search. Keep only the descriptive
                         venue words (e.g. "waterfront conference room").

Rules:
- If a single capacity like "for 50 people" is given with no range wording,
  set min_amount to that number and leave max_amount null.
- Numbers only for amounts/prices — strip currency symbols and commas.
- Return ONLY the JSON object, no prose, no markdown fences.
"""


@dataclass
class LLMExtraction:
    """Structured result of the LLM extraction."""

    min_amount: Optional[int] = None
    max_amount: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    date: Optional[str] = None
    time: Optional[str] = None
    clean_query: str = ""
    raw: Optional[Dict[str, Any]] = None  # the model's parsed JSON, for debugging


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).replace(",", "").replace("$", "")))
    except (ValueError, TypeError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def extract(text: str, client: httpx.AsyncClient) -> LLMExtraction:
    """Call the LLM and parse its JSON into an LLMExtraction.

    Raises httpx.HTTPStatusError on an OpenAI error, and ValueError if the model
    returns something that isn't valid JSON.
    """
    api_key = os.environ["OPENAI_API_KEY"]

    resp = await client.post(
        OPENAI_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": DEFAULT_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        },
        timeout=30.0,
    )
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    try:
        data: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {content[:300]}") from exc

    return LLMExtraction(
        min_amount=_coerce_int(data.get("min_amount")),
        max_amount=_coerce_int(data.get("max_amount")),
        min_price=_coerce_float(data.get("min_price")),
        max_price=_coerce_float(data.get("max_price")),
        date=_coerce_str(data.get("date")),
        time=_coerce_str(data.get("time")),
        clean_query=_coerce_str(data.get("clean_query")) or text,
        raw=data,
    )


def build_filter(
    min_amount: Optional[int],
    max_amount: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Build a Qdrant `filter` from the extracted capacity range.

    The payload field `capacity` is a number (the venue's capacity), so:
        min_amount -> capacity >= min_amount
        max_amount -> capacity <= max_amount

    Price is intentionally NOT filtered here: the payload stores price as a
    formatted string which Qdrant cannot range-filter.
    """
    conditions: Dict[str, Any] = {}
    if min_amount is not None:
        conditions["gte"] = min_amount
    if max_amount is not None:
        conditions["lte"] = max_amount

    if not conditions:
        return None

    must: List[Dict[str, Any]] = [{"key": "capacity", "range": conditions}]
    return {"must": must}
