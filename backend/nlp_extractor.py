"""
NLP extraction logic (shared).
==============================

Pulls structured booking filters out of a raw natural-language query using
spaCy, and returns the remaining semantic "core" (`clean_query`).

This module is deliberately framework-free so it can be reused by:
  * the `/api/extract` endpoint (returns raw strings for a frontend), and
  * the `/api/search` prototype (needs numeric values to filter Qdrant).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import spacy
from spacy.matcher import Matcher

# ---------------------------------------------------------------------------
# Model + Matcher (loaded once)
# ---------------------------------------------------------------------------

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "The spaCy model 'en_core_web_sm' is not installed. "
        "Run: python -m spacy download en_core_web_sm"
    ) from exc

_matcher = Matcher(_nlp.vocab)
_matcher.add(
    "CAPACITY_RULE",
    [[
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["people", "pax", "guests", "guest", "persons", "person"]}},
    ]],
)

_METADATA_ENT_LABELS = {"DATE", "TIME", "MONEY"}
_FILLER_WORDS = {"for", "the", "a", "an", "of", "with", "and", "to", "in", "on"}

# Words that signal an upper price bound, e.g. "under $200", "below 150".
_PRICE_CEILING_HINTS = ("under", "below", "less than", "max", "up to", "cheaper than")


@dataclass
class Extraction:
    """Everything pulled from a raw query."""

    dates: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    prices: List[str] = field(default_factory=list)
    capacity: Optional[str] = None          # raw string, e.g. "50"
    capacity_value: Optional[int] = None    # parsed int, e.g. 50 (for filtering)
    price_ceiling: Optional[float] = None    # numeric max, e.g. 200.0 (for filtering)
    clean_query: str = ""


def _words_to_int(token_text: str) -> Optional[int]:
    """Best-effort parse of a numeric token ('50', '15') to int."""
    try:
        return int(float(token_text.replace(",", "")))
    except ValueError:
        return None


def _extract_price_ceiling(text: str, money_spans: List[str]) -> Optional[float]:
    """If the query says 'under $200' / 'below 150', return 200.0 / 150.0.

    Only treats a MONEY value as a ceiling when a hint word precedes it, so a
    plain price mention isn't mistaken for a hard cap.
    """
    lowered = text.lower()
    for money in money_spans:
        # Strip currency symbols/commas to get a number.
        numeric = re.sub(r"[^\d.]", "", money)
        if not numeric:
            continue
        try:
            value = float(numeric)
        except ValueError:
            continue

        idx = lowered.find(money.lower())
        window = lowered[max(0, idx - 20): idx] if idx != -1 else lowered
        if any(hint in window for hint in _PRICE_CEILING_HINTS):
            return value
    return None


def _remove_spans(text: str, spans: List[tuple]) -> str:
    """Remove (start, end) char spans from text, merging overlaps."""
    if not spans:
        return text

    ordered = sorted(spans, key=lambda s: s[0])
    merged: List[list] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result = text
    for start, end in reversed(merged):
        result = result[:start] + " " + result[end:]
    return result


def _normalise_clean_query(tokens: List[str]) -> str:
    keywords = [
        tok.strip().lower()
        for tok in tokens
        if tok.strip() and tok.strip().lower() not in _FILLER_WORDS
    ]
    return " ".join(keywords)


def extract(text: str) -> Extraction:
    """Run the full extraction pipeline over `text`."""
    doc = _nlp(text)
    result = Extraction()
    strip_spans: List[tuple] = []

    for ent in doc.ents:
        if ent.label_ == "DATE":
            result.dates.append(ent.text)
            strip_spans.append((ent.start_char, ent.end_char))
        elif ent.label_ == "TIME":
            result.times.append(ent.text)
            strip_spans.append((ent.start_char, ent.end_char))
        elif ent.label_ == "MONEY":
            result.prices.append(ent.text)
            strip_spans.append((ent.start_char, ent.end_char))

    # Capacity via the custom Matcher — keep the raw number and parsed int.
    for _match_id, start, end in _matcher(doc):
        span = doc[start:end]
        number_token = span[0]
        if result.capacity is None:
            result.capacity = number_token.text
            result.capacity_value = _words_to_int(number_token.text)
        strip_spans.append((span.start_char, span.end_char))

    result.price_ceiling = _extract_price_ceiling(text, result.prices)

    # Build the clean query from what's left.
    stripped = _remove_spans(text, strip_spans)
    residual = _nlp(stripped)
    residual_tokens = [
        tok.text
        for tok in residual
        if not tok.is_stop
        and not tok.is_punct
        and not tok.is_space
        and not tok.is_currency
    ]
    result.clean_query = _normalise_clean_query(residual_tokens)
    return result
