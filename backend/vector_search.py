"""
Vector search client (prototype).
=================================

Mirrors what OptimoGov.Services `SmartAssetSearchService` does, but in Python so
we can prove the hybrid flow end-to-end before porting it back to C#:

    clean_query  --embed(OpenAI)-->  vector
    vector + payload filter  --search-->  Qdrant  -->  matched assets

The key difference vs. the current .NET code: we attach a Qdrant `filter` built
from the spaCy-extracted capacity (and price, when the payload allows it), so
non-matching assets are excluded before scoring — not just embedded away.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class VectorSearchClient:
    def __init__(self) -> None:
        self.qdrant_endpoint = os.environ["QDRANT_ENDPOINT"].rstrip("/")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
        self.collection = os.environ.get("QDRANT_COLLECTION", "space_v2")
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

    # -- OpenAI embedding -------------------------------------------------

    async def embed(self, text: str, client: httpx.AsyncClient) -> List[float]:
        resp = await client.post(
            OPENAI_EMBED_URL,
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    # -- Qdrant payload index --------------------------------------------

    async def ensure_capacity_index(self, client: httpx.AsyncClient) -> None:
        """Ensure a payload index exists on `capacity` so range filters work.

        Qdrant refuses `range` filters on un-indexed payload fields (HTTP 400:
        "Index required but not found"). The OptimoGov C# sync writes `capacity`
        as a number but never indexes it, so we create the index here.

        Idempotent: creating an index that already exists returns 200, so this
        is safe to call on every startup.
        """
        headers = {"Content-Type": "application/json"}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key

        resp = await client.put(
            f"{self.qdrant_endpoint}/collections/{self.collection}/index",
            headers=headers,
            json={"field_name": "capacity", "field_schema": "integer"},
            timeout=30.0,
        )
        resp.raise_for_status()

    # -- Qdrant filter construction --------------------------------------

    @staticmethod
    def build_filter(
        capacity_value: Optional[int],
        price_ceiling: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Build a Qdrant `filter` object from extracted values.

        capacity -> the payload field `capacity` is a number (asset's max
        capacity), so we require it to be >= the requested headcount.

        NOTE: `price` in the current OptimoGov payload is a formatted STRING
        ("$50 to $200"), which Qdrant cannot range-filter. So price is left out
        of the hard filter here; see README for the payload change needed to
        enable it.
        """
        must: List[Dict[str, Any]] = []

        if capacity_value is not None:
            # Asset must seat at least this many people.
            must.append({"key": "capacity", "range": {"gte": capacity_value}})

        # price_ceiling intentionally not applied — payload price is a string.
        _ = price_ceiling

        return {"must": must} if must else None

    # -- Qdrant search ----------------------------------------------------

    async def search(
        self,
        vector: List[float],
        client: httpx.AsyncClient,
        qdrant_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        if qdrant_filter:
            body["filter"] = qdrant_filter

        headers = {"Content-Type": "application/json"}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key

        resp = await client.post(
            f"{self.qdrant_endpoint}/collections/{self.collection}/points/search",
            headers=headers,
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json().get("result", [])

        hits: List[Dict[str, Any]] = []
        for hit in result:
            payload = hit.get("payload", {}) or {}
            hits.append(
                {
                    "score": hit.get("score"),
                    "assetid": payload.get("assetid") or payload.get("asset_id"),
                    "name": payload.get("name"),
                    "venue": payload.get("venue"),
                    "facilitytype": payload.get("facilitytype"),
                    "capacity": payload.get("capacity"),
                    "price": payload.get("price"),
                }
            )
        return hits
