"""
Smart Booking Platform — Hybrid Search Prototype
================================================

Two endpoints:

  POST /api/extract
      Pure NLP filter-stripping. Returns dates/times/prices/capacity + the
      clean semantic query. (The original contract — unchanged.)

  POST /api/search   ← PROTOTYPE
      The full hybrid flow proven end-to-end:
          raw query
            -> spaCy extract (capacity / price / date + clean_query)
            -> embed(clean_query) via OpenAI
            -> Qdrant vector search WITH a payload filter (capacity >= N)
            -> matched assets
      This is the logic we port back into OptimoGov's SmartAssetSearchService
      once it's proven here.

Run with:
    uvicorn main:app --reload --port 8000
Swagger UI:
    http://127.0.0.1:8000/docs
"""

from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import nlp_extractor
from vector_search import VectorSearchClient

# Load QDRANT_* / OPENAI_* from backend/.env before anything reads them.
load_dotenv()

app = FastAPI(
    title="Smart Booking Hybrid Search",
    description="spaCy filter-stripping + Qdrant vector search prototype.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare root (e.g. Docker Desktop's port link) to Swagger UI."""
    return RedirectResponse(url="/docs")

# Lazily-initialised so /api/extract still works even if Qdrant/OpenAI env
# vars are missing (e.g. someone only wants the NLP layer).
_vector_client: Optional[VectorSearchClient] = None


def get_vector_client() -> VectorSearchClient:
    global _vector_client
    if _vector_client is None:
        try:
            _vector_client = VectorSearchClient()
        except KeyError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Vector search not configured: missing env var {exc}. "
                "Copy backend/.env.example to backend/.env and fill it in.",
            ) from exc
    return _vector_client


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    dates: List[str]
    times: List[str]
    prices: List[str]
    capacity: Optional[str]
    clean_query: str


class SearchRequest(BaseModel):
    text: str
    top_k: int = 10


class SearchResponse(BaseModel):
    original_query: str
    clean_query: str
    filters: Dict[str, Any]
    applied_qdrant_filter: Optional[Dict[str, Any]]
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    """Strip filters from a query and return metadata + clean semantic core."""
    ex = nlp_extractor.extract(request.text)
    return ExtractResponse(
        dates=ex.dates,
        times=ex.times,
        prices=ex.prices,
        capacity=ex.capacity,
        clean_query=ex.clean_query,
    )


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """PROTOTYPE: full hybrid search — spaCy strip → embed → filtered Qdrant."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    client = get_vector_client()

    # 1) Strip filters + get the clean semantic query.
    ex = nlp_extractor.extract(request.text)

    # 2) Build the Qdrant payload filter from extracted values.
    qdrant_filter = client.build_filter(
        capacity_value=ex.capacity_value,
        price_ceiling=ex.price_ceiling,
    )

    # 3) Embed the clean query and run the filtered vector search.
    try:
        async with httpx.AsyncClient() as http:
            vector = await client.embed(ex.clean_query or request.text, http)
            results = await client.search(
                vector=vector,
                client=http,
                qdrant_filter=qdrant_filter,
                top_k=request.top_k,
            )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream error from {exc.request.url}: "
            f"{exc.response.status_code} {exc.response.text[:300]}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach {exc.request.url}: {exc}",
        ) from exc

    return SearchResponse(
        original_query=request.text,
        clean_query=ex.clean_query,
        filters={
            "capacity": ex.capacity_value,
            "price_ceiling": ex.price_ceiling,
            "dates": ex.dates,
            "times": ex.times,
        },
        applied_qdrant_filter=qdrant_filter,
        results=results,
    )


@app.on_event("startup")
async def ensure_indexes() -> None:
    """Best-effort: ensure the Qdrant `capacity` payload index exists.

    Range-filtering on `capacity` requires this index. We create it once at
    startup so the first /api/search doesn't 400. Failures here are non-fatal —
    /api/extract (pure NLP) must keep working even if Qdrant/OpenAI aren't set.
    """
    try:
        client = get_vector_client()
    except HTTPException:
        # Vector search not configured (missing env). Skip silently.
        return

    try:
        async with httpx.AsyncClient() as http:
            await client.ensure_capacity_index(http)
    except Exception as exc:  # noqa: BLE001 - startup must never crash on this
        # Log-only: the app still serves; /api/search will surface a clear 502
        # if the index is genuinely missing.
        print(f"[startup] could not ensure capacity index: {exc!r}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "en_core_web_sm"}
