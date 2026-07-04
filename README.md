# Smart Booking — Hybrid Search Backend

A Python (FastAPI + spaCy) preprocessing layer for a venue & facility booking
platform. It takes a raw natural-language booking query, **strips the structured
filters** out of it (capacity, price, dates, times), and runs a **vector search**
over the remaining semantic core against a **Qdrant** collection — the same
collection OptimoGov's asset index (`space_v2`) is synced to.

```
"quiet meeting room for 20 people with wifi"
        │
        ▼  spaCy strips filters
capacity=20   clean_query="quiet meeting room wifi"
        │
        ▼  embed(clean_query) via OpenAI  →  vector
        │
        ▼  Qdrant search WITH payload filter (capacity ≥ 20)
        ▼
matched assets (all capacity ≥ 20)
```

Why this matters: OptimoGov's own `SmartAssetSearchService` embeds the **entire**
raw query (so "20 people" pollutes the semantic vector) and applies **no** hard
capacity filter. This service proves the better approach — strip first, then
filter — end-to-end in Python before porting it back to C#.

---

## Project structure

```
smart-booking-platform/
├── docker-compose.yml          # one-command run (loads backend/.env)
├── README.md
└── backend/
    ├── main.py                 # FastAPI app: routes, CORS, startup index
    ├── nlp_extractor.py        # spaCy extraction (framework-free, reusable)
    ├── vector_search.py        # OpenAI embed + Qdrant filter/search client
    ├── requirements.txt
    ├── Dockerfile              # Python 3.13-slim, bakes in the spaCy model
    ├── .dockerignore
    ├── .env                    # real secrets (gitignored — never commit)
    └── .env.example            # template
```

---

## Prerequisites

- **Docker Desktop** (recommended path), **or**
- **Python 3.13** with the `py` launcher (local path)
- A **Qdrant** endpoint + API key, and an **OpenAI** API key

---

## Configuration

All secrets are read from `backend/.env`. Copy the template and fill it in:

```bash
cp backend/.env.example backend/.env
```

```dotenv
QDRANT_ENDPOINT=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=space_v2

OPENAI_API_KEY=sk-your-openai-key
```

`.env` is gitignored. **Only Docker Compose (or the local run) loads it** — see
the warning under *Running with Docker*.

---

## Running with Docker (recommended)

From the **`smart-booking-platform/`** folder:

```bash
docker compose up -d --build   # build + start (background)
docker compose logs -f         # watch logs
docker compose down            # stop + remove
docker compose restart         # restart
```

- **Swagger UI:** http://localhost:8080/docs
  (the root `/` auto-redirects here, so Docker Desktop's port link works too)
- Host port **8080** → container **8000**.

> ⚠️ **Always start it with `docker compose`.** Your `.env` is loaded **only**
> through Compose. Do **not** use Docker Desktop's ▶️ *Run* button on the
> **image**, and do **not** run `docker run smart-booking-backend` directly —
> those start a container **without** `.env`, and every request then fails with:
>
> ```json
> { "detail": "Vector search not configured: missing env var 'QDRANT_ENDPOINT'. ..." }
> ```
>
> To start/stop from the Docker Desktop UI, only press start/stop on the
> existing **`smart-booking-backend`** container — never spawn a new one from
> the Images tab.

### Why port 8080 (not 8000)?

On some Windows/WSL2 setups a stale `uvicorn --reload` process can leave a
zombie socket squatting on host port 8000, which intermittently hijacks
requests. Mapping to **8080** sidesteps that. If you've rebooted and want 8000
back, change the `ports:` line in `docker-compose.yml` to `"8000:8000"`.

---

## Running locally (without Docker)

```bash
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm   # model isn't in requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- **Swagger UI:** http://127.0.0.1:8000/docs
- Stop with **Ctrl+C**. To restart, re-run the last command.
- Add `--reload` for auto-restart on code edits (note: this is what can leave the
  zombie socket on 8000 — kill the process cleanly when done).

---

## API

Interactive docs at `/docs`. Endpoints:

### `POST /api/extract` — NLP only

Strips filters and returns metadata + the clean semantic query. No vector DB
needed (works even if Qdrant/OpenAI aren't configured).

**Request**
```json
{ "text": "Rooftop bar for 50 pax next Friday evening under £2000" }
```

**Response**
```json
{
  "dates": ["next Friday"],
  "times": ["evening"],
  "prices": ["£2000"],
  "capacity": "50",
  "clean_query": "rooftop bar"
}
```

### `POST /api/search` — full hybrid search

Strips filters → embeds `clean_query` → runs a filtered Qdrant search. Requires
`QDRANT_*` and `OPENAI_*` env vars (else returns **503**).

**Request**
```json
{ "text": "quiet meeting room for 20 people with wifi", "top_k": 3 }
```

**Response** (abridged)
```json
{
  "original_query": "quiet meeting room for 20 people with wifi",
  "clean_query": "quiet meeting room wifi",
  "filters": { "capacity": 20, "price_ceiling": null, "dates": [], "times": [] },
  "applied_qdrant_filter": { "must": [ { "key": "capacity", "range": { "gte": 20 } } ] },
  "results": [
    { "score": 0.47, "assetid": "...", "name": "...", "venue": "...",
      "facilitytype": "...", "capacity": 30, "price": "$50 to $200" }
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "model": "en_core_web_sm" }
```

### `GET /`
Redirects to `/docs`.

---

## How it works

### 1. Filter extraction (`nlp_extractor.py`)
- **Dates / times / prices** — spaCy's `DATE`, `TIME`, `MONEY` named entities.
- **Capacity** — a custom spaCy `Matcher` rule (`CAPACITY_RULE`): a number
  followed by `people` / `pax` / `guests` / `persons`. Only the raw number is
  kept (e.g. `"50 pax"` → `50`).
- **Price ceiling** — a `MONEY` value is treated as an upper bound only when
  preceded by a hint word (`under`, `below`, `max`, `up to`, …).
- **clean_query** — the original text with all matched spans removed, then
  stop-words / punctuation / currency symbols dropped and compressed to
  keywords.

### 2. Vector search (`vector_search.py`)
- Embeds `clean_query` with OpenAI `text-embedding-3-small` (1536-dim) — matches
  what OptimoGov's `QdrantAssetService` uses to build the index.
- Builds a Qdrant payload **filter** from the extracted `capacity`
  (`capacity ≥ N`) so non-matching assets are excluded before scoring.
- On startup, ensures an **integer payload index on `capacity`** exists
  (Qdrant rejects range filters on un-indexed fields — see *Known limitations*).

---

## Known limitations

- **Price filtering is not applied.** OptimoGov stores `price` as a formatted
  string (`"$1500 to $2000"`), which Qdrant can't range-filter. `price_ceiling`
  is extracted but not enforced. Enabling it needs a numeric `price_min` /
  `price_max` field added to the C# sync payload.
- **`@app.on_event("startup")` is deprecated** in current FastAPI (works, but
  warns). Migrate to the `lifespan` handler when productionising.
- **spaCy TIME/MONEY merges.** Phrasings like `"3pm around 500 dollars"` can be
  captured as a single `TIME` entity, so the money isn't tagged. A model
  limitation, not a code bug.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `503 "missing env var 'QDRANT_ENDPOINT'"` | Container started without `.env` (raw `docker run` or Docker Desktop *Run* on the image) | `docker rm -f` the orphan(s), then `docker compose up -d` |
| `/api/search` → `404 Not Found` | Request hit the zombie socket on host 8000 | Use **8080** (the Compose port) |
| `400 "Index required but not found for capacity"` | Qdrant `capacity` field not indexed | Handled automatically at startup; ensure the app can reach Qdrant on boot |
| Port 8080 "already allocated" | An orphan container holds it | `docker ps -aq --filter ancestor=smart-booking-backend:latest \| xargs docker rm -f`, then `docker compose up -d` |
| Docker Desktop port link shows `{"detail":"Not Found"}` | Opened `/` before the redirect was added, or an old image | Rebuild (`docker compose up -d --build`); `/` now redirects to `/docs` |
