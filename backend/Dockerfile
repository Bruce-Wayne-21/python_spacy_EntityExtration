# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Smart Booking Hybrid Search — backend image
# ---------------------------------------------------------------------------
# Python 3.13 slim to match the local venv. The spaCy model is downloaded at
# BUILD time (it is not a pip requirement) so the container starts offline-ready.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

# - Don't buffer stdout/stderr so logs appear immediately in `docker logs`.
# - Don't write .pyc files inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (better layer caching: only re-runs when
# requirements.txt changes, not on every source edit).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy English model. Baked into the image so no network is
# needed at runtime and the app doesn't fail its startup model load.
RUN python -m spacy download en_core_web_sm

# Copy application source.
COPY main.py nlp_extractor.py vector_search.py ./

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable from outside the container.
# (127.0.0.1 would only be reachable from inside it.)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
