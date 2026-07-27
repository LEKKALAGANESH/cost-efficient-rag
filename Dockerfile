# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# OMP_NUM_THREADS=1: concurrency comes from Starlette's threadpool, so N
# workers x C intra-op torch threads would thrash a C-core box.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# CPU-only torch: the GPU wheel is ~2.5 GB and buys nothing for MiniLM at this
# scale. Installed first so the layer caches independently of app code.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

RUN useradd --create-home --uid 10001 app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app eval/ ./eval/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app data/raw_documents/ ./data/raw_documents/
COPY --chown=app:app data/eval_dataset.json data/threshold_calibration.json ./data/
COPY --chown=app:app env.example README.md ./

USER app

# Bake the embedding model into the image so a cold container does not need
# network access on its first request.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

# 127.0.0.1 inside the container would be unreachable from the host, so bind
# 0.0.0.0 here and publish deliberately: `docker run -p 127.0.0.1:8000:8000`.
# The service is unauthenticated by design (see README "Threat model") and must
# not be published to a public interface.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
