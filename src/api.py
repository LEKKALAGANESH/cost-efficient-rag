"""The only HTTP-facing module. Contains no business logic.

Two design points that are security decisions, not style choices:

**Handlers are ``def``, not ``async def``.**  Embedding is CPU-bound and torch
releases the GIL, so Starlette runs ``def`` handlers in a threadpool where they
genuinely parallelise.  An ``async def`` handler wrapping a blocking
``encode()`` plus a blocking HTTP client stalls the entire event loop for the
full request -- strictly worse, and it inverts the concurrency benefit it
appears to buy.

**/ingest is upload-only.**  A server-side directory parameter is an
arbitrary-file-read primitive: ``{"directory": "."}`` would walk the
filesystem, ingest everything the loaders parse, and make it verbatim
retrievable through /query, which quotes and cites chunks.  With .env at the
repo root that is a two-request path to exfiltrating API keys.  "It's
local-only" is not a valid scope boundary for a file-read primitive.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import SUPPORTED_EXTENSIONS, get_settings
from src.embeddings import EmbeddingModelMismatchError, embedding_service
from src.ingest_service import ingest_bytes
from src.llm_client import LLMError
from src.logger import configure_logging, logger
from src.rag_pipeline import RAGPipeline, get_pipeline
from src.vector_store import InvalidFilterError, VectorStoreError

# N workers x C intra-op threads thrashes a C-core box; one thread per worker
# is the right default when concurrency comes from the threadpool instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Boot-time validation: a missing env var fails here, not at first query."""
    configure_logging()
    settings = get_settings()
    logger.info(
        f"cost-efficient-rag up | backend={settings.vector_backend} "
        f"| model={settings.embedding_model_name}"
    )
    yield


app = FastAPI(
    title="cost-efficient-rag",
    version="1.0.0",
    description=(
        "RAG question answering over an embedded vector store. "
        "Unauthenticated by design (see README threat model) -- bind to "
        "127.0.0.1 and do not expose to a network."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    # An unbounded query string flows straight into the generation prompt as
    # billable tokens, so the bound is a cost control as well as a validation.
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    metadata_filter: dict[str, str] | None = Field(
        default=None,
        description="Keys must appear in ALLOWED_FILTER_KEYS; anything else is 422.",
    )


class CitationModel(BaseModel):
    source: str
    chunk_id: str


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    source: str
    chunk_index: int
    file_type: str
    similarity: float
    text: str


class QueryMetadata(BaseModel):
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    chunk_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    fallback_triggered: bool
    top_similarity: float | None
    unverified_citation_count: int
    cache_hit: bool
    model: str | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    retrieved_chunks: list[RetrievedChunkModel]
    metadata: QueryMetadata


class FileStatus(BaseModel):
    source: str
    status: str
    doc_key: str | None = None
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_skipped_cached: int = 0
    error: str | None = None


class IngestResponse(BaseModel):
    files: list[FileStatus]
    total_chunks_indexed: int
    vector_count: int


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_dim: int
    vector_backend: str
    vector_count: int
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    similarity_threshold: float


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def pipeline_dependency() -> RAGPipeline:
    """Overridable in tests so a temp-dir store can be injected."""
    return get_pipeline()


PipelineDep = Annotated[RAGPipeline, Depends(pipeline_dependency)]


# ---------------------------------------------------------------------------
# Error handling -- sanitised bodies with a correlation ID, never a stack trace
# ---------------------------------------------------------------------------
def _error_body(code: str, detail: str) -> dict[str, Any]:
    return {"error": code, "detail": detail, "correlation_id": uuid.uuid4().hex}


@app.exception_handler(InvalidFilterError)
def _handle_invalid_filter(_r: Request, exc: InvalidFilterError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body("invalid_filter", str(exc)))


@app.exception_handler(EmbeddingModelMismatchError)
def _handle_model_mismatch(_r: Request, exc: EmbeddingModelMismatchError) -> JSONResponse:
    return JSONResponse(status_code=503, content=_error_body("embedding_model_mismatch", str(exc)))


@app.exception_handler(VectorStoreError)
def _handle_store_error(_r: Request, exc: VectorStoreError) -> JSONResponse:
    return JSONResponse(status_code=503, content=_error_body("vector_store_unavailable", str(exc)))


@app.exception_handler(LLMError)
def _handle_llm_error(_r: Request, exc: LLMError) -> JSONResponse:
    # 502, and never a fabricated answer presented as a success.
    return JSONResponse(status_code=502, content=_error_body("generation_failed", str(exc)))


@app.exception_handler(Exception)
def _handle_unexpected(_r: Request, exc: Exception) -> JSONResponse:
    correlation = uuid.uuid4().hex
    logger.exception(f"Unhandled error correlation_id={correlation}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            # Deliberately generic: a filesystem path or stack frame here is an
            # information disclosure. The detail is in the server log.
            "detail": "An internal error occurred. Quote the correlation_id when reporting it.",
            "correlation_id": correlation,
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health(pipeline: PipelineDep) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        embedding_model=embedding_service.model_name,
        embedding_dim=embedding_service.dim,
        vector_backend=settings.vector_backend,
        vector_count=pipeline.store.count(),
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        default_top_k=settings.default_top_k,
        similarity_threshold=settings.similarity_threshold,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(pipeline: PipelineDep, files: list[UploadFile] = File(...)) -> IngestResponse:
    """Upload-only ingestion. No directory parameter -- see module docstring."""
    settings = get_settings()
    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=413,
            detail=f"At most {settings.max_files_per_request} files per request.",
        )

    statuses: list[FileStatus] = []
    for upload in files:
        # basename + charset allowlist. The client filename is metadata only;
        # filename="../../.env" must never become a path component.
        raw_name = os.path.basename(upload.filename or "unnamed")
        # ASCII-only: ``str.isalnum()`` is true for 'e'-acute, CJK and Cyrillic,
        # so "cafe.md" (accented) passed here and then failed the store's
        # stricter source pattern -- raising out of the per-file loop and
        # failing the whole batch instead of just that file.
        safe_name = (
            "".join(c for c in raw_name if c.isascii() and (c.isalnum() or c in "._- "))[:120]
            or "unnamed"
        )
        extension = Path(safe_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            statuses.append(
                FileStatus(
                    source=safe_name,
                    status="failed",
                    error=f"Unsupported extension {extension!r}. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
                )
            )
            continue

        payload = _read_capped(upload, settings.max_upload_bytes)
        if payload is None:
            statuses.append(
                FileStatus(
                    source=safe_name,
                    status="failed",
                    error=f"File exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes} bytes).",
                )
            )
            continue

        # The upload is not persisted to disk. Nothing ever read it back, and
        # ingestion is content-addressed by doc_key, so the copy was pure
        # accumulation: an unauthenticated caller could write
        # max_files_per_request * max_upload_bytes per request with no
        # retention path. Chunks and their text live in the vector store.
        result = ingest_bytes(payload, safe_name, pipeline.store)
        statuses.append(FileStatus(**result.__dict__))

    return IngestResponse(
        files=statuses,
        total_chunks_indexed=sum(s.chunks_embedded for s in statuses),
        vector_count=pipeline.store.count(),
    )


def _read_capped(upload: UploadFile, max_bytes: int) -> bytes | None:
    """Stream with a running counter. ``Content-Length`` is client-controlled.

    Returns ``None`` once the cap is exceeded, without buffering the rest.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        block = upload.file.read(1024 * 256)
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            return None
        chunks.append(block)
    return b"".join(chunks)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: PipelineDep) -> QueryResponse:
    """Retrieve, ground, and answer. Logging happens inside the pipeline."""
    result = pipeline.query(
        request.query, top_k=request.top_k, metadata_filter=request.metadata_filter
    )
    return QueryResponse(
        answer=result.answer,
        citations=[CitationModel(**c.__dict__) for c in result.citations],
        retrieved_chunks=[
            RetrievedChunkModel(
                chunk_id=c["chunk_id"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                file_type=c["file_type"],
                similarity=c["similarity"],
                text=c["text"],
            )
            for c in result.retrieved_chunks
        ],
        metadata=QueryMetadata(
            retrieval_latency_ms=round(result.retrieval_latency_ms, 3),
            generation_latency_ms=round(result.generation_latency_ms, 3),
            total_latency_ms=round(result.total_latency_ms, 3),
            chunk_count=result.chunk_count,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            fallback_triggered=result.fallback_triggered,
            top_similarity=result.top_similarity,
            unverified_citation_count=result.unverified_citation_count,
            cache_hit=result.cache_hit,
            model=result.model,
        ),
    )
