"""Orchestrates ``ingestion.py`` (pure transform) against a vector store.

Kept separate from ``ingestion.py`` so that module stays a hermetic
``(bytes, config) -> records`` function testable with no store dependency, and
separate from ``api.py`` so the CLI, tests, and eval harness can ingest without
booting FastAPI.
"""

from __future__ import annotations

from pathlib import Path

from src.config import SUPPORTED_EXTENSIONS
from src.ingestion import (
    DocumentParseError,
    FileIngestionResult,
    UnsupportedFileTypeError,
    build_chunk_records,
    embed_records,
)
from src.logger import configure_logging, logger
from src.vector_store import InvalidFilterError, VectorStoreManager, get_vector_store


def ingest_bytes(
    file_bytes: bytes,
    filename: str,
    store: VectorStoreManager,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    skip_reembedding: bool = True,
) -> FileIngestionResult:
    """Ingest one document. Never raises for a bad file -- reports it instead.

    A single malformed file in a batch must not abort the batch, so every
    per-file failure becomes a status row the caller can surface.
    """
    configure_logging()
    source = Path(filename).name
    try:
        records, doc_key, _file_type = build_chunk_records(
            file_bytes, filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # Validated here, inside the guard: a name the store will reject must
        # fail this file alone. Letting InvalidFilterError escape turned one bad
        # filename into a 422 for every file in the request.
        store.doc_keys_for_source(source)
    except (UnsupportedFileTypeError, DocumentParseError, InvalidFilterError) as exc:
        logger.warning(f"Ingest failed for {filename!r}: {exc}")
        return FileIngestionResult(source=source, status="failed", error=str(exc))
    if not records:
        logger.warning(f"{filename!r} produced 0 chunks (empty or unparseable content)")
        return FileIngestionResult(source=source, status="empty", doc_key=doc_key)

    # Decide whether this is an unchanged re-ingest or a genuine revision by
    # comparing content hashes stored under this logical document name.
    stored_doc_keys = store.doc_keys_for_source(source)
    unchanged = stored_doc_keys == {doc_key}

    if unchanged and skip_reembedding:
        # The pure idempotency path: zero embeddings, zero writes. Correctness
        # does not depend on reaching it -- merge_insert below would be
        # idempotent anyway -- so a concurrent writer racing this check costs
        # CPU, never duplicate rows.
        logger.info(f"{source!r} unchanged ({len(records)} chunks); nothing re-embedded")
        return FileIngestionResult(
            source=source,
            status="ok",
            doc_key=doc_key,
            chunks_created=len(records),
            chunks_embedded=0,
            chunks_skipped_cached=len(records),
        )

    if stored_doc_keys - {doc_key}:
        # The document changed. Delete by *source*, not by doc_key: doc_key is
        # sha256(file_bytes), so the revised file has a different one and
        # deleting by it cannot reach the superseded rows. They would linger as
        # orphans that retrieval still returns and generation still cites.
        logger.info(f"{source!r} changed; removing {len(stored_doc_keys)} prior version(s)")
        store.delete_by_source(source)

    existing = store.chunk_ids_for_source(source) if skip_reembedding else set()
    embedded, n_skipped = embed_records(records, existing)
    store.upsert([r.to_dict() for r in embedded])

    return FileIngestionResult(
        source=source,
        status="ok",
        doc_key=doc_key,
        chunks_created=len(records),
        chunks_embedded=len(embedded),
        chunks_skipped_cached=n_skipped,
    )


def ingest_directory(
    directory: str | Path,
    store: VectorStoreManager | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[FileIngestionResult]:
    """Ingest every supported file in a directory.

    **Local/offline use only** -- this is how the corpus and the eval labels are
    built.  It is deliberately *not* reachable from any HTTP route: a
    server-side directory parameter is an arbitrary-file-read primitive, and
    since /query quotes and cites retrieved chunks verbatim, it would be a
    two-request path to exfiltrating .env.  See README "Threat model".
    """
    store = store if store is not None else get_vector_store()
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    results: list[FileIngestionResult] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        results.append(
            ingest_bytes(
                path.read_bytes(),
                path.name,
                store,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return results
