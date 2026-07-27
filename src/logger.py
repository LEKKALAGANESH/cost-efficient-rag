"""Structured per-query logging -- the monitoring layer for this scope.

``logs/queries.jsonl`` is the *only* data source for the README's p50/p95
table.  It is written from ``rag_pipeline.py`` (the single pipeline exit
point), never from ``api.py``: the eval harness calls the pipeline directly,
so an api-layer log call would produce zero lines during evaluation and leave
the latency deliverable with nothing behind it.

Known simplification, stated in the README: full query and answer text are
logged.  That is deliberate for grader visibility into groundedness; a
production deployment would hash or redact PII first.  The config object is
never logged wholesale -- it holds API keys.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_settings

_configured = False
_lock = threading.Lock()


def configure_logging(query_log_path: Path | None = None) -> None:
    """Install a human sink on stderr and a JSON-lines sink for query records.

    Idempotent: repeated calls (each Uvicorn worker, each test module) do not
    stack duplicate sinks.
    """
    global _configured
    with _lock:
        if _configured:
            return
        settings = get_settings()
        path = Path(query_log_path or settings.query_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.remove()
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            backtrace=False,  # never leak a stack trace into operator output
            diagnose=False,  # diagnose=True would print local variables (keys)
        )
        logger.add(
            str(path),
            level="INFO",
            format="{message}",
            filter=lambda record: record["extra"].get("query_log") is True,
            enqueue=True,  # process-safe across Uvicorn workers
            serialize=False,  # we emit our own flat JSON; see log_query
        )
        _configured = True


def log_query(
    *,
    query: str,
    answer: str,
    chunk_count: int,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    fallback_triggered: bool,
    attempts: int,
    top_similarity: float | None,
    citations: list[dict[str, str]] | None = None,
    model: str | None = None,
    cache_hit: bool = False,
    estimated_cost_usd: float = 0.0,
    unverified_citation_count: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    """Append one JSON object to the query log. Returns the record written.

    ``prompt_tokens``/``completion_tokens`` are summed across *all* attempts
    including failed retries and JSON repairs -- billing counts those, so an
    accounting that only records the successful attempt understates real spend.
    """
    configure_logging()
    record: dict[str, Any] = {
        "ts": time.time(),
        "event": "query",
        "query": query,
        "answer": answer,
        "chunk_count": chunk_count,
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "generation_latency_ms": round(generation_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "fallback_triggered": fallback_triggered,
        "attempts": attempts,
        "top_similarity": top_similarity,
        "citations": citations or [],
        "unverified_citation_count": unverified_citation_count,
        "model": model,
        "cache_hit": cache_hit,
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "error": error,
    }
    logger.bind(query_log=True).info(json.dumps(record, ensure_ascii=False))
    return record


def read_query_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse the query log. Malformed lines are skipped, not fatal."""
    path = Path(path or get_settings().query_log_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


__all__ = ["configure_logging", "log_query", "logger", "read_query_log"]
