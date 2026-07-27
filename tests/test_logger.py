"""Query logging -- the only data source behind the README's p50/p95 table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.logger import configure_logging, log_query, read_query_log


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point logging at a temp file and force reconfiguration for this test."""
    import src.logger as logger_module

    path = tmp_path / "queries.jsonl"
    monkeypatch.setattr(logger_module, "_configured", False)
    configure_logging(path)
    yield path
    monkeypatch.setattr(logger_module, "_configured", False)


def _write(path: Path, **overrides: object) -> dict[str, object]:
    payload = {
        "query": "what is a vector database?",
        "answer": "A store for embeddings.",
        "chunk_count": 5,
        "retrieval_latency_ms": 24.5,
        "generation_latency_ms": 810.2,
        "total_latency_ms": 834.7,
        "prompt_tokens": 1200,
        "completion_tokens": 180,
        "fallback_triggered": False,
        "attempts": 1,
        "top_similarity": 0.71,
    }
    payload.update(overrides)
    return log_query(**payload)  # type: ignore[arg-type]


def test_log_query_writes_one_json_object_per_call(log_path: Path) -> None:
    _write(log_path)
    _write(log_path, query="second")

    from loguru import logger as loguru_logger

    loguru_logger.complete()  # enqueue=True means the sink is asynchronous

    records = read_query_log(log_path)
    assert len(records) == 2
    assert records[0]["query"] == "what is a vector database?"
    assert records[1]["query"] == "second"


def test_record_carries_every_field_r11_requires(log_path: Path) -> None:
    """R11: per-query latency, chunk count, and token usage."""
    record = _write(log_path)
    for field in (
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "chunk_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "fallback_triggered",
        "attempts",
        "top_similarity",
        "ts",
    ):
        assert field in record, f"missing {field}"
    assert record["total_tokens"] == 1200 + 180


def test_fallback_record_shows_generation_was_skipped(log_path: Path) -> None:
    """Fallback correctness is verified from the log: flag set, zero tokens."""
    record = _write(
        log_path,
        fallback_triggered=True,
        prompt_tokens=0,
        completion_tokens=0,
        generation_latency_ms=0.0,
    )
    assert record["fallback_triggered"] is True
    assert record["total_tokens"] == 0


def test_record_is_json_serialisable(log_path: Path) -> None:
    record = _write(log_path)
    assert json.loads(json.dumps(record))["chunk_count"] == 5


def test_reading_a_missing_log_returns_empty_not_an_error(tmp_path: Path) -> None:
    assert read_query_log(tmp_path / "absent.jsonl") == []


def test_malformed_lines_are_skipped_rather_than_fatal(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"event": "query", "total_latency_ms": 1.0}\n'
        "this line is not json\n"
        "\n"
        '{"event": "query", "total_latency_ms": 2.0}\n',
        encoding="utf-8",
    )
    records = read_query_log(path)
    assert len(records) == 2
    assert [r["total_latency_ms"] for r in records] == [1.0, 2.0]


def test_configure_logging_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each Uvicorn worker and each test module calls this; sinks must not stack."""
    import src.logger as logger_module

    monkeypatch.setattr(logger_module, "_configured", False)
    path = tmp_path / "q.jsonl"
    configure_logging(path)
    configure_logging(path)
    configure_logging(path)

    log_query(
        query="q",
        answer="a",
        chunk_count=1,
        retrieval_latency_ms=1.0,
        generation_latency_ms=1.0,
        total_latency_ms=2.0,
        prompt_tokens=1,
        completion_tokens=1,
        fallback_triggered=False,
        attempts=1,
        top_similarity=0.5,
    )
    from loguru import logger as loguru_logger

    loguru_logger.complete()
    assert len(read_query_log(path)) == 1, "a duplicated sink would write it twice"
    monkeypatch.setattr(logger_module, "_configured", False)


def test_percentiles_are_computable_from_the_log(log_path: Path) -> None:
    """The README's latency table is derived from this file, not measured ad hoc."""
    for value in (10.0, 20.0, 30.0, 40.0, 100.0):
        _write(log_path, retrieval_latency_ms=value)

    from loguru import logger as loguru_logger

    loguru_logger.complete()

    import statistics

    latencies = sorted(r["retrieval_latency_ms"] for r in read_query_log(log_path))
    assert statistics.median(latencies) == 30.0
    assert max(latencies) == 100.0
