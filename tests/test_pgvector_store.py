"""pgvector backend logic, verified without a database.

The full behavioural contract lives in ``tests/test_vector_store.py``, which
runs the identical suite against both backends whenever ``SUPABASE_DB_URL`` is
set.  These tests cover what can be proven offline and are the parts most
likely to be silently wrong: the score-direction transform, that filter values
are *parameterised* rather than interpolated, and that the HNSW overfiltering
mitigation is actually issued.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.vector_store import InvalidFilterError, VectorStoreError


class FakeCursor:
    """Records every statement and its parameters."""

    def __init__(self, owner: FakeConnection) -> None:
        self.owner = owner

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.owner.statements.append((" ".join(sql.split()), params))

    def executemany(self, sql: str, rows: Any) -> None:
        self.owner.statements.append((" ".join(sql.split()), list(rows)))

    def fetchone(self) -> Any:
        return self.owner.fetchone_result

    def fetchall(self) -> Any:
        return self.owner.fetchall_result


class FakeConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.fetchone_result: Any = None
        self.fetchall_result: list[Any] = []
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.pgvector_store import PgVectorStore

    instance = PgVectorStore(table_name="chunks_test", dsn="postgresql://u:p@localhost:5432/db")
    connection = FakeConnection()
    monkeypatch.setattr(instance, "_connect", lambda: connection)
    instance._conn = connection
    return instance


def sql_of(store: Any) -> list[str]:
    return [s for s, _ in store._conn.statements]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_missing_dsn_fails_with_an_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``dsn`` falls back to settings, so the setting must be cleared too.

    Without this the test passes only on a machine whose ``.env`` happens to
    have no ``SUPABASE_DB_URL`` -- green in CI, red for anyone with a database
    configured.
    """
    from src.config import get_settings
    from src.pgvector_store import PgVectorStore

    monkeypatch.setattr(get_settings(), "supabase_db_url", None)

    with pytest.raises(VectorStoreError, match="Session pooler"):
        PgVectorStore(table_name="t", dsn="")


def test_unsafe_table_name_is_rejected() -> None:
    """The table name is interpolated, so it must be validated, not trusted."""
    from src.pgvector_store import PgVectorStore

    with pytest.raises(InvalidFilterError, match="Unsafe table name"):
        PgVectorStore(table_name="chunks; DROP TABLE users; --", dsn="postgresql://x")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_uses_cosine_opclass_to_match_the_lancedb_metric(store: Any) -> None:
    """A different opclass would make the two backends silently incomparable."""
    store._ensure_schema()
    statements = " ".join(sql_of(store))
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert "USING hnsw (embedding vector_cosine_ops)" in statements
    assert f"vector({384})" in statements


def test_schema_records_the_embedding_model_for_the_mismatch_check(store: Any) -> None:
    store._ensure_schema()
    assert "model_name TEXT NOT NULL" in " ".join(sql_of(store))


def test_existing_table_with_a_different_model_fails_at_open(store: Any) -> None:
    from src.embeddings import EmbeddingModelMismatchError

    store._conn.fetchone_result = ("some-other/embedding-model",)
    with pytest.raises(EmbeddingModelMismatchError, match="not comparable"):
        store._ensure_schema()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def test_upsert_uses_on_conflict_do_update_not_a_plain_insert(store: Any) -> None:
    """The SQL equivalent of merge_insert: idempotent and race-tolerant."""
    store.upsert(
        [
            {
                "chunk_id": "c1",
                "doc_key": "a" * 64,
                "text": "t",
                "source": "s.md",
                "chunk_index": 0,
                "file_type": "md",
                "vector": [0.1] * 384,
            }
        ]
    )
    statements = " ".join(sql_of(store))
    assert "ON CONFLICT (chunk_id) DO UPDATE SET" in statements
    assert "INSERT INTO chunks_test" in statements


def test_upsert_of_empty_list_issues_no_statement(store: Any) -> None:
    assert store.upsert([]) == 0
    assert store._conn.statements == []


def test_delete_by_doc_key_parameterises_and_validates(store: Any) -> None:
    store.delete_by_doc_key("b" * 64)
    statement, params = store._conn.statements[-1]
    assert statement == "DELETE FROM chunks_test WHERE doc_key = %s"
    assert params == ("b" * 64,)


def test_delete_by_doc_key_rejects_a_non_digest(store: Any) -> None:
    with pytest.raises(InvalidFilterError):
        store.delete_by_doc_key("'; DROP TABLE chunks_test; --")


def test_delete_by_source_rejects_a_hostile_value(store: Any) -> None:
    with pytest.raises(InvalidFilterError):
        store.delete_by_source("x' OR '1'='1")


# ---------------------------------------------------------------------------
# Reads -- the score-direction invariant across backends
# ---------------------------------------------------------------------------
def test_search_selects_one_minus_distance_as_similarity(store: Any) -> None:
    """`<=>` is cosine DISTANCE. Passing it through would invert every score."""
    store._conn.fetchall_result = []
    store.search(np.zeros(384, dtype=np.float32), top_k=5)
    statements = " ".join(sql_of(store))
    assert "1 - (embedding <=> %s) AS similarity" in statements
    assert "ORDER BY embedding <=> %s" in statements  # ascending distance


def test_search_returns_higher_is_better_similarity(store: Any) -> None:
    store._conn.fetchall_result = [
        ("c1", "a" * 64, "relevant text", "s.md", 0, "md", 0.93),
        ("c2", "a" * 64, "unrelated text", "s.md", 1, "md", 0.10),
    ]
    results = store.search(np.zeros(384, dtype=np.float32), top_k=2)
    assert [r.similarity for r in results] == [0.93, 0.10]
    assert results[0].similarity > results[1].similarity
    assert results[0].chunk_id == "c1"


def test_search_parameterises_filter_values_and_never_interpolates_them(
    store: Any,
) -> None:
    store._conn.fetchall_result = []
    store.search(np.zeros(384, dtype=np.float32), top_k=5, metadata_filter={"file_type": "md"})
    select = next(
        (sql, params) for sql, params in store._conn.statements if sql.startswith("SELECT chunk_id")
    )
    sql, params = select
    assert "file_type = %s" in sql
    assert "'md'" not in sql, "the value must be bound, never interpolated"
    # params holds numpy arrays alongside the value, so compare element-wise.
    assert any(isinstance(p, str) and p == "md" for p in params)


def test_search_enables_iterative_scan_only_when_filtering(store: Any) -> None:
    """The mitigation for HNSW overfiltering, and the reason pgvector is an
    interesting second backend rather than a second row in a table."""
    store._conn.fetchall_result = []
    store.search(np.zeros(384, dtype=np.float32), top_k=5, metadata_filter={"file_type": "md"})
    scan_stmts = [s for s in sql_of(store) if "hnsw.iterative_scan" in s]
    assert scan_stmts

    # Not SET LOCAL: the connection is autocommit, so a transaction-scoped
    # setting is discarded before the SELECT runs and the mitigation silently
    # does nothing.
    assert not any("SET LOCAL" in s.upper() for s in scan_stmts)

    store._conn.statements.clear()
    store.search(np.zeros(384, dtype=np.float32), top_k=5)
    assert not any("hnsw.iterative_scan" in s for s in sql_of(store))


def test_search_rejects_a_disallowed_filter_key_before_touching_sql(store: Any) -> None:
    with pytest.raises(InvalidFilterError):
        store.search(np.zeros(384, dtype=np.float32), 5, {"text": "secret"})
    assert store._conn.statements == []


def test_backend_satisfies_the_shared_protocol(store: Any) -> None:
    from src.vector_store import VectorStoreManager

    assert isinstance(store, VectorStoreManager)
