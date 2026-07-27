"""One contract suite, parametrised over every backend.

Both LanceDB and pgvector must satisfy the identical input/output contract, so
the eval harness and the API can swap backends via one env var.  The pgvector
parametrisation self-skips unless ``SUPABASE_DB_URL`` is set, so the suite is
green offline and still proves parity when a database is reachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from src.config import get_settings
from src.embeddings import embedding_service
from src.vector_store import (
    InvalidFilterError,
    LanceDBVectorStore,
    SearchResult,
    VectorStoreManager,
    validate_metadata_filter,
)

DOC_A = "a" * 64
DOC_B = "b" * 64

CORPUS = [
    (
        "c1",
        DOC_A,
        "LanceDB is an embedded columnar vector database with no server process.",
        "store.md",
        0,
        "md",
    ),
    (
        "c2",
        DOC_A,
        "Managed vector databases bill for provisioned capacity even when idle.",
        "store.md",
        1,
        "md",
    ),
    (
        "c3",
        DOC_B,
        "Photosynthesis converts light energy into chemical energy in plant cells.",
        "bio.html",
        0,
        "html",
    ),
]


def _records() -> list[dict[str, Any]]:
    vectors = embedding_service.embed_texts([row[2] for row in CORPUS])
    return [
        {
            "chunk_id": cid,
            "doc_key": dk,
            "text": text,
            "source": src,
            "chunk_index": idx,
            "file_type": ftype,
            "vector": vec.tolist(),
        }
        for (cid, dk, text, src, idx, ftype), vec in zip(CORPUS, vectors, strict=True)
    ]


# ---------------------------------------------------------------------------
# Backend parametrisation
# ---------------------------------------------------------------------------
def _lancedb_store(tmp_path: Any) -> Iterator[VectorStoreManager]:
    store = LanceDBVectorStore(db_path=tmp_path / "lancedb", table_name="contract")
    yield store
    store.close()


def _pgvector_store(tmp_path: Any) -> Iterator[VectorStoreManager]:
    from src.pgvector_store import PgVectorStore

    store = PgVectorStore(table_name="contract_test_chunks")
    store.reset()
    yield store
    store.reset()
    store.close()


def _pgvector_skip_reason() -> str | None:
    """Why the pgvector parametrisation cannot run, or ``None`` if it can.

    Gating on ``os.environ`` alone was wrong twice over: the store resolves its
    DSN from ``.env`` as well, so a configured machine still skipped silently,
    and a *placeholder* DSN turned every parity test into a connection error
    rather than an honest skip.  Probing once, here, keeps the suite green
    offline while making the reason it did not run visible in `-rs` output.
    """
    dsn = get_settings().supabase_db_url
    if not dsn:
        return "SUPABASE_DB_URL unset; pgvector parity test skipped"
    try:
        import psycopg

        psycopg.connect(dsn, connect_timeout=5).close()
    except Exception as exc:  # any failure at all means "cannot verify parity"
        return f"SUPABASE_DB_URL set but unreachable, so parity is unproven: {exc}"
    return None


_PGVECTOR_SKIP = _pgvector_skip_reason()

BACKENDS = [
    pytest.param(_lancedb_store, id="lancedb"),
    pytest.param(
        _pgvector_store,
        id="pgvector",
        marks=pytest.mark.skipif(_PGVECTOR_SKIP is not None, reason=_PGVECTOR_SKIP or ""),
    ),
]


@pytest.fixture(params=BACKENDS)
def store(request: pytest.FixtureRequest, tmp_path: Any) -> Iterator[VectorStoreManager]:
    yield from request.param(tmp_path)


@pytest.fixture
def populated_store(store: VectorStoreManager) -> VectorStoreManager:
    store.upsert(_records())
    return store


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def test_backend_satisfies_the_protocol(store: VectorStoreManager) -> None:
    assert isinstance(store, VectorStoreManager)


# ---------------------------------------------------------------------------
# Score direction -- the two assertions that make the whole bug class impossible
# ---------------------------------------------------------------------------
def test_query_identical_to_an_ingested_chunk_scores_approximately_one(
    populated_store: VectorStoreManager,
) -> None:
    """If the metric or normalisation is wrong, this is not ~1.0."""
    query = embedding_service.embed_query(CORPUS[0][2])
    results = populated_store.search(query, top_k=3)
    assert results[0].chunk_id == "c1"
    assert results[0].similarity == pytest.approx(1.0, abs=1e-3)


def test_relevant_query_outscores_irrelevant_content(
    populated_store: VectorStoreManager,
) -> None:
    """Higher similarity must mean *more* relevant, not less."""
    query = embedding_service.embed_query("which vector database runs embedded without a server?")
    results = populated_store.search(query, top_k=3)
    by_id = {r.chunk_id: r.similarity for r in results}
    assert by_id["c1"] > by_id["c3"], (
        "The embedded-vector-DB chunk must outscore the photosynthesis chunk. "
        "If this inverts, `similarity` is carrying a distance."
    )
    assert results == sorted(results, key=lambda r: r.similarity, reverse=True)


def test_similarity_is_bounded_and_never_exposed_as_distance(
    populated_store: VectorStoreManager,
) -> None:
    results = populated_store.search(embedding_service.embed_query("anything"), top_k=3)
    assert all(-1.01 <= r.similarity <= 1.01 for r in results)
    assert not hasattr(SearchResult, "score")
    assert not hasattr(SearchResult, "distance")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def test_reupserting_identical_records_adds_no_rows(store: VectorStoreManager) -> None:
    """R3: re-ingesting the same document produces 0 new entries."""
    records = _records()
    store.upsert(records)
    count_after_first = store.count()
    store.upsert(records)
    store.upsert(records)
    assert count_after_first == len(CORPUS)
    assert store.count() == count_after_first


def test_upsert_updates_in_place_on_matching_key(store: VectorStoreManager) -> None:
    records = _records()
    store.upsert(records)
    records[0]["text"] = "REVISED TEXT for the same chunk_id"
    store.upsert(records)
    assert store.count() == len(CORPUS)
    hits = store.search(embedding_service.embed_query("REVISED TEXT"), top_k=3)
    assert any(h.text.startswith("REVISED TEXT") for h in hits)


def test_delete_by_doc_key_removes_only_that_document(
    populated_store: VectorStoreManager,
) -> None:
    populated_store.delete_by_doc_key(DOC_A)
    assert populated_store.count() == 1
    remaining = populated_store.search(embedding_service.embed_query("photosynthesis"), top_k=5)
    assert {r.doc_key for r in remaining} == {DOC_B}


def test_delete_by_doc_key_rejects_a_non_digest(store: VectorStoreManager) -> None:
    with pytest.raises(InvalidFilterError):
        store.delete_by_doc_key("'; DROP TABLE chunks; --")


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------
def test_metadata_filter_restricts_results(populated_store: VectorStoreManager) -> None:
    query = embedding_service.embed_query("energy in cells")
    assert {r.file_type for r in populated_store.search(query, 5, {"file_type": "md"})} == {"md"}
    assert {r.file_type for r in populated_store.search(query, 5, {"file_type": "html"})} == {
        "html"
    }


def test_metadata_filter_matching_nothing_returns_empty_not_error(
    populated_store: VectorStoreManager,
) -> None:
    query = embedding_service.embed_query("anything")
    assert populated_store.search(query, 5, {"file_type": "pdf"}) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_search_on_empty_store_returns_empty_list(store: VectorStoreManager) -> None:
    assert store.search(embedding_service.embed_query("anything"), top_k=5) == []
    assert store.count() == 0


def test_top_k_larger_than_corpus_returns_what_exists(
    populated_store: VectorStoreManager,
) -> None:
    results = populated_store.search(embedding_service.embed_query("database"), top_k=100)
    assert len(results) == len(CORPUS)


def test_upsert_of_empty_list_is_a_noop(store: VectorStoreManager) -> None:
    assert store.upsert([]) == 0
    assert store.count() == 0


def test_reopening_an_existing_table_in_a_fresh_handle_finds_it(tmp_path: Any) -> None:
    """A second process must open the existing table, not try to create it.

    Regression: LanceDB's list_tables() returns a ListTablesResponse, not a
    list, so a membership test against it silently returns False and the second
    open raises "Table already exists". Every test that used a fresh temp dir
    passed regardless.
    """
    first = LanceDBVectorStore(db_path=tmp_path / "db", table_name="persisted")
    first.upsert(_records())
    assert first.count() == len(CORPUS)
    first.close()

    second = LanceDBVectorStore(db_path=tmp_path / "db", table_name="persisted")
    assert second.count() == len(CORPUS)
    assert second.upsert(_records()) == len(CORPUS)
    assert second.count() == len(CORPUS)
    second.close()


# ---------------------------------------------------------------------------
# Filter validation (backend-independent, so not parametrised)
# ---------------------------------------------------------------------------
def test_filter_allowlist_accepts_configured_keys() -> None:
    assert validate_metadata_filter({"file_type": "pdf"}) == {"file_type": "pdf"}
    assert validate_metadata_filter(None) == {}
    assert validate_metadata_filter({}) == {}


def test_filter_allowlist_rejects_unlisted_key() -> None:
    with pytest.raises(InvalidFilterError, match="not allowed"):
        validate_metadata_filter({"text": "anything"})


@pytest.mark.parametrize(
    "hostile",
    ["x' OR '1'='1", "a'; DROP TABLE rag_chunks; --", "' UNION SELECT 1 --", "a" * 200, ""],
)
def test_filter_rejects_hostile_values_before_any_predicate_is_built(hostile: str) -> None:
    with pytest.raises(InvalidFilterError):
        validate_metadata_filter({"file_type": hostile})


def test_filter_rejects_non_string_value() -> None:
    with pytest.raises(InvalidFilterError):
        validate_metadata_filter({"file_type": 42})  # type: ignore[dict-item]
