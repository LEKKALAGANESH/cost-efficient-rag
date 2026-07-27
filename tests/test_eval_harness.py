"""End-to-end eval harness against a real store and a mocked LLM.

Covers the aggregation policy that the metric unit tests cannot: that the
out-of-corpus question is *excluded* from the rank metrics and *scored* under
fallback correctness, that n_ranked and n_fallback are both reported, and that
the k-sweep comes from a single retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.evaluate_answer import run as run_answer
from eval.evaluate_retrieval import run as run_retrieval
from src.config import FALLBACK_ANSWER
from src.ingest_service import ingest_bytes
from src.llm_client import LLMResponse
from src.rag_pipeline import RAGPipeline
from src.vector_store import LanceDBVectorStore

# Long enough to split into several chunks. A single-chunk corpus dilutes
# every similarity below the fallback threshold, which would make this fixture
# measure the threshold rather than the harness.
DOC = (
    b"# Storage Notes\n\n"
    b"A single embedding at 384 dimensions stored as float32 occupies 1,536 "
    b"bytes, approximately 1.5 kilobytes of storage per vector. Metadata adds "
    b"a further half kilobyte per vector, covering the document identifier, "
    b"the chunk identifier, the source name, the file type, and the chunk text "
    b"itself, so the total per-vector footprint is about two kilobytes. At that "
    b"footprint one million vectors occupy roughly two gigabytes of disk.\n\n"
    b"## Concurrency\n\n"
    b"LanceDB uses optimistic concurrency control, so concurrent writers are "
    b"permitted but each commit retries a bounded number of times. Write "
    b"contention therefore surfaces as a failed commit rather than as blocking, "
    b"which is why the team standard is a single dedicated ingestion worker per "
    b"table rather than a pool of writers competing for the same manifest.\n\n"
    b"## Compaction\n\n"
    b"Every write to a Lance table creates a new version, so without a "
    b"scheduled optimize call the disk usage grows without bound. Cleaning up "
    b"superseded index files is part of the same maintenance job. This is the "
    b"single most common operational surprise reported by teams adopting an "
    b"embedded columnar store, and it is an architecture gap as much as a cost "
    b"one because a single-box deployment has nowhere to run the job.\n"
)


class StubJudge:
    """Returns a fixed, schema-valid verdict for any input."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_with_chain(self, messages: list[dict[str, Any]], **_: Any) -> LLMResponse:
        """Generation goes through the provider chain; judging still calls a
        named model directly, so this double has to answer to both."""
        return self.complete(messages)

    def complete(self, messages: list[dict[str, Any]], **_: Any) -> LLMResponse:
        self.calls += 1
        system = messages[0]["content"]
        if "strict evaluator" in system:
            payload = {
                "faithfulness_rationale": "The context supports the claim.",
                "faithfulness_score": 4,
                "relevance_rationale": "Addresses the question asked.",
                "relevance_score": 5,
                "unsupported_claims": [],
            }
            return LLMResponse(
                text=json.dumps(payload),
                model="test/judge",
                prompt_tokens=200,
                completion_tokens=60,
            )
        import re

        match = re.search(r"\[Doc: [^,\]]+, Chunk: [0-9a-f]{64}\]", system)
        citation = match.group(0) if match else ""
        return LLMResponse(
            text=f"A vector occupies about 1.5 kilobytes {citation}.",
            model="test/gen",
            prompt_tokens=300,
            completion_tokens=40,
        )


@pytest.fixture
def populated(tmp_path: Path) -> tuple[RAGPipeline, StubJudge, dict[str, Any]]:
    store = LanceDBVectorStore(db_path=tmp_path / "db", table_name="evaltest")
    ingest_bytes(DOC, "storage_notes.md", store)
    llm = StubJudge()
    pipeline = RAGPipeline(store=store, llm_client=llm)

    chunks = (
        store._open_table(create_if_missing=False)
        .search()
        .select(["chunk_id", "text"])
        .limit(None)
        .to_list()
    )

    def chunk_containing(marker: str) -> str:
        return next(c["chunk_id"] for c in chunks if marker in c["text"])

    dataset = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunking": {"chunk_size": 500, "chunk_overlap": 50},
        "corpus_chunk_count": len(chunks),
        "questions": [
            {
                "id": "t1",
                "question": "How many bytes does a 384-dimension float32 embedding occupy?",
                "ground_truth_answer": "1,536 bytes, about 1.5 KB.",
                "expects_fallback": False,
                "relevant_chunk_ids": [chunk_containing("1,536")],
            },
            {
                "id": "t2",
                "question": "What happens to disk usage without a scheduled optimize call?",
                "ground_truth_answer": "It grows without bound.",
                "expects_fallback": False,
                "relevant_chunk_ids": [chunk_containing("grows without bound")],
            },
            {
                "id": "t3",
                "question": "What is the best recipe for sourdough bread?",
                "ground_truth_answer": None,
                "expects_fallback": True,
                "relevant_chunk_ids": [],
            },
        ],
    }
    return pipeline, llm, dataset


# ---------------------------------------------------------------------------
# Retrieval harness
# ---------------------------------------------------------------------------
def test_retrieval_excludes_the_unjudgeable_query_from_every_rank_metric(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    section = run_retrieval(pipeline, dataset, primary_k=5)

    assert section["n_questions"] == 3
    assert section["n_ranked"] == 2
    assert section["n_fallback"] == 1
    # Both denominators are auditable, which is the point of reporting both.
    assert section["aggregate"]["recall"]["n"] == 2

    rows = {r["id"]: r for r in section["per_question"]}
    assert rows["t3"]["judgeable"] is False
    assert "metrics" not in rows["t3"]
    assert "empty relevant set" in rows["t3"]["excluded_reason"]


def test_retrieval_scores_a_missed_but_judgeable_query_as_zero(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    """Relevant chunks exist but are unreachable at k=1 -> 0, not excluded."""
    pipeline, _, dataset = populated
    dataset = {
        **dataset,
        "questions": [
            {
                "id": "miss",
                "question": "What happens to disk usage without a scheduled optimize call?",
                "ground_truth_answer": "x",
                "expects_fallback": False,
                # A real chunk id that this query will not rank first.
                "relevant_chunk_ids": ["0" * 64],
            }
        ],
    }
    section = run_retrieval(pipeline, dataset, primary_k=5)
    assert section["n_ranked"] == 1
    assert section["aggregate"]["recall"]["mean"] == 0.0
    assert section["aggregate"]["mrr"]["mean"] == 0.0


def test_retrieval_reports_the_precision_ceiling_and_ratio(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    agg = run_retrieval(pipeline, dataset, primary_k=5)["aggregate"]
    # One relevant chunk each at k=5 -> a structural ceiling of 0.20.
    assert agg["max_achievable_precision"]["mean"] == pytest.approx(0.2)
    assert "precision_ratio_of_achievable" in agg


def test_k_sweep_covers_every_k_from_one_retrieval(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    section = run_retrieval(pipeline, dataset, primary_k=5)
    assert set(section["k_sweep"]) == {"k=1", "k=3", "k=5", "k=10", "k=20"}
    assert "ndcg" in section["k_sweep"]["k=5"]
    assert "ndcg" in section["k_sweep"]["k=10"]
    # Recall is monotone non-decreasing in k.
    recalls = [section["k_sweep"][f"k={k}"]["recall"]["mean"] for k in (1, 3, 5, 10, 20)]
    assert recalls == sorted(recalls)


def test_retrieval_harness_makes_no_llm_calls(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    """Retrieval eval must cost zero quota, so it can loop for a stable p95."""
    pipeline, llm, dataset = populated
    run_retrieval(pipeline, dataset, primary_k=5, latency_loops=3)
    assert llm.calls == 0


def test_latency_loops_multiply_the_sample_count(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    single = run_retrieval(pipeline, dataset, primary_k=5, latency_loops=1)
    looped = run_retrieval(pipeline, dataset, primary_k=5, latency_loops=4)
    assert single["retrieval_latency_ms"]["n_samples"] == 3
    assert looped["retrieval_latency_ms"]["n_samples"] == 12


# ---------------------------------------------------------------------------
# Answer harness
# ---------------------------------------------------------------------------
def test_answer_harness_scores_fallback_separately_from_faithfulness(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    section = run_answer(pipeline, StubJudge(), dataset, judge_model="test/judge", top_k=5)

    assert section["n_questions"] == 3
    assert section["n_fallback_questions"] == 1
    assert section["fallback_correctness"]["n_correct"] == 1
    assert section["fallback_correctness"]["rate"] == 1.0

    rows = {r["id"]: r for r in section["per_question"]}
    assert rows["t3"]["scored"] is False
    assert rows["t3"]["scored_under"] == "fallback_correctness"
    assert rows["t3"]["answer"] == FALLBACK_ANSWER
    assert rows["t3"]["fallback_exact_string"] is True
    # Generation was skipped entirely: a correctness property AND a cost saving.
    assert rows["t3"]["generation_skipped"] is True
    assert rows["t3"]["generation_completion_tokens"] == 0
    # And it never reached the faithfulness/relevance denominators.
    assert section["faithfulness"]["n"] == 2


def test_answer_harness_sums_tokens_across_generation_and_judging(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    section = run_answer(pipeline, StubJudge(), dataset, judge_model="test/judge", top_k=5)
    # 2 answered questions: generation 300+40 each, judging 200+60 each.
    assert section["tokens"]["prompt"] == 2 * (300 + 200)
    assert section["tokens"]["completion"] == 2 * (40 + 60)
    assert (
        section["tokens"]["total"] == section["tokens"]["prompt"] + section["tokens"]["completion"]
    )


def test_answer_harness_records_verified_citations(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    pipeline, _, dataset = populated
    section = run_answer(pipeline, StubJudge(), dataset, judge_model="test/judge", top_k=5)
    answered = [r for r in section["per_question"] if r.get("scored")]
    assert answered
    for row in answered:
        assert row["citations"], "a non-fallback answer must carry a citation"
        assert row["unverified_citation_count"] == 0


def test_false_refusal_is_counted_not_excluded(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    """An in-corpus question the system refuses must hurt the score.

    Excluding it would let a system that refuses everything post a perfect
    faithfulness mean.
    """
    pipeline, _, dataset = populated
    dataset = {
        **dataset,
        "questions": [
            {
                "id": "refused",
                "question": "What is the airspeed velocity of an unladen swallow?",
                "ground_truth_answer": "x",
                "expects_fallback": False,  # we claim it IS answerable
                "relevant_chunk_ids": ["0" * 64],
            }
        ],
    }
    section = run_answer(pipeline, StubJudge(), dataset, judge_model="test/judge", top_k=5)
    assert section["false_refusals"] == 1

    # It must not vanish -- but it is scored by rule, so it belongs in the
    # imputed bucket rather than inflating the measured faithfulness mean.
    imputed = section["false_refusal_imputation"]
    assert imputed["n"] == 1
    assert section["faithfulness"]["n"] == 0, "a rule-assigned score is not a measurement"
    assert imputed["answer_relevance_including_imputed"]["n"] == 1
    assert imputed["answer_relevance_including_imputed"]["mean"] == 1.0, "it still hurts"

    row = section["per_question"][0]
    assert row["false_refusal"] is True
    assert row["scored_under"] == "false_refusal_imputed"
    assert row["relevance_score"] == 1  # answered nothing
    assert row["faithfulness_score"] == 5  # but invented nothing either


def test_judge_error_is_reported_and_excluded_never_silently_dropped(
    populated: tuple[RAGPipeline, StubJudge, dict[str, Any]],
) -> None:
    class BrokenJudge(StubJudge):
        def complete(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
            if "strict evaluator" in messages[0]["content"]:
                return LLMResponse(
                    text="not json", model="test/judge", prompt_tokens=10, completion_tokens=5
                )
            return super().complete(messages, **kw)

    pipeline, _, dataset = populated
    section = run_answer(pipeline, BrokenJudge(), dataset, judge_model="test/judge", top_k=5)

    # A broken judge no longer empties the section: the run degrades to the
    # local evaluator so the harness still reports rows. What must never happen
    # is the two instruments being pooled, or the degradation being invisible.
    assert section["n_scored"] == 0, "no row may be credited to the LLM judge"
    assert section["faithfulness"]["n"] == 0
    assert section["local_fallback_scores"]["n"] == 2, "the local evaluator covered both"

    degraded = [r for r in section["per_question"] if r.get("degraded")]
    assert len(degraded) == 2, "every degraded row must say so"
    assert all(r["instrument"] == "local-embedding" for r in degraded)
    assert all("unavailable" in r["degraded"] for r in degraded)
