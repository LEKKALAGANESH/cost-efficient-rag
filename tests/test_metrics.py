"""IR metrics against a hand-computed toy fixture.

Every expected value below was computed by hand and the arithmetic is shown in
the docstring, so a reviewer can check the test rather than trusting it.  A bug
in these functions produces a plausible-looking but wrong number that nothing
else in the pipeline would flag, which is why they are tested before ever being
pointed at the real question set.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    aggregate,
    bootstrap_ci,
    context_precision_at_k,
    dcg_at_k,
    evaluate_query,
    hit_rate,
    max_achievable_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)

# Toy corpus: chunks c1..c8. Rankings are written as explicit lists so the
# expected values can be recomputed by eye.
PERFECT = ["c1", "c2", "c3", "c4", "c5"]


# ---------------------------------------------------------------------------
# Hit Rate
# ---------------------------------------------------------------------------
def test_hit_rate_one_when_any_relevant_present() -> None:
    assert hit_rate(["c9", "c8", "c3"], {"c3"}, k=5) == 1.0


def test_hit_rate_zero_when_none_present() -> None:
    assert hit_rate(["c9", "c8", "c7"], {"c3"}, k=5) == 0.0


def test_hit_rate_respects_k_truncation() -> None:
    """c3 sits at rank 4, so it is invisible at k=3."""
    assert hit_rate(["c9", "c8", "c7", "c3"], {"c3"}, k=3) == 0.0
    assert hit_rate(["c9", "c8", "c7", "c3"], {"c3"}, k=4) == 1.0


def test_hit_rate_is_one_even_when_recall_is_partial() -> None:
    """The exact case where Hit Rate and Recall@k must diverge."""
    retrieved, relevant = ["c1", "c9", "c8"], {"c1", "c2", "c3", "c4"}
    assert hit_rate(retrieved, relevant, k=3) == 1.0
    assert recall_at_k(retrieved, relevant, k=3) == 0.25


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------
def test_recall_single_relevant_found() -> None:
    assert recall_at_k(["c9", "c1"], {"c1"}, k=5) == 1.0


def test_recall_multi_relevant_partial() -> None:
    """2 of 4 relevant retrieved -> 2/4 = 0.5."""
    assert recall_at_k(["c1", "c9", "c3", "c8"], {"c1", "c2", "c3", "c4"}, k=5) == 0.5


def test_recall_cannot_exceed_one_with_duplicates() -> None:
    """A duplicated ID in the ranking must not double-count."""
    assert recall_at_k(["c1", "c1", "c1"], {"c1"}, k=5) == 1.0


def test_recall_when_relevant_exceeds_k_is_capped_by_k() -> None:
    """7 relevant, k=5: the best possible recall is 5/7."""
    relevant = {f"c{i}" for i in range(1, 8)}
    assert recall_at_k(PERFECT, relevant, k=5) == pytest.approx(5 / 7)


def test_recall_raises_on_empty_relevant_set() -> None:
    """0/0 must be a loud error, not a silent NaN poisoning every mean."""
    with pytest.raises(ValueError, match="undefined for an empty relevant set"):
        recall_at_k(["c1"], set(), k=5)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        (["c1", "c9", "c8"], 1.0),  # rank 1 -> 1/1
        (["c9", "c1", "c8"], 0.5),  # rank 2 -> 1/2
        (["c9", "c8", "c1"], 1 / 3),  # rank 3 -> 1/3
        (["c9", "c8", "c7"], 0.0),  # absent -> 0, NOT excluded
    ],
)
def test_reciprocal_rank(retrieved: list[str], expected: float) -> None:
    assert reciprocal_rank(retrieved, {"c1"}, k=5) == pytest.approx(expected)


def test_reciprocal_rank_uses_first_relevant_only() -> None:
    """c3 at rank 2 wins; c1 at rank 3 is irrelevant to the score."""
    assert reciprocal_rank(["c9", "c3", "c1"], {"c1", "c3"}, k=5) == 0.5


def test_reciprocal_rank_beyond_k_scores_zero() -> None:
    assert reciprocal_rank(["c9", "c8", "c7", "c1"], {"c1"}, k=3) == 0.0


# ---------------------------------------------------------------------------
# nDCG@k  -- including the truncated-IDCG guard
# ---------------------------------------------------------------------------
def test_dcg_rank_one_is_undiscounted() -> None:
    """log2(1+1) == 1, so a rank-1 hit contributes exactly 1.0."""
    assert dcg_at_k(["c1"], {"c1"}, k=5) == 1.0


def test_dcg_discounts_by_log2_of_rank_plus_one() -> None:
    """Hit at rank 3 -> 1/log2(4) = 0.5."""
    assert dcg_at_k(["c9", "c8", "c1"], {"c1"}, k=5) == pytest.approx(0.5)


def test_ndcg_perfect_single_relevant_is_one() -> None:
    assert ndcg_at_k(["c1", "c9"], {"c1"}, k=5) == 1.0


def test_ndcg_hand_computed_partial() -> None:
    """Relevant {c1,c2}; ranking [c9, c1, c8, c2, c7], k=5.

    DCG  = 1/log2(3) + 1/log2(5) = 0.6309298 + 0.4306766 = 1.0616063
    IDCG = 1/log2(2) + 1/log2(3) = 1.0000000 + 0.6309298 = 1.6309298
    nDCG = 1.0616063 / 1.6309298 = 0.6509209
    """
    value = ndcg_at_k(["c9", "c1", "c8", "c2", "c7"], {"c1", "c2"}, k=5)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert value == pytest.approx(expected)
    assert value == pytest.approx(0.6509209, abs=1e-6)


def test_ndcg_is_exactly_one_when_relevant_exceeds_k() -> None:
    """THE truncated-IDCG guard.

    7 relevant chunks, k=5, and the top 5 are all relevant. IDCG@5 must sum
    only min(7, 5) = 5 terms. Summing all 7 would give
    DCG/IDCG = 2.9484591 / 3.6379996 = 0.8105 -- a flawless ranking reported as
    a failure. The correct value is exactly 1.0.
    """
    relevant = {f"c{i}" for i in range(1, 8)}
    assert ndcg_at_k(PERFECT, relevant, k=5) == pytest.approx(1.0)


def test_ndcg_untruncated_idcg_would_have_been_wrong() -> None:
    """Pin the specific wrong number, so a regression is unmistakable."""
    relevant = {f"c{i}" for i in range(1, 8)}
    dcg = dcg_at_k(PERFECT, relevant, k=5)
    untruncated_idcg = sum(1 / math.log2(i + 1) for i in range(1, 8))
    assert dcg / untruncated_idcg == pytest.approx(0.8105, abs=1e-4)
    assert ndcg_at_k(PERFECT, relevant, k=5) == pytest.approx(1.0)


def test_ndcg_zero_when_nothing_relevant_retrieved() -> None:
    assert ndcg_at_k(["c9", "c8"], {"c1"}, k=5) == 0.0


def test_ndcg_raises_on_empty_relevant_set() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(["c1"], set(), k=5)


# ---------------------------------------------------------------------------
# Context precision and its ceiling
# ---------------------------------------------------------------------------
def test_context_precision_hand_computed() -> None:
    """2 relevant among the 5 fetched -> 2/5 = 0.4."""
    assert context_precision_at_k(["c1", "c9", "c2", "c8", "c7"], {"c1", "c2"}, k=5) == 0.4


def test_context_precision_ceiling_with_one_relevant_at_k5() -> None:
    """The number the README must publish so 0.19 is not misread as failure."""
    assert max_achievable_precision_at_k({"c1"}, k=5) == 0.2
    assert context_precision_at_k(["c1", "c9", "c8", "c7", "c6"], {"c1"}, k=5) == 0.2


def test_context_precision_ceiling_caps_at_one() -> None:
    relevant = {f"c{i}" for i in range(1, 8)}
    assert max_achievable_precision_at_k(relevant, k=5) == 1.0


def test_context_precision_raises_on_empty_relevant_set() -> None:
    with pytest.raises(ValueError):
        context_precision_at_k(["c1"], set(), k=5)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_evaluate_query_returns_every_metric() -> None:
    row = evaluate_query(["c1", "c9", "c2"], {"c1", "c2"}, k=3)
    assert row["hit_rate"] == 1.0
    assert row["recall"] == 1.0
    assert row["mrr"] == 1.0
    assert row["context_precision"] == pytest.approx(2 / 3)
    assert row["max_achievable_precision"] == pytest.approx(2 / 3)


def test_aggregate_reports_mean_std_and_ci() -> None:
    rows = [
        evaluate_query(["c1", "c9", "c8", "c7", "c6"], {"c1"}, k=5),
        evaluate_query(["c9", "c8", "c7", "c6", "c5"], {"c1"}, k=5),
    ]
    out = aggregate(rows)
    assert out["hit_rate"]["n"] == 2
    assert out["hit_rate"]["mean"] == 0.5
    assert "ci_low" in out["hit_rate"] and "ci_high" in out["hit_rate"]


def test_aggregate_reports_precision_ratio_of_achievable() -> None:
    """Achieved 0.2 against a ceiling of 0.2 is 100% of achievable, not 20%."""
    rows = [evaluate_query(["c1", "c9", "c8", "c7", "c6"], {"c1"}, k=5)]
    out = aggregate(rows)
    assert out["context_precision"]["mean"] == 0.2
    assert out["max_achievable_precision"]["mean"] == 0.2
    assert out["precision_ratio_of_achievable"] == 1.0


def test_summarise_of_empty_input_is_not_a_crash() -> None:
    out = summarise([])
    assert out["n"] == 0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first = bootstrap_ci(values)
    assert first == bootstrap_ci(values)  # seeded
    assert first[0] <= sum(values) / len(values) <= first[1]


def test_bootstrap_ci_of_single_value_is_degenerate_not_an_error() -> None:
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


# ---------------------------------------------------------------------------
# The full toy fixture, end to end
# ---------------------------------------------------------------------------
def test_toy_fixture_end_to_end() -> None:
    """Three queries covering the three cases the methodology requires.

    q_single : relevant {c1}, ranking [c9,c1,c8,c7,c6]
        hit=1, recall=1/1=1.0, mrr=1/2=0.5,
        ndcg = (1/log2(3)) / 1 = 0.6309298, precision = 1/5 = 0.2, ceiling 0.2
    q_multi  : relevant {c1,c2,c3}, ranking [c1,c2,c9,c8,c7]
        hit=1, recall=2/3=0.666667, mrr=1.0,
        DCG = 1 + 1/log2(3) = 1.6309298
        IDCG(min(3,5)=3) = 1 + 1/log2(3) + 1/log2(4) = 2.1309298
        ndcg = 0.7653606, precision = 2/5 = 0.4, ceiling = 3/5 = 0.6
    q_over_k : relevant c1..c7, ranking [c1..c5]  (|relevant| > k)
        hit=1, recall=5/7=0.714286, mrr=1.0, ndcg=1.0,
        precision = 5/5 = 1.0, ceiling = 1.0
    """
    single = evaluate_query(["c9", "c1", "c8", "c7", "c6"], {"c1"}, k=5)
    assert single["recall"] == 1.0
    assert single["mrr"] == 0.5
    assert single["ndcg"] == pytest.approx(0.6309298, abs=1e-6)
    assert single["context_precision"] == pytest.approx(0.2)

    multi = evaluate_query(["c1", "c2", "c9", "c8", "c7"], {"c1", "c2", "c3"}, k=5)
    assert multi["recall"] == pytest.approx(2 / 3)
    assert multi["mrr"] == 1.0
    assert multi["ndcg"] == pytest.approx(0.7653606, abs=1e-6)
    assert multi["context_precision"] == pytest.approx(0.4)
    assert multi["max_achievable_precision"] == pytest.approx(0.6)

    over_k = evaluate_query(PERFECT, {f"c{i}" for i in range(1, 8)}, k=5)
    assert over_k["recall"] == pytest.approx(5 / 7)
    assert over_k["ndcg"] == pytest.approx(1.0)
    assert over_k["context_precision"] == 1.0

    out = aggregate([single, multi, over_k])
    expected_recall = (1.0 + 2 / 3 + 5 / 7) / 3
    assert out["recall"]["mean"] == pytest.approx(round(expected_recall, 4))
    assert out["recall"]["n"] == 3


def test_zero_relevant_query_is_excluded_not_scored_as_zero() -> None:
    """The out-of-corpus question must never reach these functions.

    Every rank metric raises on an empty relevant set rather than returning
    0.0, so the harness cannot silently fold an unjudgeable query into a mean.
    """
    # hit_rate belongs in this loop. It was the one metric that returned a
    # silent 0.0 instead, protected only by dict-literal evaluation order in
    # evaluate_query -- recall_at_k happened to raise on the next line. That is
    # a guarantee by accident, and it is exactly the "silent 0.0" hole the
    # module docstring claims to have closed.
    for func in (hit_rate, recall_at_k, reciprocal_rank, ndcg_at_k, context_precision_at_k):
        with pytest.raises(ValueError):
            func(["c1", "c2"], set(), 5)
