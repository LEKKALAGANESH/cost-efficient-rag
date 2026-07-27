"""Exact Match and token F1.

Offline and pure -- no model, no store, no key.
"""

from __future__ import annotations

import pytest

from eval.text_metrics import exact_match, normalize_answer, token_f1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Cost Model", "cost model"),
        ("a read unit, per query.", "read unit per query"),
        ("  spaced   out  ", "spaced out"),
        ("UPPER lower", "upper lower"),
    ],
)
def test_normalisation_strips_case_punctuation_and_articles(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_citations_are_stripped_before_comparison() -> None:
    """Citations are this system's output format, not content the gold answer
    could ever contain. Left in, every EM would be 0 and every F1 diluted."""
    answer = "One read unit per gigabyte. [Doc: cost_model_reference.pdf, Chunk: abc123]"
    assert normalize_answer(answer) == "one read unit per gigabyte"


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------
def test_exact_match_ignores_surface_differences() -> None:
    assert exact_match("The answer is 42.", "answer is 42") == 1.0


def test_exact_match_is_zero_for_different_content() -> None:
    assert exact_match("The answer is 42.", "The answer is 43.") == 0.0


def test_exact_match_is_zero_when_the_prediction_merely_contains_the_gold() -> None:
    """EM is equality, not containment -- a verbose but correct answer scores 0.
    That is the metric's defining weakness here, so it is pinned deliberately."""
    assert exact_match("Well, the answer is 42, as the document states.", "the answer is 42") == 0.0


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------
def test_f1_is_one_for_an_exact_match() -> None:
    assert token_f1("read units per gigabyte", "read units per gigabyte") == 1.0


def test_f1_is_zero_when_no_tokens_are_shared() -> None:
    assert token_f1("chocolate cake recipe", "vector store latency") == 0.0


def test_f1_rewards_partial_overlap() -> None:
    score = token_f1("one read unit per gigabyte scanned", "one read unit per gigabyte")
    assert 0.0 < score < 1.0
    # 5 shared / 6 predicted, 5 shared / 5 gold -> 2*(5/6)*(1)/((5/6)+1)
    assert score == pytest.approx(2 * (5 / 6) * 1.0 / ((5 / 6) + 1.0))


def test_repeated_tokens_are_not_double_counted() -> None:
    """Counter intersection caps credit at the gold multiplicity; without it,
    padding an answer with one gold word would inflate recall."""
    assert token_f1("cost cost cost cost", "cost model") == pytest.approx(2 * 0.25 * 0.5 / 0.75)


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("", "", 1.0),  # both empty: identical
        ("", "something", 0.0),  # nothing shared
        ("something", "", 0.0),
    ],
)
def test_empty_strings_do_not_divide_by_zero(pred: str, gold: str, expected: float) -> None:
    assert token_f1(pred, gold) == expected


def test_f1_is_symmetric() -> None:
    a, b = "one read unit per gigabyte", "read unit per gigabyte scanned"
    assert token_f1(a, b) == pytest.approx(token_f1(b, a))
