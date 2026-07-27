"""Probes that validate the judge itself.

A 20-point rubric line resting on an unvalidated instrument is an assertion,
not evidence.  Each probe has a *known* correct verdict that follows from the
rubric's own definitions, so a judge that fails one is demonstrably measuring
the wrong thing:

  - ``faithful_but_false``  - the classic conflation. The answer repeats an
    error that is present in the context. Faithfulness must be HIGH (it is
    grounded); the judge must not punish it for being untrue in the world.
  - ``true_but_ungrounded`` - the inverse. The claim is true in the world but
    absent from the context. Faithfulness must be LOW.
  - ``grounded_but_off_topic`` - separates the two axes. Perfectly grounded,
    answers a different question. Faithfulness HIGH, relevance LOW.
  - ``verbose_padding``     - length must not buy a score. A padded restatement
    of a correct short answer must not outscore it.
"""

from __future__ import annotations

from typing import Any

CONTEXT_COST = (
    "[Doc: cost_model_reference.pdf, Chunk: probe1]\n"
    "The marginal compute row used in this reference is therefore zero dollars "
    "at 100,000 vectors, approximately 12 dollars per month at one million, "
    "and approximately 35 to 85 dollars per month at ten million. Attached "
    "block storage on general-purpose SSD is priced at 0.08 dollars per "
    "gigabyte-month."
)

CONTEXT_CHUNKING = (
    "[Doc: ingestion_operations_notes.md, Chunk: probe2]\n"
    "The defaults are a chunk size of 500 characters and an overlap of 50 "
    "characters, roughly ten percent. These are fixed a priori from published "
    "practice and are not selected against the evaluation set."
)

PROBES: list[dict[str, Any]] = [
    {
        "id": "faithful_but_false",
        "question": "What is the marginal compute cost at one million vectors?",
        "context": CONTEXT_COST,
        # Deliberately wrong in the world; exactly what the context says.
        "answer": (
            "The marginal compute cost at one million vectors is approximately "
            "12 dollars per month [Doc: cost_model_reference.pdf, Chunk: probe1]."
        ),
        "expectation": "faithfulness_high",
        "rationale": (
            "Grounded in the context verbatim. A judge scoring this low is "
            "grading world-truth instead of groundedness."
        ),
        "assert_faithfulness_min": 4,
    },
    {
        "id": "true_but_ungrounded",
        "question": "What is the marginal compute cost at one million vectors?",
        "context": CONTEXT_COST,
        # Plausible and possibly true, but nowhere in the context.
        "answer": (
            "The marginal compute cost at one million vectors is about 40 dollars "
            "per month on a t3.medium instance in us-east-1, plus a further 15 "
            "dollars for provisioned IOPS."
        ),
        "expectation": "faithfulness_low",
        "rationale": "No span of the context supports these figures.",
        "assert_faithfulness_max": 3,
    },
    {
        "id": "grounded_but_off_topic",
        "question": "What is the default chunk overlap?",
        "context": CONTEXT_CHUNKING,
        # Every claim is in the context; it answers a different question.
        "answer": (
            "The defaults were fixed a priori from published practice and were "
            "not selected against the evaluation set "
            "[Doc: ingestion_operations_notes.md, Chunk: probe2]."
        ),
        "expectation": "faithful_high_relevance_low",
        "rationale": (
            "The two axes must move independently: grounded, but does not "
            "state the overlap the question asked for."
        ),
        "assert_faithfulness_min": 4,
        "assert_relevance_max": 3,
    },
    {
        "id": "verbose_padding",
        "question": "What is the default chunk overlap?",
        "context": CONTEXT_CHUNKING,
        "answer": (
            "This is an excellent and very important question about the "
            "configuration of the ingestion pipeline, and it deserves a "
            "thorough treatment. Chunking is one of the most consequential "
            "decisions in any retrieval-augmented generation system, affecting "
            "precision, recall, and cost alike. Having considered the matter "
            "carefully and at length, and weighing the many trade-offs "
            "involved in the design of such systems, the answer is that the "
            "default chunk overlap is 50 characters, roughly ten percent of the "
            "500-character chunk size "
            "[Doc: ingestion_operations_notes.md, Chunk: probe2]."
        ),
        "expectation": "relevance_not_inflated_by_length",
        "rationale": (
            "Correct but padded. Compared against the terse-correct control "
            "below; length must not buy a higher score."
        ),
        "control_answer": (
            "The default chunk overlap is 50 characters "
            "[Doc: ingestion_operations_notes.md, Chunk: probe2]."
        ),
    },
]


def check_expectation(probe: dict[str, Any], verdict: Any) -> tuple[bool, str]:
    """Did the judge behave as the rubric requires? Returns ``(passed, note)``."""
    if not verdict.ok:
        return False, f"judge errored: {verdict.error}"

    failures: list[str] = []
    minimum = probe.get("assert_faithfulness_min")
    if minimum is not None and verdict.faithfulness_score < minimum:
        failures.append(f"faithfulness {verdict.faithfulness_score} < required {minimum}")
    maximum = probe.get("assert_faithfulness_max")
    if maximum is not None and verdict.faithfulness_score > maximum:
        failures.append(f"faithfulness {verdict.faithfulness_score} > allowed {maximum}")
    relevance_max = probe.get("assert_relevance_max")
    if relevance_max is not None and verdict.relevance_score > relevance_max:
        failures.append(f"relevance {verdict.relevance_score} > allowed {relevance_max}")

    if failures:
        return False, "; ".join(failures)
    return True, "as expected"
