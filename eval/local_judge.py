"""Local, offline answer scorer: the last link in the evaluation chain.

**Read this before reading a number it produced.** This is *not* an LLM judge
and its scores are not interchangeable with one. It is a deterministic,
embedding-based evaluator that runs with no network and no credential, and it
exists for one reason: an evaluation harness whose only instrument is a remote
API produces nothing at all when that API is unreachable. That is what happened
here — one dead credential emptied an entire rubric line.

So the chain degrades rather than aborts, and every artifact records **which
instrument produced each score** so the two are never silently pooled.

What it measures, and how honestly:

* **Groundedness** — maximum cosine similarity between each answer sentence and
  the retrieved context sentences, averaged over sentences. A sentence with no
  close support in the context drags the score down. This is a genuine signal
  and a weak proxy for entailment: it detects *unsupported* content, not
  *contradicted* content. A fluent sentence that contradicts the context while
  reusing its vocabulary will score high. An NLI cross-encoder would catch that;
  cosine similarity cannot, and pretending otherwise would be the same
  overclaiming this evaluator was written to avoid.

* **Answer relevance** — cosine similarity between the question and the answer.
  Standard, and it fails in the standard way: a fluent, on-topic, wrong answer
  scores well.

Both are reported on the same 1-5 scale as the LLM judge so the columns line up,
with `instrument: "local-embedding"` on every row. The right reading of a local
score is "nothing here is obviously ungrounded", not "this answer is correct".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.embeddings import embedding_service

#: Cosine similarity thresholds mapping to the 1-5 rubric scale. Calibrated so
#: that verbatim-supported text lands at 5 and unrelated text at 1; the interior
#: bands are deliberately wide because this instrument cannot resolve finer
#: distinctions and a 7-point claim from a 2-signal metric would be false
#: precision.
_SCORE_BANDS: tuple[tuple[float, int], ...] = (
    (0.80, 5),
    (0.65, 4),
    (0.50, 3),
    (0.35, 2),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str, *, min_chars: int = 15) -> list[str]:
    """Split into scoreable units. Fragments shorter than ``min_chars`` are
    dropped: a citation marker or a stray bullet is not a claim, and scoring it
    as one adds noise in whichever direction it happens to fall."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if len(s.strip()) >= min_chars]


def _to_score(similarity: float) -> int:
    for threshold, score in _SCORE_BANDS:
        if similarity >= threshold:
            return score
    return 1


@dataclass
class LocalVerdict:
    """One locally-scored answer. Field names mirror the LLM judge's verdict so
    both can be aggregated by the same code, with ``instrument`` keeping them
    distinguishable in every report."""

    faithfulness_score: int
    faithfulness_rationale: str
    relevance_score: int
    relevance_rationale: str
    instrument: str = "local-embedding"
    ok: bool = True
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    measured: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness_score": self.faithfulness_score,
            "faithfulness_rationale": self.faithfulness_rationale,
            "relevance_score": self.relevance_score,
            "relevance_rationale": self.relevance_rationale,
            "instrument": self.instrument,
            "measured": {k: round(v, 4) for k, v in self.measured.items()},
        }


def _cosine_matrix(left: list[str], right: list[str]) -> np.ndarray:
    """Pairwise cosine similarity. Vectors are L2-normalised by the embedding
    service at both ends, so a dot product *is* the cosine."""
    left_vectors = np.asarray(embedding_service.embed_texts(left), dtype=np.float32)
    right_vectors = np.asarray(embedding_service.embed_texts(right), dtype=np.float32)
    return left_vectors @ right_vectors.T


def judge_answer_locally(question: str, context: str, answer: str) -> LocalVerdict:
    """Score one answer with no network access.

    Returns a verdict even for degenerate input rather than raising: the whole
    point of this path is that the harness keeps producing rows.
    """
    answer_sentences = _sentences(answer)
    context_sentences = _sentences(context, min_chars=10)

    if not answer_sentences:
        return LocalVerdict(
            faithfulness_score=1,
            faithfulness_rationale="The answer contains no scoreable sentence.",
            relevance_score=1,
            relevance_rationale="The answer contains no scoreable sentence.",
            measured={"groundedness": 0.0, "relevance": 0.0},
        )

    if not context_sentences:
        # No context is not the same as an ungrounded answer, and scoring it as
        # faithfulness=1 would blame the generator for a retrieval failure.
        relevance = float(_cosine_matrix([question], [" ".join(answer_sentences)])[0, 0])
        return LocalVerdict(
            faithfulness_score=1,
            faithfulness_rationale=(
                "No retrieved context to check against, so nothing in the answer is supported. "
                "This is a retrieval failure, not evidence that the generator hallucinated."
            ),
            relevance_score=_to_score(relevance),
            relevance_rationale=f"Question-answer cosine similarity {relevance:.3f}.",
            measured={"groundedness": 0.0, "relevance": relevance},
        )

    # Groundedness: every answer sentence needs *some* close support.
    support = _cosine_matrix(answer_sentences, context_sentences)
    best_support_per_sentence = support.max(axis=1)
    groundedness = float(best_support_per_sentence.mean())
    weakest = float(best_support_per_sentence.min())

    relevance = float(_cosine_matrix([question], [" ".join(answer_sentences)])[0, 0])

    return LocalVerdict(
        faithfulness_score=_to_score(groundedness),
        faithfulness_rationale=(
            f"Mean best-match similarity between each of {len(answer_sentences)} answer "
            f"sentence(s) and the retrieved context is {groundedness:.3f}; the least-supported "
            f"sentence scores {weakest:.3f}. Similarity detects unsupported content, not "
            f"contradiction."
        ),
        relevance_score=_to_score(relevance),
        relevance_rationale=f"Question-answer cosine similarity {relevance:.3f}.",
        measured={
            "groundedness": groundedness,
            "weakest_sentence_support": weakest,
            "relevance": relevance,
            "n_answer_sentences": float(len(answer_sentences)),
        },
    )
