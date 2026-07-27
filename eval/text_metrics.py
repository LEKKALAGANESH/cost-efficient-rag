"""Exact Match and token-level F1 against gold answers.

Both are SQuAD-style: normalise, then compare. They are reported because the
assignment asks for them, but they are the weakest instruments in this report
and the numbers should be read with that in mind.

EM asks whether a generated sentence is *string-identical* to the gold one
after normalisation. This system answers in grounded prose with inline
citations; the gold answers are single declarative sentences written by hand.
Two answers can be equally correct and share no surface form, so EM here is
close to a formatting check. Token F1 is more forgiving -- it scores bag-of-
token overlap -- but it still rewards echoing the gold phrasing rather than
being right, and it cannot see a citation, a hedge, or a contradiction.

They are kept because a low EM alongside a high groundedness score is itself
informative: it says the system is not parroting the gold text.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")

#: Citations are part of a correct answer here, but they are this system's
#: formatting rather than content the gold answer could ever contain. Left in,
#: every EM would be 0 and every F1 diluted by tokens the gold cannot match.
_CITATION = re.compile(r"\[Doc:[^\]]*\]")


def normalize_answer(text: str) -> str:
    """Lowercase, drop citations/articles/punctuation, collapse whitespace."""
    text = _CITATION.sub(" ", text)
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(prediction: str, gold: str) -> float:
    """1.0 when the normalised strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Harmonic mean of token precision and recall over normalised tokens.

    Multiplicity is preserved (``Counter`` intersection), so a prediction that
    repeats a gold token five times gets credit once, not five times.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    # Two empty strings are identical, so F1 is 1.0. One empty and one not
    # shares no tokens, so it is 0.0. Falling through to the formula below
    # would divide by zero in both cases.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    shared = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
