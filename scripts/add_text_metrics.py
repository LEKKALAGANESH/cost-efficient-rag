"""Add Exact Match / token F1 to results/eval_results.json.

Separate from `eval.evaluate_answer` on purpose: EM and F1 need only the stored
answers and the gold set, so they can be recomputed without re-running
generation. Re-running generation costs free-tier quota and would change the
answers being scored, which is the wrong way to add a metric to a finished run.

    python scripts/add_text_metrics.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.text_metrics import exact_match, token_f1

RESULTS = REPO_ROOT / "results" / "eval_results.json"
DATASET = REPO_ROOT / "data" / "eval_dataset.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    results_path = Path(args.results)
    report = json.loads(results_path.read_text(encoding="utf-8"))
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    gold = {
        q["id"]: q["ground_truth_answer"]
        for q in dataset["questions"]
        if not q["expects_fallback"] and q.get("ground_truth_answer")
    }

    rows = report.get("answer", {}).get("per_question", [])
    if not rows:
        print("No answer section in the results file; run eval.evaluate_answer first.")
        return 1

    em: list[float] = []
    f1: list[float] = []
    scored_ids: list[str] = []
    for row in rows:
        reference = gold.get(row["id"])
        # Skip the out-of-corpus question (no gold answer exists) and any
        # question the system refused: scoring a refusal against a gold answer
        # measures the refusal policy, which fallback_correctness already
        # reports, not the quality of an answer.
        if not reference or row.get("fallback_triggered") or not (row.get("answer") or "").strip():
            continue
        em.append(exact_match(row["answer"], reference))
        f1.append(token_f1(row["answer"], reference))
        scored_ids.append(row["id"])

    if not em:
        print("No answered questions with a gold answer; nothing to score.")
        return 1

    section = {
        "n": len(em),
        "scored_question_ids": scored_ids,
        "exact_match": round(statistics.fmean(em), 4),
        "token_f1": round(statistics.fmean(f1), 4),
        "token_f1_min": round(min(f1), 4),
        "token_f1_max": round(max(f1), 4),
        "definition": (
            "SQuAD-style. Both normalise by lowercasing and removing inline "
            "citations, articles and punctuation. EM is string equality after "
            "normalisation; token_f1 is the harmonic mean of bag-of-token "
            "precision and recall, with multiplicity preserved."
        ),
        "excluded": (
            "Refused questions and the out-of-corpus question are excluded: "
            "scoring a refusal against a gold answer measures the refusal "
            "policy, which fallback_correctness already reports separately."
        ),
        "caveat": (
            "Weak instruments for this task. Answers are grounded prose with "
            "citations; gold answers are hand-written single sentences. Two "
            "answers can be equally correct and share little surface form, so "
            "EM approximates a formatting check and F1 rewards echoing the "
            "gold phrasing. Reported because the assignment asks for them, and "
            "read alongside groundedness rather than instead of it."
        ),
    }

    if args.check:
        print(json.dumps(section, indent=2))
        return 0

    report["answer"]["text_overlap"] = section
    results_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"n={section['n']}  EM={section['exact_match']}  token_F1={section['token_f1']}")
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
