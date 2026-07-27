"""Pick the fallback threshold tau from evidence, not from a guess.

Scores top-1 similarity for in-corpus and out-of-corpus probes (which are
disjoint from the frozen eval set), sweeps candidate thresholds, and reports
the confusion matrix at each.  Writes ``results/threshold_calibration.json``.

    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rag_pipeline import RAGPipeline

PROBES = REPO_ROOT / "data" / "threshold_calibration.json"
OUTPUT = REPO_ROOT / "results" / "threshold_calibration.json"
CANDIDATES = [round(0.05 * i, 2) for i in range(1, 13)]  # 0.05 .. 0.60
MAX_FALSE_REFUSAL_RATE = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()

    def top1(question: str) -> float:
        hits, _ = pipeline.retrieve(question, top_k=args.top_k)
        return hits[0].similarity if hits else -1.0

    in_scores = [top1(q) for q in probes["in_corpus"]]
    out_scores = [top1(q) for q in probes["out_of_corpus"]]

    print(
        f"in-corpus  n={len(in_scores):<3} "
        f"mean={statistics.mean(in_scores):.3f} min={min(in_scores):.3f} max={max(in_scores):.3f}"
    )
    print(
        f"out-corpus n={len(out_scores):<3} "
        f"mean={statistics.mean(out_scores):.3f} min={min(out_scores):.3f} max={max(out_scores):.3f}"
    )
    print(f"\n{'tau':>6} {'false_refusal':>14} {'false_answer':>13} {'accuracy':>9}")
    print("-" * 46)

    sweep: list[dict[str, float]] = []
    for tau in CANDIDATES:
        false_refusals = sum(1 for s in in_scores if s < tau)  # answerable, refused
        false_answers = sum(1 for s in out_scores if s >= tau)  # unanswerable, answered
        row = {
            "tau": tau,
            "false_refusals": false_refusals,
            "false_refusal_rate": round(false_refusals / len(in_scores), 4),
            "false_answers": false_answers,
            "false_answer_rate": round(false_answers / len(out_scores), 4),
            "accuracy": round(
                ((len(in_scores) - false_refusals) + (len(out_scores) - false_answers))
                / (len(in_scores) + len(out_scores)),
                4,
            ),
        }
        sweep.append(row)
        print(
            f"{tau:>6.2f} {row['false_refusal_rate']:>14.2%} "
            f"{row['false_answer_rate']:>13.2%} {row['accuracy']:>9.2%}"
        )

    # Selection rule, in two stages.
    #
    # (1) Keep only thresholds whose false-refusal rate is within budget --
    #     wrongly refusing an answerable question is the costlier error here,
    #     because the model's own prompted abstention still catches the other
    #     direction downstream.
    # (2) Among those, take the one furthest from BOTH classes rather than the
    #     largest. The largest admissible tau sits flush against the lowest
    #     in-corpus score, so a single slightly-harder question would tip into
    #     a false refusal. Maximising the margin is what makes the choice
    #     robust rather than merely optimal on this sample.
    eligible = [r for r in sweep if r["false_refusal_rate"] <= MAX_FALSE_REFUSAL_RATE]
    if eligible:
        best_accuracy = max(r["accuracy"] for r in eligible)
        top = [r for r in eligible if r["accuracy"] == best_accuracy]
        chosen = max(
            top,
            key=lambda r: min(
                min(s for s in in_scores) - r["tau"],  # headroom above out-of-corpus
                r["tau"] - max(s for s in out_scores),
            ),
        )
        rationale = (
            "highest accuracy within the false-refusal budget, and among those "
            "ties the threshold furthest from both classes"
        )
    else:
        chosen = min(sweep, key=lambda r: r["false_refusal_rate"])
        rationale = "no threshold met the false-refusal budget; minimising it instead"

    margin = min(min(in_scores) - chosen["tau"], chosen["tau"] - max(out_scores))
    print(f"\nSelected tau = {chosen['tau']} ({rationale})")
    print(
        f"  separating band: [{max(out_scores):.3f}, {min(in_scores):.3f}]"
        f"  margin to nearer class: {margin:.3f}"
    )
    print("Set SIMILARITY_THRESHOLD in .env to this value.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "method": (
                    "Top-1 cosine similarity scored over in-corpus and out-of-corpus "
                    "probes disjoint from the frozen eval set. Among thresholds whose "
                    f"false-refusal rate stays at or below {MAX_FALSE_REFUSAL_RATE:.0%}, "
                    "tau is the highest-accuracy candidate that is furthest from both "
                    "classes, so the choice is robust rather than flush against the "
                    "lowest in-corpus score."
                ),
                "caveat": (
                    "The clean separation here is partly an artefact of probe design: "
                    "the out-of-corpus probes are far off-topic. Near-miss topical "
                    "questions would land inside the 0.25-0.45 band where relevant and "
                    "merely-topical pairs overlap. tau is therefore treated as a cheap "
                    "pre-filter, and the model's own prompted abstention as the primary "
                    "correctness mechanism."
                ),
                "selection_rationale": rationale,
                "separating_band": [round(max(out_scores), 4), round(min(in_scores), 4)],
                "margin_to_nearer_class": round(margin, 4),
                "top_k": args.top_k,
                "n_in_corpus": len(in_scores),
                "n_out_of_corpus": len(out_scores),
                "in_corpus_scores": [round(s, 4) for s in in_scores],
                "out_of_corpus_scores": [round(s, 4) for s in out_scores],
                "in_corpus_mean": round(statistics.mean(in_scores), 4),
                "out_of_corpus_mean": round(statistics.mean(out_scores), 4),
                "sweep": sweep,
                "selected_tau": chosen["tau"],
                "confusion_at_selected_tau": chosen,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
