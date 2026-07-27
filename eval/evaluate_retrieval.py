"""Retrieval-only evaluation over the frozen question set.

Runs no generation, so it costs zero LLM quota and isolates retrieval quality
from generation quality -- the decomposition the Discussion's weak-link
analysis depends on.

Retrieves **once** at k=20 and truncates for the k-sweep, so Recall@{1,3,5,10,20}
and nDCG@{5,10} cost no extra retrievals.

    python -m eval.evaluate_retrieval
    python -m eval.evaluate_retrieval --smoke          # 2 questions, for CI
    python -m eval.evaluate_retrieval --latency-loops 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import METRIC_NAMES, aggregate, evaluate_query, summarise
from src.config import get_settings
from src.rag_pipeline import RAGPipeline

DATASET = REPO_ROOT / "data" / "eval_dataset.json"
RESULTS = REPO_ROOT / "results" / "eval_results.json"

RETRIEVAL_DEPTH = 20
K_SWEEP = (1, 3, 5, 10, 20)
NDCG_KS = (5, 10)


def _nearest_rank(n: int, p: float) -> int:
    """Index of the nearest-rank percentile in a sorted sequence of length ``n``."""
    return min(max(math.ceil(p * n), 1), n) - 1


def load_dataset(path: Path = DATASET) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python scripts/build_eval_dataset.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    pipeline: RAGPipeline,
    dataset: dict[str, Any],
    *,
    primary_k: int,
    latency_loops: int = 1,
) -> dict[str, Any]:
    """Score every question. Returns the retrieval section of the results file."""
    per_question: list[dict[str, Any]] = []
    judgeable_rows: list[dict[str, float]] = []
    sweep_rows: dict[int, list[dict[str, float]]] = {k: [] for k in K_SWEEP}
    latencies_ms: list[float] = []

    for item in dataset["questions"]:
        relevant = set(item["relevant_chunk_ids"])

        started = time.perf_counter()
        hits, _ = pipeline.retrieve(item["question"], top_k=RETRIEVAL_DEPTH)
        latencies_ms.append((time.perf_counter() - started) * 1000)

        # Extra loops for a stable p95: p95 of 15 observations is just the max.
        for _ in range(max(0, latency_loops - 1)):
            started = time.perf_counter()
            pipeline.retrieve(item["question"], top_k=RETRIEVAL_DEPTH)
            latencies_ms.append((time.perf_counter() - started) * 1000)

        retrieved_ids = [h.chunk_id for h in hits]
        row: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "expects_fallback": item["expects_fallback"],
            "n_relevant": len(relevant),
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            "retrieved_chunk_ids": retrieved_ids[:primary_k],
            "retrieved_top1_similarity": round(hits[0].similarity, 4) if hits else None,
            "retrieved_sources": [h.source for h in hits[:primary_k]],
        }

        if not relevant:
            # Unjudgeable for every rank metric. Excluded here, scored under
            # fallback correctness in evaluate_answer.py.
            row["judgeable"] = False
            row["excluded_reason"] = "empty relevant set (out-of-corpus probe)"
            per_question.append(row)
            continue

        row["judgeable"] = True
        scores = evaluate_query(retrieved_ids, relevant, primary_k)
        row["metrics"] = {name: round(scores[name], 4) for name in scores}
        judgeable_rows.append(scores)

        for k in K_SWEEP:
            sweep_rows[k].append(evaluate_query(retrieved_ids, relevant, k))
        per_question.append(row)

    aggregated = aggregate(judgeable_rows)
    sweep = {
        f"k={k}": {
            "recall": summarise([r["recall"] for r in rows]),
            "hit_rate": summarise([r["hit_rate"] for r in rows]),
            "mrr": summarise([r["mrr"] for r in rows]),
            **({"ndcg": summarise([r["ndcg"] for r in rows])} if k in NDCG_KS else {}),
            "context_precision": summarise([r["context_precision"] for r in rows]),
            "max_achievable_precision": summarise([r["max_achievable_precision"] for r in rows]),
        }
        for k, rows in sweep_rows.items()
    }

    ordered = sorted(latencies_ms)
    return {
        "primary_k": primary_k,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "n_questions": len(dataset["questions"]),
        "n_ranked": len(judgeable_rows),
        "n_fallback": len(dataset["questions"]) - len(judgeable_rows),
        "aggregate": aggregated,
        "k_sweep": sweep,
        "retrieval_latency_ms": {
            "n_samples": len(ordered),
            "p50": round(statistics.median(ordered), 3),
            # Nearest-rank (ceil), not floor: the floor form returns the 14th of
            # 15 samples for both p95 and p99 at the default --latency-loops 1.
            "p95": round(ordered[_nearest_rank(len(ordered), 0.95)], 3),
            "p99": round(ordered[_nearest_rank(len(ordered), 0.99)], 3),
            "mean": round(statistics.fmean(ordered), 3),
            "min": round(ordered[0], 3),
            "max": round(ordered[-1], 3),
        },
        "per_question": per_question,
    }


def print_summary(section: dict[str, Any]) -> None:
    agg = section["aggregate"]
    print(
        f"\nRetrieval @ k={section['primary_k']}  "
        f"(n_ranked={section['n_ranked']}, n_fallback={section['n_fallback']})"
    )
    print("-" * 68)
    print(f"{'metric':<24}{'mean':>8}{'std':>8}{'95% CI':>22}")
    for name in METRIC_NAMES:
        s = agg[name]
        # Built separately rather than nested in the f-string: reusing the outer
        # quote inside an f-string is 3.12+ syntax, and this project targets 3.10.
        ci = "[{:.3f}, {:.3f}]".format(s["ci_low"], s["ci_high"])
        print(f"{name:<24}{s['mean']:>8.4f}{s['std']:>8.4f}{ci:>22}")
    ceiling = agg["max_achievable_precision"]["mean"]
    print(
        f"\ncontext_precision ceiling  {ceiling:.4f}  "
        f"-> achieved {agg['precision_ratio_of_achievable']:.1%} of achievable"
    )

    print(f"\n{'k':>4}{'recall':>10}{'hit_rate':>10}{'mrr':>10}{'precision':>11}")
    print("-" * 46)
    for key, row in section["k_sweep"].items():
        k = key.split("=")[1]
        print(
            f"{k:>4}{row['recall']['mean']:>10.4f}{row['hit_rate']['mean']:>10.4f}"
            f"{row['mrr']['mean']:>10.4f}{row['context_precision']['mean']:>11.4f}"
        )

    lat = section["retrieval_latency_ms"]
    print(
        f"\nretrieval latency (n={lat['n_samples']}): "
        f"p50={lat['p50']}ms  p95={lat['p95']}ms  p99={lat['p99']}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="first 2 questions only")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--latency-loops",
        type=int,
        default=1,
        help="repeat each retrieval N times for a stable p95",
    )
    parser.add_argument("--output", default=str(RESULTS))
    args = parser.parse_args()

    settings = get_settings()
    dataset = load_dataset()
    if args.smoke:
        dataset = {**dataset, "questions": dataset["questions"][:2]}

    pipeline = RAGPipeline()
    if pipeline.store.count() == 0:
        print("Vector store is empty. Run: python scripts/ingest_corpus.py --reset")
        return 1

    section = run(
        pipeline,
        dataset,
        primary_k=args.top_k or settings.default_top_k,
        latency_loops=args.latency_loops,
    )
    print_summary(section)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    existing["retrieval"] = section
    existing.setdefault("meta", {})
    existing["meta"].update(
        {
            "embedding_model": dataset["embedding_model"],
            "embedding_dim": dataset["embedding_dim"],
            "chunking": dataset["chunking"],
            "vector_backend": settings.vector_backend,
            "similarity_threshold": settings.similarity_threshold,
            "corpus_chunk_count": dataset["corpus_chunk_count"],
            "retrieval_run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    # allow_nan=False: a bare NaN token is valid Python but not valid JSON
    # (RFC 8259 has no such literal), so a run with an empty metric would emit
    # an artifact that jq, JavaScript, Go and serde all reject. Fail loudly here
    # rather than ship a results file no standard parser can open.
    output.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
