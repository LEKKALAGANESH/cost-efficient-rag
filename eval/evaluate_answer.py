"""Answer-quality evaluation: faithfulness, relevance, fallback correctness.

Runs each question through the *same* ``rag_pipeline`` the API uses -- never a
parallel hand-copied eval path that could silently drift from what is
deployed -- then scores the answer with a cross-family judge.

Out-of-corpus questions are not scored on faithfulness or relevance.  They are
scored under **fallback correctness**: did the exact refusal string come back,
and was the generation model left uncalled?

    python -m eval.evaluate_answer
    python -m eval.evaluate_answer --smoke        # 2 questions, for CI
    python -m eval.evaluate_answer --probes-only  # judge validation only
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

from eval.adversarial_probes import PROBES, check_expectation
from eval.judge import judge_answer
from eval.local_judge import judge_answer_locally
from eval.metrics import summarise
from src.config import FALLBACK_ANSWER, get_settings
from src.llm_client import LLMError, get_llm_client
from src.providers import has_credential
from src.rag_pipeline import RAGPipeline

DATASET = REPO_ROOT / "data" / "eval_dataset.json"
RESULTS = REPO_ROOT / "results" / "eval_results.json"


def _pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: index ``ceil(p*n) - 1`` over sorted values.

    The floor-based form used elsewhere in the harness is off by one whenever
    ``p*n`` is not an integer -- at n=15 it returns the 14th of 15 for both p95
    and p99.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(p * len(ordered))
    return round(ordered[min(max(rank, 1), len(ordered)) - 1], 3)


def judge_with_fallback(
    llm_client: Any,
    question: str,
    context: str,
    answer: str,
    *,
    model: str,
    allow_local: bool = True,
) -> Any:
    """Judge one answer, degrading to the local evaluator rather than aborting.

    A remote judge is preferred and tried first. When it fails -- bad key, dead
    quota, provider outage -- the alternative is not "no number", it is a number
    from a weaker instrument that is labelled as such. Returning nothing lets a
    single dead credential empty an entire rubric line, which is exactly what
    happened to this project before the chain existed.

    The returned verdict always carries ``instrument``, so the aggregator can
    keep remote and local scores in separate buckets instead of pooling two
    different measurement devices into one mean.
    """
    settings = get_settings()
    chain = [m.strip() for m in settings.judge_provider_chain.split(",") if m.strip()]
    # The chain is authoritative when configured. Forcing the caller's model to
    # the front made the chain's *order* unreachable: `judge_model` defaults to
    # the first historical choice, so reordering the chain to put a provider
    # with remaining quota first had no effect at all. The caller's model is
    # appended as a last resort if the chain does not already contain it.
    ordered = list(chain) if model in chain else [*chain, model]

    failures: list[str] = []
    for candidate in ordered:
        if not has_credential(candidate):
            failures.append(f"{candidate}: no credential")
            continue
        verdict = judge_answer(llm_client, question, context, answer, model=candidate)
        if verdict.ok:
            verdict.instrument = f"llm-judge:{candidate}"
            return verdict
        failures.append(f"{candidate}: {verdict.error}")

    if not allow_local:
        verdict.error = " | ".join(failures)
        return verdict

    local = judge_answer_locally(question, context, answer)
    local.error = "every judge in the chain was unavailable (" + " | ".join(failures) + ")"
    return local


def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Doc: {c['source']}, Chunk: {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks
    )


def run_probes(llm_client: Any, judge_model: str) -> dict[str, Any]:
    """Validate the judge before trusting any number it produces."""
    rows: list[dict[str, Any]] = []
    for probe in PROBES:
        verdict = judge_answer(
            llm_client,
            probe["question"],
            probe["context"],
            probe["answer"],
            model=judge_model,
        )
        passed, note = check_expectation(probe, verdict)
        row: dict[str, Any] = {
            "id": probe["id"],
            "expectation": probe["expectation"],
            "why_it_matters": probe["rationale"],
            "faithfulness_score": verdict.faithfulness_score,
            "relevance_score": verdict.relevance_score,
            "faithfulness_rationale": verdict.faithfulness_rationale,
            "relevance_rationale": verdict.relevance_rationale,
            "passed": passed,
            "note": note,
        }

        # The length probe is a paired comparison, so it needs its control.
        if probe.get("control_answer"):
            control = judge_answer(
                llm_client,
                probe["question"],
                probe["context"],
                probe["control_answer"],
                model=judge_model,
            )
            row["control_relevance_score"] = control.relevance_score
            row["control_faithfulness_score"] = control.faithfulness_score
            inflated = verdict.relevance_score > control.relevance_score
            row["length_inflated_score"] = inflated
            row["passed"] = verdict.ok and control.ok and not inflated
            row["note"] = f"padded={verdict.relevance_score} vs terse={control.relevance_score}" + (
                " -- LENGTH BIAS DETECTED" if inflated else " -- no length bias"
            )
        rows.append(row)

    passed = sum(1 for r in rows if r["passed"])
    return {
        "n_probes": len(rows),
        "n_passed": passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "interpretation": (
            "Probes assert properties that follow from the rubric's own "
            "definitions: faithfulness tracks grounding rather than world-truth, "
            "the two axes move independently, and answer length does not buy a "
            "score. A failure means the judge is measuring something other than "
            "what it claims."
        ),
        "probes": rows,
    }


def run(
    pipeline: RAGPipeline,
    llm_client: Any,
    dataset: dict[str, Any],
    *,
    judge_model: str,
    top_k: int,
) -> dict[str, Any]:
    per_question: list[dict[str, Any]] = []
    faithfulness: list[float] = []
    relevance: list[float] = []
    # False refusals are scored by rule, not by the judge. They are kept in
    # their own bucket so the headline means stay strictly *measured*, which is
    # what the rubric asks for; the combined figure is reported beside them.
    imputed_faithfulness: list[float] = []
    imputed_relevance: list[float] = []
    # Scores from the offline evaluator, kept apart from LLM-judge scores.
    local_faithfulness: list[float] = []
    local_relevance: list[float] = []
    fallback_rows: list[dict[str, Any]] = []
    judge_errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_generation_ms: list[float] = []

    for item in dataset["questions"]:
        row: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "expects_fallback": item["expects_fallback"],
        }
        try:
            result = pipeline.query(item["question"], top_k=top_k)
        except LLMError as exc:
            row.update({"error": f"generation_failed: {exc}", "scored": False})
            per_question.append(row)
            judge_errors += 1
            continue

        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        total_generation_ms.append(result.generation_latency_ms)

        row.update(
            {
                "answer": result.answer,
                "fallback_triggered": result.fallback_triggered,
                "chunk_count": result.chunk_count,
                "top_similarity": result.top_similarity,
                "citations": [c.__dict__ for c in result.citations],
                "unverified_citation_count": result.unverified_citation_count,
                "generation_prompt_tokens": result.prompt_tokens,
                "generation_completion_tokens": result.completion_tokens,
                "retrieval_latency_ms": round(result.retrieval_latency_ms, 3),
                "generation_latency_ms": round(result.generation_latency_ms, 3),
            }
        )

        if item["expects_fallback"]:
            # Scored under fallback correctness only. Two conditions, both
            # checkable: the exact string, and zero generation tokens billed.
            exact = result.answer.strip() == FALLBACK_ANSWER
            skipped_generation = result.completion_tokens == 0
            row.update(
                {
                    "scored": False,
                    "scored_under": "fallback_correctness",
                    "fallback_exact_string": exact,
                    "generation_skipped": skipped_generation,
                    "fallback_correct": exact and skipped_generation,
                }
            )
            fallback_rows.append(row)
            per_question.append(row)
            continue

        if result.fallback_triggered:
            # An in-corpus question that the system refused. Not a judge error;
            # a retrieval/threshold failure, and it must count against the
            # scores rather than quietly vanish from the denominator.
            row.update(
                {
                    "scored": True,
                    "scored_under": "false_refusal_imputed",
                    "faithfulness_score": 5,  # refusing invents nothing
                    "relevance_score": 1,  # but answers nothing either
                    "judge_note": (
                        "System refused an answerable question. Scored without "
                        "calling the judge: a refusal is trivially faithful and "
                        "wholly irrelevant. Counted, not excluded -- but kept out "
                        "of the measured means, because a rule-assigned score is "
                        "not a measurement."
                    ),
                    "false_refusal": True,
                }
            )
            imputed_faithfulness.append(5.0)
            imputed_relevance.append(1.0)
            per_question.append(row)
            continue

        verdict = judge_with_fallback(
            llm_client,
            item["question"],
            format_context(result.retrieved_chunks),
            result.answer,
            model=judge_model,
        )
        total_prompt_tokens += verdict.prompt_tokens
        total_completion_tokens += verdict.completion_tokens

        if not verdict.ok and getattr(verdict, "instrument", "") == "llm-judge":
            judge_errors += 1
            row.update({"scored": False, "judge_error": verdict.error})
            per_question.append(row)
            continue

        instrument = getattr(verdict, "instrument", "llm-judge")
        is_local = instrument.startswith("local")
        row.update(
            {
                "scored": True,
                "scored_under": "faithfulness_relevance",
                "false_refusal": False,
                **verdict.to_dict(),
            }
        )
        if verdict.error:
            row["degraded"] = verdict.error
        per_question.append(row)

        # Separate buckets: the local evaluator is a different instrument, and
        # averaging it together with LLM-judge scores would report one number
        # for two measurement devices.
        if is_local:
            local_faithfulness.append(float(verdict.faithfulness_score))
            local_relevance.append(float(verdict.relevance_score))
        else:
            faithfulness.append(float(verdict.faithfulness_score))
            relevance.append(float(verdict.relevance_score))

    n_fallback_correct = sum(1 for r in fallback_rows if r["fallback_correct"])
    return {
        "judge_model": judge_model,
        "generation_model": get_settings().generation_model,
        "top_k": top_k,
        "n_questions": len(dataset["questions"]),
        "n_scored": len(faithfulness),
        "n_fallback_questions": len(fallback_rows),
        "n_judge_errors": judge_errors,
        "scale": "1-5 integer, per criterion",
        "faithfulness": summarise(faithfulness),
        "answer_relevance": summarise(relevance),
        "local_fallback_scores": {
            "n": len(local_faithfulness),
            "instrument": "local-embedding",
            "policy": (
                "Produced by eval/local_judge.py when every provider in the judge chain was "
                "unreachable. Reported separately and never pooled with LLM-judge scores: "
                "cosine-similarity groundedness detects unsupported content, not contradiction, "
                "so it is a weaker instrument measuring a narrower property."
            ),
            "faithfulness": summarise(local_faithfulness),
            "answer_relevance": summarise(local_relevance),
        },
        "false_refusal_imputation": {
            "n": len(imputed_faithfulness),
            "policy": (
                "A refusal of an answerable question is assigned faithfulness=5, "
                "relevance=1 by rule rather than by the judge. Excluded from the "
                "measured means above and reported here so the imputation is "
                "visible instead of blended in."
            ),
            "faithfulness_including_imputed": summarise(faithfulness + imputed_faithfulness),
            "answer_relevance_including_imputed": summarise(relevance + imputed_relevance),
        },
        "fallback_correctness": {
            "n": len(fallback_rows),
            "n_correct": n_fallback_correct,
            "rate": round(n_fallback_correct / len(fallback_rows), 4) if fallback_rows else None,
            "definition": (
                "Exact refusal string returned AND zero completion tokens billed "
                "(generation skipped entirely)."
            ),
        },
        "false_refusals": sum(1 for r in per_question if r.get("false_refusal")),
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_prompt_tokens + total_completion_tokens,
            "note": (
                "Summed across generation and judging, including every failed "
                "attempt and JSON repair, because billing counts those too."
            ),
        },
        "generation_latency_ms": {
            "n": len(total_generation_ms),
            "p50": _pct(total_generation_ms, 0.50),
            # p95 is on the assignment's mandatory list alongside retrieval, and
            # was previously not reported for generation at all.
            "p95": _pct(total_generation_ms, 0.95),
            "mean": round(statistics.fmean(total_generation_ms), 3)
            if total_generation_ms
            else None,
        },
        "per_question": per_question,
    }


def print_summary(section: dict[str, Any], probes: dict[str, Any] | None) -> None:
    print(
        f"\nAnswer quality  (judge={section['judge_model']}, "
        f"generator={section['generation_model']})"
    )
    print("-" * 68)
    for name in ("faithfulness", "answer_relevance"):
        s = section[name]
        print(
            f"{name:<20} mean={s['mean']:.3f}  std={s['std']:.3f}  "
            f"95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]  n={s['n']}"
        )
    fb = section["fallback_correctness"]
    print(f"{'fallback_correct':<20} {fb['n_correct']}/{fb['n']}")
    print(f"{'false_refusals':<20} {section['false_refusals']}")
    print(f"{'judge_errors':<20} {section['n_judge_errors']}")
    print(f"{'tokens (all attempts)':<20} {section['tokens']['total']:,}")

    if probes:
        print(f"\nJudge validation: {probes['n_passed']}/{probes['n_probes']} probes passed")
        for probe in probes["probes"]:
            mark = "PASS" if probe["passed"] else "FAIL"
            print(f"  [{mark}] {probe['id']:<24} {probe['note']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="first 2 questions only")
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default=str(RESULTS))
    args = parser.parse_args()

    settings = get_settings()
    llm_client = get_llm_client()
    probes = None if args.skip_probes else run_probes(llm_client, settings.judge_model)

    section: dict[str, Any] | None = None
    if not args.probes_only:
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        if args.smoke:
            # Keep the out-of-corpus question in the smoke subset so CI still
            # exercises the fallback path.
            questions = dataset["questions"]
            dataset = {**dataset, "questions": [questions[0], questions[-1]]}
        pipeline = RAGPipeline()
        if pipeline.store.count() == 0:
            print("Vector store is empty. Run: python scripts/ingest_corpus.py --reset")
            return 1
        section = run(
            pipeline,
            llm_client,
            dataset,
            judge_model=settings.judge_model,
            top_k=args.top_k or settings.default_top_k,
        )
        print_summary(section, probes)
    elif probes:
        print(f"Judge validation: {probes['n_passed']}/{probes['n_probes']} passed")
        for probe in probes["probes"]:
            print(f"  [{'PASS' if probe['passed'] else 'FAIL'}] {probe['id']}: {probe['note']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if section is not None:
        existing["answer"] = section
    if probes is not None:
        existing["judge_validation"] = probes
    existing.setdefault("meta", {})
    existing["meta"].update(
        {
            "answer_run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "judge_model": settings.judge_model,
            "generation_model": settings.generation_model,
            "generation_temperature": settings.generation_temperature,
        }
    )
    # allow_nan=False: see eval/evaluate_retrieval.py -- a bare NaN token makes
    # the artifact unparseable by every non-Python JSON reader.
    output.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
