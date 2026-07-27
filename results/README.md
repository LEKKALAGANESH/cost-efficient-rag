# results/

Everything in this directory is the **actual output of a real run**. Nothing
here was written by hand or estimated. Each file names the command that
produced it, and each is regenerated from a clean clone by re-running it.

| File | Produced by | Status |
|---|---|---|
| `eval_results.json` | `python -m eval.evaluate_retrieval --latency-loops 20` | **Retrieval section: real.** Answer section: **not present** -- see below |
| `threshold_calibration.json` | `python scripts/calibrate_threshold.py` | Real |
| `cost_benchmark_table.md` | `python -m eval.cost_analysis` | Real |
| `cost_benchmark.json` | `python -m eval.cost_analysis` | Real |
| `query_log_sample.jsonl` | all 15 eval questions through `rag_pipeline.query()` | Real, with one caveat below |

### About `query_log_sample.jsonl`

`logs/` is gitignored, so this is a committed excerpt proving the per-query
logging required by R11 actually works. All 15 questions were run through the
real pipeline against the real corpus. Every line carries a genuine
`retrieval_latency_ms`, `chunk_count`, `top_similarity`, `fallback_triggered`
and the full token fields.

**The caveat, stated plainly:** with no API key, 14 of the 15 lines carry
`"error": "generation_failed..."` and therefore `total_tokens: 0`. One line
(the out-of-corpus probe) shows the fallback path working exactly as designed:
`fallback_triggered: true`, generation skipped, zero tokens billed. The
token-summing logic itself -- including summing across failed and repaired
attempts -- is proven separately by `tests/test_logger.py` and
`tests/test_judge.py::test_tokens_are_summed_across_every_attempt_including_failures`.

## What is missing, and why

**`eval_results.json` has no `answer` or `judge_validation` section.**

The machine that produced these files had no `GROQ_API_KEY` and no
`GEMINI_API_KEY`, so the generation and LLM-judge legs could not be executed.
Fabricating faithfulness and relevance numbers would have been worse than
omitting them, so they are omitted and this file says so.

Everything needed to produce them is implemented, tested offline against
mocked clients (`tests/test_judge.py`, `tests/test_eval_harness.py`), and runs
with one command once keys are present:

```bash
cp env.example .env          # then fill in the two keys
python -m eval.evaluate_answer          # writes the answer + judge sections
```

That command adds three things to `eval_results.json`:

- `answer` -- per-question faithfulness and relevance (1-5, with the judge's
  rationale for each), fallback correctness, false-refusal count, and tokens
  summed across generation, judging, and every failed or repaired attempt;
- `judge_validation` -- the four adversarial probes that check the judge is
  measuring groundedness rather than world-truth, that faithfulness and
  relevance move independently, and that answer length does not buy a score;
- `meta.answer_run_at`, the judge model ID, and the temperature used.

It also populates `logs/queries.jsonl`, which is where the per-query latency
and token numbers come from.

## Reproducing the retrieval numbers

```bash
python scripts/build_corpus.py        # deterministic; same bytes every time
python scripts/ingest_corpus.py --reset
python -m eval.evaluate_retrieval --latency-loops 20
```

The corpus is byte-reproducible, so this yields identical chunk IDs and
identical metrics on any machine. `tests/test_corpus_determinism.py` asserts
that property, and that every gold chunk ID in `data/eval_dataset.json` still
resolves against the committed corpus.

Latency figures vary with hardware. They were captured on a Windows 11 laptop
(CPU-only inference, Python 3.14.3) over 300 samples -- 15 questions x 20
loops, because a p95 taken from 15 observations is just the maximum.

## Auditing a single row by hand

Open `eval_results.json` -> `retrieval.per_question`, pick any row, and compare
its `retrieved_chunk_ids` against its `relevant_chunk_ids`. Recall@5 is
`|intersection| / |relevant|` and MRR is `1 / (1-indexed rank of the first
relevant id)`. `q01` is worked through line by line in the main README under
"Hand-verifying one row".

The out-of-corpus question (`q15`) carries `"judgeable": false` and no
`metrics` block: it is excluded from all five rank metrics and scored only
under fallback correctness. `n_ranked` (14) and `n_fallback` (1) are reported
separately so both denominators are auditable.
