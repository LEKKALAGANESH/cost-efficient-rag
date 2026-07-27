<div align="center">

# cost-efficient-rag

**A retrieval-augmented QA service on an embedded vector store — with the cost argument actually measured.**

[![tests](https://img.shields.io/badge/tests-244%20passed-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![ruff](https://img.shields.io/badge/ruff-clean-brightgreen)](pyproject.toml)
[![offline](https://img.shields.io/badge/tests-no%20API%20key%20needed-informational)](#testing)
[![vector store](https://img.shields.io/badge/vector%20store-LanceDB-orange)](https://lancedb.com)

</div>

---

A knowledge base that is **large but lightly queried** pays full 24/7 rent on a managed vector
database. This project measures what you actually give up by running the index in-process
instead — and reports the number that survives being argued with, not the flattering one.

> **The headline is 3.7–9× cheaper than provisioned managed capacity, not 20–47×.**
> An earlier revision of the cost model produced the larger figure by assuming a managed service
> holds raw `float32` vectors *and* the chunk text in RAM. Neither is how they are deployed. The
> corrected model is in [`eval/cost_analysis.py`](eval/cost_analysis.py), and a test pins the
> claim so code and documentation cannot drift apart again.

---

## Table of contents

- [Results](#results)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Design decisions worth defending](#design-decisions-worth-defending)
- [Cost model](#cost-model)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Limitations](#limitations)

---

## Results

Every figure below is read from a committed artifact in [`results/`](results/). Nothing is
asserted from code inspection.

### Retrieval quality — n=14 judged, 1 out-of-corpus

| Metric | Value | Note |
|---|---:|---|
| **Recall@5** | **0.714** | 95% CI 0.50–0.93 |
| Recall@10 / @20 | 0.941 / 0.964 | ranking is the bottleneck, not the embedder |
| **Hit Rate@5** | 0.786 | reported separately from Recall — they coincide only at one relevant chunk |
| **MRR** | 0.667 | first-relevant rank; 0 when nothing relevant is retrieved |
| **nDCG@5** | 0.631 | IDCG truncated to `min(|relevant|, k)` |
| **Context precision@5** | 0.200 | against an achievable ceiling of **0.329** = **61% of achievable** |

The ceiling matters: with one relevant chunk at `k=5`, a *perfect* system scores 0.20. Publishing
the raw number alone would misrepresent it.

### Latency

| Stage | p50 | p95 | p99 | n |
|---|---:|---:|---:|---:|
| Retrieval | **28.0 ms** | **33.4 ms** | 37.1 ms | 300 |
| Generation | **787 ms** | **4147 ms** | — | 14 |

Generation dominates the median by ~28×. Optimising the vector store would move the smaller term.

### Answer quality

| Metric | Value | Instrument |
|---|---:|---|
| Groundedness | 4.00 / 5 | `local-embedding` (n=9) |
| Answer relevance | 3.89 / 5 | `local-embedding` (n=9) |
| Fallback correctness | **1/1** | exact refusal string + 0 completion tokens |

> **On the instrument column.** These are **not** LLM-judge scores. They come from
> [`eval/local_judge.py`](eval/local_judge.py), a cosine-similarity evaluator that runs offline
> when every provider is unreachable. It detects *unsupported* content, not *contradicted*
> content. Every row records `instrument`, and local scores are never pooled into an LLM-judge
> mean. Reporting them under an "LLM-as-judge" heading would show a better number and be false.

### Cost at scale — monthly, USD

| Vectors | Storage | Embedded (fully loaded) | Serverless | Provisioned | Activity-tier |
|---|---:|---:|---:|---:|---:|
| 100K | 0.21 GiB | **$0.08** | $50.00 | $3.27 | $45.00 |
| 1M | 2.08 GiB | **$12.21** | $50.00 | $32.67 | $80.00 |
| 10M | 20.82 GiB | **$37.14 – $87.14** | $50.00 | $326.68 | $300.00 |

**Against serverless at this query volume it is roughly a wash.** That is the honest result, and
the win there is operational — no vendor, no egress metering, no data-residency question — not
economic.

---

## Quickstart

```bash
pip install -r requirements.txt
cp env.example .env                  # two keys; see the table below

python scripts/build_corpus.py
python scripts/ingest_corpus.py --reset
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/docs** for the interactive API, or:

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Why must IDCG be truncated at k?","top_k":5}'
```

### What needs an API key

| Runs with **no key at all** | Needs a key |
|---|---|
| Ingestion (PDF / HTML / Markdown) | Grounded answer generation |
| Embedding, indexing, retrieval | The LLM judge |
| Every retrieval metric | |
| The entire cost model | |
| **All 244 tests** | |

Embeddings are local (`all-MiniLM-L6-v2`, 384-dim) — $0 marginal cost and, more importantly,
**no quota**.

---

## Architecture

```
  client ──► api.py  (/ingest  /query  /health)
               │                    │
      ingest_service.py      rag_pipeline.py
      load → normalize       embed → search →
      → split → hash         threshold → prompt
      → embed → upsert       → generate → cite
               │                    │        └──► llm_client.py
               └────────┬───────────┘              (cache, provider chain,
                        │                           retries, token pacing)
                 vector_store.py ◄──── embeddings.py
                 VectorStoreManager     (one model-load site,
                 ├── LanceDB (default, in-process)   always L2-normalised)
                 └── pgvector (optional second backend)

  rag_pipeline.py is the single exit point ──► logger.py ──► logs/queries.jsonl
```

Handlers are `def`, not `async def`: embedding is CPU-bound and torch releases the GIL, so
Starlette's threadpool genuinely parallelises. `vector_store.py` imports no FastAPI;
`cost_analysis.py` imports nothing from `src/`; configuration is read in exactly one place.

---

## Design decisions worth defending

<details>
<summary><b>Why LanceDB</b></summary>

<br>

Zero-process and — decisively — **zero-account**, so `git clone && pip install && python
scripts/ingest_corpus.py` actually works for a reviewer.

Rejected: **Qdrant**'s Python local mode has *no HNSW index at all* (brute-force, ~20K points —
its published benchmarks describe the server binary). **FAISS** is an index library, not a
database: no metadata store, no transactional persistence. **ChromaDB**'s metadata filtering is
weaker and its storage efficiency drops above ~10M vectors.

</details>

<details>
<summary><b>Idempotency is a test, not a claim</b></summary>

<br>

`doc_key = sha256(file_bytes)` — never the filesystem path. Hashing the path would mean a reviewer
who ingests by upload gets different chunk IDs, `retrieved ∩ relevant = {}`, and **every retrieval
metric silently reporting 0.0**.

Writes go through `merge_insert` keyed on `chunk_id`, never `add()`: read-then-append is a TOCTOU
race where two concurrent ingests both see an empty set and both append.

Proven by re-ingesting the corpus (72 → 72 chunks, 0 embedded) *and* by HTTP upload of identical
bytes.

</details>

<details>
<summary><b>Score direction</b></summary>

<br>

LanceDB returns `_distance` (lower = better) and defaults to L2. `metric("cosine")` is passed on
**every** `search()` call — `create_table` takes no metric argument, so the per-query call, not the
table, is the durable guarantee. Vectors are L2-normalised at both ends, and `search()` returns
`similarity = 1 - _distance`, a field deliberately never named `score`.

Two assertions make the whole bug class impossible: a query identical to an ingested chunk must
score ≈1.0, and a relevant query must outscore an irrelevant one. Both backends face them via one
shared contract suite.

</details>

<details>
<summary><b>Provider fallback — why evaluation never aborts</b></summary>

<br>

This harness once produced **zero measurements** because a single configured provider returned
`Invalid API Key`. One dead credential emptied an entire evaluation.

[`src/providers.py`](src/providers.py) makes the *chain* the unit of configuration, not the model:

```bash
LLM_PROVIDER_CHAIN=groq/openai/gpt-oss-120b,gemini/gemini-3.5-flash,ollama/llama3
```

- **Transient** failures (429, 5xx, timeout) retry the *same* provider with exponential backoff and
  full jitter — moving on immediately would spend the next provider's quota on a problem that was
  about to resolve itself.
- **Fatal** failures (bad key, unknown model) **open the circuit immediately**. A bad key does not
  heal by being asked 40 more times.
- **Token-per-minute pacing** runs before every attempt. TPM binds before RPM for this workload.
- A **local evaluator** terminates the chain, so a total outage degrades a measurement instead of
  deleting it.

14 providers are supported by prefix, switchable by environment variable with no code change.

</details>

---

## Cost model

Run it standalone — no API key, no application boot:

```bash
python -m eval.cost_analysis
python -m eval.cost_analysis --queries-per-month 500000
```

Assumptions are named constants with sourced pricing and capture dates:

| Assumption | Value |
|---|---|
| Embedding dimensions | 384 (`all-MiniLM-L6-v2`) |
| Bytes per vector | 1,536 B (float32) |
| Metadata per vector | **700 B** — 512 B could not cover `chunk_id` + `doc_key` + `source` + `file_type` + `chunk_index` **and** the chunk text |
| Query volume | 50,000/month (0.019 QPS) |
| Quantization | int8, ~4× — the default on provisioned offerings |
| Payload location | disk, not RAM |

Two things this model **refuses** to do, both of which would make it look better:

1. **Claim $0 marginal compute.** CPU genuinely is free at 0.019 QPS. Memory is not, and it does
   not vanish because the process is shared with the API.
2. **Report a uniform 100× win.** Against serverless it is a wash; below the free-tier storage cap
   the managed option is outright cheaper.

### When to switch back to managed

Switch when **any one** holds. Query volume is deliberately *not* on the list — per the sensitivity
table it favours staying embedded.

1. The corpus exceeds what a single node can index and hold in its working set.
2. Multi-region HA or a compliance guarantee is required.
3. Write concurrency from independent services exceeds a single-writer ingestion path.

---

## Evaluation

```bash
python -m eval.evaluate_retrieval --latency-loops 20   # no API key needed
python -m eval.evaluate_answer                         # needs a key; degrades if unavailable
python -m eval.cost_analysis                           # no API key needed
```

The evaluation set is **15 frozen questions** ([`data/eval_dataset.json`](data/eval_dataset.json)),
built by resolving hand-written marker sentences against the live index so labels cannot rot
silently when the corpus changes. It deliberately includes a multi-relevant question, a
`|relevant| > k` question, and an out-of-corpus probe.

Questions with no relevant chunk **raise** rather than scoring a silent 0.0 — they are excluded
from the ranked denominator and reported separately as `n_ranked` vs `n_fallback`.

---

## Testing

```bash
python -m pytest                        # 244 passed, 13 skipped — no API key, no network
ruff check . && ruff format --check .
```

| Suite | Tests | Covers |
|---|---:|---|
| `test_metrics.py` | 32 | All five IR formulas against a hand-computed fixture that also pins the *wrong* untruncated-IDCG value, so a regression is unmistakable |
| `test_ingestion.py` | 28 | Loaders, chunking, hashing, idempotency, malformed files |
| `test_api.py` | 23 | Endpoints, batch isolation, upload caps, filter injection, non-ASCII filenames |
| `test_llm_client.py` | 21 | Cache, retries, transient-vs-permanent classification, token accounting |
| `test_judge.py` | 20 | Rubric, repair loop, adversarial probes |
| `test_vector_store.py` | 18 | One contract suite parametrised over **both** backends |
| `test_pgvector_store.py` | 16 | Score direction, parameterised filters, `SET` vs `SET LOCAL` |
| `test_cost_analysis.py` | 15 | Cost arithmetic and the single-digit-advantage bound |
| `test_providers.py` | 12 | Routing vs family, circuit breaker, backoff jitter |

The 13 skips are honest: they report *"SUPABASE_DB_URL set but unreachable, so parity is
unproven"* rather than claiming the variable is unset.

CI runs the suite on **Python 3.10 and 3.12**, so the declared `requires-python` floor is verified
rather than asserted.

---

## Configuration

Everything is environment-driven via [`env.example`](env.example) — models, chunk size, top-k,
thresholds, timeouts, retries, provider chains, TPM ceilings and cost assumptions. No call site
names a vendor.

Secrets are never logged: fields are `repr=False`, `.env` is gitignored, and CI runs a secret scan.

---

## Limitations

Stated plainly, because a reviewer will find them anyway:

- **The corpus is synthetic** — 4 self-authored documents (1 PDF, 1 HTML, 2 Markdown) → 72 chunks.
  "Robust chunking" is never tested against a messy real-world PDF.
- **n=14 judged questions** gives ±0.20 confidence intervals. Nothing here supports fine-grained
  tuning claims.
- **Labels are self-authored.** Pooling and committed excerpts make them auditable, but Recall@5 of
  0.71 against my own labels is weaker evidence than the same number against someone else's.
- **19 of 24 gold labels sit within the pooling depth**, so `recall@10 = 0.941` is partly a
  construction artifact. Read the sweep's tail as softer than it looks.
- **Answer-quality scores come from the offline evaluator**, not an LLM judge — see the note under
  [Results](#results).
- **The second backend is implemented, not benchmarked.** pgvector has 16 offline tests and shares
  the contract suite, but no reachable database produced comparative numbers.

---

<div align="center">
<sub>Built by <a href="https://github.com/LEKKALAGANESH">LEKKALA GANESH</a></sub>
</div>
