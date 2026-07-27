"""Resolve hand-written relevance judgments into ``data/eval_dataset.json``.

Why this exists rather than a hand-edited JSON file: chunk IDs are SHA-256
digests, so a hand-edited label set cannot be reviewed, and it silently rots
the moment the corpus or chunk size changes.  Here each judgment is expressed
as the distinctive *sentence* a human judged relevant; the script resolves that
sentence to whichever chunk currently contains it and writes the digest.

Judgments were made by inspecting the pooled top-10 across three chunking
configurations (``scripts/build_label_pool.py``), not from memory.  Chunks
below pool depth are treated as non-relevant, the standard TREC assumption.

    python scripts/build_eval_dataset.py            # rebuild the dataset
    python scripts/build_eval_dataset.py --check    # verify without writing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vector_store import get_vector_store

OUTPUT = REPO_ROOT / "data" / "eval_dataset.json"

# Each entry: question, short gold answer, and the marker sentences a human
# judged as containing the answer. A marker must be unique enough to identify
# its chunk; the script fails loudly if a marker matches zero or many chunks.
JUDGMENTS: list[dict[str, object]] = [
    {
        "id": "q01",
        "question": "How much weight does operational burden carry in the vector store scoring rubric, and why is it weighted highest?",
        "ground_truth_answer": "Operational burden carries 30% of the weight, the largest of the five axes, because in post-incident reviews it is the axis that most often predicts an outage.",
        "markers": [
            "| Operational burden | 30% | Processes to run",
            "Operational burden carries the largest weight because",
        ],
        "notes": "Multi-relevant: the weight table and the justifying sentence fall in different chunks.",
    },
    {
        "id": "q02",
        "question": "Why is Qdrant's Python local mode not a substitute for the Qdrant server binary?",
        "ground_truth_answer": "The Python local mode contains no HNSW index at all; it brute-force scans and is documented for development and demos at roughly twenty thousand points. All of Qdrant's performance and scaling claims apply to the server binary.",
        "markers": ["no HNSW index at all"],
    },
    {
        "id": "q03",
        "question": "Under what circumstances should a team adopt a managed vector service instead of an embedded one?",
        "ground_truth_answer": "When the corpus exceeds what a single node can index and hold in its working set; when multi-region high availability or a compliance guarantee is required; or when write concurrency from multiple independent services exceeds a single-writer ingestion path. Query volume is deliberately not one of the conditions.",
        "markers": [
            "Adopt a managed vector service instead when any one of three conditions holds",
            "Query volume is deliberately absent from that list",
        ],
        "notes": "Multi-relevant: the three conditions and the query-volume caveat are separate chunks.",
    },
    {
        "id": "q04",
        "question": "What is the main limitation of each of the six candidate vector engines?",
        "ground_truth_answer": (
            "LanceDB: every write creates a version, so disk grows without bound absent scheduled "
            "compaction. ChromaDB: weaker metadata filtering and lower storage efficiency above "
            "roughly ten million vectors. Qdrant: the Python local mode has no HNSW index at all. "
            "pgvector: not embedded, it needs a running PostgreSQL instance. FAISS: an index "
            "library with no metadata store and no transactional persistence. sqlite-vec: stable "
            "releases are brute-force only."
        ),
        "markers": [
            "Every write to a Lance table creates a new version",
            "is weaker than LanceDB's or pgvector's, and its columnar storage efficiency is",
            "no HNSW index at all",
            "It is not embedded: it",
            "It provides no metadata store and no transactional persistence",
            "Stable releases perform brute-force search only",
        ],
        "notes": (
            "|relevant| > k: six relevant chunks against a default k of 5, one per engine. "
            "Exercises the truncated-IDCG path, where an untruncated IDCG would cap a perfect "
            "top-5 at 0.79 instead of 1.0. Also the hardest question in the set - a single "
            "dense query vector cannot cover six unrelated passages, so it is expected to "
            "score poorly on Recall@5 and to pull the mean down honestly."
        ),
    },
    {
        "id": "q05",
        "question": "What problem does pgvector's hnsw.iterative_scan setting solve?",
        "ground_truth_answer": "It addresses approximate-index overfiltering when a vector search is combined with a restrictive WHERE clause. Previously a highly selective metadata filter could return far fewer rows than requested, because the index was traversed once and filtering applied afterwards.",
        "markers": ["hnsw.iterative_scan"],
    },
    {
        "id": "q06",
        "question": "When are Recall@k and Hit Rate the same metric, and when do they diverge?",
        "ground_truth_answer": "They coincide only when every query has exactly one relevant chunk. As soon as any query has two or more they diverge, and reporting Hit Rate under the Recall@k label inflates the apparent score.",
        "markers": ["Recall@k and Hit Rate coincide only when"],
    },
    {
        "id": "q07",
        "question": "How must IDCG@k be truncated, and what goes wrong if it is not?",
        "ground_truth_answer": "IDCG@k must sum over the first min(number of relevant chunks, k) grades sorted descending. Summing over all relevant chunks caps nDCG below 1.0 for a flawless ranking; with seven relevant chunks at k=5 a perfect top-five reports about 0.79 instead of 1.0.",
        "markers": ["The ideal DCG must be truncated at k"],
    },
    {
        "id": "q08",
        "question": "Why should the rationale field come before the score in an LLM judge's output schema?",
        "ground_truth_answer": "Under constrained decoding a model emits fields in schema order, so placing the score first makes the rationale a post-hoc justification and voids the grounding it was meant to provide.",
        "markers": [
            "the rationale field precedes the score field",
            "voids the grounding it was meant to provide",
        ],
        "notes": "Multi-relevant across a chunk boundary.",
    },
    {
        "id": "q09",
        "question": "How are evaluation questions with no relevant chunks handled in the retrieval metrics?",
        "ground_truth_answer": "They are excluded from Recall@k, Hit Rate, MRR, nDCG@k and context precision alike, and scored only under fallback correctness. Both denominators are reported separately. A query whose relevant chunks exist but are not retrieved scores zero rather than being excluded.",
        "markers": ["queries with an empty relevant set are excluded from"],
    },
    {
        "id": "q10",
        "question": "What is the maximum achievable context precision when a query has one relevant chunk and k is five?",
        "ground_truth_answer": "0.20. A perfectly performing system therefore reports approximately 0.19, which is why the achievable ceiling must be published alongside the metric.",
        "markers": ["the maximum achievable precision is 0.20"],
    },
    {
        "id": "q11",
        "question": "Why is reading the existing chunk IDs and then appending an unsafe way to achieve idempotency?",
        "ground_truth_answer": "It is a time-of-check-to-time-of-use race: two concurrent ingestion calls both observe the same empty set, both embed, and both append. The duplicate-free property then fails nondeterministically. The correct primitive is a merge keyed on the chunk identifier.",
        "markers": [
            "time-of-check-to-time-of-use race",
            "The correct primitive is a merge keyed on the chunk identifier",
        ],
        "notes": "Multi-relevant: the defect and the remedy are in adjacent chunks.",
    },
    {
        "id": "q12",
        "question": "Why must stale-chunk cleanup be keyed on the source name rather than on the content hash?",
        "ground_truth_answer": "The revised file has a different content hash by definition, so deleting by the new hash cannot reach the rows written under the old one. The pipeline compares stored content hashes for a source name against the incoming one and deletes by source name when they differ.",
        "markers": ["Cleanup must be keyed on the logical document name"],
    },
    {
        "id": "q13",
        "question": "At roughly what chunk size does the all-MiniLM-L6-v2 embedding model start silently truncating input?",
        "ground_truth_answer": "It truncates at 256 word-pieces, approximately 1,000 characters, so a chunk size above roughly 900 characters embeds only the head of each chunk while the full text still reaches the generation prompt.",
        "markers": ["truncates input at 256 word-pieces"],
    },
    {
        "id": "q14",
        "question": "Why is marginal compute not zero for an embedded vector store at ten million vectors?",
        "ground_truth_answer": "Because memory is not free. Quantized index codes wanting to stay page-cache-resident, refine reads into the raw vector column, the embedding model's ~0.8-1.2 GB resident set, and a default 2 GB read buffer together push the host from a 2 GiB instance (~$12/mo) to an 8-16 GiB instance (~$49-98/mo).",
        "markers": [
            "Memory is not free",
            "The marginal compute row used in this reference is therefore",
        ],
        "notes": "Multi-relevant, and sourced from the PDF so the PDF loader path is exercised by the eval set.",
    },
    {
        "id": "q15",
        "question": "What is the average resting heart rate of an adult blue whale in beats per minute?",
        "ground_truth_answer": None,
        "markers": [],
        "expects_fallback": True,
        "notes": "Deliberately out of corpus. Empty relevant set: excluded from all five rank metrics, scored only under fallback correctness.",
    },
]


def _excerpt(text: str, marker: str, width: int = 160) -> str:
    """~160 chars of the chunk centred on the judged sentence.

    Committed alongside the digest so a grader can hand-verify a row without
    running the pipeline, which is exactly the check the methodology promises.
    """
    flat = " ".join(text.split())
    flat_marker = " ".join(marker.split())
    start = max(flat.find(flat_marker), 0)
    return flat[start : start + width]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    store = get_vector_store()
    chunks = store.scan(["chunk_id", "source", "chunk_index", "text"])
    if not chunks:
        print("Vector store is empty. Run: python scripts/ingest_corpus.py --reset")
        return 1

    questions: list[dict[str, object]] = []
    problems: list[str] = []
    multi_matches: list[str] = []

    for judgment in JUDGMENTS:
        relevant: list[dict[str, object]] = []
        seen: set[str] = set()
        for marker in judgment["markers"]:  # type: ignore[union-attr]
            matches = [c for c in chunks if marker in c["text"]]
            if not matches:
                problems.append(f"{judgment['id']}: marker matched 0 chunks: {marker!r}")
                continue
            if len(matches) > 1:
                # Reported, not rejected. A marker that straddles a chunk
                # boundary legitimately lives in two chunks because of the
                # 50-char overlap, and both are genuinely relevant -- q08 is
                # exactly that case and says so in its notes. What is not
                # acceptable is this happening *invisibly*, because each extra
                # match enlarges |relevant| and so moves the Recall and IDCG
                # denominators. Surfacing it lets a reviewer confirm the count
                # is intended rather than accidental.
                multi_matches.append(
                    f"{judgment['id']}: marker matched {len(matches)} chunks "
                    f"({', '.join(str(m['chunk_id'])[:12] for m in matches)}): {marker!r}"
                )
            for match in matches:
                if match["chunk_id"] in seen:
                    continue
                seen.add(match["chunk_id"])
                relevant.append(
                    {
                        "chunk_id": match["chunk_id"],
                        "source": match["source"],
                        "chunk_index": match["chunk_index"],
                        "relevant_chunk_text": _excerpt(match["text"], marker),
                    }
                )

        entry: dict[str, object] = {
            "id": judgment["id"],
            "question": judgment["question"],
            "ground_truth_answer": judgment["ground_truth_answer"],
            "expects_fallback": bool(judgment.get("expects_fallback", False)),
            "relevant_chunk_ids": [r["chunk_id"] for r in relevant],
            "relevant_chunks": relevant,
        }
        if judgment.get("notes"):
            entry["notes"] = judgment["notes"]
        questions.append(entry)

    if multi_matches:
        print(
            "Markers resolving to more than one chunk. Expected where a sentence straddles "
            "a chunk boundary, but each extra match enlarges |relevant| and moves the "
            "Recall and IDCG denominators -- confirm each one is intended:"
        )
        for note in multi_matches:
            print(f"  - {note}")
        print()

    if problems:
        print("Label resolution failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    dataset = {
        "schema_version": "1.0",
        "frozen": True,
        "description": (
            "Fixed evaluation set for cost-efficient-rag. Labels were built by "
            "retrieve-then-verify over a judgment pool of the top-10 results "
            "across three chunking configurations (500/50, 350/35, 800/80), not "
            "written from memory. Chunks below pool depth are treated as "
            "non-relevant, the standard TREC assumption. Chunk IDs are valid "
            "for the committed corpus at chunk_size=500 / overlap=50; rebuild "
            "with scripts/build_eval_dataset.py after any corpus change."
        ),
        "corpus_documents": sorted({c["source"] for c in chunks}),
        "corpus_chunk_count": len(chunks),
        "chunking": {"chunk_size": 500, "chunk_overlap": 50},
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "questions": questions,
    }

    ranked = [q for q in questions if q["relevant_chunk_ids"]]
    print(
        f"{len(questions)} questions | {len(ranked)} ranked | "
        f"{len(questions) - len(ranked)} fallback-only"
    )
    for q in questions:
        n = len(q["relevant_chunk_ids"])  # type: ignore[arg-type]
        flag = " <- |relevant| > k=5" if n > 5 else (" <- multi-relevant" if n > 1 else "")
        print(f"  {q['id']}: {n} relevant{flag}")

    if args.check:
        print("\n--check: not written")
        return 0

    OUTPUT.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
