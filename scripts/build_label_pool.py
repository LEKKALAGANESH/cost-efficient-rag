"""Build a TREC-style judgment pool for hand-labelling relevance.

Labels written from memory over a corpus of thousands of chunks are
*incomplete*: genuinely relevant chunks that were never labelled get scored as
errors, biasing every metric downward by an unmeasured amount.  This is the
classical TREC incomplete-judgments problem.

The mitigation is pooling.  For each question this script runs top-N retrieval
under several chunking configurations, takes the union of retrieved chunk IDs
as the judgment pool, and prints that pool for a human to accept or reject.
Chunks below pool depth are treated as non-relevant, the standard TREC
assumption, which the README states explicitly.

    python scripts/build_label_pool.py --questions scripts/candidate_questions.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import embedding_service
from src.ingest_service import ingest_directory
from src.vector_store import LanceDBVectorStore

#: Pool across configurations, not just across k. A chunk that only surfaces
#: under a different chunk size is exactly the label a single-config pool
#: misses.
POOL_CONFIGS: list[tuple[int, int]] = [(500, 50), (350, 35), (800, 80)]
POOL_DEPTH = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="one question per line")
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data" / "raw_documents"))
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    questions = [
        line.strip()
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # The *primary* configuration is first; its chunk IDs are the ones that end
    # up in the shipped eval set, because it is the configuration the service
    # actually runs. Other configurations only widen the pool.
    pools: dict[str, dict[str, dict]] = {q: {} for q in questions}
    primary_ids: dict[str, list[str]] = {q: [] for q in questions}

    for config_index, (size, overlap) in enumerate(POOL_CONFIGS):
        workdir = Path(tempfile.mkdtemp(prefix=f"pool_{size}_"))
        try:
            store = LanceDBVectorStore(db_path=workdir, table_name="pool")
            ingest_directory(args.corpus, store, chunk_size=size, chunk_overlap=overlap)
            for question in questions:
                hits = store.search(embedding_service.embed_query(question), args.depth)
                if config_index == 0:
                    primary_ids[question] = [h.chunk_id for h in hits]
                for rank, hit in enumerate(hits, start=1):
                    entry = pools[question].setdefault(
                        hit.chunk_id,
                        {
                            "chunk_id": hit.chunk_id,
                            "source": hit.source,
                            "chunk_index": hit.chunk_index,
                            "text": hit.text,
                            "seen_in": [],
                            "best_rank": rank,
                            "similarity": round(hit.similarity, 4),
                        },
                    )
                    entry["seen_in"].append(f"{size}/{overlap}@{rank}")
                    entry["best_rank"] = min(entry["best_rank"], rank)
            store.close()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    for question in questions:
        print(f"\n{'=' * 100}\nQ: {question}\n{'=' * 100}")
        ranked = sorted(pools[question].values(), key=lambda e: e["best_rank"])
        for entry in ranked:
            in_primary = "P" if entry["chunk_id"] in primary_ids[question] else " "
            preview = " ".join(entry["text"].split())[:220]
            print(
                f"\n [{in_primary}] rank={entry['best_rank']:<3} "
                f"sim={entry['similarity']:<7} {entry['source']}#{entry['chunk_index']}"
            )
            print(f"     {entry['chunk_id']}")
            print(f"     {preview}")
        print(f"\n  pool size: {len(ranked)} (primary top-{args.depth} marked P)")

    if args.json_out:
        payload = {
            q: {
                "pool": sorted(pools[q].values(), key=lambda e: e["best_rank"]),
                "primary_top_k": primary_ids[q],
            }
            for q in questions
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
