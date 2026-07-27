"""Chunk size / overlap sweep against the marker-labelled eval questions.

    python scripts/sweep_chunking.py

Read-only: writes nothing, touches no store, needs no API key. Retrieval is
scored by exact cosine over freshly embedded chunks, which is what the vector
store returns at this corpus size anyway.

Gold labels are re-derived per configuration from the same marker sentences
`build_eval_dataset.py` uses. That re-derivation is the whole point: chunk_id
is a function of the chunking, so scoring 300/30 chunks against labels built
for 500/50 would report every setting except the shipped one as a total miss.

Read the output with the corpus size in mind. Recall@k is not comparable
across configurations for free -- fewer, larger chunks mean the same k covers
a larger fraction of the corpus, so Recall@5 rises mechanically as chunk size
grows. The `corpus covered` column makes that visible instead of letting it
masquerade as a quality gain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_eval_dataset import JUDGMENTS  # noqa: E402
from src.embeddings import embedding_service  # noqa: E402
from src.ingestion import build_chunk_records  # noqa: E402

CORPUS = REPO_ROOT / "data" / "raw_documents"
K = 5

#: (chunk_size, chunk_overlap). 1200 is deliberately past the embedding
#: model's 256 word-piece ceiling; 500/0 and 500/150 isolate overlap.
CONFIGS = [(200, 20), (300, 30), (500, 50), (800, 80), (1200, 120), (500, 0), (500, 150)]

SHIPPED = (500, 50)


def flatten(text: str) -> str:
    return " ".join(text.split())


def score(chunk_size: int, chunk_overlap: int) -> dict[str, float]:
    texts: list[str] = []
    for path in sorted(CORPUS.iterdir()):
        if not path.is_file():
            continue
        records, _, _ = build_chunk_records(
            path.read_bytes(), path.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts.extend(record.text for record in records)

    matrix = np.asarray(embedding_service.embed_texts(texts), dtype=np.float32)
    flattened = [flatten(text) for text in texts]

    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    unresolved = 0

    for item in JUDGMENTS:
        markers = [flatten(marker) for marker in item["markers"]]
        if not markers:
            continue  # the out-of-corpus question has no relevant chunk by design
        relevant = {i for i, text in enumerate(flattened) if any(m in text for m in markers)}
        if not relevant:
            # A marker that no longer resolves means this chunking split the
            # answer span across a boundary -- itself a finding, so it is
            # counted rather than silently skipped.
            unresolved += 1
            continue

        query = np.asarray(embedding_service.embed_query(item["question"]), dtype=np.float32)
        ranked = np.argsort(-(matrix @ query))[:K].tolist()
        found = [rank for rank in ranked if rank in relevant]

        recalls.append(len(found) / len(relevant))
        hits.append(1.0 if found else 0.0)
        mrrs.append(1.0 / (ranked.index(found[0]) + 1) if found else 0.0)

        dcg = sum(1.0 / np.log2(pos + 2) for pos, idx in enumerate(ranked) if idx in relevant)
        idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), K)))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        "chunks": len(texts),
        "mean_chars": float(np.mean([len(t) for t in texts])),
        "covered": K / len(texts),
        "recall": float(np.mean(recalls)),
        "hit_rate": float(np.mean(hits)),
        "mrr": float(np.mean(mrrs)),
        "ndcg": float(np.mean(ndcgs)),
        "unresolved": unresolved,
    }


def main() -> int:
    header = (
        f"{'size/overlap':<14}{'chunks':>7}{'avg chars':>10}{'corpus covered':>16}"
        f"{'Recall@5':>10}{'Hit@5':>8}{'MRR':>8}{'nDCG@5':>9}{'unlabelled':>12}"
    )
    print(header)
    print("-" * len(header))
    for chunk_size, chunk_overlap in CONFIGS:
        r = score(chunk_size, chunk_overlap)
        note = "  <- shipped default" if (chunk_size, chunk_overlap) == SHIPPED else ""
        print(
            f"{f'{chunk_size}/{chunk_overlap}':<14}{r['chunks']:>7}{r['mean_chars']:>10.0f}"
            f"{r['covered']:>15.1%}{r['recall']:>10.3f}{r['hit_rate']:>8.3f}"
            f"{r['mrr']:>8.3f}{r['ndcg']:>9.3f}{r['unresolved']:>12}{note}"
        )
    print(
        "\nn=14 ranked questions. The Recall@5 95% CI at the shipped setting is "
        "0.50-0.93, so differences below ~0.15 here are not distinguishable from "
        "noise on a set this size."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
