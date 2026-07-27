"""Ingest ``data/raw_documents/`` into the configured vector store.

Offline convenience for building the corpus and the gold labels.  The HTTP API
has no directory-ingest route on purpose (that would be an arbitrary-file-read
primitive); this script is the local-only equivalent.

    python scripts/ingest_corpus.py
    python scripts/ingest_corpus.py --reset      # drop the store first
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_settings
from src.ingest_service import ingest_directory
from src.vector_store import get_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(REPO_ROOT / "data" / "raw_documents"))
    parser.add_argument("--reset", action="store_true", help="delete the store first")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.reset and settings.vector_store_path.exists():
        shutil.rmtree(settings.vector_store_path)
        print(f"Removed {settings.vector_store_path}")

    store = get_vector_store()
    results = ingest_directory(
        args.directory,
        store,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"{'source':<45} {'status':<8} {'chunks':>7} {'embedded':>9} {'cached':>7}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.source:<45} {r.status:<8} {r.chunks_created:>7} "
            f"{r.chunks_embedded:>9} {r.chunks_skipped_cached:>7}"
            + (f"  {r.error}" if r.error else "")
        )
    print("-" * 80)
    print(f"vector_count = {store.count()}")
    return 0 if all(r.status != "failed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
