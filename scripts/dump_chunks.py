"""Print every stored chunk so a human can hand-label relevance.

Labelling ``relevant_chunk_ids`` is not performable without this: SHA-256
digests are not human-readable, and the eval dataset is keyed on them.

    python scripts/dump_chunks.py                    # all chunks, 120-char preview
    python scripts/dump_chunks.py --width 400        # longer previews
    python scripts/dump_chunks.py --source notes.md  # one document
    python scripts/dump_chunks.py --json             # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vector_store import get_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=120, help="preview characters")
    parser.add_argument("--source", help="restrict to one source filename")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    store = get_vector_store()
    rows = store.scan(["chunk_id", "doc_key", "source", "chunk_index", "file_type", "text"])
    if args.source:
        rows = [r for r in rows if r["source"] == args.source]
    rows.sort(key=lambda r: (r["source"], r["chunk_index"]))

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("No chunks found. Run ingestion first:")
        print("  python -m scripts.ingest_corpus")
        return 1

    current: str | None = None
    for row in rows:
        if row["source"] != current:
            current = row["source"]
            print(f"\n{'=' * 100}\n{current}  ({row['file_type']})\n{'=' * 100}")
        preview = " ".join(row["text"].split())[: args.width]
        print(f"\n[{row['chunk_index']:>3}] {row['chunk_id']}")
        print(f"      {preview}{'...' if len(row['text']) > args.width else ''}")

    print(f"\n{'-' * 100}\n{len(rows)} chunks across {len({r['source'] for r in rows})} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
