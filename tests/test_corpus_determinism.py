"""The corpus must be byte-reproducible, or the gold labels silently rot.

``doc_key = sha256(file_bytes)`` and ``chunk_id`` hashes ``doc_key``, so a
generated document whose bytes differ between builds produces brand-new chunk
IDs every build.  The committed ``data/eval_dataset.json`` labels for that
document then stop resolving, every metric for those questions drops to zero,
and nothing raises.  A grader regenerating the corpus would see different
numbers from the README and have no way to tell why.

This was a real defect: reportlab embeds a creation timestamp and document ID
by default, so the PDF differed on every build until ``invariant=1`` was set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_corpus import build_pdf
from src.ingestion import build_chunk_records, compute_doc_key

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "raw_documents"
DATASET = REPO_ROOT / "data" / "eval_dataset.json"


def test_generated_pdf_is_byte_identical_across_builds(tmp_path: Path) -> None:
    first = build_pdf(tmp_path / "a.pdf").read_bytes()
    second = build_pdf(tmp_path / "b.pdf").read_bytes()
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest(), (
        "The generated PDF is not reproducible. doc_key hashes file bytes, so "
        "non-deterministic output invalidates every committed label for it."
    )


def test_rebuilding_the_pdf_preserves_its_doc_key(tmp_path: Path) -> None:
    committed = (CORPUS_DIR / "cost_model_reference.pdf").read_bytes()
    rebuilt = build_pdf(tmp_path / "rebuilt.pdf").read_bytes()
    assert compute_doc_key(rebuilt) == compute_doc_key(committed), (
        "Rebuilding the corpus changed the PDF's doc_key. Regenerate the "
        "dataset with scripts/build_eval_dataset.py, or restore determinism."
    )


def test_rebuilding_the_pdf_preserves_every_chunk_id(tmp_path: Path) -> None:
    committed, _, _ = build_chunk_records(
        (CORPUS_DIR / "cost_model_reference.pdf").read_bytes(), "cost_model_reference.pdf"
    )
    rebuilt, _, _ = build_chunk_records(
        build_pdf(tmp_path / "r.pdf").read_bytes(), "cost_model_reference.pdf"
    )
    assert [c.chunk_id for c in committed] == [c.chunk_id for c in rebuilt]


@pytest.mark.parametrize(
    "filename",
    [
        "cost_model_reference.pdf",
        "ingestion_operations_notes.md",
        "rag_evaluation_methodology.html",
        "vector_store_selection_guide.md",
    ],
)
def test_every_corpus_document_is_present_and_parses(filename: str) -> None:
    path = CORPUS_DIR / filename
    assert path.exists(), f"{filename} is missing from the corpus"
    records, _, file_type = build_chunk_records(path.read_bytes(), filename)
    assert records, f"{filename} produced no chunks"
    assert file_type in {"pdf", "md", "html"}


def test_corpus_covers_all_three_required_formats() -> None:
    """R1: PDF, HTML and Markdown ingestion must each be exercised."""
    suffixes = {p.suffix.lower() for p in CORPUS_DIR.iterdir() if p.is_file()}
    assert {".pdf", ".html", ".md"} <= suffixes


# ---------------------------------------------------------------------------
# The committed dataset must stay consistent with the committed corpus
# ---------------------------------------------------------------------------
def test_every_gold_chunk_id_resolves_against_the_committed_corpus() -> None:
    """The check that would have caught the non-deterministic PDF immediately."""
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    available: set[str] = set()
    for path in sorted(CORPUS_DIR.iterdir()):
        if not path.is_file():
            continue
        records, _, _ = build_chunk_records(path.read_bytes(), path.name)
        available.update(r.chunk_id for r in records)

    for question in dataset["questions"]:
        missing = set(question["relevant_chunk_ids"]) - available
        assert not missing, (
            f"{question['id']} references {len(missing)} chunk id(s) that no "
            f"longer exist in the corpus. Rebuild with "
            f"scripts/build_eval_dataset.py."
        )


def test_gold_excerpts_actually_appear_in_the_chunks_they_label() -> None:
    """The excerpts let a grader hand-verify a row without running anything.

    They are only worth committing if they are true, so this asserts they are.
    """
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    by_id: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.iterdir()):
        if not path.is_file():
            continue
        records, _, _ = build_chunk_records(path.read_bytes(), path.name)
        by_id.update({r.chunk_id: " ".join(r.text.split()) for r in records})

    for question in dataset["questions"]:
        for entry in question.get("relevant_chunks", []):
            excerpt = " ".join(entry["relevant_chunk_text"].split())
            chunk_text = by_id[entry["chunk_id"]]
            assert excerpt in chunk_text, (
                f"{question['id']}: the committed excerpt is not present in "
                f"chunk {entry['chunk_id'][:12]}."
            )


def test_dataset_meets_the_structural_requirements() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    assert 15 <= len(questions) <= 30, "R12: the set must hold 15-30 questions"
    assert len({q["id"] for q in questions}) == len(questions), "duplicate ids"

    counts = [len(q["relevant_chunk_ids"]) for q in questions]
    assert any(c == 0 for c in counts), "need an out-of-corpus question"
    assert any(c > 1 for c in counts), "need a multi-relevant question (Recall != Hit Rate)"
    assert any(c > 5 for c in counts), "need |relevant| > k to exercise truncated IDCG"

    fallback = [q for q in questions if q["expects_fallback"]]
    assert len(fallback) >= 1
    assert all(q["relevant_chunk_ids"] == [] for q in fallback)
    assert all(q["ground_truth_answer"] for q in questions if not q["expects_fallback"]), (
        "every answerable question needs a gold answer"
    )
