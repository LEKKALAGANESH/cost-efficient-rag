"""Unit tests for hashing determinism, normalisation, and chunking.

Hermetic: no vector store, no embedding model, no network.
"""

from __future__ import annotations

import pytest

from src.ingestion import (
    UnsupportedFileTypeError,
    build_chunk_records,
    compute_doc_key,
    generate_chunk_id,
    load_document,
    normalize_text,
    split_text,
)

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_doc_key_is_content_derived_not_path_derived() -> None:
    """The same bytes must key identically regardless of arrival path.

    This is the invariant that keeps gold labels valid when a grader ingests
    by upload instead of from data/raw_documents/.
    """
    payload = b"identical content"
    assert compute_doc_key(payload) == compute_doc_key(payload)
    assert compute_doc_key(payload) != compute_doc_key(b"different content")


def test_doc_key_is_sha256_of_bytes() -> None:
    import hashlib

    payload = b"the quick brown fox"
    assert compute_doc_key(payload) == hashlib.sha256(payload).hexdigest()


def test_chunk_id_is_deterministic() -> None:
    a = generate_chunk_id("dk", 3, "some chunk text")
    b = generate_chunk_id("dk", 3, "some chunk text")
    assert a == b
    assert len(a) == 64


@pytest.mark.parametrize(
    ("doc_key", "index", "text"),
    [("other", 3, "same"), ("dk", 4, "same"), ("dk", 3, "different")],
)
def test_chunk_id_varies_with_every_input(doc_key: str, index: int, text: str) -> None:
    baseline = generate_chunk_id("dk", 3, "same")
    assert generate_chunk_id(doc_key, index, text) != baseline


def test_chunk_id_delimiter_prevents_field_smear() -> None:
    """(doc_key='ab', idx=1) must not collide with (doc_key='a', idx=..).

    Naive f-string concatenation without a delimiter makes distinct field
    tuples hash identically.
    """
    assert generate_chunk_id("ab", 1, "x") != generate_chunk_id("a", 1, "x")


def test_same_file_ingested_twice_yields_identical_chunk_ids(
    sample_markdown_bytes: bytes,
) -> None:
    first, key_a, _ = build_chunk_records(sample_markdown_bytes, "a.md")
    # Same bytes, different filename: IDs must still match, because identity
    # is content-derived.
    second, key_b, _ = build_chunk_records(sample_markdown_bytes, "elsewhere/b.md")
    assert key_a == key_b
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_dehyphenates_across_line_breaks() -> None:
    out = normalize_text(["The embed-\nding model produces vectors."])
    assert "embedding model" in out
    assert "embed-" not in out


def test_normalize_handles_consecutive_hyphenated_breaks() -> None:
    out = normalize_text(["multi-\nline-\nbreak here"])
    assert "multilinebreak here" in out


def test_normalize_collapses_intra_paragraph_newlines_preserving_breaks() -> None:
    out = normalize_text(["Line one\nline two.\n\nSecond paragraph."])
    assert "Line one line two." in out
    assert "\n\n" in out
    assert out.count("\n\n") == 1


def test_normalize_strips_headers_recurring_on_over_half_of_pages() -> None:
    pages = [f"ACME CONFIDENTIAL\nBody text for page {i}.\nFooter line 42" for i in range(1, 6)]
    out = normalize_text(pages)
    assert "ACME CONFIDENTIAL" not in out
    assert "Footer line 42" not in out
    assert "Body text for page 3." in out


def test_normalize_keeps_lines_recurring_on_under_half_of_pages() -> None:
    pages = ["SHARED\nunique one", "SHARED\nunique two", "unique three", "unique four"]
    out = normalize_text(pages)
    assert "SHARED" in out  # 2 of 4 pages is not > 50%


def test_normalize_does_not_strip_boilerplate_on_short_documents() -> None:
    """With <3 pages a repeated line is as likely to be content as a header."""
    out = normalize_text(["Important\nbody a", "Important\nbody b"])
    assert "Important" in out


def test_normalize_drops_bare_page_numbers() -> None:
    out = normalize_text(["Real content here.\n7\nPage 3 of 12\nMore content."])
    assert "Real content here." in out
    assert "More content." in out
    assert "Page 3 of 12" not in out


def test_normalize_normalises_whitespace() -> None:
    out = normalize_text(["too    many\t\tspaces"])
    assert out == "too many spaces"


def test_normalize_empty_input() -> None:
    assert normalize_text([""]) == ""
    assert normalize_text([]) == ""


# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------


def test_split_respects_chunk_size_without_overlap() -> None:
    text = ". ".join(f"sentence number {i} with filler words" for i in range(60))
    chunks = split_text(text, chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_split_prefers_paragraph_boundaries() -> None:
    text = "Para one is short.\n\nPara two is also short.\n\nPara three."
    chunks = split_text(text, chunk_size=30, chunk_overlap=0)
    assert "Para one is short." in chunks
    assert "Para three." in chunks


def test_split_overlap_carries_tail_forward() -> None:
    text = "\n\n".join(f"Paragraph {i} content here." for i in range(6))
    chunks = split_text(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1
    previous_tail = chunks[0][-10:]
    assert chunks[1].startswith(previous_tail.strip()[:5])


def test_split_returns_single_chunk_when_text_fits() -> None:
    assert split_text("short text", chunk_size=500, chunk_overlap=50) == ["short text"]


def test_split_empty_text_yields_no_chunks() -> None:
    assert split_text("", 500, 50) == []
    assert split_text("   \n  ", 500, 50) == []


def test_split_rejects_overlap_at_or_above_chunk_size() -> None:
    with pytest.raises(ValueError, match="must be <"):
        split_text("text", chunk_size=100, chunk_overlap=100)


def test_split_handles_unbroken_token_longer_than_window() -> None:
    """A pathological input must terminate, not loop or raise."""
    chunks = split_text("x" * 1000, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 10
    assert all(len(c) == 100 for c in chunks)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_markdown_loader(sample_markdown_bytes: bytes) -> None:
    pages, file_type = load_document(sample_markdown_bytes, "doc.md")
    assert file_type == "md"
    assert "Vector Databases" in pages[0]


def test_html_loader_strips_script_and_style(sample_html_bytes: bytes) -> None:
    pages, file_type = load_document(sample_html_bytes, "doc.html")
    assert file_type == "html"
    text = pages[0]
    assert "LanceDB is an embedded columnar" in text
    assert "should not be ingested" not in text
    assert "color:red" not in text


def test_unsupported_extension_rejected() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported extension"):
        load_document(b"data", "archive.zip")


def test_pdf_named_file_that_is_not_a_pdf_raises_parse_error() -> None:
    from src.ingestion import DocumentParseError

    with pytest.raises(DocumentParseError):
        load_document(b"this is definitely not a PDF", "fake.pdf")


def test_build_chunk_records_populates_metadata(sample_markdown_bytes: bytes) -> None:
    records, doc_key, file_type = build_chunk_records(
        sample_markdown_bytes, "nested/dir/doc.md", chunk_size=120, chunk_overlap=20
    )
    assert file_type == "md"
    assert records
    assert all(r.doc_key == doc_key for r in records)
    # Only the basename is retained; a client-supplied path is never a path.
    assert all(r.source == "doc.md" for r in records)
    assert [r.chunk_index for r in records] == list(range(len(records)))


def test_empty_document_yields_zero_chunks_not_an_error() -> None:
    records, _, _ = build_chunk_records(b"", "empty.md")
    assert records == []
