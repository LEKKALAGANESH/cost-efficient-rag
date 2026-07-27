# Ingestion Pipeline Operations Notes

*Synthetic reference document authored for the cost-efficient-rag evaluation
corpus. Contains no confidential or third-party material.*

## Idempotency contract

Re-ingesting an unmodified document must produce zero new vectors. The pipeline
guarantees this for identical triples of file content, chunk size, and chunk
overlap. It does *not* guarantee it across parameter changes: re-ingesting the
same file with a different chunk size legitimately produces different chunk
boundaries, therefore different identifiers, therefore additional rows. That is
a documented limitation of the contract, not a defect in it.

Document identity is `doc_key`, the SHA-256 digest of the raw file bytes. Chunk
identity is the SHA-256 digest of the document key, the chunk index, and the
chunk text joined by a delimiter. Deriving identity from file content rather
than from the filesystem path is what allows the same document to be ingested
by directory scan and by HTTP upload and receive identical identifiers both
ways.

## Why upsert and not append

The naive idempotency implementation reads the set of existing identifiers,
filters the incoming chunks against it, and appends the remainder. That
sequence is a time-of-check-to-time-of-use race. Two concurrent ingestion calls
 - from multiple workers, or from a client timeout followed by a retry - both
observe the same empty set, both embed, and both append. The duplicate-free
property then fails nondeterministically, and it fails on precisely the check a
reviewer is most likely to run.

The correct primitive is a merge keyed on the chunk identifier, which is
idempotent by construction and tolerant of concurrent execution. The
existing-identifier pre-filter is retained purely as an embedding-cost
optimization and carries no correctness role.

## Orphaned chunks on revision

Because the chunk identifier hashes the chunk text, editing a document changes
the identifiers of exactly the chunks that changed. A merge alone would insert
the new rows and leave the superseded rows in place, where retrieval can still
return them and generation can still cite them as current.

Cleanup must be keyed on the logical document name rather than on the content
hash. The revised file has a different content hash by definition, so deleting
by the new hash cannot reach the rows written under the old one. The pipeline
therefore compares the stored content hashes for a source name against the
incoming one, and deletes by source name when they differ.

## Text normalization

Normalization runs between the loader and the splitter and is the single
highest-leverage quality step for PDF corpora. PDF text extraction returns hard
line wraps in the middle of sentences, words hyphenated across line breaks, and
running headers and footers interleaved with the body.

Feeding that directly to a recursive splitter defeats the paragraph and
sentence boundary detection that is the splitter's entire advantage over
fixed-width splitting, and pollutes chunks with strings like "Page 7 of 42".

The normalizer performs five operations in order: it drops lines recurring on
more than fifty percent of pages, removes bare page numbers, rejoins words
hyphenated across line breaks, collapses single newlines inside a paragraph
while preserving blank-line paragraph breaks, and normalizes horizontal
whitespace. Boilerplate stripping is suppressed on documents shorter than three
pages, where a line shared by both pages is as likely to be content as a
running header.

## Chunking defaults

The defaults are a chunk size of 500 characters and an overlap of 50
characters, roughly ten percent. These are fixed a priori from published
practice and are not selected against the evaluation set.

An upper bound applies that is easy to miss. The all-MiniLM-L6-v2 embedding
model truncates input at 256 word-pieces, approximately 1,000 characters, and
does so silently with no warning. A chunk size above roughly 900 characters
therefore embeds only the head of each chunk while the full text still reaches
the generation prompt. In an evaluation this presents as "larger chunks
retrieve worse", which is true but for the wrong reason.

## Failure isolation

A single malformed file must never abort an ingestion batch. Every per-file
failure is caught, recorded as a status row with the source name and the error
text, and the remaining files continue processing. The response carries a
per-file status list so partial success is visible rather than silently
swallowed.

Password-protected PDFs, files whose extension does not match their actual
content, and files exceeding the configured upload ceiling all follow this
path. The upload ceiling is enforced by a streaming byte counter rather than by
reading the Content-Length header, which is client-controlled and therefore not
a limit.

## Embedding batching

Ingestion embeds in batches of 32 by default. Batching does not amortize model
load - that happens once, at construction of the singleton - but it does
amortize per-call framework overhead and produces larger matrix multiplications
with better cache and SIMD utilization. The encoder sorts inputs by length
internally to minimize padding waste, so batches of wildly heterogeneous chunk
lengths realize less of the benefit.
