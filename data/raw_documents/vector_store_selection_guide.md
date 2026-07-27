# Vector Store Selection Guide

*Synthetic reference document authored for the cost-efficient-rag evaluation
corpus. Contains no confidential or third-party material.*

## 1. Purpose and scope

This guide describes how the Platform Data team selects a vector store for a
new retrieval workload. It covers six candidate engines, the decision criteria
we apply, and the operational obligations each choice creates. It does not
cover embedding model selection, which is documented separately in the
Embedding Standards note.

The guide assumes a workload shaped like an internal knowledge base: a corpus
that is large in document count but queried infrequently, on the order of tens
of thousands of queries per month rather than tens per second.

## 2. Decision criteria

We score every candidate on five axes, weighted as follows:

| Axis | Weight | What it measures |
|---|---|---|
| Operational burden | 30% | Processes to run, accounts to hold, upgrades to schedule |
| Metadata filtering | 25% | Expressiveness of the predicate language |
| Scaling headroom | 20% | Vectors per node before a re-architecture is forced |
| Recovery story | 15% | Backup, restore, and point-in-time recovery |
| Cost at idle | 10% | What the workload bills when nobody is querying |

Operational burden carries the largest weight because in our post-incident
reviews it is the axis that most often predicts an outage. A team that must
remember to run a nightly compaction job will eventually forget.

## 3. Candidate engines

### 3.1 LanceDB

LanceDB is an embedded, disk-native columnar vector database built on the
Lance format, which is itself built on Apache Arrow and DataFusion. It runs
in-process; there is no server to operate and no account to create.

Reads go through an asynchronous I/O scheduler layered over an object-store
abstraction, which is why the same code path serves both local disk and S3.
That scheduler holds an explicit read-buffer budget, controlled by the
`io_buffer_size` setting, which defaults to two gigabytes. This matters for
capacity planning: LanceDB is disk-resident but not memory-free.

LanceDB uses optimistic concurrency control. Concurrent writers are permitted,
but each commit retries a bounded number of times, so write contention surfaces
as failed commits rather than as blocking. The team standard is therefore a
single dedicated ingestion worker per table.

The project's own documentation places comfortable single-node capacity in the
low single-digit millions of vectors, and directs larger deployments to the
enterprise product. Claims of billion-row capacity refer to Lance-format
lakehouse scans, not to single-node vector index serving.

Every write to a Lance table creates a new version. Without a scheduled
`optimize()` call and cleanup of superseded index files, disk usage grows
without bound. This is the single most common operational surprise reported by
teams adopting LanceDB.

### 3.2 ChromaDB

ChromaDB offers the lowest ingestion friction of any candidate and a
Python-first API that most engineers can use productively within an hour. Its
persistent local mode requires no infrastructure. Its metadata filter language
is weaker than LanceDB's or pgvector's, and its columnar storage efficiency is
lower, which becomes visible above roughly ten million vectors in embedded
mode.

### 3.3 Qdrant

Qdrant is a Rust vector engine with an HNSW implementation that benchmarks
extremely well, and a genuinely expressive payload filtering DSL. It is
available both as a self-hosted server binary and as a managed cloud service.

A critical caveat applies to its Python client's local mode: that mode contains
no HNSW index at all. It performs a brute-force scan and is documented for
development, testing, and demonstrations at approximately twenty thousand
points. Every performance and scaling claim about Qdrant applies to the server
binary, which is a process that must be run and managed. Teams that read the
benchmark numbers and then adopt local mode are comparing two different
systems.

### 3.4 pgvector

pgvector adds vector columns and approximate-nearest-neighbour indexes to
PostgreSQL, supporting both IVFFlat and HNSW index types. Its decisive
advantage is that it runs inside infrastructure most teams already operate:
backups, replication, monitoring, and access control already exist and already
have owners.

It also offers full ACID transactions and native joins between vectors and
relational metadata, which no other candidate provides. It is not embedded: it
requires a running PostgreSQL instance, self-hosted or on a managed tier.

Version 0.8.0 added `hnsw.iterative_scan`, which addresses approximate-index
overfiltering when a vector search is combined with a restrictive `WHERE`
clause. Before that setting existed, a highly selective metadata filter could
return far fewer rows than requested because the index was traversed once and
filtering was applied afterwards.

### 3.5 FAISS

FAISS is an index library, not a database. It provides an exceptional selection
of algorithms - Flat, IVF, HNSW, and product quantization variants - along with
`write_index` and `read_index` serialization and `IDSelector`-based filtering.

It provides no metadata store and no transactional persistence. A team adopting
FAISS is committing to build both. Most production users place FAISS behind a
wrapper that supplies those layers.

### 3.6 sqlite-vec

sqlite-vec is a SQLite extension for vector search. It is genuinely
zero-process and single-file, works anywhere SQLite works including serverless
functions, and inherits the whole of SQLite's SQL dialect for metadata
filtering, which makes it strictly more expressive than a predicate-string
interface.

It is the youngest candidate. Stable releases perform brute-force search only.
DiskANN support has appeared in the pre-release line but has not yet shipped in
a stable release, so index availability must be verified against the exact
version pinned.

## 4. Standing recommendation

For a large, lightly queried corpus with a single ingestion writer, the team
standard is LanceDB, with pgvector as the alternative whenever the workload
also needs relational joins, multi-writer concurrency, or an existing
PostgreSQL operational envelope.

Adopt a managed vector service instead when any one of three conditions holds:
the corpus exceeds what a single node can index and hold in its working set;
the workload requires multi-region high availability or a compliance guarantee
that would otherwise need bespoke engineering; or write concurrency from
multiple independent services exceeds what a single-writer ingestion path can
absorb.

Query volume is deliberately absent from that list. Higher query volume
strengthens rather than weakens the case for the self-hosted path, because
usage-based managed pricing scales with queries while an already-provisioned
host does not.

## 5. Obligations accepted

Choosing an embedded store transfers four responsibilities to the adopting
team. They must be staffed, not merely acknowledged.

1. **Backup and point-in-time recovery.** No managed service is providing
   these. For a twenty-gigabyte store, snapshot storage costs roughly fifty
   cents to one dollar per month, which is thirty to sixty percent of the
   storage line it footnotes.
2. **Compaction.** A scheduled `optimize()` and index cleanup, or disk grows
   without bound.
3. **Index build capacity.** At ten million vectors the IVF k-means training
   sample alone is approximately 1.24 gigabytes before the streaming write
   phase begins, realistically requiring eight to sixteen gibibytes of memory
   and several hours. A single-box architecture has no offline build path.
4. **Write serialization.** One ingestion worker, enforced by deployment
   topology rather than by convention.
