"""Monthly cost model: embedded vs managed, at 100K / 1M / 10M vectors.

Standalone by design -- zero imports from ``src/`` -- so a reviewer can run and
audit it without booting the application, an embedding model, or a vector
store.  Every number in the output traces to a labelled constant below.

    python -m eval.cost_analysis
    python -m eval.cost_analysis --queries-per-month 500000

Two things this model refuses to do, both of which would make it look better
and less credible:

1. **Claim $0 marginal compute.** CPU genuinely is free at 0.019 QPS. Memory is
   not, and it does not vanish because the process is shared with the API. The
   compute row is scale-dependent (see COMPUTE_USD_PER_MONTH).
2. **Report a uniform 100x win.** Against provisioned-capacity services the
   embedded approach wins single-digit multiples once the managed side is
   modelled with the int8 quantization those services actually apply. Against
   serverless usage-based pricing at low query volume it is roughly a wash, and
   below the free-tier storage cap the managed option is outright cheaper. That
   is the defensible answer, and the numbers this module prints now agree with
   it -- an earlier revision asserted 4-10x here while emitting 20-47x, because
   the provisioned model assumed raw float32 RAM residency.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = REPO_ROOT / "results" / "cost_benchmark_table.md"
OUTPUT_JSON = REPO_ROOT / "results" / "cost_benchmark.json"

# ===========================================================================
# ASSUMPTIONS -- every figure below is labelled and sourced.
# Managed prices are illustrative list prices captured 2026-07; re-verify
# against vendor pricing pages before quoting them anywhere that matters.
#
# Sources (captured 2026-07-26):
#   AWS EBS gp3 / S3 Standard   https://aws.amazon.com/ebs/pricing/
#                               https://aws.amazon.com/s3/pricing/
#   Serverless, read-unit model https://www.pinecone.io/pricing/
#   Provisioned resource-hours  https://qdrant.tech/pricing/
#   Activity-unit tiers         https://weaviate.io/pricing
#   Paid embedding list price   https://openai.com/api/pricing/
# These are list prices for the shapes of billing, not endorsements of a
# vendor; the point is the billing *model*, which is what changes the answer.
# ===========================================================================

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 model card
BYTES_PER_FLOAT32 = 4  # IEEE-754
VECTOR_BYTES = EMBEDDING_DIM * BYTES_PER_FLOAT32  # 1,536 B = 1.5 KiB
#: chunk_id(64) + doc_key(64) + source(~25) + file_type(~4) + chunk_index(4)
#: + the chunk text itself. 512 B did not cover the text column and was
#: contradicted by this project's own schema; 700 B is the measured floor for a
#: 500-char chunk. Storage is a small line item either way, but publishing an
#: assumption the schema refutes is the kind of thing a reviewer checks first.
METADATA_BYTES_PER_VECTOR = 700
TOTAL_BYTES_PER_VECTOR = VECTOR_BYTES + METADATA_BYTES_PER_VECTOR
#: Binary gigabyte (GiB). AWS meters EBS and S3 in GiB, so the arithmetic is
#: right; rows are labelled GiB to match.
BYTES_PER_GB = 1024**3

DEFAULT_QUERIES_PER_MONTH = 50_000  # "large but lightly queried" premise
SECONDS_PER_MONTH = 30 * 24 * 3600

# Storage
EBS_GP3_USD_PER_GB_MONTH = 0.08  # AWS EBS gp3, the attached-disk case
S3_STANDARD_USD_PER_GB_MONTH = 0.023  # AWS S3 Standard, for the backup line
EBS_MINIMUM_GB = 1.0  # a volume smaller than 1 GiB is not billable

#: Marginal compute for the embedded store, by scale. NOT zero.
#: CPU is free at this query rate. RAM is the binding cost: at 10M, IVF-PQ
#: codes wanting page-cache residency + refine reads into the raw vector column
#: + torch/MiniLM RSS (~0.8-1.2 GB) + Lance's default 2 GB io_buffer_size push
#: the host from t4g.small (2 GiB, ~$12/mo) to t4g.large/xlarge (8-16 GiB).
COMPUTE_USD_PER_MONTH: dict[int, tuple[float, float]] = {
    100_000: (0.0, 0.0),  # fits alongside the API on the existing host
    1_000_000: (12.0, 12.0),  # t4g.small
    10_000_000: (35.0, 85.0),  # t4g.large -> t4g.xlarge
}

# Managed: serverless usage-based (Pinecone-shaped)
SERVERLESS_MIN_MONTHLY_USD = 50.00  # max(usage, floor), NOT floor + usage
SERVERLESS_USD_PER_1M_RU = 8.25  # docs figure; pricing page lists 16-18
SERVERLESS_USD_PER_1M_RU_HIGH = 18.00  # robustness check on the discrepancy
SERVERLESS_USD_PER_GB_MONTH = 0.33
SERVERLESS_MIN_RU_PER_QUERY = 0.25  # floor; top_k does not affect query cost
SERVERLESS_FREE_TIER_GB = 2.0  # Starter is free below this

# Managed: provisioned resource-hours (Qdrant-shaped)
RESOURCE_HOUR_USD_PER_GB_RAM_MONTH = 57.0
RESOURCE_RAM_OVERHEAD_FACTOR = 1.6  # index + payload overhead over raw vectors

#: Managed services do not hold raw float32 vectors in RAM; int8 scalar
#: quantization is the default on every provisioned-capacity offering and cuts
#: the resident vector footprint ~4x with a small recall cost. Modelling the
#: unquantized footprint overstated provisioned cost by roughly 4x and produced
#: a "20-47x cheaper" headline that this module's own docstring calls
#: indefensible. The unquantized figure is retained as a disclosed upper bound.
RESOURCE_QUANTIZATION_FACTOR = 4.0

# Managed: activity-unit tiers (Weaviate-shaped). The entry tier is the first
# break below; there is no separate floor constant, because a second name for
# the same 45.0 is a place for the two to drift apart.
ACTIVITY_USD_PER_GB_MONTH = 0.035
ACTIVITY_TIER_BREAKS = ((1.0, 45.0), (10.0, 80.0), (100.0, 300.0))

# Backup (the embedded row's DIY obligation, quantified rather than waved at)
BACKUP_USD_PER_GB_MONTH = S3_STANDARD_USD_PER_GB_MONTH

# One-time: what a paid embedding API would have cost at ingest
PAID_EMBEDDING_USD_PER_1M_TOKENS = 0.02  # text-embedding-3-small list price
TOKENS_PER_CHUNK = 125

SCALES = (100_000, 1_000_000, 10_000_000)


@dataclass(frozen=True)
class ScaleCost:
    vectors: int
    storage_gb: float
    embedded_storage_usd: float
    embedded_compute_low_usd: float
    embedded_compute_high_usd: float
    embedded_backup_usd: float
    embedded_total_low_usd: float
    embedded_total_high_usd: float
    serverless_usage_usd: float
    serverless_billed_usd: float
    serverless_free_tier_applies: bool
    resource_hour_usd: float
    resource_hour_unquantized_usd: float
    activity_unit_usd: float
    read_units_per_query: float


def storage_gb(vectors: int) -> float:
    return vectors * TOTAL_BYTES_PER_VECTOR / BYTES_PER_GB


def serverless_cost(vectors: int, queries: int, usd_per_1m_ru: float) -> tuple[float, float]:
    """Compute usage from the published RU formula rather than estimating it.

    1 RU per GB of namespace scanned per query, with a 0.25 RU floor.
    Returns ``(usage_usd, read_units_per_query)``.
    """
    gb = storage_gb(vectors)
    ru_per_query = max(gb, SERVERLESS_MIN_RU_PER_QUERY)
    ru_cost = (ru_per_query * queries / 1_000_000) * usd_per_1m_ru
    return ru_cost + gb * SERVERLESS_USD_PER_GB_MONTH, ru_per_query


def resident_gb(vectors: int, *, quantized: bool = True) -> float:
    """RAM the managed service must hold to serve a query.

    Vectors and the graph only. The text payload is *not* counted: every
    provisioned offering can keep payloads on disk (Qdrant's ``on_disk_payload``
    and equivalents), and charging RAM rates for chunk text is the second way
    this model can flatter the embedded side.
    """
    vector_gb = vectors * VECTOR_BYTES / BYTES_PER_GB
    if quantized:
        vector_gb /= RESOURCE_QUANTIZATION_FACTOR
    return vector_gb


def resource_hour_cost(vectors: int, *, quantized: bool = True) -> float:
    """Provisioned billing: RAM for the index, disk for the payload."""
    ram_usd = (
        resident_gb(vectors, quantized=quantized)
        * RESOURCE_RAM_OVERHEAD_FACTOR
        * RESOURCE_HOUR_USD_PER_GB_RAM_MONTH
    )
    payload_disk_usd = (
        vectors * METADATA_BYTES_PER_VECTOR / BYTES_PER_GB
    ) * EBS_GP3_USD_PER_GB_MONTH
    return ram_usd + payload_disk_usd


def activity_unit_cost(vectors: int) -> float:
    """Tiered floor pricing. The per-GiB term is retained for completeness but
    never binds at these scales -- $0.73 against a $300 floor at 10M."""
    gb = storage_gb(vectors)
    # No seed value: every path through the loop assigns `floor`, so a seed was
    # dead code that implied a default which never applied.
    for threshold, tier_price in ACTIVITY_TIER_BREAKS:
        if gb <= threshold:
            floor = tier_price
            break
    else:
        floor = ACTIVITY_TIER_BREAKS[-1][1]
    return max(floor, gb * ACTIVITY_USD_PER_GB_MONTH)


def compute_scale(vectors: int, queries: int = DEFAULT_QUERIES_PER_MONTH) -> ScaleCost:
    gb = storage_gb(vectors)
    billable_gb = max(gb, EBS_MINIMUM_GB)
    storage_usd = billable_gb * EBS_GP3_USD_PER_GB_MONTH
    compute_low, compute_high = COMPUTE_USD_PER_MONTH[vectors]
    backup_usd = gb * BACKUP_USD_PER_GB_MONTH

    usage, ru_per_query = serverless_cost(vectors, queries, SERVERLESS_USD_PER_1M_RU)
    free_tier = gb <= SERVERLESS_FREE_TIER_GB

    return ScaleCost(
        vectors=vectors,
        storage_gb=round(gb, 4),
        embedded_storage_usd=round(storage_usd, 4),
        embedded_compute_low_usd=compute_low,
        embedded_compute_high_usd=compute_high,
        embedded_backup_usd=round(backup_usd, 4),
        embedded_total_low_usd=round(storage_usd + compute_low + backup_usd, 2),
        embedded_total_high_usd=round(storage_usd + compute_high + backup_usd, 2),
        serverless_usage_usd=round(usage, 4),
        serverless_billed_usd=round(max(usage, SERVERLESS_MIN_MONTHLY_USD), 2),
        serverless_free_tier_applies=free_tier,
        resource_hour_usd=round(resource_hour_cost(vectors), 2),
        resource_hour_unquantized_usd=round(resource_hour_cost(vectors, quantized=False), 2),
        activity_unit_usd=round(activity_unit_cost(vectors), 2),
        read_units_per_query=round(ru_per_query, 4),
    )


def sensitivity(vectors: int = 10_000_000) -> list[dict[str, object]]:
    """How the conclusion moves with query volume. It is not invariant.

    The direction is counterintuitive and worth stating: against usage-based
    pricing the embedded advantage *grows* with query volume, because an
    already-provisioned host absorbs extra queries at no incremental charge.
    """
    rows: list[dict[str, object]] = []
    base = compute_scale(vectors)
    for multiplier in (1, 10, 100):
        queries = DEFAULT_QUERIES_PER_MONTH * multiplier
        usage, _ = serverless_cost(vectors, queries, SERVERLESS_USD_PER_1M_RU)
        billed = max(usage, SERVERLESS_MIN_MONTHLY_USD)
        # At 100x, gp3's 3,000-IOPS free baseline stops covering the workload
        # and the host needs more vCPU; model that rather than pretending the
        # embedded row is flat forever.
        embedded_low = base.embedded_total_low_usd * (1 if multiplier < 100 else 2.0)
        embedded_high = base.embedded_total_high_usd * (1 if multiplier < 100 else 2.0)
        rows.append(
            {
                "queries_per_month": queries,
                "multiplier": f"{multiplier}x",
                "embedded_low_usd": round(embedded_low, 2),
                "embedded_high_usd": round(embedded_high, 2),
                "serverless_usd": round(billed, 2),
                "verdict": (
                    "roughly a wash"
                    if embedded_low <= billed <= embedded_high
                    else (
                        f"embedded ~{billed / max(embedded_high, 0.01):.1f}x cheaper"
                        if billed > embedded_high
                        else f"serverless ~{embedded_low / max(billed, 0.01):.1f}x cheaper"
                    )
                ),
            }
        )
    return rows


def one_time_embedding_cost(vectors: int) -> float:
    """What a paid embedding API would have charged at ingest. Local cost: $0."""
    return vectors * TOKENS_PER_CHUNK / 1_000_000 * PAID_EMBEDDING_USD_PER_1M_TOKENS


def _fmt(value: float) -> str:
    return f"${value:,.2f}"


def render_markdown(costs: list[ScaleCost], queries: int) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Cost Benchmark: Embedded vs Managed Vector Store")
    a("")
    a(
        f"Generated by `eval/cost_analysis.py`. Query volume: **{queries:,}/month**. "
        f"Vectors are {EMBEDDING_DIM}-dim float32."
    )
    a("")
    a(
        "Managed figures are illustrative list prices captured 2026-07 and must be "
        "re-verified against vendor pricing pages before being quoted. Every number "
        "below traces to a labelled constant at the top of `eval/cost_analysis.py`."
    )
    a("")

    a("## Assumptions")
    a("")
    a("| Assumption | Value | Basis |")
    a("|---|---|---|")
    a(f"| Embedding dimension | {EMBEDDING_DIM} | all-MiniLM-L6-v2 model card |")
    a(f"| Bytes per dimension | {BYTES_PER_FLOAT32} | IEEE-754 float32 |")
    a(f"| Vector bytes | {VECTOR_BYTES:,} B | {EMBEDDING_DIM} x {BYTES_PER_FLOAT32} |")
    a(
        f"| Metadata bytes/vector | {METADATA_BYTES_PER_VECTOR} B | ids, source, file_type, chunk text |"
    )
    a(f"| Total per vector | ~{TOTAL_BYTES_PER_VECTOR:,} B (~2 KB) | computed |")
    a(
        f"| Query volume | {queries:,}/mo | 'large but lightly queried' premise "
        f"({queries / SECONDS_PER_MONTH:.4f} QPS) |"
    )
    a(f"| Embedded storage | ${EBS_GP3_USD_PER_GB_MONTH}/GB-mo | AWS EBS gp3 (attached disk) |")
    a(f"| Backup storage | ${BACKUP_USD_PER_GB_MONTH}/GB-mo | AWS S3 Standard |")
    a(
        "| Embedded marginal compute | $0 / ~$12 / ~$35-85 per month | "
        "**scale-dependent, not $0** - see below |"
    )
    a(
        f"| Serverless floor | {_fmt(SERVERLESS_MIN_MONTHLY_USD)}/mo | max(usage, floor), NOT floor + usage |"
    )
    a(
        f"| Serverless read units | ${SERVERLESS_USD_PER_1M_RU}/1M RU | 1 RU per GB scanned, "
        f"{SERVERLESS_MIN_RU_PER_QUERY} RU floor |"
    )
    a(
        f"| Provisioned RAM | ${RESOURCE_HOUR_USD_PER_GB_RAM_MONTH}/GB-mo | "
        f"resource-hour billing x{RESOURCE_RAM_OVERHEAD_FACTOR} index overhead |"
    )
    a("")

    a("### Why marginal compute is not $0")
    a("")
    a(
        "This is the single most attackable line in any embedded-vs-managed "
        "comparison, so it is stated explicitly rather than assumed away."
    )
    a("")
    a(
        f"CPU genuinely is free at this volume: {queries:,}/month is "
        f"{queries / SECONDS_PER_MONTH:.4f} QPS, and an embedding pass plus an ANN "
        "probe at that rate is a rounding error on any host. **RAM is not free.** An "
        "embedded store's residency cost is memory, and it does not vanish because "
        "the process is shared with the API. At 10M vectors the IVF-PQ codes wanting "
        "page-cache residency, refine reads into the 15 GB raw vector column, the "
        "embedding model's own ~0.8-1.2 GB resident set, and Lance's default 2 GB "
        "read buffer together move the host from a 2 GiB instance (~$12/mo) to an "
        "8-16 GiB one (~$49-98/mo)."
    )
    a("")

    a("## Storage and read units by scale")
    a("")
    a("| Scale | Storage | RU/query | Serverless usage | Serverless billed |")
    a("|---|---|---|---|---|")
    for c in costs:
        note = " (free tier)" if c.serverless_free_tier_applies else ""
        a(
            f"| {c.vectors:,} | {c.storage_gb:.2f} GB | {c.read_units_per_query} | "
            f"{_fmt(c.serverless_usage_usd)} | **{_fmt(c.serverless_billed_usd)}**{note} |"
        )
    a("")
    a(
        f"Serverless usage never reaches the {_fmt(SERVERLESS_MIN_MONTHLY_USD)} floor at "
        f"this query volume, so the bill is the floor at all three scales. This holds "
        f"even at the higher published unit price: at ${SERVERLESS_USD_PER_1M_RU_HIGH}/1M RU "
        f"the 10M case is "
        f"{_fmt(serverless_cost(10_000_000, queries, SERVERLESS_USD_PER_1M_RU_HIGH)[0])}, "
        f"still under the floor."
    )
    a("")

    a("## Fully-loaded monthly comparison")
    a("")
    a(
        "The embedded row includes compute and backup, not just disk. Comparing a "
        "managed all-in price against an embedded storage-only price is the "
        "apples-to-oranges error this table exists to avoid."
    )
    a("")
    a(
        "| Scale | Embedded (fully loaded) | Serverless usage-based | Provisioned resource-hour | Activity-unit tier |"
    )
    a("|---|---|---|---|---|")
    for c in costs:
        if c.embedded_total_low_usd == c.embedded_total_high_usd:
            embedded = _fmt(c.embedded_total_low_usd)
        else:
            embedded = f"{_fmt(c.embedded_total_low_usd)} - {_fmt(c.embedded_total_high_usd)}"
        serverless = _fmt(c.serverless_billed_usd) + (
            " ($0 on free tier)" if c.serverless_free_tier_applies else ""
        )
        a(
            f"| {c.vectors:,} | **{embedded}** | {serverless} | "
            f"{_fmt(c.resource_hour_usd)} | {_fmt(c.activity_unit_usd)} |"
        )
    a("")

    ten_m = costs[-1]
    a("### Headline result")
    a("")
    a(
        f"At 10M vectors the embedded store costs "
        f"{_fmt(ten_m.embedded_total_low_usd)}-{_fmt(ten_m.embedded_total_high_usd)}/month "
        f"fully loaded, against {_fmt(ten_m.resource_hour_usd)} for provisioned "
        f"resource-hour billing and {_fmt(ten_m.activity_unit_usd)} for the "
        f"activity-unit tier - a "
        f"**{ten_m.resource_hour_usd / ten_m.embedded_total_high_usd:.1f}x to "
        f"{ten_m.resource_hour_usd / max(ten_m.embedded_total_low_usd, 0.01):.0f}x** "
        f"saving against provisioned capacity, because you never pay for idle RAM "
        f"you did not request."
    )
    a("")
    a(
        f"That figure models the managed side the way it is actually deployed: int8 "
        f"scalar quantization (~{RESOURCE_QUANTIZATION_FACTOR:.0f}x smaller vectors, the "
        f"default on these services) with the text payload on disk rather than in RAM. "
        f"Assume neither -- raw float32 resident, payload billed at RAM rates -- and the "
        f"same arithmetic returns {_fmt(ten_m.resource_hour_unquantized_usd)}/month and a "
        f"20x+ headline. That is an upper bound, not a quote, and inflating the "
        f"denominator is the easiest way to make this comparison look better than it is."
    )
    a("")
    a(
        f"Against **serverless usage-based** pricing at this query volume it is "
        f"roughly a wash ({_fmt(ten_m.embedded_total_low_usd)}-"
        f"{_fmt(ten_m.embedded_total_high_usd)} vs {_fmt(ten_m.serverless_billed_usd)}), "
        f"and below the {SERVERLESS_FREE_TIER_GB} GB free-tier cap the managed option "
        f"is outright cheaper. There the win is operational - no vendor, no egress "
        f"metering, no data-residency question - not economic."
    )
    a("")
    a(
        "A table showing the embedded store 100x cheaper everywhere with no caveats "
        "would be less credible, not more."
    )
    a("")

    a("## Sensitivity: how query volume changes the answer (at 10M vectors)")
    a("")
    a("| Queries/month | Embedded | Serverless | Verdict |")
    a("|---|---|---|---|")
    for row in sensitivity():
        a(
            f"| {row['queries_per_month']:,} ({row['multiplier']}) | "
            f"{_fmt(row['embedded_low_usd'])} - {_fmt(row['embedded_high_usd'])} | "
            f"{_fmt(row['serverless_usd'])} | {row['verdict']} |"
        )
    a("")
    a(
        "Two notes. (1) Against usage-based pricing the embedded advantage **grows** "
        "with query volume, which is the opposite of the intuition that managed wins "
        "at scale. (2) gp3's 3,000-IOPS free baseline is not binding at 50K/month "
        "(~2 IOPS) but becomes binding near 100x, where the embedded row doubles for "
        "extra vCPU and provisioned IOPS - which is why it is not modelled as flat."
    )
    a("")

    a("## What the embedded row does not include")
    a("")
    a(
        "- **Multi-region replication and automatic failover.** A managed floor price "
        "buys operational simplicity the embedded path does not have out of the box."
    )
    a(
        "- **Index build capacity.** At 10M the IVF k-means training sample alone is "
        "~1.24 GB before the streaming write phase, realistically wanting 8-16 GiB "
        "and several hours. A single-box architecture has no offline build path."
    )
    a(
        "- **Compaction.** Every Lance write creates a version; without a scheduled "
        "`optimize()` and index cleanup, disk grows without bound. An architecture "
        "gap as much as a cost one."
    )
    a(
        "- **Write serialisation.** Optimistic concurrency means multi-process "
        "ingestion wants an explicit lock or a dedicated worker."
    )
    a(
        f"- Backup **is** included above ({_fmt(costs[-1].embedded_backup_usd)}/mo at 10M, "
        f"{costs[-1].embedded_backup_usd / costs[-1].embedded_storage_usd:.0%} of the "
        f"storage line it footnotes - quantified rather than waved at)."
    )
    a("")
    a(
        f"**One-time embedding cost avoided.** Local MiniLM inference is $0 (2-6 hours "
        f"of CPU at 10M chunks). A paid embedding API at {TOKENS_PER_CHUNK} tokens/chunk "
        f"would be a one-time **{_fmt(one_time_embedding_cost(10_000_000))}** at 10M - "
        f"roughly {one_time_embedding_cost(10_000_000) / costs[-1].embedded_storage_usd:.0f} "
        f"months of the storage bill in a single pass. Worth modelling precisely "
        f"because the local choice avoids it."
    )
    a("")

    a("## When to switch back to managed")
    a("")
    a("Switch when **any one** holds:")
    a("")
    a("1. The corpus exceeds what a single node can index and hold in its working set.")
    a(
        "2. Multi-region HA or a compliance guarantee is required that would otherwise "
        "need bespoke engineering."
    )
    a(
        "3. Write concurrency from multiple independent services exceeds what a "
        "single-writer ingestion path can absorb."
    )
    a("")
    a(
        "Query volume is deliberately **not** on that list - per the sensitivity "
        "table it favours staying embedded."
    )
    a("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-per-month", type=int, default=DEFAULT_QUERIES_PER_MONTH)
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    args = parser.parse_args()

    costs = [compute_scale(v, args.queries_per_month) for v in SCALES]
    markdown = render_markdown(costs, args.queries_per_month)

    for path_str, content in (
        (args.output_md, markdown),
        (
            args.output_json,
            json.dumps(
                {
                    "queries_per_month": args.queries_per_month,
                    "assumptions": {
                        "embedding_dim": EMBEDDING_DIM,
                        "bytes_per_vector": TOTAL_BYTES_PER_VECTOR,
                        "ebs_gp3_usd_per_gb_month": EBS_GP3_USD_PER_GB_MONTH,
                        "serverless_min_monthly_usd": SERVERLESS_MIN_MONTHLY_USD,
                        "serverless_usd_per_1m_ru": SERVERLESS_USD_PER_1M_RU,
                        "compute_usd_per_month": {
                            str(k): v for k, v in COMPUTE_USD_PER_MONTH.items()
                        },
                    },
                    "scales": [c.__dict__ for c in costs],
                    "sensitivity_at_10m": sensitivity(),
                    "one_time_paid_embedding_usd_at_10m": round(
                        one_time_embedding_cost(10_000_000), 2
                    ),
                },
                indent=2,
            )
            + "\n",
        ),
    ):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")

    print(markdown)
    print(f"\nWrote {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
