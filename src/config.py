"""Single source of truth for runtime configuration.

Every other module imports ``settings`` from here; nothing else reads
``os.environ`` directly.  Required secrets have no defaults, so a missing key
is a *boot* failure with a named field rather than a mid-request crash.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Named constants shared across modules.  The fallback string is asserted on
# verbatim by tests and by the eval harness, so it lives in exactly one place.
# --------------------------------------------------------------------------
FALLBACK_ANSWER = (
    "I do not have sufficient information in the provided context to answer this question."
)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".markdown": "md",
    ".txt": "md",  # plain text goes through the markdown path unchanged
}

#: ``all-MiniLM-L6-v2`` silently truncates input at 256 word-pieces
#: (~1000 characters).  Chunks larger than this embed only their head.
EMBEDDING_CHAR_TRUNCATION_WARNING_THRESHOLD = 900


class Settings(BaseSettings):
    """Typed, validated view of the process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (optional: the chain decides what it can reach) -----------
    #: No longer required at boot. A hard requirement here meant one dead
    #: credential stopped the whole application, including the parts that need
    #: no LLM at all -- ingestion, retrieval and every retrieval metric. The
    #: provider chain reports what it can reach and degrades instead.
    groq_api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)
    openrouter_api_key: str | None = Field(default=None, repr=False)
    together_api_key: str | None = Field(default=None, repr=False)
    deepinfra_api_key: str | None = Field(default=None, repr=False)
    azure_api_key: str | None = Field(default=None, repr=False)
    mistral_api_key: str | None = Field(default=None, repr=False)
    cohere_api_key: str | None = Field(default=None, repr=False)
    supabase_db_url: str | None = Field(default=None, repr=False)

    # --- Provider routing --------------------------------------------------
    #: Ordered fallback chain, tried left to right. Switching providers is an
    #: edit here; no call site names a vendor.
    llm_provider_chain: str = "groq/openai/gpt-oss-120b,gemini/gemini-3.5-flash"
    #: Groq first: Gemini's free tier is metered at 20 requests per *day* for
    #: this model (measured -- Google does not publish it), so it is a poor
    #: primary for a 19-call evaluation. It stays in the chain as a cross-family
    #: alternate, which is what the self-enhancement mitigation needs.
    judge_provider_chain: str = "groq/openai/gpt-oss-120b,gemini/gemini-3.5-flash"
    #: Consecutive transient failures before a provider's circuit opens.
    provider_failure_threshold: int = Field(default=2, ge=1, le=10)
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    retry_backoff_cap_seconds: float = Field(default=45.0, ge=0.0, le=300.0)
    #: When every provider in the chain is unreachable, fall back to a local
    #: deterministic evaluator so the harness reports a measurement instead of
    #: aborting. Clearly labelled in output -- it is a different instrument.
    enable_local_fallback: bool = True

    #: Provider tokens-per-minute ceilings, paced client-side before each
    #: request. TPM binds before RPM for this workload: gpt-oss-120b allows
    #: 8,000 TPM and a judge call costs ~2,000, so ~4 calls/minute regardless
    #: of the request ceiling. Measured, not assumed.
    default_tokens_per_minute: int = 8000

    @property
    def tokens_per_minute_by_model(self) -> dict[str, int]:
        return {
            "groq/openai/gpt-oss-120b": 8000,
            "gemini/gemini-3.5-flash": 12000,
        }

    # --- Vector backend ----------------------------------------------------
    vector_backend: Literal["lancedb", "supabase"] = "lancedb"
    vector_store_path: Path = Path("./data/lancedb_store")
    vector_table_name: str = "rag_chunks"

    # --- Embeddings --------------------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    # --- Chunking ----------------------------------------------------------
    default_chunk_size: int = Field(default=500, ge=50, le=8000)
    default_chunk_overlap: int = Field(default=50, ge=0, le=4000)

    # --- Retrieval ---------------------------------------------------------
    default_top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.3, ge=-1.0, le=1.0)

    # --- Generation / judge ------------------------------------------------
    generation_model: str = "groq/openai/gpt-oss-120b"
    judge_model: str = "gemini/gemini-3.5-flash"
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=1024, ge=64, le=8192)
    llm_timeout_seconds: int = Field(default=60, ge=1, le=600)
    #: 5, not 2. Generation and judging share one 8,000 TPM ceiling, so a judge
    #: call routinely arrives while the window is still draining and is asked to
    #: wait ~2s. Three attempts with an 8s backoff ceiling gave up inside that
    #: window and fell through to the offline evaluator, which cost a *worse
    #: measurement* rather than saving any quota -- a refused call bills nothing.
    llm_max_retries: int = Field(default=5, ge=0, le=10)
    llm_cache_path: Path = Path("./.cache/llm")

    # --- API limits --------------------------------------------------------
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    max_files_per_request: int = Field(default=20, ge=1, le=200)
    max_query_chars: int = Field(default=2000, ge=1, le=100_000)
    max_pdf_pages: int = Field(default=500, ge=1, le=10_000)
    allowed_filter_keys: str = "file_type,source,doc_key"

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    query_log_path: Path = Path("./logs/queries.jsonl")

    # ----------------------------------------------------------------------
    @property
    def filter_key_allowlist(self) -> frozenset[str]:
        """Keys a client may use in ``metadata_filter``.

        Anything outside this set is rejected with 422 *before* a predicate
        string is built, so untrusted input never reaches the query planner.
        """
        return frozenset(k.strip() for k in self.allowed_filter_keys.split(",") if k.strip())

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError(
                "default_chunk_overlap "
                f"({self.default_chunk_overlap}) must be strictly less than "
                f"default_chunk_size ({self.default_chunk_size}); otherwise the "
                "splitter cannot advance and would loop forever."
            )
        if self.default_chunk_size > EMBEDDING_CHAR_TRUNCATION_WARNING_THRESHOLD:
            warnings.warn(
                f"default_chunk_size={self.default_chunk_size} exceeds "
                f"{EMBEDDING_CHAR_TRUNCATION_WARNING_THRESHOLD} chars. "
                "all-MiniLM-L6-v2 truncates at 256 word-pieces (~1000 chars) "
                "with no error, so the tail of every chunk would be dropped "
                "from its embedding while remaining in the generation context.",
                stacklevel=2,
            )
        if self.vector_backend == "supabase" and not self.supabase_db_url:
            raise ValueError(
                "VECTOR_BACKEND=supabase requires SUPABASE_DB_URL to be set "
                "(use the IPv4 Session pooler string, not the direct string)."
            )
        return self


_settings: Settings | None = None


def get_settings(*, reload: bool = False) -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Lazily constructed so that importing this module never raises; the
    validation error surfaces at first use (app startup), naming the missing
    field.  ``reload=True`` is for tests that mutate the environment.
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
