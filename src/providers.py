"""Provider routing: an ordered fallback chain, configured by environment only.

The problem this solves is concrete. This project's evaluation harness was
complete, tested, and produced **no numbers at all**, because the single
configured provider returned `Invalid API Key`. One dead credential took out an
entire rubric line. A pipeline whose evidence depends on one vendor being
reachable is not an evaluation pipeline; it is a demo.

So the chain is the unit of configuration, not the model:

    LLM_PROVIDER_CHAIN=groq/openai/gpt-oss-120b,gemini/gemini-3.5-flash,ollama/llama3

Each entry is tried in order. A provider is skipped when its key is absent,
and abandoned when it fails in a way that retrying cannot fix (bad key, unknown
model, permission denied). Transient failures — 429, 5xx, timeouts — are
retried against the *same* provider with exponential backoff and jitter before
the chain moves on, because moving on immediately would spend the next
provider's quota on a problem that was about to resolve itself.

Switching providers is an env edit. There is no code path that names a vendor.

**Circuit breaker.** A provider that has failed fatally is marked down for the
process lifetime and skipped on subsequent calls. Without this, a 40-question
evaluation re-attempts a known-dead key 40 times, turning one misconfiguration
into a slow, confusing run.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field

from src.config import get_settings

#: Environment variable holding each provider's key, by routing prefix. Adding a
#: provider is a row here plus a chain entry -- never a change to call sites.
PROVIDER_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "azure": "AZURE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
}

#: Providers that serve a model on the local machine. They need no credential,
#: which is exactly why they belong at the end of a chain: a local model is the
#: difference between "the evaluation degraded" and "the evaluation aborted".
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "lm_studio", "hosted_vllm", "openai_like"})


@dataclass
class ProviderState:
    """Health of one chain entry, for the lifetime of the process."""

    model: str
    consecutive_failures: int = 0
    is_open: bool = False
    """Circuit open: the provider is presumed dead and skipped."""
    last_error: str | None = None

    @property
    def provider(self) -> str:
        return routing_provider(self.model)

    @property
    def requires_key(self) -> bool:
        return self.provider not in LOCAL_PROVIDERS

    @property
    def key_env(self) -> str | None:
        return PROVIDER_KEY_ENV.get(self.provider)


def routing_provider(model: str) -> str:
    """The routing prefix, which is *not* the model family.

    ``groq/openai/gpt-oss-120b`` is routed by Groq and belongs to OpenAI's
    lineage. Conflating the two is how a cross-family judging invariant passes
    for the wrong reason, so the two concepts stay separate functions.
    """
    return model.split("/", 1)[0] if "/" in model else "openai"


def has_credential(model: str) -> bool:
    """Whether this entry could possibly work right now."""
    provider = routing_provider(model)
    if provider in LOCAL_PROVIDERS:
        return True
    env = PROVIDER_KEY_ENV.get(provider)
    if env is None:
        return False
    return bool(os.environ.get(env) or _settings_key(env))


def _settings_key(env: str) -> str | None:
    """Keys also arrive via ``.env`` through pydantic-settings, not only the
    process environment. Checking one and not the other is how a configured
    machine reports itself unconfigured."""
    settings = get_settings()
    return getattr(settings, env.lower(), None)


class ProviderChain:
    """An ordered set of providers with health tracking.

    Thread-safe: the evaluation harness may drive several calls concurrently and
    a torn circuit-breaker state would let a dead provider back in.
    """

    def __init__(self, models: list[str], *, failure_threshold: int = 2) -> None:
        if not models:
            raise ValueError("provider chain is empty; set LLM_PROVIDER_CHAIN")
        self._states = [ProviderState(model=m) for m in models]
        self._failure_threshold = failure_threshold
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> ProviderChain:
        settings = get_settings()
        raw = settings.llm_provider_chain.strip()
        models = [m.strip() for m in raw.split(",") if m.strip()]
        return cls(models, failure_threshold=settings.provider_failure_threshold)

    @property
    def models(self) -> list[str]:
        return [s.model for s in self._states]

    def available(self) -> list[ProviderState]:
        """Entries worth attempting: circuit closed and credential present."""
        with self._lock:
            return [s for s in self._states if not s.is_open and has_credential(s.model)]

    def skipped(self) -> list[tuple[str, str]]:
        """(model, why) for entries the chain will not attempt. Reported rather
        than hidden, so a run that silently used the third choice is visible."""
        out: list[tuple[str, str]] = []
        with self._lock:
            for state in self._states:
                if state.is_open:
                    out.append((state.model, f"circuit open: {state.last_error}"))
                elif not has_credential(state.model):
                    out.append((state.model, f"no credential in {state.key_env}"))
        return out

    def record_success(self, model: str) -> None:
        with self._lock:
            for state in self._states:
                if state.model == model:
                    state.consecutive_failures = 0

    def record_fatal(self, model: str, error: str) -> None:
        """Open the circuit immediately. A bad key does not heal by being asked
        again 40 more times."""
        with self._lock:
            for state in self._states:
                if state.model == model:
                    state.is_open = True
                    state.last_error = error

    def record_transient(self, model: str, error: str) -> None:
        with self._lock:
            for state in self._states:
                if state.model == model:
                    state.consecutive_failures += 1
                    state.last_error = error
                    if state.consecutive_failures >= self._failure_threshold:
                        state.is_open = True

    def health(self) -> list[dict[str, object]]:
        """Serialisable health snapshot for the run report."""
        with self._lock:
            return [
                {
                    "model": s.model,
                    "provider": s.provider,
                    "requires_key": s.requires_key,
                    "credential_present": has_credential(s.model),
                    "circuit_open": s.is_open,
                    "consecutive_failures": s.consecutive_failures,
                    "last_error": s.last_error,
                }
                for s in self._states
            ]


def backoff_delay(
    attempt: int, *, base: float, cap: float, rng: random.Random | None = None
) -> float:
    """Exponential backoff with full jitter.

    Full jitter rather than a fixed multiplier: several concurrent callers that
    all back off by exactly the same amount retry in lockstep and re-collide,
    which is how a single 429 becomes a synchronised thundering herd.
    """
    rand = rng or random
    ceiling = min(cap, base * (2 ** max(0, attempt - 1)))
    return rand.uniform(0.0, ceiling)


@dataclass
class AttemptRecord:
    """One provider attempt, for the audit trail."""

    model: str
    attempt: int
    ok: bool
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class ChainResult:
    """Outcome of driving the chain, including everything that was tried."""

    model_used: str | None
    attempts: list[AttemptRecord] = field(default_factory=list)
    degraded: bool = False
    """True when the answer came from a provider that was not first choice."""

    @property
    def ok(self) -> bool:
        return self.model_used is not None


def sleep_with_backoff(attempt: int, *, base: float, cap: float) -> float:
    delay = backoff_delay(attempt, base=base, cap=cap)
    if delay > 0:
        time.sleep(delay)
    return delay
