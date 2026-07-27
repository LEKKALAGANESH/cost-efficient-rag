"""One call layer for every LLM request (generation and judge).

Three things this owns that callers must not reimplement:

1. **Disk-backed response cache** keyed on ``sha256(model | messages | params)``.
   Free-tier quota is the actual scarce resource -- Groq's 200K TPD is roughly
   111 calls/day at 1,800 tokens -- and dev iteration burns far more calls than
   the final run.  A cached re-run costs zero quota.

2. **Token accounting summed across every attempt**, including failed retries
   and JSON repairs.  Billing counts those; an accounting that records only the
   successful attempt understates real spend.

3. **The two load-bearing Groq settings.** ``gpt-oss-120b`` is a reasoning
   model and reasoning tokens are emitted as output.  At default effort they
   can double or triple spend against a hard daily cap, so ``reasoning_effort``
   is pinned low and ``include_reasoning`` disabled.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.providers import ProviderChain, credential_for

#: Fraction of a provider's stated TPM the client will actually spend, as
#: headroom for token-estimation error and a window this process did not
#: start empty.
TOKEN_BUDGET_SAFETY_FACTOR = 0.75

#: Public list prices, for the "what would this have cost on a paid tier?"
#: line in the Discussion.  The run itself is $0 on both free tiers.
#: USD per 1M tokens, (prompt, completion).
LIST_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "groq/openai/gpt-oss-120b": (0.15, 0.75),
    "gemini/gemini-3.5-flash": (0.30, 2.50),
}


class LLMError(RuntimeError):
    """Generation or judging failed after every retry. Never fabricate instead."""


@dataclass
class LLMResponse:
    """One completed call, with usage summed over all attempts made."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    cache_hit: bool = False
    latency_ms: float = 0.0
    raw_attempts: list[str] = field(default_factory=list)
    degraded_from: str | None = None
    """Set when the chain's first choice was unavailable and a later provider
    answered instead. A run that quietly used its third choice produced numbers
    from a different model than the report claims, so this is recorded rather
    than inferred."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """What this call *would* cost at list price. Actual spend is $0."""
        prompt_rate, completion_rate = LIST_PRICES_USD_PER_MTOK.get(self.model, (0.0, 0.0))
        return (
            self.prompt_tokens * prompt_rate + self.completion_tokens * completion_rate
        ) / 1_000_000


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class DiskCache:
    """Content-addressed JSON cache. One file per (model, messages, params)."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root or get_settings().llm_cache_path)

    @staticmethod
    def key(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._root / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt cache entry is a miss, never a failure

    def put(self, key: str, value: dict[str, Any]) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            (self._root / f"{key}.json").write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # an unwritable cache degrades to no cache, never to an error


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
#: Retryable HTTP statuses. Checked against the exception's ``status_code``
#: rather than by substring: a bare "500" matched messages like "max 500
#: characters", turning a permanent bad request into three retries.
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Fallback for exceptions that carry the status only in their message.
#: The lookarounds reject digits *and* decimal points, so neither "1500" nor
#: "0.500" reads as a 500. Plain ``\b`` is not enough: "." is a word boundary.
_TRANSIENT_STATUS_RE = re.compile(r"(?<![\d.])(?:408|409|429|500|502|503|504)(?![\d.])")

#: Transport faults that carry no status code at all. "connection" alone was
#: too broad -- it retried ConnectionRefusedError against a permanently wrong
#: host three times before failing.
_TRANSIENT_MARKERS = (
    "rate limit",
    "ratelimit",
    "timeout",
    "timed out",
    "overloaded",
    "temporarily",
    "connection reset",
    "connection aborted",
    "connection error",
    "service unavailable",
)


def _is_transient(exc: Exception) -> bool:
    """Retry throttling and transport faults; fail fast on auth/schema errors."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS_CODES
    text = f"{type(exc).__name__} {exc}".lower()
    return bool(_TRANSIENT_STATUS_RE.search(text)) or any(m in text for m in _TRANSIENT_MARKERS)


def _estimate_tokens(messages: list[dict[str, Any]], max_tokens: int) -> int:
    """Rough prompt+completion size for pacing. ~4 characters per token is close
    enough: the window is re-measured every call, so a small overestimate costs
    a little throughput and nothing else."""
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return chars // 4 + max_tokens


class LLMClient:
    """Thin, cached, retrying wrapper over ``litellm.completion``."""

    def __init__(self, cache: DiskCache | None = None) -> None:
        self._cache = cache if cache is not None else DiskCache()
        # Per-model sliding token windows. Tokens-per-minute, not requests, is
        # what actually binds this workload: gpt-oss-120b allows 8,000 TPM and a
        # judge call costs ~2,000, so roughly four calls a minute however many
        # requests the tier permits. Discovering that by being refused wastes a
        # round trip and a retry budget on every rejection.
        self._token_windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._throttle_lock = threading.Lock()

    def _throttle(self, model: str, estimated_tokens: int) -> float:
        """Wait until ``estimated_tokens`` fit the trailing-minute budget."""
        settings = get_settings()
        ceiling = settings.tokens_per_minute_by_model.get(model, settings.default_tokens_per_minute)
        if not ceiling:
            return 0.0
        # Spend a fraction of the stated ceiling: the 4-chars-per-token estimate
        # can undershoot, and the provider's window is already partly consumed
        # when this process starts with an empty one.
        budget = max(1, int(ceiling * TOKEN_BUDGET_SAFETY_FACTOR))
        estimated = min(estimated_tokens, budget)
        slept = 0.0
        while True:
            with self._throttle_lock:
                window = self._token_windows[model]
                now = time.monotonic()
                while window and now - window[0][0] >= 60.0:
                    window.popleft()
                if sum(t for _, t in window) + estimated <= budget or not window:
                    window.append((now, estimated))
                    return slept
                wait = max(0.05, min(60.0 - (now - window[0][0]), 60.0))
            time.sleep(wait)
            slept += wait

    def _provider_params(self, model: str) -> dict[str, Any]:
        """Provider-specific knobs that materially change cost."""
        if model.startswith("groq/"):
            # gpt-oss-120b is a reasoning model; see module docstring.
            return {"reasoning_effort": "low", "include_reasoning": False}
        return {}

    def complete_with_chain(
        self,
        messages: list[dict[str, Any]],
        *,
        chain: ProviderChain,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Try each provider in order until one answers.

        The chain, not the model, is the unit of configuration. A provider whose
        credential is missing is skipped without an attempt; one that fails
        fatally has its circuit opened so the remaining N-1 calls in a run do
        not re-ask a key that is already known bad.

        Raises :class:`LLMError` only when *every* entry is exhausted, and the
        message names what each one did -- a single "invalid api key" tells you
        nothing about which of four providers produced it.
        """
        candidates = chain.available()
        if not candidates:
            raise LLMError(
                "no provider in the chain is usable: "
                + "; ".join(f"{m} ({why})" for m, why in chain.skipped())
            )

        failures: list[str] = []
        for index, state in enumerate(candidates):
            try:
                response = self.complete(
                    messages,
                    model=state.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    use_cache=use_cache,
                )
            except LLMError as exc:
                message = str(exc)
                failures.append(f"{state.model}: {message}")
                if _is_transient(exc):
                    chain.record_transient(state.model, message)
                else:
                    chain.record_fatal(state.model, message)
                continue
            chain.record_success(state.model)
            response.degraded_from = candidates[0].model if index else None
            return response

        raise LLMError("every provider in the chain failed -- " + " | ".join(failures))

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Call one model. Raises :class:`LLMError` rather than inventing text."""
        settings = get_settings()
        model = model or settings.generation_model
        params: dict[str, Any] = {
            "temperature": settings.generation_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.generation_max_tokens,
            **self._provider_params(model),
        }
        if response_format is not None:
            params["response_format"] = response_format

        cache_key = DiskCache.key(model, messages, params)
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return LLMResponse(
                    text=cached["text"],
                    model=model,
                    prompt_tokens=cached.get("prompt_tokens", 0),
                    completion_tokens=cached.get("completion_tokens", 0),
                    attempts=cached.get("attempts", 1),
                    cache_hit=True,
                    latency_ms=0.0,
                )

        import litellm

        prompt_tokens = 0
        completion_tokens = 0
        raw_attempts: list[str] = []
        last_error: Exception | None = None
        started = time.perf_counter()

        for attempt in range(1, settings.llm_max_retries + 2):
            # Paced before *every* attempt, not just the first. Throttling once
            # outside the loop lets one paced call emit a burst of unpaced
            # retries, which is precisely what the pacing exists to prevent.
            self._throttle(model, _estimate_tokens(messages, params["max_tokens"]))
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    timeout=settings.llm_timeout_seconds,
                    # Explicit, not ambient. See credential_for(): without it
                    # litellm reads os.environ, which .env never populates.
                    api_key=credential_for(model),
                    **params,
                )
            except Exception as exc:
                last_error = exc
                raw_attempts.append(f"<error attempt {attempt}: {exc}>")
                if not _is_transient(exc) or attempt > settings.llm_max_retries:
                    break
                # Ceiling must exceed the longest retryDelay a provider asks
                # for, or the retry gives up while still inside the window it
                # was told to wait out.
                time.sleep(min(2.0 ** (attempt - 1), settings.retry_backoff_cap_seconds))
                continue

            # Usage is read off the response, not from litellm's own cost
            # tracker: litellm's model table lags both providers and carries no
            # entry for gemini-3.5-flash, so its accounting is unreliable here.
            usage = getattr(response, "usage", None)
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            text = (response.choices[0].message.content or "").strip()
            raw_attempts.append(text)

            result = LLMResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attempts=attempt,
                cache_hit=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_attempts=raw_attempts,
            )
            if use_cache:
                self._cache.put(
                    cache_key,
                    {
                        "text": text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "attempts": attempt,
                        "model": model,
                    },
                )
            return result

        # len(raw_attempts), not the configured ceiling: a non-transient error
        # breaks after the first call, and reporting "3 attempt(s)" there sent
        # anyone reading the query log looking for a retry storm that never
        # happened.
        raise LLMError(f"{model} failed after {len(raw_attempts)} attempt(s): {last_error}")


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Process-wide client singleton (shares one cache directory)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
