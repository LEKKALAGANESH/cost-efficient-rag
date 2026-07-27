"""LLM client: caching, retry classification, and token accounting.

All offline -- ``litellm.completion`` is monkeypatched, so no key is needed and
no free-tier quota is consumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import get_settings
from src.llm_client import (
    TOKEN_BUDGET_SAFETY_FACTOR,
    DiskCache,
    LLMClient,
    LLMError,
    LLMResponse,
    _estimate_tokens,
    _is_transient,
)


class FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content: str, prompt: int = 100, completion: int = 20) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(prompt, completion)


@pytest.fixture
def client(tmp_path: Path) -> LLMClient:
    return LLMClient(cache=DiskCache(tmp_path / "llmcache"))


def patch_completion(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> list[dict[str, Any]]:
    """Replace litellm.completion; returns a list recording each call's kwargs."""
    import litellm

    calls: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = behaviour(len(calls)) if callable(behaviour) else behaviour
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(litellm, "completion", fake)
    return calls


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_key_is_stable_and_order_independent() -> None:
    messages = [{"role": "user", "content": "hi"}]
    a = DiskCache.key("m", messages, {"temperature": 0.0, "max_tokens": 10})
    b = DiskCache.key("m", messages, {"max_tokens": 10, "temperature": 0.0})
    assert a == b and len(a) == 64


@pytest.mark.parametrize(
    ("model", "messages", "params"),
    [
        ("other-model", [{"role": "user", "content": "hi"}], {"temperature": 0.0}),
        ("m", [{"role": "user", "content": "different"}], {"temperature": 0.0}),
        ("m", [{"role": "user", "content": "hi"}], {"temperature": 0.7}),
    ],
)
def test_cache_key_changes_with_every_input(
    model: str, messages: list[dict[str, str]], params: dict[str, Any]
) -> None:
    baseline = DiskCache.key("m", [{"role": "user", "content": "hi"}], {"temperature": 0.0})
    assert DiskCache.key(model, messages, params) != baseline


def test_second_identical_call_is_served_from_cache(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free-tier quota is the scarce resource; a re-run must cost zero calls."""
    calls = patch_completion(monkeypatch, FakeCompletion("the answer"))
    messages = [{"role": "user", "content": "question"}]

    first = client.complete(messages, model="groq/test")
    second = client.complete(messages, model="groq/test")

    assert len(calls) == 1, "the second call must not reach the provider"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text
    assert second.prompt_tokens == first.prompt_tokens


def test_cache_can_be_bypassed(client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_completion(monkeypatch, FakeCompletion("x"))
    messages = [{"role": "user", "content": "q"}]
    client.complete(messages, model="groq/test")
    client.complete(messages, model="groq/test", use_cache=False)
    assert len(calls) == 2


def test_a_corrupt_cache_entry_is_a_miss_not_a_failure(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c")
    cache.put("abc", {"text": "ok"})
    (tmp_path / "c" / "abc.json").write_text("{not json", encoding="utf-8")
    assert cache.get("abc") is None


def test_an_unwritable_cache_degrades_to_no_cache(tmp_path: Path) -> None:
    """An unwritable cache must never turn into a request failure."""
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    cache = DiskCache(blocker)
    cache.put("key", {"text": "value"})  # must not raise
    assert cache.get("key") is None


# ---------------------------------------------------------------------------
# Provider parameters
# ---------------------------------------------------------------------------
def test_groq_calls_pin_reasoning_effort_low_and_disable_reasoning_output(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gpt-oss-120b emits reasoning as output tokens; defaults multiply spend
    against a hard 200K/day cap."""
    calls = patch_completion(monkeypatch, FakeCompletion("x"))
    client.complete([{"role": "user", "content": "q"}], model="groq/openai/gpt-oss-120b")
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["include_reasoning"] is False


def test_non_groq_calls_do_not_carry_groq_specific_parameters(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = patch_completion(monkeypatch, FakeCompletion("x"))
    client.complete([{"role": "user", "content": "q"}], model="gemini/gemini-3.5-flash")
    assert "reasoning_effort" not in calls[0]
    assert "include_reasoning" not in calls[0]


def test_temperature_and_max_tokens_are_forwarded(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = patch_completion(monkeypatch, FakeCompletion("x"))
    client.complete([{"role": "user", "content": "q"}], model="m", temperature=0.7, max_tokens=256)
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["max_tokens"] == 256


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "429 rate limit exceeded",
        "Request timed out",
        "503 Service Unavailable",
        "connection reset",
        "model is overloaded",
    ],
)
def test_transient_errors_are_classified_for_retry(message: str) -> None:
    assert _is_transient(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "401 invalid api key",
        "model not found",
        "invalid request: bad schema",
        # A permanent error whose text merely contains a retryable code as a
        # substring. Naive matching retried these three times before failing.
        "input exceeds the maximum of 1500 characters",
        "temperature must be between 0.500 and 2.0",
    ],
)
def test_permanent_errors_are_not_retried(message: str) -> None:
    assert _is_transient(RuntimeError(message)) is False


def test_status_code_attribute_wins_over_message_text() -> None:
    """A typed provider error is authoritative; its prose is not."""

    class ProviderError(RuntimeError):
        status_code = 401

    # Message looks retryable, the status says otherwise.
    assert _is_transient(ProviderError("gateway timeout 503")) is False

    class ThrottledError(RuntimeError):
        status_code = 429

    assert _is_transient(ThrottledError("slow down")) is True


def test_transient_failure_is_retried_then_succeeds(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)  # no real backoff in tests

    def behaviour(attempt: int) -> Any:
        return RuntimeError("429 rate limit") if attempt == 1 else FakeCompletion("recovered")

    calls = patch_completion(monkeypatch, behaviour)
    result = client.complete([{"role": "user", "content": "q"}], model="m")
    assert len(calls) == 2
    assert result.text == "recovered"
    assert result.attempts == 2


def test_permanent_failure_fails_fast_without_burning_retries(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = patch_completion(monkeypatch, RuntimeError("401 invalid api key"))
    with pytest.raises(LLMError, match="401"):
        client.complete([{"role": "user", "content": "q"}], model="m")
    assert len(calls) == 1


def test_exhausted_retries_raise_rather_than_fabricate_an_answer(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = patch_completion(monkeypatch, RuntimeError("503 unavailable"))

    # Read the budget rather than hardcoding it. The literal 3 here silently
    # encoded llm_max_retries=2, so raising the default to 5 broke this test in
    # CI while it kept passing locally against a `.env` that still said 2.
    expected_attempts = get_settings().llm_max_retries + 1

    with pytest.raises(LLMError, match="failed after"):
        client.complete([{"role": "user", "content": "q"}], model="m")
    assert len(calls) == expected_attempts, "one initial attempt plus llm_max_retries"


def test_a_failed_call_is_never_cached(client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    patch_completion(monkeypatch, RuntimeError("503 unavailable"))
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "q"}], model="m")

    calls = patch_completion(monkeypatch, FakeCompletion("now working"))
    assert client.complete([{"role": "user", "content": "q"}], model="m").text == "now working"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
def test_usage_is_read_off_the_response_not_from_litellms_tracker(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """litellm's model table lags both providers, so its accounting is unreliable."""
    patch_completion(monkeypatch, FakeCompletion("x", prompt=777, completion=333))
    result = client.complete([{"role": "user", "content": "q"}], model="m")
    assert result.prompt_tokens == 777
    assert result.completion_tokens == 333
    assert result.total_tokens == 1110


def test_a_response_without_usage_metadata_does_not_crash(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoUsage:
        choices = [FakeChoice("text")]

    patch_completion(monkeypatch, NoUsage())
    result = client.complete([{"role": "user", "content": "q"}], model="m")
    assert result.total_tokens == 0


def test_estimated_cost_uses_list_prices_for_a_known_model() -> None:
    """The run is $0; this reports what it would have cost on a paid tier."""
    response = LLMResponse(
        text="x",
        model="groq/openai/gpt-oss-120b",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert response.estimated_cost_usd == pytest.approx(0.15 + 0.75)


def test_estimated_cost_of_an_unknown_model_is_zero_not_an_error() -> None:
    assert LLMResponse(text="x", model="unknown/model", prompt_tokens=999).estimated_cost_usd == 0.0


def test_cached_response_round_trips_through_json(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c")
    payload = {"text": "hello", "prompt_tokens": 5, "completion_tokens": 2, "attempts": 1}
    cache.put("k", payload)
    assert cache.get("k") == json.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# Token pacing
#
# Tokens-per-minute, not requests-per-minute, is what actually binds this
# workload. The limiter shipped untested, which is how it reached CI still
# pacing calls to a *mocked* provider -- no quota to protect, ~2 minutes of
# pure sleeping per run. conftest disables it suite-wide; these tests opt back
# in against a fake clock, so they assert the real waits without spending them.
# ---------------------------------------------------------------------------
PACED_MODEL = "test/paced-model"  # deliberately absent from tokens_per_minute_by_model
PACED_CEILING = 1000
PACED_BUDGET = int(PACED_CEILING * TOKEN_BUDGET_SAFETY_FACTOR)  # 750


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("time.monotonic", fake.monotonic)
    monkeypatch.setattr("time.sleep", fake.sleep)
    return fake


@pytest.fixture
def paced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-enable pacing for one test; conftest turns it off for the suite."""
    monkeypatch.setattr(get_settings(), "default_tokens_per_minute", PACED_CEILING)


def test_pacing_is_off_when_no_ceiling_is_configured(
    client: LLMClient, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with no published TPM limit must not be paced at all."""
    monkeypatch.setattr(get_settings(), "default_tokens_per_minute", 0)
    assert client._throttle(PACED_MODEL, 10_000) == 0.0
    assert clock.slept == []


def test_calls_within_the_budget_never_wait(
    client: LLMClient, clock: FakeClock, paced: None
) -> None:
    assert client._throttle(PACED_MODEL, 300) == 0.0
    assert client._throttle(PACED_MODEL, 300) == 0.0  # 600 of 750
    assert clock.slept == []


def test_only_the_safety_factor_of_the_ceiling_is_spendable(
    client: LLMClient, clock: FakeClock, paced: None
) -> None:
    """The 4-chars-per-token estimate can undershoot and the provider's own
    window is already partly spent when this process starts, so the limiter
    must hold back a margin rather than aim at the stated ceiling."""
    client._throttle(PACED_MODEL, 400)
    client._throttle(PACED_MODEL, 400)  # 800: under the ceiling, over the budget

    assert clock.slept, f"800 tokens fit under {PACED_CEILING} but must exceed {PACED_BUDGET}"


def test_exceeding_the_budget_waits_for_the_window_to_slide(
    client: LLMClient, clock: FakeClock, paced: None
) -> None:
    client._throttle(PACED_MODEL, 700)
    waited = client._throttle(PACED_MODEL, 700)

    assert waited > 0.0, "the second call must be paced"
    assert waited <= 60.0, "never wait longer than the window it is waiting on"
    assert clock.now == pytest.approx(1000.0 + waited)


def test_the_window_frees_up_once_a_minute_has_passed(
    client: LLMClient, clock: FakeClock, paced: None
) -> None:
    client._throttle(PACED_MODEL, PACED_BUDGET)
    clock.now += 60.0  # a full window elapses with no calls
    assert client._throttle(PACED_MODEL, PACED_BUDGET) == 0.0
    assert clock.slept == [], "an expired window must not cost a wait"


def test_a_request_larger_than_the_whole_budget_still_proceeds(
    client: LLMClient, clock: FakeClock, paced: None
) -> None:
    """Clamped to the budget, and admitted against an empty window. Without
    both, an over-large estimate could never satisfy the check and the caller
    would wait forever."""
    assert client._throttle(PACED_MODEL, PACED_BUDGET * 100) == 0.0
    assert clock.slept == []


def test_each_model_is_paced_independently(
    client: LLMClient, clock: FakeClock, paced: None
) -> None:
    """Limits are per-model at the provider, so one model's spend must not
    throttle another's."""
    client._throttle(PACED_MODEL, PACED_BUDGET)
    assert client._throttle("test/other-model", PACED_BUDGET) == 0.0
    assert clock.slept == []


@pytest.mark.parametrize(
    ("messages", "max_tokens", "expected"),
    [
        ([], 100, 100),  # completion budget alone
        ([{"role": "user", "content": "x" * 400}], 100, 200),  # 400 chars -> 100 tokens
        ([{"role": "user", "content": None}], 50, 50),  # a null content must not raise
    ],
)
def test_token_estimate_counts_prompt_and_completion(
    messages: list[dict[str, Any]], max_tokens: int, expected: int
) -> None:
    assert _estimate_tokens(messages, max_tokens) == expected
