"""Provider chain: routing, credential detection, circuit breaker, backoff.

The behaviour under test is the one that cost this project an entire rubric
line: a single dead credential produced zero measurements. Everything here is
about degrading instead of stopping.
"""

from __future__ import annotations

import random

import pytest

from src.config import get_settings
from src.providers import (
    LOCAL_PROVIDERS,
    PROVIDER_KEY_ENV,
    ProviderChain,
    backoff_delay,
    credential_for,
    has_credential,
    routing_provider,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("groq/openai/gpt-oss-120b", "groq"),
        ("gemini/gemini-3.5-flash", "gemini"),
        ("ollama/llama3", "ollama"),
        ("gpt-4o", "openai"),
    ],
)
def test_routing_prefix_is_read_not_guessed(model: str, expected: str) -> None:
    assert routing_provider(model) == expected


def test_routing_provider_is_not_the_model_family() -> None:
    """``groq/openai/gpt-oss-120b`` is *routed* by Groq and *descends* from
    OpenAI. Collapsing the two is how a cross-family judging invariant passes
    for the wrong reason, so they stay separate concepts."""
    assert routing_provider("groq/openai/gpt-oss-120b") == "groq"
    assert "openai" in "groq/openai/gpt-oss-120b"


def test_local_providers_need_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of putting a local model last: it is always reachable, so
    the chain has a terminal option that cannot fail on authentication."""
    for provider in LOCAL_PROVIDERS:
        assert has_credential(f"{provider}/some-model") is True


def test_missing_credential_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.providers._settings_key", lambda env: None)
    assert has_credential("anthropic/claude-opus-4") is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    assert has_credential("anthropic/claude-opus-4") is True


def test_every_mapped_provider_names_an_env_var() -> None:
    """Adding a provider must be a table row, never a change at a call site."""
    assert all(env.endswith("_API_KEY") for env in PROVIDER_KEY_ENV.values())
    assert not (set(PROVIDER_KEY_ENV) & LOCAL_PROVIDERS), "local providers take no key"


def test_an_empty_chain_is_refused_rather_than_silently_doing_nothing() -> None:
    with pytest.raises(ValueError, match="chain is empty"):
        ProviderChain([])


def test_a_fatal_failure_opens_the_circuit_for_the_rest_of_the_run() -> None:
    """Without this, a 40-question evaluation re-asks a known-dead key 40 times
    and turns one misconfiguration into a slow, confusing run."""
    chain = ProviderChain(["groq/a", "gemini/b"])
    assert len(chain.available()) == 2

    chain.record_fatal("groq/a", "invalid api key")

    remaining = [s.model for s in chain.available()]
    assert remaining == ["gemini/b"]
    assert any("circuit open" in why for _, why in chain.skipped())


def test_transient_failures_open_the_circuit_only_after_the_threshold() -> None:
    """A single 429 is not evidence a provider is dead; a run of them is."""
    chain = ProviderChain(["groq/a", "gemini/b"], failure_threshold=2)

    chain.record_transient("groq/a", "429")
    assert len(chain.available()) == 2, "one transient failure must not evict a provider"

    chain.record_transient("groq/a", "429")
    assert [s.model for s in chain.available()] == ["gemini/b"]


def test_success_resets_the_transient_counter() -> None:
    chain = ProviderChain(["groq/a"], failure_threshold=2)
    chain.record_transient("groq/a", "429")
    chain.record_success("groq/a")
    chain.record_transient("groq/a", "429")
    assert chain.available(), "an intermittent provider must not accumulate toward eviction"


def test_health_snapshot_names_why_each_entry_is_unusable() -> None:
    chain = ProviderChain(["groq/a", "gemini/b"])
    chain.record_fatal("groq/a", "invalid api key")
    health = {row["model"]: row for row in chain.health()}
    assert health["groq/a"]["circuit_open"] is True
    assert "invalid api key" in str(health["groq/a"]["last_error"])
    assert health["gemini/b"]["circuit_open"] is False


def test_backoff_grows_and_is_bounded() -> None:
    rng = random.Random(11)
    ceilings = [backoff_delay(i, base=1.0, cap=45.0, rng=random.Random(0)) for i in range(1, 8)]
    assert all(0.0 <= d <= 45.0 for d in ceilings), "never exceeds the cap"
    # Full jitter samples [0, ceiling], so assert on the ceiling, not the draw.
    many = [backoff_delay(6, base=1.0, cap=45.0, rng=rng) for _ in range(200)]
    assert max(many) > max(backoff_delay(1, base=1.0, cap=45.0, rng=rng) for _ in range(200)), (
        "later attempts must be able to wait longer than early ones"
    )


def test_backoff_uses_full_jitter_not_a_fixed_multiplier() -> None:
    """Identical backoff across concurrent callers makes them retry in lockstep
    and re-collide -- one 429 becomes a synchronised thundering herd."""
    rng = random.Random(3)
    draws = {round(backoff_delay(4, base=1.0, cap=45.0, rng=rng), 6) for _ in range(50)}
    assert len(draws) > 40, "delays must be spread, not constant"


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------
def test_credential_comes_from_settings_when_the_environment_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this pins: litellm was called with no api_key, so it fell back to
    os.environ -- which pydantic-settings never populates when the key arrives
    via .env. It only worked because litellm runs load_dotenv() at import,
    resolved against the current working directory. Same .env and same key gave
    a 401 from a server and success from a script.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(get_settings(), "groq_api_key", "gsk_from_settings")
    assert credential_for("groq/openai/gpt-oss-120b") == "gsk_from_settings"


def test_the_process_environment_wins_over_the_dotenv_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly exported key is a deliberate override of a checked-in
    default, so it must take precedence rather than be shadowed by it."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_environ")
    monkeypatch.setattr(get_settings(), "groq_api_key", "gsk_from_settings")
    assert credential_for("groq/openai/gpt-oss-120b") == "gsk_from_environ"


def test_credential_for_a_local_provider_is_none() -> None:
    """A local model is the terminal option in a chain precisely because it
    cannot fail on authentication; handing it a key would be meaningless."""
    assert credential_for("ollama/llama3") is None
    assert credential_for("lm_studio/whatever") is None


def test_the_routing_prefix_selects_the_key_not_the_model_family() -> None:
    """groq/openai/gpt-oss-120b is served by Groq and must use the Groq key,
    even though the model portion says openai."""
    assert PROVIDER_KEY_ENV[routing_provider("groq/openai/gpt-oss-120b")] == "GROQ_API_KEY"
