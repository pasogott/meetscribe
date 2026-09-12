"""A missing `tinfoil` SDK must degrade readably, not explode.

Until 0.20.1 the SDK lived in the optional `[tee]` extra while
`DEFAULT_SUMMARY_BACKEND` was already `tinfoil`, so a plain
`pip install millet-pipeline` could not summarize on its own default path.

Worse, availability was decided by "is an API key resolvable?" alone.  With a
key set but no SDK the backend reported itself *available*, so the run walked
past the guard and into a bare `from tinfoil import TinfoilAI` — surfacing as
a ModuleNotFoundError traceback mid-job, and (with a preset requested) with no
fallback at all.

The SDK is now a base dependency, and availability additionally checks that it
imports.  These tests pin the degraded behaviour for the environments that can
still lack it: constraint files, partial upgrades, pre-0.20.1 installs.
"""
from __future__ import annotations

import pytest

from millet import summarize as sm
from millet.summarize import MeetingSummary, SummaryConfig, summarize


@pytest.fixture
def no_sdk(monkeypatch):
    """Simulate an environment where `import tinfoil` fails."""
    monkeypatch.setattr(sm, "tinfoil_sdk_installed", lambda: False)


@pytest.fixture
def with_key(monkeypatch):
    """A resolvable API key — the case that used to mask the missing SDK."""
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk-test")


class TestAvailability:
    def test_key_without_sdk_is_not_available(self, no_sdk, with_key):
        """The regression: a key alone used to be enough."""
        assert not sm.is_backend_available(SummaryConfig(backend="tinfoil"))

    def test_sdk_without_key_is_not_available(self, monkeypatch):
        monkeypatch.setattr(sm, "tinfoil_sdk_installed", lambda: True)
        monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: None)
        assert not sm.is_backend_available(SummaryConfig(backend="tinfoil"))

    def test_both_present_is_available(self, monkeypatch, with_key):
        monkeypatch.setattr(sm, "tinfoil_sdk_installed", lambda: True)
        assert sm.is_backend_available(SummaryConfig(backend="tinfoil"))

    def test_real_probe_agrees_with_import(self):
        """The helper must reflect reality, not a constant."""
        try:
            import tinfoil  # noqa: F401
        except ImportError:
            assert not sm.tinfoil_sdk_installed()
        else:
            assert sm.tinfoil_sdk_installed()


class TestMessage:
    def test_missing_sdk_message_names_the_fix(self, no_sdk, with_key):
        msg = sm._backend_not_available_message(SummaryConfig(backend="tinfoil"))
        assert "tinfoil" in msg and "not installed" in msg
        assert "pip install" in msg
        # The local escape hatch stays discoverable.
        assert "ollama" in msg

    def test_missing_key_message_is_unchanged(self, monkeypatch):
        """A present SDK with no key must still talk about the key."""
        monkeypatch.setattr(sm, "tinfoil_sdk_installed", lambda: True)
        monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: None)
        msg = sm._backend_not_available_message(SummaryConfig(backend="tinfoil"))
        assert "TINFOIL_API_KEY" in msg
        assert "not installed" not in msg


class TestEndToEnd:
    def test_falls_back_to_ollama_when_sdk_missing(self, no_sdk, with_key, monkeypatch):
        """No preset requested: the private local backend takes over."""
        monkeypatch.setattr(sm, "is_ollama_available", lambda url: True)
        monkeypatch.setattr(
            sm, "_dispatch",
            lambda backend, s, u, cfg, **kw: MeetingSummary(
                markdown="# Summary\n\nLong enough to be a real body.",
                model="ollama-model", elapsed_seconds=1.0, backend=backend,
            ),
        )
        result = summarize("transcript text", SummaryConfig())
        assert result.backend == "ollama"
        assert result.fallback_used is True

    def test_preset_fails_with_readable_message_not_importerror(
        self, no_sdk, with_key, monkeypatch
    ):
        """The worst old path: preset + key + no SDK raised ModuleNotFoundError
        from deep inside the backend.  It must now fail at the guard."""
        monkeypatch.setattr(sm, "is_ollama_available", lambda url: True)

        def exploding_dispatch(*a, **kw):  # pragma: no cover - must never run
            raise AssertionError("dispatch reached despite unavailable backend")

        monkeypatch.setattr(sm, "_dispatch", exploding_dispatch)
        with pytest.raises(RuntimeError) as exc:
            summarize("transcript text", SummaryConfig(preset="confidential"))
        assert "not installed" in str(exc.value)
        assert not isinstance(exc.value, ImportError)

    def test_no_sdk_no_ollama_names_both_causes(self, no_sdk, with_key, monkeypatch):
        """Nothing usable: every backend is skipped, so nothing is ever
        dispatched and `last_error` stays None.  The error used to read
        "All summary backends failed. Last error: None" — true, and useless.
        It must name why each backend was skipped."""
        monkeypatch.setattr(sm, "is_ollama_available", lambda url: False)
        with pytest.raises(RuntimeError) as exc:
            summarize("transcript text", SummaryConfig())
        msg = str(exc.value)
        assert "Last error: None" not in msg
        assert "not installed" in msg      # the tinfoil SDK
        assert "ollama" in msg.lower()     # and the local alternative
        assert "No module named" not in msg
