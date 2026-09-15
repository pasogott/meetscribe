"""Tests for backend resolution, the fallback chain, and preset aliasing.

0.19.0 removed the non-private cloud backends (claudemax, openrouter,
openai).  What used to be a privacy/quality tradeoff across five backends is
now two private ones -- a hardware-attested TEE and local Ollama -- so:

  - the preset axis is retired; the three historical names are kept as
    aliases for the default so existing callers (vezir passes
    --summary-preset on every job) and ~580 stored jobs keep working;
  - MILLET_SUMMARY_PRESET_FALLBACK is gone, because the cloud backends it
    existed to reach are gone.  An explicitly requested preset now always
    fails loud rather than silently producing something else;
  - stale environment config naming a removed backend must degrade with a
    warning, not crash every job on a deployment that upgrades.
"""
from __future__ import annotations

import json

import pytest

from millet import summarize as sm
from millet.summarize import (
    BACKENDS,
    DEFAULT_FALLBACK_ORDER,
    DEFAULT_SUMMARY_BACKEND,
    DEFAULT_TINFOIL_MODEL,
    RETIRED_BACKENDS,
    SUMMARY_PRESETS,
    MeetingSummary,
    SummaryConfig,
    _default_model_for_backend,
    _resolve_backend,
    _resolve_fallback_order,
    _resolve_model,
    summarize,
)

# ─── Backend registry ──────────────────────────────────────────────────────


class TestBackendRegistry:
    def test_only_private_backends_remain(self):
        # Every backend is private: tinfoil/venice/near are attested TEEs,
        # ollama is local.  No plaintext cloud backend may reappear.
        assert set(BACKENDS) == {"tinfoil", "ollama", "venice", "near"}
        assert not (set(BACKENDS) & set(RETIRED_BACKENDS))

    def test_retired_backends_rejected_when_explicit(self):
        """An explicit backend= is a caller bug and must fail loudly."""
        for name in RETIRED_BACKENDS:
            with pytest.raises(ValueError, match="Unknown summary backend"):
                SummaryConfig(backend=name)

    def test_default_backend_is_the_tee(self):
        assert DEFAULT_SUMMARY_BACKEND == "tinfoil"


# ─── Stale environment config (upgrade path) ───────────────────────────────


class TestRetiredBackendEnvDowngrade:
    @pytest.mark.parametrize("retired", RETIRED_BACKENDS)
    def test_env_naming_retired_backend_falls_back_to_default(
        self, monkeypatch, retired, caplog
    ):
        """Deployments carry MEETSCRIBE_SUMMARY_BACKEND=claudemax in systemd
        env files that outlive an upgrade.  Hard-failing there would turn a
        version bump into an outage."""
        monkeypatch.setattr(sm, "_WARNED_BACKENDS", set())
        monkeypatch.setenv("MILLET_SUMMARY_BACKEND", retired)
        with caplog.at_level("WARNING"):
            assert _resolve_backend() == DEFAULT_SUMMARY_BACKEND
        assert retired in caplog.text
        assert "removed in 0.19.0" in caplog.text

    def test_legacy_env_alias_also_downgraded(self, monkeypatch):
        monkeypatch.setattr(sm, "_WARNED_BACKENDS", set())
        monkeypatch.delenv("MILLET_SUMMARY_BACKEND", raising=False)
        monkeypatch.setenv("MEETSCRIBE_SUMMARY_BACKEND", "claudemax")
        assert SummaryConfig().backend == DEFAULT_SUMMARY_BACKEND

    def test_warning_emitted_once_per_name(self, monkeypatch, caplog):
        monkeypatch.setattr(sm, "_WARNED_BACKENDS", set())
        monkeypatch.setenv("MILLET_SUMMARY_BACKEND", "claudemax")
        with caplog.at_level("WARNING"):
            _resolve_backend()
            _resolve_backend()
        assert caplog.text.count("removed in 0.19.0") == 1

    def test_valid_backend_env_untouched(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_BACKEND", "ollama")
        assert _resolve_backend() == "ollama"


class TestStaleOllamaModelGuard:
    """Before 0.19.0 the default backend was ollama, so a bare
    MILLET_SUMMARY_MODEL names an Ollama tag.  Sending that to the enclave
    would 404 at request time."""

    def test_ollama_tag_ignored_for_tee(self, monkeypatch, caplog):
        monkeypatch.setattr(sm, "_WARNED_MODELS", set())
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "qwen3.8:27b")
        with caplog.at_level("WARNING"):
            assert _resolve_model("tinfoil") == DEFAULT_TINFOIL_MODEL
        assert "looks like an Ollama tag" in caplog.text

    def test_ollama_tag_honored_for_ollama(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "qwen3.8:27b")
        assert _resolve_model("ollama") == "qwen3.8:27b"

    def test_plain_model_name_still_honored_for_tee(self, monkeypatch):
        monkeypatch.setattr(sm, "_WARNED_MODELS", set())
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "deepseek-v4-1-flash")
        assert _resolve_model("tinfoil") == "deepseek-v4-1-flash"

    def test_env_model_does_not_leak_across_backends(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "some-ollama-only:9b")
        assert _default_model_for_backend("tinfoil") == DEFAULT_TINFOIL_MODEL


# ─── Fallback chain ────────────────────────────────────────────────────────


class TestResolveFallbackOrder:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MILLET_SUMMARY_FALLBACK_ORDER", raising=False)
        order = _resolve_fallback_order()
        assert order == DEFAULT_FALLBACK_ORDER
        assert order == ("tinfoil", "venice", "near", "ollama")

    def test_every_destination_is_private(self):
        """The chain must not be able to downgrade confidentiality.

        tinfoil/venice/near are attested TEEs, ollama is local — all private.
        """
        assert all(
            b in ("tinfoil", "venice", "near", "ollama")
            for b in DEFAULT_FALLBACK_ORDER
        )

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "ollama,tinfoil")
        assert _resolve_fallback_order() == ("ollama", "tinfoil")

    def test_retired_names_dropped_from_override(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai,claudemax,ollama")
        assert _resolve_fallback_order() == ("ollama",)

    def test_all_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai,bogus")
        assert _resolve_fallback_order() == DEFAULT_FALLBACK_ORDER


# ─── Preset aliasing ───────────────────────────────────────────────────────


class TestPresetAliasing:
    @pytest.mark.parametrize("preset", ["high-quality", "confidential", "alternative"])
    def test_legacy_presets_still_accepted(self, preset):
        """vezir passes these on every job; rejecting them breaks production."""
        cfg = SummaryConfig(preset=preset)
        assert cfg.backend == DEFAULT_SUMMARY_BACKEND
        assert cfg.model == DEFAULT_TINFOIL_MODEL

    def test_all_presets_resolve_identically(self):
        configs = {(v["backend"], v["model"]) for v in SUMMARY_PRESETS.values()}
        assert len(configs) == 1

    def test_deprecated_preset_logs_once(self, monkeypatch, caplog):
        monkeypatch.setattr(sm, "_WARNED_PRESETS", set())
        with caplog.at_level("INFO"):
            sm._warn_deprecated_preset("high-quality")
            sm._warn_deprecated_preset("high-quality")
        assert caplog.text.count("is deprecated") == 1


# ─── Preset never falls back ───────────────────────────────────────────────


def _fake_summary(backend: str) -> MeetingSummary:
    return MeetingSummary(
        markdown="# Summary\n\nReal content, definitely long enough.",
        model=f"{backend}-model",
        elapsed_seconds=1.0,
        backend=backend,
    )


def _patch_backends(monkeypatch, available: set[str], failing: set[str]):
    """Stub availability checks and dispatch for the given backends."""
    monkeypatch.setattr(sm, "is_backend_available", lambda cfg: cfg.backend in available)

    def fake_dispatch(backend, system_prompt, user_prompt, config, **kwargs):
        if backend in failing:
            raise RuntimeError(f"{backend} upstream quota exhausted")
        return _fake_summary(backend)

    monkeypatch.setattr(sm, "_dispatch", fake_dispatch)


class TestPresetNeverFallsBack:
    @pytest.mark.parametrize("preset", ["high-quality", "confidential", "alternative"])
    def test_requested_preset_fails_loud(self, monkeypatch, preset):
        """No preset falls back, and the opt-in env var is gone -- setting it
        must not resurrect the behaviour."""
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        _patch_backends(monkeypatch, available={"tinfoil", "ollama"}, failing={"tinfoil"})
        with pytest.raises(RuntimeError, match="quota exhausted"):
            summarize("transcript text", SummaryConfig(preset=preset))

    def test_unavailable_primary_raises_with_actionable_message(self, monkeypatch):
        _patch_backends(monkeypatch, available={"ollama"}, failing=set())
        with pytest.raises(RuntimeError, match="requires the 'tinfoil' backend"):
            summarize("transcript text", SummaryConfig(preset="confidential"))

    def test_primary_success_tags_result(self, monkeypatch):
        _patch_backends(monkeypatch, available={"tinfoil"}, failing=set())
        result = summarize("transcript text", SummaryConfig(preset="confidential"))
        assert result.backend == "tinfoil"
        assert result.preset == "confidential"
        assert result.fallback_used is False


class TestUnpresettedFallback:
    """Without a preset the chain still degrades -- but only between private
    backends, so confidentiality is preserved either way."""

    def test_falls_back_to_ollama_when_tee_unavailable(self, monkeypatch):
        _patch_backends(monkeypatch, available={"ollama"}, failing=set())
        result = summarize("transcript text", SummaryConfig(backend="tinfoil"))
        assert result.backend == "ollama"
        assert result.fallback_used is True

    def test_all_backends_failing_raises(self, monkeypatch):
        _patch_backends(
            monkeypatch,
            available={"tinfoil", "ollama"},
            failing={"tinfoil", "ollama"},
        )
        with pytest.raises(RuntimeError, match="All summary backends failed"):
            summarize("transcript text", SummaryConfig(backend="tinfoil"))


# ─── Meta sidecar provenance ───────────────────────────────────────────────


class TestMetaSidecarProvenance:
    def test_meta_records_preset_and_fallback(self, tmp_path):
        summary = _fake_summary("tinfoil")
        summary.preset = "confidential"
        summary.fallback_used = True
        summary.save(tmp_path, "session1")
        meta = json.loads((tmp_path / "session1.summary.meta.json").read_text(encoding="utf-8"))
        assert meta["backend"] == "tinfoil"
        assert meta["preset"] == "confidential"
        assert meta["fallback_used"] is True

    def test_meta_defaults_no_preset_no_fallback(self, tmp_path):
        summary = _fake_summary("ollama")
        summary.save(tmp_path, "session2")
        meta = json.loads((tmp_path / "session2.summary.meta.json").read_text(encoding="utf-8"))
        assert meta["preset"] is None
        assert meta["fallback_used"] is False
