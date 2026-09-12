"""Tests for opt-in preset fallback (MILLET_SUMMARY_PRESET_FALLBACK),
the MILLET_SUMMARY_FALLBACK_ORDER chain override, per-backend
MILLET_OPENAI_MODEL resolution, and fallback provenance in the
.summary.meta.json sidecar.
"""

from __future__ import annotations

import json

import pytest

from millet import summarize as sm
from millet.summarize import (
    DEFAULT_FALLBACK_ORDER,
    DEFAULT_OPENAI_COMPAT_MODEL,
    MeetingSummary,
    SummaryConfig,
    _default_model_for_backend,
    _effective_temperature,
    _preset_fallback_allowed,
    _resolve_fallback_order,
    summarize,
)

# ─── Fallback order resolution ─────────────────────────────────────────────


class TestResolveFallbackOrder:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MILLET_SUMMARY_FALLBACK_ORDER", raising=False)
        assert _resolve_fallback_order() == DEFAULT_FALLBACK_ORDER
        assert "openai" not in _resolve_fallback_order()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai,ollama")
        assert _resolve_fallback_order() == ("openai", "ollama")

    def test_unknown_and_duplicates_dropped(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai, bogus ,OPENAI,ollama")
        assert _resolve_fallback_order() == ("openai", "ollama")

    def test_all_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "bogus,nope")
        assert _resolve_fallback_order() == DEFAULT_FALLBACK_ORDER


# ─── Per-backend openai model resolution ───────────────────────────────────


class TestOpenaiModelResolution:
    def test_openai_model_env_honored(self, monkeypatch):
        monkeypatch.setenv("MILLET_OPENAI_MODEL", "kimi-k3")
        assert _default_model_for_backend("openai") == "kimi-k3"

    def test_openai_model_env_default(self, monkeypatch):
        monkeypatch.delenv("MILLET_OPENAI_MODEL", raising=False)
        assert _default_model_for_backend("openai") == DEFAULT_OPENAI_COMPAT_MODEL

    def test_summary_model_env_still_ignored_for_fallback(self, monkeypatch):
        # The chain-wide MILLET_SUMMARY_MODEL must not leak into a fallback
        # backend; only the per-backend MILLET_OPENAI_MODEL applies.
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "some-ollama-only:9b")
        monkeypatch.delenv("MILLET_OPENAI_MODEL", raising=False)
        assert _default_model_for_backend("openai") == DEFAULT_OPENAI_COMPAT_MODEL


# ─── Preset fallback gate ──────────────────────────────────────────────────


class TestPresetFallbackAllowed:
    @pytest.mark.parametrize("val", ["1", "true", "True", "YES", "on"])
    def test_truthy_enables_non_confidential(self, monkeypatch, val):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", val)
        assert _preset_fallback_allowed("high-quality") is True
        assert _preset_fallback_allowed("alternative") is True

    def test_confidential_never_falls_back(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        assert _preset_fallback_allowed("confidential") is False

    def test_unset_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("MILLET_SUMMARY_PRESET_FALLBACK", raising=False)
        assert _preset_fallback_allowed("high-quality") is False

    def test_unknown_or_missing_preset(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        assert _preset_fallback_allowed(None) is False
        assert _preset_fallback_allowed("bogus") is False


# ─── summarize() dispatch behavior ─────────────────────────────────────────


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


class TestSummarizePresetFallback:
    def test_fallback_fires_when_enabled(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai")
        _patch_backends(monkeypatch, available={"claudemax", "openai"}, failing={"claudemax"})
        cfg = SummaryConfig(preset="high-quality")
        result = summarize("transcript text", cfg)
        assert result.backend == "openai"
        assert result.preset == "high-quality"
        assert result.fallback_used is True

    def test_no_fallback_by_default(self, monkeypatch):
        monkeypatch.delenv("MILLET_SUMMARY_PRESET_FALLBACK", raising=False)
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai")
        _patch_backends(monkeypatch, available={"claudemax", "openai"}, failing={"claudemax"})
        cfg = SummaryConfig(preset="high-quality")
        with pytest.raises(RuntimeError, match="quota exhausted"):
            summarize("transcript text", cfg)

    def test_confidential_never_falls_back(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai")
        _patch_backends(monkeypatch, available={"tinfoil", "openai"}, failing={"tinfoil"})
        cfg = SummaryConfig(preset="confidential")
        with pytest.raises(RuntimeError, match="quota exhausted"):
            summarize("transcript text", cfg)

    def test_fallback_when_primary_unavailable(self, monkeypatch):
        # Health check fails (e.g. proxy down): with the opt-in, the chain
        # proceeds instead of raising the preset-unavailable error.
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai")
        _patch_backends(monkeypatch, available={"openai"}, failing=set())
        cfg = SummaryConfig(preset="high-quality")
        result = summarize("transcript text", cfg)
        assert result.backend == "openai"
        assert result.fallback_used is True

    def test_primary_unavailable_raises_without_opt_in(self, monkeypatch):
        monkeypatch.delenv("MILLET_SUMMARY_PRESET_FALLBACK", raising=False)
        _patch_backends(monkeypatch, available={"openai"}, failing=set())
        cfg = SummaryConfig(preset="high-quality")
        with pytest.raises(RuntimeError, match="requires the 'claudemax' backend"):
            summarize("transcript text", cfg)

    def test_all_backends_failing_raises(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "openai")
        _patch_backends(
            monkeypatch,
            available={"claudemax", "openai"},
            failing={"claudemax", "openai"},
        )
        cfg = SummaryConfig(preset="high-quality")
        with pytest.raises(RuntimeError, match="All summary backends failed"):
            summarize("transcript text", cfg)

    def test_primary_success_tags_result(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_PRESET_FALLBACK", "1")
        _patch_backends(monkeypatch, available={"claudemax"}, failing=set())
        cfg = SummaryConfig(preset="high-quality")
        result = summarize("transcript text", cfg)
        assert result.backend == "claudemax"
        assert result.preset == "high-quality"
        assert result.fallback_used is False


# ─── Kimi K-series temperature clamp ───────────────────────────────────────


class TestEffectiveTemperature:
    @pytest.mark.parametrize("model", ["kimi-k3", "kimi-k2.6", "kimi-for-coding", "Kimi-K3"])
    def test_kimi_kseries_forced_to_1(self, model):
        assert _effective_temperature(model, 0.3) == 1.0

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "moonshot-v1-8k", "k3", "glm-5-3-flash"])
    def test_other_models_keep_configured(self, model):
        assert _effective_temperature(model, 0.3) == 0.3


# ─── Meta sidecar provenance ───────────────────────────────────────────────


class TestMetaSidecarProvenance:
    def test_meta_records_preset_and_fallback(self, tmp_path):
        summary = _fake_summary("openai")
        summary.preset = "high-quality"
        summary.fallback_used = True
        summary.save(tmp_path, "session1")
        meta = json.loads((tmp_path / "session1.summary.meta.json").read_text(encoding="utf-8"))
        assert meta["backend"] == "openai"
        assert meta["preset"] == "high-quality"
        assert meta["fallback_used"] is True

    def test_meta_defaults_no_preset_no_fallback(self, tmp_path):
        summary = _fake_summary("ollama")
        summary.save(tmp_path, "session2")
        meta = json.loads((tmp_path / "session2.summary.meta.json").read_text(encoding="utf-8"))
        assert meta["preset"] is None
        assert meta["fallback_used"] is False
