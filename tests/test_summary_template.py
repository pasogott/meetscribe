"""Tests for summary templates (--summary-template / MILLET_SUMMARY_TEMPLATE).

Covers prompt-file resolution, SummaryConfig validation, dispatch routing
(templates force the single-pass Ollama flow), artifact naming in
MeetingSummary.save(), and end-to-end provenance through summarize().
"""

from __future__ import annotations

import json

import pytest

import millet.summarize as sm
from millet.frontmatter import FrontmatterContext
from millet.summarize import (
    MeetingSummary,
    SummaryConfig,
    _build_system_prompt,
    _dispatch,
    _load_user_prompt_template,
)


class TestTemplateSystemPrompt:
    def test_iteration_plan_template_is_used(self):
        prompt = _build_system_prompt("en", template="iteration-plan")
        assert "product iteration assistant" in prompt
        assert "## Issues" in prompt
        assert "## UX Notes" in prompt
        # Keeps the fenced-JSON contract downstream tooling parses.
        assert "```json" in prompt
        assert "action_items" in prompt

    def test_unknown_template_falls_back_to_default(self):
        prompt = _build_system_prompt("en", template="no-such-template")
        assert "Meeting Overview" in prompt

    def test_no_template_is_default(self):
        assert _build_system_prompt("en") == _build_system_prompt("en", template=None)

    def test_iteration_plan_prompt_has_no_format_leftovers(self):
        """All placeholders in the template file must be consumed by .format."""
        prompt = _build_system_prompt("en", template="iteration-plan")
        assert "{transcript}" not in prompt
        assert "{lang_instruction}" not in prompt


class TestTemplateUserPrompt:
    def test_iteration_plan_user_template(self):
        tpl = _load_user_prompt_template("iteration-plan")
        assert "iteration plan" in tpl.lower()
        assert "{transcript}" in tpl

    def test_unknown_template_falls_back_to_default(self):
        assert _load_user_prompt_template("no-such-template") == _load_user_prompt_template()


class TestSummaryConfigTemplate:
    def test_template_explicit(self):
        cfg = SummaryConfig(template="iteration-plan")
        assert cfg.template == "iteration-plan"

    def test_template_normalized(self):
        cfg = SummaryConfig(template=" Iteration-Plan ")
        assert cfg.template == "iteration-plan"

    def test_template_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_TEMPLATE", "iteration-plan")
        cfg = SummaryConfig()
        assert cfg.template == "iteration-plan"

    def test_template_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_TEMPLATE", "other")
        cfg = SummaryConfig(template="iteration-plan")
        assert cfg.template == "iteration-plan"

    def test_template_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="Invalid summary template"):
            SummaryConfig(template="../etc/passwd")

    def test_template_rejects_bad_charset(self):
        with pytest.raises(ValueError, match="Invalid summary template"):
            SummaryConfig(template="bad name!")

    def test_template_default_is_none(self, monkeypatch):
        monkeypatch.delenv("MILLET_SUMMARY_TEMPLATE", raising=False)
        monkeypatch.delenv("MEETSCRIBE_SUMMARY_TEMPLATE", raising=False)
        assert SummaryConfig().template is None


class TestDispatchTemplateRouting:
    def test_template_forces_ollama_singlepass(self, monkeypatch):
        seen: list[str] = []

        def fake_twopass(transcript_text, config, language=None):
            seen.append("twopass")
            return MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="ollama",
            )

        def fake_singlepass(system_prompt, user_prompt, config):
            seen.append("singlepass")
            return MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="ollama",
            )

        monkeypatch.setattr(sm, "_summarize_ollama_twopass", fake_twopass)
        monkeypatch.setattr(sm, "_summarize_ollama", fake_singlepass)

        cfg = SummaryConfig(
            backend="ollama", ollama_singlepass=False, template="iteration-plan"
        )
        _dispatch(
            "ollama", "sys", "usr", cfg,
            transcript_text="some transcript", language="en",
        )
        assert seen == ["singlepass"]

    def test_template_survives_fallback_config(self, monkeypatch):
        """A fallback backend's rebuilt config must keep the template."""
        captured: list[SummaryConfig] = []

        def fake_tinfoil(system_prompt, user_prompt, config):
            captured.append(config)
            return MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="tinfoil",
            )

        monkeypatch.setattr(sm, "_summarize_tinfoil", fake_tinfoil)

        cfg = SummaryConfig(backend="ollama", template="iteration-plan")
        _dispatch("tinfoil", "sys", "usr", cfg)
        assert captured[0].template == "iteration-plan"


class TestSaveArtifactNaming:
    def _summary(self) -> MeetingSummary:
        return MeetingSummary(
            markdown="## Overview\nDemoed the thing.",
            model="test-model",
            elapsed_seconds=1.5,
            backend="ollama",
            template="iteration-plan",
        )

    def test_default_artifact_unchanged(self, tmp_path):
        path = self._summary().save(tmp_path, "sess")
        assert path.name == "sess.summary.md"
        assert (tmp_path / "sess.summary.meta.json").exists()

    def test_template_artifact_names(self, tmp_path):
        path = self._summary().save(tmp_path, "sess", artifact="iteration-plan")
        assert path.name == "sess.iteration-plan.md"
        assert (tmp_path / "sess.iteration-plan.meta.json").exists()
        # Does not clobber the regular summary name.
        assert not (tmp_path / "sess.summary.md").exists()

    def test_template_meta_records_template(self, tmp_path):
        self._summary().save(tmp_path, "sess", artifact="iteration-plan")
        meta = json.loads((tmp_path / "sess.iteration-plan.meta.json").read_text())
        assert meta["template"] == "iteration-plan"

    def test_template_frontmatter_sidecar_name(self, tmp_path):
        self._summary().save(
            tmp_path, "sess",
            frontmatter_context=FrontmatterContext(title="Demo"),
            artifact="iteration-plan",
        )
        assert (tmp_path / "sess.iteration-plan.frontmatter.json").exists()
        assert not (tmp_path / "sess.frontmatter.json").exists()

    def test_template_and_lang_suffix_compose(self, tmp_path):
        path = self._summary().save(
            tmp_path, "sess", lang_suffix="de", artifact="iteration-plan"
        )
        assert path.name == "sess.iteration-plan.de.md"
        assert (tmp_path / "sess.iteration-plan.de.meta.json").exists()


class TestSummarizeEndToEndTemplate:
    def test_template_prompts_and_provenance(self, monkeypatch, tmp_path):
        captured: dict[str, str] = {}

        def fake_singlepass(system_prompt, user_prompt, config):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return MeetingSummary(
                markdown="## Overview\nx" + "y" * 500,
                model=config.model, elapsed_seconds=1.0, backend="ollama",
            )

        monkeypatch.setattr(sm, "_summarize_ollama", fake_singlepass)
        monkeypatch.setattr(sm, "is_backend_available", lambda cfg: True)

        cfg = SummaryConfig(
            backend="ollama", ollama_singlepass=True, template="iteration-plan"
        )
        result = sm.summarize("[00:00:01] YOU: demo starts", cfg, language="en")

        assert "product iteration assistant" in captured["system"]
        assert "iteration plan" in captured["user"].lower()
        assert "[00:00:01] YOU: demo starts" in captured["user"]
        assert result.template == "iteration-plan"

        path = result.save(tmp_path, "sess", artifact=result.template or "summary")
        assert path.name == "sess.iteration-plan.md"
