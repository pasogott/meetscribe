"""Tests for the attested OpenAI-compatible fallback backends (venice/near).

Covers the three things the new code must get right:
  * vision-gating — a frames job never falls back to a text-only tier;
  * attestation gate — a failed enclave attestation is loud, not a silent
    unattested summary;
  * chain wiring — registry/order/model defaults are consistent.

No network, no real SDK: the openai client and attestation verifier are
monkeypatched.
"""

from __future__ import annotations

import pytest

import millet.attestation as att
import millet.summarize as sm


class TestRegistry:
    def test_fallback_order_and_backends(self):
        assert sm.DEFAULT_FALLBACK_ORDER == ("tinfoil", "venice", "near", "ollama")
        for b in ("venice", "near"):
            assert b in sm.BACKENDS
            assert b in sm.ATTESTED_BACKENDS

    def test_default_models(self):
        # Real ids verified live 2026-09-15: Venice attested models need the
        # e2ee- prefix; NEAR uses z-ai/ (slash).
        assert sm._default_model_for_backend("venice") == "e2ee-glm-5-3-flash"
        assert sm._default_model_for_backend("near") == "z-ai/glm-5.3-flash"

    def test_vision_capability(self):
        assert sm.backend_supports_vision("tinfoil") is True
        assert sm.backend_supports_vision("venice") is True  # has qwen3-vl
        assert sm.backend_supports_vision("near") is False  # text-only
        assert sm.backend_supports_vision("ollama") is False


class TestVisionGating:
    def test_frames_job_skips_text_only_fallback(self, monkeypatch, tmp_path):
        """A frames session must not fall back to near/ollama (text-only)."""
        frame = tmp_path / "cue.png"
        frame.write_bytes(b"\x89PNG\r\n")
        logs: list[str] = []

        # Every backend "available"; make tinfoil (primary) fail so the loop
        # walks the fallback order and we can see which tiers it skips.
        monkeypatch.setattr(sm, "is_backend_available", lambda cfg: True)

        def _dispatch(backend, *a, **k):
            if not sm.backend_supports_vision(backend):
                raise AssertionError(f"frames routed to text-only tier {backend!r}")
            raise RuntimeError(f"{backend} boom")  # force continue

        monkeypatch.setattr(sm, "_dispatch", _dispatch)

        cfg = sm.SummaryConfig(backend="tinfoil", frames=[frame])
        with pytest.raises(RuntimeError):
            sm.summarize("transcript", cfg, progress_callback=logs.append)

        # near + ollama were skipped explicitly as text-only, never dispatched.
        assert any("text-only" in m and "near" in m for m in logs)
        assert any("text-only" in m and "ollama" in m for m in logs)


class TestAttestationGate:
    def test_attestation_failure_is_loud(self, monkeypatch):
        """A failed attestation raises before any prompt is sent."""
        monkeypatch.setattr(sm, "_resolve_attested_api_key", lambda b: "k")
        monkeypatch.setattr(sm, "verify_model", lambda *a, **k: None)

        def _fail(url, **k):
            raise att.AttestationError("bad quote")

        monkeypatch.setattr(att, "verify_attestation", _fail)

        # openai must never be reached if attestation fails first.
        import sys
        import types

        boom = types.ModuleType("openai")
        boom.OpenAI = lambda **k: (_ for _ in ()).throw(AssertionError("reached openai"))
        monkeypatch.setitem(sys.modules, "openai", boom)

        cfg = sm.SummaryConfig(backend="venice", model="e2ee-glm-5-3-flash")
        with pytest.raises(att.AttestationError):
            sm._summarize_attested_oai("venice", "sys", "user", cfg)

    def test_missing_key_fails_before_attestation(self, monkeypatch):
        monkeypatch.setattr(sm, "_resolve_attested_api_key", lambda b: None)
        cfg = sm.SummaryConfig(backend="near", model="z-ai/glm-5.3-flash")
        with pytest.raises(RuntimeError, match="NEAR_AI_API_KEY"):
            sm._summarize_attested_oai("near", "sys", "user", cfg)


class TestAttestationVerifier:
    """Real two-step flow: GET doc -> extract nvidia_payload -> POST to NRAS ->
    require overall PASS + our nonce.  NRAS is mocked (needs a real GPU + key);
    the payload-extraction and verdict/nonce gates are exercised for real.
    """

    @staticmethod
    def _get_resp(doc):
        class _R:
            status_code = 200
            ok = True

            def raise_for_status(self):
                pass

            def json(self):
                return doc

        return _R()

    @staticmethod
    def _nras_resp(nonce, overall=True):
        import jwt as pyjwt

        eat = pyjwt.encode(
            {"iss": "https://nras.attestation.nvidia.com",
             "x-nvidia-overall-att-result": overall, "eat_nonce": nonce},
            "s",
        )

        class _R:
            status_code = 200
            ok = True

            def json(self):
                return [["JWT", eat], {"GPU-0": eat}]

        return _R()

    def test_accepted(self, monkeypatch):
        doc = {"model_attestations": [{"nvidia_payload": {"e": 1}}], "nonce": "abc123"}
        monkeypatch.setattr(att.requests, "get", lambda *a, **k: self._get_resp(doc))
        monkeypatch.setattr(att.requests, "post",
                            lambda *a, **k: self._nras_resp("abc123", overall=True))
        att.verify_attestation("https://x/att", api_key="k", nonce="abc123")  # no raise

    def test_nonce_mismatch_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        monkeypatch.setattr(att.requests, "get", lambda *a, **k: self._get_resp(doc))
        monkeypatch.setattr(att.requests, "post",
                            lambda *a, **k: self._nras_resp("DIFFERENT", overall=True))
        with pytest.raises(att.AttestationError, match="nonce"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_overall_fail_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        monkeypatch.setattr(att.requests, "get", lambda *a, **k: self._get_resp(doc))
        monkeypatch.setattr(att.requests, "post",
                            lambda *a, **k: self._nras_resp("abc123", overall=False))
        with pytest.raises(att.AttestationError, match="overall result"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_no_payload_rejected(self, monkeypatch):
        doc = {"note": "no nvidia_payload here"}
        monkeypatch.setattr(att.requests, "get", lambda *a, **k: self._get_resp(doc))
        with pytest.raises(att.AttestationError, match="no nvidia_payload"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")


class TestBillingClassification:
    @pytest.mark.parametrize("msg", [
        "HTTP 402", "Insufficient USD or Diem balance", "no_limit_configured",
        "No spending limit configured", "add credits",
    ])
    def test_billing_errors_detected(self, msg):
        assert sm._is_billing_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", ["connection reset", "timeout", "500 server error"])
    def test_non_billing_not_flagged(self, msg):
        assert sm._is_billing_error(Exception(msg)) is False

    def test_billing_error_fails_loud_in_backend(self, monkeypatch):
        """A 402 mid-completion must raise with a credits message, not retry."""
        monkeypatch.setattr(sm, "_resolve_attested_api_key", lambda b: "k")
        monkeypatch.setattr(sm, "verify_model", lambda *a, **k: None)

        from millet import attestation as _att
        monkeypatch.setattr(_att, "verify_attestation", lambda *a, **k: None)

        import sys
        import types

        mod = types.ModuleType("openai")

        class _Client:
            def __init__(self, **k):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **k):
                raise Exception("Error code: 402 - Insufficient USD or Diem balance")

        mod.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", mod)

        cfg = sm.SummaryConfig(backend="venice", model="e2ee-glm-5-3-flash")
        with pytest.raises(RuntimeError, match="billing"):
            sm._summarize_attested_oai("venice", "sys", "user", cfg)
