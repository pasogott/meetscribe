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
        # 0.21.8 (audit M-2): venice/near stay registered + usable, but are
        # no longer in the default chain — their attestation is not bound to
        # the transport, so reaching them requires an explicit opt-in.
        assert sm.DEFAULT_FALLBACK_ORDER == ("tinfoil", "ollama")
        for b in ("venice", "near"):
            assert b in sm.BACKENDS
            assert b in sm.ATTESTED_BACKENDS
            assert b not in sm.DEFAULT_FALLBACK_ORDER

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
        # Opt the unbound tiers in explicitly so the gating is exercised on
        # the full chain (they are not in the default order since 0.21.8).
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "tinfoil,venice,near,ollama")
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

    # A fake NRAS with a real P-384 key: since 0.22.0 the EAT signature is
    # verified against NVIDIA's JWKS (alg pinned ES384, issuer pinned, exp
    # required), so the fake must look exactly like the real thing.
    @staticmethod
    def _nras_keypair():
        from cryptography.hazmat.primitives.asymmetric import ec

        if not hasattr(TestAttestationVerifier, "_kp"):
            TestAttestationVerifier._kp = ec.generate_private_key(ec.SECP384R1())
        return TestAttestationVerifier._kp

    @classmethod
    def _jwks_doc(cls, kid="nv-eat-kid-test"):
        import jwt as pyjwt

        pub = cls._nras_keypair().public_key()
        jwk = pyjwt.algorithms.ECAlgorithm.to_jwk(pub, as_dict=True)
        jwk["kid"] = kid
        return {"keys": [jwk]}

    @classmethod
    def _eat(cls, nonce, overall=True, *, alg="ES384", kid="nv-eat-kid-test",
             iss="https://nras.attestation.nvidia.com", key=None, exp_delta=3600):
        import time

        import jwt as pyjwt

        claims = {"iss": iss, "x-nvidia-overall-att-result": overall, "eat_nonce": nonce,
                  "iat": int(time.time()), "exp": int(time.time()) + exp_delta}
        if alg.startswith("HS"):
            return pyjwt.encode(claims, "s", algorithm=alg, headers={"kid": kid})
        return pyjwt.encode(claims, key or cls._nras_keypair(), algorithm=alg,
                            headers={"kid": kid})

    @classmethod
    def _nras_resp(cls, nonce, overall=True, **eat_kw):
        eat = cls._eat(nonce, overall, **eat_kw)

        class _R:
            status_code = 200
            ok = True

            def json(self):
                return [["JWT", eat], {"GPU-0": eat}]

        return _R()

    @classmethod
    def _wire(cls, monkeypatch, doc, nonce, overall=True, **eat_kw):
        """requests.get serves the attestation doc AND the JWKS; post = NRAS."""
        jwks = cls._jwks_doc()

        def _get(url, *a, **k):
            if "jwks" in url:
                return cls._get_resp(jwks)
            return cls._get_resp(doc)

        monkeypatch.setattr(att.requests, "get", _get)
        monkeypatch.setattr(att.requests, "post",
                            lambda *a, **k: cls._nras_resp(nonce, overall, **eat_kw))
        monkeypatch.setattr(att, "_JWKS_CACHE", {})
        monkeypatch.setattr(att, "_JWKS_FETCHED_AT", 0.0)

    def test_accepted(self, monkeypatch):
        doc = {"model_attestations": [{"nvidia_payload": {"e": 1}}], "nonce": "abc123"}
        self._wire(monkeypatch, doc, "abc123", overall=True)
        att.verify_attestation("https://x/att", api_key="k", nonce="abc123")  # no raise

    def test_nonce_mismatch_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        self._wire(monkeypatch, doc, "DIFFERENT", overall=True)
        with pytest.raises(att.AttestationError, match="nonce"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_overall_fail_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        self._wire(monkeypatch, doc, "abc123", overall=False)
        with pytest.raises(att.AttestationError, match="overall result"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    # ── 0.22.0 (audit M-2): the EAT signature is actually verified ──────────

    def test_eat_signed_by_wrong_key_rejected(self, monkeypatch):
        from cryptography.hazmat.primitives.asymmetric import ec

        doc = {"nvidia_payload": {"e": 1}}
        other = ec.generate_private_key(ec.SECP384R1())
        self._wire(monkeypatch, doc, "abc123", key=other)
        with pytest.raises(att.AttestationError, match="signature/claims rejected"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_eat_with_hs256_rejected(self, monkeypatch):
        """Before 0.22.0 an HS256 token with ANY secret was accepted
        (verify_signature=False).  Alg is pinned now."""
        doc = {"nvidia_payload": {"e": 1}}
        self._wire(monkeypatch, doc, "abc123", alg="HS256")
        with pytest.raises(att.AttestationError, match="unexpected alg"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_eat_unknown_kid_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        self._wire(monkeypatch, doc, "abc123", kid="nv-eat-kid-unknown")
        with pytest.raises(att.AttestationError, match="not in NVIDIA JWKS"):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_eat_wrong_issuer_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        self._wire(monkeypatch, doc, "abc123", iss="https://nras.attestation.nvidia.com.evil")
        with pytest.raises(att.AttestationError):
            att.verify_attestation("https://x/att", api_key="k", nonce="abc123")

    def test_eat_expired_rejected(self, monkeypatch):
        doc = {"nvidia_payload": {"e": 1}}
        self._wire(monkeypatch, doc, "abc123", exp_delta=-10)
        with pytest.raises(att.AttestationError, match="signature/claims rejected"):
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
                completions = types.SimpleNamespace(
                    create=self._create,
                    with_raw_response=types.SimpleNamespace(create=self._create),
                )
                self.chat = types.SimpleNamespace(completions=completions)

            def _create(self, **k):
                raise Exception("Error code: 402 - Insufficient USD or Diem balance")

        mod.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", mod)

        cfg = sm.SummaryConfig(backend="venice", model="e2ee-glm-5-3-flash")
        with pytest.raises(RuntimeError, match="billing"):
            sm._summarize_attested_oai("venice", "sys", "user", cfg)


# ── 0.22.0 (audit M-2): NEAR provider-key response signatures ───────────────


def _secp_keypair():
    from cryptography.hazmat.primitives.asymmetric import ec

    priv = ec.generate_private_key(ec.SECP256K1())
    nums = priv.public_key().public_numbers()
    pub_hex = f"{nums.x:064x}{nums.y:064x}"
    return priv, pub_hex


def _eip191_sign(priv, text: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        Prehashed,
        decode_dss_signature,
    )

    from millet.keccak import keccak256

    msg = text.encode()
    digest = keccak256(b"\x19Ethereum Signed Message:\n" + str(len(msg)).encode() + msg)
    der = priv.sign(digest, ec.ECDSA(Prehashed(hashes.SHA256())))
    r, s_ = decode_dss_signature(der)
    return "0x" + r.to_bytes(32, "big").hex() + s_.to_bytes(32, "big").hex() + "1b"


class TestResponseSignature:
    def _wire_sig(self, monkeypatch, doc):
        monkeypatch.setattr(att.requests, "get", lambda *a, **k: TestAttestationVerifier._get_resp(doc))

    def test_keccak_known_answers(self):
        from millet.keccak import keccak256

        assert keccak256(b"").hex().startswith("c5d2460186f7233c")
        # NOT sha3_256 (different padding).
        import hashlib

        assert keccak256(b"abc") != hashlib.sha3_256(b"abc").digest()

    def test_signing_key_extracted_and_checked(self):
        from millet.keccak import keccak256

        _priv, pub = _secp_keypair()
        addr = keccak256(bytes.fromhex(pub))[-20:].hex()
        doc = {"model_attestations": [{
            "signing_address": "0x" + addr, "signing_public_key": pub, "request_nonce": "ab",
        }]}
        out = att._signing_key_from_doc(doc, "AB")
        assert out == {"signing_address": addr, "signing_public_key": pub}
        bad = {"model_attestations": [{
            "signing_address": "0x" + "0" * 40, "signing_public_key": pub, "request_nonce": "ab",
        }]}
        with pytest.raises(att.AttestationError, match="does not match"):
            att._signing_key_from_doc(bad, "ab")
        with pytest.raises(att.AttestationError, match="nonce"):
            att._signing_key_from_doc(doc, "other")
        assert att._signing_key_from_doc({"nvidia_payload": "x"}, "ab") is None  # venice shape

    def test_valid_signature_verifies(self, monkeypatch):
        import hashlib

        priv, pub = _secp_keypair()
        req, resp = b'{"model":"m","messages":[]}', b'{"id":"c1","choices":[]}'
        text = f"m:{hashlib.sha256(req).hexdigest()}:{hashlib.sha256(resp).hexdigest()}"
        from millet.keccak import keccak256

        self._wire_sig(monkeypatch, {
            "text": text, "signature": _eip191_sign(priv, text),
            "signing_address": "0x" + keccak256(bytes.fromhex(pub))[-20:].hex(),
        })
        att.verify_response_signature(
            "https://x/sig/c1", api_key="k", model="m",
            request_bytes=req, response_bytes=resp, signing_public_key=pub,
        )

    @pytest.mark.parametrize("tamper", ["request", "response", "model", "key", "sig"])
    def test_tampering_is_detected(self, monkeypatch, tamper):
        import hashlib

        priv, pub = _secp_keypair()
        req, resp = b'{"model":"m"}', b'{"id":"c1"}'
        text = f"m:{hashlib.sha256(req).hexdigest()}:{hashlib.sha256(resp).hexdigest()}"
        sig = _eip191_sign(priv, text)
        check_req, check_resp, check_model, check_pub = req, resp, "m", pub
        if tamper == "request":
            check_req = req + b" "
        elif tamper == "response":
            check_resp = resp + b" "
        elif tamper == "model":
            check_model = "other"
        elif tamper == "key":
            _, check_pub = _secp_keypair()
        elif tamper == "sig":
            sig = sig[:-4] + ("0000" if sig[-4:] != "0000" else "1111")
        self._wire_sig(monkeypatch, {"text": text, "signature": sig})
        with pytest.raises(att.AttestationError):
            att.verify_response_signature(
                "https://x/sig/c1", api_key="k", model=check_model,
                request_bytes=check_req, response_bytes=check_resp,
                signing_public_key=check_pub,
            )

    def test_near_path_labels_provider_signed_and_fails_closed(self, monkeypatch):
        """A valid signature earns only the honest provider-key label; a bad
        one raises AttestationError (→ fallback loop, never saved)."""
        import hashlib
        import sys
        import types

        priv, pub = _secp_keypair()
        from millet.keccak import keccak256

        addr = keccak256(bytes.fromhex(pub))[-20:].hex()
        req_bytes = b'{"model":"z-ai/glm-5.3-flash","messages":[]}'
        resp_bytes = b'{"id":"chat-1","choices":[{"message":{"content":"# Summary"}}]}'
        good_text = (
            f"z-ai/glm-5.3-flash:{hashlib.sha256(req_bytes).hexdigest()}:"
            f"{hashlib.sha256(resp_bytes).hexdigest()}"
        )
        state = {"sig_text": good_text}

        monkeypatch.setattr(sm, "_resolve_attested_api_key", lambda b: "k")
        monkeypatch.setattr(sm, "verify_model", lambda *a, **k: None)
        monkeypatch.setattr(
            att, "verify_attestation",
            lambda *a, **k: {"signing_address": addr, "signing_public_key": pub},
        )

        def _get(url, *a, **k):
            assert url.endswith("/v1/signature/chat-1")
            return TestAttestationVerifier._get_resp({
                "text": state["sig_text"], "signature": _eip191_sign(priv, state["sig_text"]),
                "signing_address": "0x" + addr,
            })

        monkeypatch.setattr(att.requests, "get", _get)

        mod = types.ModuleType("openai")

        class _Raw:
            http_response = types.SimpleNamespace(
                request=types.SimpleNamespace(content=req_bytes), content=resp_bytes,
            )

            def parse(self):
                return types.SimpleNamespace(
                    id="chat-1",
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="# Summary"))],
                )

        class _Client:
            def __init__(self, **k):
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                    with_raw_response=types.SimpleNamespace(create=lambda **kw: _Raw()),
                ))

        mod.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", mod)

        cfg = sm.SummaryConfig(backend="near", model="z-ai/glm-5.3-flash")
        out = sm._summarize_attested_oai("near", "sys", "user", cfg)
        assert out.model == "z-ai/glm-5.3-flash (response-signed, provider-key)"
        assert "TEE" not in out.model
        assert "enclave" not in out.model
        assert "attested" not in out.model

        state["sig_text"] = good_text[:-1] + ("0" if good_text[-1] != "0" else "1")
        with pytest.raises(att.AttestationError):
            sm._summarize_attested_oai("near", "sys", "user", cfg)
