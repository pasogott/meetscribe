"""First-party TEE attestation checks for the OpenAI-compatible fallback backends.

The Tinfoil backend verifies its enclave through the Tinfoil SDK.  The generic
attested backends (venice, near) talk plain OpenAI-compatible HTTP, so before we
trust one enough to badge a summary "attested" we verify the enclave ourselves.

Both providers expose the SAME Phala/dstack-style attestation (verified live
2026-09-15): a GET returns a document containing an opaque ``nvidia_payload``
(GPU evidence) plus the enclave's ``signing_address`` and the ``nonce`` we sent.
The NVIDIA GPU evidence is NOT a ready JWT — it must be POSTed to NVIDIA's
Remote Attestation Service (NRAS), which returns a signed Entity Attestation
Token (EAT).  We then require, from that NVIDIA-signed EAT:

  1. ``x-nvidia-overall-att-result == True`` — NVIDIA vouches the GPU is a
     genuine Confidential-Computing device in a good state; and
  2. ``eat_nonce`` equals the fresh nonce we sent — this is not a replayed or
     cached attestation.

Since 0.22.0 (security audit 2026-10-30, M-2) two more things are checked:

  3. the EAT's **signature** — against NVIDIA's published JWKS
     (``/.well-known/jwks.json``), algorithm pinned to ES384, issuer pinned,
     ``exp`` enforced.  Until 0.22.0 the EAT was decoded with
     ``verify_signature=False``, so its integrity rested on the TLS session
     to nvidia.com alone; and
  4. for NEAR, the **per-response signature**: every completion can be
     fetched back from ``/v1/signature/{id}`` as
     ``"<model>:<sha256(request)>:<sha256(response)>"`` signed (EIP-191,
     secp256k1) by the model enclave's ``signing_address`` — the same key the
     attestation report binds into the TDX ``report_data``.  Verifying it
     with the attested ``signing_public_key`` proves the bytes we received
     were produced inside the attested enclave for the bytes we sent, not
     by the gateway or anything in between.  This is output integrity and
     origin binding; it is still NOT proof that the gateway's TLS
     termination point cannot read the request, which is why NEAR stays
     opt-in and is recorded as "(TEE, response-signed)" rather than as a
     Tinfoil-grade attestation.

What we deliberately DO NOT re-derive is full Intel TDX quote validation
(``intel_quote`` — MRTD/RTMR measurement policy + the Intel DCAP certificate
chain / TCB collateral).  That is a large, security-critical crypto pipeline; a
subtly wrong reimplementation would mint a *false* "attested" badge, worse than
none.  Providers publish reference verifiers for it (NEAR's
nearai-cloud-verifier; Phala's dcap-qvl).  We verify the NVIDIA-signed GPU
evidence + nonce freshness ourselves and treat deep TDX-quote validation as a
delegated ceiling.

# ponytail: verifies NRAS GPU EAT (overall result + nonce) only; full Intel TDX
#   quote/DCAP validation is the ceiling — wire dcap-qvl or a provider reference
#   verifier here if TDX-level assurance is ever required, don't hand-roll DCAP.

A failure here is loud: the caller (summarize) treats it like the Tinfoil
attestation error — fall through to the next backend, never save an unverified
summary as attested.
"""

from __future__ import annotations

import json
import logging
import secrets

import requests

logger = logging.getLogger("millet.attestation")

# NVIDIA Remote Attestation Service.  v4 is current (verified live 2026-09-15);
# v3 kept as a fallback since NEAR's docs still show it.
NRAS_ATTEST_URLS = (
    "https://nras.attestation.nvidia.com/v4/attest/gpu",
    "https://nras.attestation.nvidia.com/v3/attest/gpu",
)

NRAS_JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
NRAS_ISSUER = "https://nras.attestation.nvidia.com"
# NRAS signs EATs with ES384 (verified live 2026-10-30: kid nv-eat-kid-prod-*,
# crv P-384).  Pinned — a token claiming another alg is rejected outright.
NRAS_EAT_ALGS = ("ES384",)
_HTTP_TIMEOUT = 30

# JWKS cache: NRAS rotates keys (~200 live kids); a kid miss re-fetches once.
_JWKS_CACHE: dict[str, object] = {}
_JWKS_FETCHED_AT = 0.0
_JWKS_TTL = 6 * 3600.0


class AttestationError(RuntimeError):
    """Raised when a backend's enclave attestation cannot be verified.

    Kept distinct so summarize() can treat it like the Tinfoil attestation
    fault: a provider-side trust failure, never an excuse to fall through to an
    unattested backend silently.
    """


def new_nonce() -> str:
    """Fresh 32-byte hex nonce (64 chars) for one attestation challenge."""
    return secrets.token_hex(32)


def _extract_nvidia_payload(doc: dict) -> object | None:
    """Pull the NVIDIA GPU evidence payload out of a provider attestation doc.

    Both providers nest it slightly differently: NEAR under
    ``model_attestations[0].nvidia_payload``, Venice at the top level (attested
    ``e2ee-`` models) — and sometimes both.  Return the first one found.
    """
    atts = doc.get("model_attestations")
    if isinstance(atts, list) and atts and isinstance(atts[0], dict):
        p = atts[0].get("nvidia_payload")
        if p:
            return p
    return doc.get("nvidia_payload")


def _verify_nras(payload: object, nonce: str) -> None:
    """POST GPU evidence to NRAS; require overall PASS + our nonce bound.

    NRAS returns ``[["JWT", "<eat>"], {"GPU-0": "<eat>"}, ...]``.  We verify the
    top EAT's signature against NVIDIA's JWKS and check the overall result
    flag and eat_nonce.
    """
    body = payload if isinstance(payload, (dict, list)) else json.loads(payload)

    last_err: Exception | None = None
    for url in NRAS_ATTEST_URLS:
        try:
            r = requests.post(
                url,
                headers={"accept": "application/json", "content-type": "application/json"},
                json=body,
                timeout=_HTTP_TIMEOUT,
            )
        except Exception as e:  # transient reaching NRAS — try next/raise
            last_err = e
            continue
        if r.status_code == 404:  # wrong API version, try the next
            last_err = RuntimeError(f"{url} -> 404")
            continue
        if not r.ok:
            raise AttestationError(
                f"NVIDIA NRAS rejected the GPU evidence ({url}): "
                f"HTTP {r.status_code} {r.text[:200]}"
            )
        try:
            data = r.json()
            token = data[0][1] if isinstance(data, list) and isinstance(data[0], list) else None
            if not token:
                raise ValueError(f"unexpected NRAS response shape: {type(data)}")
        except Exception as e:
            raise AttestationError(f"could not parse NVIDIA NRAS EAT: {e}") from e
        claims = _decode_eat_verified(token)

        if claims.get("iss") != NRAS_ISSUER:
            raise AttestationError(f"NRAS EAT issuer is not NVIDIA: {claims.get('iss')!r}")
        if claims.get("x-nvidia-overall-att-result") is not True:
            raise AttestationError(
                "NVIDIA attestation overall result is not PASS "
                f"(x-nvidia-overall-att-result={claims.get('x-nvidia-overall-att-result')!r})"
            )
        eat_nonce = str(claims.get("eat_nonce", ""))
        if eat_nonce.lower() != nonce.lower():
            raise AttestationError(
                "NVIDIA attestation did not bind our freshness nonce "
                "(replay/stale evidence); refusing to trust it"
            )
        logger.info("NVIDIA NRAS attestation PASS + nonce bound (%s)", url)
        return

    raise AttestationError(f"could not reach NVIDIA NRAS to verify GPU evidence: {last_err}")


def _nras_jwks(force: bool = False) -> dict[str, object]:
    """NVIDIA's JWKS, keyed by kid; cached for a few hours."""
    import time

    global _JWKS_CACHE, _JWKS_FETCHED_AT
    if not force and _JWKS_CACHE and time.time() - _JWKS_FETCHED_AT < _JWKS_TTL:
        return _JWKS_CACHE
    r = requests.get(NRAS_JWKS_URL, timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    keys = r.json().get("keys") or []
    _JWKS_CACHE = {k["kid"]: k for k in keys if isinstance(k, dict) and k.get("kid")}
    _JWKS_FETCHED_AT = time.time()
    return _JWKS_CACHE


def _decode_eat_verified(token: str) -> dict:
    """Decode an NRAS EAT with its signature verified against NVIDIA's JWKS.

    Algorithm and issuer pinned; ``exp`` enforced.  A kid not in the cached
    set triggers exactly one re-fetch (key rotation), never a loop.
    """
    import jwt as pyjwt

    try:
        header = pyjwt.get_unverified_header(token)
    except Exception as e:
        raise AttestationError(f"malformed NVIDIA NRAS EAT: {e}") from e
    alg = header.get("alg")
    if alg not in NRAS_EAT_ALGS:
        raise AttestationError(f"NRAS EAT uses unexpected alg {alg!r}; expected {NRAS_EAT_ALGS}")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise AttestationError("NRAS EAT has no kid")
    try:
        jwk = _nras_jwks().get(kid) or _nras_jwks(force=True).get(kid)
    except Exception as e:
        raise AttestationError(f"could not fetch NVIDIA NRAS JWKS: {e}") from e
    if jwk is None:
        raise AttestationError(f"NRAS EAT kid {kid!r} not in NVIDIA JWKS")
    try:
        key = pyjwt.PyJWK(jwk).key
        return pyjwt.decode(
            token, key, algorithms=list(NRAS_EAT_ALGS), issuer=NRAS_ISSUER,
            options={"require": ["exp", "iss"]},
        )
    except Exception as e:
        raise AttestationError(f"NVIDIA NRAS EAT signature/claims rejected: {e}") from e


def verify_attestation(
    attestation_url: str,
    *,
    api_key: str | None = None,
    nonce: str | None = None,
) -> dict | None:
    """Fetch and verify a backend's enclave attestation. Raise on any failure.

    ``attestation_url`` must accept a ``nonce`` query parameter (Venice:
    ``/api/v1/tee/attestation?model=<e2ee-id>&nonce=...``; NEAR:
    ``/v1/attestation/report?model=<id>&signing_algo=ecdsa&nonce=...``).  We send
    a fresh nonce, pull the ``nvidia_payload`` from the response, POST it to
    NRAS, and require an overall-PASS EAT that binds our nonce.

    Returns the model enclave's signing key (``{"signing_address",
    "signing_public_key"}``) when the report carries one (NEAR), else None;
    raises :class:`AttestationError` on any failure.
    """
    nonce = nonce or new_nonce()
    sep = "&" if "?" in attestation_url else "?"
    url = f"{attestation_url}{sep}nonce={nonce}"
    headers = {"accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        doc = resp.json()
    except Exception as e:
        raise AttestationError(
            f"could not fetch attestation from {attestation_url}: {e}"
        ) from e

    if not isinstance(doc, dict):
        raise AttestationError(f"attestation from {attestation_url} was not a JSON object")

    payload = _extract_nvidia_payload(doc)
    if not payload:
        raise AttestationError(
            f"no nvidia_payload in the attestation document from {attestation_url}; "
            f"refusing to treat this backend as attested"
        )

    _verify_nras(payload, nonce)
    logger.info("attestation verified for %s (NVIDIA NRAS GPU + nonce)", attestation_url)
    return _signing_key_from_doc(doc, nonce)


def _signing_key_from_doc(doc: dict, nonce: str) -> dict | None:
    """Extract the model enclave's signing key from a NEAR-style report.

    Returns ``{"signing_address", "signing_public_key"}`` (lower-case hex,
    no ``0x``) when present and self-consistent — the address must equal
    ``keccak256(pubkey)[-20:]`` and the report's ``request_nonce`` must be
    ours — else None (Venice's doc carries no per-model key).
    """
    atts = doc.get("model_attestations")
    if not (isinstance(atts, list) and atts and isinstance(atts[0], dict)):
        return None
    ma = atts[0]
    addr = ma.get("signing_address")
    pub = ma.get("signing_public_key")
    if not (isinstance(addr, str) and isinstance(pub, str)):
        return None
    rn = ma.get("request_nonce")
    if isinstance(rn, str) and rn.lower() != nonce.lower():
        raise AttestationError("model attestation does not carry our nonce")
    addr = addr.lower().removeprefix("0x")
    pub = pub.lower().removeprefix("0x")
    if len(pub) != 128 or len(addr) != 40:
        raise AttestationError("model attestation signing key has an unexpected shape")
    from .keccak import keccak256

    if keccak256(bytes.fromhex(pub))[-20:].hex() != addr:
        raise AttestationError("model attestation signing_address does not match its public key")
    return {"signing_address": addr, "signing_public_key": pub}


def verify_response_signature(
    signature_url: str,
    *,
    api_key: str | None,
    model: str,
    request_bytes: bytes,
    response_bytes: bytes,
    signing_public_key: str,
) -> None:
    """Prove a completion was produced by the attested model enclave.

    NEAR exposes ``/v1/signature/{chat_id}`` returning
    ``text = "<model>:<sha256(request)>:<sha256(response)>"`` and an
    Ethereum-style (EIP-191 personal-sign, secp256k1) ``signature`` by the
    enclave's key.  We recompute both hashes from the exact bytes we sent
    and received, require ``text`` to match, and verify the signature with
    the ``signing_public_key`` the attestation report bound into the TDX
    ``report_data`` (0.22.0, audit M-2).  Raises :class:`AttestationError`.
    """
    import hashlib

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        Prehashed,
        encode_dss_signature,
    )

    from .keccak import keccak256

    headers = {"accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.get(signature_url, headers=headers, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        sig_doc = r.json()
    except Exception as e:
        raise AttestationError(f"could not fetch response signature: {e}") from e
    if not isinstance(sig_doc, dict):
        raise AttestationError("response signature document is not a JSON object")

    expected = (
        f"{model}:{hashlib.sha256(request_bytes).hexdigest()}:"
        f"{hashlib.sha256(response_bytes).hexdigest()}"
    )
    text = sig_doc.get("text")
    if text != expected:
        raise AttestationError(
            "response signature covers different bytes than we sent/received "
            "(request or response was altered in transit, or the id is wrong)"
        )
    sig_hex = str(sig_doc.get("signature", "")).lower().removeprefix("0x")
    if len(sig_hex) != 130:
        raise AttestationError("response signature has an unexpected length")
    addr = str(sig_doc.get("signing_address", "")).lower().removeprefix("0x")
    pub = signing_public_key.lower().removeprefix("0x")
    if addr and keccak256(bytes.fromhex(pub))[-20:].hex() != addr:
        raise AttestationError("response was signed by a key other than the attested one")

    raw = bytes.fromhex(sig_hex)
    r_int, s_int = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:64], "big")
    msg = text.encode("utf-8")
    digest = keccak256(b"\x19Ethereum Signed Message:\n" + str(len(msg)).encode() + msg)
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), b"\x04" + bytes.fromhex(pub),
        )
        key.verify(
            encode_dss_signature(r_int, s_int), digest, ec.ECDSA(Prehashed(hashes.SHA256())),
        )
    except (InvalidSignature, ValueError) as e:
        raise AttestationError(f"response signature does not verify: {e}") from e
    logger.info("response signature verified against the attested enclave key (%s)", model)


if __name__ == "__main__":
    # Offline self-check: payload extraction (both nesting shapes) + the NRAS
    # verdict/nonce gate, with NRAS mocked. Guards the branches most likely to rot.
    import types

    # payload extraction, both shapes
    assert _extract_nvidia_payload({"nvidia_payload": "x"}) == "x", "top-level payload"
    assert _extract_nvidia_payload(
        {"model_attestations": [{"nvidia_payload": "y"}]}
    ) == "y", "nested payload"
    assert _extract_nvidia_payload({"nope": 1}) is None, "no payload -> None"

    # nonce/verdict gate with a fake NRAS
    import jwt as pyjwt

    import millet.attestation as A

    def _fake_post(good=True, nonce="n"):
        eat = pyjwt.encode(
            {"iss": "https://nras.attestation.nvidia.com",
             "x-nvidia-overall-att-result": good, "eat_nonce": nonce},
            "s",
        )
        return types.SimpleNamespace(
            status_code=200, ok=True, json=lambda: [["JWT", eat], {"GPU-0": eat}]
        )

    orig = A.requests.post
    try:
        pl = {"evidence": "..."}  # dict payload, like the real nvidia_payload
        A.requests.post = lambda *a, **k: _fake_post(good=True, nonce="abc")
        _verify_nras(pl, "abc")  # should pass
        A.requests.post = lambda *a, **k: _fake_post(good=True, nonce="different")
        try:
            _verify_nras(pl, "abc")
            raise SystemExit("nonce mismatch not caught")
        except AttestationError:
            pass
        A.requests.post = lambda *a, **k: _fake_post(good=False, nonce="abc")
        try:
            _verify_nras(pl, "abc")
            raise SystemExit("bad verdict not caught")
        except AttestationError:
            pass
    finally:
        A.requests.post = orig
    print("attestation self-check ok")
