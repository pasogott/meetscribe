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

_HTTP_TIMEOUT = 30


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

    NRAS returns ``[["JWT", "<eat>"], {"GPU-0": "<eat>"}, ...]``.  We decode the
    top EAT (NVIDIA-signed) and check the overall result flag and eat_nonce.
    """
    import jwt as pyjwt

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
            claims = pyjwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            raise AttestationError(f"could not parse NVIDIA NRAS EAT: {e}") from e

        if claims.get("iss", "").lower().find("nvidia") < 0:
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


def verify_attestation(
    attestation_url: str,
    *,
    api_key: str | None = None,
    nonce: str | None = None,
) -> None:
    """Fetch and verify a backend's enclave attestation. Raise on any failure.

    ``attestation_url`` must accept a ``nonce`` query parameter (Venice:
    ``/api/v1/tee/attestation?model=<e2ee-id>&nonce=...``; NEAR:
    ``/v1/attestation/report?model=<id>&signing_algo=ecdsa&nonce=...``).  We send
    a fresh nonce, pull the ``nvidia_payload`` from the response, POST it to
    NRAS, and require an overall-PASS EAT that binds our nonce.

    Returns None on success; raises :class:`AttestationError` otherwise.
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
