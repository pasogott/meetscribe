"""Keccak-256 — the pre-FIPS variant Ethereum uses (NOT ``hashlib.sha3_256``,
which applies the 0x06 SHA-3 domain padding and yields different digests).

Needed to check NEAR's per-response signatures (``millet.attestation``):
the enclave signs with an Ethereum-style secp256k1 key, so the message is
``keccak256("\\x19Ethereum Signed Message:\\n" + len + text)`` and the
attested ``signing_address`` is ``keccak256(pubkey_xy)[-20:]``.  Pure
Python on purpose — the inputs are a few hundred bytes, and pulling in
pycryptodome / eth-hash for one digest is not worth a dependency.
Self-checked against the known empty-input digest on import.
"""

from __future__ import annotations

_RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
_ROT = (
    (0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56), (27, 20, 39, 8, 14),
)
_MASK = (1 << 64) - 1
_RATE = 136  # bytes, for a 256-bit digest


def _rotl(v: int, r: int) -> int:
    return ((v << r) | (v >> (64 - r))) & _MASK if r else v


def _keccak_f(a: list[list[int]]) -> list[list[int]]:
    for rc in _RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        a = [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl(a[x][y], _ROT[x][y])
        a = [[b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]) for y in range(5)]
             for x in range(5)]
        a[0][0] ^= rc
    return a


def keccak256(data: bytes) -> bytes:
    """Keccak-256 digest of ``data`` (Ethereum padding, 0x01 … 0x80)."""
    state = [[0] * 5 for _ in range(5)]
    buf = bytearray(data)
    buf.append(0x01)
    while len(buf) % _RATE:
        buf.append(0)
    buf[-1] |= 0x80
    for off in range(0, len(buf), _RATE):
        block = buf[off:off + _RATE]
        for i in range(_RATE // 8):
            state[i % 5][i // 5] ^= int.from_bytes(block[8 * i:8 * i + 8], "little")
        state = _keccak_f(state)
    return b"".join(state[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


# Known-answer check: a wrong round constant or rotation would silently
# produce garbage, and this digest gates whether a response is trusted.
assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
assert keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
