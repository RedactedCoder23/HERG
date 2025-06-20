from typing import Iterable


def xor_chunks(*chunks: Iterable[bytes]) -> bytes:
    """Return XOR parity of given byte-like chunks."""
    if not chunks:
        return b""
    it = iter(chunks)
    parity = bytearray(next(it))
    for c in it:
        for i, b in enumerate(c):
            parity[i] ^= b
    return bytes(parity)
