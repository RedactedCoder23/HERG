import os
from .chunk import HyperChunk, ENTRY_SIZE, ChecksumError
from .parity import xor_chunks


def scrub(path: str) -> None:
    """Verify CRCs in all chunks at path and attempt repair using parity."""
    chunks = [f for f in os.listdir(path) if f.endswith('.chk')]
    chunks.sort()
    data_chunks = [os.path.join(path, c) for c in chunks if not c.endswith('p.chk')]
    parity_chunks = [os.path.join(path, c) for c in chunks if c.endswith('p.chk')]
    if not parity_chunks:
        return
    for d0, d1, d2, p in zip(data_chunks[0::3], data_chunks[1::3], data_chunks[2::3], parity_chunks):
        for fname in [d0, d1, d2]:
            chunk = HyperChunk(fname)
            for i in range(chunk.count):
                off = 64 + i * ENTRY_SIZE
                try:
                    chunk.read(off)
                except ChecksumError:
                    rebuild(fname, d0, d1, d2, p)
                    break
            chunk.close()


def rebuild(target: str, c0: str, c1: str, c2: str, parity: str) -> None:
    """Rebuild missing chunk using XOR parity."""
    with open(parity, 'rb') as f:
        pbytes = f.read()
    with open(c0, 'rb') as f:
        b0 = f.read()
    b1 = b''
    if os.path.exists(c1):
        with open(c1, 'rb') as f:
            b1 = f.read()
    with open(c2, 'rb') as f:
        b2 = f.read()
    missing = xor_chunks(pbytes, b0, b1, b2)
    with open(target, 'wb') as f:
        f.write(missing)

