import numpy as np

_POPCNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

_QUAD_DIFF = np.zeros((256, 256), dtype=np.uint8)
for a in range(256):
    for b in range(256):
        d = 0
        for i in range(4):
            d += (((a >> (2 * i)) & 3) != ((b >> (2 * i)) & 3))
        _QUAD_DIFF[a, b] = d

_NIBBLE_DIFF = np.zeros((256, 256), dtype=np.uint8)
for a in range(256):
    for b in range(256):
        d = 0
        for i in range(2):
            d += (((a >> (4 * i)) & 0xF) != ((b >> (4 * i)) & 0xF))
        _NIBBLE_DIFF[a, b] = d


def hybrid_hamming(a: np.ndarray, b: np.ndarray, lane_split=(4096, 2048, 2048)) -> int:
    """Return hybrid Hamming distance between two packed vectors."""
    a = a.astype(np.uint8, copy=False)
    b = b.astype(np.uint8, copy=False)
    d1, d2, d3 = lane_split
    idx = 0
    dist = 0

    b1 = d1 // 8
    if b1:
        x = np.bitwise_xor(a[idx:idx+b1], b[idx:idx+b1])
        dist += int(_POPCNT[x].sum())
        idx += b1

    b2 = (2 * d2) // 8
    if b2:
        dist += int(_QUAD_DIFF[a[idx:idx+b2], b[idx:idx+b2]].sum())
        idx += b2

    b3 = (4 * d3) // 8
    if b3:
        dist += int(_NIBBLE_DIFF[a[idx:idx+b3], b[idx:idx+b3]].sum())

    return dist
