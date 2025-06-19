
# ❑ CODEX: replace seed_to_hyper with the deterministic algorithm below.
#
# ALGORITHM (deterministic, collision-free, zero RNG state):
# 1. Accept `seed: bytes` (32-byte SHA-256 digest).
# 2. Interpret it as *big-endian* unsigned int `S`.
# 3. For each vector index i = 0 … dim-1:
#      a. Compute 64-bit keyed hash:
#           h_i = (1103515245 * (S ^ i) + 12345) mod 2**31
#      b. Bit-parity rule:
#           bit = (h_i >> 15) & 1
#           value = +1 if bit == 1 else -1
# 4. Pack the ±1 values into *signed* int8 tensor of shape (dim,).
# 5. If `ternary` flag is True: zero every element whose index i
#    is not divisible by 3  ➜ sparse 1/3 density.
# 6. Return `torch.IntTensor` on the requested `device`.
#
# The procedure is branchless, GPU-friendly, reproducible across
# Python, C++, and CUDA implementations.

"""Prototype hypervector encoder."""

import hashlib

import torch

ENC_DIM = 2048


def seed_to_hyper(seed: bytes, dim: int = ENC_DIM, ternary: bool = False, device: str = "cpu") -> torch.IntTensor:
    """Deterministically encode ``seed`` into a bipolar/ternary hypervector."""

    if not isinstance(seed, (bytes, bytearray)):
        raise TypeError("seed must be bytes")
    if len(seed) != 32:
        # allow hashing of arbitrary inputs for convenience
        seed = hashlib.sha256(seed).digest()

    S = int.from_bytes(seed, "big")

    vec = torch.empty(dim, dtype=torch.int8, device=device)
    for i in range(dim):
        h_i = (1103515245 * (S ^ i) + 12345) % (2 ** 31)
        bit = (h_i >> 15) & 1
        val = 1 if bit == 1 else -1
        if ternary and i % 3 != 0:
            val = 0
        vec[i] = val

    return vec


# ❑ CODEX: in forward()    << explain where it is if needed
# 1. For each incoming token `tok`:
#     a. seed = sha256(tok.encode()).digest()
#     b. cap = store.spawn(seed)     # returns existing or new capsule
#     c. messages = edge_table.neighbors(cap.id)
#     d. capsule_vec = gnn_step(cap.vec, messages)  # simple W*vec + GELU
# 2. Concat `[hidden_state ‖ capsule_vec]` → project to logits.
# 3. After loss.backward(): store.update(cap.id, Δvec, ts_now)

