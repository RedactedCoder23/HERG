1 Drop-in spec for deterministic hyper-vector encoding
Add this verbatim at the top of src/encoder.py:

```python
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
#    is not divisible by 3  ⟩ sparse 1/3 density.
# 6. Return `torch.IntTensor` on the requested `device`.
#
# The procedure is branchless, GPU-friendly, reproducible across
# Python, C++, and CUDA implementations.
```

2 Capsule & edge scaffolding
Insert in src/graph_caps/__init__.py (create the folder):

```python
# ❑ CODEX: implement:
#
#   class Capsule:
#       id: int                    # uint64
#       vec: torch.IntTensor       # int8, shape (DIM,)
#       last_used: int             # Unix epoch seconds
#       edge_ids: list[int]        # neighbour ids
#       edge_wts: list[int]        # signed int16, same len as edge_ids
#
#   class EdgeCOO:
#       add_edge(src:int, dst:int, w:int16) -> None
#       neighbors(id:int) -> tuple[list[int], list[int]]
#       prune_edges(threshold:int16) -> None
#
# Requirements:
#   • store edge arrays in `torch.int16` tensors (dense chunk) but expose
#     COO-style API.
#   • max 5 hot edges per capsule; truncate oldest if limit exceeded.
```

3 CapsuleStore with sticky-pool policy
Append to src/graph_caps/store.py (new file):

```python
# ❑ CODEX: create CapsuleStore
#
# API:
#   spawn(seed:bytes) -> Capsule
#   read(id:int) -> Capsule | None
#   update(id:int, delta_vec:torch.IntTensor, ts:int) -> None
#   prune(now:int) -> None
#
# Sticky-pool rules:
#   • VRAM budget = 100_000 capsules (≈100 MB at 1 k dim / int8).
#   • LRU eviction, *unless* capsule has cosine similarity > 0.95
#     with > 1 active neighbour  ⟩ mark “sticky” (immune for 10 min).
#   • Evicted capsules are serialised to `capsules.sqlite` using
#     `pickle.dumps(capsule)` under key = capsule.id.
#   • prune(now) runs every 1 000 calls to update().
```

4 Forward-pass integration hook
Append near the end of src/encoder.py (or wherever the transformer lives):

```python
# ❑ CODEX: in forward()    << explain where it is if needed
# 1. For each incoming token `tok`:
#     a. seed = sha256(tok.encode()).digest()
#     b. cap = store.spawn(seed)     # returns existing or new capsule
#     c. messages = edge_table.neighbors(cap.id)
#     d. capsule_vec = gnn_step(cap.vec, messages)  # simple W*vec + GELU
# 2. Concat `[hidden_state ‖ capsule_vec]` → project to logits.
# 3. After loss.backward(): store.update(cap.id, Δvec, ts_now)
```

5 codex_todo_runner implementation details
Replace stub in .github/scripts/codex_todo_runner.py:

```python
# ❑ CODEX: full script
#
# 0. Read $GITHUB_SHA, repo URL, and OPENAI_API_KEY from env.
# 1. Scan committed files for lines matching r'^# ❑ CODEX: (.*)$'.
# 2. For each match, accumulate into one mega-prompt:
#       "### File:{path}\n{preceding 20 lines}\n### TODO:{todo_line}"
# 3. Call OpenAI chat/completions, model=gpt-4o-codex-preview,
#    temperature=0, max_tokens=600.
# 4. Parse ````diff ... ````, write patched file(s) on a new branch
#    `codex-auto/<timestamp>`.
# 5. Use `gh pr create --fill` to open draft PR.
# 6. Exit 0 if PR opened, else non-zero to mark workflow failed.
#
#   (Include robust try/except and rate-limit handling.)
```

6 Minimal unit tests
Create tests/test_encoder.py:

```python
# ❑ CODEX: write tests
#   • test_determinism(): same seed → identical vector.
#   • test_bipolar_values(): all entries ∈ {-1,+1} (or 0 if ternary).
#   • test_ternary_sparsity(): with --ternary flag density ≈ 1/3 ±2 %.
```

How to use
Commit this file (DOCS/TODO_ROADMAP.md) and copy the TODO blocks into their target .py files.

`git add . && git commit -m "feat: seed TODO blocks for Codex"` → push.

The Codex Action will see “TODO” in the diff, run, and open PR #1 implementing everything above.
