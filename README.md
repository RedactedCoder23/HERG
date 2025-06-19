# HERG — Hypervector Emergent Reasoning Graph

A lightweight, self-evolving hyper-vector reasoning graph.

**How Codex/GPT fits in →** see [`HERG_CODEx_GUIDE.md`](HERG_CODEx_GUIDE.md).

Experimental engine that fuses
* deterministic hypervector encodings,
* persistent “capsule” nodes,
* a sparse capsule-to-capsule graph, and
* a lightweight GNN message pass

…to enable continual, branching reasoning on commodity GPUs.

See `DOCS/capsule_graph.md` for the design spec.

## Backends

| Backend | Device | Env var |
|---------|--------|---------|
| NumPy   | CPU    | *(default)* |
| Torch   | GPU/CPU| `HERG_BACKEND=torch` |
| CuPy    | GPU    | `HERG_BACKEND=cupy` |

Enable GPU by exporting the desired `HERG_BACKEND` before running. Capsule updates follow the BHRE low-rank ADF math.
