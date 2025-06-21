# HERG — Hypervector Emergent Reasoning Graph

A lightweight, self-evolving hyper-vector reasoning graph.

### Quick-start

```bash
git clone https://github.com/YourOrg/HERG.git
cd HERG
pip install -e .[dev]          # editable install
herg-run --demo text
```


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

## Quick-Start

Spin up a demo node and router then insert a capsule:

```bash
export NODE_KEY=$(openssl rand -hex 16)
python -m agent.node &
python -m agent.router &
curl -H "x-api-key: $NODE_KEY" http://localhost:9000/health
payload=$(python -m agent.utils prefix '{"seed":"hi","text":"hello","reward":0.1}')
curl -H "x-api-key: $NODE_KEY" -XPOST \
  -d "$payload" \
  http://localhost:8000/insert
payload=$(python -m agent.utils prefix '{"seed":"hi","top_k":1}')
curl -H "x-api-key: $NODE_KEY" -XPOST \
  -d "$payload" \
  http://localhost:8000/query
```

Queries hit the router on port 8000 as well.
Use the same `x-api-key` header for `/health`, `/insert`, and `/query`.
Inserts may include `{"seed":"abc", "reward":1.0}`.
