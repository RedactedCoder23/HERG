[![CI](https://github.com/RedactedCoder23/HERG/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RedactedCoder23/HERG/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/RedactedCoder23/HERG/branch/main/graph/badge.svg)](https://codecov.io/gh/RedactedCoder23/HERG)
[![PyPI version](https://img.shields.io/pypi/v/herg.svg)](https://pypi.org/project/herg/)
# HERG — Hypervector Emergent Reasoning Graph

A lightweight, self-evolving hyper-vector reasoning graph.

### Quick-start

```bash
git clone https://github.com/RedactedCoder23/HERG.git
cd HERG
pip install -e .[dev]          # editable install
# try a quick text demo with a few seeds
herg-run --demo text --seed foo --seed bar --seed baz
# Faiss is optional; without it some demo features may raise ImportError.
# If `pip install faiss-cpu` fails, try `conda install -c conda-forge faiss-cpu`
# or set `USE_FLAT=1` to fall back to a slower index.
```

### Code Coverage

Coverage reports are published to [Codecov](https://codecov.io/gh/RedactedCoder23/HERG) for each commit.

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

Licensed under the MIT License—see [LICENSE](LICENSE) for details.

## Releases

See the [GitHub releases](https://github.com/RedactedCoder23/HERG/releases) page for changelogs.
