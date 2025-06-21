"""
One shard:  • query / update API
            • local Faiss IVF+PQ index
            • background tasks: push_closed_chunks, hydrate_prefix
"""

import asyncio, os, time, logging
from pathlib import Path
import numpy as np
import faiss, orjson, uvicorn
from fastapi import FastAPI, HTTPException, Request
from agent.utils import safe_search, cosine
from herg.faiss_wrapper import make_index
from herg.hvlogfs import HVLogFS          # assuming you expose this
from agent.encoder_ext import encode, prefix
from agent.memory import MemoryCapsule, SelfCapsule, maybe_branch
if os.getenv("S3_BUCKET"):
    from agent.replicator import push_closed_chunks, hydrate_prefix
else:  # disable S3 sync in dev without credentials
    async def push_closed_chunks():
        pass

    async def hydrate_prefix(_):
        pass

log = logging.getLogger("node")

HVLOG_DIR = Path(os.getenv("HVLOG_DIR", "/data/hvlog"))
SHARD_KEY  = os.getenv("SHARD_KEY")       # e.g. "7a"
if SHARD_KEY is None:
    raise SystemExit("must set SHARD_KEY env")

DIM = 2048
NLIST = 4096               # IVF coarse bins
M = 64                     # PQ code size

API_KEY = os.getenv("NODE_KEY")
app = FastAPI()
hvlog = HVLogFS(str(HVLOG_DIR))
index = faiss.IndexIDMap(make_index(DIM))
id_map = {}         # capsule_id -> (chunk_offset, meta_dict, mu)

SIM_THR = 0.88


class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._next_id = 1

    def add(self, cap):
        self.nodes[cap.id_int] = cap

    def add_edge(self, src, dst, route=""):
        self.edges.append((src, dst, route))

    def __getitem__(self, key):
        return self.nodes[key]

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def get_or_create(self, key, cls):
        cap = self.nodes.get(key)
        if cap is None:
            cap = cls() if callable(cls) else cls
            self.nodes[key] = cap
        return cap


self_cap = SelfCapsule()
graph = _Graph()

def _auth(request: Request) -> None:
    """Check API key header if API_KEY is set."""
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="unauth")

# ---------------------------------------------------------------------------

@app.get("/health", tags=["_infra"])
async def health():
    return {"ok": True}

@app.on_event("startup")
async def _load():
    asyncio.create_task(hydrate_prefix(SHARD_KEY))
    await _rebuild()
    # load persisted SELF capsule if present
    global self_cap
    for cap in hvlog.iter_capsules(prefix=SHARD_KEY):
        if cap.id_int == 0:
            self_cap.mu = cap.mu.astype(np.float32)
            self_cap.step = int(cap.meta.get("step", 0))
            self_cap.mean_reward = float(cap.meta.get("mean_reward", 0.0))
            self_cap.entropy = float(cap.meta.get("entropy", 0.0))
            break
    asyncio.create_task(_rebuild_periodic())

async def _rebuild():
    vecs, ids = [], []
    for cap in hvlog.iter_capsules(prefix=SHARD_KEY):
        if not getattr(cap, "active", True):
            continue
        vecs.append(cap.mu.astype(np.float32))
        ids.append(cap.id_int)
        id_map[cap.id_int] = (cap.chunk, cap.meta, cap.mu.astype(np.float32))
    if vecs:
        xb = np.stack(vecs)
        index.add_with_ids(faiss.vector_to_array(xb).reshape(-1, DIM), np.array(ids))
    log.info("Indexed %d vectors", len(vecs))

async def _rebuild_periodic():
    while True:
        await asyncio.sleep(300)
        await _rebuild()

# ---------------------------------------------------------------------------

@app.post("/query")
async def query(request: Request):
    _auth(request)
    req = await request.body()
    """
    Body = b"{prefix_hex}{json_payload}"
    json = { "seed": str, "top_k": int }
    """
    body = req.decode()
    pref = body[:2]
    if pref != SHARD_KEY:
        raise HTTPException(400, "wrong shard")
    data = orjson.loads(body[2:])
    vec, _ = encode(data["seed"])
    xb = np.asarray([vec], np.float32)
    D, I = safe_search(index, xb, data.get("top_k", 8))
    if I.size == 0:
        return []
    results = []
    for dist, cid in zip(D[0], I[0]):
        if cid == -1:
            continue
        cap = graph.nodes.get(cid)
        meta = cap.meta if cap else {}
        results.append({"capsule": int(cid), "dist": float(dist), "meta": meta})
    return results

@app.post("/insert")
async def insert(request: Request):
    _auth(request)
    req = await request.body()
    """
    Body = b"{prefix_hex}{json}"
    json = { "seed": str, "text": str, "reward": float }
    """
    body = req.decode()
    pref = body[:2]
    if pref != SHARD_KEY:
        raise HTTPException(400, "wrong shard")
    data = orjson.loads(body[2:])
    reward = float(data.get("reward", 0.0))
    text = data.get("text", "")
    vec, _ = encode(data["seed"])

    xb = np.asarray([vec], np.float32)
    D, I = safe_search(index, xb, 1)
    if I.size and D[0][0] > SIM_THR:
        cap = graph[I[0][0]]
        cap.update(vec, reward)
    else:
        cap = MemoryCapsule(graph.new_id(), vec.copy(), {"text": text})
        graph.add(cap)
        hvlog.append_cap(SHARD_KEY, cap.id_int, cap.mu, cap.meta)
        index.add_with_ids(xb, np.array([cap.id_int], np.int64))

    child = maybe_branch(graph, cap, vec, reward)
    self_cap = graph.get_or_create("SELF", SelfCapsule)
    self_cap.bump(reward)
    hvlog.append_cap(SHARD_KEY, 0, self_cap.mu, {
        "step": self_cap.step,
        "mean_reward": self_cap.mean_reward,
        "entropy": self_cap.entropy,
    })
    return {"ok": True}

# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _bg_tasks():
    asyncio.create_task(_push_loop())

async def _push_loop():
    while True:
        await push_closed_chunks()
        await asyncio.sleep(30)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
