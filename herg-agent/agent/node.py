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
from agent.utils import safe_search
from herg.hvlogfs import HVLogFS          # assuming you expose this
from agent.encoder_ext import encode, prefix
from agent.memory import MemoryCapsule, SelfCapsule, maybe_branch
from herg.backend import cosine
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

NODE_KEY = os.getenv("NODE_KEY")
app = FastAPI()
hvlog = HVLogFS(str(HVLOG_DIR))
index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
id_map = {}         # capsule_id -> (chunk_offset, meta_dict, mu)

SIM_THR = 0.9


class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add(self, cap):
        self.nodes[cap.id_int] = cap

    def add_edge(self, src, dst, route=""):
        self.edges.append((src, dst, route))


self_cap = SelfCapsule()
graph = _Graph()


def _check_key(request: Request) -> None:
    if NODE_KEY and request.headers.get("x-api-key") != NODE_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

# ---------------------------------------------------------------------------

@app.get("/health", tags=["_infra"])
async def health():
    return {"ok": True}

@app.on_event("startup")
async def _load():
    asyncio.create_task(hydrate_prefix(SHARD_KEY))
    await _rebuild()
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
    _check_key(request)
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
    D, I = safe_search(index, vec.astype(np.float32)[None, :], data.get("top_k", 8))
    if I.size == 0:
        return []
    results = []
    for dist, cid in zip(D[0], I[0]):
        if cid == -1:
            continue
        chunk, meta = id_map.get(cid, (None, None))
        results.append({"capsule": int(cid), "dist": float(dist), "meta": meta})
    return results

@app.post("/insert")
async def insert(request: Request):
    _check_key(request)
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
    vec, h = encode(data["seed"])

    # look up nearest capsule
    D, I = safe_search(index, vec.astype(np.float32)[None, :], 1)
    cap = None
    if I.size and I[0][0] != -1:
        cid = int(I[0][0])
        chunk, meta, mu = id_map[cid]
        cap = MemoryCapsule(cid, mu.copy(), meta, meta.get("energy", 1.0))
        if cosine(vec, cap.mu) > SIM_THR:
            cap.update(vec, reward)
            hvlog.append_cap(prefix=pref, cap_id=cid, mu=cap.mu, meta=cap.meta)
            index.remove_ids(np.array([cid], dtype=np.int64))
            index.add_with_ids(cap.mu.astype(np.float32)[None, :], np.array([cid], dtype=np.int64))
            id_map[cid] = ("current_chunk", cap.meta, cap.mu.copy())
        else:
            cap = None

    if cap is None:
        cap_id = h
        meta = {"text": data.get("text", ""), "energy": reward, "ts": time.time()}
        hvlog.append_cap(prefix=pref, cap_id=cap_id, mu=vec, meta=meta)
        index.add_with_ids(vec.astype(np.float32)[None, :], np.array([cap_id], dtype=np.int64))
        id_map[int(cap_id)] = ("current_chunk", meta, vec.astype(np.float32))
        cap = MemoryCapsule(cap_id, vec.copy(), meta)

    maybe_branch(graph, cap, vec, reward)
    self_cap.bump(reward, 0.0)
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
