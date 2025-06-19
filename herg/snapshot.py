# ◇ CODEX_IMPLEMENT: implement snapshot CLI helpers
import pickle
from pathlib import Path
from herg.graph_caps import CapsuleStore


def save_snapshot(store: CapsuleStore, path: str) -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "caps": list(store.caps.values()),
        "edges": (store.edges.src, store.edges.dst, store.edges.wts),
    }
    with open(p, "wb") as f:
        pickle.dump(data, f)


def load_snapshot(path: str) -> CapsuleStore:
    with open(path, "rb") as f:
        data = pickle.load(f)
    store = CapsuleStore()
    for cap in data.get("caps", []):
        store.caps[cap.id] = cap
    src, dst, wts = data.get("edges", ([], [], []))
    store.edges.src, store.edges.dst, store.edges.wts = src, dst, wts
    return store

