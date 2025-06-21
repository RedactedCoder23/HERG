import argparse, sys
import numpy as np
from pathlib import Path

AGENT_PATH = Path(__file__).resolve().parents[1] / "herg-agent"
if AGENT_PATH.exists():
    sys.path.append(str(AGENT_PATH))
from herg.memory import MemoryCapsule, maybe_branch
from herg.storage.hvlogfs import HVLogFS
from herg.faiss_wrapper import HybridIndex
from agent.encoder_ext import expand_seed, prefix, LANE_SPLIT
from agent.utils import safe_search, cosine

class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._next = 1

    def add(self, cap):
        self.nodes[cap.id_int] = cap

    def add_edge(self, src, dst, route=""):
        self.edges.append((src, dst, route))

    def __getitem__(self, key):
        return self.nodes[key]

    def new_id(self):
        nid = self._next
        self._next += 1
        return nid
from herg import config

def demo_text(seeds):
    print("== HERG text demo ==")

    hvlog = HVLogFS("/tmp/hvlog")
    index = HybridIndex(LANE_SPLIT)
    id_map = []
    graph = _Graph()

    if not seeds:
        seeds = ["hello"]

    for seed in seeds:
        vec, h = expand_seed(seed)
        D, I = safe_search(index, vec, 1)
        if I.size:
            cid = id_map[int(I[0][0])]
            cap = graph[cid]
            cos = cosine(cap.mu, vec.astype(np.float32))
            if cos > 0.88:
                cap.update(vec.astype(np.float32))
                hvlog.append_cap(prefix(h), cid, cap.mu, cap.meta)
                maybe_branch(graph, cap, vec.astype(np.float32), 0.0)
                print(f"Queried 1 vector, best match cos={cos:.2f}")
                continue

        cid = graph.new_id()
        cap = MemoryCapsule(cid, vec.astype(np.float32), {})
        graph.add(cap)
        hvlog.append_cap(prefix(h), cid, cap.mu, cap.meta)
        index.add(vec)
        id_map.append(cid)
        print(f"Inserted 1 vector, total capsules: {len(graph.nodes)}")

def main(argv=None):
    cfg = config.load()
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["text"], default="text")
    parser.add_argument("--seed", action="append", default=[],
                        help="seed string; may be repeated")
    parser.add_argument("--alpha", default=None,
                        help="comma-separated floats for kernel scale")
    parser.add_argument("--kernel", choices=["separable", "radial"],
                        default=None)
    args = parser.parse_args(argv)

    if args.alpha is not None:
        vals = [float(v) for v in args.alpha.split(',')] if ',' in args.alpha else float(args.alpha)
        cfg.kernel_alpha = vals
    if args.kernel is not None:
        cfg.kernel_mode = args.kernel
    config.atomic_save(cfg)

    if args.demo == "text":
        demo_text(args.seed)

if __name__ == "__main__":
    main()
