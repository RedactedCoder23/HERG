# ◇ CODEX_IMPLEMENT: implement herg/viz.py visualization helper
from herg.graph_caps import CapsuleStore


def viz_dot(store: CapsuleStore, last_n: int = 500) -> str:
    nodes = list(store.caps.keys())[-last_n:]
    edges = [
        (src, dst, wt)
        for src in nodes
        for dst, wt in zip(*store.edges.neighbors(src))
    ]
    dot = ["digraph G {"]
    for n in nodes:
        dot.append(f'  "{n}";')
    for s, d, _ in edges:
        dot.append(f'  "{s}" -> "{d}";')
    dot.append("}")
    return "\n".join(dot)

