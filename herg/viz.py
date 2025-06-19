# ◇ CODEX_IMPLEMENT: implement herg/viz.py visualization helper
from herg.graph_caps.store import CapsuleStore


def viz_dot(store: CapsuleStore, last_n: int = 500) -> str:
    nodes = list(store.caps.keys())[-last_n:]
    dot_lines = ["digraph G {"]
    for node_id in nodes:
        dot_lines.append(f'  "{node_id}";')
        dsts, _ = store.edges.neighbors(node_id)
        for dst in dsts:
            dot_lines.append(f'  "{node_id}" -> "{dst}";')
    dot_lines.append("}")
    return "\n".join(dot_lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("herg viz")
    parser.add_argument("--last", type=int, default=500)
    args = parser.parse_args()

    from herg.graph_caps.store import CapsuleStore

    store = CapsuleStore()
    print(viz_dot(store, last_n=args.last))

