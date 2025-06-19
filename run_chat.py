# ◇ CODEX_IMPLEMENT: create run_chat.py REPL
import sys
import time
import signal

from herg.graph_caps.store import CapsuleStore
from herg.snapshot import save_snapshot, load_snapshot
from herg import backend as B


def main() -> None:
    try:
        store = load_snapshot("brains/auto.pkl")
    except Exception:
        store = CapsuleStore()

    def handler(sig, frame):
        store.prune()
        save_snapshot(store, "brains/auto.pkl")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    token_count = 0
    for line in sys.stdin:
        tokens = line.strip().split()
        for t in tokens:
            store.spawn(t.encode(), ts=int(time.time()))
            token_count += 1
            if token_count % 50 == 0:
                vecs = [cap.vec for cap in store.caps.values()]
                avg_cos = (
                    sum(B.cosine(a, b) for a, b in zip(vecs, vecs[1:]))
                    / max(len(vecs) - 1, 1)
                )
                print(f"{len(store.caps)} capsules, avg_cos={avg_cos:.2f}")


if __name__ == "__main__":
    main()

