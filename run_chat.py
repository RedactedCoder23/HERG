# ◇ CODEX_IMPLEMENT: create run_chat.py REPL
import sys
import time
import signal

from herg.graph_caps import CapsuleStore
from herg.snapshot import save_snapshot, load_snapshot
from herg import backend as B


def main() -> None:
    try:
        store = load_snapshot("brains/auto.pkl")
    except Exception:
        store = CapsuleStore()

    stop = False

    def handle_sigint(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    token_count = 0
    cos_sum = 0.0
    prev_vec = None

    try:
        for line in sys.stdin:
            if stop:
                break
            tokens = line.strip().split()
            for tok in tokens:
                cap = store.spawn(tok.encode(), ts=int(time.time()))
                if prev_vec is not None:
                    cos_sum += B.cosine(cap.vec, prev_vec)
                prev_vec = cap.vec
                token_count += 1
                if token_count % 50 == 0:
                    avg_cos = cos_sum / max(1, token_count - 1)
                    print(f"{len(store.caps)} capsules, avg_cos={avg_cos:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        store.prune()
        save_snapshot(store, "brains/auto.pkl")


if __name__ == "__main__":
    main()

