# ◇ CODEX_IMPLEMENT: snapshot CLI
import argparse
import sys

from herg.snapshot import save_snapshot, load_snapshot
from herg.graph_caps.store import CapsuleStore


def main():
    parser = argparse.ArgumentParser("herg snapshot")
    subs = parser.add_subparsers(dest="cmd")
    p_save = subs.add_parser("save")
    p_save.add_argument("path")
    p_load = subs.add_parser("load")
    p_load.add_argument("path")
    args = parser.parse_args()
    if args.cmd == "save":
        try:
            store = load_snapshot(args.path)
        except Exception:
            store = CapsuleStore()
        save_snapshot(store, args.path)
        print(f"Saved {len(store.caps)} capsules")
    elif args.cmd == "load":
        store = load_snapshot(args.path)
        print(f"Loaded {len(store.caps)} capsules")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
