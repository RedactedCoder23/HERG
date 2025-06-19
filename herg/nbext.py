"""Jupyter Notebook extension for HERG magics."""

from IPython.core.magic import Magics, magics_class, line_magic
from herg.graph_caps.store import CapsuleStore
from herg.snapshot import save_snapshot, load_snapshot
from herg.viz import viz_dot
from IPython.display import SVG, display
import argparse


@magics_class
class HergMagics(Magics):
    """IPython magics for HERG."""

    @line_magic
    def herg_viz(self, line: str = ""):
        parser = argparse.ArgumentParser(prog="%herg_viz", add_help=False)
        parser.add_argument("last_n", nargs="?", type=int, default=500)
        try:
            args = parser.parse_args(line.split())
        except SystemExit:
            return "Usage: %herg_viz [last_n]"
        store = CapsuleStore()
        dot = viz_dot(store, last_n=args.last_n)
        try:
            display(SVG(data=dot))
        except Exception:
            pass
        return dot

    @line_magic
    def herg_snapshot(self, line: str = ""):
        parser = argparse.ArgumentParser(prog="%herg_snapshot", add_help=False)
        sub = parser.add_subparsers(dest="cmd")
        p_save = sub.add_parser("save")
        p_save.add_argument("path")
        p_load = sub.add_parser("load")
        p_load.add_argument("path")
        try:
            args = parser.parse_args(line.split())
        except SystemExit:
            return "Usage: %herg_snapshot <save|load> <path>"

        if args.cmd == "save":
            store = CapsuleStore()
            save_snapshot(store, args.path)
            return f"Saved {len(store.caps)} capsules"
        elif args.cmd == "load":
            store = load_snapshot(args.path)
            msg = f"Loaded {len(store.caps)} capsules"
            print(msg)
            return msg
        else:
            return "Usage: %herg_snapshot <save|load> <path>"


def load_ipython_extension(ip):
    """Register HERG magics."""
    ip.register_magics(HergMagics)
