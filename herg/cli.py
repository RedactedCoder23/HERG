import argparse
import sys
from pathlib import Path

from herg.snapshot import save_snapshot, load_snapshot
from herg.graph_caps.store import CapsuleStore
from . import config
from .prof import Prof


def main() -> None:
    cfg = config.load()
    parser = argparse.ArgumentParser(prog='herg')
    sub = parser.add_subparsers(dest='cmd')

    p_run = sub.add_parser('run-sim')
    p_run.add_argument('--radius', type=int, default=cfg.radius)
    p_run.add_argument('--alpha-u', type=float, default=cfg.alpha_u)
    p_run.add_argument('--alpha-b', type=float, default=cfg.alpha_b)
    p_run.add_argument('--eta', type=float, default=cfg.eta)
    p_run.add_argument('--block-size', type=int, default=cfg.block_size)
    p_run.add_argument('--backend', default=cfg.backend)
    p_run.add_argument('--scrub-interval', type=int, default=cfg.scrub_interval)
    p_run.add_argument('--gossip-every', type=int, default=cfg.gossip_every)
    p_run.add_argument('--profile', action='store_true')

    p_bench = sub.add_parser('bench')
    bench_sub = p_bench.add_subparsers(dest='benchcmd')
    p_ing = bench_sub.add_parser('ingest')
    p_ing.add_argument('--n', type=int, default=100000)

    p_hv = sub.add_parser('hvlog')
    hv_sub = p_hv.add_subparsers(dest='hvcmd')
    p_mount = hv_sub.add_parser('mount')
    p_mount.add_argument('path')

    p_save = sub.add_parser('save')
    p_save.add_argument('path')
    p_load = sub.add_parser('load')
    p_load.add_argument('path')

    args = parser.parse_args()

    if args.cmd == 'run-sim':
        cfg.radius = args.radius
        cfg.alpha_u = args.alpha_u
        cfg.alpha_b = args.alpha_b
        cfg.eta = args.eta
        cfg.block_size = args.block_size
        cfg.backend = args.backend
        cfg.scrub_interval = args.scrub_interval
        cfg.gossip_every = args.gossip_every
        config.save(cfg)
        ctx = Prof() if args.profile else nullcontext()
        with ctx:
            print('Simulation config updated')
    elif args.cmd == 'bench' and args.benchcmd == 'ingest':
        from scripts.bench_ingest import run_bench
        run_bench(args.n)
    elif args.cmd == 'hvlog' and args.hvcmd == 'mount':
        from herg.storage.hvlogfs.fuse_mount import mount
        mount(args.path)
    elif args.cmd == 'save':
        try:
            store = load_snapshot(args.path)
        except Exception:
            store = CapsuleStore()
        save_snapshot(store, args.path)
        print(f"Saved {len(store.caps)} capsules")
    elif args.cmd == 'load':
        store = load_snapshot(args.path)
        print(f"Loaded {len(store.caps)} capsules")
    else:
        parser.print_help()


if __name__ == '__main__':
    from contextlib import nullcontext
    main()
