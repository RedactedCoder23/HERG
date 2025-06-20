import argparse
import sys
import time
from pathlib import Path
from contextlib import nullcontext

from herg.snapshot import save_snapshot, load_snapshot
from herg.graph_caps.store import CapsuleStore
from . import config
from .prof import Prof
from .auto.metrics import MetricStore


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
    p_run.add_argument('--ticks', type=int, default=1000)
    p_run.add_argument('--metrics', type=int, default=0,
                        help='print metrics every N seconds')

    p_bench = sub.add_parser('bench')
    bench_sub = p_bench.add_subparsers(dest='benchcmd')
    p_ing = bench_sub.add_parser('ingest')
    p_ing.add_argument('--n', type=int, default=100000)

    p_hv = sub.add_parser('hvlog')
    hv_sub = p_hv.add_subparsers(dest='hvcmd')
    p_mount = hv_sub.add_parser('mount')
    p_mount.add_argument('path')

    p_auto = sub.add_parser('auto-run')
    p_auto.add_argument('--nvec', type=int, default=20000)
    p_auto.add_argument('--ticks', type=int, default=1000)
    p_auto.add_argument('--radius', type=int, default=cfg.radius)
    p_auto.add_argument('--tune-interval', type=int, default=30)
    p_auto.add_argument('--goal', default='retention', choices=['throughput','retention','latency'])
    p_auto.add_argument('--profile', action='store_true')

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
        config.atomic_save(cfg)
        ctx = Prof() if args.profile else nullcontext()
        metrics = MetricStore()
        next_print = time.time() + args.metrics if args.metrics else None
        with ctx:
            from herg.graph_caps.step import k_radius_pass, adf_update
            from herg.graph_caps.gossip import gap_junction_gossip
            from herg.graph_caps.prune import sticky_pool_prune
            store = CapsuleStore()
            prev_mu = None
            for tick in range(args.ticks):
                cap = store.spawn(str(tick).encode())
                if prev_mu is not None:
                    import numpy as np
                    mu_d = float(np.linalg.norm(cap.mu - prev_mu))
                else:
                    mu_d = 0.0
                prev_mu = cap.mu.copy()
                k_radius_pass(store, cfg.radius)
                if tick % cfg.gossip_every == 0:
                    for c in list(store.caps.values()):
                        adf_update(c, c.fast, 1.0, cfg.eta)
                    gap_junction_gossip(store)
                    sticky_pool_prune(store)
                metrics.update(
                    ingest_bytes=1028,
                    query_latency=None,
                    retention_value=0.5 + 0.05 * cfg.radius,
                    mu_drift=mu_d,
                )
                if next_print and time.time() >= next_print:
                    print(metrics.snapshot())
                    next_print = time.time() + args.metrics
                time.sleep(0.001)
            print('Simulation complete')
    elif args.cmd == 'bench' and args.benchcmd == 'ingest':
        from scripts.bench_ingest import run_bench
        run_bench(args.n)
    elif args.cmd == 'hvlog' and args.hvcmd == 'mount':
        from herg.storage.hvlogfs.fuse_mount import mount
        mount(args.path)
    elif args.cmd == 'auto-run':
        store = CapsuleStore()
        metrics = MetricStore()
        from herg.auto import daemon
        tuner_thread = daemon.start(metrics, cfg, args.tune_interval, args.goal)
        ctx = Prof() if args.profile else nullcontext()
        with ctx:
            start = None
            for tick in range(args.ticks):
                store.spawn(str(tick).encode())
                target = 0.5 + 0.05 * cfg.radius
                metrics.update(ingest_bytes=1024, retention_value=target)
                if start is None:
                    start = metrics.retention
                time.sleep(0.001)
            metrics.running = False
            tuner_thread.join()
            print(
                f"{args.goal} improved from {start:.2f} -> {metrics.retention:.2f} after {metrics.adjustments} adjustments"
            )
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
    main()
