from __future__ import annotations

import threading
import time
import logging
import json
from pathlib import Path

from .metrics import MetricStore
from .tuner import HillClimbTuner, BanditTuner
from .. import config

logger = logging.getLogger(__name__)


LOG_PATH = Path.home() / ".cache" / "herg" / "autotune.log"


def start(store: MetricStore, cfg: config.Config, interval: int, goal: str) -> threading.Thread:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if getattr(cfg, 'tuner', 'hill') == 'bandit':
        tuner = BanditTuner()
    else:
        tuner = HillClimbTuner()

    def loop() -> None:
        while getattr(store, "running", True):
            time.sleep(interval)
            snap = store.snapshot()
            try:
                delta = tuner.suggest(snap, goal, cfg)
                if delta:
                    cfg.apply(delta)
                    config.atomic_save(cfg)
                    store.adjustments += 1
                    with open(LOG_PATH, "a") as f:
                        json.dump({
                            "timestamp": int(time.time()),
                            "metrics": snap,
                            "delta": delta,
                        }, f)
                        f.write("\n")
                    logger.info("Auto-tune applied: %s", delta)
            except Exception:  # never crash main loop
                logger.exception("auto-tune failure")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
