from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque


class BoundError(Exception):
    pass


PARAM_BOUNDS = {
    "radius": (1, 4),
    "block_size": (64, 4096),
    "alpha_u": (0.01, 1.0),
    "alpha_b": (0.01, 1.0),
    "eta": (0.001, 1.0),
    "energy_drain": (0.0, 10.0),
}


@dataclass
class HillClimbTuner:
    """Heuristic tuner with simple roll-back."""

    step: dict[str, float] | None = None

    def __post_init__(self):
        if self.step is None:
            self.step = {
                "radius": 1,
                "block_size": 64,
                "alpha_u": 0.05,
                "alpha_b": 0.05,
                "eta": 0.01,
                "energy_drain": 0.1,
            }
        self.bad: deque[float] = deque(maxlen=3)
        self.last_good_metric: float | None = None
        self.last_good_cfg: dict | None = None

    def suggest(self, metrics: dict, goal: str, cfg) -> dict:
        updates: dict[str, int | float] = {}
        metric_map = {
            "retention": "retention",
            "throughput": "ingest_rate",
            "latency": "latency_p95",
        }
        val = metrics.get(metric_map.get(goal, "retention"), 0.0)
        if self.last_good_metric is None:
            self.last_good_metric = val
            self.last_good_cfg = asdict(cfg)

        if val < self.last_good_metric:
            self.bad.append(val)
        else:
            self.bad.clear()
            self.last_good_metric = val
            self.last_good_cfg = asdict(cfg)

        if len(self.bad) >= 3:
            self.bad.clear()
            self.step = {k: v / 2 for k, v in self.step.items()}
            revert = {}
            for k in self.last_good_cfg:
                cur = getattr(cfg, k, None)
                good = self.last_good_cfg[k]
                if cur != good:
                    revert[k] = good
            if revert:
                return revert

        if goal == "retention" and val < 0.70:
            new = min(cfg.radius + self.step["radius"], PARAM_BOUNDS["radius"][1])
            if new != cfg.radius:
                updates["radius"] = new
        elif goal == "throughput" and val < 1.0:
            new = min(cfg.block_size + self.step["block_size"], PARAM_BOUNDS["block_size"][1])
            if new != cfg.block_size:
                updates["block_size"] = new
        elif goal == "latency" and val > 0.05:
            new = max(cfg.block_size - self.step["block_size"], PARAM_BOUNDS["block_size"][0])
            if new != cfg.block_size:
                updates["block_size"] = new
        return updates
