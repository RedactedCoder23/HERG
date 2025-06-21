from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
import random


class BoundError(Exception):
    pass


def find_eta_max(successes: int, trials: int, beta: float = 0.05) -> float:
    """Chernoff bound upper confidence on success rate."""
    if trials == 0:
        return 1.0
    from math import log, sqrt
    p = successes / trials
    eps = sqrt(log(1.0 / beta) / (2 * trials))
    return min(1.0, p + eps)


PARAM_BOUNDS = {
    "radius": (1, 4),
    "block_size": (64, 4096),
    "alpha_u": (0.01, 1.0),
    "alpha_b": (0.01, 1.0),
    "eta": (0.001, 1.0),
    "energy_drain": (0.0, 10.0),
    "lane_split": (1024, 8192),
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
                "lane_split": 512,
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


@dataclass
class BanditTuner(HillClimbTuner):
    """\u03b5-greedy bandit tuner."""

    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    decay_steps: int = 200
    rng: random.Random | None = None
    steps: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rng = self.rng or random.Random()
        self.q: dict[tuple[int, str], float] = {}
        self.prev_state: int | None = None
        self.prev_action: str | None = None
        self.prev_val: float | None = None

    # --------------------------------------------------------------
    def _epsilon(self) -> float:
        frac = min(self.decay_steps, self.steps) / self.decay_steps
        return max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * frac)

    # --------------------------------------------------------------
    def suggest(self, metrics: dict, goal: str, cfg) -> dict:
        updates: dict[str, int | float] = {}
        metric_map = {
            "retention": "retention",
            "throughput": "ingest_rate",
            "latency": "latency_p95",
        }
        val = metrics.get(metric_map.get(goal, "retention"), 0.0)

        state = int(val * 10)
        if self.prev_action is not None and self.prev_state is not None and self.prev_val is not None:
            reward = val - self.prev_val
            if "lane_split" in self.prev_action:
                r_now = find_eta_max(int(val * 1000), 1000)
                r_prev = find_eta_max(int(self.prev_val * 1000), 1000)
                reward = r_now - r_prev
            key = (self.prev_state, self.prev_action)
            old = self.q.get(key, 0.0)
            self.q[key] = 0.8 * old + 0.2 * reward

        actions = []
        for p in ["radius", "eta", "alpha_u", "block_size", "lane_split"]:
            step = self.step[p]
            actions.append(f"{p}+{step}")
            actions.append(f"{p}-{step}")

        def apply(act: str) -> dict:
            param, op = act.split("+") if "+" in act else act.split("-")
            delta = self.step[param]
            if "-" in act:
                delta = -delta
            cur = getattr(cfg, param)
            if param == "lane_split":
                d1 = int(cur[0] + delta)
                lo, hi = PARAM_BOUNDS[param]
                d1 = max(lo, min(hi, d1))
                new = (d1, d1 // 2, d1 // 2)
            else:
                new = cur + delta
                lo, hi = PARAM_BOUNDS[param]
                new = max(lo, min(hi, new))
            if new != cur:
                return {param: new}
            return {}

        # pick action
        eps = self._epsilon()
        best = None
        if self.rng.random() < eps:
            best = self.rng.choice(actions)
        else:
            qvals = {a: self.q.get((state, a), 0.0) for a in actions}
            best = max(qvals, key=qvals.get)

        updates = apply(best)

        if val < 0.97 * (self.last_good_metric or val):
            self.bad.append(val)
        else:
            self.bad.clear()
            self.last_good_metric = val
            self.last_good_cfg = asdict(cfg)

        if len(self.bad) >= 3 and self.last_good_cfg:
            self.bad.clear()
            self.step = {k: v / 2 for k, v in self.step.items()}
            revert = {}
            for k in self.last_good_cfg:
                cur = getattr(cfg, k, None)
                good = self.last_good_cfg[k]
                if cur != good:
                    revert[k] = good
            if revert:
                updates = revert

        self.prev_state = state
        self.prev_action = best
        self.prev_val = val
        self.steps += 1
        return updates
