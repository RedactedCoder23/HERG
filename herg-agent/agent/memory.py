from dataclasses import dataclass, field
import os, time
import numpy as np
from herg.backend import cosine

DECAY = 0.99
LR = 0.25
NEW_THR = 0.12

@dataclass
class MemoryCapsule:
    id_int: int
    mu: np.ndarray
    meta: dict
    energy: float = 1.0

    def update(self, vec: np.ndarray, reward: float) -> None:
        self.mu = (1 - LR) * self.mu + LR * vec
        self.energy = self.energy * DECAY + 0.01 * reward
        self.meta["ts"] = time.time()


@dataclass
class SelfCapsule:
    mu: np.ndarray = field(default_factory=lambda: np.zeros(2048, np.float32))
    step: int = 0
    mean_reward: float = 0.0
    entropy: float = 0.0

    def bump(self, reward: float, routing_entropy: float) -> None:
        self.step += 1
        self.mean_reward = 0.99 * self.mean_reward + 0.01 * reward
        self.entropy = routing_entropy
        self.mu[:3] = [self.step & 0xFF, self.mean_reward, self.entropy]


def maybe_branch(graph, parent_cap: MemoryCapsule, vec: np.ndarray, reward: float):
    if reward < -0.2 or cosine(vec, parent_cap.mu) < NEW_THR:
        child_id = int.from_bytes(os.urandom(4), "big")
        child = MemoryCapsule(child_id, vec.copy(), {"energy": 1.0})
        if hasattr(graph, "add"):
            graph.add(child)
        if hasattr(graph, "add_edge"):
            graph.add_edge(parent_cap.id_int, child_id, route="branch")
        return child
    return None

