from dataclasses import dataclass, field
from typing import Any, List
from herg import backend as B


@dataclass(slots=True)
class Capsule:
    """Light-weight capsule node with fast and slow state."""
    id: int
    fast: Any  # int8 tensor shape (D,)
    mu: Any    # float32 tensor shape (D,)
    L: Any     # low-rank perturbation (r, D)
    edges: List[int] = field(default_factory=list)
    energy: float = 1.0

    # legacy aliases
    @property
    def vec(self):
        return self.fast

    @vec.setter
    def vec(self, value):
        self.fast = value

    @property
    def mean(self):
        return self.mu

    @mean.setter
    def mean(self, value):
        self.mu = value

    # --------------------------------------------------------------
    def to(self, device: str | None = None) -> None:
        self.fast = B.tensor(B.as_numpy(self.fast), dtype=B.as_numpy(self.fast).dtype, device=device)
        self.mu = B.tensor(B.as_numpy(self.mu), dtype=B.as_numpy(self.mu).dtype, device=device)
        self.L = B.tensor(B.as_numpy(self.L), dtype=B.as_numpy(self.L).dtype, device=device)

    def promote(self, dim: int = 6000) -> None:
        if B.device_of(self.fast) != "cpu":
            return
        from herg.cupy_encoder import seed_to_cupy
        self.fast = seed_to_cupy(self.id.to_bytes(32, "big"), dim=dim)

    def demote(self) -> None:
        self.fast = B.tensor(B.as_numpy(self.fast), dtype=B.as_numpy(self.fast).dtype, device="cpu")
