from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import os
import logging

from .config_lock import lock_path

CONFIG_PATH = Path.home() / '.config' / 'herg' / 'config.yml'

@dataclass
class Config:
    radius: int = 2
    alpha_u: float = 0.1
    alpha_b: float = 0.1
    eta: float = 0.05
    block_size: int = 512
    backend: str = 'stub'
    scrub_interval: int = 60
    gossip_every: int = 8
    energy_drain: float = 0.0
    tuner: str = 'bandit'
    lane_split: tuple[int, int, int] = (4096, 2048, 2048)
    kernel_alpha: float | list[float] = 1.0
    kernel_mode: str = 'separable'

    def apply(self, delta: dict) -> None:
        for k, v in delta.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)


def load(path: Path | None = None) -> 'Config':
    p = Path(path or CONFIG_PATH).expanduser()
    if p.exists():
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception:
            data = {}
        if 'lane_split' in data:
            data['lane_split'] = tuple(data['lane_split'])
        if 'kernel_alpha' in data and isinstance(data['kernel_alpha'], list):
            data['kernel_alpha'] = [float(v) for v in data['kernel_alpha']]
        return Config(**{**asdict(Config()), **data})
    cfg = Config()
    save(cfg, p)
    return cfg


def save(cfg: 'Config', path: Path | None = None) -> None:
    p = Path(path or CONFIG_PATH).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    data['lane_split'] = list(cfg.lane_split)
    if isinstance(cfg.kernel_alpha, tuple):
        data['kernel_alpha'] = list(cfg.kernel_alpha)
    p.write_text(yaml.dump(data))


def atomic_save(cfg: 'Config', path: Path | None = None) -> None:
    """Save config with file lock and atomic replace."""
    p = Path(path or CONFIG_PATH).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    try:
        with lock_path(p):
            data = asdict(cfg)
            data['lane_split'] = list(cfg.lane_split)
            if isinstance(cfg.kernel_alpha, tuple):
                data['kernel_alpha'] = list(cfg.kernel_alpha)
            tmp.write_text(yaml.dump(data))
            os.replace(tmp, p)
    except Exception as e:  # pragma: no cover - file I/O errors
        logging.error("atomic_save failed: %s", e)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
