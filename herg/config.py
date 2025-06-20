from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

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


def load(path: Path | None = None) -> 'Config':
    p = Path(path or CONFIG_PATH).expanduser()
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
        return Config(**{**asdict(Config()), **data})
    cfg = Config()
    save(cfg, p)
    return cfg


def save(cfg: 'Config', path: Path | None = None) -> None:
    p = Path(path or CONFIG_PATH).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(asdict(cfg)))
