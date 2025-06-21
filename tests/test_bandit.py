from herg.auto.tuner import BanditTuner
from herg import config
import random

class DummyStore:
    def __init__(self, val=0.5):
        self.val = val
    def snapshot(self):
        return {'retention': self.val}


def test_bandit_improves():
    store = DummyStore()
    cfg = config.Config()
    tuner = BanditTuner(rng=random.Random(42))
    for _ in range(10):
        delta = tuner.suggest(store.snapshot(), 'retention', cfg)
        if delta:
            cfg.apply(delta)
            store.val += 0.01
    assert store.val - 0.5 >= 0.05


def test_bandit_lane_split():
    cfg = config.Config()
    tuner = BanditTuner(rng=random.Random(0), epsilon_start=1.0, epsilon_end=1.0)
    delta = tuner.suggest({'retention': 0.5}, 'retention', cfg)
    if 'lane_split' in delta:
        cfg.apply(delta)
    assert isinstance(cfg.lane_split, tuple)
