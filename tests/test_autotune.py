from herg.auto.metrics import MetricStore
from herg.auto.tuner import HillClimbTuner
from herg import config
import sys
from unittest import mock


def test_autotune_hillclimb():
    store = MetricStore()
    cfg = config.Config(radius=1)
    tuner = HillClimbTuner()
    delta = tuner.suggest({'retention': 0.6}, 'retention', cfg)
    assert delta.get('radius') == 2


def test_integration_run(tmp_path, monkeypatch, capsys):
    home = tmp_path
    monkeypatch.setenv('HOME', str(home))
    argv = [
        'herg',
        'auto-run',
        '--ticks', '200',
        '--tune-interval', '1',
        '--goal', 'retention',
        '--tuner', 'hill',
    ]
    with mock.patch.object(sys, 'argv', argv):
        from herg.cli import main

        main()
        out = capsys.readouterr().out
    assert 'improved' in out
