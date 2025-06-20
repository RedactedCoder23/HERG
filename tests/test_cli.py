import subprocess
import sys
import yaml
from pathlib import Path


def test_cli_roundtrip(tmp_path, monkeypatch):
    home = tmp_path
    monkeypatch.setenv('HOME', str(home))
    cfg_path = home / '.config' / 'herg' / 'config.yml'
    subprocess.run([sys.executable, '-m', 'herg.cli', 'run-sim', '--radius', '3'], check=True)
    data = yaml.safe_load(cfg_path.read_text())
    assert data['radius'] == 3
