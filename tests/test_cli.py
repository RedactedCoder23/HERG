import subprocess
from pathlib import Path
import sys


def test_cli_demo():
    script = Path(__file__).resolve().parents[1] / "herg-run"
    capture = subprocess.run(
        [sys.executable, str(script), '--demo', 'text'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'HERG text demo' in capture.stdout
