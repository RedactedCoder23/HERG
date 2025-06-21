import subprocess


def test_cli_demo():
    capture = subprocess.run([
        "herg-run",
        '--demo', 'text'
    ], check=True, capture_output=True, text=True)
    assert 'HERG text demo' in capture.stdout
