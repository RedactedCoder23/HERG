import tempfile, subprocess, sys, os
from herg.graph_caps.store import CapsuleStore
from herg.snapshot import save_snapshot, load_snapshot


def test_save_and_load(tmp_path):
    store = CapsuleStore()
    store.spawn(b"x", ts=0)
    file = tmp_path / "brain.pkl"
    save_snapshot(store, str(file))
    new = load_snapshot(str(file))
    assert isinstance(new, CapsuleStore)
    assert len(new.caps) == 1


def test_cli_save_load(tmp_path, capsys):
    file = tmp_path / "brain.pkl"
    subprocess.run([sys.executable, "-m", "herg.cli", "save", str(file)], check=True)
    assert file.exists()
    capture = subprocess.run([
        sys.executable,
        "-m",
        "herg.cli",
        "load",
        str(file),
    ], check=True, capture_output=True, text=True)
    assert "Loaded 1 capsules" in capture.stdout
