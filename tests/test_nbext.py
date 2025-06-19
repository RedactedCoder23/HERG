import sys
import pytest
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from herg.nbext import load_ipython_extension


def get_shell():
    ip = get_ipython()
    if ip is None:
        ip = TerminalInteractiveShell.instance()
    return ip


def test_herg_magics(tmp_path):
    ip = get_shell()
    load_ipython_extension(ip)

    dot = ip.run_line_magic("herg_viz", "2")
    assert isinstance(dot, str) and "digraph" in dot

    tmp = tmp_path / "b.pkl"
    out = ip.run_line_magic("herg_snapshot", f"save {tmp}")
    assert tmp.exists()
    out2 = ip.run_line_magic("herg_snapshot", f"load {tmp}")
    assert "Loaded" in out2
