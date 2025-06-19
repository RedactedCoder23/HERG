import os
import pathlib
import subprocess
import tempfile
import textwrap
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / '.github' / 'scripts'))

import codex_todo_runner as ctr


def test_gather_todos_single_marker(tmp_path):
    f = tmp_path / 'a.py'
    content = 'line1\n' + ctr.MARKER + ' foo\nline3\n'
    f.write_text(content)
    blocks = ctr.gather_todos(tmp_path)
    assert len(blocks) == 1
    path, lineno, ctx = blocks[0]
    assert path == pathlib.Path('a.py')
    assert lineno == 2
    assert ctr.MARKER in ctx


def test_build_prompt_includes_header_and_context():
    blocks = [(pathlib.Path('x.py'), 1, 'ctx')]
    prompt = ctr.build_prompt(blocks)
    assert ctr.HEADER.strip() in prompt
    assert '### FILE: x.py' in prompt
    assert 'ctx' in prompt


def test_apply_diff_and_pr_bad_diff(tmp_path):
    repo = tmp_path
    subprocess.run(['git', 'init'], cwd=repo, check=True, stdout=subprocess.PIPE)
    (repo / 'foo.txt').write_text('hello')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-m', 'init'], cwd=repo, check=True, stdout=subprocess.PIPE)
    bad_diff = '```diff\n@@\n--- a/foo.txt\n+++ b/foo.txt\n@@\n+bad\n```'
    with pytest.raises(subprocess.CalledProcessError):
        ctr.apply_diff_and_pr(bad_diff, branch_prefix='test')

