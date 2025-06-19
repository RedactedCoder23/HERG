"""Automation helper for Codex-driven TODO resolution."""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import openai


TODO_RE = re.compile(r"^# ❑ CODEX: (.*)$")


def gather_todos(repo_root: Path) -> str:
    entries = []
    for path in repo_root.rglob("*.py"):
        rel = path.relative_to(repo_root)
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            m = TODO_RE.match(line)
            if m:
                start = max(0, idx - 20)
                context = "\n".join(lines[start:idx])
                entries.append(f"### File:{rel}\n{context}\n### TODO:{m.group(1)}")
    return "\n".join(entries)


def apply_patch(diff_text: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(diff_text)
        tmp_path = tmp.name
    subprocess.run(["patch", "-p1", "-i", tmp_path], check=True)
    os.unlink(tmp_path)


def main() -> int:
    sha = os.getenv("GITHUB_SHA", "")
    repo = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    prompt = gather_todos(Path("."))
    if not prompt.strip():
        return 0

    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-codex-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
    except Exception as e:
        print(f"OpenAI error: {e}", file=sys.stderr)
        return 1

    text = resp.choices[0].message.content
    diffs = re.findall(r"```diff\n(.*?)```", text, re.S)
    if not diffs:
        print("No diff found", file=sys.stderr)
        return 1

    branch = f"codex-auto/{int(time.time())}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)

    for diff in diffs:
        apply_patch(diff)

    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "codex:auto apply"], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)

    pr_result = subprocess.run(["gh", "pr", "create", "--draft", "--fill"], check=False)
    return pr_result.returncode


if __name__ == "__main__":
    sys.exit(main())


