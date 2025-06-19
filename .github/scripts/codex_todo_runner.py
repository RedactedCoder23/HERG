"""Automation helper for Codex-driven development."""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import openai


HEADER = (
    "You are Codex.  \n"
    "Generate *executable* Python code that fully satisfies every ◇ CODEX_IMPLEMENT: instruction below.  \n"
    "Return **one unified diff** inside ```diff fences — no explanations, no TODO echoes.""
)

IMPL_RE = re.compile(r"\u25c7 CODEX_IMPLEMENT:")


def gather_files(repo_root: Path) -> str:
    blocks = []
    for path in repo_root.rglob("*.py"):
        text = path.read_text()
        if IMPL_RE.search(text):
            rel = path.relative_to(repo_root)
            blocks.append(f"### FILE: {rel}\n```python\n{text}\n```")
    return "\n".join(blocks)


def apply_patch(diff: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(diff)
        tmp_path = tmp.name
    subprocess.run(["patch", "-p1", "-i", tmp_path], check=True)
    os.unlink(tmp_path)


def main() -> int:
    repo_root = Path(".")
    payload = gather_files(repo_root)
    if not payload.strip():
        return 0

    prompt = f"{HEADER}\n{payload}"
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai.api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    while True:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-codex-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            break
        except openai.error.RateLimitError:
            time.sleep(5)
        except Exception as e:
            print(f"OpenAI error: {e}", file=sys.stderr)
            return 1

    m = re.search(r"```diff\n(.*?)```", resp.choices[0].message.content, re.S)
    if not m:
        print("No diff found", file=sys.stderr)
        return 1

    diff_text = m.group(1)
    apply_patch(diff_text)

    branch = f"codex-auto/{int(time.time())}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "codex:auto apply"], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)
    subprocess.run(["gh", "pr", "create", "--draft", "--fill"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

