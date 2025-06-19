"""
GitHub Action helper that scans the repository for special CODEX markers,
sends them to OpenAI Codex, applies the returned patch, commits it on a
new branch, and opens a draft pull-request.

Environment vars required:
    OPENAI_API_KEY   – OpenAI credential
    GITHUB_TOKEN     – automatically provided by GitHub Actions

The script is run from the repo root by workflows, never manually.
"""

from __future__ import annotations
import os, re, sys, subprocess, time, tempfile, pathlib, textwrap
from typing import List, Tuple
import openai

HEADER = textwrap.dedent(
    """\
    ### SYSTEM
    You are Codex.  Implement every task below **in code**.
    Return one unified diff inside ```diff fences.  No explanations.
    """
)

MARKER = "\u25c7 CODEX_IMPLEMENT:"

# ── PHASE A ─────────────────────────────────────────────────────────────
# gather_todos(repo_root: pathlib.Path,
#              marker: str = MARKER) -> List[Tuple[pathlib.Path, int, str]]
#     • Walk all *.py files under repo_root (skip .git, .venv, .pytest_cache)
#     • For each line containing `marker`, capture ±15 lines of context
#       and return (file_path, lineno, context_string) tuples ordered by file path.
#     • Raise ValueError if > 400 markers are found.
# ────────────────────────────────────────────────────────────────────────

def gather_todos(repo_root: pathlib.Path, marker: str = MARKER) -> List[Tuple[pathlib.Path, int, str]]:
    results: List[Tuple[pathlib.Path, int, str]] = []
    for path in repo_root.rglob("*.py"):
        if any(part in {".git", ".venv", ".pytest_cache"} for part in path.parts):
            continue
        try:
            lines = path.read_text().splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, 1):
            if marker in line:
                start = max(0, idx - 16)
                end = idx + 15
                context = "\n".join(lines[start:end])
                results.append((path.relative_to(repo_root), idx, context))
    if len(results) > 400:
        raise ValueError("too many markers")
    results.sort(key=lambda t: str(t[0]))
    return results

# ── PHASE B ─────────────────────────────────────────────────────────────
# build_prompt(blocks: List[Tuple[pathlib.Path, int, str]]) -> str
#     • Compose the final prompt string:
#       HEADER + two newlines +
#       each block → "### FILE: {relative_path}\n{context}\n"
#     • Return the resulting prompt.
# ────────────────────────────────────────────────────────────────────────

def build_prompt(blocks: List[Tuple[pathlib.Path, int, str]]) -> str:
    parts = [HEADER, ""]
    for path, _lineno, context in blocks:
        parts.append(f"### FILE: {path}\n{context}\n")
    return "\n".join(parts)

# ── PHASE C ─────────────────────────────────────────────────────────────
# apply_diff_and_pr(diff_text: str,
#                   branch_prefix: str = "codex-auto") -> None
#     • Write `diff_text` to a temp file.
#     • Create new branch `{branch_prefix}/{epoch_ts}`.
#     • Apply patch (`patch -p1 -i temp_file`); abort if patch fails.
#     • `git add -u` → commit with message "Codex auto-impl".
#     • Open draft PR (`gh pr create --draft --fill`).
#     • Exit non-zero on any failure so the workflow fails visibly.
# ────────────────────────────────────────────────────────────────────────

def apply_diff_and_pr(diff_text: str, branch_prefix: str = "codex-auto") -> None:
    m = re.search(r"```diff\n(.*)```", diff_text, re.S)
    body = m.group(1) if m else diff_text
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(body)
        tmp_path = tmp.name
    branch = f"{branch_prefix}/{int(time.time())}"
    try:
        subprocess.run(["git", "checkout", "-b", branch], check=True)
        subprocess.run(["patch", "-p1", "-i", tmp_path], check=True)
        subprocess.run(["git", "add", "-u"], check=True)
        subprocess.run(["git", "commit", "-m", "Codex auto-impl"], check=True)
        subprocess.run(["gh", "pr", "create", "--draft", "--fill"], check=True)
    finally:
        os.unlink(tmp_path)


def main() -> None:                            #   <<< do NOT edit this body
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    blocks = gather_todos(repo_root)
    if not blocks:       # nothing to do
        print("No markers found; exiting 0.")
        return

    prompt = build_prompt(blocks)
    print("→ Sending prompt to OpenAI ({} bytes)…".format(len(prompt)))

    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o-codex-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    diff_text = response.choices[0].message.content.strip()
    if not diff_text.startswith("```diff"):
        sys.stderr.write("ERROR: No diff found in Codex response\n")
        sys.exit(1)

    apply_diff_and_pr(diff_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"Runner failed: {exc}\n")
        sys.exit(1)
