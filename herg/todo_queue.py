import datetime
import pathlib


def add_todo(file_path: str, todo_line: str) -> None:
    path = pathlib.Path(file_path)
    marker = f"\u25c7 CODEX_IMPLEMENT: {todo_line}"
    lines = path.read_text().splitlines() if path.exists() else []
    if any(marker in line for line in lines):
        return
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    insert_idx = len(lines)
    found_import = False
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            found_import = True
        elif found_import and not line.strip():
            insert_idx = i + 1
            break
    lines.insert(insert_idx, marker)
    lines.insert(insert_idx, f"# \u23F0 {ts}")
    path.write_text("\n".join(lines) + "\n")

