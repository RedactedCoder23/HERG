import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Install lightweight stubs for heavy deps when missing
try:
    from herg import _ci_stubs  # noqa: F401
except Exception:
    pass
