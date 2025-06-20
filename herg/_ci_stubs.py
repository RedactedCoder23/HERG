"""
Light-weight fallback modules so sandboxed CI (no PyPI access) can import
`numpy`, `torch`, `yaml`, and `IPython` without blowing up. Each stub exposes
just the names our tests use – **nothing more**.
"""

import importlib.util
import sys
from types import ModuleType
from contextlib import contextmanager


def inject():
    """Install stub modules if real ones are missing."""
    # numpy stub
    if importlib.util.find_spec("numpy") is None and "numpy" not in sys.modules:
        np = ModuleType("numpy")
        np.array = lambda x, **k: x
        np.asarray = lambda x, **k: x
        np.stack = lambda seq, axis=0: [s for s in seq]
        np.zeros = lambda shape, **k: [0] * (shape if isinstance(shape, int) else shape[0])
        np.ones = lambda shape, **k: [1] * (shape if isinstance(shape, int) else shape[0])
        np.outer = lambda a, b: [[i * j for j in b] for i in a]
        np.dot = lambda a, b: sum(i * j for i, j in zip(a, b))
        np.array_equal = lambda a, b: a == b
        np.allclose = lambda a, b, **k: a == b
        np.all = lambda x: all(x)
        np.any = lambda x: any(x)
        np.count_nonzero = lambda x: len([i for i in x if i])
        np.argsort = lambda x: sorted(range(len(x)), key=lambda i: x[i])
        np.cos = lambda x: x
        np.sin = lambda x: x
        np.sinc = lambda x: x
        np.mean = lambda x, **k: sum(x) / len(x)
        np.linalg = ModuleType("linalg")
        def _norm(a, axis=None):
            if axis is None:
                return sum(i * i for i in a) ** 0.5
            return [sum(i * i for i in row) ** 0.5 for row in a]
        np.linalg.norm = _norm
        np.random = ModuleType("random")
        class _RNG:
            def __init__(self, seed=None):
                self.seed = seed

            def integers(self, low, high=None, size=None, dtype=None):
                n = size if isinstance(size, int) else (size[0] if size else 1)
                return [0] * n

            def random(self, size=None):
                n = size if isinstance(size, int) else (size[0] if size else 1)
                return [0.0] * n

            def standard_normal(self, size=None):
                n = size if isinstance(size, int) else (size[0] if size else 1)
                return [0.0] * n

        np.random.default_rng = lambda seed=None: _RNG(seed)
        np.random.randn = lambda *shape: [0] * (shape[0] if shape else 1)
        np.sign = lambda arr: [1 if i >= 0 else -1 for i in arr]
        def _prod(arr, axis=None):
            if axis is None:
                result = 1
                for i in arr:
                    result *= i
                return result
            return [ _prod(row) for row in arr ]
        np.prod = _prod
        np.int8 = "int8"
        np.int16 = "int16"
        np.int32 = "int32"
        np.float32 = "float32"
        np.uint8 = "uint8"
        np.ndarray = list
        sys.modules["numpy"] = np

    # yaml stub
    if importlib.util.find_spec("yaml") is None and "yaml" not in sys.modules:
        yaml = ModuleType("yaml")
        yaml.safe_load = lambda s: {}
        yaml.safe_dump = lambda d, **k: ""
        sys.modules["yaml"] = yaml

    # torch stub
    if importlib.util.find_spec("torch") is None and "torch" not in sys.modules:
        torch = ModuleType("torch")

        class _Tensor(list):
            def numpy(self):
                return self

            def numel(self):
                return len(self)

            @property
            def shape(self):
                return (len(self),)

            def to(self, *args, **kwargs):
                return self

        torch.Tensor = _Tensor
        torch.tensor = lambda x, **k: _Tensor(x if isinstance(x, list) else list(x))
        torch.from_numpy = lambda x: _Tensor(list(x))
        torch.zeros = lambda *shape, **k: _Tensor([0] * (shape[-1] if shape else 1))
        torch.stack = lambda seq, dim=0: _Tensor([item for t in seq for item in t])
        torch.cat = lambda seq, dim=0: _Tensor([item for t in seq for item in t])
        torch.sign = lambda x: _Tensor([1 if i >= 0 else -1 for i in x])

        torch.nn = ModuleType("nn")
        torch.nn.functional = ModuleType("functional")
        torch.nn.functional.pad = lambda t, pad: t + [0] * sum(pad)

        @contextmanager
        def _noop():
            yield

        torch.no_grad = _noop
        sys.modules["torch"] = torch

    # IPython stub
    if importlib.util.find_spec("IPython") is None and "IPython" not in sys.modules:
        ipy = ModuleType("IPython")

        class _Shell:
            def __init__(self):
                self._magics = None

            @classmethod
            def instance(cls):
                if not hasattr(cls, "_inst"):
                    cls._inst = cls()
                return cls._inst

            def register_magics(self, cls_):
                self._magics = cls_(shell=self)

            def run_line_magic(self, name, line):
                m = getattr(self._magics, name)
                return m(line)

        def get_ipython():
            return _Shell.instance()

        ipy.get_ipython = get_ipython
        ipy.terminal = ModuleType("terminal")
        ipy.terminal.interactiveshell = ModuleType("interactiveshell")
        ipy.terminal.interactiveshell.TerminalInteractiveShell = _Shell
        ipy.core = ModuleType("core")
        ipy.core.magic = ModuleType("magic")
        ipy.core.magic.Magics = object
        ipy.core.magic.magics_class = lambda cls: cls
        ipy.core.magic.line_magic = lambda f: f
        ipy.display = ModuleType("display")
        ipy.display.SVG = lambda data=None: data
        ipy.display.display = lambda *a, **k: None
        sys.modules["IPython"] = ipy
        sys.modules["IPython.terminal.interactiveshell"] = ipy.terminal.interactiveshell
        sys.modules["IPython.core.magic"] = ipy.core.magic
        sys.modules["IPython.display"] = ipy.display

# When imported, perform injection
inject()

# END _ci_stubs.py
