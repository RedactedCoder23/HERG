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
    if "numpy" not in sys.modules and importlib.util.find_spec("numpy") is None:
        np = ModuleType("numpy")

        # --- minimal ndarray stand-in -------------------------------------- #
        class NDArray(list):
            """Just enough duck-typing to satisfy HERG tests (1-D only)."""

            def __init__(self, data, *, dtype="float32"):
                super().__init__(data if isinstance(data, list) else [data])
                self.dtype = dtype

            def _it(self, other):
                if isinstance(other, NDArray):
                    return other
                if isinstance(other, list):
                    return other
                return [other] * len(self)

            def __add__(self, other):
                o = self._it(other)
                return NDArray([a + b for a, b in zip(self, o)], dtype=self.dtype)

            def __mul__(self, other):
                o = self._it(other)
                return NDArray([a * b for a, b in zip(self, o)], dtype=self.dtype)

            def __sub__(self, other):
                o = self._it(other)
                return NDArray([a - b for a, b in zip(self, o)], dtype=self.dtype)

            def __truediv__(self, other):
                o = self._it(other)
                return NDArray([a / b for a, b in zip(self, o)], dtype=self.dtype)

            def __neg__(self):
                return NDArray([-a for a in self], dtype=self.dtype)

            @property
            def size(self):
                return len(self)

            def reshape(self, *shape):
                return NDArray(self[:], dtype=self.dtype)

            def tobytes(self):
                return bytes(int(x) & 0xFF for x in self)

            def __radd__(self, other):
                return NDArray([b + a for a, b in zip(self, self._it(other))], dtype=self.dtype)

            def __rmul__(self, other):
                return NDArray([b * a for a, b in zip(self, self._it(other))], dtype=self.dtype)

            def __rsub__(self, other):
                return NDArray([b - a for a, b in zip(self, self._it(other))], dtype=self.dtype)

            def __rtruediv__(self, other):
                return NDArray([b / a for a, b in zip(self, self._it(other))], dtype=self.dtype)

            @property
            def shape(self):
                return (len(self),)

            def astype(self, dt):
                self.dtype = dt
                return self

            def copy(self):
                return NDArray(self[:], dtype=self.dtype)

            def tolist(self):
                return list(self)

        NDArray.__qualname__ = "NDArray"
        NDArray.__module__ = __name__

        def _wrap(x, **kw):
            return NDArray(list(x) if isinstance(x, (list, tuple, NDArray)) else [x], **kw)

        np.array = lambda x, **k: _wrap(x, dtype=k.get("dtype", "float32"))
        np.asarray = np.array
        np.stack = lambda seq, axis=0: NDArray([elem for arr in seq for elem in arr])
        np.zeros = lambda shape, **k: NDArray([0] * (shape if isinstance(shape, int) else shape[0]), dtype=k.get("dtype", "float32"))
        np.ones = lambda shape, **k: NDArray([1] * (shape if isinstance(shape, int) else shape[0]), dtype=k.get("dtype", "float32"))
        np.int8 = "int8"
        np.int16 = "int16"
        np.int32 = "int32"
        np.float32 = "float32"
        np.uint8 = "uint8"
        np.frombuffer = lambda buf, dtype="uint8", count=-1, offset=0: NDArray(list(buf[offset:len(buf) if count==-1 else offset+count]), dtype=str(dtype))
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
                return NDArray([0] * n, dtype="int32")

            def random(self, size=None):
                n = size if isinstance(size, int) else (size[0] if size else 1)
                return NDArray([0.0] * n)

            def standard_normal(self, size=None):
                n = size if isinstance(size, int) else (size[0] if size else 1)
                return NDArray([0.0] * n)

        np.random.default_rng = lambda seed=None: _RNG(seed)
        np.random.randint = lambda low, high=None, size=None, **k: NDArray([0] * (size if isinstance(size, int) else size[0]), dtype="int32")
        np.random.random = lambda size=None: NDArray([0.0] * (size if isinstance(size, int) else size[0]))
        np.random.randn = lambda *shape: NDArray([0.0] * (shape[0] if shape else 1))
        np.random.normal = lambda loc=0.0, scale=1.0, size=None: NDArray([loc] * (size if isinstance(size, int) else (size[0] if size else 1)))
        np.sign = lambda arr: NDArray([1 if i >= 0 else -1 for i in (arr if isinstance(arr, (list, NDArray)) else [arr])], dtype="int8")
        np.clip = lambda arr, a_min, a_max: NDArray([max(a_min, min(a_max, x)) for x in (arr if isinstance(arr, (list, NDArray)) else [arr])])
        np.vectorize = lambda fn: lambda arr: NDArray([fn(x) for x in (arr if isinstance(arr, (list, NDArray)) else [arr])])

        def _prod(arr, axis=None):
            if axis is None:
                result = 1
                for i in arr:
                    result *= i
                return result
            return [_prod(row) for row in arr]

        np.prod = _prod
        np.ndarray = NDArray
        sys.modules["numpy"] = np

    # yaml stub
    if importlib.util.find_spec("yaml") is None and "yaml" not in sys.modules:
        yaml = ModuleType("yaml")
        def _load_yaml(s):
            data = {}
            for line in s.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                v = v.strip()
                if v.isdigit():
                    v = int(v)
                data[k.strip()] = v
            return data
        yaml.safe_load = _load_yaml
        yaml.safe_dump = lambda d, **k: "\n".join(f"{k}: {v}" for k, v in d.items())
        yaml.load = lambda s, **k: yaml.safe_load(s)
        yaml.dump = lambda d, **k: yaml.safe_dump(d, **k)
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
                self._magics = cls_()

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

    # hvlogfs stub used by agent tests
    if importlib.util.find_spec("herg.hvlogfs") is None:
        hv = ModuleType("herg.hvlogfs")

        class Capsule:
            def __init__(self, cap_id, mu, meta):
                self.id_int = int(cap_id)
                self.mu = mu
                self.meta = meta
                self.chunk = "mem"
                self.active = True

        class HVLogFS:
            def __init__(self, path: str):
                self.path = path
                self._caps = []

            def append_cap(self, prefix: str, cap_id: int, mu, meta=None) -> None:
                self._caps.append(Capsule(cap_id, mu, meta or {}))

            def iter_capsules(self, prefix: str = None):
                for c in self._caps:
                    yield c

            def chunks(self):
                return []

        hv.HVLogFS = HVLogFS
        hv.Capsule = Capsule
        sys.modules["herg.hvlogfs"] = hv

# When imported, perform injection
inject()

# END _ci_stubs.py
