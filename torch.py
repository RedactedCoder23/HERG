import numpy as np

int8 = np.int8
int16 = np.int16
int32 = np.int32
float32 = np.float32

class Tensor:
    def __init__(self, data, dtype=None, device=None):
        self.data = np.array(data, dtype=dtype)
        self.device = device or "cpu"

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def to(self, device=None, dtype=None):
        return Tensor(self.data.astype(dtype or self.data.dtype), device or self.device)

    def view(self, *shape):
        return Tensor(self.data.reshape(*shape), self.data.dtype, self.device)

    def mean(self, dim=None):
        return Tensor(self.data.mean(axis=dim), self.data.dtype, self.device)

    def numel(self):
        return self.data.size

    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype else self.data

    def __matmul__(self, other):
        other_arr = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data @ other_arr, self.data.dtype, self.device)

    def __add__(self, other):
        other_arr = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data + other_arr, self.data.dtype, self.device)

    def __getitem__(self, key):
        res = self.data[key]
        if isinstance(res, np.ndarray):
            return Tensor(res, res.dtype, self.device)
        return res


def tensor(data, dtype=None, device=None):
    return Tensor(data, dtype, device)

def as_tensor(data, dtype=None, device=None):
    return tensor(data, dtype, device)

def zeros(*size, dtype=float32, device=None):
    return Tensor(np.zeros(size, dtype=dtype), dtype, device)

def stack(seq, dim=0):
    arr = np.stack([s.data if isinstance(s, Tensor) else s for s in seq], axis=dim)
    return Tensor(arr, arr.dtype, seq[0].device if seq else "cpu")

def cat(seq, dim=0):
    arr = np.concatenate([s.data if isinstance(s, Tensor) else s for s in seq], axis=dim)
    return Tensor(arr, arr.dtype, seq[0].device if seq else "cpu")

class nn:
    class functional:
        @staticmethod
        def pad(t, pad):
            left, right = pad
            arr = np.pad(t.data if isinstance(t, Tensor) else t, ((0,0),(left,right)), constant_values=0)
            return Tensor(arr, t.dtype if isinstance(t, Tensor) else arr.dtype, getattr(t, 'device', 'cpu'))

def erf(x):
    arr = np.erf(x.data if isinstance(x, Tensor) else x)
    return Tensor(arr, getattr(x, 'dtype', arr.dtype), getattr(x, 'device', 'cpu'))

class cuda:
    @staticmethod
    def is_available():
        return False
