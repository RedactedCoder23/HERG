# CI stub injector – must run **before** any heavy imports
from ._ci_stubs import *  # noqa: F401,F403

from .backend import tensor, as_numpy, dot, cosine, stack, device_of  # noqa: E402

__all__ = [
    'tensor', 'as_numpy', 'dot', 'cosine', 'stack', 'device_of'
]
