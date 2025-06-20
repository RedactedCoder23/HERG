"""FUSE mount stub for hvlogfs (non-functional on tests)."""

try:
    import fuse
except Exception:  # pragma: no cover - optional
    fuse = None


def mount(*args, **kwds) -> None:
    if fuse is None:
        raise RuntimeError('fuse not available')
    raise NotImplementedError('FUSE mount not implemented')
