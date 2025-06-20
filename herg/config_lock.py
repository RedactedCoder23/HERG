import logging
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
    HAVE_FCNTL = True
except Exception:  # pragma: no cover
    HAVE_FCNTL = False
    fcntl = None

try:
    import portalocker
except Exception:  # pragma: no cover
    portalocker = None

@contextmanager
def lock_path(path: Path):
    if HAVE_FCNTL:
        f = open(path, 'a+')
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield f
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            finally:
                f.close()
    elif portalocker:
        f = open(path, 'a+')
        try:
            portalocker.lock(f, portalocker.LOCK_EX)
            yield f
        finally:
            portalocker.unlock(f)
            f.close()
    else:
        logging.warning('no file locking module available')
        yield None
