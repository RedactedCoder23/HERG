from .chunk import HyperChunk, ChecksumError, VECTOR_SIZE, ENTRY_SIZE, CHUNK_SIZE
from .index import MetaIndex
from .journal import WriteAheadJournal
from .backend import DAXBackend, SPDKBackend
from .graph import DiskHNSW
from .scrub import scrub


class Capsule:
    def __init__(self, cap_id, mu, meta):
        self.id_int = int(cap_id)
        self.mu = mu
        self.meta = meta
        self.chunk = "mem"
        self.active = True


class MemChunk:
    """In-memory chunk with minimal API for dev jobs."""

    def __init__(self, caps):
        self._caps = caps
        self.path = "mem"

    def capsules(self):
        return self._caps

    def tombstone(self, cap_id: int):
        for c in self._caps:
            if c.id_int == cap_id:
                c.active = False

    def flush(self):
        pass

    def is_closed(self):
        return True


class HVLogFS:
    """Minimal in-memory HVLogFS used for development."""

    def __init__(self, path: str):
        self.path = path
        self._caps = []
        self._chunk = MemChunk(self._caps)
        self._chunks = [self._chunk]

    def append_cap(self, prefix: str, cap_id: int, mu, meta: dict) -> None:
        self._caps.append(Capsule(cap_id, mu, meta))

    def iter_capsules(self, prefix: str = ""):
        from agent.encoder_ext import prefix as _pfx
        for c in self._caps:
            if prefix and _pfx(c.id_int) != prefix:
                continue
            yield c

    def chunks(self, active_only: bool = True):
        return self._chunks

__all__ = [
    'HyperChunk', 'ChecksumError', 'MetaIndex', 'WriteAheadJournal',
    'DAXBackend', 'SPDKBackend', 'DiskHNSW', 'scrub',
    'Capsule', 'HVLogFS',
    'VECTOR_SIZE', 'ENTRY_SIZE', 'CHUNK_SIZE'
]
