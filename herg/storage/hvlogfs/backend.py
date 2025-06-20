import mmap
import os


class BackendBase:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT)

    def mmap(self, length: int) -> mmap.mmap:
        return mmap.mmap(self.fd, length)

    def close(self):
        os.close(self.fd)


class DAXBackend(BackendBase):
    pass


class SPDKBackend(BackendBase):
    pass
