import os


class WriteAheadJournal:
    """Append-only journal for durability."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, 'ab')

    def append(self, data: bytes) -> None:
        self.f.write(data + b"\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()
