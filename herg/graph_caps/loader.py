from typing import Iterator, Tuple
import numpy as np
from ..storage.hvlogfs import HyperChunk, ENTRY_SIZE


class HVLogLoader:
    def __init__(self, chunk_path: str):
        self.chunk = HyperChunk(chunk_path)

    def __iter__(self) -> Iterator[Tuple[bytes, np.ndarray]]:
        for i in range(self.chunk.count):
            off = 64 + i * ENTRY_SIZE
            yield b'', np.frombuffer(self.chunk.read(off), dtype=np.int8)
