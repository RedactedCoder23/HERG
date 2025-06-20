import os
import pickle
from typing import Tuple, Optional


class MetaIndex:
    """Very small meta index mapping seed hash to (chunk_path, offset)."""

    def __init__(self, path: str):
        self.path = path
        try:
            with open(self.path, 'rb') as f:
                self._index = pickle.load(f)
        except Exception:
            self._index = {}

    def put(self, seed_hash: bytes, location: Tuple[str, int]) -> None:
        self._index[seed_hash] = location
        with open(self.path, 'wb') as f:
            pickle.dump(self._index, f)

    def get(self, seed_hash: bytes) -> Optional[Tuple[str, int]]:
        return self._index.get(seed_hash)
