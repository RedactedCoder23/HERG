import numpy as np
from herg.storage.hvlogfs import HyperChunk, ENTRY_SIZE


def test_roundtrip(tmp_path):
    path = tmp_path / 'c0.chk'
    chunk = HyperChunk(str(path))
    vec = bytes([1]) * 1024
    offs = chunk.append([vec])
    read = chunk.read(offs[0])
    assert read == vec
    chunk.close()
