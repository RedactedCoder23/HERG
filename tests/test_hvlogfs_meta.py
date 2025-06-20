import time
import numpy as np
from herg.storage.hvlogfs import HyperChunk, MetaIndex


def test_lookup_latency(tmp_path):
    chunk = HyperChunk(str(tmp_path / 'c.chk'))
    idx = MetaIndex(str(tmp_path / 'meta.idx'))
    vec = b'B' * 1024
    [off] = chunk.append([vec])
    idx.put(b'seed', (str(tmp_path / 'c.chk'), off))

    t0 = time.perf_counter()
    loc = idx.get(b'seed')
    chunk2 = HyperChunk(loc[0])
    out = chunk2.read(loc[1])
    dt = time.perf_counter() - t0
    assert dt < 0.002
    assert out == vec
    chunk.close(); chunk2.close()

