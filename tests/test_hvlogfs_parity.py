import os
import numpy as np
from herg.storage.hvlogfs import HyperChunk
from herg.storage.hvlogfs.parity import xor_chunks
from herg.storage.hvlogfs.scrub import rebuild


def test_rebuild(tmp_path):
    d0 = HyperChunk(str(tmp_path / 'd0.chk'))
    d1 = HyperChunk(str(tmp_path / 'd1.chk'))
    d2 = HyperChunk(str(tmp_path / 'd2.chk'))

    vec = b'A' * 1024
    d0.append([vec])
    d1.append([vec])
    d2.append([vec])
    d0.close(); d1.close(); d2.close()

    with open(tmp_path / 'd0.chk', 'rb') as f:
        b0 = f.read()
    with open(tmp_path / 'd1.chk', 'rb') as f:
        b1 = f.read()
    with open(tmp_path / 'd2.chk', 'rb') as f:
        b2 = f.read()
    parity = xor_chunks(b0, b1, b2)
    with open(tmp_path / 'p.chk', 'wb') as f:
        f.write(parity)

    os.remove(tmp_path / 'd1.chk')
    rebuild(str(tmp_path / 'd1.chk'), str(tmp_path / 'd0.chk'), str(tmp_path / 'd1.chk'), str(tmp_path / 'd2.chk'), str(tmp_path / 'p.chk'))
    rebuilt = HyperChunk(str(tmp_path / 'd1.chk'))
    got = rebuilt.read(64)
    assert got == vec
    rebuilt.close()

