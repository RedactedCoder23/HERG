import argparse
import os
import time
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from herg.storage.hvlogfs import HyperChunk


def run_bench(n: int) -> None:
    path = 'bench.chk'
    if os.path.exists(path):
        os.remove(path)
    chunk = HyperChunk(path)
    vec = bytes([0]) * 1024
    from herg.storage.hvlogfs import ENTRY_SIZE, CHUNK_SIZE
    capacity = (CHUNK_SIZE - 64) // ENTRY_SIZE
    batch = [vec] * capacity
    start = time.time()
    i = 0
    while i < n:
        step = min(capacity, n - i)
        try:
            chunk.append(batch[:step])
        except OSError:
            idx = i // capacity
            chunk.close()
            chunk = HyperChunk(f'bench{idx}.chk')
            chunk.append(batch[:step])
        i += step
    elapsed = time.time() - start
    mbps = (n * 1024) / (1024 * 1024) / elapsed
    vps = n / elapsed
    print(f"{mbps:.1f} MB/s {vps:.0f} vectors/s")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000000)
    args = p.parse_args()
    run_bench(args.n)
