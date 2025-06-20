import os
import mmap
import struct
import zlib
from typing import List

VECTOR_SIZE = 1024  # bytes per vector (8192 bits)
ENTRY_SIZE = VECTOR_SIZE + 4  # extra CRC32 per vector
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB
HEADER_FMT = '<8sIII44s'  # 64 bytes total
MAGIC = b'HVLGCHNK'


class ChecksumError(Exception):
    pass


class HyperChunk:
    """Simple hypervector chunk reader/writer with CRC checks."""

    def __init__(self, path: str, mode: str = 'r+b') -> None:
        self.path = path
        exists = os.path.exists(path)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT)
        if not exists:
            os.ftruncate(self.fd, CHUNK_SIZE)
        self.mm = mmap.mmap(self.fd, CHUNK_SIZE)
        if not exists:
            header = struct.pack(HEADER_FMT, MAGIC, 0, VECTOR_SIZE, 0, b'')
            self.mm[:64] = header
        else:
            magic, self.count, vsize, crc, _ = struct.unpack(HEADER_FMT, self.mm[:64])
            if magic != MAGIC:
                raise ValueError('Bad chunk magic')
            if vsize != VECTOR_SIZE:
                raise ValueError('Vector size mismatch')
            if crc and zlib.crc32(self.mm[64:64 + self.count * ENTRY_SIZE]) != crc:
                raise ChecksumError('Chunk CRC mismatch')
        if not exists:
            self.count = 0

    # ------------------------------------------------------------
    def append(self, vectors: List[bytes]) -> List[int]:
        """Append vectors, returning offsets."""
        offsets = []
        for vec in vectors:
            if len(vec) != VECTOR_SIZE:
                raise ValueError('Vector must be 1024 bytes')
            if 64 + (self.count + 1) * ENTRY_SIZE > CHUNK_SIZE:
                raise IOError('chunk full')
            off = 64 + self.count * ENTRY_SIZE
            crc = zlib.crc32(vec)
            self.mm[off:off + VECTOR_SIZE] = vec
            self.mm[off + VECTOR_SIZE: off + ENTRY_SIZE] = struct.pack('<I', crc)
            offsets.append(off)
            self.count += 1
        self._write_header()
        return offsets

    # ------------------------------------------------------------
    def read(self, offset: int) -> bytes:
        vec = self.mm[offset:offset + ENTRY_SIZE]
        data = vec[:VECTOR_SIZE]
        crc_stored = struct.unpack('<I', vec[VECTOR_SIZE:ENTRY_SIZE])[0]
        if zlib.crc32(data) != crc_stored:
            raise ChecksumError('vector crc mismatch')
        return data

    # ------------------------------------------------------------
    def _write_header(self) -> None:
        chunk_crc = zlib.crc32(self.mm[64:64 + self.count * ENTRY_SIZE])
        header = struct.pack(HEADER_FMT, MAGIC, self.count, VECTOR_SIZE, chunk_crc, b'')
        self.mm[:64] = header
        self.mm.flush()

    def close(self) -> None:
        self._write_header()
        self.mm.close()
        os.close(self.fd)

