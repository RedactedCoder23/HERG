# hvlogfs

`hvlogfs` provides a simple log-structured store for hypervectors. Data is
written in 4 MiB *hyperchunks* where each vector occupies 1024 bytes plus a CRC32
trailer. Every group of three chunks is protected with an XOR parity file.  The
implementation used for tests is intentionally small but mirrors the production
layout.

Chunks are memory-mapped for O(1) random access.  The `MetaIndex` maps a
32‑byte seed hash to a `(chunk_path, offset)` pair.  On retrieval the CRC is
verified before returning the bytes as a NumPy array.

A tiny write‑ahead journal provides crash safety and is replayed by higher
levels in real deployments.
