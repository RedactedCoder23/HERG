# hvlogfs backends

`DAXBackend` and `SPDKBackend` expose the same mmap‑style API.  The former uses
ordinary file descriptors while the latter would talk to SPDK in production.
For the unit tests both simply wrap Python's `mmap` module so switching is
trivial:

```python
from herg.storage.hvlogfs import DAXBackend, SPDKBackend
```

Choose a backend at open time depending on your environment.
