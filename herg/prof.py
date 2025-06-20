import time
from contextlib import ContextDecorator
from herg import backend as B


class Prof(ContextDecorator):
    def __enter__(self):
        self.start = time.perf_counter()
        self.cuda_evt_start = None
        self.cuda_evt_end = None
        if getattr(B, '_TORCH', False):
            import torch
            self.cuda_evt_start = torch.cuda.Event(enable_timing=True)
            self.cuda_evt_end = torch.cuda.Event(enable_timing=True)
            self.cuda_evt_start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.perf_counter()
        wall = end - self.start
        if self.cuda_evt_start is not None:
            import torch
            self.cuda_evt_end.record()
            torch.cuda.synchronize()
            cuda_ms = self.cuda_evt_start.elapsed_time(self.cuda_evt_end)
            print(f"wall={wall:.3f}s cuda={cuda_ms/1000:.3f}s")
        else:
            print(f"wall={wall:.3f}s")
