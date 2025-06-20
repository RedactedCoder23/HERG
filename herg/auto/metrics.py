import time
import statistics
from collections import deque


class MetricStore:
    """Collects runtime metrics for the auto-tuner."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.ingest_rate = 0.0
        self.query_qps = 0.0
        self.retention = 0.0
        self.latencies: deque[float] = deque(maxlen=window)
        self.retention_hist: deque[float] = deque(maxlen=window)
        self.mu_drift_hist: deque[float] = deque(maxlen=window)
        self._last_ingest_ts = time.time()
        self._ingest_bytes = 0
        self._last_query_ts = time.time()
        self._query_count = 0
        self.adjustments = 0
        self.running = True

    # ------------------------------------------------------------------
    def update(
        self,
        ingest_bytes: int = 0,
        query_latency: float | None = None,
        retention_value: float | None = None,
        mu_drift: float = 0.0,
    ) -> None:
        now = time.time()
        if ingest_bytes:
            self._ingest_bytes += ingest_bytes
            dt = now - self._last_ingest_ts or 1e-6
            rate = self._ingest_bytes / dt / 1_000_000
            self.ingest_rate = 0.8 * self.ingest_rate + 0.2 * rate if self.ingest_rate else rate
            self._ingest_bytes = 0
            self._last_ingest_ts = now
        if query_latency is not None:
            self.latencies.append(query_latency)
            self._query_count += 1
            dt = now - self._last_query_ts or 1e-6
            qps = self._query_count / dt
            self.query_qps = 0.8 * self.query_qps + 0.2 * qps if self.query_qps else qps
            self._query_count = 0
            self._last_query_ts = now
        if retention_value is not None:
            self.retention_hist.append(retention_value)
            self.retention = sum(self.retention_hist) / len(self.retention_hist)
        if mu_drift:
            self.mu_drift_hist.append(mu_drift)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        p50 = statistics.median(self.latencies) if self.latencies else 0.0
        p95 = (
            statistics.quantiles(self.latencies, n=20)[-1]
            if len(self.latencies) >= 20
            else p50
        )
        mu_d = sum(self.mu_drift_hist) / len(self.mu_drift_hist) if self.mu_drift_hist else 0.0
        return {
            "ingest_rate": self.ingest_rate,
            "query_qps": self.query_qps,
            "retention": self.retention,
            "latency_p50": p50,
            "latency_p95": p95,
            "mu_drift": mu_d,
        }
