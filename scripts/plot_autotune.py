import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def main(log_path: str) -> None:
    path = Path(log_path)
    ts = []
    metrics = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ts.append(obj["timestamp"])
            for k, v in obj["metrics"].items():
                metrics.setdefault(k, []).append(v)
    for key, vals in metrics.items():
        plt.plot(ts[: len(vals)], vals, label=key)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Auto-tune metrics")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".cache" / "herg" / "autotune.log"))
