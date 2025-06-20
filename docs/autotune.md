# Auto-Tuner

HERG ships with a tiny auto-tuning daemon that adjusts runtime parameters
while the simulator runs.  It watches ingest throughput, query latency and
retention accuracy via `MetricStore`.  Every few seconds the daemon asks a
`HillClimbTuner` for parameter updates and applies them live to `Config`.

## Config keys
- `radius` – message propagation radius
- `block_size` – hypervector block granularity
- `alpha_u`, `alpha_b`, `eta`, `energy_drain` – learning constants

All changes are persisted back to `~/.config/herg/config.yml`.

## Safety
Parameters never exceed predefined bounds.  Failures are logged to
`~/.cache/herg/autotune.log` as JSON lines but never crash the main loop.
You can visualise the metrics with `scripts/plot_autotune.py autotune.log`.

## Bandit tuner

`BanditTuner` implements a simple \u03b5-greedy algorithm that explores
parameter tweaks at random early on then gradually exploits the best known
updates.  Select it with `--tuner bandit` on the CLI.
