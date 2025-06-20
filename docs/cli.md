# HERG CLI

Basic usage:

```bash
herg run-sim --radius 3 --gossip-every 4
herg bench ingest --n 100000
```

Flags:
- `--radius`          k-hop message radius
- `--alpha-u`         update rate
- `--alpha-b`         bundling rate
- `--eta`             learning rate
- `--block-size`      micro-vector block size
- `--backend`         hvlogfs backend (dax/spdk/stub)
- `--scrub-interval`  scrubber interval seconds
- `--gossip-every`    gossip tick interval
- `--tuner`           which auto-tuner to use (`bandit` default, `hill` legacy)
