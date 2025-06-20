#!/usr/bin/env bash
set -e
python -m herg.cli auto-run \
       --nvec 20000 --ticks 500 \
       --radius 2 --tune-interval 10 --goal retention --tuner ${TUNER:-hill} --profile \
       | tee smoke_out.txt
mkdir -p artifacts
cp smoke_out.txt artifacts/
cp ~/.cache/herg/autotune.log artifacts/ || true
