#!/usr/bin/env bash
set -e
export SHARD_KEY=aa
export HVLOG_DIR=/tmp/hvlog
PYTHONPATH=herg-agent:$PYTHONPATH uvicorn agent.node:app --port 8000 --host 0.0.0.0 &
SERV_PID=$!
sleep 2
python -m herg.cli auto-run \
       --nvec 20000 --ticks 500 \
       --radius 2 --tune-interval 10 --goal retention --tuner ${TUNER:-hill} --profile \
       | tee smoke_out.txt
curl -H "x-api-key: smoke" http://localhost:8000/health
kill $SERV_PID
mkdir -p artifacts
cp smoke_out.txt artifacts/
cp ~/.cache/herg/autotune.log artifacts/ || true
