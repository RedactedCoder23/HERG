# Quick Start

```bash
git clone --recurse-submodules https://github.com/you/herg-agent
cd herg-agent
docker compose up --build      # two-shard demo
# in another shell
python samples/feed.py         # push some seeds
python samples/query.py        # pull them back
```

feed.py and query.py are tiny demos that POST to /insert & /query.

### Required environment

- `SHARD_KEY` – two-hex shard identifier
- `S3_BUCKET` and `AWS_REGION` – object storage location for replication
- `NODE_KEY` – value for `X-API-KEY` header when inserting/querying
