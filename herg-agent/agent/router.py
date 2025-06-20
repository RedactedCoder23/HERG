"""
Stateless front-door that routes requests to the correct shard
via HTTP 307 redirect (cheapest) or proxy-pass (Envoy style).
"""

import os, hashlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse

SHARD_MAP = os.getenv("SHARD_MAP",
    "00-3f=http://node-a:9000,40-7f=http://node-b:9000,80-bf=http://node-c:9000,c0-ff=http://node-d:9000")

# parse env "xx-yy=url"
ranges = []
for entry in SHARD_MAP.split(","):
    span, url = entry.split("=")
    lo, hi = [int(x, 16) for x in span.split("-")]
    ranges.append((lo, hi, url))

app = FastAPI()

def _pick(prefix: str) -> str:
    val = int(prefix, 16)
    for lo, hi, url in ranges:
        if lo <= val <= hi:
            return url
    raise HTTPException(500, "no shard for prefix")

@app.api_route("/{path:path}", methods=["POST"])
async def redirect(path: str, req: Request):
    body = await req.body()
    prefix = body[:2].decode()       # client must prepend shard-prefix ascii
    dest = _pick(prefix)
    if dest.endswith(req.url.path):
        dest_url = dest
    else:
        dest_url = dest + req.url.path
    return RedirectResponse(url=dest_url, status_code=307)
