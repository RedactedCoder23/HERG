"""
Asynchronous push/pull of closed hvlogfs chunks to object storage.
"""

import asyncio, os, hashlib, logging
from pathlib import Path
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "herg-brain")
S3_PREFIX = os.getenv("S3_PREFIX", "chunks/")
LOCAL_DIR  = Path(os.getenv("HVLOG_DIR", "/data/hvlog"))
PARALLEL = 4

s3 = boto3.client("s3")

# ---------------------------------------------------------------------------

def _etag(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

@retry(stop=stop_after_attempt(5), wait=wait_exponential())
async def _upload(chunk: Path):
    key = f"{S3_PREFIX}{chunk.name}"
    etag_local = _etag(chunk)
    try:
        obj = s3.head_object(Bucket=S3_BUCKET, Key=key)
        if obj["ETag"].strip('"') == etag_local:
            log.debug("%s already in bucket", chunk.name)
            return
    except s3.exceptions.NoSuchKey:
        pass

    log.info("Uploading %s → s3://%s/%s", chunk, S3_BUCKET, key)
    await asyncio.to_thread(s3.upload_file, str(chunk), S3_BUCKET, key)

@retry(stop=stop_after_attempt(5), wait=wait_exponential())
async def _download(key: str):
    dest = LOCAL_DIR / key.split("/")[-1]
    if dest.exists():
        return
    log.info("Fetching %s", key)
    await asyncio.to_thread(s3.download_file, S3_BUCKET, key, str(dest))
    os.sync()

async def push_closed_chunks():
    tasks = []
    for chunk in LOCAL_DIR.glob("chunk-*.dat"):
        tasks.append(_upload(chunk))
        if len(tasks) >= PARALLEL:
            await asyncio.gather(*tasks)
            tasks.clear()
    if tasks:
        await asyncio.gather(*tasks)

async def hydrate_prefix(prefix: str):
    """
    Pull any remote chunks whose filename starts with this prefix
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}{prefix}"):
        tasks = []
        for obj in page.get("Contents", []):
            key = obj["Key"]
            tasks.append(_download(key))
            if len(tasks) >= PARALLEL:
                await asyncio.gather(*tasks)
                tasks.clear()
        if tasks:
            await asyncio.gather(*tasks)
