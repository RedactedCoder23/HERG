import os
import sys
import importlib
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "herg-agent"))

from agent import utils, node


def _make_client(with_key: bool):
    os.environ['SHARD_KEY'] = 'aa'
    os.environ['HVLOG_DIR'] = '/tmp/hvlog'
    if with_key:
        os.environ['NODE_KEY'] = 'test'
    else:
        os.environ.pop('NODE_KEY', None)
    importlib.reload(node)
    return TestClient(node.app)


def test_insert_requires_key():
    try:
        client = _make_client(with_key=True)
        resp = client.post('/insert', data=utils.add_prefix({'seed': 'hello'}))
        assert resp.status_code == 401
    finally:
        os.environ.pop('NODE_KEY', None)


def test_insert_with_key_ok():
    try:
        client = _make_client(with_key=True)
        resp = client.post('/insert', data=utils.add_prefix({'seed': 'hello'}), headers={'x-api-key': 'test'})
        assert resp.status_code == 200
    finally:
        os.environ.pop('NODE_KEY', None)
