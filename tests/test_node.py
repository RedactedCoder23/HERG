import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "herg-agent"))

os.environ['SHARD_KEY'] = 'aa'
os.environ['HVLOG_DIR'] = '/tmp/hvlog'
try:
    from agent import node
    from agent import utils
except ModuleNotFoundError:
    pytest.skip("agent deps missing", allow_module_level=True)

client = TestClient(node.app)


def test_insert_and_query():
    payload = {'seed': 'hello', 'text': 'hi', 'reward': 0.1}
    r = client.post('/insert', data=utils.add_prefix(payload))
    assert r.status_code == 200
    q = {'seed': 'hello', 'top_k': 1}
    out = client.post('/query', data=utils.add_prefix(q)).json()
    assert out and out[0]['dist'] == 0.0
