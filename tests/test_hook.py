# ◇ CODEX_IMPLEMENT: test hook_forward shape
import torch
from herg.graph_caps.store import CapsuleStore
from integrations.llm_hook import hook_forward


def test_hook_forward():
    store = CapsuleStore()
    hidden = torch.zeros(1, 768)
    seed = b"x"
    out = hook_forward(hidden, [seed], store)
    assert out.shape[1] == hidden.shape[1] * 2
