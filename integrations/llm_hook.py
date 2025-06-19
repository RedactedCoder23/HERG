# ◇ CODEX_IMPLEMENT: stub LLM hook to concat capsule context
from typing import List
from herg.graph_caps import CapsuleStore


def hook_forward(hidden_states, token_ids: List[int], store: CapsuleStore):
    import torch

    cap_vecs = []
    for tid in token_ids:
        cap = store.read(tid)
        if cap is None:
            cap_vecs.append(torch.zeros(store.dim))
        else:
            cap_vecs.append(torch.tensor(cap.vec, dtype=torch.float32))
    cap_batch = torch.stack(cap_vecs, dim=1)
    return torch.cat([hidden_states, cap_batch], dim=-1)

