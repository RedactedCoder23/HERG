# ◇ CODEX_IMPLEMENT: stub LLM hook to concat capsule context
from typing import List
from herg.graph_caps.store import CapsuleStore


def hook_forward(hidden_states, token_ids: List[int], store: CapsuleStore):
    import torch

    hidden_dim = hidden_states.shape[-1]
    cap_vecs = []
    for tid in token_ids:
        cap = store.read(tid)
        if cap is None:
            vec = torch.zeros(hidden_dim)
        else:
            vec = torch.tensor(cap.vec, dtype=torch.float32)
            if vec.numel() > hidden_dim:
                vec = vec[:hidden_dim]
            elif vec.numel() < hidden_dim:
                vec = torch.nn.functional.pad(vec, (0, hidden_dim - vec.numel()))
        cap_vecs.append(vec)
    cap_batch = torch.stack(cap_vecs, dim=0)
    return torch.cat([hidden_states, cap_batch], dim=-1)

