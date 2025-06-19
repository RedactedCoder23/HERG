# ◇ CODEX_IMPLEMENT: stub LLM hook to concat capsule context
from typing import List
from herg.graph_caps.store import CapsuleStore


def hook_forward(
    hidden_states,
    token_seeds: List[bytes],
    store: CapsuleStore,
):
    import torch

    hidden_dim = hidden_states.shape[-1]
    cap_vecs = []
    for seed in token_seeds:
        cap = store.spawn(seed, ts=None)
        vec = torch.tensor(cap.vec, dtype=hidden_states.dtype, device=hidden_states.device)
        if vec.numel() > hidden_dim:
            vec = vec[:hidden_dim]
        elif vec.numel() < hidden_dim:
            vec = torch.nn.functional.pad(vec, (0, hidden_dim - vec.numel()))
        cap_vecs.append(vec)

    cap_batch = torch.stack(cap_vecs, dim=0)

    return torch.cat([hidden_states, cap_batch], dim=-1)

