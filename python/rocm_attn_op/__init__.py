import torch

from . import rocm_attn_ext


def attn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    return rocm_attn_ext.attn_forward(q, k, v, causal)
