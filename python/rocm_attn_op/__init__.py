import torch
import torch.nn.functional as F

from . import rocm_attn_ext


def _sdpa_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)


def attn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] >= 64:
        return _sdpa_fallback(q, k, v, causal)
    return rocm_attn_ext.attn_forward(q, k, v, causal)
