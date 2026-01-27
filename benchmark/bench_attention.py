import argparse
import time
from typing import Optional

import torch
import torch.nn.functional as F


def _torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, iters: int, warmup: int) -> float:
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    _sync(q.device)
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    _sync(q.device)
    t0 = time.time()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    _sync(q.device)
    return (time.time() - t0) / iters


def bench_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, iters: int, warmup: int) -> Optional[float]:
    try:
        from unifolm_wma.modules.flash_attn_rocm_wrapper import flash_attn_func
    except Exception:
        try:
            from flash_attn import flash_attn_func
        except Exception:
            return None

    _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    _sync(q.device)
    for _ in range(warmup):
        _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    _sync(q.device)
    t0 = time.time()
    for _ in range(iters):
        _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    _sync(q.device)
    return (time.time() - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SDPA vs FlashAttention")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = _torch_dtype(args.dtype)
    device = torch.device(args.device)

    q_sdpa = torch.randn(args.batch, args.heads, args.seqlen, args.headdim, device=device, dtype=dtype)
    k_sdpa = torch.randn_like(q_sdpa)
    v_sdpa = torch.randn_like(q_sdpa)

    q_flash = q_sdpa.permute(0, 2, 1, 3).contiguous()
    k_flash = k_sdpa.permute(0, 2, 1, 3).contiguous()
    v_flash = v_sdpa.permute(0, 2, 1, 3).contiguous()

    sdpa_t = bench_sdpa(q_sdpa, k_sdpa, v_sdpa, args.iters, args.warmup)
    flash_t = bench_flash(q_flash, k_flash, v_flash, args.iters, args.warmup)

    print("=== Attention Benchmark ===")
    print(f"device: {device}")
    print(f"shape: B={args.batch}, H={args.heads}, L={args.seqlen}, D={args.headdim}")
    print(f"dtype: {args.dtype}")
    print(f"SDPA: {sdpa_t*1000:.3f} ms/iter")
    if flash_t is None:
        print("FlashAttention: unavailable")
    else:
        print(f"FlashAttention: {flash_t*1000:.3f} ms/iter")
        if flash_t > 0:
            print(f"Speedup: {sdpa_t/flash_t:.2f}x")


if __name__ == "__main__":
    main()
