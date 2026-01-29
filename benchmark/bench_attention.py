import argparse
import time
from typing import Optional, List, Tuple

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


def _parse_shapes(spec: str) -> List[Tuple[int, int, int, int, int]]:
    shapes: List[Tuple[int, int, int, int, int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        values = [int(x) for x in part.split(",")]
        if len(values) == 4:
            b, h, lq, d = values
            lk = lq
        elif len(values) == 5:
            b, h, lq, lk, d = values
        else:
            raise ValueError(
                "Invalid --shapes entry. Expected B,H,Lq,D or B,H,Lq,Lk,D; got: " + part
            )
        shapes.append((b, h, lq, lk, d))
    if not shapes:
        raise ValueError("--shapes is empty")
    return shapes


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


def bench_rocm_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, iters: int, warmup: int) -> Optional[float]:
    try:
        from rocm_attn_op import attn_forward
    except Exception:
        return None

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    _ = attn_forward(q, k, v)
    _sync(q.device)
    for _ in range(warmup):
        _ = attn_forward(q, k, v)
    _sync(q.device)
    t0 = time.time()
    for _ in range(iters):
        _ = attn_forward(q, k, v)
    _sync(q.device)
    return (time.time() - t0) / iters


def bench_sdpa_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    flash: bool,
    mem_efficient: bool,
    iters: int,
    warmup: int,
) -> float:
    def make_ctx():
        if hasattr(torch.nn.attention, "sdpa_kernel"):
            backend = (
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
                if flash
                else torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            )
            return torch.nn.attention.sdpa_kernel(backend)
        return torch.backends.cuda.sdp_kernel(
            enable_flash=flash,
            enable_math=False,
            enable_mem_efficient=mem_efficient,
        )

    with make_ctx():
        _ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
    _sync(q.device)
    with make_ctx():
        for _ in range(warmup):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
    _sync(q.device)
    t0 = time.time()
    with make_ctx():
        for _ in range(iters):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
    _sync(q.device)
    return (time.time() - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ROCm attn vs SDPA vs FlashAttention")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--seqlen-k", type=int, default=None, help="K/V sequence length (cross-attn)")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Semicolon-separated list: B,H,Lq,D or B,H,Lq,Lk,D",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--check-backends", action="store_true", help="Run SDPA backend checks")
    args = parser.parse_args()

    dtype = _torch_dtype(args.dtype)
    device = torch.device(args.device)

    if args.shapes:
        shapes = _parse_shapes(args.shapes)
    else:
        seqlen_k = args.seqlen_k if args.seqlen_k is not None else args.seqlen
        shapes = [(args.batch, args.heads, args.seqlen, seqlen_k, args.headdim)]

    for idx, (batch, heads, seqlen, seqlen_k, headdim) in enumerate(shapes, start=1):
        q_sdpa = torch.randn(batch, heads, seqlen, headdim, device=device, dtype=dtype)
        k_sdpa = torch.randn(batch, heads, seqlen_k, headdim, device=device, dtype=dtype)
        v_sdpa = torch.randn_like(k_sdpa)

        q_flash = q_sdpa.permute(0, 2, 1, 3).contiguous()
        k_flash = k_sdpa.permute(0, 2, 1, 3).contiguous()
        v_flash = v_sdpa.permute(0, 2, 1, 3).contiguous()

        if args.check_backends:
            sdpa_flash_t = bench_sdpa_backend(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                flash=True,
                mem_efficient=False,
                iters=args.iters,
                warmup=args.warmup,
            )
            sdpa_mem_t = bench_sdpa_backend(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                flash=False,
                mem_efficient=True,
                iters=args.iters,
                warmup=args.warmup,
            )
            print(f"SDPA flash-only: {sdpa_flash_t*1000:.3f} ms/iter")
            print(f"SDPA mem_efficient-only: {sdpa_mem_t*1000:.3f} ms/iter")

        sdpa_t = bench_sdpa(q_sdpa, k_sdpa, v_sdpa, args.iters, args.warmup)
        rocm_t = bench_rocm_attn(q_sdpa, k_sdpa, v_sdpa, args.iters, args.warmup)
        flash_t = bench_flash(q_flash, k_flash, v_flash, args.iters, args.warmup)

        title = "=== Attention Benchmark ===" if len(shapes) == 1 else f"=== Attention Benchmark ({idx}/{len(shapes)}) ==="
        print(title)
        print(f"device: {device}")
        if seqlen_k == seqlen:
            print(f"shape: B={batch}, H={heads}, L={seqlen}, D={headdim}")
        else:
            print(f"shape: B={batch}, H={heads}, Lq={seqlen}, Lk={seqlen_k}, D={headdim}")
        print(f"dtype: {args.dtype}")
        print(f"SDPA: {sdpa_t*1000:.3f} ms/iter")
        if rocm_t is None:
            print("ROCm attn: unavailable")
        else:
            print(f"ROCm attn: {rocm_t*1000:.3f} ms/iter")
            if rocm_t > 0:
                print(f"Speedup vs SDPA: {sdpa_t/rocm_t:.2f}x")
        if flash_t is None:
            print("FlashAttention: unavailable")
        else:
            print(f"FlashAttention: {flash_t*1000:.3f} ms/iter")
            if flash_t > 0:
                print(f"Speedup: {sdpa_t/flash_t:.2f}x")


if __name__ == "__main__":
    main()
