import torch

from rocm_attn_op import attn_forward


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP is not available")

    q = torch.randn(1, 1, 2, 4, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, 2, 4, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 1, 2, 4, device="cuda", dtype=torch.float32)

    out = attn_forward(q, k, v)
    assert out.shape == q.shape, f"unexpected shape: {out.shape}"

    # correctness check against PyTorch SDPA
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    max_diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), f"mismatch vs sdpa: max_diff={max_diff}"

    torch.cuda.synchronize()
    print("torch attn_forward ok, shape=", tuple(out.shape), "max_diff=", max_diff)


if __name__ == "__main__":
    main()
