import torch

from rocm_attn_op import attn_forward


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP is not available")

    torch.manual_seed(0)

    # self-attention (non-causal)
    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = attn_forward(q, k, v, False)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    max_diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), f"self non-causal mismatch: max_diff={max_diff}"
    print("self non-causal ok, max_diff=", max_diff)

    # self-attention (causal)
    out_causal = attn_forward(q, k, v, True)
    ref_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    max_diff = (out_causal - ref_causal).abs().max().item()
    assert torch.allclose(out_causal, ref_causal, rtol=1e-3, atol=1e-3), f"self causal mismatch: max_diff={max_diff}"
    print("self causal ok, max_diff=", max_diff)

    # cross-attention (non-causal)
    qx = torch.randn(1, 2, 6, 16, device="cuda", dtype=torch.float32)
    kx = torch.randn(1, 2, 9, 16, device="cuda", dtype=torch.float32)
    vx = torch.randn_like(kx)
    out_cross = attn_forward(qx, kx, vx, False)
    ref_cross = torch.nn.functional.scaled_dot_product_attention(qx, kx, vx, dropout_p=0.0, is_causal=False)
    max_diff = (out_cross - ref_cross).abs().max().item()
    assert torch.allclose(out_cross, ref_cross, rtol=1e-3, atol=1e-3), f"cross non-causal mismatch: max_diff={max_diff}"
    print("cross non-causal ok, max_diff=", max_diff)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
