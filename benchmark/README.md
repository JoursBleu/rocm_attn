# ROCm Attention Benchmark

This folder contains a minimal benchmark to compare **SDPA** (PyTorch scaled dot-product attention) and **FlashAttention** on ROCm.

## What it does
- Creates random Q/K/V tensors
- Runs warmup iterations (skipped from timing)
- Measures average time per iteration for SDPA and FlashAttention
- Prints speedup ratio

## Script
- `bench_attention.py`

## Usage
```bash
# Example (ROCm GPU 3)
CUDA_VISIBLE_DEVICES=3 python3 benchmark/bench_attention.py \
  --batch 1 \
  --seqlen 512 \
  --heads 8 \
  --headdim 64 \
  --dtype fp16 \
  --iters 500 \
  --warmup 50
```

## Notes
- FlashAttention uses the ROCm Triton backend if available.
- If FlashAttention is unavailable, the script reports it and only prints SDPA timing.
- Increase `--seqlen`/`--heads` to evaluate larger workloads.
