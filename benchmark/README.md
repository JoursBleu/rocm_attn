# ROCm 注意力基准

benchmark/ ROCm 环境下对比 **SDPA**（PyTorch scaled dot-product attention）与 **FlashAttention** 的性能。

## 功能说明
- 生成随机 Q/K/V 张量
- 先进行若干次 warmup（不计入计时）
- 统计 SDPA 与 FlashAttention 的平均单次耗时
- 输出速度对比

## 脚本
- `bench_attention.py`

## 用法
```bash
# 示例（使用 ROCm GPU 3）
CUDA_VISIBLE_DEVICES=3 python3 benchmark/bench_attention.py   --batch 1   --seqlen 512   --heads 8   --headdim 64   --dtype fp16   --iters 500   --warmup 50   --backend both

# 仅测 SDPA
CUDA_VISIBLE_DEVICES=3 python3 benchmark/bench_attention.py   --backend sdpa

# 仅测 FlashAttention
CUDA_VISIBLE_DEVICES=3 python3 benchmark/bench_attention.py   --backend flash
```

## 备注
- `--backend` 可选：`sdpa` / `flash` / `both`（默认）。
- 若 FlashAttention 不可用，会提示并跳过。
- 可增大 `--seqlen` / `--heads` 以测试更大规模。
