# ROCm Attention for Radeon

 Radeon 目标进行优化的 ROCm Attention（attn）库，提供高性能的注意力算子实现与可复用的算子接口，便于在训练与推理场景中集成。

## 特性

- 面向 Radeon GPU 的优化实现
- 关注性能、显存占用与数值稳定性
- 可扩展的算子设计，便于集成到现有模型/框架
- 适配常见注意力变体的扩展空间

## 目录结构

```
rocm_attn/
  src/              核心实现源码
  benchmark/        性能基准（如有）
  examples/         示例代码（如有）
  scripts/          工具脚本（如有）
  tests/            测试（如有）
```

> 注：部分目录可能尚在建设中，会随开发进度逐步补齐。

## 快速开始

### 1) 克隆仓库

```
git clone <repo-url>
cd rocm_attn
```

### 2) 构建与安装（占位）

ls

```
# TODO: 补充构建步骤
# 例如：cmake / hip / pip 安装等
```

### 3) 使用方式（占位）

```
# TODO: 补充使用示例
```

### 4) 性能基准（常见配置）

下面示例采用常见 LLM 配置：$B=1, H=32, L=2048, D=128, fp16$。

```
PYTHONPATH=./python python benchmark/bench_attention.py \
  --batch 1 --heads 32 --seqlen 2048 --headdim 128 --dtype fp16 --check-backends
```

### 5) 环境与结果（参考）

环境：
- OS: Linux 6.14.0-37-generic (Ubuntu 24.04.1)
- ROCm: 7.1.0
- PyTorch: 2.9.1+rocm7.1.0
- HIP: 7.1.25424-4179531dcd
- GPU: gfx1201 (Device ID 0x7551), 4x
- CPU: AMD Ryzen Threadripper PRO 9995WX 96-Cores

结果（$B=1, H=32, L=2048, D=128, fp16$）：
- SDPA flash-only: 1.859 ms/iter
- SDPA mem-efficient-only: 1.589 ms/iter
- SDPA: 1.413 ms/iter
- ROCm attn: 1.310 ms/iter
- FlashAttention: 1.615 ms/iter


## 设计目标

- 为 Radeon 提供可落地的高性能注意力实现
- 在可维护性与性能之间取得平衡
- 为后续算子优化与新特性扩展提供清晰结构

## 兼容性

- 目标平台：Radeon GPU
- 后端：ROCm

> 具体支持的 ROCm 版本与硬件型号会在后续补充。

## 路线图（简要）

- [ ] 完成基础注意力算子实现
- [ ] 性能基准与回归测试
- [ ] 增加更多变体与可选优化

## 贡献指南

 Issue 与 PR。若要贡献代码，请确保：

- 通过基础测试与格式检查
- 说明变更目的与性能/正确性影响

## 许可证

 Apache License 2.0。
