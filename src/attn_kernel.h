#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch scaled dot-product attention forward.
// Layout: Q [B, H, Sq, D], K/V [B, H, Skv, D], O [B, H, Sq, D]
// Supports causal mask for self-attention when Sq == Skv.
enum AttnDType {
    ATTN_F16 = 0,
    ATTN_BF16 = 1,
    ATTN_F32 = 2,
};

void launch_attn_forward(const void* q, const void* k, const void* v, void* out,
                         int B, int H, int Sq, int Skv, int D,
                         bool causal, AttnDType dtype, hipStream_t stream);

#ifdef __cplusplus
}
#endif
