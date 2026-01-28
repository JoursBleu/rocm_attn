#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch naive scaled dot-product attention forward.
// Layout: Q, K, V, O are [B, H, S, D] contiguous in row-major order.
void launch_attn_forward(const float* q, const float* k, const float* v, float* out,
                         int B, int H, int S, int D, hipStream_t stream);

#ifdef __cplusplus
}
#endif
