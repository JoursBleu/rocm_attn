#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <math.h>
#include <rocwmma/rocwmma.hpp>

#include "attn_kernel.h"

using namespace rocwmma;

__device__ __forceinline__ int idx4(int b, int h, int s, int d, int H, int S, int D) {
    return (((b * H + h) * S + s) * D + d);
}

template <typename T>
struct AttnTraits;

template <>
struct AttnTraits<float> {
    __device__ __forceinline__ static float load(const float* p) { return *p; }
    __device__ __forceinline__ static float store(float v) { return v; }
};

template <>
struct AttnTraits<__half> {
    __device__ __forceinline__ static float load(const __half* p) { return __half2float(*p); }
    __device__ __forceinline__ static __half store(float v) { return __float2half_rn(v); }
};

template <>
struct AttnTraits<hip_bfloat16> {
    __device__ __forceinline__ static float load(const hip_bfloat16* p) { return static_cast<float>(*p); }
    __device__ __forceinline__ static hip_bfloat16 store(float v) { return hip_bfloat16(v); }
};

template <typename T>
__global__ void attn_forward_kernel_simple(const T* q, const T* k, const T* v, T* out,
                                           int B, int H, int Sq, int Skv, int D, int causal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Sq;
    if (tid >= total) return;

    int t = tid;
    int s = t % Sq;
    t /= Sq;
    int h = t % H;
    int b = t / H;

    float scale = rsqrtf((float)D);

    float max_score = -INFINITY;
    for (int ks = 0; ks < Skv; ++ks) {
        if (causal && ks > s) continue;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            int q_idx = idx4(b, h, s, d, H, Sq, D);
            int k_idx = idx4(b, h, ks, d, H, Skv, D);
            score += AttnTraits<T>::load(q + q_idx) * AttnTraits<T>::load(k + k_idx);
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    for (int d = 0; d < D; ++d) {
        int o_idx = idx4(b, h, s, d, H, Sq, D);
        out[o_idx] = AttnTraits<T>::store(0.0f);
    }

    for (int ks = 0; ks < Skv; ++ks) {
        if (causal && ks > s) continue;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            int q_idx = idx4(b, h, s, d, H, Sq, D);
            int k_idx = idx4(b, h, ks, d, H, Skv, D);
            score += AttnTraits<T>::load(q + q_idx) * AttnTraits<T>::load(k + k_idx);
        }
        score = score * scale - max_score;
        float w = expf(score);
        denom += w;
        for (int d = 0; d < D; ++d) {
            int v_idx = idx4(b, h, ks, d, H, Skv, D);
            int o_idx = idx4(b, h, s, d, H, Sq, D);
            float acc = AttnTraits<T>::load(out + o_idx);
            acc += w * AttnTraits<T>::load(v + v_idx);
            out[o_idx] = AttnTraits<T>::store(acc);
        }
    }

    float inv_denom = 1.0f / denom;
    for (int d = 0; d < D; ++d) {
        int o_idx = idx4(b, h, s, d, H, Sq, D);
        float acc = AttnTraits<T>::load(out + o_idx);
        out[o_idx] = AttnTraits<T>::store(acc * inv_denom);
    }
}

template <typename T>
__global__ void attn_forward_kernel_block(const T* q, const T* k, const T* v, T* out,
                                          int B, int H, int Sq, int Skv, int D, int causal) {
    int query_idx = blockIdx.x;
    int total = B * H * Sq;
    if (query_idx >= total) return;

    int t = query_idx;
    int s = t % Sq;
    t /= Sq;
    int h = t % H;
    int b = t / H;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float smem[];
    float* red = smem;
    float* scores = red + threads;
    float* qbuf = scores + Skv;

    if (tid < D) {
        int q_idx = idx4(b, h, s, tid, H, Sq, D);
        qbuf[tid] = AttnTraits<T>::load(q + q_idx);
    }
    __syncthreads();

    float max_score = -INFINITY;
    for (int ks = 0; ks < Skv; ++ks) {
        if (causal && ks > s) {
            if (tid == 0) scores[ks] = -INFINITY;
            __syncthreads();
            continue;
        }
        float partial = 0.0f;
        if (tid < D) {
            int k_idx = idx4(b, h, ks, tid, H, Skv, D);
            partial = qbuf[tid] * AttnTraits<T>::load(k + k_idx);
        }
        red[tid] = partial;
        __syncthreads();

        for (int stride = threads / 2; stride >= 1; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float score = red[0] * rsqrtf((float)D);
            scores[ks] = score;
            if (score > max_score) max_score = score;
            red[0] = max_score;
        }
        __syncthreads();
        max_score = red[0];
    }

    float local_sum = 0.0f;
    for (int ks = tid; ks < Skv; ks += threads) {
        float score = scores[ks];
        local_sum += expf(score - max_score);
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int stride = threads / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    float denom = red[0];
    float inv_denom = 1.0f / denom;

    if (tid < D) {
        float acc = 0.0f;
        for (int ks = 0; ks < Skv; ++ks) {
            float w = expf(scores[ks] - max_score) * inv_denom;
            int v_idx = idx4(b, h, ks, tid, H, Skv, D);
            acc += w * AttnTraits<T>::load(v + v_idx);
        }
        int o_idx = idx4(b, h, s, tid, H, Sq, D);
        out[o_idx] = AttnTraits<T>::store(acc);
    }
}

__global__ void attn_forward_kernel_wmma_f16(const __half* q, const __half* k, const __half* v, __half* out,
                                             int B, int H, int Sq, int Skv, int D, int causal) {
    int bh = blockIdx.x;
    int tile_s = blockIdx.y * 16;
    int b = bh / H;
    int h = bh % H;
    if (b >= B) return;

    int tid = threadIdx.x;

    __shared__ __half q_sh[16][16];
    __shared__ __half k_sh[16][16];
    __shared__ __half v_sh[16][16];
    __shared__ float scores_sh[16][16];
    __shared__ float out_sh[16][16];
    __shared__ float max_sh[16];
    __shared__ float sum_sh[16];

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c = idx % 16;
        out_sh[r][c] = 0.0f;
    }
    for (int r = tid; r < 16; r += blockDim.x) {
        max_sh[r] = -INFINITY;
        sum_sh[r] = 0.0f;
    }
    __syncthreads();

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c = idx % 16;
        int s = tile_s + r;
        if (s < Sq) {
            int q_idx = idx4(b, h, s, c, H, Sq, D);
            q_sh[r][c] = q[q_idx];
        } else {
            q_sh[r][c] = __float2half(0.0f);
        }
    }
    __syncthreads();

    for (int kb = 0; kb < Skv; kb += 16) {
        for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
            int d = idx / 16;
            int n = idx % 16;
            int ks = kb + n;
            if (ks < Skv) {
                int k_idx = idx4(b, h, ks, d, H, Skv, D);
                k_sh[d][n] = k[k_idx];
                v_sh[n][d] = v[k_idx];
            } else {
                k_sh[d][n] = __float2half(0.0f);
                v_sh[n][d] = __float2half(0.0f);
            }
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, __half, row_major> a;
        fragment<matrix_b, 16, 16, 16, __half, row_major> bfrag;
        fragment<accumulator, 16, 16, 16, float> acc;
        fill_fragment(acc, 0.0f);
        load_matrix_sync(a, &q_sh[0][0], 16);
        load_matrix_sync(bfrag, &k_sh[0][0], 16);
        mma_sync(acc, a, bfrag, acc);
        store_matrix_sync(&scores_sh[0][0], acc, 16, mem_row_major);
        __syncthreads();

        for (int r = tid; r < 16; r += blockDim.x) {
            int s = tile_s + r;
            if (s >= Sq) continue;
            float tile_max = -INFINITY;
            for (int c = 0; c < 16; ++c) {
                int ks = kb + c;
                float score = scores_sh[r][c] * rsqrtf((float)D);
                if (ks >= Skv || (causal && ks > s)) {
                    score = -INFINITY;
                }
                scores_sh[r][c] = score;
                if (score > tile_max) tile_max = score;
            }

            float prev_max = max_sh[r];
            float new_max = fmaxf(prev_max, tile_max);
            float scale = (prev_max == -INFINITY) ? 0.0f : expf(prev_max - new_max);
            float sum = sum_sh[r] * scale;
            for (int d = 0; d < 16; ++d) {
                out_sh[r][d] *= scale;
            }

            for (int c = 0; c < 16; ++c) {
                float score = scores_sh[r][c];
                if (score == -INFINITY) continue;
                float w = expf(score - new_max);
                sum += w;
                for (int d = 0; d < 16; ++d) {
                    out_sh[r][d] += w * __half2float(v_sh[c][d]);
                }
            }

            max_sh[r] = new_max;
            sum_sh[r] = sum;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int d = idx % 16;
        int s = tile_s + r;
        if (s < Sq) {
            float denom = sum_sh[r];
            float val = (denom > 0.0f) ? (out_sh[r][d] / denom) : 0.0f;
            int o_idx = idx4(b, h, s, d, H, Sq, D);
            out[o_idx] = __float2half_rn(val);
        }
    }
}

extern "C" void launch_attn_forward(const void* q, const void* k, const void* v, void* out,
                                     int B, int H, int Sq, int Skv, int D,
                                     bool causal, AttnDType dtype, hipStream_t stream) {
    int total = B * H * Sq;
    int threads = 1;
    while (threads < D) threads <<= 1;
    if (threads > 256) threads = 256;
    size_t smem_bytes = (size_t)(threads + Skv + D) * sizeof(float);
    bool use_block = (D <= 256) && (smem_bytes <= 48 * 1024);
    int blocks_simple = (total + 256 - 1) / 256;
    int causal_flag = causal ? 1 : 0;

    if (dtype == ATTN_F16 && D == 16) {
        dim3 grid(B * H, (Sq + 15) / 16);
        hipLaunchKernelGGL(attn_forward_kernel_wmma_f16, grid, dim3(32), 0, stream,
                           static_cast<const __half*>(q), static_cast<const __half*>(k),
                           static_cast<const __half*>(v), static_cast<__half*>(out),
                           B, H, Sq, Skv, D, causal_flag);
        return;
    }

    switch (dtype) {
        case ATTN_F16:
            if (use_block) {
                hipLaunchKernelGGL(attn_forward_kernel_block<__half>, dim3(total), dim3(threads), smem_bytes, stream,
                                   static_cast<const __half*>(q), static_cast<const __half*>(k),
                                   static_cast<const __half*>(v), static_cast<__half*>(out),
                                   B, H, Sq, Skv, D, causal_flag);
            } else {
                hipLaunchKernelGGL(attn_forward_kernel_simple<__half>, dim3(blocks_simple), dim3(256), 0, stream,
                                   static_cast<const __half*>(q), static_cast<const __half*>(k),
                                   static_cast<const __half*>(v), static_cast<__half*>(out),
                                   B, H, Sq, Skv, D, causal_flag);
            }
            break;
        case ATTN_BF16:
            if (use_block) {
                hipLaunchKernelGGL(attn_forward_kernel_block<hip_bfloat16>, dim3(total), dim3(threads), smem_bytes, stream,
                                   static_cast<const hip_bfloat16*>(q), static_cast<const hip_bfloat16*>(k),
                                   static_cast<const hip_bfloat16*>(v), static_cast<hip_bfloat16*>(out),
                                   B, H, Sq, Skv, D, causal_flag);
            } else {
                hipLaunchKernelGGL(attn_forward_kernel_simple<hip_bfloat16>, dim3(blocks_simple), dim3(256), 0, stream,
                                   static_cast<const hip_bfloat16*>(q), static_cast<const hip_bfloat16*>(k),
                                   static_cast<const hip_bfloat16*>(v), static_cast<hip_bfloat16*>(out),
                                   B, H, Sq, Skv, D, causal_flag);
            }
            break;
        case ATTN_F32:
        default:
            if (use_block) {
                hipLaunchKernelGGL(attn_forward_kernel_block<float>, dim3(total), dim3(threads), smem_bytes, stream,
                                   static_cast<const float*>(q), static_cast<const float*>(k),
                                   static_cast<const float*>(v), static_cast<float*>(out),
                                   B, H, Sq, Skv, D, causal_flag);
            } else {
                hipLaunchKernelGGL(attn_forward_kernel_simple<float>, dim3(blocks_simple), dim3(256), 0, stream,
                                   static_cast<const float*>(q), static_cast<const float*>(k),
                                   static_cast<const float*>(v), static_cast<float*>(out),
                                   B, H, Sq, Skv, D, causal_flag);
            }
            break;
    }
}
