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
    float scale_log2 = scale * 1.4426950408889634f; // log2(e)

    float max_score = -INFINITY;
    for (int ks = 0; ks < Skv; ++ks) {
        if (causal && ks > s) continue;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            int q_idx = idx4(b, h, s, d, H, Sq, D);
            int k_idx = idx4(b, h, ks, d, H, Skv, D);
            score += AttnTraits<T>::load(q + q_idx) * AttnTraits<T>::load(k + k_idx);
        }
        score *= scale_log2;
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
        score = score * scale_log2 - max_score;
        float w = exp2f(score);
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
    float scale_log2 = rsqrtf((float)D) * 1.4426950408889634f; // log2(e)

    constexpr int TILE_K = 32;
    extern __shared__ float smem[];
    float* red = smem;                        // threads
    float* qbuf = red + threads;              // D
    float* kbuf = qbuf + D;                   // TILE_K * D
    float* vbuf = kbuf + TILE_K * D;          // TILE_K * D
    float* scores = vbuf + TILE_K * D;        // TILE_K
    float* meta = scores + TILE_K;            // 3 floats: max, sum, scale

    if (tid < D) {
        int q_idx = idx4(b, h, s, tid, H, Sq, D);
        qbuf[tid] = AttnTraits<T>::load(q + q_idx);
    }
    if (tid == 0) {
        meta[0] = -INFINITY;
        meta[1] = 0.0f;
        meta[2] = 1.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int ks0 = 0; ks0 < Skv; ks0 += TILE_K) {
        int tile_len = (ks0 + TILE_K <= Skv) ? TILE_K : (Skv - ks0);
        int tile_elems = tile_len * D;
        for (int idx = tid; idx < tile_elems; idx += threads) {
            int ks = idx / D;
            int d = idx - ks * D;
            int k_idx = idx4(b, h, ks0 + ks, d, H, Skv, D);
            int v_idx = idx4(b, h, ks0 + ks, d, H, Skv, D);
            kbuf[ks * D + d] = AttnTraits<T>::load(k + k_idx);
            vbuf[ks * D + d] = AttnTraits<T>::load(v + v_idx);
        }
        __syncthreads();

        for (int ks = 0; ks < tile_len; ++ks) {
            float partial = 0.0f;
            if (tid < D) {
                partial = qbuf[tid] * kbuf[ks * D + tid];
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
                float score = red[0] * scale_log2;
                if (causal && (ks0 + ks) > s) score = -INFINITY;
                scores[ks] = score;
            }
            __syncthreads();
        }

        if (tid == 0) {
            float tile_max = -INFINITY;
            for (int ks = 0; ks < tile_len; ++ks) {
                float sc = scores[ks];
                if (sc > tile_max) tile_max = sc;
            }
            float prev_max = meta[0];
            float new_max = fmaxf(prev_max, tile_max);
            float scale = (prev_max == -INFINITY) ? 0.0f : exp2f(prev_max - new_max);
            float tile_sum = 0.0f;
            for (int ks = 0; ks < tile_len; ++ks) {
                float sc = scores[ks];
                if (sc != -INFINITY) tile_sum += exp2f(sc - new_max);
            }
            meta[0] = new_max;
            meta[1] = meta[1] * scale + tile_sum;
            meta[2] = scale;
        }
        __syncthreads();

        if (tid < D) {
            float scale = meta[2];
            float new_max = meta[0];
            acc *= scale;
            for (int ks = 0; ks < tile_len; ++ks) {
                float sc = scores[ks];
                if (sc == -INFINITY) continue;
                float w = exp2f(sc - new_max);
                acc += w * vbuf[ks * D + tid];
            }
        }
        __syncthreads();
    }

    float denom = meta[1];
    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    if (tid < D) {
        int o_idx = idx4(b, h, s, tid, H, Sq, D);
        out[o_idx] = AttnTraits<T>::store(acc * inv_denom);
    }
}
__global__ __launch_bounds__(128) void attn_forward_kernel_wmma_f16(const __half* q, const __half* k, const __half* v, __half* out,
                                             int B, int H, int Sq, int Skv, int D, int causal) {
    int bh = blockIdx.x;
    int tile_s = blockIdx.y * 32;
    int b = bh / H;
    int h = bh % H;
    if (b >= B) return;

    int tid = threadIdx.x;
    const float scale_log2 = rsqrtf((float)D) * 1.4426950408889634f; // log2(e)

    __shared__ __half q_sh0[16][16];
    __shared__ __half q_sh1[16][16];
    __shared__ __half k_sh0[16][16];
    __shared__ __half k_sh1[16][16];
    __shared__ __half v_sh0[16][16];
    __shared__ __half v_sh1[16][16];
    __shared__ float scores_sh[16][16];
    __shared__ __half p_sh[16][16];
    __shared__ float out_sh0[16][16];
    __shared__ float out_sh1[16][16];
    __shared__ float max_sh[32];
    __shared__ float sum_sh[32];

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c = idx % 16;
        out_sh0[r][c] = 0.0f;
        out_sh1[r][c] = 0.0f;
    }
    for (int r = tid; r < 32; r += blockDim.x) {
        max_sh[r] = -INFINITY;
        sum_sh[r] = 0.0f;
    }
    __syncthreads();

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c = idx % 16;
        if (c & 1) continue;
        int s0 = tile_s + r;
        int s1 = tile_s + 16 + r;
        if (s0 < Sq) {
            int q_idx0 = idx4(b, h, s0, c, H, Sq, D);
            const __half2 qv0 = *reinterpret_cast<const __half2*>(q + q_idx0);
            q_sh0[r][c] = __low2half(qv0);
            q_sh0[r][c + 1] = __high2half(qv0);
        } else {
            q_sh0[r][c] = __float2half(0.0f);
            q_sh0[r][c + 1] = __float2half(0.0f);
        }
        if (s1 < Sq) {
            int q_idx1 = idx4(b, h, s1, c, H, Sq, D);
            const __half2 qv1 = *reinterpret_cast<const __half2*>(q + q_idx1);
            q_sh1[r][c] = __low2half(qv1);
            q_sh1[r][c + 1] = __high2half(qv1);
        } else {
            q_sh1[r][c] = __float2half(0.0f);
            q_sh1[r][c + 1] = __float2half(0.0f);
        }
    }
    __syncthreads();

    auto load_kv = [&](int kb, __half k_sh[16][16], __half v_sh[16][16]) {
        if (tid >= 32) return;
        int lane = tid & 31;
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            int d = idx / 16;
            int n = idx % 16;
            if (d & 1) continue;
            int ks = kb + n;
            if (ks < Skv) {
                int k_idx = idx4(b, h, ks, d, H, Skv, D);
                const __half2 kv = *reinterpret_cast<const __half2*>(k + k_idx);
                const __half2 vv = *reinterpret_cast<const __half2*>(v + k_idx);
                k_sh[d][n] = __low2half(kv);
                k_sh[d + 1][n] = __high2half(kv);
                v_sh[n][d] = __low2half(vv);
                v_sh[n][d + 1] = __high2half(vv);
            } else {
                k_sh[d][n] = __float2half(0.0f);
                k_sh[d + 1][n] = __float2half(0.0f);
                v_sh[n][d] = __float2half(0.0f);
                v_sh[n][d + 1] = __float2half(0.0f);
            }
        }
    };

    load_kv(0, k_sh0, v_sh0);
    __syncthreads();

    for (int kb = 0; kb < Skv; kb += 16) {
        int buf = (kb / 16) & 1;
        __half (*k_sh)[16] = buf ? k_sh1 : k_sh0;
        __half (*v_sh)[16] = buf ? v_sh1 : v_sh0;

        int next_kb = kb + 16;
        if (next_kb < Skv) {
            if (buf == 0) {
                load_kv(next_kb, k_sh1, v_sh1);
            } else {
                load_kv(next_kb, k_sh0, v_sh0);
            }
        }

        fragment<matrix_a, 16, 16, 16, __half, row_major> a0;
        fragment<matrix_a, 16, 16, 16, __half, row_major> a1;
        fragment<matrix_b, 16, 16, 16, __half, row_major> bfrag;
        fragment<accumulator, 16, 16, 16, float> acc;

        fill_fragment(acc, 0.0f);
        load_matrix_sync(a0, &q_sh0[0][0], 16);
        load_matrix_sync(bfrag, &k_sh[0][0], 16);
        mma_sync(acc, a0, bfrag, acc);
        store_matrix_sync(&scores_sh[0][0], acc, 16, mem_row_major);
        __syncthreads();

        // warp-level softmax + output for tile 0
        {
            int warp = tid >> 5;
            int lane = tid & 31;
            int row_in_warp = lane >> 4; // 0 or 1
            int col = lane & 15;
            for (int row_base = 0; row_base < 16; row_base += 8) {
                int r = row_base + warp * 2 + row_in_warp;
                if (r >= 16) continue;
                int s = tile_s + r;
                if (s >= Sq) continue;

                float score = scores_sh[r][col] * scale_log2;
                if ((kb + col) >= Skv || (causal && (kb + col) > s)) {
                    score = -INFINITY;
                }
                scores_sh[r][col] = score;

                float maxv = score;
                // reduce max within 16-lane group
                for (int offset = 8; offset > 0; offset >>= 1) {
                    float v = __shfl_xor(maxv, offset, 16);
                    if (v > maxv) maxv = v;
                }

                float prev_max = max_sh[r];
                float new_max = fmaxf(prev_max, maxv);
                float scale = (prev_max == -INFINITY) ? 0.0f : exp2f(prev_max - new_max);

                out_sh0[r][col] *= scale;

                float w = (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
                float sum = w;
                for (int offset = 8; offset > 0; offset >>= 1) {
                    sum += __shfl_xor(sum, offset, 16);
                }

                if (col == 0) {
                    max_sh[r] = new_max;
                    sum_sh[r] = sum_sh[r] * scale + sum;
                }

            }
        }
        __syncthreads();

        for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
            int r = idx / 16;
            int c = idx % 16;
            float score = scores_sh[r][c];
            float w = (score == -INFINITY) ? 0.0f : exp2f(score - max_sh[r]);
            p_sh[r][c] = __float2half_rn(w);
        }
        __syncthreads();

        load_matrix_sync(a0, &p_sh[0][0], 16);
        load_matrix_sync(bfrag, &v_sh[0][0], 16);
        load_matrix_sync(acc, &out_sh0[0][0], 16, mem_row_major);
        mma_sync(acc, a0, bfrag, acc);
        store_matrix_sync(&out_sh0[0][0], acc, 16, mem_row_major);
        __syncthreads();

        fill_fragment(acc, 0.0f);
        load_matrix_sync(a1, &q_sh1[0][0], 16);
        load_matrix_sync(bfrag, &k_sh[0][0], 16);
        mma_sync(acc, a1, bfrag, acc);
        store_matrix_sync(&scores_sh[0][0], acc, 16, mem_row_major);
        __syncthreads();

        // warp-level softmax + output for tile 1
        {
            int warp = tid >> 5;
            int lane = tid & 31;
            int row_in_warp = lane >> 4; // 0 or 1
            int col = lane & 15;
            for (int row_base = 0; row_base < 16; row_base += 8) {
                int r = row_base + warp * 2 + row_in_warp;
                if (r >= 16) continue;
                int s = tile_s + 16 + r;
                if (s >= Sq) continue;

                float score = scores_sh[r][col] * scale_log2;
                if ((kb + col) >= Skv || (causal && (kb + col) > s)) {
                    score = -INFINITY;
                }
                scores_sh[r][col] = score;

                float maxv = score;
                for (int offset = 8; offset > 0; offset >>= 1) {
                    float v = __shfl_xor(maxv, offset, 16);
                    if (v > maxv) maxv = v;
                }

                int row = 16 + r;
                float prev_max = max_sh[row];
                float new_max = fmaxf(prev_max, maxv);
                float scale = (prev_max == -INFINITY) ? 0.0f : exp2f(prev_max - new_max);

                out_sh1[r][col] *= scale;

                float w = (score == -INFINITY) ? 0.0f : exp2f(score - new_max);
                float sum = w;
                for (int offset = 8; offset > 0; offset >>= 1) {
                    sum += __shfl_xor(sum, offset, 16);
                }

                if (col == 0) {
                    max_sh[row] = new_max;
                    sum_sh[row] = sum_sh[row] * scale + sum;
                }

            }
        }
        for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
            int r = idx / 16;
            int c = idx % 16;
            float score = scores_sh[r][c];
            float w = (score == -INFINITY) ? 0.0f : exp2f(score - max_sh[16 + r]);
            p_sh[r][c] = __float2half_rn(w);
        }
        __syncthreads();

        load_matrix_sync(a1, &p_sh[0][0], 16);
        load_matrix_sync(bfrag, &v_sh[0][0], 16);
        load_matrix_sync(acc, &out_sh1[0][0], 16, mem_row_major);
        mma_sync(acc, a1, bfrag, acc);
        store_matrix_sync(&out_sh1[0][0], acc, 16, mem_row_major);
        __syncthreads();
    }

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int d = idx % 16;
        int s0 = tile_s + r;
        if (s0 < Sq) {
            float denom = sum_sh[r];
            float val = (denom > 0.0f) ? (out_sh0[r][d] / denom) : 0.0f;
            int o_idx = idx4(b, h, s0, d, H, Sq, D);
            out[o_idx] = __float2half_rn(val);
        }
        int s1 = tile_s + 16 + r;
        if (s1 < Sq) {
            float denom = sum_sh[16 + r];
            float val = (denom > 0.0f) ? (out_sh1[r][d] / denom) : 0.0f;
            int o_idx = idx4(b, h, s1, d, H, Sq, D);
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
    const int tile_k = 32;
    size_t smem_bytes = (size_t)(threads + D + tile_k * D * 2 + tile_k + 3) * sizeof(float);
    bool use_block = (D <= 128) && (smem_bytes <= 48 * 1024);
    int blocks_simple = (total + 256 - 1) / 256;
    int causal_flag = causal ? 1 : 0;

    if (dtype == ATTN_F16 && D == 16) {
        dim3 grid(B * H, (Sq + 31) / 32);
        hipLaunchKernelGGL(attn_forward_kernel_wmma_f16, grid, dim3(128), 0, stream,
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
