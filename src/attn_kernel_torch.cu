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
    int query_base = blockIdx.x * 2;
    int total = B * H * Sq;
    if (query_base >= total) return;

    int tid = threadIdx.x;
    int threads = blockDim.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (threads + 31) >> 5;
    float scale_log2 = rsqrtf((float)D) * 1.4426950408889634f; // log2(e)

    constexpr int TILE_K = 32;
    extern __shared__ float smem[];
    float* red = smem;                        // threads
    float* qbuf = red + threads;              // 2 * D
    float* kbuf = qbuf + 2 * D;               // TILE_K * D
    float* vbuf = kbuf + TILE_K * D;          // TILE_K * D
    float* scores = vbuf + TILE_K * D;        // 2 * TILE_K
    float* meta = scores + 2 * TILE_K;        // 6 floats: max0,sum0,scale0,max1,sum1,scale1

    int qidx0 = query_base;
    int qidx1 = query_base + 1;

    int t0 = qidx0;
    int s0 = t0 % Sq;
    t0 /= Sq;
    int h0 = t0 % H;
    int b0 = t0 / H;

    bool valid1 = qidx1 < total;
    int b1 = b0;
    int h1 = h0;
    int s1 = s0;
    if (valid1) {
        int t1 = qidx1;
        s1 = t1 % Sq;
        t1 /= Sq;
        h1 = t1 % H;
        b1 = t1 / H;
        valid1 = (b1 == b0) && (h1 == h0);
    }

    if (tid < D) {
        int q_idx0 = idx4(b0, h0, s0, tid, H, Sq, D);
        qbuf[tid] = AttnTraits<T>::load(q + q_idx0);
        if (valid1) {
            int q_idx1 = idx4(b1, h1, s1, tid, H, Sq, D);
            qbuf[D + tid] = AttnTraits<T>::load(q + q_idx1);
        } else {
            qbuf[D + tid] = 0.0f;
        }
    }
    if (tid == 0) {
        meta[0] = -INFINITY; // max0
        meta[1] = 0.0f;      // sum0
        meta[2] = 1.0f;      // scale0
        meta[3] = -INFINITY; // max1
        meta[4] = 0.0f;      // sum1
        meta[5] = 1.0f;      // scale1
    }
    __syncthreads();

    float acc0_0 = 0.0f;
    float acc0_1 = 0.0f;
    float acc1_0 = 0.0f;
    float acc1_1 = 0.0f;

    for (int ks0 = 0; ks0 < Skv; ks0 += TILE_K) {
        int tile_len = (ks0 + TILE_K <= Skv) ? TILE_K : (Skv - ks0);
        int tile_elems = tile_len * D;
        if constexpr (std::is_same<T, __half>::value) {
            int halfD = (D + 1) >> 1;
            int tile_elems2 = tile_len * halfD;
            for (int idx = tid; idx < tile_elems2; idx += threads) {
                int ks = idx / halfD;
                int d2 = idx - ks * halfD;
                int d0 = d2 * 2;
                int d1 = d0 + 1;
                int k_idx = idx4(b0, h0, ks0 + ks, d0, H, Skv, D);
                int v_idx = idx4(b0, h0, ks0 + ks, d0, H, Skv, D);
                if (d1 < D) {
                    const __half2 kv = *reinterpret_cast<const __half2*>(k + k_idx);
                    const __half2 vv = *reinterpret_cast<const __half2*>(v + v_idx);
                    const float2 kf = __half22float2(kv);
                    const float2 vf = __half22float2(vv);
                    kbuf[ks * D + d0] = kf.x;
                    kbuf[ks * D + d1] = kf.y;
                    vbuf[ks * D + d0] = vf.x;
                    vbuf[ks * D + d1] = vf.y;
                } else {
                    kbuf[ks * D + d0] = AttnTraits<T>::load(k + k_idx);
                    vbuf[ks * D + d0] = AttnTraits<T>::load(v + v_idx);
                }
            }
        } else {
            for (int idx = tid; idx < tile_elems; idx += threads) {
                int ks = idx / D;
                int d = idx - ks * D;
                int k_idx = idx4(b0, h0, ks0 + ks, d, H, Skv, D);
                int v_idx = idx4(b0, h0, ks0 + ks, d, H, Skv, D);
                kbuf[ks * D + d] = AttnTraits<T>::load(k + k_idx);
                vbuf[ks * D + d] = AttnTraits<T>::load(v + v_idx);
            }
        }
        __syncthreads();

        for (int ks = 0; ks < tile_len; ++ks) {
            int base = ks * D;
            float partial0 = 0.0f;
            float partial1 = 0.0f;
            if (tid < D) {
                partial0 = qbuf[tid] * kbuf[base + tid];
                if (valid1) {
                    partial1 = qbuf[D + tid] * kbuf[base + tid];
                }
            }
            float sum = partial0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down(sum, offset, 32);
            }
            if (lane == 0) {
                red[warp_id] = sum;
            }
            __syncthreads();
            if (warp_id == 0) {
                float warp_sum = (lane < num_warps) ? red[lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    warp_sum += __shfl_down(warp_sum, offset, 32);
                }
                if (lane == 0) {
                    red[0] = warp_sum;
                }
            }
            __syncthreads();
            if (tid == 0) {
                float score0 = red[0] * scale_log2;
                if (causal && (ks0 + ks) > s0) score0 = -INFINITY;
                scores[ks] = score0;
            }
            __syncthreads();

            float sum1 = partial1;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum1 += __shfl_down(sum1, offset, 32);
            }
            if (lane == 0) {
                red[warp_id] = sum1;
            }
            __syncthreads();
            if (warp_id == 0) {
                float warp_sum = (lane < num_warps) ? red[lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    warp_sum += __shfl_down(warp_sum, offset, 32);
                }
                if (lane == 0) {
                    red[0] = warp_sum;
                }
            }
            __syncthreads();
            if (tid == 0) {
                float score1 = red[0] * scale_log2;
                if (!valid1 || (causal && (ks0 + ks) > s1)) score1 = -INFINITY;
                scores[TILE_K + ks] = score1;
            }
            __syncthreads();
        }

        if (tid == 0) {
            float tile_max0 = -INFINITY;
            float tile_max1 = -INFINITY;
            for (int ks = 0; ks < tile_len; ++ks) {
                float sc0 = scores[ks];
                float sc1 = scores[TILE_K + ks];
                if (sc0 > tile_max0) tile_max0 = sc0;
                if (sc1 > tile_max1) tile_max1 = sc1;
            }
            float prev_max0 = meta[0];
            float prev_max1 = meta[3];
            float new_max0 = fmaxf(prev_max0, tile_max0);
            float new_max1 = fmaxf(prev_max1, tile_max1);
            float scale0 = (prev_max0 == -INFINITY) ? 0.0f : exp2f(prev_max0 - new_max0);
            float scale1 = (prev_max1 == -INFINITY) ? 0.0f : exp2f(prev_max1 - new_max1);
            float tile_sum0 = 0.0f;
            float tile_sum1 = 0.0f;
            for (int ks = 0; ks < tile_len; ++ks) {
                float sc0 = scores[ks];
                float sc1 = scores[TILE_K + ks];
                if (sc0 != -INFINITY) tile_sum0 += exp2f(sc0 - new_max0);
                if (sc1 != -INFINITY) tile_sum1 += exp2f(sc1 - new_max1);
            }
            meta[0] = new_max0;
            meta[1] = meta[1] * scale0 + tile_sum0;
            meta[2] = scale0;
            meta[3] = new_max1;
            meta[4] = meta[4] * scale1 + tile_sum1;
            meta[5] = scale1;
        }
        __syncthreads();

        if (tid < (D + 1) / 2) {
            int d0 = tid * 2;
            int d1 = d0 + 1;
            float scale0 = meta[2];
            float scale1 = meta[5];
            float new_max0 = meta[0];
            float new_max1 = meta[3];
            acc0_0 *= scale0;
            acc1_0 *= scale0;
            if (d1 < D) {
                acc0_1 *= scale0;
                acc1_1 *= scale0;
            }
            for (int ks = 0; ks < tile_len; ++ks) {
                int base = ks * D;
                float sc0 = scores[ks];
                float sc1 = scores[TILE_K + ks];
                if (sc0 != -INFINITY) {
                    float w0 = exp2f(sc0 - new_max0);
                    acc0_0 += w0 * vbuf[base + d0];
                    if (d1 < D) acc0_1 += w0 * vbuf[base + d1];
                }
                if (sc1 != -INFINITY) {
                    float w1 = exp2f(sc1 - new_max1);
                    acc1_0 += w1 * vbuf[base + d0];
                    if (d1 < D) acc1_1 += w1 * vbuf[base + d1];
                }
            }
        }
        __syncthreads();
    }

    float denom0 = meta[1];
    float denom1 = meta[4];
    float inv_denom0 = (denom0 > 0.0f) ? (1.0f / denom0) : 0.0f;
    float inv_denom1 = (denom1 > 0.0f) ? (1.0f / denom1) : 0.0f;
    if (tid < (D + 1) / 2) {
        int d0 = tid * 2;
        int d1 = d0 + 1;
        int o_idx0 = idx4(b0, h0, s0, d0, H, Sq, D);
        out[o_idx0] = AttnTraits<T>::store(acc0_0 * inv_denom0);
        if (d1 < D) {
            int o_idx1 = idx4(b0, h0, s0, d1, H, Sq, D);
            out[o_idx1] = AttnTraits<T>::store(acc0_1 * inv_denom0);
        }
        if (valid1) {
            int o_idx2 = idx4(b1, h1, s1, d0, H, Sq, D);
            out[o_idx2] = AttnTraits<T>::store(acc1_0 * inv_denom1);
            if (d1 < D) {
                int o_idx3 = idx4(b1, h1, s1, d1, H, Sq, D);
                out[o_idx3] = AttnTraits<T>::store(acc1_1 * inv_denom1);
            }
        }
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
    size_t smem_bytes = (size_t)(threads + D * 2 + tile_k * D * 2 + tile_k * 2 + 6) * sizeof(float);
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
                hipLaunchKernelGGL(attn_forward_kernel_block<__half>, dim3((total + 1) / 2), dim3(threads), smem_bytes, stream,
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
                hipLaunchKernelGGL(attn_forward_kernel_block<hip_bfloat16>, dim3((total + 1) / 2), dim3(threads), smem_bytes, stream,
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
                hipLaunchKernelGGL(attn_forward_kernel_block<float>, dim3((total + 1) / 2), dim3(threads), smem_bytes, stream,
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
