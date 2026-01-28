#include <hip/hip_runtime.h>
#include <math.h>

// Naive scaled dot-product attention forward (reference implementation)
// Layout: Q, K, V, O are [B, H, S, D] contiguous in row-major order.

__device__ __forceinline__ int idx4(int b, int h, int s, int d, int H, int S, int D) {
    return (((b * H + h) * S + s) * D + d);
}

__global__ void attn_forward_kernel(const float* q, const float* k, const float* v, float* out,
                                    int B, int H, int S, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S;
    if (tid >= total) return;

    int t = tid;
    int s = t % S;
    t /= S;
    int h = t % H;
    int b = t / H;

    float scale = rsqrtf((float)D);

    // Compute max score for numerical stability
    float max_score = -INFINITY;
    for (int ks = 0; ks < S; ++ks) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            int q_idx = idx4(b, h, s, d, H, S, D);
            int k_idx = idx4(b, h, ks, d, H, S, D);
            score += q[q_idx] * k[k_idx];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Compute softmax denominator
    float denom = 0.0f;
    for (int ks = 0; ks < S; ++ks) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            int q_idx = idx4(b, h, s, d, H, S, D);
            int k_idx = idx4(b, h, ks, d, H, S, D);
            score += q[q_idx] * k[k_idx];
        }
        score = score * scale - max_score;
        denom += expf(score);
    }

    // Compute output
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int ks = 0; ks < S; ++ks) {
            float score = 0.0f;
            for (int kd = 0; kd < D; ++kd) {
                int q_idx = idx4(b, h, s, kd, H, S, D);
                int k_idx = idx4(b, h, ks, kd, H, S, D);
                score += q[q_idx] * k[k_idx];
            }
            score = score * scale - max_score;
            float w = expf(score) / denom;
            int v_idx = idx4(b, h, ks, d, H, S, D);
            acc += w * v[v_idx];
        }
        int o_idx = idx4(b, h, s, d, H, S, D);
        out[o_idx] = acc;
    }
}

extern "C" void launch_attn_forward(const float* q, const float* k, const float* v, float* out,
                                     int B, int H, int S, int D, hipStream_t stream) {
    int total = B * H * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(attn_forward_kernel, dim3(blocks), dim3(threads), 0, stream,
                       q, k, v, out, B, H, S, D);
}
