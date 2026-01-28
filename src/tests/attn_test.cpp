#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attn_kernel.h"

static void cpu_attn_forward(const float* q, const float* k, const float* v, float* out,
                             int B, int H, int S, int D) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                // scores
                std::vector<float> scores(S, 0.0f);
                float max_score = -INFINITY;
                for (int ks = 0; ks < S; ++ks) {
                    float score = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        int q_idx = (((b * H + h) * S + s) * D + d);
                        int k_idx = (((b * H + h) * S + ks) * D + d);
                        score += q[q_idx] * k[k_idx];
                    }
                    score *= scale;
                    scores[ks] = score;
                    if (score > max_score) max_score = score;
                }

                float denom = 0.0f;
                for (int ks = 0; ks < S; ++ks) {
                    denom += std::exp(scores[ks] - max_score);
                }

                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int ks = 0; ks < S; ++ks) {
                        float w = std::exp(scores[ks] - max_score) / denom;
                        int v_idx = (((b * H + h) * S + ks) * D + d);
                        acc += w * v[v_idx];
                    }
                    int o_idx = (((b * H + h) * S + s) * D + d);
                    out[o_idx] = acc;
                }
            }
        }
    }
}

int main() {
    const int B = 1, H = 1, S = 2, D = 4;
    const int total = B * H * S * D;

    std::vector<float> h_q(total), h_k(total), h_v(total), h_out(total), h_ref(total);
    for (int i = 0; i < total; ++i) {
        h_q[i] = 0.01f * (i + 1);
        h_k[i] = 0.02f * (i + 1);
        h_v[i] = 0.03f * (i + 1);
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    hipMalloc(&d_q, total * sizeof(float));
    hipMalloc(&d_k, total * sizeof(float));
    hipMalloc(&d_v, total * sizeof(float));
    hipMalloc(&d_out, total * sizeof(float));

    hipMemcpy(d_q, h_q.data(), total * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_k, h_k.data(), total * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_v, h_v.data(), total * sizeof(float), hipMemcpyHostToDevice);

    launch_attn_forward(d_q, d_k, d_v, d_out, B, H, S, D, nullptr);
    hipDeviceSynchronize();

    hipMemcpy(h_out.data(), d_out, total * sizeof(float), hipMemcpyDeviceToHost);

    cpu_attn_forward(h_q.data(), h_k.data(), h_v.data(), h_ref.data(), B, H, S, D);

    float max_diff = 0.0f;
    for (int i = 0; i < total; ++i) {
        float diff = std::fabs(h_out[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_out);

    const float tol = 1e-3f;
    if (max_diff > tol) {
        std::printf("attn_test FAILED: max_diff=%f\n", max_diff);
        return 1;
    }

    std::printf("attn_test PASSED: max_diff=%f\n", max_diff);
    return 0;
}
