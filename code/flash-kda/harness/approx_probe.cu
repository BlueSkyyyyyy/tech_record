// P2-2 探针：量化 K1 用到的三个近似指令的误差。
//   ex2_approx_ftz_f32      (utils.cuh:38-42)  base-2 exp
//   tanh_approx_f32         (utils.cuh:44-48)  PTX tanh.approx.f32
//   sigmoid_tanh_approx_f32 (utils.cuh:50-53)  0.5*tanh(x/2)+0.5
// 以及门控衰减乘子 2^(gate_scale * sigmoid_approx(x))（fwd_kernel1.cuh:321-328）
// 相对「精确双精度参照」的误差。
//
// 编译（在 kernel_lab 内）：
//   nvcc -O2 -arch=sm_90a approx_probe.cu -o /tmp/approx_probe && /tmp/approx_probe
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float ex2_approx(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float tanh_approx(float x) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float sig_approx(float x) {
    return tanh_approx(x * 0.5f) * 0.5f + 0.5f;
}

__global__ void probe(const float* __restrict__ xs, int n,
                      float* __restrict__ ex2,
                      float* __restrict__ th,
                      float* __restrict__ sg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = xs[i];
        ex2[i] = ex2_approx(x);
        th[i]  = tanh_approx(x);
        sg[i]  = sig_approx(x);
    }
}

static double max_rel(double a, double b) {
    double d = std::fabs(a - b);
    return d / std::fmax(std::fabs(b), 1e-30);
}

int main() {
    const int n = 2000001;
    const float lo = -30.0f, hi = 30.0f;
    float* hx = new float[n];
    float* he = new float[n];
    float* ht = new float[n];
    float* hs = new float[n];
    for (int i = 0; i < n; ++i) hx[i] = lo + (hi - lo) * i / (n - 1);

    float *dx, *de, *dt, *ds;
    cudaMalloc(&dx, n * 4); cudaMalloc(&de, n * 4);
    cudaMalloc(&dt, n * 4); cudaMalloc(&ds, n * 4);
    cudaMemcpy(dx, hx, n * 4, cudaMemcpyHostToDevice);
    probe<<<(n + 255) / 256, 256>>>(dx, n, de, dt, ds);
    cudaDeviceSynchronize();
    cudaMemcpy(he, de, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ht, dt, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hs, ds, n * 4, cudaMemcpyDeviceToHost);

    // ex2: 关注 g 的实际范围（gate 非正 → cumsum ∈ (-inf,0]），取 [-30,0]
    double e_abs = 0, e_rel = 0;
    for (int i = 0; i < n; ++i) {
        if (hx[i] > 0) continue;
        double ref = std::exp2((double)hx[i]);
        e_abs = std::fmax(e_abs, std::fabs(he[i] - ref));
        e_rel = std::fmax(e_rel, max_rel(he[i], ref));
    }
    printf("ex2.approx  x in [-30,0] : max abs=%.3e  max rel=%.3e\n", e_abs, e_rel);

    // tanh.approx
    double t_abs = 0, t_rel = 0;
    for (int i = 0; i < n; ++i) {
        double ref = std::tanh((double)hx[i]);
        t_abs = std::fmax(t_abs, std::fabs(ht[i] - ref));
        t_rel = std::fmax(t_rel, max_rel(ht[i], ref));
    }
    printf("tanh.approx x in [-30,30]: max abs=%.3e  max rel=%.3e\n", t_abs, t_rel);

    // sigmoid_tanh_approx
    double s_abs = 0, s_rel = 0;
    for (int i = 0; i < n; ++i) {
        double ref = 1.0 / (1.0 + std::exp(-(double)hx[i]));
        s_abs = std::fmax(s_abs, std::fabs(hs[i] - ref));
        s_rel = std::fmax(s_rel, max_rel(hs[i], ref));
    }
    printf("sigmoid_tanh_approx x in [-30,30]: max abs=%.3e  max rel=%.3e\n", s_abs, s_rel);
    // 常用门控 logit 范围
    s_abs = 0; s_rel = 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(hx[i]) > 10) continue;
        double ref = 1.0 / (1.0 + std::exp(-(double)hx[i]));
        s_abs = std::fmax(s_abs, std::fabs(hs[i] - ref));
        s_rel = std::fmax(s_rel, max_rel(hs[i], ref));
    }
    printf("sigmoid_tanh_approx x in [-10,10]: max abs=%.3e  max rel=%.3e\n", s_abs, s_rel);

    // 端到端：decay = 2^(gate_scale * sigmoid(x))，kernel 用 ex2_approx(gate_scale * sig_approx(x))
    printf("\ndecay = 2^(gate_scale*sigmoid(x)), x in [-10,10]\n");
    const double log2e = 1.4426950408889634;
    for (double lb : {-1.0, -2.0, -5.0}) {
        double gs = lb * log2e;  // gate_scale = lower_bound * log2(e)
        double d_abs = 0, d_rel = 0;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(hx[i]) > 10) continue;
            double sref = 1.0 / (1.0 + std::exp(-(double)hx[i]));
            double kdec = std::exp2(gs * (double)hs[i]);          // kernel 路径
            double rdec = std::exp2(gs * sref);                   // 精确 sigmoid
            d_abs = std::fmax(d_abs, std::fabs(kdec - rdec));
            d_rel = std::fmax(d_rel, max_rel(kdec, rdec));
        }
        printf("  lower_bound=%.1f: max abs=%.3e  max rel=%.3e\n", lb, d_abs, d_rel);
    }

    cudaFree(dx); cudaFree(de); cudaFree(dt); cudaFree(ds);
    delete[] hx; delete[] he; delete[] ht; delete[] hs;
    return 0;
}
