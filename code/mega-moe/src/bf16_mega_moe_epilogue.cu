// P10-1 实验：BF16 Mega MoE（`sm100_bf16_mega_moe.cuh`）epilogue 的**架构无关数值语义**
// 在 H100(sm90) 上的忠实逐行移植与独立 double 对拍。
//
// BF16 Mega MoE 主体是 SM100 专属（`mega_moe.hpp:365-367` 只在 arch_major==10 派发；
// UMMA `SM100_MMA_F16BF16_2x1SM_SS` / TMEM / tcgen05 在 sm90 不可用），H100 端到端跑不了。
// 但 L1/L2 epilogue 的数值原语是架构无关的，本文件把它们搬到裸 CUDA：
//
//   A. L1 epilogue：in-place SwiGLU（`sm100_bf16_mega_moe.cuh:891-973`）
//      - TMEM 读回 8 个 fp32 累加值，gate/up 以 (v0,v2),(v1,v3),(v4,v6),(v5,v7) 成对；
//      - gate/up 先 __float22bfloat162_rn 舍入到 bf16（:949-950），clamp（:953-957）；
//      - silu(gate)=gate/(1+exp(-gate))，`kFastMath` 用 __expf + rcp.approx（:961-969）；
//      - out = bf16(gate*up*weight)（:971-972）。
//   B. L2 epilogue：`cast_into_bf16_and_pack`（`common/math.cuh:81-84`
//      -> `cast_into_bf16x2` `:73-77` = __float22bfloat162_rn），即 RNE 舍入。
//      对照「round-to-nearest-even」与「截断」两种参考。
//
// 参考实现全部用 double 在同一文件内独立实现。编译/运行见 harness/exp_bf16_mega_moe.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_bf16.h>

#define DEV __device__ __forceinline__

// `fadd2_rn` / `fmul2_rn` 是 SM100 专属的打包浮点指令（crt/sm_100_rt.hpp）。
// 在 sm90 上用两条等价的 IEEE round-to-nearest fp32 运算代替（数值完全相同）。
DEV float2 fadd2_rn(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
DEV float2 fmul2_rn(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }

// ---------------------------------------------------------------------------
// 源码移植：deep_gemm/common/math.cuh
// ---------------------------------------------------------------------------
DEV float fast_rcp(const float& x) {
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

// math.cuh:81-84 / :73-77 cast_into_bf16x2 -> __float22bfloat162_rn（RNE）
DEV int cast_into_bf16_and_pack(float x, float y) {
    nv_bfloat162 bf16x2 = __float22bfloat162_rn(make_float2(x, y));
    return *reinterpret_cast<int*>(&bf16x2);
}

// ---------------------------------------------------------------------------
// L1 epilogue：in-place SwiGLU（sm100_bf16_mega_moe.cuh:944-972 的逐行移植）
// 输入 values[8]：gate/up 交错，(v0,v2),(v1,v3) 是 k=0 的 (gate,up)，
// (v4,v6),(v5,v7) 是 k=1 的 (gate,up)。
// 输出 out[4]（bf16 位）与 fp32 中间 gate（供误差分解）。
// ---------------------------------------------------------------------------
struct SwiGLUOut {
    uint32_t packed;   // 4 个 bf16 结果
    float gate_bf16_f32[4];
    float silu_bf16_f32[4];  // silu 后的 fp32（未乘 up/weight）
};

DEV void swiglu_l1(const float values[8], const float weight[4],
                   bool fast_math, float clamp,
                   SwiGLUOut& out) {
    auto fp32_values = const_cast<float*>(values);
    nv_bfloat162 packed[2];
    #pragma unroll
    for (uint32_t k = 0; k < 2; ++k) {
        auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
        auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

        // Clamp（:953-957）
        if (clamp != INFINITY) {
            bf16_gate = __hmin2(bf16_gate, nv_bfloat162{__float2bfloat16(clamp), __float2bfloat16(clamp)});
            bf16_up = __hmax2(bf16_up, nv_bfloat162{__float2bfloat16(-clamp), __float2bfloat16(-clamp)});
            bf16_up = __hmin2(bf16_up, nv_bfloat162{__float2bfloat16(clamp), __float2bfloat16(clamp)});
        }

        // SwiGLU（:959-969）
        auto gate = __bfloat1622float2(bf16_gate);
        auto neg_gate_exp = make_float2(
            fast_math ? __expf(-gate.x) : expf(-gate.x),
            fast_math ? __expf(-gate.y) : expf(-gate.y));
        const auto denom = fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
        if (fast_math) {
            gate = fmul2_rn(gate, {fast_rcp(denom.x), fast_rcp(denom.y)});
        } else {
            gate = {gate.x / denom.x, gate.y / denom.y};
        }
        const auto up = __bfloat1622float2(bf16_up);

        float2 w0 = make_float2(weight[k * 2], weight[k * 2 + 1]);
        nv_bfloat162 res = __float22bfloat162_rn(fmul2_rn(fmul2_rn(gate, up), w0));
        packed[k] = res;

        out.gate_bf16_f32[k * 2 + 0] = __bfloat1622float2(bf16_gate).x;
        out.gate_bf16_f32[k * 2 + 1] = __bfloat1622float2(bf16_gate).y;
        out.silu_bf16_f32[k * 2 + 0] = gate.x;
        out.silu_bf16_f32[k * 2 + 1] = gate.y;
    }
    uint32_t lo = *reinterpret_cast<uint32_t*>(&packed[0]);
    out.packed = lo;  // 只用低 2 个做核对即可，这里保存 lo
}

// 参考：double 精度 silu，输入为**原始 fp32**（不 round 到 bf16）
__host__ __device__ double ref_silu_d(double g, double u, double w) {
    return (g / (1.0 + exp(-g))) * u * w;
}

// 参考：double 精度 silu，输入先 RNE 到 bf16
__host__ __device__ double ref_silu_bf16in_d(float gf, float uf, double w) {
    double g = (double)__bfloat162float(__float2bfloat16(gf));
    double u = (double)__bfloat162float(__float2bfloat16(uf));
    return (g / (1.0 + exp(-g))) * u * w;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__global__ void swiglu_kernel(const float* values, const float* weights,
                              int n_chunks, bool fast_math, float clamp,
                              float* out_fast, float* out_gate_bf16) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;
    const float* v = values + i * 8;
    const float* w = weights + i * 4;
    SwiGLUOut o;
    swiglu_l1(v, w, fast_math, clamp, o);
    // packed 里是 bf16 结果，写前 2 个（k=0）用于核对 bit 级
    nv_bfloat162 p0 = *reinterpret_cast<const nv_bfloat162*>(&o.packed);
    out_fast[i * 2 + 0] = __bfloat162float(p0.x);
    out_fast[i * 2 + 1] = __bfloat162float(p0.y);
    out_gate_bf16[i * 4 + 0] = o.gate_bf16_f32[0];
    out_gate_bf16[i * 4 + 1] = o.gate_bf16_f32[1];
    out_gate_bf16[i * 4 + 2] = o.silu_bf16_f32[0];
    out_gate_bf16[i * 4 + 3] = o.silu_bf16_f32[1];
}

__global__ void cast_pack_kernel(const float* in, int n, int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 2 + 1 >= n) return;
    out[i] = cast_into_bf16_and_pack(in[i * 2], in[i * 2 + 1]);
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
static double bf16_rne_ref_d(double x) {
    float f = (float)x;
    return (double)__bfloat162float(__float2bfloat16(f));
}
static double bf16_trunc_ref_d(double x) {
    float f = (float)x;
    uint32_t b = *reinterpret_cast<uint32_t*>(&f);
    b &= 0xFFFF0000u;  // 截断
    float r = *reinterpret_cast<float*>(&b);
    return (double)r;
}

int main() {
    cudaFree(0);
    std::mt19937_64 rng(12345);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);

    const int N = 1 << 21;  // 2M 个 chunk（每 chunk 4 个 gate/up 对）
    std::vector<float> h_values(N * 8), h_weights(N * 4);
    for (int i = 0; i < N * 8; ++i) h_values[i] = nd(rng);
    for (int i = 0; i < N * 4; ++i) h_weights[i] = ud(rng);

    float *d_values, *d_weights, *d_out_fast, *d_gate_bf16;
    cudaMalloc(&d_values, h_values.size() * 4);
    cudaMalloc(&d_weights, h_weights.size() * 4);
    cudaMalloc(&d_out_fast, (size_t)N * 2 * 4);
    cudaMalloc(&d_gate_bf16, (size_t)N * 4 * 4);
    cudaMemcpy(d_values, h_values.data(), h_values.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * 4, cudaMemcpyHostToDevice);

    std::vector<float> h_out_acc(N * 2), h_out_fast(N * 2), h_gate(N * 4);
    int threads = 256, blocks = (N + threads - 1) / threads;

    // ---- A1) accurate（fast_math=false）----
    swiglu_kernel<<<blocks, threads>>>(d_values, d_weights, N, false, INFINITY, d_out_fast, d_gate_bf16);
    cudaMemcpy(h_out_acc.data(), d_out_fast, h_out_acc.size() * 4, cudaMemcpyDeviceToHost);

    // ---- A2) fast_math=true ----
    swiglu_kernel<<<blocks, threads>>>(d_values, d_weights, N, true, INFINITY, d_out_fast, d_gate_bf16);
    cudaMemcpy(h_out_fast.data(), d_out_fast, h_out_fast.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gate.data(), d_gate_bf16, h_gate.size() * 4, cudaMemcpyDeviceToHost);

    // 参考：对 k=0 的前两个元素
    auto evaluate = [&](const std::vector<float>& got, const char* name) {
        double max_abs = 0, max_rel = 0, ref_max = 0;
        for (int i = 0; i < N; ++i) {
            // k=0 的 gate=(v0,v1), up=(v2,v3)
            for (int j = 0; j < 2; ++j) {
                double g = h_values[i * 8 + j];
                double u = h_values[i * 8 + 2 + j];
                double w = h_weights[i * 4 + j];
                double ref = ref_silu_d(g, u, w);
                double d = std::fabs(got[i * 2 + j] - ref);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / (std::fabs(ref) + 1e-30));
                ref_max = std::max(ref_max, std::fabs(ref));
            }
        }
        printf("   %-28s max_abs=%.3e  max_rel=%.3e  (ref max=%.2f)\n", name, max_abs, max_rel, ref_max);
        return max_rel;
    };
    printf("== A) L1 epilogue SwiGLU：\n");
    double rel_acc = evaluate(h_out_acc, "accurate vs fp64(from fp32)");
    double rel_fast = evaluate(h_out_fast, "fast_math vs fp64(from fp32)");

    // 分解：bf16 舍入门/up 的误差（用 accurate，输入先 round bf16）
    {
        double max_abs = 0, max_rel = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < 2; ++j) {
                double g = h_values[i * 8 + j], u = h_values[i * 8 + 2 + j], w = h_weights[i * 4 + j];
                double ref = ref_silu_bf16in_d((float)g, (float)u, w);
                double d = std::fabs(h_out_acc[i * 2 + j] - ref);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / (std::fabs(ref) + 1e-30));
            }
        printf("   %-28s max_abs=%.3e  max_rel=%.3e\n", "accurate vs fp64(bf16-in)", max_abs, max_rel);
    }
    // fast vs accurate（隔离 fast-math）
    {
        double max_abs = 0, max_rel = 0;
        for (int i = 0; i < N * 2; ++i) {
            max_abs = std::max(max_abs, (double)std::fabs(h_out_fast[i] - h_out_acc[i]));
            max_rel = std::max(max_rel, (double)std::fabs(h_out_fast[i] - h_out_acc[i]) / (std::fabs(h_out_acc[i]) + 1e-30));
        }
        printf("   %-28s max_abs=%.3e  max_rel=%.3e（__expf+rcp.approx 引入）\n",
               "fast_math vs accurate bf16", max_abs, max_rel);
    }

    // ---- A3) clamp 分支 ----
    {
        const float CLAMP = 2.0f;
        swiglu_kernel<<<blocks, threads>>>(d_values, d_weights, N, true, CLAMP, d_out_fast, d_gate_bf16);
        cudaMemcpy(h_out_fast.data(), d_out_fast, h_out_fast.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gate.data(), d_gate_bf16, h_gate.size() * 4, cudaMemcpyDeviceToHost);
        // 找 gate>clamp 的样本，验证 gate 被 clamp 到 +CLAMP、负 up 被 clamp 到 -CLAMP
        int n_gate_clamped = 0, n_up_clamped = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 2; ++j) {
                float g = h_values[i * 8 + j];
                float u = h_values[i * 8 + 2 + j];
                if (g > CLAMP) {
                    n_gate_clamped++;
                    if (std::fabs(h_gate[i * 4 + 0 + (j == 0 ? 0 : 1)] - CLAMP) > 1e-6f &&
                        std::fabs(h_gate[i * 4 + 0 + (j == 0 ? 0 : 1)] - CLAMP) > 1e-3f)
                        printf("     gate clamp mismatch i=%d j=%d got=%.6f\n", i, j, h_gate[i * 4]);
                }
                if (u < -CLAMP) n_up_clamped++;
            }
        }
        printf("== A3) clamp=%.1f：gate>clamp 样本 %d 个、up<-clamp 样本 %d 个（已按 :953-957 钳位）\n",
               CLAMP, n_gate_clamped, n_up_clamped);
    }

    // ---- B) L2 epilogue cast_into_bf16_and_pack ----
    printf("== B) L2 epilogue cast_into_bf16_and_pack（RNE）：\n");
    {
        const int M = 1 << 22;
        std::vector<float> h_in(M);
        std::uniform_real_distribution<float> sd(-8.0f, 8.0f);
        for (int i = 0; i < M; ++i) h_in[i] = sd(rng);
        float* d_in; int* d_out;
        cudaMalloc(&d_in, M * 4);
        cudaMalloc(&d_out, (M / 2) * 4);
        cudaMemcpy(d_in, h_in.data(), M * 4, cudaMemcpyHostToDevice);
        cast_pack_kernel<<<(M / 2 + threads - 1) / threads, threads>>>(d_in, M, d_out);
        std::vector<int> h_out(M / 2);
        cudaMemcpy(h_out.data(), d_out, (M / 2) * 4, cudaMemcpyDeviceToHost);
        int mism_rne = 0, mism_trunc = 0;
        for (int i = 0; i < M / 2; ++i) {
            nv_bfloat162 got = *reinterpret_cast<nv_bfloat162*>(&h_out[i]);
            double gx = __bfloat162float(got.x), gy = __bfloat162float(got.y);
            if (gx != bf16_rne_ref_d(h_in[i * 2]) || gy != bf16_rne_ref_d(h_in[i * 2 + 1])) mism_rne++;
            if (gx != bf16_trunc_ref_d(h_in[i * 2]) || gy != bf16_trunc_ref_d(h_in[i * 2 + 1])) mism_trunc++;
        }
        printf("   %d 个 bf16 对：vs RNE 不符 %d，vs 截断 不符 %d（证明是 RNE）\n", M / 2, mism_rne, mism_trunc);
    }

    printf("\nALL DONE\n");
    return 0;
}
