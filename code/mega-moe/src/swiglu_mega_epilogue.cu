// P153-1 实验：把 Mega MoE SM100 主 kernel 的 **L1 epilogue** 算术
//   （`sm100_fp8_fp4_mega_moe.cuh:1066-1205`：bf16 round → clamp → SwiGLU → 乘 top-k 权重 →
//    per-`num_per_channels` amax → UE8M0 幂次 scale → E4M3 量化 → MN-major SF）在
//   H100(sm90) 上**用真实 DeepGEMM 设备头文件**逐元素复刻，作为 `third-party/tilelang_ops/
//   swiglu_apply_weight_to_fp8.py` 基线（`tests/test_mega_moe.py` 的 `run_baseline` 用它）
//   的**设备端算术对照**。
//
// 背景：Mega MoE 端到端在 SM100 上跑不了，但它的 L1 epilogue 数学与量化语义架构无关；
//   官方验收测试要求 `torch.equal(fused_y, baseline_y)`（逐位），而 baseline 的 SwiGLU 段就是
//   这个 tilelang kernel。本实验把 mega epilogue 的**算术**（`__expf` + `fast_rcp` 的 silu、
//   `get_ue8m0_sf_exp/inv` 的进位取整幂次、`__nv_fp8_e4m3` 的 RN 量化）原样搬到 sm90 上跑，
//   与真实 tilelang kernel 的输出逐位对拍，回答「两者是否 bit-compatible」。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/common/math.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

static void dump(const char* path, const void* ptr, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("cannot open %s\n", path); std::exit(1); }
    fwrite(ptr, 1, bytes, f);
    fclose(f);
}

static std::vector<uint8_t> load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); std::exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t)n) { printf("short read %s\n", path); std::exit(1); }
    fclose(f);
    return buf;
}

// 每个 block 处理一个 (token, group-of-G) 通道组；blockDim.x == G（32 或 128）。
// 每个线程一个通道：读 gate/up(bf16) → clamp → silu → 乘权重 → amax 归约(G) →
// get_ue8m0_sf_exp → get_ue8m0_sf_inv → E4M3 量化 → 写 fp8 与 SF 指数。
__global__ void k_swiglu_fp8(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             const float* w, uint8_t* y, uint8_t* sf_out,
                             int T, int H, int G, float clamp_val, int fast, int use_weight) {
    const int num_groups = H / G;
    const int blk = blockIdx.x;            // in [0, T * num_groups)
    const int token = blk / num_groups;
    const int grp = blk % num_groups;
    const int t = threadIdx.x;             // in [0, G)
    const int col = grp * G + t;

    float g = __bfloat162float(gate[token * H + col]);
    float u = __bfloat162float(up[token * H + col]);
    // mega: gate/up 先 bf16 round（此处输入已是 bf16，round 为恒等），再
    //   gate = min(gate, c); up = min(c, max(-c, up))；本实验按 tilelang 用 fp32 做同样 clamp。
    if (clamp_val < 3.0e38f) {
        g = fminf(g, clamp_val);
        u = fminf(fmaxf(u, -clamp_val), clamp_val);
    }
    float silu;
    if (fast) {
        // mega `:1124-1132`：kFastMath 用 __expf + fast_rcp
        const float e = __expf(-g);
        silu = g * math::fast_rcp(1.0f + e);
    } else {
        silu = g / (1.0f + expf(-g));
    }
    float val = silu * u;
    if (use_weight) val *= w[token];

    __shared__ float red[128];
    red[t] = fabsf(val);
    __syncthreads();
    for (int s = G / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float amax = red[0];
    // mega `:1173-1176`：get_ue8m0_sf_exp / get_ue8m0_sf_inv（整数进位幂次）
    const uint32_t sf_exp = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(amax);
    const float sf_inv = math::get_ue8m0_sf_inv<float>(sf_exp);
    // mega `:1183`：__nv_fp8x4_e4m3（RN）；此处单元素用 __nv_fp8_e4m3
    const __nv_fp8_e4m3 q(static_cast<float>(val) * sf_inv);
    y[token * H + col] = reinterpret_cast<const uint8_t&>(q);
    if (t == 0) sf_out[token * num_groups + grp] = static_cast<uint8_t>(sf_exp);
}

int main(int argc, char** argv) {
    // argv: T H G clamp fast use_weight
    if (argc < 7) { printf("usage: %s T H G clamp fast use_weight\n", argv[0]); return 1; }
    const int T = atoi(argv[1]);
    const int H = atoi(argv[2]);
    const int G = atoi(argv[3]);
    const float clamp_val = strtof(argv[4], nullptr);
    const int fast = atoi(argv[5]);
    const int use_weight = atoi(argv[6]);

    auto gb = load("/tmp/swiglu_gate.bin");
    auto ub = load("/tmp/swiglu_up.bin");
    auto wb = load("/tmp/swiglu_w.bin");

    __nv_bfloat16 *gate, *up;
    float* w;
    CK(cudaMalloc(&gate, gb.size()));
    CK(cudaMalloc(&up, ub.size()));
    CK(cudaMalloc(&w, wb.size()));
    CK(cudaMemcpy(gate, gb.data(), gb.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(up, ub.data(), ub.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w, wb.data(), wb.size(), cudaMemcpyHostToDevice));

    const int num_groups = H / G;
    std::vector<uint8_t> y(T * H, 0), sf(T * num_groups, 0);
    uint8_t* dy; uint8_t* dsf;
    CK(cudaMalloc(&dy, y.size()));
    CK(cudaMalloc(&dsf, sf.size()));

    k_swiglu_fp8<<<T * num_groups, G>>>(gate, up, w, dy, dsf, T, H, G, clamp_val, fast, use_weight);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y.data(), dy, y.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sf.data(), dsf, sf.size(), cudaMemcpyDeviceToHost));

    dump("/tmp/swiglu_mega_y.bin", y.data(), y.size());
    dump("/tmp/swiglu_mega_sf.bin", sf.data(), sf.size());

    // 打印几个样本便于人工核对
    printf("mega-epilogue: T=%d H=%d G=%d clamp=%g fast=%d weight=%d  y[0..5]=",
           T, H, G, clamp_val, fast, use_weight);
    for (int i = 0; i < 6 && i < T * H; ++i) printf("%02x ", y[i]);
    printf(" sf[0..5]=");
    for (int i = 0; i < 6 && i < (int)sf.size(); ++i) printf("%u ", sf[i]);
    printf("\n");
    return 0;
}
