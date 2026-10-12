// P21-1 实验：DeepGEMM **GEMM epilogue 算子**（`deep_gemm/epilogue/operators.cuh`）的 H100(sm90) 数值验证。
//
// 背景：所有 SM100 GEMM 的尾部都挂一个 `EpilogueClass`（csrc/jit_kernels/impls/epilogue_class.hpp），
// 它选出 device 侧 `epilogue::operators` 里的**一个**算子对象，随 kernel 一起 launch
// （`sm100_fp8_fp4_gemm_1d1d.cuh:41` 的 `__grid_constant__ epilogue_operator_t`）。算子在
// `sm100_store_cd.cuh:124` / `sm100_store_cd_swap_ab.cuh:112/100` 里对从 TMEM 读回寄存器的一批
// 累加器值 `apply_values()`。四种算子：
//   Identity            —— 无操作
//   ScaleByAlpha        —— 只缩放乘积项（BLAS 线性组合的 alpha）
//   StochasticRoundToBF16 —— 随机舍入到 BF16（quartet 哈希 PRNG + `cvt.rs.bf16x2.f32`）
//   QuantizeToFP8       —— BF16 舍入 -> per-32 列 amax -> UE8M0 SF -> E4M3，SF 按 MN-major 打包
//
// 本机 H100（sm90）**跑不了** SM100 GEMM（`gemm.hpp:119` 对自定义 epilogue 直接断言 "Custom epilogues
// require SM100"），且两个算子用的硬件原语是 SM100 专属：
//   * `StochasticRoundToBF16::cast_quartet` 调 `ptx::cvt_rs_bf16x2_f32`（`cvt.rs.bf16x2.f32`），
//     ptxas 在 sm_90 上直接报 "Feature '.rs' not supported on .target 'sm_90'"；
//   * `ScaleByAlpha::apply_values` 用 `__fmul2_rn`（CUDA 13.2 里只声明在 `crt/sm_100_rt.h`）。
// 但算子的**核心算法**（哈希 PRNG、SF/量化数学、SF 寻址）都是架构无关的整数/浮点运算。因此本实验：
//   1) 由 harness 把 `operators.cuh:12-92`（四个 struct）**逐行原样抽取**编译；两个 SM100 硬件原语
//      用**明确标注的等价模拟**补上（`__fmul2_rn` = 逐分量 `__fmul_rn`（sm90 device pass），`cvt.rs`
//      = 无偏随机舍入软件等价）；
//   2) 直接运行真实算子方法，验证：ScaleByAlpha 缩放正确；四元组哈希的均匀性/去相关；随机舍入的
//      无偏性 vs RNE；QuantizeToFP8 的 D/SFD 与 torch 参考逐位一致。
//
// 编译/运行见 harness/exp_epilogue_operators.sh（nvcc -arch=sm_90）。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// host 侧的位转换辅助（设备内建 __float_as_uint 等只在 device 可用）
static inline uint32_t f2u(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static inline float    u2f(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
static inline uint16_t bf16_low_host(float x) { return static_cast<uint16_t>(f2u(x) >> 16); }
static inline float    bf16_bits_to_float(uint16_t b) { return u2f(static_cast<uint32_t>(b) << 16); }

// ---------------------------------------------------------------------------
// SM100 专属硬件原语的**等价模拟**（H100 上不可用，见文件头注释）
// ---------------------------------------------------------------------------
// 注意：`__fmul2_rn`（packed fp32x2 乘，sm_100 指令 `mul.f32x2`）由 cutlass 间接引入的
// `crt/sm_100_rt.h` 声明，但声明被 `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000`
// 包住：**host pass 可见、sm90 device pass 不可见**。因此这里仅在 sm90 device pass 补一个
// 逐分量 `__fmul_rn` 的等价实现（与头文件文档的 “per component same as __fmul_rn” 一致）。
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
__device__ __forceinline__ float2 __fmul2_rn(float2 a, float2 b) {
    return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
}
#endif

// `cvt.rs.bf16x2.f32 d, upper, lower, rand`：带 16 位随机数的无偏随机舍入到 bf16x2。
// 语义等价模拟：把随机数加到将被丢弃的低 16 位尾数上再截断（对正负号都按幅值工作），
// 这是对「以概率=小数部分/ulp 向上舍入」这一无偏定义的软件实现。
namespace deep_gemm::ptx {
CUTLASS_DEVICE uint32_t cvt_rs_bf16x2_f32(const float& lower, const float& upper,
                                          const uint32_t& random_bits) {
    auto sr_one = [](const float& x, const uint32_t& r) -> uint16_t {
        uint32_t b = __float_as_uint(x);
        b += static_cast<uint16_t>(r);      // 加 [0, 2^16) 的随机扰动
        b &= 0xffff0000u;                   // 截断低 16 位（bf16 只留高 16 位）
        return static_cast<uint16_t>(b >> 16);
    };
    const uint32_t lo = sr_one(lower, random_bits & 0xffffu);
    const uint32_t hi = sr_one(upper, random_bits >> 16);
    return lo | (hi << 16);
}
}  // namespace deep_gemm::ptx

// 由 harness 生成的抽取文件：operators.cuh:12-92（Identity/ScaleByAlpha/StochasticRoundToBF16/
// QuantizeToFP8 四个 struct），包在 `deep_gemm::epilogue::operators` 命名空间里。
#include <epilogue_operators.inc>

using namespace deep_gemm::epilogue::operators;

// ---------------------------------------------------------------------------
// A) ScaleByAlpha
// ---------------------------------------------------------------------------
__global__ void k_scale_alpha(const float* in, float* out, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ScaleByAlpha op;
    op.alpha = alpha;
    uint32_t v[2] = { __float_as_uint(in[i]), __float_as_uint(0.0f) };
    op.apply_values(v);
    out[i] = __uint_as_float(v[0]);
}

// ---------------------------------------------------------------------------
// B) StochasticRoundToBF16 的四元组哈希统计
// ---------------------------------------------------------------------------
__global__ void k_hash_stats(const uint32_t* values, int n,
                             uint32_t* bits01,      // offset 0 / 1 的 16 位输出
                             uint32_t* hist,        // 低 16 位直方图（64 桶，每桶 1024）
                             uint32_t* decorr) {    // offset 两两是否相等的计数（6 对）
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    using SR = StochasticRoundToBF16;
    const uint32_t h = SR::partial_hash(values[i], 0) + SR::partial_hash(values[i ^ 1u], 1) +
                       SR::partial_hash(values[i ^ 2u], 2) + SR::partial_hash(values[i ^ 3u], 3) + (uint32_t)i;
    uint32_t b[4];
    for (int o = 0; o < 4; ++ o) b[o] = SR::select_random_bits(h, (uint32_t)o) & 0xffffu;
    bits01[i] = (b[0]) | (b[1] << 16);
    atomicAdd(&hist[(b[0] >> 10) & 63u], 1u);
    int idx = 0;
    for (int a = 0; a < 4; ++ a)
        for (int c = a + 1; c < 4; ++ c, ++ idx)
            if (b[a] == b[c]) atomicAdd(&decorr[idx], 1u);
}

// ---------------------------------------------------------------------------
// C) 随机舍入的无偏性：模拟 cvt.rs（用真实 cvt_rs 模拟函数）+ 对比 RNE
// ---------------------------------------------------------------------------
__global__ void k_sr_round(const float* in, int n, const uint32_t* randbits,
                           uint16_t* sr_out, uint16_t* rne_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = in[i];
    const uint32_t packed = ptx::cvt_rs_bf16x2_f32(x, x, randbits[i]);
    sr_out[i] = static_cast<uint16_t>(packed & 0xffffu);
    rne_out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

// ---------------------------------------------------------------------------
// D) QuantizeToFP8 的 device 管线（复刻 sm100_store_cd.cuh:144-174 的 kWithOutputSF 分支）
//    输入 fp32 累加器 [M, N]（N=32 的整数倍），输出 E4M3 D [M,N] 与 MN-major 打包 SFD。
// ---------------------------------------------------------------------------
__global__ void k_quantize_fp8(const float* acc, int M, int N,
                               uint8_t* d_out, uint32_t* sfd, uint32_t sfd_stride) {
    const int row = blockIdx.x;
    if (row >= M) return;
    QuantizeToFP8 op;
    op.sfd = sfd;
    op.sfd_stride = sfd_stride;
    op.shape_m = (uint32_t)M;
    op.shape_n = (uint32_t)N;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        // 每个线程负责一个 32 列 SF 组（store_cd 里一个 warp 用 32 个 lane 各覆盖一组）
        if (col % 32 != 0) continue;
        const float* p = acc + (size_t)row * N + col;
        nv_bfloat162 bf[16];
        #pragma unroll
        for (int i = 0; i < 16; ++ i) {
            float lo = p[2 * i], hi = p[2 * i + 1];
            bf[i] = math::cast_into_bf16x2(lo, hi);
        }
        const auto amax = math::get_packed_bf16_amax(bf);
        const auto sf_exp = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(amax);
        const auto sf_inv = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const uint32_t word = math::scale_bf16x2_into_fp8x4(bf[i], bf[i + 1], sf_inv, sf_inv);
            d_out[(size_t)row * N + col + i * 2 + 0] = (uint8_t)(word & 0xffu);
            d_out[(size_t)row * N + col + i * 2 + 1] = (uint8_t)((word >> 8) & 0xffu);
            d_out[(size_t)row * N + col + i * 2 + 2] = (uint8_t)((word >> 16) & 0xffu);
            d_out[(size_t)row * N + col + i * 2 + 3] = (uint8_t)((word >> 24) & 0xffu);
        }
        op.store_sf((uint32_t)row, (uint32_t)col, 0u, (uint8_t)sf_exp);
    }
}

int main() {
    printf("== P21-1 DeepGEMM GEMM epilogue 算子（operators.cuh:12-92 原样抽取）H100 实测 ==\n\n");

    // ---- A) ScaleByAlpha ----
    {
        const int n = 1024;
        std::vector<float> h(n);
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> dist(-4.f, 4.f);
        for (auto& x : h) x = dist(rng);
        float *din, *dout;
        CK(cudaMalloc(&din, n * sizeof(float))); CK(cudaMalloc(&dout, n * sizeof(float)));
        CK(cudaMemcpy(din, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        k_scale_alpha<<<(n + 255) / 256, 256>>>(din, dout, n, 0.5f);
        CK(cudaDeviceSynchronize());
        std::vector<float> out(n);
        CK(cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
        int mism = 0; double maxrel = 0;
        for (int i = 0; i < n; ++ i) {
            float ref = h[i] * 0.5f;
            if (f2u(out[i]) != f2u(ref)) ++ mism;
            maxrel = std::max(maxrel, (double)std::fabs((double)out[i] - (double)ref) / (std::fabs((double)ref) + 1e-30));
        }
        printf("A) ScaleByAlpha(alpha=0.5)  n=%d  bitwise_mismatch=%d  max_rel=%.3g\n", n, mism, maxrel);
        cudaFree(din); cudaFree(dout);
    }

    // ---- B) 四元组哈希统计 ----
    {
        const int n = 1 << 20;
        std::vector<uint32_t> h(n);
        std::mt19937 rng(7);
        for (auto& x : h) x = f2u((float)(rng() % 100000) * 1e-3f);
        uint32_t* dv; uint32_t* dbits; uint32_t* dhist; uint32_t* ddecorr;
        CK(cudaMalloc(&dv, n * 4)); CK(cudaMalloc(&dbits, n * 4));
        CK(cudaMalloc(&dhist, 64 * 4)); CK(cudaMalloc(&ddecorr, 6 * 4));
        CK(cudaMemcpy(dv, h.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dhist, 0, 64 * 4)); CK(cudaMemset(ddecorr, 0, 6 * 4));
        k_hash_stats<<<(n + 255) / 256, 256>>>(dv, n, dbits, dhist, ddecorr);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> hist(64), decorr(6), bits(n);
        CK(cudaMemcpy(hist.data(), dhist, 64 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(decorr.data(), ddecorr, 6 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bits.data(), dbits, n * 4, cudaMemcpyDeviceToHost));

        // 低 16 位均匀性（bit 级：每个 bit 的 1 比例）+ 64 桶卡方
        std::vector<uint64_t> ones(16, 0);
        for (int i = 0; i < n; ++ i) {
            uint32_t b = bits[i] & 0xffffu;
            for (int k = 0; k < 16; ++ k) ones[k] += (b >> k) & 1u;
        }
        double max_dev = 0;
        for (int k = 0; k < 16; ++ k) max_dev = std::max(max_dev, std::fabs((double)ones[k] / n - 0.5));
        double exp = (double)n / 64, chi2 = 0;
        for (int b = 0; b < 64; ++ b) { double d = hist[b] - exp; chi2 += d * d / exp; }
        // 均值应 ~0.5（16 位无符号归一化）
        double mean = 0; for (int i = 0; i < n; ++ i) mean += (bits[i] & 0xffffu); mean /= n; mean /= 65535.0;
        const char* names[6] = {"0=1","0=2","0=3","1=2","1=3","2=3"};
        printf("B) StochasticRoundToBF16 四元组哈希  n=%d\n", n);
        printf("   每 bit 的 1 比例偏离 0.5 最大 %.4g；64 桶卡方=%.1f（df=63，期望≈63）；归一化均值=%.5f\n",
               max_dev, chi2, mean);
        printf("   offset 两两相等计数：");
        for (int i = 0; i < 6; ++ i) printf("%s=%u ", names[i], decorr[i]);
        printf("（应≈0，即各 offset 独立）\n");
    }

    // ---- C) 随机舍入无偏性 vs RNE ----
    {
        // 选「恰好落在两个 bf16 中点」与「非中点」两类值
        const int rep = 200000;
        std::vector<float> vals;
        std::vector<int> kinds;
        // bf16 ulp(1.0)=2^-7；下列三值的 fp32 低 16 位尾数分别 = 0x8000/0x2000/0x6000
        float mid = 0x1.01p0f;           // 1.0 与下一个 bf16 的精确中点（低 16 位 0x8000）
        float q1  = 0x1.004p0f;          // 1/8 处（低 16 位 0x2000）
        float q3  = 0x1.00cp0f;          // 3/4 处（低 16 位 0x6000）
        for (int k = 0; k < 3; ++ k) { float v = (k==0)?mid:(k==1)?q1:q3; for (int i=0;i<rep;++i){ vals.push_back(v); kinds.push_back(k);} }
        int n = vals.size();
        std::vector<uint32_t> rbits(n);
        std::mt19937 rng(3);
        for (auto& r : rbits) r = rng();
        float* dv; uint32_t* dr; uint16_t* dsr; uint16_t* drne;
        CK(cudaMalloc(&dv, n*4)); CK(cudaMalloc(&dr, n*4)); CK(cudaMalloc(&dsr, n*2)); CK(cudaMalloc(&drne, n*2));
        CK(cudaMemcpy(dv, vals.data(), n*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dr, rbits.data(), n*4, cudaMemcpyHostToDevice));
        k_sr_round<<<(n+255)/256,256>>>(dv, n, dr, dsr, drne);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> sr(n), rne(n);
        CK(cudaMemcpy(sr.data(), dsr, n*2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rne.data(), drne, n*2, cudaMemcpyDeviceToHost));
        const char* kn[3] = {"exact midpoint (low 0x8000)", "1/8 up (low 0x2000)", "3/8 up (low 0x6000)"};
        const float kx[3] = {mid, q1, q3};
        for (int k = 0; k < 3; ++ k) {
            double s_sr = 0, s_rne = 0; int up_sr = 0, up_rne = 0;
            for (int i = 0; i < n; ++ i) if (kinds[i] == k) {
                double fsr = (double)bf16_bits_to_float(sr[i]);
                double frne = (double)bf16_bits_to_float(rne[i]);
                s_sr += fsr; s_rne += frne;
                if (sr[i] > bf16_low_host(vals[i])) ++up_sr;
                if (rne[i] > bf16_low_host(vals[i])) ++up_rne;
            }
            double m_sr = s_sr / rep, m_rne = s_rne / rep, v = kx[k];
            printf("C) %-22s x=%.6f  SR均值=%.6f (bias=%+.2e, 向上比例=%.4f)  RNE均值=%.6f (bias=%+.2e)\n",
                   kn[k], (double)v, m_sr, m_sr - v, (double)up_sr / rep, m_rne, m_rne - v);
        }
    }

    // ---- D) QuantizeToFP8：输出 D 与 SFD（torch 参考在 harness 里对拍）----
    {
        const int M = 64, N = 128;
        const uint32_t sfd_stride = 8;
        std::vector<float> acc((size_t)M * N);
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> dist(-8.f, 8.f);
        for (auto& x : acc) x = dist(rng);
        // 注入一个极端行（大动态范围）与一个全零组
        for (int j = 0; j < N; ++ j) acc[0 * N + j] *= 100.f;
        for (int j = 0; j < 32; ++ j) acc[1 * N + j] = 0.f;
        float* dacc; uint8_t* dd; uint32_t* dsfd;
        CK(cudaMalloc(&dacc, acc.size()*4)); CK(cudaMalloc(&dd, acc.size()));
        CK(cudaMalloc(&dsfd, (size_t)sfd_stride * (M / 4 + 1) * 4));
        CK(cudaMemcpy(dacc, acc.data(), acc.size()*4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dd, 0, acc.size())); CK(cudaMemset(dsfd, 0, (size_t)sfd_stride*(M/4+1)*4));
        k_quantize_fp8<<<M, 32>>>(dacc, M, N, dd, dsfd, sfd_stride);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> hd(acc.size());
        std::vector<uint32_t> hsfd(sfd_stride * (M / 4 + 1));
        CK(cudaMemcpy(hd.data(), dd, acc.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hsfd.data(), dsfd, hsfd.size()*4, cudaMemcpyDeviceToHost));
        FILE* f = fopen("/tmp/epilogue_quant.bin", "wb");
        uint32_t hdr[4] = {(uint32_t)M,(uint32_t)N,sfd_stride,(uint32_t)acc.size()};
        fwrite(hdr, 4, 4, f); fwrite(acc.data(), 4, acc.size(), f);
        fwrite(hd.data(), 1, acc.size(), f); fwrite(hsfd.data(), 4, hsfd.size(), f); fclose(f);
        printf("D) QuantizeToFP8  M=%d N=%d  D/SFD 已写 /tmp/epilogue_quant.bin（D 前 8 字节 %02x %02x %02x %02x %02x %02x %02x %02x）\n",
               M, N, hd[0], hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7]);
        cudaFree(dacc); cudaFree(dd); cudaFree(dsfd);
    }

    printf("\n== 完成 ==\n");
    return 0;
}
