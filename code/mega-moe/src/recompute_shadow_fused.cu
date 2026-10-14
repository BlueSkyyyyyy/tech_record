// P173-1 实验：把第 182 篇（P172-1）留下的口子做成单 kernel 实验——
//   「重算（compute-bound）能不能在一个 kernel 内被 shadow（访存受限腿）吸收，还是被 shadow 的
//    compute-bound 段（dgrad）挡住」。
//
// 第 182 篇量到：重算 RECOMPUTE 356.7us（计算受限 1349 TFLOPS）、shadow = dgrad L2 193.3us
//   （计算受限 1244 TFLOPS）+ wgrad L2 535.9us（访存受限 449 TFLOPS），串行 1086us；两 kernel 两
//   stream 最好只有 1.09x。本篇把「重算 × shadow」拆成**资源**问题：shadow 里 wgrad 是访存受限、dgrad
//   是计算受限；重算也是计算受限（张量核）。故**只有 wgrad 段能吸收重算**、dgrad 段不能。
//
// 本文件在 H100(sm90a) 上用**真实 WGMMA**（`deep_gemm/mma/sm90.cuh` + `ptx/wgmma.cuh`）当计算腿
//   （proxy 重算 / dgrad），用一个**访存受限的 global RMW 流式腿** proxy wgrad，做成**单 kernel、
//   两个 warpgroup、`setmaxnreg` 分区**，量两两之间的重叠效率：
//     * COMP + MEM （计算腿 + 访存腿）→ 期望高（不同资源：张量核 vs LSU/HBM）；
//     * COMP + TENS（计算腿 + 第二个计算腿）→ 期望低（同一张量核，抢同一条发射/张量核通路）；
//   并对照第 182 篇两 kernel 两 stream 的 1.09x。
//
//   Mega MoE 融合 kernel 本体仍是 SM100 专属 → 仅分析；这里跑的是架构无关的 WGMMA/访存/寄存器原语。
//
// 编译（需 DeepGEMM 头）：
//   nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
//     -I$DG/deep_gemm/include -I$DG/deep_gemm/include/cutlass -I$DG/deep_gemm/include/cute \
//     src/recompute_shadow_fused.cu -o /tmp/recompute_shadow_fused
// 运行：CUDA_VISIBLE_DEVICES=1 /tmp/recompute_shadow_fused

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/numeric/integer_sequence.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

using namespace deep_gemm;
namespace mma90 = deep_gemm::mma::sm90;

#define CUDA_CHECK(x)                                                          \
    do { cudaError_t e = (x); if (e) { std::fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); std::exit(1);} } while (0)

template <int R> __device__ __forceinline__ void rinc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(R)); }
template <int R> __device__ __forceinline__ void rdec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(R)); }

using Bf16 = nv_bfloat16;
constexpr int kBM = 64, kBN = 128, kBK = 64;
using MMA = mma90::BF16MMASelector<kBN, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;

__device__ __forceinline__ void fill_swz(Bf16* smem, int rows, int k) {
    constexpr int EPB = 16 / sizeof(Bf16);
    for (int i = threadIdx.x & 127; i < rows * k; i += 128) {
        int row = i / k, col = i % k, chunk = col / EPB, within = col % EPB;
        smem[row * k + (chunk ^ (row & 7)) * EPB + within] = (Bf16)((row * 7 + col * 3) % 64) * (Bf16)0.01f;
    }
}

// 一个 warpgroup 做 reps 次 64x64x64 BF16 WGMMA（= 4 个 k=64x64x16 步），把累加器求和防优化。
__device__ __forceinline__ void wgmma_leg(Bf16* sa, Bf16* sb, float* out, long long reps, int wg) {
    fill_swz(sa, kBM, kBK);
    fill_swz(sb, kBN, kBK);
    __syncwarp();
    constexpr int kNumSteps = kBK / MMA::K;
    float accum[MMA::kNumAccum] = {};
    for (long long rep = 0; rep < reps; ++rep) {
        ptx::warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < kNumSteps; ++k) {
            auto a_desc = mma90::make_gmma_desc<cute::UMMA::Major::K, kBM, kBK, 128, Bf16>(sa, 0, (k * MMA::K) % kBK);
            auto b_desc = mma90::make_gmma_desc<cute::UMMA::Major::K, kBN, kBK, 128, Bf16>(sb, 0, (k * MMA::K) % kBK);
            MMA::wgmma(a_desc, b_desc, accum, k == 0 ? 0 : 1);
        }
        ptx::warpgroup_commit_batch();
        ptx::warpgroup_wait<0>();
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < MMA::kNumAccum; ++i) s += accum[i];
    if (s == 0.1234567f) out[threadIdx.x] = s;
}

// 访存受限腿：对 [n] 个 uint4 做 read-modify-write（RMW），proxy wgrad 的「读操作数 + 回读/写输出」。
__device__ __forceinline__ void mem_leg(uint4* __restrict__ buf, size_t n, long long iters, float* out) {
    uint4 v = make_uint4(1, 2, 3, (unsigned)threadIdx.x);
    const size_t lane = threadIdx.x & 127;
    const size_t stride = (size_t)gridDim.x * 128;
    const size_t base = (size_t)blockIdx.x * 128 + lane;
    for (long long it = 0; it < iters; ++it) {
        for (size_t i = base; i < n; i += stride) {
            uint4 r = buf[i];
            r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w;
            buf[i] = r;
        }
    }
    if (v.x == 0xdeadbeef) out[threadIdx.x] = (float)v.y;
}

// 只做一次遍历、只到 n_used 个 uint4：让标定能连续调 MEM 腿时长（1 次 512 MiB 遍历 ~1 ms，太粗）。
__device__ __forceinline__ void mem_leg_once(uint4* __restrict__ buf, size_t n_used, float* out) {
    mem_leg(buf, n_used, 1, out);
}

// 融合 kernel：1 compute WG (WG0) + 1 第二腿 (WG1)。256 线程。
//   kDoComp：WG0 是否做 WGMMA；kLeg：0=idle、1=MEM（访存 RMW）、2=TENS（第二个 WGMMA）。
template <bool kDoComp, int kLeg, int kRegC, int kRegL, bool kRealloc>
__global__ void __launch_bounds__(256, 1)
fused(uint4* __restrict__ buf, size_t mem_n, float* __restrict__ out,
      long long reps, long long mem_iters) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    Bf16* sa0 = reinterpret_cast<Bf16*>(smem_raw);
    Bf16* sb0 = sa0 + kBM * kBK;
    Bf16* sa1 = sb0 + kBN * kBK;
    Bf16* sb1 = sa1 + kBM * kBK;
    const int wg = threadIdx.x >> 7;
    if (wg == 0) {
        if constexpr (kRealloc) rdec<kRegC>();
        if constexpr (kDoComp) wgmma_leg(sa0, sb0, out, reps, 0);
        if constexpr (kRealloc) rinc<kRegC>();
    } else {
        if constexpr (kRealloc) rdec<kRegL>();
        if constexpr (kLeg == 1) mem_leg(buf, mem_n, mem_iters, out);
        else if constexpr (kLeg == 2) wgmma_leg(sa1, sb1, out, reps, 1);
        if constexpr (kRealloc) rinc<kRegL>();
    }
}

static constexpr size_t kBufBytes = 512ull * 1024 * 1024;   // 512 MiB streaming buffer

int main() {
    int nd; CUDA_CHECK(cudaGetDeviceCount(&nd));
    CUDA_CHECK(cudaSetDevice(0));
    int sm; CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
    std::printf("visible GPUs = %d, SM count = %d, stream buffer = %zu MiB\n", nd, sm, kBufBytes >> 20);
    const int grid = sm;
    const size_t n_u4 = kBufBytes / 16;

    uint4* buf; float* out;
    CUDA_CHECK(cudaMalloc(&buf, kBufBytes));
    CUDA_CHECK(cudaMalloc(&out, 4096 * sizeof(float)));
    CUDA_CHECK(cudaMemset(buf, 0x01, kBufBytes));
    const size_t smem = (kBM * kBK + kBN * kBK) * 2 * sizeof(Bf16);

    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    long long reps = 1;
    size_t mem_n = n_u4;   // MEM 腿一次遍历覆盖的 uint4 数（标定后 ≈ COMP 腿时长）
    auto time_variant = [&](int kind, long long r, size_t mn, int iters) -> double {
        std::vector<double> ts;
        for (int t = 0; t < iters; ++t) {
            CUDA_CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 3; ++i) {
                switch (kind) {
                    case 0: fused<true, 0, 40, 40, false><<<grid, 256, smem>>>(buf, mn, out, r, 0); break;
                    case 1: fused<false, 1, 40, 40, false><<<grid, 256, smem>>>(buf, mn, out, 0, 1); break;
                    case 2: fused<false, 2, 40, 40, false><<<grid, 256, smem>>>(buf, mn, out, r, 0); break;
                    case 3: fused<true, 1, 40, 40, false><<<grid, 256, smem>>>(buf, mn, out, r, 1); break;
                    case 4: fused<true, 2, 40, 40, false><<<grid, 256, smem>>>(buf, mn, out, r, 0); break;
                }
                cudaError_t le = cudaPeekAtLastError();
                if (le) { std::fprintf(stderr, "launch kind=%d err=%s\n", kind, cudaGetErrorString(le)); std::exit(1); }
            }
            CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ts.push_back(ms / 3.0 * 1000.0);   // us per launch
        }
        std::sort(ts.begin(), ts.end());
        return ts[ts.size() / 2];
    };

    // 标定 COMP：找 reps 使单次 ≈ 350us（对齐第 182 篇 重算 356.7us）
    for (reps = 200; ; reps *= 2) {
        double t = time_variant(0, reps, 0, 3);
        std::printf("  probe COMP reps=%lld -> %.1f us\n", reps, t);
        if (t > 320.0 || reps > (1 << 24)) break;
    }
    { long long lo = reps / 2, hi = reps; while (lo < hi) { long long mid = (lo + hi) / 2; double t = time_variant(0, mid, 0, 3);
        if (t < 350.0) lo = mid + 1; else hi = mid; } reps = lo; }
    double t_comp = time_variant(0, reps, 0, 5);

    // 标定 MEM：二分 mem_n（一次遍历覆盖的 uint4 数）使单次 ≈ t_comp（两条腿等长，读重叠效率最直观）。
    { size_t lo = 1, hi = n_u4; while (lo < hi) { size_t mid = lo + (hi - lo) / 2;
        double t = time_variant(1, 0, mid, 3); if (t < t_comp) lo = mid + 1; else hi = mid; } mem_n = lo; }

    const int it = 9;
    double tc = time_variant(0, reps, 0, it);      // COMP only
    double tm = time_variant(1, 0, mem_n, it);     // MEM only
    double tt = time_variant(2, reps, 0, it);      // TENS only
    double tcm = time_variant(3, reps, mem_n, it); // COMP + MEM
    double tct = time_variant(4, reps, 0, it);     // COMP + TENS

    auto eff = [](double a, double b, double t) { double s = a + b, ideal = std::max(a, b); return (s / t - 1) / (s / ideal - 1); };
    auto gbps = [&](double us) { return (double)mem_n * 16 * 2 / (us * 1e-6) / 1e9; };  // RMW = 读+写

    std::printf("\n# 结果（单 kernel，1 CTA/SM × %d SM，256 线程 = 1 compute WG + 1 第二腿 WG）\n", sm);
    std::printf("  计算腿 tile = %dx%dx%d BF16 WGMMA，reps=%lld；MEM 腿遍历 %.1f MiB\n",
                kBM, kBN, kBK, reps, mem_n * 16.0 / 1048576.0);
    std::printf("  COMP (WGMMA/张量核)   only = %8.1f us\n", tc);
    std::printf("  TENS (WGMMA/张量核)   only = %8.1f us\n", tt);
    std::printf("  MEM  (RMW/访存)       only = %8.1f us  (%.1f GB/s 读+写)\n", tm, gbps(tm));
    std::printf("  COMP + MEM   fused    = %8.1f us  eff=%.3f  (ideal=max(%.1f,%.1f), serial=%.1f)\n",
                tcm, eff(tc, tm, tcm), tc, tm, tc + tm);
    std::printf("  COMP + TENS  fused    = %8.1f us  eff=%.3f  (ideal=max(%.1f,%.1f), serial=%.1f)\n",
                tct, eff(tc, tt, tct), tc, tt, tc + tt);
    std::printf("  eff=1 理想重叠、eff=0 完全串行\n");

    // ---- 预算推演：用第 182 篇的真实数字 + 本实验的 eff，算「重算 + shadow」的可达上限 ----
    const double rec = 356.7, dgrad = 193.3, wgrad = 535.9;
    const double eff_f = eff(tc, tm, tcm);      // 访存腿吸收计算腿的实测重叠效率
    // 只有 wgrad(访存) 能与 rec(计算) 重叠；dgrad(计算) 与 rec 抢张量核 → 串行。
    const double serial = rec + dgrad + wgrad;
    const double budget = std::max(rec, wgrad) + dgrad;           // 理想：rec 藏进 wgrad
    const double fused  = (rec + wgrad) * eff_f > 0 ? (rec + wgrad) / (1 + eff_f * ((rec + wgrad) / std::max(rec, wgrad) - 1)) : serial;
    const double real   = fused + dgrad;
    std::printf("\n# 预算推演（第 182 篇真实数字 rec=%.1f dgrad=%.1f wgrad=%.1f us）\n", rec, dgrad, wgrad);
    std::printf("  串行 3 段          = %.1f us\n", serial);
    std::printf("  理想（rec 全藏进 wgrad）= max(rec,wgrad)+dgrad = %.1f us  → %.2fx\n", budget, serial / budget);
    std::printf("  用实测 eff=%.3f    → %.1f us  → %.2fx（两 kernel 两 stream 实测仅 1.09x）\n", eff_f, real, serial / real);
    return 0;
}
