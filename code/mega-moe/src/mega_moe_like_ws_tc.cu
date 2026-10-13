// P157-1 实验（配套 B）：为什么 Mega MoE 的单 kernel 融合在**目标卡**上成立，在 H100 CUDA-core 上却不理想？
//   假设：Mega MoE 的计算段是 **异步张量核 UMMA**（`SM100_MMA_MXF8F6F4_2x1SM_SS` / `SM100_MMA_F16BF16_`,
//   SM100），它不占用 CUDA core 的发射/LSU 通路，所以 dispatch warp 能真正并发；而本系列 A 部分用
//   CUDA-core FMA 当计算腿，会和通信腿抢同一批发射槽。
//
//   本文件在 H100(sm90a) 上用**真实 WGMMA**（SM90 的异步张量核，`deep_gemm/mma/sm90.cuh` + `ptx/wgmma.cuh`）
//   当计算腿，和 CUDA-core FMA 腿做对照：同为「1 comm WG + 1 compute WG、2 CTA/SM、setmaxnreg 分区」，
//   量两者和通信腿的重叠效率。若 WGMMA 腿重叠明显更好 ⇒ 坐实「异步张量核是融合的前提」。
//
//   Mega MoE 融合 kernel 本体仍是 SM100 专属 → 仅分析；这里跑的是架构无关的 WGMMA/comm/寄存器原语。
//
// 编译（需 DeepGEMM 头）：
//   nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
//     -I$DG/deep_gemm/include -I$DG/deep_gemm/include/cutlass -I$DG/deep_gemm/include/cute \
//     src/mega_moe_like_ws_tc.cu -o /tmp/mega_moe_like_ws_tc
// 运行：CUDA_VISIBLE_DEVICES=1..7 /tmp/mega_moe_like_ws_tc

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <ctime>
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

__device__ __forceinline__ size_t ctot(int w, size_t b) { return (size_t)(w - 1) * b; }
__device__ __forceinline__ void cstep(const uint4* s, uint4* const* r, int me, int w, size_t b, size_t i) {
    int pi = (int)(i / b); int j = (pi < me) ? pi : pi + 1; size_t off = i - (size_t)pi * b;
    r[j][(size_t)me * b + off] = s[(size_t)j * b + off];
}

// CUDA-core FMA 腿
constexpr int kAcc = 144;
__device__ __forceinline__ float burn(long long it, float a) {
    float acc[kAcc];
    #pragma unroll
    for (int i = 0; i < kAcc; ++i) acc[i] = a + i;
    #pragma unroll 2
    for (long long t = 0; t < it; ++t) {
        #pragma unroll
        for (int i = 0; i < kAcc; ++i) acc[i] = fmaf(acc[i], 1.0000001f, a);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < kAcc; ++i) s += acc[i];
    return s;
}

using Bf16 = nv_bfloat16;
constexpr int kBM = 64, kBN = 64, kBK = 64;
using MMA = mma90::BF16MMASelector<kBN, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;

__device__ __forceinline__ void fill_swz(Bf16* smem, int rows, int k) {
    constexpr int EPB = 16 / sizeof(Bf16);
    for (int i = threadIdx.x & 127; i < rows * k; i += 128) {
        int row = i / k, col = i % k, chunk = col / EPB, within = col % EPB;
        smem[row * k + (chunk ^ (row & 7)) * EPB + within] = (Bf16)((row * 7 + col * 3) % 64) * (Bf16)0.01f;
    }
}

// 融合 kernel：1 comm WG + 1 compute WG = 256 线程。kUseMMA 决定计算腿是 WGMMA(异步张量核) 还是 FMA(CUDA core)。
// reps：WGMMA 的重复次数（每次 64x64x64 = 4 个 64x64x16 步）；fma_iters：FMA 轮数。
template <bool kUseMMA, int kRegComm, int kRegComp, bool kRealloc>
__global__ void __launch_bounds__(256, 1)
fused(const uint4* __restrict__ send, uint4* const* __restrict__ recvs, int me, int world, size_t blk_u4,
      float* __restrict__ out, long long reps, long long fma_iters, int do_comm) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    const int wg = threadIdx.x >> 7;
    if (wg == 0) {
        if constexpr (kRealloc) rdec<kRegComm>();
        const size_t T = ctot(world, blk_u4);
        if (do_comm)
            for (size_t i = (size_t)blockIdx.x * 128 + (threadIdx.x & 127); i < T; i += (size_t)gridDim.x * 128)
                cstep(send, recvs, me, world, blk_u4, i);
    } else {
        if constexpr (kRealloc) rinc<kRegComp>();
        if constexpr (kUseMMA) {
            Bf16* sa = reinterpret_cast<Bf16*>(smem_raw);
            Bf16* sb = sa + kBM * kBK;
            fill_swz(sa, kBM, kBK);
            fill_swz(sb, kBN, kBK);
            __syncwarp();
            constexpr int kNumSteps = kBK / MMA::K;
            float accum[MMA::kNumAccum] = {};
            // 每个 warpgroup 用其 128 个线程参与
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
        } else {
            float s = burn(fma_iters, (float)threadIdx.x);
            if (s == 0.1234567f) out[threadIdx.x] = s;
        }
    }
}

#ifndef WORLD
#define WORLD 7
#endif
static constexpr int kWorld = WORLD;
static constexpr size_t kCommBytesPerRank = 168ull * 1024 * 1024;
static constexpr size_t kBlkBytes = kCommBytesPerRank / kWorld;

struct St { uint4* send; uint4* recv; uint4** d; float* out; cudaStream_t st; cudaEvent_t e0, e1; };

int main() {
    int nd; CUDA_CHECK(cudaGetDeviceCount(&nd)); std::printf("visible GPUs = %d\n", nd);
    if (nd < kWorld) { std::printf("need %d\n", kWorld); return 1; }
    int sm; CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
    std::printf("SM count = %d, comm=%.1f MB/rank\n", sm, kBlkBytes / 1048576.0);
    const size_t blk_u4 = kBlkBytes / 16;
    std::vector<St> S(kWorld);
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&S[d].send, kWorld * kBlkBytes)); CUDA_CHECK(cudaMalloc(&S[d].recv, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].out, 4096 * sizeof(float))); CUDA_CHECK(cudaMalloc((void**)&S[d].d, kWorld * sizeof(uint4*)));
        CUDA_CHECK(cudaMemset(S[d].send, 0x3f, kWorld * kBlkBytes));
        CUDA_CHECK(cudaStreamCreate(&S[d].st)); CUDA_CHECK(cudaEventCreate(&S[d].e0)); CUDA_CHECK(cudaEventCreate(&S[d].e1));
    }
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d)); std::vector<uint4*> h(kWorld);
        for (int j = 0; j < kWorld; ++j) { if (j == d) continue; int c; CUDA_CHECK(cudaDeviceCanAccessPeer(&c, d, j)); if (c) CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0)); }
        for (int j = 0; j < kWorld; ++j) h[j] = S[j].recv;
        CUDA_CHECK(cudaMemcpy(S[d].d, h.data(), kWorld * sizeof(uint4*), cudaMemcpyHostToDevice));
    }
    auto sync = [&] { for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); } };
    const int grid = 2 * sm;
    const size_t smem = (kBM * kBK + kBN * kBK) * sizeof(Bf16);

    auto once = [&](bool mma, long long reps, long long fma_iters, int do_comm, int iters) -> double {
        sync();
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; ++it) for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d)); St& s = S[d];
            if (mma) fused<true, 40, 216, true><<<grid, 256, smem, s.st>>>(s.send, s.d, d, kWorld, blk_u4, s.out, reps, 0, do_comm);
            else     fused<false, 40, 216, true><<<grid, 256, 0, s.st>>>(s.send, s.d, d, kWorld, blk_u4, s.out, 0, fma_iters, do_comm);
            { cudaError_t le = cudaPeekAtLastError(); if (le) { std::fprintf(stderr, "launch mma=%d dev=%d err=%s\n", (int)mma, d, cudaGetErrorString(le)); std::exit(1); } }
        }
        sync();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        return ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / iters;
    };
    auto med = [&](bool mma, long long reps, long long fma_iters, int dc, int iters) {
        double v[3]; for (int i = 0; i < 3; ++i) v[i] = once(mma, reps, fma_iters, dc, iters);
        std::sort(v, v + 3); return v[1];
    };

    const int it = 10;
    // 标定 WGMMA：找 reps 使 compute-only ≈ comm
    long long probe = 20000;
    for (;;) {
        double t = med(true, probe, 0, 0, 4);
        std::printf("  probe WGMMA reps=%lld -> %.1f us\n", probe, t * 1e6);
        if (t < 0.35e-3) probe *= 4; else if (t > 1.2e-3) probe /= 2; else break;
        if (probe < 1 || probe > 100000000) break;
    }
    long long reps = probe;
    std::printf("\n# 用 WGMMA reps=%lld（64x64x64/rep）；FMA 用 6000 轮（kAcc=%d）\n", reps, kAcc);

    double tc_comm = med(true, 0, 0, 1, it);          // comm only
    double tc_mma = med(true, reps, 0, 0, it);        // WGMMA only
    double tc_fma = med(false, 0, 6000, 0, it);     // FMA only
    double tc_both_mma = med(true, reps, 0, 1, it);   // comm + WGMMA
    double tc_both_fma = med(false, 0, 6000, 1, it);// comm + FMA
    auto eff = [](double tc, double tg, double t) { double s = tc + tg, ideal = std::max(tc, tg); return (s / t - 1) / (s / ideal - 1); };
    std::printf("\n# 结果（1 CTA/SM，256 线程 = 1 comm WG + 1 compute WG，setmaxnreg 40/216）\n");
    std::printf("  comm only          = %8.1f us (%.1f GB/s)\n", tc_comm * 1e6, (kWorld - 1) * kBlkBytes / tc_comm / 1e9);
    std::printf("  WGMMA(异步张量核)  only = %8.1f us\n", tc_mma * 1e6);
    std::printf("  FMA(CUDA core)     only = %8.1f us\n", tc_fma * 1e6);
    std::printf("  comm + WGMMA       = %8.1f us  eff=%.3f  (ideal=max(%.1f,%.1f))\n",
                tc_both_mma * 1e6, eff(tc_comm, tc_mma, tc_both_mma), tc_comm * 1e6, tc_mma * 1e6);
    std::printf("  comm + FMA         = %8.1f us  eff=%.3f  (ideal=max(%.1f,%.1f))\n",
                tc_both_fma * 1e6, eff(tc_comm, tc_fma, tc_both_fma), tc_comm * 1e6, tc_fma * 1e6);
    std::printf("  eff=1 理想重叠、eff=0 完全串行；**异步张量核腿应显著高于 CUDA-core 腿**\n");

    // 正确性：comm 段
    for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaMemset(S[d].recv, 0, kWorld * kBlkBytes)); }
    sync();
    for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); fused<true, 40, 216, true><<<grid, 256, smem, S[d].st>>>(S[d].send, S[d].d, d, kWorld, blk_u4, S[d].out, 1, 0, 1); }
    sync();
    int bad = 0;
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d)); std::vector<uint8_t> h(kWorld * kBlkBytes);
        CUDA_CHECK(cudaMemcpy(h.data(), S[d].recv, kWorld * kBlkBytes, cudaMemcpyDeviceToHost));
        for (int j = 0; j < kWorld; ++j) { if (j == d) continue; const uint8_t* p = h.data() + (size_t)j * kBlkBytes; if (p[0] != 0x3f || p[kBlkBytes - 1] != 0x3f) ++bad; }
    }
    std::printf("\n# 正确性：comm 段 bad slots = %d\n", bad);
    return 0;
}
