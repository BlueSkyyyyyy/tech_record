// P157-1 实验：把 Mega MoE 的「单 kernel 内 通信 warp + 计算 warp 按寄存器预算分区」搬到 H100 上做原型。
//
// 问题（第 144/155 篇留下的）：
//   * 第 144 篇：NCCL 侧流与 GEMM **不重叠**（NCCL 抓满 SM）。
//   * 第 155 篇：把流水线推进到 micro-batch 维度——真实 grouped GEMM × NCCL a2a 的通信腿只被藏住
//     一小部分（ovf 1.21×/1.09× vs 理想 1.39/1.86）；**正对照**（限块 NVLink scatter）给了小 SM 预算后
//     能到 **~99%** 理想重叠。但那仍**是两个 kernel 靠硬件 co-residency**。
//   Mega MoE 真正做的是第三个方案：**同一 grid、同一 CTA** 里用 warp specialization + **动态寄存器
//     重分配**把通信段和 MMA 段放进同一块 SM。本文件把这条路径在 H100 上做成可跑原型。
//
// 真实依据（源码已核对，见 docs/167）：
//   * DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`：
//       :56   `__launch_bounds__(kNumThreads, 1)`
//       :326-335  寄存器预算 `{kNumDispatch,kNumNonEpilogue,kNumEpilogue}Registers` = {48,40,208}
//                 （或 {96,72,168}），`DG_STATIC_ASSERT(... <= 64512, "Too many registers")`
//       :344   dispatch warps   `warpgroup_reg_dealloc<kNumDispatchRegisters>()`
//       :692/:758/:821/:941  MMA non-epilogue warps `warpgroup_reg_dealloc<kNumNonEpilogueRegisters>()`
//       :948   epilogue/combine warps `warpgroup_reg_alloc<kNumEpilogueRegisters>()`
//   * DeepEP `deep_ep/include/deep_ep/impls/ep/hybrid_combine.cuh`：
//       :106-112  scale-up(NVLink) warps `ptx::warpgroup_reg_realloc<40>()`
//       :353-359  forward(RDMA) warps   `ptx::warpgroup_reg_realloc<256-40>()`
//       底层 `deep_ep/include/deep_ep/common/ptx.cuh:456-469` = `setmaxnreg.inc/dec.sync.aligned.u32`。
//
// 本文件用 `setmaxnreg.dec/inc` 内联 PTX（与 DeepEP `ptx.cuh:466-468` 同指令）建四种方案：
//   mode0  serial      : 通信 kernel → 计算 kernel（单流，无重叠）
//   mode1  coresident  : 通信 kernel ∥ 计算 kernel（两条流、**自然网格**，靠硬件 co-residency）
//   mode2  coresident_lim : 通信 kernel B 块 ∥ 计算 kernel (SM-B) 块（两条流、**限块**，第 155 篇正对照）
//   mode3  fused_ws    : **单 kernel**，每 CTA 1 个 comm warpgroup + 3 个 compute warpgroup，
//                         按寄存器预算 dealloc/alloc，同一 CTA 内并发（= Mega MoE 的形态）
//   mode4  fused_ws_norealloc : 同 mode3 但**不做** setmaxnreg（计算路径因寄存器不足而 spill）
//
// 计算段故意做得**吃寄存器**（kAcc=144 个独立累加链，约需 152 个寄存器）：这样「给不给寄存器」可测，
// 也复现了 Mega MoE epilogue warp 需要 208 > 均匀 128 的处境（均匀预算下会被 ptxas spill）。
//
// 跑不了的：Mega MoE 融合 kernel 本体是 SM100 专属（`mega_moe.hpp` 只在 arch10 派发）→ 仅分析。
// 本文件跑的是**架构无关**的 warp specialization + setmaxnreg + NVLink peer copy 原语，H100(sm90a) 可跑。
//
// 编译：nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr \
//         src/mega_moe_like_ws.cu -o /tmp/mega_moe_like_ws
// 运行：/tmp/mega_moe_like_ws   （输出见 harness/exp_mega_moe_like_ws.out.txt）

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e), __FILE__, __LINE__);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// setmaxnreg 原语（与 DeepEP ptx.cuh:466-468 同指令；每 warpgroup 128 线程统一执行）
// ---------------------------------------------------------------------------
template <int kNumRegs>
__device__ __forceinline__ void reg_dec() {
    static_assert(kNumRegs % 8 == 0 && kNumRegs >= 24 && kNumRegs <= 256, "invalid regs");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}
template <int kNumRegs>
__device__ __forceinline__ void reg_inc() {
    static_assert(kNumRegs % 8 == 0 && kNumRegs >= 24 && kNumRegs <= 256, "invalid regs");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}

// ---------------------------------------------------------------------------
// 通信段：本 rank 把一段发送写进各 peer 的 recv（NVLink 远端写）。与第 155 篇同款。
// ---------------------------------------------------------------------------
__device__ __forceinline__ size_t comm_total(int world, size_t blk_u4) {
    return (size_t)(world - 1) * blk_u4;
}
__device__ __forceinline__ void comm_step(const uint4* __restrict__ send,
                                          uint4* const* __restrict__ recvs,
                                          int me, int world, size_t blk_u4, size_t idx) {
    const int pi = (int)(idx / blk_u4);
    const int j = (pi < me) ? pi : pi + 1;
    const size_t off = idx - (size_t)pi * blk_u4;
    recvs[j][(size_t)me * blk_u4 + off] = send[(size_t)j * blk_u4 + off];
}

// 独立通信 kernel（mode0/1/2）
__global__ void comm_kernel(const uint4* __restrict__ send, uint4* const* __restrict__ recvs,
                            int me, int world, size_t blk_u4) {
    const size_t total = comm_total(world, blk_u4);
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x)
        comm_step(send, recvs, me, world, blk_u4, idx);
}

// ---------------------------------------------------------------------------
// 计算段：寄存器压力型 FMA 燃烧（kAcc 个独立累加链）。kAcc=144 需 ~152 寄存器，
// 用来让「给不给寄存器」可测（对照 Mega MoE epilogue warp 208 regs）。
// ---------------------------------------------------------------------------
constexpr int kAcc = 144;
__device__ __forceinline__ float burn(long long iters, float a) {
    float acc[kAcc];
    #pragma unroll
    for (int i = 0; i < kAcc; ++i) acc[i] = a + i;
    #pragma unroll 2
    for (long long t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < kAcc; ++i) acc[i] = fmaf(acc[i], 1.0000001f, a);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < kAcc; ++i) s += acc[i];
    return s;
}

// 独立计算 kernel（mode0/1/2）：256 线程/块 → 均匀 255 regs，充足。
__global__ void __launch_bounds__(256, 1) burn_kernel(float* __restrict__ out, long long iters) {
    float s = burn(iters, (float)blockIdx.x);
    if (s == 0.1234567f) out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// 融合 kernel（mode3/4）：单 grid，每 CTA = 1 个 comm warpgroup + 3 个 compute warpgroup。
//   realloc=true  : comm WG dec 到 kRegComm，compute WG inc 到 kRegComp（Mega MoE 形态）
//   realloc=false : 不做重分配 → 均匀 128 regs，计算路径 spill
// ---------------------------------------------------------------------------
// 两种 CTA 形态（对应 Mega MoE 512 线程/1 CTA/SM 与 DeepEP hybrid_combine 2 warpgroup）：
//   A: 1 comm WG + 3 comp WG = 512 线程 → 63488 regs/CTA，1 CTA/SM
//   B: 1 comm WG + 1 comp WG = 256 线程 → comp 拿 216 regs，grid=2*SM
constexpr int kCommWGs = 1;
constexpr int kRegComm = 40;                 // 对照 Mega MoE kNumNonEpilogueRegisters=40
constexpr int kRegCompA = 152;               // (65536 - 40*128) / (3*128) 向下 8 的倍数
constexpr int kRegCompB = 216;               // 256 线程/CTA 时 compute WG 可得的上限

template <int kCompWGs, int kRegComp, bool kRealloc>
__global__ void __launch_bounds__((kCommWGs + kCompWGs) * 128, 1)
fused_kernel(const uint4* __restrict__ send, uint4* const* __restrict__ recvs,
             int me, int world, size_t blk_u4, float* __restrict__ out, long long comp_iters,
             int do_comm) {
    const int wg = (int)(threadIdx.x >> 7);
    const size_t total = comm_total(world, blk_u4);
    if (wg == 0) {
        if constexpr (kRealloc) reg_dec<kRegComm>();
        const int stride = (int)gridDim.x * 128;
        if (do_comm)
            for (size_t idx = (size_t)blockIdx.x * 128 + (threadIdx.x & 127); idx < total; idx += stride)
                comm_step(send, recvs, me, world, blk_u4, idx);
    } else {
        if constexpr (kRealloc) reg_inc<kRegComp>();
        const long long cthreads = (long long)kCompWGs * 128 * gridDim.x;
        const long long iters = comp_iters / cthreads / kAcc;
        float s = burn(iters, (float)blockIdx.x + (float)wg);
        if (s == 0.1234567f) out[(size_t)blockIdx.x * 128 + (threadIdx.x & 127)] = s;
    }
}

// ---------------------------------------------------------------------------
#ifndef WORLD
#define WORLD 8
#endif
static constexpr int kWorld = WORLD;  // 可用 -DWORLD=7 + CUDA_VISIBLE_DEVICES 避开被占用的卡
static constexpr size_t kCommBytesPerRank = 168ull * 1024 * 1024;  // 与第 155 篇同量级（可被 world*16 整除）
static constexpr size_t kBlkBytes = kCommBytesPerRank / kWorld;

struct DevState {
    uint4* send = nullptr;
    uint4* recv = nullptr;
    uint4** d_ptr = nullptr;
    float* burn_out = nullptr;
    cudaStream_t s_comm, s_comp, s_timing;
    cudaEvent_t e0, e1;
};

int main() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    std::printf("visible GPUs = %d\n", ndev);
    if (ndev < kWorld) { std::printf("need >= %d GPUs; abort\n", kWorld); return 1; }
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    std::printf("SM count = %d, comm bytes/rank = %.1f MB (%.1f MB/peer)\n",
                sm_count, kCommBytesPerRank / 1048576.0, kBlkBytes / 1048576.0);
    std::printf("fused A = 512 thr (1 comm WG + 3 comp WG, 1 CTA/SM, comp->%d); "
                "fused B = 256 thr (1 comm WG + 1 comp WG, grid=2*SM, comp->%d); kAcc = %d; comm->%d\n",
                kRegCompA, kRegCompB, kAcc, kRegComm);

    const size_t blk_u4 = kBlkBytes / 16;

    std::vector<DevState> S(kWorld);
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&S[d].send, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].recv, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].burn_out, 4096 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&S[d].d_ptr), kWorld * sizeof(uint4*)));
        CUDA_CHECK(cudaMemset(S[d].send, 0x3f, kWorld * kBlkBytes));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_comm));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_comp));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_timing));
        CUDA_CHECK(cudaEventCreate(&S[d].e0));
        CUDA_CHECK(cudaEventCreate(&S[d].e1));
    }
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        std::vector<uint4*> host_ptr(kWorld, nullptr);
        for (int j = 0; j < kWorld; ++j) {
            if (j == d) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, d, j));
            if (can) CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
        }
        for (int j = 0; j < kWorld; ++j) host_ptr[j] = S[j].recv;
        CUDA_CHECK(cudaMemcpy(S[d].d_ptr, host_ptr.data(), kWorld * sizeof(uint4*),
                              cudaMemcpyHostToDevice));
    }

    auto sync_all = [&]() {
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); }
    };

    // 单次测量：mode + 网格 + 计算量
    auto timed_once = [&](int mode, int cblocks, int gblocks, long long comp_fma, int iters) -> double {
        sync_all();
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_timing); }
        for (int it = 0; it < iters; ++it) {
            for (int d = 0; d < kWorld; ++d) {
                CUDA_CHECK(cudaSetDevice(d));
                DevState& s = S[d];
                // 独立 kernel 的每块 FMA 数
                const long long per = gblocks > 0 ? comp_fma / ((long long)gblocks * 256) : 0;
                const long long per_iters = per / kAcc;
                if (mode == 0) {
                    if (cblocks > 0) comm_kernel<<<cblocks, 256, 0, s.s_timing>>>(s.send, s.d_ptr, d, kWorld, blk_u4);
                    if (gblocks > 0) burn_kernel<<<gblocks, 256, 0, s.s_timing>>>(s.burn_out, per_iters);
                } else if (mode == 1) {
                    if (cblocks > 0) comm_kernel<<<cblocks, 256, 0, s.s_comm>>>(s.send, s.d_ptr, d, kWorld, blk_u4);
                    if (gblocks > 0) burn_kernel<<<gblocks, 256, 0, s.s_comp>>>(s.burn_out, per_iters);
                } else if (mode == 2) {  // 限块 coresident
                    if (cblocks > 0) comm_kernel<<<cblocks, 256, 0, s.s_comm>>>(s.send, s.d_ptr, d, kWorld, blk_u4);
                    if (gblocks > 0) burn_kernel<<<gblocks, 256, 0, s.s_comp>>>(s.burn_out, per_iters);
                } else if (mode == 3) {   // fused B（256 thr, grid=2*SM）：comm + compute
                    fused_kernel<1, kRegCompB, true><<<2 * sm_count, 256, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, comp_fma, 1);
                } else if (mode == 4) {   // fused B 无 realloc
                    fused_kernel<1, kRegCompB, false><<<2 * sm_count, 256, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, comp_fma, 1);
                } else if (mode == 5) {   // fused B：只有 comm
                    fused_kernel<1, kRegCompB, true><<<2 * sm_count, 256, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, 0, 1);
                } else if (mode == 6) {   // fused B：只有 compute
                    fused_kernel<1, kRegCompB, true><<<2 * sm_count, 256, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, comp_fma, 0);
                } else if (mode == 7) {   // fused A（1 CTA/SM）：comm + compute
                    fused_kernel<3, kRegCompA, true><<<sm_count, 512, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, comp_fma, 1);
                } else if (mode == 8) {   // fused A：只有 comm
                    fused_kernel<3, kRegCompA, true><<<sm_count, 512, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, 0, 1);
                } else {                  // mode 9：fused A 只有 compute
                    fused_kernel<3, kRegCompA, true><<<sm_count, 512, 0, s.s_timing>>>(
                        s.send, s.d_ptr, d, kWorld, blk_u4, s.burn_out, comp_fma, 0);
                }
            }
        }
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            cudaEvent_t ec, ep;
            CUDA_CHECK(cudaEventCreateWithFlags(&ec, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ep, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(ec, S[d].s_comm));
            CUDA_CHECK(cudaEventRecord(ep, S[d].s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(S[d].s_timing, ec, 0));
            CUDA_CHECK(cudaStreamWaitEvent(S[d].s_timing, ep, 0));
            CUDA_CHECK(cudaEventRecord(S[d].e1, S[d].s_timing));
            cudaEventDestroy(ec); cudaEventDestroy(ep);
        }
        sync_all();
        double t = 0;
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, S[d].e0, S[d].e1)); t += ms;
        }
        return t / kWorld / iters / 1e3;
    };

    auto timed = [&](int mode, int cblocks, int gblocks, long long comp_fma, int iters) -> double {
        double v[3];
        for (int i = 0; i < 3; ++i) v[i] = timed_once(mode, cblocks, gblocks, comp_fma, iters);
        std::sort(v, v + 3);
        return v[1];
    };

    // 标定用的固定计算量（让 t_comp 与 t_comm 同量级）：先粗测一次
    const int iters = 12;
    const long long kBaseFMA = (long long)sm_count * 2 * 256 * 200000LL;  // ≈ 13.5 G FMA@132SM

    // ---- 标定：独立的通信 / 计算（都取「充分并行」网格）
    const int cgrid_iso = sm_count * 2;      // 264 块 comm
    const int ggrid_iso = sm_count * 2;      // 264 块 compute
    double t_comm_iso = timed(1, cgrid_iso, 0, 0, iters);
    double t_comp_iso = timed(1, 0, ggrid_iso, kBaseFMA, iters);
    std::printf("\n# 标定（独立 kernel）\n");
    std::printf("  t_comm_iso = %8.1f us (%.1f GB/s, %d 块)\n",
                t_comm_iso * 1e6, (kWorld - 1) * kBlkBytes / t_comm_iso / 1e9, cgrid_iso);
    std::printf("  t_comp_iso = %8.1f us (%.1f TFLOPS, %d 块, %.2f G FMA)\n",
                t_comp_iso * 1e6, 2.0 * kBaseFMA / t_comp_iso / 1e12, ggrid_iso, kBaseFMA / 1e9);

    // ---- 融合 kernel 内 通信腿 / 计算腿 单独跑（B 形态：256 thr, grid=2*SM）
    double tB_comm = timed(5, 0, 0, 0, iters);
    double tB_comp = timed(6, 0, 0, kBaseFMA, iters);
    double tA_comm = timed(8, 0, 0, 0, iters);
    double tA_comp = timed(9, 0, 0, kBaseFMA, iters);
    std::printf("  fusedB(256thr, grid=2*SM) comm 单独 = %8.1f us (%.1f GB/s) | comp 单独 = %8.1f us (%.1f TFLOPS)\n",
                tB_comm * 1e6, (kWorld - 1) * kBlkBytes / tB_comm / 1e9,
                tB_comp * 1e6, 2.0 * kBaseFMA / tB_comp / 1e12);
    std::printf("  fusedA(1 CTA/SM, 512thr) comm 单独 = %8.1f us (%.1f GB/s) | comp 单独 = %8.1f us (%.1f TFLOPS)\n",
                tA_comm * 1e6, (kWorld - 1) * kBlkBytes / tA_comm / 1e9,
                tA_comp * 1e6, 2.0 * kBaseFMA / tA_comp / 1e12);

    // ---- A) 寄存器分区对「吃寄存器」的计算段的效果（fused B 形态：realloc on vs off）
    double t_fused_r = tB_comp;
    double t_fused_nr = timed(4, 0, 0, kBaseFMA, iters);
    std::printf("\n# A) 融合 kernel 计算段：setmaxnreg 开/关（kAcc=%d，需 ~152 regs；B 形态 256thr）\n", kAcc);
    std::printf("  realloc=on  %8.1f us (%.1f TFLOPS)\n", t_fused_r * 1e6, 2.0 * kBaseFMA / t_fused_r / 1e12);
    std::printf("  realloc=off %8.1f us (%.1f TFLOPS)   off/on = %.2fx\n",
                t_fused_nr * 1e6, 2.0 * kBaseFMA / t_fused_nr / 1e12,
                t_fused_r > 0 ? t_fused_nr / t_fused_r : 0);

    // ---- B) 重叠：固定 comm、扫计算量。每方案用**自己的腿**算 eff（避免拿别方案的算核做参照）。
    const int Bcomm = std::max(8, sm_count / 4);
    const int glim = std::max(1, sm_count - Bcomm);
    std::printf("\n# B) 重叠对照（comm 固定 %.1f MB/rank；扫计算量）\n", kCommBytesPerRank / 1048576.0);
    std::printf("  cores_lim: comm %d 块 ∥ compute %d 块；cores_nat: 两者都 %d 块自然网格\n",
                Bcomm, glim, cgrid_iso);
    std::printf("  fusedA=1 comm WG+3 comp WG, 1 CTA/SM; fusedB=1 comm WG+1 comp WG, 256 thr grid=2*SM\n");
    std::printf("%-7s | %-22s | %-22s | %-22s | %-22s | %-22s\n",
                "scale", "serial [eff]", "cores_nat [eff]", "cores_lim [eff]", "fusedA [eff]", "fusedB [eff]");
    std::printf("%-7s | %6s %6s %6s | %6s %6s %6s | %6s %6s %6s | %6s %6s %6s | %6s %6s %6s\n",
                "", "t", "tc", "tg", "t", "tc", "tg", "t", "tc", "tg", "t", "tc", "tg", "t", "tc", "tg");
    for (double scale : {0.5, 1.0, 2.0}) {
        const long long fma = (long long)(kBaseFMA * scale);
        // 各方案自己的腿
        double s_c = timed(1, cgrid_iso, 0, 0, iters);
        double s_g = timed(1, 0, ggrid_iso, fma, iters);
        double l_c = timed(1, Bcomm, 0, 0, iters);
        double l_g = timed(1, 0, glim, fma, iters);
        double fB_c = timed(5, 0, 0, 0, iters);
        double fB_g = timed(6, 0, 0, fma, iters);
        double fA_c = timed(8, 0, 0, 0, iters);
        double fA_g = timed(9, 0, 0, fma, iters);
        // 总时间
        double t_serial = timed(0, cgrid_iso, ggrid_iso, fma, iters);
        double t_cores_nat = timed(1, cgrid_iso, ggrid_iso, fma, iters);
        double t_cores_lim = timed(2, Bcomm, glim, fma, iters);
        double t_fA = timed(7, 0, 0, fma, iters);
        double t_fB = timed(3, 0, 0, fma, iters);
        auto eff = [](double tc, double tg, double t) {
            double s = tc + tg, ideal = std::max(tc, tg);
            return (s / t - 1.0) / (s / ideal - 1.0);
        };
        std::printf("%-7.1f | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f\n",
                    scale, t_serial * 1e6, s_c * 1e6, s_g * 1e6,
                    t_cores_nat * 1e6, s_c * 1e6, s_g * 1e6,
                    t_cores_lim * 1e6, l_c * 1e6, l_g * 1e6,
                    t_fA * 1e6, fA_c * 1e6, fA_g * 1e6,
                    t_fB * 1e6, fB_c * 1e6, fB_g * 1e6);
        std::printf("%-7s | %-22.3f | %-22.3f | %-22.3f | %-22.3f | %-22.3f\n", "",
                    eff(s_c, s_g, t_serial), eff(s_c, s_g, t_cores_nat),
                    eff(l_c, l_g, t_cores_lim), eff(fA_c, fA_g, t_fA), eff(fB_c, fB_g, t_fB));
    }
    std::printf("  eff=1 为理想重叠（t=max(腿)）、eff=0 为完全串行（t=腿之和）；>1 为测量噪声/超理想。\n");

    // ---- 正确性：fused comm 段写入 peer recv 槽位
    {
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaMemset(S[d].recv, 0, kWorld * kBlkBytes); }
        sync_all();
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            fused_kernel<1, kRegCompB, true><<<2 * sm_count, 256, 0, S[d].s_timing>>>(
                S[d].send, S[d].d_ptr, d, kWorld, blk_u4, S[d].burn_out, 0, 1);
        }
        sync_all();
        int bad = 0;
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            std::vector<uint8_t> host(kWorld * kBlkBytes);
            CUDA_CHECK(cudaMemcpy(host.data(), S[d].recv, kWorld * kBlkBytes, cudaMemcpyDeviceToHost));
            for (int j = 0; j < kWorld; ++j) {
                if (j == d) continue;
                const uint8_t* p = host.data() + (size_t)j * kBlkBytes;
                if (p[0] != 0x3f || p[kBlkBytes - 1] != 0x3f) ++bad;
            }
        }
        std::printf("\n# 正确性：fused comm 段写入 peer recv 槽位（bad slots = %d）\n", bad);
    }

    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaFree(S[d].send); cudaFree(S[d].recv); cudaFree(S[d].burn_out); cudaFree(S[d].d_ptr);
        cudaStreamDestroy(S[d].s_comm); cudaStreamDestroy(S[d].s_comp); cudaStreamDestroy(S[d].s_timing);
        cudaEventDestroy(S[d].e0); cudaEventDestroy(S[d].e1);
    }
    return 0;
}
