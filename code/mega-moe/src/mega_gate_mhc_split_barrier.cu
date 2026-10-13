// P92-1 实验：Mega Gate / Mega mHC 的**跨 CTA / 跨 split 分段屏障**与工作区布局。
//   - `deep_gemm/include/deep_gemm/scheduler/mega_gate.cuh` 的 `ScoreBarrier<kNumLogicalCtas>`
//   - `deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh` 的 `SplitBarrier<kBarrierIndex>`（`Norm`/`Mix`）
//   - 它们所依赖的 `layout/mega_gate.cuh` 的 `Workspace` 与 `layout/mega_mhc.cuh` 的 `Workspace`
// 在 H100(sm90) 上**直接 include 真实头文件、编译运行真实设备代码**（只用 ptx 内联 asm，架构无关）。
//
// 背景：第 5 篇（Mega Gate）只在 `epilogue:39-78` 一带提过「split-K 时经 score scratch + ScoreBarrier
//   跨 CTA 汇聚（scheduler:13-39）」；第 5 篇下篇（Mega mHC）只在 `scheduler:13-49` 提过
//   「SplitBarrier 用 grid 序号+到达数一次编码、免复位（Norm/Mix 两个）」。这两个调度器文件本身
//   （共 96 行）从没被单独拆过，也没有被真机跑过。本篇补上。
//
// 覆盖（全部 include 真实头文件，非手抄）：
//   A) `ScoreBarrier` 全协议——C 个逻辑 CTA 一 init 其余 wait_init，逐个 arrive，再 wait；
//      真机验证终止态 = `encode_state(grid_idx, C)`，并**连续两次 launch 不重置缓冲**证明免复位。
//   B) `ScoreBarrier::wait_init` 的**无符号距离**语义——用「延迟 init」实验区分
//      ① 陈旧上一轮终止态（必须拒绝、等到本轮 init）、② 本轮已 arm（立即通过）、
//      ③ 本轮已有部分到达（立即通过）。
//   C) `SplitBarrier::get_ptr` 的 line 偏移（`kBarrierIndex*kNumMaxMBlocks + m_block_idx`）与
//      Norm(0)/Mix(1) 两个索引独立性；以及 `SplitBarrier` 全协议（跨 SM 的 owner init + 其余
//      wait_init + 每 split 一次 arrive），连续两次 launch 免复位。
//   D) 两个 `Workspace` 的 host 侧算术（scratch 字节数、score/partial 指针偏移、m_blocks 公式）。
//
// 编译/运行见 harness/exp_mega_gate_mhc_split_barrier.sh。目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include <deep_gemm/scheduler/mega_gate.cuh>
#include <deep_gemm/scheduler/mega_mhc.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/layout/mega_mhc.cuh>

using namespace deep_gemm;
namespace gg = deep_gemm::layout::mega_gate;
namespace mm = deep_gemm::layout::mega_mhc;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

static int g_fail = 0;
static void report(const char* name, bool ok) {
    printf("%-46s %s\n", name, ok ? "PASS" : "FAIL");
    if (not ok) g_fail += 1;
}

// 真实 `arrive()` 调的是 `ptx::red_async_inc_rel`（`red.async.release.gpu.global.add.u64`）——
// ptxas 在 sm_90a 直接报 `Feature 'st.async/red.async with .global state space' requires
// .target sm_100 or higher`。故 H100 上无法运行真实 arrive；用**同语义、非 async** 的
// `red.release.gpu.global.add.u64` 作为替身（其余 init/wait_init/wait/get_ptr/encode_state 全是真实头文件代码）。
static __device__ __forceinline__ void subst_arrive(uint64_t* ptr) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(ptr) : "memory");
}

// ---------------------------------------------------------------------------
// A) ScoreBarrier 全协议：C 个逻辑 CTA，一个 token block（数组里多块）
// ---------------------------------------------------------------------------
using GateWS = gg::Workspace<4, 128>;   // 仅用 barrier 指针，模板参数不影响

template <uint32_t kNumLogicalCtas>
__global__ void k_gate_protocol(void* scratch, void* barriers, uint32_t num_token_blocks,
                                uint64_t* out_terminal, uint32_t* out_mismatch) {
    using SB = sched::mega_gate::ScoreBarrier<kNumLogicalCtas>;
    const GateWS ws(scratch, barriers);
    const uint32_t rank = blockIdx.x;
    uint32_t mism = 0;
    for (uint32_t tb = 0; tb < num_token_blocks; ++ tb) {
        uint64_t* ptr = ws.get_score_barrier_ptr(tb);
        if (threadIdx.x == 0) {
            if (rank == 0) SB::init(ptr);
            else           SB::wait_init(ptr);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            subst_arrive(ptr);   // 真实 arrive 是 sm100-only 的 red.async，见文件头注释
            SB::wait(ptr);
        }
        __syncthreads();
        if (threadIdx.x == 0 and rank == 0) {
            const uint64_t term   = ptx::ld_acq_gpu(ptr);
            const uint64_t expect = SB::encode_state(ptx::get_grid_idx(), kNumLogicalCtas);
            if (term != expect) ++ mism;
            if (tb == 0) out_terminal[0] = term;
        }
    }
    if (threadIdx.x == 0) out_mismatch[rank] = mism;
}

template <uint32_t kNumLogicalCtas>
static void part_a(void* scratch, void* barriers, uint32_t num_token_blocks,
                   const char* label, bool zero_before) {
    if (zero_before) CK(cudaMemset(barriers, 0, (size_t)gg::kNumMaxTokenBlocks * gg::kScoreBarrierLineBytes));

    uint64_t* d_term; uint32_t* d_mism;
    CK(cudaMalloc(&d_term, sizeof(uint64_t)));
    CK(cudaMalloc(&d_mism, kNumLogicalCtas * sizeof(uint32_t)));
    CK(cudaMemset(d_term, 0, sizeof(uint64_t)));
    CK(cudaMemset(d_mism, 0, kNumLogicalCtas * sizeof(uint32_t)));

    // 连续两次 launch，**中间不重置缓冲**——验证免复位
    for (int rep = 0; rep < 2; ++ rep) {
        k_gate_protocol<kNumLogicalCtas><<<kNumLogicalCtas, 128>>>(scratch, barriers, num_token_blocks,
                                                                   d_term, d_mism);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }

    std::vector<uint32_t> mism(kNumLogicalCtas);
    uint64_t term = 0;
    CK(cudaMemcpy(mism.data(), d_mism, kNumLogicalCtas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&term, d_term, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint32_t total = 0; for (auto v : mism) total += v;
    const uint64_t arrivals = term % gg::kNumMaxLogicalCtas;
    printf("GATE_PROTO C=%u tb=%u zero_before=%d mismatch=0x%x arrival_field=%llu\n",
           kNumLogicalCtas, num_token_blocks, zero_before ? 1 : 0, total,
           (unsigned long long)arrivals);
    report(label, total == 0 and arrivals == kNumLogicalCtas);
    CK(cudaFree(d_term)); CK(cudaFree(d_mism));
}

// ---------------------------------------------------------------------------
// B) ScoreBarrier::wait_init 的无符号距离语义
//    block0：真实 `wait_init` 后读 flag（判断「init 是否已在它返回前发生」）
//    block1：自旋 spin_delay 后按 mode 决定是否 `init`，再置 flag=1
// ---------------------------------------------------------------------------
template <uint32_t kNumLogicalCtas>
__global__ void k_gate_stale(uint64_t* ptr, unsigned long long* flag, uint32_t* out,
                             unsigned long long spin_delay, int mode) {
    using SB = sched::mega_gate::ScoreBarrier<kNumLogicalCtas>;
    const uint64_t g = ptx::get_grid_idx();
    const uint64_t S = gg::kNumMaxLogicalCtas;
    if (threadIdx.x == 0) {
        if (mode == 0)      ptx::st_rel(ptr, g * S + kNumLogicalCtas);  // 上一轮的终止态（陈旧）
        else if (mode == 1) ptx::st_rel(ptr, (g + 1) * S);              // 本轮已 arm（arrivals=0）
        else                ptx::st_rel(ptr, (g + 1) * S + 1);          // 本轮已有 1 个到达
    }
    __syncthreads();

    if (blockIdx.x == 0 and threadIdx.x == 0) {
        SB::wait_init(ptr);
        out[0] = (uint32_t)atomicAdd(flag, 0ull);   // init 是否已经在 wait_init 返回前发生
    } else if (blockIdx.x == 1 and threadIdx.x == 0) {
        const uint64_t t0 = clock64();
        while (clock64() - t0 < spin_delay) {}
        if (mode == 0) SB::init(ptr);               // 只有陈旧态才需要（由 owner）重新 arm
        __threadfence();
        atomicExch(flag, 1ull);
    }
}

template <uint32_t kNumLogicalCtas>
static void part_b(void* barriers, int mode, const char* label, uint32_t expect_seen) {
    CK(cudaMemset(barriers, 0, (size_t)gg::kNumMaxTokenBlocks * gg::kScoreBarrierLineBytes));
    uint64_t* d_ptr = nullptr; unsigned long long* d_flag = nullptr; uint32_t* d_out = nullptr;
    CK(cudaMalloc(&d_ptr, sizeof(uint64_t)));
    CK(cudaMalloc(&d_flag, sizeof(unsigned long long)));
    CK(cudaMalloc(&d_out, sizeof(uint32_t)));
    CK(cudaMemset(d_ptr, 0, sizeof(uint64_t)));
    CK(cudaMemset(d_flag, 0, sizeof(unsigned long long)));
    CK(cudaMemset(d_out, 0, sizeof(uint32_t)));

    const unsigned long long delay = 30000000ull;   // ~15 ms @2GHz
    k_gate_stale<kNumLogicalCtas><<<2, 32>>>(d_ptr, d_flag, d_out, delay, mode);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    uint32_t seen = 0;
    CK(cudaMemcpy(&seen, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("GATE_STALE mode=%d init_seen_before_return=%u (expect %u)\n", mode, seen, expect_seen);
    report(label, seen == expect_seen);
    CK(cudaFree(d_ptr)); CK(cudaFree(d_flag)); CK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// C) SplitBarrier::get_ptr line 偏移 + 全协议
// ---------------------------------------------------------------------------
template <uint32_t kBarrierIndex>
__global__ void k_mhc_ptr(uint64_t* base, uint64_t* out_off, uint32_t num_mb) {
    using SB = sched::mega_mhc::SplitBarrier<kBarrierIndex>;
    if (threadIdx.x == 0)
        for (uint32_t mb = 0; mb < num_mb; ++ mb)
            out_off[mb] = (uint64_t)((char*)SB::get_ptr(base, mb) - (char*)base);
}

template <uint32_t kNumSplits, uint32_t kBarrierIndex>
__global__ void k_mhc_protocol(uint64_t* barriers, uint32_t num_m_blocks,
                               uint64_t* out_terminal, uint32_t* out_mismatch) {
    using SB = sched::mega_mhc::SplitBarrier<kBarrierIndex>;
    const uint32_t sm = blockIdx.x, nsms = gridDim.x;
    uint32_t mism = 0;
    // owner：每个 SM 负责一段 m_block 的 init（与 impl 的 `for mb=blockIdx.x; mb<num_m_blocks; mb+=kNumSMs` 一致）
    if (threadIdx.x == 0)
        for (uint32_t mb = sm; mb < num_m_blocks; mb += nsms) SB::init(barriers, mb);
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t num_tasks = num_m_blocks * kNumSplits;
        for (uint32_t task = sm; task < num_tasks; task += nsms) {
            const uint32_t mb = task / kNumSplits;
            SB::wait_init<kNumSplits>(barriers, mb);
            subst_arrive(SB::get_ptr(barriers, mb));   // 同上：替身 arrive
        }
        const uint64_t expect = SB::encode_state(ptx::get_grid_idx(), kNumSplits);
        for (uint32_t mb = sm; mb < num_m_blocks; mb += nsms) {
            SB::wait<kNumSplits>(barriers, mb);
            const uint64_t term = ptx::ld_acq_gpu(SB::get_ptr(barriers, mb));
            if (term != expect) ++ mism;
            if (mb == 0) out_terminal[kBarrierIndex] = term;
        }
    }
    if (threadIdx.x == 0) out_mismatch[sm] = mism;
}

template <uint32_t kNumSplits, uint32_t kBarrierIndex>
static void part_c_proto(uint64_t* barriers, uint32_t nsms, uint32_t num_m_blocks,
                         const char* label, bool zero_before) {
    if (zero_before) CK(cudaMemset(barriers, 0, (size_t)mm::kNumSplitBarriers * mm::kNumMaxMBlocks * mm::kSplitBarrierLineBytes));
    uint64_t* d_term; uint32_t* d_mism;
    CK(cudaMalloc(&d_term, mm::kNumSplitBarriers * sizeof(uint64_t)));
    CK(cudaMalloc(&d_mism, nsms * sizeof(uint32_t)));
    CK(cudaMemset(d_term, 0, mm::kNumSplitBarriers * sizeof(uint64_t)));
    CK(cudaMemset(d_mism, 0, nsms * sizeof(uint32_t)));

    for (int rep = 0; rep < 2; ++ rep) {
        k_mhc_protocol<kNumSplits, kBarrierIndex><<<nsms, 128>>>(barriers, num_m_blocks, d_term, d_mism);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }
    std::vector<uint32_t> mism(nsms);
    uint64_t term = 0;
    CK(cudaMemcpy(mism.data(), d_mism, nsms * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&term, d_term + kBarrierIndex, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint32_t total = 0; for (auto v : mism) total += v;
    const uint64_t arrivals = term % (mm::kNumSplitBarriers * mm::kNumMaxSplits);
    printf("MHC_PROTO idx=%u splits=%u sms=%u mb=%u mismatch=0x%x arrival_field=%llu\n",
           kBarrierIndex, kNumSplits, nsms, num_m_blocks, total, (unsigned long long)arrivals);
    report(label, total == 0 and arrivals == kNumSplits);
    CK(cudaFree(d_term)); CK(cudaFree(d_mism));
}

// ---------------------------------------------------------------------------
// D) host 侧算术
// ---------------------------------------------------------------------------
static void part_d() {
    // D1 gate Workspace scratch / score_ptr
    {
        const uint32_t tb = 5, splits = 4, bt = 256, e = 128;
        const uint64_t got = GateWS::get_num_scratch_bytes(tb, splits, bt, e);
        const uint64_t exp = (uint64_t)tb * splits * bt * e * sizeof(float);
        report("D1 gate scratch bytes", got == exp);
        char buf[8]; // 只借地址算偏移
        GateWS ws((void*)buf, (void*)buf);
        const float* p = ws.get_score_ptr(/*token*/3, /*split*/2);
        const int64_t off = (int64_t)(p - (float*)buf);
        report("D2 gate score_ptr offset", off == (int64_t)3 * (splits * e) + (int64_t)2 * e);
    }
    // D3 mHC Workspace m_blocks / scratch bytes / partial 指针
    {
        const uint32_t T = 1025, mb = mm::Workspace<>::get_num_m_blocks(T);
        const uint32_t exp_mb = (T + mm::BLOCK_M - 1) / mm::BLOCK_M;
        report("D3 mHC num_m_blocks", mb == exp_mb);
        const uint32_t splits = 16;
        const uint64_t got = mm::Workspace<>::get_num_scratch_bytes(mb, splits);
        const uint64_t per_task = (uint64_t)mm::kNumGemmPartialElementsPerTask + 2ull * mm::kNumSqrSumPartialElementsPerTask;
        const uint64_t exp = (uint64_t)mb * splits * per_task * sizeof(float);
        report("D4 mHC scratch bytes", got == exp);
        char buf[8];
        float* base = (float*)buf;
        mm::Workspace<> ws((void*)buf, mb);
        const auto* g  = ws.get_gemm_partial_ptr(1);
        const auto* ns = ws.get_hc_norm_sqr_sum_partial_ptr(0);
        const auto* xs = ws.get_x1_sqr_sum_partial_ptr(0);
        const int64_t total = (int64_t)mb * splits;
        report("D5 mHC gemm_partial offset", (g - base) == (int64_t)mm::kNumGemmPartialElementsPerTask);
        report("D6 mHC norm_sqr offset", (ns - base) == total * (int64_t)mm::kNumGemmPartialElementsPerTask);
        report("D7 mHC x1_sqr offset",
               (xs - base) == total * (int64_t)mm::kNumGemmPartialElementsPerTask
                              + total * (int64_t)mm::kNumSqrSumPartialElementsPerTask);
    }
    printf("GATE consts BLOCK_K=%u kScoreBarrierLineBytes=%u kNumMaxLogicalCtas=%u kNumMaxTokenBlocks=%u\n",
           gg::BLOCK_K, gg::kScoreBarrierLineBytes, gg::kNumMaxLogicalCtas, gg::kNumMaxTokenBlocks);
    printf("MHC  consts BLOCK_M=%u kNumSplitBarriers=%u kNumMaxSplits=%u kNumMaxMBlocks=%u line_bytes=%llu\n",
           mm::BLOCK_M, mm::kNumSplitBarriers, mm::kNumMaxSplits, mm::kNumMaxMBlocks,
           (unsigned long long)mm::kSplitBarrierLineBytes);
}

int main() {
    int dev = 0, nsms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&nsms, cudaDevAttrMultiProcessorCount, dev));
    printf("Device sm_count=%d\n", nsms);

    void* scratch = nullptr;
    CK(cudaMalloc(&scratch, (size_t)gg::Workspace<4, 128>::get_num_scratch_bytes(4, 4, 256, 128)));
    void* gate_bar = nullptr;
    CK(cudaMalloc(&gate_bar, (size_t)gg::kNumMaxTokenBlocks * gg::kScoreBarrierLineBytes));
    void* mhc_bar = nullptr;
    CK(cudaMalloc(&mhc_bar, (size_t)mm::kNumSplitBarriers * mm::kNumMaxMBlocks * mm::kSplitBarrierLineBytes));

    // ---- Part A ----
    printf("\n== Part A: ScoreBarrier 全协议（免复位）==\n");
    part_a<4>(scratch, gate_bar, 3, "A1 ScoreBarrier C=4 x2 launches (no reset)", true);
    part_a<8>(scratch, gate_bar, 3, "A2 ScoreBarrier C=8 x2 launches (no reset)", false);
    part_a<1>(scratch, gate_bar, 3, "A3 ScoreBarrier C=1 x2 launches (no reset)", false);

    // ---- Part B ----
    printf("\n== Part B: ScoreBarrier::wait_init 无符号距离 ==\n");
    part_b<4>(gate_bar, 0, "B1 stale terminal rejected (wait until init)", 1);
    part_b<4>(gate_bar, 1, "B2 current-grid armed accepted immediately", 0);
    part_b<4>(gate_bar, 2, "B3 current-grid partial arrival accepted", 0);

    // ---- Part C ----
    printf("\n== Part C: SplitBarrier get_ptr 偏移 + 全协议 ==\n");
    {
        const uint32_t nmb = 6;
        uint64_t* d_off; CK(cudaMalloc(&d_off, nmb * sizeof(uint64_t)));
        bool ok = true; std::vector<uint64_t> off0(nmb), off1(nmb);
        k_mhc_ptr<0><<<1, 32>>>((uint64_t*)mhc_bar, d_off, nmb); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(off0.data(), d_off, nmb * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        k_mhc_ptr<1><<<1, 32>>>((uint64_t*)mhc_bar, d_off, nmb); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(off1.data(), d_off, nmb * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (uint32_t mb = 0; mb < nmb; ++ mb) {
            const uint64_t e0 = (uint64_t)mb * mm::kSplitBarrierLineBytes;
            const uint64_t e1 = (uint64_t)(mm::kNumMaxMBlocks + mb) * mm::kSplitBarrierLineBytes;
            if (off0[mb] != e0 or off1[mb] != e1) ok = false;
        }
        printf("MHC_PTR off0=[%llu,%llu,%llu,...] off1=[%llu,%llu,%llu,...] line=%llu\n",
               (unsigned long long)off0[0], (unsigned long long)off0[1], (unsigned long long)off0[2],
               (unsigned long long)off1[0], (unsigned long long)off1[1], (unsigned long long)off1[2],
               (unsigned long long)mm::kSplitBarrierLineBytes);
        report("C1 SplitBarrier get_ptr line offsets", ok);
        CK(cudaFree(d_off));
    }
    part_c_proto<4, 0>( (uint64_t*)mhc_bar, 8, 5, "C2 Norm(idx0) splits=4 x2 (no reset)", true);
    part_c_proto<4, 1>( (uint64_t*)mhc_bar, 8, 5, "C3 Mix (idx1) splits=4 x2 (no reset)", false);
    part_c_proto<16, 0>((uint64_t*)mhc_bar, 8, 5, "C4 Norm(idx0) splits=16 x2 (no reset)", false);
    part_c_proto<16, 1>((uint64_t*)mhc_bar, 8, 5, "C5 Mix (idx1) splits=16 x2 (no reset)", false);

    // ---- Part D ----
    printf("\n== Part D: Workspace host 算术 ==\n");
    part_d();

    CK(cudaFree(scratch)); CK(cudaFree(gate_bar)); CK(cudaFree(mhc_bar));
    printf("\n%s (%d FAIL)\n", g_fail == 0 ? "ALL PASS" : "HAS FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
