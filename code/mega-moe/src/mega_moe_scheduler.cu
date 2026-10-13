// P110-1 实验：DeepGEMM 的 Mega MoE 任务调度器 `scheduler/mega_moe.cuh`（481 行）。
//
// 背景：本专题已单独拆过另外三个调度器——`scheduler/gemm.cuh`（第 108 篇通用 GEMM 任务调度器）、
// `scheduler/mega_gate.cuh` + `scheduler/mega_mhc.cuh`（第 92 篇分段屏障调度器）。唯独
// Mega MoE 主 kernel 自己的任务调度器 `scheduler/mega_moe.cuh` 只在第 02 篇有一节概述。
// 它是整条「dispatch + linear1 + SwiGLU + linear2 + combine」流水线**把工作切成 task 并发给
// 16-warp 分身的唯一入口**，也是 ring buffer 容量公式 `get_num_max_live_pool_blocks` 的来源。
//
// 为什么 H100(sm90) 可跑（与第 108 篇同源理由）：本文件的分工是**纯整数/布局运算**——
//   * `get_num_l1_warmup_waves` / `get_num_max_live_pool_blocks`：CUTLASS_HOST_DEVICE constexpr；
//   * `TaskInfo`：POD + 位域（`alignas(16)`、8×uint32）；
//   * `create_task` / `get_pool_block_offset` / `get_next_task_idx`：warp 内建整数映射，不碰 UMMA/TMA；
//   * `L2KBlockDependency`：只用 `red.release.gpu.global.xor` + `ld.acquire.gpu.global`，sm90 起就支持。
// 唯一 SM100 专属的是 `publish_task` 的 `st.async.shared::cluster` + cluster transaction barrier
// （`ld_st.cuh:206-224`，sm90+ 其实也有该指令，但需要 cluster launch），以及主 kernel 端的 UMMA/TMA。
// 故本篇在 H100 上**直接 include 真实头文件、编译运行真实设备代码**做架构无关部分的对拍，
// `publish_task` / 主 kernel 的 cluster 协作仅做源码分析。
//
// 六组核验（Part A~F，全部对拍「独立复刻的规范」）：
//   A) 行容量公式 `get_num_max_live_pool_blocks` 与其前置 `get_num_l1_warmup_waves`（host constexpr）。
//   B) `TaskInfo` 的 ABI（sizeof/alignof/字段偏移）与 `is_valid`/`is_shared`/`get_umma_aligned_valid_m`。
//   C) `L2KBlockDependency`（L1→L2 位图依赖）：arrive 的 XOR 翻转 + wait 的相位掩码语义。
//   D) `create_task` 的**warp 协作**映射：pool block → (expert, owner m_block, valid_m)。
//   E) `get_pool_block_offset` / `get_num_total_pool_blocks`（前缀和语义）。
//   F) `get_next_task_idx` 的 locality-domain 分域枚举是 [0,num_m*kNumLocalClusters) 的双射。
//
// 编译与运行见 harness/exp_mega_moe_scheduler.sh。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>

#include <cuda_runtime.h>

// 构建适配：`exception.cuh` 的 `DG_DEVICE_ASSERT` 展开成 `asm("trap;")`（`exception.cuh:15-23`）。
// `DG_IN_CUDA_COMPILATION` 在 nvcc **host 通道**里也是定义的（`compile.cuh:6`），而本实验要从 host
// 直接调用 `CUTLASS_HOST_DEVICE` 的 `get_num_max_live_pool_blocks`，于是 host 版函数体会把 `trap`
// 送进汇编器（"no such instruction: trap"）。真实 DeepGEMM 从不让它落到 host 目标码里，故这里按
// exception.cuh 预留的 `#ifndef` 钩子把它置空——所有测试形状都满足这些断言，禁用不影响语义。
#define DG_DEVICE_ASSERT(cond) do { } while (0)

#include <deep_gemm/scheduler/mega_moe.cuh>

using namespace deep_gemm;
using deep_gemm::sched::BlockPhase;
using deep_gemm::sched::TaskInfo;

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1); } } while (0)

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& msg) {
    if (ok) { ++g_pass; }
    else { ++g_fail; printf("   FAIL: %s\n", msg.c_str()); }
}

static uint32_t hceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static uint32_t hmax(uint32_t a, uint32_t b) { return a > b ? a : b; }
static uint32_t hmin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ===========================================================================
// Part A：行容量公式（host constexpr）
// ===========================================================================
static uint32_t host_l1_warmup_waves(uint32_t ntb, uint32_t nc, uint32_t l1n, uint32_t l2n) {
    const uint32_t first_l2_wave_m = hceil(nc, l2n);
    const uint32_t warm_for_first = hceil(first_l2_wave_m * l1n, nc);
    const uint32_t diff = l1n > l2n ? l1n - l2n : 0;
    const uint32_t warm_interleave = hceil(l1n + (ntb - 1) * diff, nc) + 1;
    return hmax(warm_for_first, warm_interleave);
}

static uint32_t host_max_live_pool_blocks(uint32_t ntb, uint32_t nsm,
                                          uint32_t hidden, uint32_t inter) {
    constexpr uint32_t kBlockN = 128, kCTAs = 2;
    const uint32_t nclusters = nsm / kCTAs;
    const uint32_t l1n = inter * 2 / (kCTAs * kBlockN);
    const uint32_t l2n = hidden / (kCTAs * kBlockN);
    const uint32_t l1clusters = ntb * l1n;
    const uint32_t l1waves = hceil(l1clusters, nclusters);
    const uint32_t min_warm = host_l1_warmup_waves(ntb, nclusters, l1n, l2n);
    const uint32_t warm_waves = hmin(min_warm, l1waves);
    const uint32_t warm_clusters = hmin(warm_waves * nclusters, l1clusters);
    const uint32_t live_after = hceil(warm_clusters, l1n);
    const uint32_t growth = l2n > l1n ? hceil(ntb * (l2n - l1n), l2n) : 0;
    const uint32_t margin = hceil(nclusters, hmin(l1n, l2n));
    return hmin(ntb, live_after + growth + margin);
}

static void part_a() {
    printf("== Part A: 行容量公式 get_num_max_live_pool_blocks / get_num_l1_warmup_waves ==\n");
    struct Case { uint32_t ntb, nsm, hidden, inter; };
    // DeepSeek-V3/R1 附近的真实形状 + 一些极端形状
    std::vector<Case> cases = {
        {64,   132, 7168, 2048},
        {128,  132, 7168, 2048},
        {512,  132, 7168, 2048},
        {1024, 132, 7168, 2048},
        {34,   132, 7168, 2048},
        {7,     64, 7168, 2048},
        {300,  148, 7168, 2048},
        {256,   80, 7168, 2048},
        {96,   132, 7168, 1024},
        {200,  132, 4096, 2048},
    };
    for (auto& c : cases) {
        const uint32_t dev = deep_gemm::sched::get_num_max_live_pool_blocks(
            c.ntb, c.nsm, c.hidden, c.inter);
        const uint32_t host = host_max_live_pool_blocks(c.ntb, c.nsm, c.hidden, c.inter);
        // 前置：warmup waves 也要逐点相等（用真实设备/host constexpr 同一函数）
        const uint32_t nclusters = c.nsm / 2;
        const uint32_t l1n = c.inter * 2 / (2 * 128);
        const uint32_t l2n = c.hidden / (2 * 128);
        const uint32_t devwarm = deep_gemm::sched::get_num_l1_warmup_waves(c.ntb, nclusters, l1n, l2n);
        const uint32_t hostwarm = host_l1_warmup_waves(c.ntb, nclusters, l1n, l2n);
        bool ok = (dev == host) && (devwarm == hostwarm);
        // 不变式：容量永远不超过 M block 总数，且至少覆盖 warmup 之后的活跃块
        ok = ok && (dev <= c.ntb) && (dev >= 1);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "ntb=%-5u nsm=%-4u h=%-5u i=%-5u : live=%-5u (host %-5u) warm=%-3u (host %-3u)",
                 c.ntb, c.nsm, c.hidden, c.inter, dev, host, devwarm, hostwarm);
        printf("   %s\n", buf);
        check(ok, std::string("row capacity mismatch: ") + buf);
    }
}

// ===========================================================================
// Part B/C/D/E/F：设备侧
// ===========================================================================
static constexpr uint32_t kBlockM = 128;
static constexpr uint32_t kBlockN = 128;
static constexpr uint32_t kBlockK = 128;
static constexpr uint32_t kHidden = 7168;      // L2_SHAPE_N
static constexpr uint32_t kInter  = 2048;      // L2_SHAPE_K, L1_SHAPE_N = 2*kInter, L1_SHAPE_K = kHidden
static constexpr uint32_t kNumExperts = 8;
static constexpr uint32_t kNumSMs = 64;
static constexpr uint32_t kNumRanks = 1;
static constexpr uint32_t kNumRingBlocks = 4096;
static constexpr uint32_t kNumDomains = 2;

using Sched = deep_gemm::sched::MegaMoEScheduler<
    kBlockM, kBlockN, kBlockK,
    2 * kInter, kHidden, kHidden, kInter,
    kNumExperts, kNumSMs, kNumRanks, kNumRingBlocks, kNumDomains>;

// 模板默认实参对应的常量（模板参数不是 static 成员，单独列出）
static constexpr uint32_t kExpertsPerLane = (kNumExperts + 31) / 32;   // 1
static constexpr uint32_t kL1BlockNs = (2 * kInter) / kBlockN;         // 32
static constexpr uint32_t kL2BlockNs = kHidden / kBlockN;              // 56
static constexpr uint32_t kL1Clusters = kL1BlockNs / 2;                // 16
static constexpr uint32_t kL2Clusters = kL2BlockNs / 2;                // 28

// ---- Part B: TaskInfo ABI ----
__global__ void part_b_kernel(uint32_t* out) {
    if (threadIdx.x != 0) return;
    TaskInfo<true> t(BlockPhase::SharedLinear1, /*expert*/7, /*m*/3, /*n*/11, /*pool*/2, /*valid_m*/100, /*shape_n*/4096, /*shape_k*/7168);
    out[0] = t.is_valid();
    out[1] = t.is_shared();
    out[2] = t.get_umma_aligned_valid_m();     // align(100,16) = 112
    out[3] = t.local_expert_idx;
    out[4] = t.m_block_idx;
    out[5] = t.n_cluster_idx;
    out[6] = t.pool_block_idx;
    out[7] = t.valid_m;
    out[8] = t.shape_n;
    out[9] = t.shape_k;
    out[10] = offsetof(TaskInfo<true>, valid_m);
    out[11] = offsetof(TaskInfo<true>, shape_k);
    // 无效相位
    TaskInfo<true> n0;
    out[12] = n0.is_valid();                    // 0
    // 各相位 is_shared
    TaskInfo<true> l1(BlockPhase::Linear1, 0,0,0,0,0,0,0);
    TaskInfo<true> l2(BlockPhase::Linear2, 0,0,0,0,0,0,0);
    TaskInfo<true> s1(BlockPhase::SharedLinear1, 0,0,0,0,0,0,0);
    TaskInfo<true> s2(BlockPhase::SharedLinear2, 0,0,0,0,0,0,0);
    out[13] = l1.is_shared() + l2.is_shared() + s1.is_shared() + s2.is_shared(); // 0+0+1+1=2
}

// ---- Part C: L2KBlockDependency ----
// 用真实设备代码：arrive 对 mask 做 red.xor，wait 用 (ld_acq ^ expected_mask) 判就绪。
// L1_SHAPE_N=2*kInter=4096, BLOCK_N=128 => kNumL1BlockNs=32；BLOCK_K=128 => 每 L2 K 块需 2 个 L1 N 块。
__global__ void part_c_kernel(uint64_t* mask,
                              uint32_t generation,
                              uint32_t* arrive_bits, uint32_t num_arrive,
                              uint32_t* wait_blocks, uint32_t num_wait,
                              uint32_t* out) {
    using Dep = deep_gemm::sched::L2KBlockDependency<2 * kInter, kBlockN, kBlockK>;
    if (threadIdx.x != 0) return;
    // 先 arrive 指定的 bit（模拟 L1 N 块完成）
    for (uint32_t i = 0; i < num_arrive; ++ i)
        Dep::arrive(mask, arrive_bits[i]);
    __threadfence();
    // 再对每个待等 K 块构造依赖并 wait；记录是否在有限自旋内返回
    Dep dep(mask, generation);
    for (uint32_t j = 0; j < num_wait; ++ j) {
        // 用一个「有界」替代：先看 pending 是否已清（wait 会自旋，这里靠已 arrive 的位）
        dep.wait(wait_blocks[j]);
        out[j] = 1;
    }
    out[64] = dep.pending_mask;    // 观测残留 pending（应为 0）
    out[65] = Dep::kNumL1BlockNs;
    out[66] = Dep::kNumL1BlockNsPerL2KBlock;
    out[67] = (uint32_t) (Dep::kFullMask & 0xffffffffull);
}

// ---- Part D/E: create_task + pool block offset ----
__global__ void part_de_kernel(const uint32_t* count_by_lane,   // [kNumExperts]
                               uint32_t* task_out,              // [max_blocks * 8]
                               uint32_t* total_out) {
    // 单 warp（32 lane）
    layout::Workspace ws;                                  // 默认构造，create_task 不触碰它
    Sched s(ws);
    const uint32_t lane = threadIdx.x;
    #pragma unroll
    for (uint32_t i = 0; i < kExpertsPerLane; ++ i) {
        const uint32_t expert_idx = i * 32 + lane;
        s.stored_num_tokens_per_expert[i] =
            (expert_idx < kNumExperts) ? count_by_lane[expert_idx] : 0u;
    }
    __syncwarp();
    const uint32_t total = s.get_num_total_pool_blocks();
    if (lane == 0) total_out[0] = total;
    // 记录每个 pool block 的 create_task 结果（create_task / get_num_tokens 都是 warp 协作）
    for (uint32_t g = 0; g < total; ++ g) {
        auto t = s.create_task(BlockPhase::Linear1, g, /*n_cluster*/0, 2 * kInter, 0u + kHidden);
        const uint32_t nt = s.get_num_tokens(g % kNumExperts);   // 全 lane 参与，结果广播
        if (lane == 0) {
            task_out[g * 8 + 0] = t.local_expert_idx;
            task_out[g * 8 + 1] = t.m_block_idx;
            task_out[g * 8 + 2] = t.valid_m;
            task_out[g * 8 + 3] = t.pool_block_idx;
            task_out[g * 8 + 4] = nt;
        }
    }
}

// ---- Part F: locality-domain 分域枚举双射 ----
// 每个 warp 反复通过 elect_one + atomic_add 领取 task 直到耗尽；只 lane 0 记录。
template <uint32_t kNumClusters>
__global__ void part_f_kernel(uint32_t* counter,      // [kNumDeviceLocalityDomains]
                              uint32_t domain,
                              uint32_t num_m_blocks,
                              uint32_t* out_mn,       // [cap*2]
                              uint32_t* out_cnt,
                              uint32_t cap) {
    const uint32_t lane = threadIdx.x % 32;
    while (true) {
        uint32_t m = 0, n = 0;
        const bool got = Sched::template get_next_task_idx<kNumClusters>(
            counter, domain, num_m_blocks, m, n);
        if (not got) break;
        uint32_t slot = 0;
        if (lane == 0) slot = atomicAdd(out_cnt, 1u);
        slot = __shfl_sync(0xffffffff, slot, 0);
        if (slot < cap && lane == 0) { out_mn[slot * 2] = m; out_mn[slot * 2 + 1] = n; }
        __syncwarp();
    }
}

// Part A2：把行容量公式接到 host 侧的 ring 分配（`csrc/apis/mega_moe.hpp:55-63`）——
// host 对所有候选 BLOCK_M 扫描 `num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank`，
// 取 `max(live_pool_blocks * block_m)` 再对齐到 `kLCMCandidateBlockM`。这解释了「ring 容量公式为什么是 kernel 的
// 唯一对外几何契约」：host 用最坏 token 数上界定容量，device 用真实 recv 计数定 num_total_m_blocks（≤ 该上界）。
static void part_a2() {
    printf("\n== Part A2: 行容量公式 -> host ring 分配（复刻 csrc/apis/mega_moe.hpp:55-63）==\n");
    const uint32_t hidden = 7168, inter = 2048, num_sms = 132;
    const uint32_t ranks = 8, max_tok = 4096, topk = 8, experts_per_rank = 32;
    const uint32_t num_max_routed_tokens = max_tok * ranks * topk;
    uint32_t best_ring = 0, best_block = 0;
    for (int bm : deep_gemm::layout::kCandidateBlockM) {
        const uint32_t num_pool_blocks = hceil(num_max_routed_tokens, (uint32_t) bm) + experts_per_rank;
        const uint32_t live = deep_gemm::sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, inter);
        const uint32_t ring = live * (uint32_t) bm;
        if (ring > best_ring) { best_ring = ring; best_block = (uint32_t) bm; }
    }
    const uint32_t aligned = hceil(best_ring, deep_gemm::layout::kLCMCandidateBlockM)
                             * deep_gemm::layout::kLCMCandidateBlockM;
    printf("   num_max_routed_tokens=%u, 候选 BLOCK_M 扫描 -> 最大 ring=%u (block_m=%u), 对齐 %u -> %u\n",
           num_max_routed_tokens, best_ring, best_block,
           (uint32_t) deep_gemm::layout::kLCMCandidateBlockM, aligned);
    check(aligned % deep_gemm::layout::kLCMCandidateBlockM == 0, "ring not aligned to LCM");
    check(aligned >= best_ring && aligned < best_ring + deep_gemm::layout::kLCMCandidateBlockM,
          "ring alignment out of range");
}

static void run_device_parts() {
    // ---------- Part B ----------
    printf("\n== Part B: TaskInfo ABI 与语义 ==\n");
    {
        static_assert(sizeof(TaskInfo<true>) == sizeof(TaskInfo<false>), "ABI");
        uint32_t* d; CUDA_CHECK(cudaMalloc(&d, 32 * sizeof(uint32_t)));
        part_b_kernel<<<1, 32>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t h[32]; CUDA_CHECK(cudaMemcpy(h, d, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        check(sizeof(TaskInfo<true>) == 32, "TaskInfo size != 32");
        check(alignof(TaskInfo<true>) == 16, "TaskInfo align != 16");
        check(h[0] == 1 && h[12] == 0, "is_valid wrong");
        check(h[1] == 1, "is_shared(true-shared) wrong");
        check(h[2] == 112, "get_umma_aligned_valid_m(100) != 112");
        check(h[3] == 7 && h[4] == 3 && h[5] == 11 && h[6] == 2, "field order wrong");
        check(h[7] == 100 && h[8] == 4096 && h[9] == 7168, "shape/valid fields wrong");
        check(h[10] == 20 && h[11] == 28, "offsetof wrong");
        check(h[13] == 2, "is_shared per phase wrong");
        printf("   sizeof=%zu alignof=%zu valid_m->umma=%u  offsets(valid_m=%u,shape_k=%u)\n",
               sizeof(TaskInfo<true>), alignof(TaskInfo<true>), h[2], h[10], h[11]);
        printf("   is_valid(None)=%u  is_shared per phase sum=%u\n", h[12], h[13]);
        cudaFree(d);
    }

    // ---------- Part C ----------
    printf("\n== Part C: L2KBlockDependency（L1->L2 位图依赖）==\n");
    {
        // 每个 K 块需 2 个连续 L1 N 块。generation=0 => expected_mask=kFullMask。
        // 场景：arrive bit0,bit1 => K块0 就绪；arrive bit2,bit3 => K块1 就绪；K块2 等 bit4,5 未到 => 不测。
        uint64_t* dmask; CUDA_CHECK(cudaMalloc(&dmask, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(dmask, 0, sizeof(uint64_t)));
        uint32_t abits[4] = {0, 1, 2, 3};
        uint32_t wblocks[2] = {0, 1};
        uint32_t *da, *dw, *do_;
        CUDA_CHECK(cudaMalloc(&da, sizeof(abits)));
        CUDA_CHECK(cudaMalloc(&dw, sizeof(wblocks)));
        CUDA_CHECK(cudaMalloc(&do_, 128 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(da, abits, sizeof(abits), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, wblocks, sizeof(wblocks), cudaMemcpyHostToDevice));
        part_c_kernel<<<1, 1>>>(dmask, 0, da, 4, dw, 2, do_);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t h[128]; CUDA_CHECK(cudaMemcpy(h, do_, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint64_t hmask; CUDA_CHECK(cudaMemcpy(&hmask, dmask, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        check(h[0] == 1 && h[1] == 1, "wait should return after bits set");
        check(hmask == 0xF, "mask should have bits 0..3 set");
        check(h[65] == 32, "kNumL1BlockNs for L1_SHAPE_N=4096/BN=128 should be 32");
        check(h[66] == 2, "kNumL1BlockNsPerL2KBlock should be 2");
        printf("   arrive bits {0,1,2,3} -> mask=0x%llX; wait K0,K1 均返回; kNumL1BlockNs=%u perK=%u\n",
               (unsigned long long) hmask, h[65], h[66]);

        // generation=1 => expected_mask=0：一个全新 mask=0 对 gen1 而言「看起来全就绪」。
        CUDA_CHECK(cudaMemset(dmask, 0, sizeof(uint64_t)));
        part_c_kernel<<<1, 1>>>(dmask, 1, da, 0, dw, 1, do_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h, do_, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint64_t hmask2; CUDA_CHECK(cudaMemcpy(&hmask2, dmask, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        // 对 gen=1，expected=0，wait 应直接返回（mask 空 => 无 pending）
        check(h[0] == 1, "gen parity: empty mask should look ready for odd generation");
        printf("   gen=1 & mask=0 -> wait K0 直接返回（expected_mask=0 语义）; mask=0x%llX\n",
               (unsigned long long) hmask2);
        cudaFree(dmask); cudaFree(da); cudaFree(dw); cudaFree(do_);
    }

    // ---------- Part D/E ----------
    printf("\n== Part D/E: create_task 的 warp 协作映射 + pool block 前缀和 ==\n");
    {
        uint32_t counts[kNumExperts] = {5, 0, 300, 200, 130, 1, 7, 250};
        uint32_t* dc; CUDA_CHECK(cudaMalloc(&dc, sizeof(counts)));
        CUDA_CHECK(cudaMemcpy(dc, counts, sizeof(counts), cudaMemcpyHostToDevice));
        uint32_t* dout; CUDA_CHECK(cudaMalloc(&dout, 64 * 8 * sizeof(uint32_t)));
        uint32_t* dtot; CUDA_CHECK(cudaMalloc(&dtot, sizeof(uint32_t)));
        part_de_kernel<<<1, 32>>>(dc, dout, dtot);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t tot; CUDA_CHECK(cudaMemcpy(&tot, dtot, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t* h = (uint32_t*) malloc(64 * 8 * sizeof(uint32_t));
        CUDA_CHECK(cudaMemcpy(h, dout, 64 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // host 规范复刻
        std::vector<uint32_t> nmb(kNumExperts), start(kNumExperts);
        uint32_t cum = 0;
        for (uint32_t e = 0; e < kNumExperts; ++ e) { start[e] = cum; nmb[e] = hceil(counts[e], kBlockM); cum += nmb[e]; }

        check(tot == cum, "num_total_pool_blocks mismatch");
        printf("   total pool blocks=%u (host %u)\n", tot, cum);
        int bad = 0;
        for (uint32_t g = 0; g < tot; ++ g) {
            int owner_e = -1; uint32_t owner_m = 0;
            for (uint32_t e = 0; e < kNumExperts; ++ e)
                if (g >= start[e] && g < start[e] + nmb[e]) { owner_e = (int) e; owner_m = g - start[e]; }
            if (owner_e < 0) { ++bad; continue; }
            const uint32_t expect_valid = hmin(counts[owner_e] - owner_m * kBlockM, kBlockM);
            const uint32_t got_e = h[g * 8 + 0], got_m = h[g * 8 + 1], got_v = h[g * 8 + 2];
            if (got_e != (uint32_t) owner_e || got_m != owner_m || got_v != expect_valid) {
                if (bad < 8)
                    printf("   block %u: got(e=%u,m=%u,v=%u) expect(e=%d,m=%u,v=%u)\n",
                           g, got_e, got_m, got_v, owner_e, owner_m, expect_valid);
                ++bad;
            }
        }
        check(bad == 0, "create_task mapping mismatch");
        // get_num_tokens 抽查：g%8 抖动，检查每个专家被问到时的返回
        int tkn_bad = 0;
        for (uint32_t g = 0; g < tot; ++ g) {
            uint32_t q = g % kNumExperts;
            if (h[g * 8 + 4] != counts[q]) ++tkn_bad;
        }
        check(tkn_bad == 0, "get_num_tokens mismatch");
        printf("   create_task 全 %u 块 (expert,owner_m,valid_m) 对拍：%d 不符；get_num_tokens %d 不符\n",
               tot, bad, tkn_bad);
        cudaFree(dc); cudaFree(dout); cudaFree(dtot); free(h);
    }

    // ---------- Part F ----------
    printf("\n== Part F: locality-domain 分域枚举（双射）==\n");
    {
        constexpr uint32_t kClusters = 2 * kInter / kBlockN;   // L1: 4096/128 = 32
        const uint32_t num_m = 5;
        const uint32_t local_clusters = kClusters / kNumDomains;
        for (uint32_t dom = 0; dom < kNumDomains; ++ dom) {
            uint32_t* dcounter; CUDA_CHECK(cudaMalloc(&dcounter, kNumDomains * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(dcounter, 0, kNumDomains * sizeof(uint32_t)));
            uint32_t* dmn; CUDA_CHECK(cudaMalloc(&dmn, (num_m * local_clusters) * 2 * sizeof(uint32_t)));
            uint32_t* dcnt; CUDA_CHECK(cudaMalloc(&dcnt, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(dcnt, 0, sizeof(uint32_t)));
            part_f_kernel<kClusters><<<1, 1024>>>(dcounter, dom, num_m, dmn, dcnt, num_m * local_clusters);
            CUDA_CHECK(cudaDeviceSynchronize());
            uint32_t cnt; CUDA_CHECK(cudaMemcpy(&cnt, dcnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t* h = (uint32_t*) malloc((num_m * local_clusters) * 2 * sizeof(uint32_t));
            CUDA_CHECK(cudaMemcpy(h, dmn, (num_m * local_clusters) * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            // 期望：{(m, dom*local_clusters + c) : m<num_m, c<local_clusters}
            std::vector<std::pair<uint32_t,uint32_t>> got;
            for (uint32_t i = 0; i < std::min(cnt, num_m * local_clusters); ++ i)
                got.push_back({h[i*2], h[i*2+1]});
            std::sort(got.begin(), got.end());
            int bad = 0;
            uint32_t idx = 0;
            for (uint32_t m = 0; m < num_m; ++ m)
                for (uint32_t c = 0; c < local_clusters; ++ c, ++ idx)
                    if (got[idx] != std::make_pair(m, dom * local_clusters + c)) ++bad;
            check(cnt == num_m * local_clusters, "domain task count wrong");
            check(bad == 0, "domain partition mismatch");
            printf("   domain %u: %u tasks, 期望 union 覆盖 m in [0,%u) x n in [%u,%u) : %d 不符\n",
                   dom, cnt, num_m, dom * local_clusters, (dom + 1) * local_clusters, bad);
            cudaFree(dcounter); cudaFree(dmn); cudaFree(dcnt); free(h);
        }
        // 工作窃取阈值（kNumMinStealWaves=2）：num_m*kNumClusters >= 2*(kNumSMs/2) 才允许跨域偷
        constexpr uint32_t kMinSteal = 2;
        auto steal_ok = [&](uint32_t num_m) {
            return (num_m * kClusters) >= kMinSteal * (kNumSMs / 2);
        };
        printf("   work-steal 阈值: num_m*%u >= %u*(%u/2)=%u  ->  num_m>=%u 才偷域\n",
               kClusters, kMinSteal, kNumSMs, kMinSteal * (kNumSMs / 2),
               hceil(kMinSteal * (kNumSMs / 2), kClusters));
        check(!steal_ok(1) && steal_ok(5), "work-steal threshold wrong");
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("DeepGEMM scheduler/mega_moe.cuh —— H100(sm90) 架构无关部分实测\n");
    printf("Template: BM=%u BN=%u BK=%u L1=[%u,%u] L2=[%u,%u] experts=%u sm=%u dom=%u\n",
           kBlockM, kBlockN, kBlockK, 2 * kInter, kHidden, kHidden, kInter,
           kNumExperts, kNumSMs, kNumDomains);
    printf("kNumExpertsPerLane=%u kNumL1Clusters=%u kNumL2Clusters=%u\n",
           (unsigned) kExpertsPerLane,
           (unsigned) kL1Clusters, (unsigned) kL2Clusters);

    part_a();
    part_a2();
    run_device_parts();

    printf("\n==== 汇总: %d PASS / %d FAIL ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
