// P53-1 实验设备代码：DeepEP 的多 rank 屏障与 symmetric-memory 句柄层
//   （deep_ep/include/deep_ep/comm/barrier.cuh + comm/handle.cuh + layout/common/barrier.cuh）。
//
// 为什么 H100(sm90) 可跑：
//   * 整条 dispatch / combine / bucket collective 的“开始/结束对齐”都靠 comm/barrier.cuh 的
//     `gpu_barrier` 家族。它分两条腿：**scaleout 腿走 NCCL GIN（RDMA）**，**scaleup 腿走 NVLink**。
//     NVLink 腿 `nvlink_barrier_wo_local_sync` 本身只用 `ptx::red_add_rel_sys` /
//     `ptx::ld_acquire_sys` / `atomicAdd`（全是架构无关内联 asm），唯一与 NCCL 相关的是
//     `NCCLGin::get_sym_ptr`——一个**纯地址翻译**（把本地 offset 映射到目标 rank 的对称窗口）。
//   * 因此把 `timeout_while`（barrier.cuh:30-54）与 `nvlink_barrier_wo_local_sync`
//     （barrier.cuh:88-131）**逐行原样抽取**（见下方 VERBATIM 标注），只把 `NCCLGin` 换成
//     一个语义等价的**对称指针桩**（`get_sym_ptr(ptr, i) = rank_signals[i] + (ptr - local)`，
//     与 DeepGEMM `SymBuffer`/DeepEP `ncclGetLsaPointer` 的语义一致）。
//   * `get_qp_mode`（barrier.cuh:56-86）是纯 constexpr 的 SM/QP 映射，直接函数的 host/device 双态调用。
//   * 真正的 `gpu_barrier` / `gin_barrier_wo_local_sync` 用 `ncclGin` / `ncclDevComm`，本机
//     NCCL 2.29.7 头不提供 GIN 设备 API → 仅源码分析，不在本文件编译。
//
// 编译/运行见 harness/exp_deepep_comm_barrier.sh。

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/common/barrier.cuh>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ncclTeamTagLsa 在真实代码里来自 <nccl_device.h>；此处只作为模板标签，给个空壳。
struct ncclTeamTagLsa {};

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

// 记录设备结果，供 Python oracle 独立复核。
static std::string g_dump;
static void dump(const std::string& line) { g_dump += line; g_dump += "\n"; }

// =============================================================================
// 对称指针桩：只提供 nvlink_barrier_wo_local_sync 需要的 get_sym_ptr。
//   真实实现（handle.cuh:64-92）走 ncclGetLsaPointer；这里用“同一 offset 映射到目标 rank
//   signals 基址”的等价映射——这正是对称窗口的语义。
// =============================================================================
namespace deep_ep::comm {
struct NCCLGin {
    deep_ep::layout::BarrierSignals* rank_signals;   // 所有 rank 的 signals（device）
    deep_ep::layout::BarrierSignals* local;          // 本 rank 的 signals（device）

    template <typename team_t, typename dtype_t = void*>
    __device__ __forceinline__
    dtype_t* get_sym_ptr(dtype_t* ptr, const int& dst_rank_idx) const {
        const auto offset = reinterpret_cast<uint8_t*>(ptr) - reinterpret_cast<uint8_t*>(local);
        return reinterpret_cast<dtype_t*>(
            reinterpret_cast<uint8_t*>(&rank_signals[dst_rank_idx]) + offset);
    }
};
}  // namespace deep_ep::comm

// ===================== VERBATIM: comm/barrier.cuh:13 =====================
namespace deep_ep::comm {
static constexpr int64_t kNumOneSecCycles = 2000000000;  // An approximation of the GPU clock at 2000 MHz
// ===================== VERBATIM: comm/barrier.cuh:30-49 =====================
template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {
    // User may share a start clock for multiple waits
    if (start_clock == 0)
        start_clock = clock64();

    while (condition) {
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;
        if (func(timeout))
            break;

        if (timeout) {
            // Wait another 1 second to let all threads print information and trap
            start_clock = clock64();
            while (clock64() - start_clock < kNumOneSecCycles) {}
            ptx::trap();
        }
    }
}
// ===================== VERBATIM: comm/barrier.cuh:51-54 =====================
template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);
}
// ===================== VERBATIM: comm/barrier.cuh:88-131 =====================
template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles, int kTag = 0>
__forceinline__ __device__ void nvlink_barrier_wo_local_sync(
    const NCCLGin& gin,
    layout::BarrierSignals& signals,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    // This barrier only uses 1 SM
    if (kNumSMs > 1 and sm_idx > 0)
        return;

    // Read the current barrier phase first
    const auto counter_ptr = signals.get_nvl_barrier_counter_ptr();
    const int status = static_cast<int>((*counter_ptr) & 3);
    const int phase = status & 1, sign = status >> 1;
    const auto signal_ptr = signals.get_nvl_barrier_signal_ptr(phase);

    for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
        const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(signal_ptr, i);
        ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
    }
    __syncthreads();

    // NOTES: we need `2^64 / 1e6 / 3600 / 24 / 365 = 571000` years to make the counter overflow (1 barrier per us)
    // Add the phase counter
    if (thread_idx == 0)
        atomicAdd(counter_ptr, 1);
    __syncwarp();

    // Check timeout
    const auto target = sign ? 0 : kNumRanks;
    timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int>(signal_ptr);
        if (signal == target)
            return true;

        if (is_last_check) {
            printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, "
                   "status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n",
                   kTag, rank_idx, thread_idx, status, signal, phase, target,
                   *counter_ptr);
        }
        return false;
    });
    __syncwarp();
}
}  // namespace deep_ep::comm

// ===================== VERBATIM: comm/barrier.cuh:56-86 =====================
// 把 ncclGinResourceSharingMode 换成 int：0=CTA，1=GPU（与原常量一一对应）。
namespace deep_ep::comm {
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __host__ __forceinline__ int2 get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp) {
    constexpr int kSharingCTA = 0;
    constexpr int kSharingGrid = kNumSMs == 1 ? 0 : 1;

    // Only one QP
    if constexpr (kNumQPs == 1)
        return make_int2(0, kSharingGrid);

    // The notify warp always use 1 SM and 1 QP
    if (is_notify_warp)
        return make_int2(0, kSharingCTA);

    // Data channels
    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // A single SM uses an entire QP
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        return make_int2(kQPStartIdx + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, kSharingCTA);
    } else {
        // All SMs share all QPs
        const auto global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return make_int2(kQPStartIdx + (global_channel_idx % kNumAvailableQPs), kSharingGrid);
    }
}
}  // namespace deep_ep::comm

// =============================================================================
// Part 1: NVLink barrier 协议——用多 block 模拟多 rank，在一个 GPU 上跑真实设备代码。
// =============================================================================

namespace sim {
using deep_ep::layout::BarrierSignals;

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kTimeout, int kTag>
__global__ void __launch_bounds__(kNumThreads, 1)
k_nvlink_barrier(BarrierSignals* per_rank_signals, const int num_iters) {
    const int rank_idx = blockIdx.x / kNumSMs;
    const int sm_idx   = blockIdx.x % kNumSMs;
    const int thread_idx = threadIdx.x;
    deep_ep::comm::NCCLGin gin;
    gin.rank_signals = per_rank_signals;
    gin.local = &per_rank_signals[rank_idx];
    auto& signals = per_rank_signals[rank_idx];
    for (int it = 0; it < num_iters; ++ it) {
        __syncthreads();   // == gpu_barrier 的 kSyncAtStart
        deep_ep::comm::nvlink_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kTimeout, kTag>(
            gin, signals, rank_idx, sm_idx, thread_idx);
        __syncthreads();   // == gpu_barrier 的 kSyncAtEnd
    }
}

// host 侧期望模型：第 it 次调用看到 counter==it（4 相位循环）→ status=it&3、phase=status&1、sign=status>>1；
// 对 phase 槽做 ±1（±kNumRanks 累计后回到 0 或 kNumRanks），target=sign?0:kNumRanks。
struct Expected {
    int counter;
    long long slot[2];
};
static std::vector<Expected> model(const int kNumRanks, const int num_iters) {
    std::vector<Expected> out(num_iters + 1);
    int counter = 0; long long s0 = 0, s1 = 0;
    out[0] = {0, {0, 0}};
    for (int it = 0; it < num_iters; ++ it) {
        const int status = counter & 3, phase = status & 1, sign = status >> 1;
        long long* slot = phase ? &s1 : &s0;
        *slot += sign ? -kNumRanks : +kNumRanks;
        ++ counter;
        out[it + 1] = {counter, {s0, s1}};
    }
    return out;
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kTimeout, int kTag>
static void run_case(const std::string& tag, const int num_iters = 8) {
    BarrierSignals* dev = nullptr;
    CK(cudaMalloc(&dev, sizeof(BarrierSignals) * kNumRanks));
    CK(cudaMemset(dev, 0, sizeof(BarrierSignals) * kNumRanks));
    dim3 grid(kNumRanks * kNumSMs), block(kNumThreads);
    k_nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, kTimeout, kTag><<<grid, block>>>(dev, num_iters);
    CK(cudaDeviceSynchronize());
    std::vector<BarrierSignals> host(kNumRanks);
    CK(cudaMemcpy(host.data(), dev, sizeof(BarrierSignals) * kNumRanks, cudaMemcpyDeviceToHost));
    CK(cudaFree(dev));

    const auto exp = model(kNumRanks, num_iters);
    bool ok = true;
    for (int r = 0; r < kNumRanks; ++ r) {
        if ((int)host[r].nvl_barrier_counter != num_iters) ok = false;
        if (host[r].nvl_barrier_signals[0] != exp[num_iters].slot[0]) ok = false;
        if (host[r].nvl_barrier_signals[1] != exp[num_iters].slot[1]) ok = false;
    }
    printf("    %s: counter=%llu slot0=%d slot1=%d（期望 counter=%d slot0=%lld slot1=%lld）\n",
           tag.c_str(), (unsigned long long)host[0].nvl_barrier_counter,
           host[0].nvl_barrier_signals[0], host[0].nvl_barrier_signals[1],
           num_iters, exp[num_iters].slot[0], exp[num_iters].slot[1]);
    check(ok, tag + "：" + std::to_string(num_iters) + " 次 barrier 后所有 rank 的 counter 与两相位槽均等于模型值");
    dump("barrier ranks=" + std::to_string(kNumRanks) + " sms=" + std::to_string(kNumSMs) +
         " iters=" + std::to_string(num_iters) +
         " counter=" + std::to_string((int)host[0].nvl_barrier_counter) +
         " slot0=" + std::to_string(host[0].nvl_barrier_signals[0]) +
         " slot1=" + std::to_string(host[0].nvl_barrier_signals[1]));
}
}  // namespace sim

static void part1_barrier_protocol() {
    printf("\n=== Part 1: NVLink barrier 协议（comm/barrier.cuh:88-131，真实设备代码 + 对称指针桩）===\n");
    constexpr int64_t kTimeout = 1000000000000LL;  // ~500s，永不超时
    printf("  [model] 相位循环：counter&3 -> phase/sign，+1/-1 读写复用两个槽（免复位）\n");
    sim::run_case<2,  1, 64, kTimeout, 1>("ranks=2  sms=1");
    sim::run_case<4,  1, 64, kTimeout, 2>("ranks=4  sms=1");
    sim::run_case<8,  1, 128, kTimeout, 3>("ranks=8  sms=1");
    sim::run_case<16, 1, 128, kTimeout, 4>("ranks=16 sms=1");
    sim::run_case<32, 1, 128, kTimeout, 5>("ranks=32 sms=1");
    sim::run_case<64, 1, 128, kTimeout, 6>("ranks=64 sms=1");
    sim::run_case<128, 1, 128, kTimeout, 7>("ranks=128 sms=1");
    // SM gating：每 rank 2 个 SM，只有 sm_idx==0 参与；counter 仍每 rank +1/iter。
    sim::run_case<16, 2, 128, kTimeout, 8>("ranks=16 sms=2");
    sim::run_case<8,  4, 128, kTimeout, 9>("ranks=8  sms=4");
    // 奇数次调用：信号槽应留下非零残值（坐实 ±1 真的发生过，而非巧合为 0）。
    sim::run_case<8,  1, 128, kTimeout, 10>("ranks=8  sms=1 iters=5", 5);
    sim::run_case<64, 1, 128, kTimeout, 11>("ranks=64 sms=1 iters=7", 7);
    sim::run_case<16, 1, 128, kTimeout, 12>("ranks=16 sms=1 iters=1", 1);
}

// =============================================================================
// Part 2: get_qp_mode 的 SM/QP 映射（comm/barrier.cuh:56-86）
// =============================================================================
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
static void qp_mode_case() {
    printf("  [config] kNumSMs=%d kNumQPs=%d kNumChannelsPerSM=%d kWithNotifyWarps=%d\n",
           kNumSMs, kNumQPs, kNumChannelsPerSM, (int)kWithNotifyWarps);
    bool ok = true;
    // 每个 (sm, channel) 都应在 [0, kNumQPs) 内；数据通道不得与 notify warp 的 QP0 冲突。
    for (int sm = 0; sm < kNumSMs; ++ sm) {
        for (int ch = 0; ch < kNumChannelsPerSM; ++ ch) {
            const int2 m = deep_ep::comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(sm, ch, false);
            if (m.x < 0 or m.x >= kNumQPs) ok = false;
            if (kWithNotifyWarps and m.x == 0) ok = false;
        }
    }
    // notify warp：恒 QP0 + CTA
    if (kWithNotifyWarps) {
        const int2 n = deep_ep::comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(kNumSMs - 1, kNumChannelsPerSM - 1, true);
        if (n.x != 0 or n.y != 0) ok = false;
    }
    // 打印 SM0 的前 4 个通道
    char buf[256];
    std::string line;
    for (int ch = 0; ch < std::min(4, kNumChannelsPerSM); ++ ch) {
        const int2 m = deep_ep::comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(0, ch, false);
        snprintf(buf, sizeof(buf), "SM0/ch%d->QP%d(%s) ", ch, m.x, m.y ? "GPU" : "CTA");
        line += buf;
    }
    printf("    %s\n", line.c_str());
    check(ok, "get_qp_mode 映射落在合法 QP 范围");
    dump("qpmode sms=" + std::to_string(kNumSMs) + " qps=" + std::to_string(kNumQPs) +
         " ch=" + std::to_string(kNumChannelsPerSM) + " notify=" + std::to_string((int)kWithNotifyWarps));
    for (int sm = 0; sm < kNumSMs; ++ sm) {
        std::string row = "  qpmode_map ";
        for (int ch = 0; ch < kNumChannelsPerSM; ++ ch) {
            const int2 m = deep_ep::comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(sm, ch, false);
            row += std::to_string(m.x) + "," + std::to_string(m.y) + " ";
        }
        dump(row);
    }
}

static void part2_qp_mode() {
    printf("\n=== Part 2: get_qp_mode 的 SM/QP 映射（comm/barrier.cuh:56-86）===\n");
    qp_mode_case<3, 10, 4, false>();
    qp_mode_case<8, 8, 1, false>();
    qp_mode_case<4, 2, 8, false>();
    qp_mode_case<8, 8, 1, true>();
    qp_mode_case<1, 1, 4, false>();
}

// =============================================================================
// Part 3: BarrierSignals 结构（layout/common/barrier.cuh:11-25）
// =============================================================================
static void part3_signals_layout() {
    printf("\n=== Part 3: BarrierSignals 布局（layout/common/barrier.cuh:11-25）===\n");
    check(alignof(deep_ep::layout::BarrierSignals) == deep_ep::kNumRDMAAlignmentBytes,
          "BarrierSignals 对齐到 kNumRDMAAlignmentBytes(64)");
    check(sizeof(deep_ep::layout::BarrierSignals) == deep_ep::kNumRDMAAlignmentBytes,
          "BarrierSignals 大小 = 64B（counter 8B + 两位相位信号 8B，补齐到 64B）");
    printf("    sizeof=%zu alignof=%zu kNumRDMAAlignmentBytes=%d\n",
           sizeof(deep_ep::layout::BarrierSignals), alignof(deep_ep::layout::BarrierSignals),
           deep_ep::kNumRDMAAlignmentBytes);
    dump("signals sizeof=" + std::to_string(sizeof(deep_ep::layout::BarrierSignals)) +
         " align=" + std::to_string(alignof(deep_ep::layout::BarrierSignals)));
}

int main(int argc, char** argv) {
    printf("DeepEP 多 rank 屏障与 symmetric-memory 句柄层实验 —— H100 (sm90a)，直接跑真实设备代码\n");
    part3_signals_layout();
    part1_barrier_protocol();
    part2_qp_mode();

    printf("\n=== 汇总：PASS=%d FAIL=%d ===\n", g_pass, g_fail);
    if (argc > 1) {
        FILE* f = fopen(argv[1], "w");
        if (f) { fputs(g_dump.c_str(), f); fclose(f); }
        printf("已写出设备结果到 %s\n", argv[1]);
    }
    return g_fail == 0 ? 0 : 1;
}
