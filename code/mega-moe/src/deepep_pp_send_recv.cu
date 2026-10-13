// P58-1 实验设备代码：DeepEP 的流水线并行 send/recv 原语
//   （deep_ep/include/deep_ep/impls/pp/pp_send_recv.cuh +
//    deep_ep/include/deep_ep/layout/pp/workspace.cuh +
//    csrc/kernels/pp/pp_send_recv.hpp / csrc/buffers/pp.hpp）。
//
// 为什么 H100(sm90) 可跑：
//   * PP 的「数据面」是**纯 TMA 1D 搬运**（`pp_tma_copy`）——它只调 `ptx::tma_load_1d` /
//     `ptx::tma_store_1d` / mbarrier，全部是 sm90 起可用的架构无关 ISA。
//   * PP 的「寻址面」是**纯整数映射**（`get_pp_buffer_offset` 与 4 段对称缓冲的 block 索引、
//     arrival/release 信号索引），与 NCCL 毫无关系。
//   * 真正与 NCCL 有关的只有数据面的第二跳：`comm::NCCLGin::put`（把本地 send 槽 push 到对端
//     recv 槽）与 `NCCLGin::signal`（StrongVASignalAdd 通知 arrival/release）。它们需要
//     `nccl_device.h` 的 GIN 设备 API（本机 NCCL 2.29.7 不提供）→ **仅源码分析**，不在本文件编译。
//     本文件把 `get_pp_buffer_offset`（cuh:20-25）与 `pp_tma_copy`（cuh:27-102）**逐行原样抽取**
//     （见 VERBATIM 标注），并复用 `layout::PPSignals` 真实头文件，跑真实设备代码。
//
// 编译/运行见 harness/exp_deepep_pp.sh。

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/pp/workspace.cuh>

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

static std::string g_dump;
static void dump(const std::string& line) { g_dump += line; g_dump += "\n"; }

namespace deep_ep::pp {

// ===================== VERBATIM: impls/pp/pp_send_recv.cuh:20-25 =====================
// Returns `(local_idx, peer_idx)`. `local_idx` is this rank's buffer half for the peer (0 for the
// next rank, 1 for the previous rank) and `peer_idx` is the symmetric index the peer uses to
// address the same half.
template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_pp_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
}
// ===================== END VERBATIM =====================

// ===================== VERBATIM: impls/pp/pp_send_recv.cuh:27-102 =====================
template <int kNumSMs,
          int kNumSmemBytes,
          int kNumStages = 2,
          int kNumTMABytesPerStage = math::constexpr_align<int, false>(
              (kNumSmemBytes - kNumStages * static_cast<int>(sizeof(ptx::mbarrier))) / kNumStages,
              kNumTMAAlignmentBytes),
          int kNumTMABlocksPerStage = kNumTMABytesPerStage / kNumTMAAlignmentBytes>
__device__ __forceinline__ void pp_tma_copy(
    void* src_ptr, void* dst_ptr,
    const int64_t& num_bytes, const int& sm_idx) {
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    const auto tma_buffers = smem;
    const auto mbarriers = reinterpret_cast<ptx::mbarrier*>(smem + kNumStages * kNumTMABytesPerStage);
    EP_STATIC_ASSERT(kNumTMABytesPerStage > 0, "Invalid shared memory bytes");
    EP_STATIC_ASSERT(kNumStages >= 2, "Need at least 2 stages for pipelining");

    // Init mbarriers
    ptx::arrival_phase phases[kNumStages];
    #pragma unroll
    for (int s = 0; s < kNumStages; ++ s)
        phases[s] = 0, ptx::mbarrier_init_with_fence(mbarriers + s, 1);

    // Work partitioning across SMs
    EP_DEVICE_ASSERT(num_bytes % kNumTMAAlignmentBytes == 0);
    const auto num_tma_blocks = num_bytes / kNumTMAAlignmentBytes;
    const auto num_tma_blocks_per_sm = math::ceil_div<int64_t>(num_tma_blocks, kNumSMs);
    const auto start_block_idx = sm_idx * num_tma_blocks_per_sm;
    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_sm, num_tma_blocks);
    const auto num_iterations = math::ceil_div<int64_t>(end_block_idx - start_block_idx, kNumTMABlocksPerStage);

    auto get_iter_info = [&](const int64_t& iter_idx) {
        const auto i = start_block_idx + iter_idx * kNumTMABlocksPerStage;
        const auto offset = i * kNumTMAAlignmentBytes;
        const auto num_transaction_bytes =
            std::min<int>(kNumTMABlocksPerStage, end_block_idx - i) * kNumTMAAlignmentBytes;
        return std::make_pair(offset, num_transaction_bytes);
    };

    // Fill pipeline: issue loads for the first kNumStages iterations
    for (int64_t iter_idx = 0; iter_idx < kNumStages and iter_idx < num_iterations; ++ iter_idx) {
        const auto [load_offset, num_load_bytes] = get_iter_info(iter_idx);
        ptx::tma_load_1d(
            tma_buffers + iter_idx * kNumTMABytesPerStage,
            math::advance_ptr(src_ptr, load_offset),
            mbarriers + iter_idx, num_load_bytes);
        ptx::mbarrier_arrive_and_set_tx(mbarriers + iter_idx, num_load_bytes);
    }

    for (int64_t iter_idx = 0; iter_idx < num_iterations; ++ iter_idx) {
        const auto stage_idx = static_cast<int>(iter_idx % kNumStages);
        const auto [store_offset, num_store_bytes] = get_iter_info(iter_idx);

        // Wait this stage's load and issue store
        ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);
        ptx::tma_store_1d(
            math::advance_ptr(dst_ptr, store_offset),
            tma_buffers + stage_idx * kNumTMABytesPerStage,
            num_store_bytes);
        ptx::tma_store_commit();

        // Prefetch: wait until this stage's store is completed, then issue next load
        const auto next_iter_idx = iter_idx + kNumStages;
        if (next_iter_idx < num_iterations) {
            ptx::tma_store_wait();
            const auto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);
            ptx::tma_load_1d(
                tma_buffers + stage_idx * kNumTMABytesPerStage,
                math::advance_ptr(src_ptr, load_offset),
                mbarriers + stage_idx, num_load_bytes);
            ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);
        }
    }

    // Drain all outstanding stores
    ptx::tma_store_wait();
}
// ===================== END VERBATIM =====================

}  // namespace deep_ep::pp

// =============================================================================
// Part A：`get_pp_buffer_offset` + 4 段对称缓冲 block 索引 + 信号索引（纯整数映射）
//   对每个有序对 (src, dst)（src != dst）核出 send 侧与 recv 侧用到的全部索引，dump 给
//   Python 独立 oracle 复核。字段 15 个：
//     N src dst is_next lo peer  send_count stag release_s arrival_s  recv_count recv_blk release_r arrival_r put_blk
// =============================================================================
template <int kNumRanks>
__global__ void k_pp_offsets(int* out, int stride) {
    if (threadIdx.x != 0) return;
    using namespace deep_ep::pp;
    int idx = 0;
    for (int src = 0; src < kNumRanks; ++ src) {
        for (int dst = 0; dst < kNumRanks; ++ dst) {
            if (src == dst) continue;
            const auto pr = get_pp_buffer_offset<kNumRanks>(src, dst);   // (local_idx, peer_idx)
            const int lo = pr.first, peer = pr.second;
            const int is_next = (dst == (src + 1) % kNumRanks) ? 1 : 0;
            int* o = out + idx * stride;
            o[0] = kNumRanks; o[1] = src; o[2] = dst; o[3] = is_next;
            o[4] = lo; o[5] = peer;
            // send 侧（坐标 (rank=src, dst_rank=dst)）：[local_idx_in_dst=lo, dst_idx_in_local=peer]
            o[6]  = peer;            // pp.send_count[dst_idx_in_local]
            o[7]  = peer + 2;        // send staging block = (dst_idx_in_local + 2)
            o[8]  = peer;            // release[idx] 读
            o[9]  = lo;              // arrival[local_idx_in_dst][qp]
            // recv 侧（坐标 (rank=dst, src_rank=src)）：[src_idx_in_local=lo, local_idx_in_src=peer]
            o[10] = lo;              // pp.recv_count[src_idx_in_local]
            o[11] = lo;              // recv block = (src_idx_in_local + 0)
            o[12] = peer;            // release[idx] 写
            o[13] = lo;              // arrival[src_idx_in_local][qp]
            o[14] = lo;              // put 的逻辑目标 block = local_idx_in_dst
            ++ idx;
        }
    }
}

// =============================================================================
// Part B：`pp_tma_copy` 真实 TMA 1D 两级流水搬运 + 跨 SM 分区（真实设备代码）
// =============================================================================
template <int kNumSMs, int kNumSmemBytes>
__global__ void __launch_bounds__(32, 1) k_pp_tma_copy(void* src, void* dst, int64_t num_bytes) {
    const int sm_idx = static_cast<int>(blockIdx.x);
    if (deep_ep::ptx::elect_one_sync())
        deep_ep::pp::pp_tma_copy<kNumSMs, kNumSmemBytes>(src, dst, num_bytes, sm_idx);
}

template <int kNumSMs, int kNumSmemBytes>
static void run_pp_tma_case(int64_t num_bytes, int& pass, int& fail) {
    void* src = nullptr; void* dst = nullptr;
    CK(cudaMalloc(&src, num_bytes));
    CK(cudaMalloc(&dst, num_bytes));
    CK(cudaMemset(dst, 0xAB, num_bytes));

    std::vector<uint8_t> host(num_bytes);
    for (int64_t i = 0; i < num_bytes; ++ i)
        host[i] = static_cast<uint8_t>((i * 2654435761u) >> 13);
    CK(cudaMemcpy(src, host.data(), num_bytes, cudaMemcpyHostToDevice));

    CK(cudaFuncSetAttribute(k_pp_tma_copy<kNumSMs, kNumSmemBytes>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, kNumSmemBytes));
    k_pp_tma_copy<kNumSMs, kNumSmemBytes><<<kNumSMs, 32, kNumSmemBytes>>>(src, dst, num_bytes);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> back(num_bytes);
    CK(cudaMemcpy(back.data(), dst, num_bytes, cudaMemcpyDeviceToHost));
    int mism = 0;
    for (int64_t i = 0; i < num_bytes; ++ i) if (back[i] != host[i]) ++ mism;
    printf("    num_sms=%-3d bytes=%-9lld  mismatch=%d\n", kNumSMs, (long long)num_bytes, mism);
    if (mism == 0) ++ pass; else ++ fail;
    CK(cudaFree(src)); CK(cudaFree(dst));
}

int main(int argc, char** argv) {
    const char* dump_path = argc > 1 ? argv[1] : "/tmp/deepep_pp_dev.txt";

    printf("=== Part C: PPSignals 布局（layout/pp/workspace.cuh）===\n");
    using namespace deep_ep::layout;
    printf("    sizeof(PPSignals)=%zu  alignof=%zu  kNumMaxSignalBytes=%d\n",
           sizeof(PPSignals), alignof(PPSignals), deep_ep::kNumMaxSignalBytes);
    printf("    offset barrier=%zu send_count=%zu recv_count=%zu arrival=%zu release=%zu\n",
           offsetof(PPSignals, barrier), offsetof(PPSignals, send_count),
           offsetof(PPSignals, recv_count), offsetof(PPSignals, arrival),
           offsetof(PPSignals, release));
    check(sizeof(PPSignals) <= deep_ep::kNumMaxSignalBytes, "PPSignals 落在 16 MiB 信号区内");
    check(alignof(PPSignals) == deep_ep::kNumRDMAAlignmentBytes, "PPSignals 按 RDMA 64B 对齐");
    check(sizeof(BarrierSignals) == deep_ep::kNumRDMAAlignmentBytes,
          "BarrierSignals = 64B（counter 8 + 2 × int 4，补齐到 64）");

    printf("\n=== Part A: get_pp_buffer_offset + 缓冲/信号索引（设备真实代码）===\n");
    for (int N = 2; N <= 8; ++ N) {
        int pairs = N * (N - 1);
        int* d_out = nullptr;
        CK(cudaMalloc(&d_out, sizeof(int) * pairs * 15));
        int* h_out = new int[pairs * 15];
        if (N == 2) k_pp_offsets<2><<<1, 32>>>(d_out, 15);
        else if (N == 3) k_pp_offsets<3><<<1, 32>>>(d_out, 15);
        else if (N == 4) k_pp_offsets<4><<<1, 32>>>(d_out, 15);
        else if (N == 5) k_pp_offsets<5><<<1, 32>>>(d_out, 15);
        else if (N == 6) k_pp_offsets<6><<<1, 32>>>(d_out, 15);
        else if (N == 7) k_pp_offsets<7><<<1, 32>>>(d_out, 15);
        else             k_pp_offsets<8><<<1, 32>>>(d_out, 15);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_out, d_out, sizeof(int) * pairs * 15, cudaMemcpyDeviceToHost));
        for (int i = 0; i < pairs; ++ i)
            dump(std::string("A ") + std::to_string(N) + " " +
                 std::to_string(h_out[i*15+1]) + " " + std::to_string(h_out[i*15+2]) + " " +
                 std::to_string(h_out[i*15+3]) + " " + std::to_string(h_out[i*15+4]) + " " +
                 std::to_string(h_out[i*15+5]) + " " + std::to_string(h_out[i*15+6]) + " " +
                 std::to_string(h_out[i*15+7]) + " " + std::to_string(h_out[i*15+8]) + " " +
                 std::to_string(h_out[i*15+9]) + " " + std::to_string(h_out[i*15+10]) + " " +
                 std::to_string(h_out[i*15+11]) + " " + std::to_string(h_out[i*15+12]) + " " +
                 std::to_string(h_out[i*15+13]) + " " + std::to_string(h_out[i*15+14]));
        delete[] h_out;
        CK(cudaFree(d_out));
    }
    printf("    已 dump 2..8 rank 的全部 (src,dst) 索引到 %s\n", dump_path);

    printf("\n=== Part B: pp_tma_copy 真实 TMA 两级流水搬运（sm90 真机）===\n");
    constexpr int kSmem = 96 * 1024;
    int bpass = 0, bfail = 0;
    const int64_t sizes[] = {32, 512, 4096, 12288, 65536, 100000, 1048576, 159968};
    for (int64_t sz : sizes) run_pp_tma_case<1,  kSmem>(sz, bpass, bfail);
    for (int64_t sz : sizes) run_pp_tma_case<2,  kSmem>(sz, bpass, bfail);
    for (int64_t sz : sizes) run_pp_tma_case<4,  kSmem>(sz, bpass, bfail);
    for (int64_t sz : sizes) run_pp_tma_case<8,  kSmem>(sz, bpass, bfail);
    for (int64_t sz : sizes) run_pp_tma_case<32, kSmem>(sz, bpass, bfail);
    printf("    Part B: %d/%d byte-exact\n", bpass, bpass + bfail);
    check(bfail == 0, "pp_tma_copy 在所有 (num_sms, bytes) 组合上 byte-exact");

    printf("\n===== 设备汇总: PASS=%d FAIL=%d =====\n", g_pass, g_fail);

    FILE* f = fopen(dump_path, "w");
    if (f) { fwrite(g_dump.data(), 1, g_dump.size(), f); fclose(f); }
    return g_fail == 0 ? 0 : 1;
}
