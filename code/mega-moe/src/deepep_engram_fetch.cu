// P59-1 实验设备代码：DeepEP 的 Engram 远程抓取原语
//   （deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh +
//    deep_ep/include/deep_ep/impls/engram/engram_fetch_wait.cuh +
//    deep_ep/include/deep_ep/layout/engram/workspace.cuh +
//    csrc/kernels/engram/engram.hpp / csrc/buffers/engram.hpp）。
//
// 为什么 H100(sm90) 可跑：
//   * Engram 的「布局层」`EngramLayout<>`（前缀和 / 存储字节偏移 / 接收字节偏移 / SF 行号与
//     分片号）全是 constexpr 整数运算，架构无关（真实头文件 `layout/engram/workspace.cuh`）。
//   * Engram 的「发射面」`EngramIssueWarpLayout`（issue warp ↔ (rdma_peer, gin_context, token tile)
//     的确定映射）、以及 issue 循环里「哪些 entry 归本 peer、落到哪个 window / 源偏移 / 目的偏移」
//     也全是整数映射与 `__ballot_sync`/`__popc`，与 NCCL 无关。
//   * 真正与 NCCL 有关的只有：`gin.get`（RDMA 拉取）、`gin.flush_async`（门铃）、
//     `gin.wait`（`engram_fetch_wait`）与 SF 的 `gin.get_sym_ptr<ncclTeamTagLsa>`（NVLink 本地
//     分片寻址）。它们需要 `nccl_device.h` 的 GIN 设备 API（本机 NCCL 2.29.7 不提供）
//     → **仅源码分析**，不在本文件编译。
//     本文件把 `EngramIssueWarpLayout`（cuh:13-39）与 issue 循环（cuh:99-171）**逐行原样抽取**
//     （见 VERBATIM 标注），把 `gin.get` 换成一个只记录参数的 recorder，从而验证「索引映射」；
//     并复用真实头文件里的 `layout::EngramLayout` / `layout::EngramWorkspace` 跑真实设备代码。
//
// 编译/运行见 harness/exp_deepep_engram.sh。

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/engram/workspace.cuh>

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

static std::string g_dump;
static void dump(const std::string& line) { g_dump += line; g_dump += "\n"; }

// ============================ 记录器（替代 gin.get）============================
struct EngramRequest {
    int layer;
    int rdma_peer;
    int entry_idx;         // 本 token 槽序号 = request 低 32 位
    int window;            // = intra_peer_rank_idx，即本地哪个 RDMA storage window
    long long src_off;     // storage 源字节偏移
    long long dst_off;     // 接收缓冲目的字节偏移
};

__device__ EngramRequest* g_requests;
__device__ int* g_num_requests;
__device__ void record_request(const int layer, const int peer, const int entry_idx,
                               const int window, const long long src_off, const long long dst_off) {
    const auto slot = atomicAdd(g_num_requests, 1);
    EngramRequest& r = g_requests[slot];
    r.layer = layer; r.rdma_peer = peer; r.entry_idx = entry_idx;
    r.window = window; r.src_off = src_off; r.dst_off = dst_off;
}

namespace deep_ep::engram {

// ===================== VERBATIM: impls/engram/engram_fetch.cuh:13-39 =====================
template <int kNumRDMAPeers, int kNumGinContexts, int kNumIssueWarps>
struct EngramIssueWarpLayout {
    int rdma_peer_idx;
    int gin_context_idx;
    int peer_context_idx;
    int num_peer_context_warps;
    int peer_warp_idx;
    int num_peer_warps;

    __device__ __forceinline__ explicit EngramIssueWarpLayout(const int issue_warp_idx) {
        constexpr int kNumPeerContexts = kNumRDMAPeers * kNumGinContexts;

        peer_context_idx = issue_warp_idx % kNumPeerContexts;
        rdma_peer_idx = peer_context_idx % kNumRDMAPeers;
        gin_context_idx = peer_context_idx / kNumRDMAPeers;
        num_peer_context_warps = math::ceil_div(kNumIssueWarps - peer_context_idx, kNumPeerContexts);

        peer_warp_idx = issue_warp_idx / kNumRDMAPeers;
        num_peer_warps = math::ceil_div(kNumIssueWarps - rdma_peer_idx, kNumRDMAPeers);
    }

    __device__ __forceinline__ int get_num_active_peer_context_warps(const int num_token_tiles) const {
        const auto num_token_warps = gin_context_idx < num_token_tiles
            ? math::ceil_div(num_token_tiles - gin_context_idx, kNumGinContexts) : 0;
        return num_token_warps < num_peer_context_warps ? num_token_warps : num_peer_context_warps;
    }
};
// ===================== END VERBATIM =====================

// ------------------- Part A: 布局算术（真实头文件 layout::EngramLayout）-------------------
template <typename engram_t>
__global__ void k_layout(int64_t* out, long long* storage_off, long long* recv_off,
                         int* sf_row, int* sf_shard, int* sf_g, int num_layer, int num_entries_per_token,
                         int num_max_tokens, int num_ranks) {
    if (threadIdx.x != 0) return;
    int p = 0;
    for (int l = 0; l < num_layer; ++l) {
        const auto e = engram_t::get_num_entries(l);
        for (int i = 0; i < e; ++i) {
            storage_off[p] = engram_t::get_storage_byte_offset(l, i);
            const int entry_slot = l * num_entries_per_token + i;   // 仅展示层内插值
            recv_off[p] = engram_t::get_recv_byte_offset(l, entry_slot, num_max_tokens);
            ++p;
        }
    }
    out[0] = engram_t::get_num_prefix_entries(num_layer - 1) + engram_t::get_num_entries(num_layer - 1);
    // SF：对若干 global_idx 记录 row / shard
    int q = 0;
    for (int l = 0; l < num_layer; ++l) {
        const auto e = engram_t::get_num_entries(l);
        for (int g = 0; g < num_ranks * e; g += (g < 8 ? 1 : 7)) {
            sf_g[q] = g;
            sf_row[q] = static_cast<int>(engram_t::get_sf_row_idx(l, g));
            sf_shard[q] = static_cast<int>(engram_t::get_sf_shard_idx(l, g));
            ++q;
        }
    }
    out[1] = q;
}

// ------------------- Part B: workspace 偏移 -------------------
__global__ void k_workspace(int64_t* out) {
    if (threadIdx.x != 0) return;
    out[0] = sizeof(layout::EngramWorkspace);
    out[1] = offsetof(layout::EngramWorkspace, barrier);
    out[2] = offsetof(layout::EngramWorkspace, issue_counters);
    layout::EngramWorkspace w;
    out[3] = reinterpret_cast<int64_t>(w.get_issue_counter_ptr(3, 5)) - reinterpret_cast<int64_t>(&w);
    out[4] = layout::EngramWorkspace::kNumMaxLayers;
    out[5] = layout::EngramWorkspace::kNumMaxPeerContexts;
}

// ------------------- Part C: issue warp layout dump -------------------
template <int kNumRDMAPeers, int kNumGinContexts, int kNumIssueWarps>
__global__ void k_issue_layout(int* out, int* ntiles_out, int num_tokens) {
    const int idx = threadIdx.x;
    if (idx >= kNumIssueWarps) return;
    const auto il = EngramIssueWarpLayout<kNumRDMAPeers, kNumGinContexts, kNumIssueWarps>(idx);
    const int num_token_tiles = math::ceil_div(num_tokens, 32);
    int* o = out + idx * 8;
    o[0] = il.rdma_peer_idx;
    o[1] = il.gin_context_idx;
    o[2] = il.peer_context_idx;
    o[3] = il.num_peer_context_warps;
    o[4] = il.peer_warp_idx;
    o[5] = il.num_peer_warps;
    o[6] = il.get_num_active_peer_context_warps(num_token_tiles);
    o[7] = num_token_tiles;
    if (idx == 0) ntiles_out[0] = num_token_tiles;
}

// ------------------- Part D: issue 循环（逐行抽取 + gin.get → recorder）-------------------
// 复刻 impls/engram/engram_fetch.cuh:80-207 的 issue warp 主体，唯一改动是把
// `gin.get<...>(storage_windows[window], src, ..., fetched+dst, hidden, peer, options)` 换成
// `record_request(layer, peer, entry_idx, window, src, dst)`。
template <int kNumSMs, int kNumQPs, int kNumIssueWarpsPerSM, int kNumMaxTokens,
          int kNumRDMAPeers, int kNumRanksPerRDMAPeer, typename engram_t>
__global__ void __launch_bounds__(kNumIssueWarpsPerSM * 32, 1)
k_issue_loop(int* indices, int num_tokens) {
    constexpr int kWarpSize = 32;
    constexpr int kNumLayers = engram_t::kNumLayers;
    constexpr int kNumEntriesPerToken = engram_t::kNumEntriesPerToken;

    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = static_cast<int>(threadIdx.x) / kWarpSize;
    const auto lane_idx = static_cast<int>(threadIdx.x) % kWarpSize;

    __shared__ int64_t pending_requests[kNumIssueWarpsPerSM][2 * kWarpSize];

    if (warp_idx >= kNumIssueWarpsPerSM) return;

    const auto issue_warp_idx = sm_idx * kNumIssueWarpsPerSM + warp_idx;
    const auto num_token_tiles = math::ceil_div(num_tokens, kWarpSize);
    const auto issue_layout = EngramIssueWarpLayout<
        kNumRDMAPeers, kNumQPs, kNumSMs * kNumIssueWarpsPerSM>(issue_warp_idx);
    if (issue_layout.peer_warp_idx >= num_token_tiles) return;

    const auto num_active_peer_context_warps =
        issue_layout.get_num_active_peer_context_warps(num_token_tiles);
    const auto doorbell_threshold = 256 >= num_active_peer_context_warps
        ? 256 / num_active_peer_context_warps : 1;
    auto* warp_pending_requests = pending_requests[warp_idx];

    const auto num_entries = num_tokens * kNumEntriesPerToken;
    #pragma unroll
    for (int layer_idx = 0; layer_idx < kNumLayers; ++ layer_idx) {
        int num_issued_requests = 0;
        int num_pending_requests = 0;
        int pending_head = 0;
        const auto num_entries_per_rank = engram_t::get_num_entries(layer_idx);
        const auto peer_global_idx_begin = issue_layout.rdma_peer_idx * kNumRanksPerRDMAPeer * num_entries_per_rank;
        const auto peer_global_idx_end = (issue_layout.rdma_peer_idx + 1) * kNumRanksPerRDMAPeer * num_entries_per_rank;

        const auto issue_batch = [&](const int batch_size, const bool is_last_batch = false) {
            const auto rings_doorbell = is_last_batch or num_issued_requests % doorbell_threshold + batch_size >= doorbell_threshold;
            num_issued_requests += batch_size;
            (void)rings_doorbell;

            if (lane_idx < batch_size) {
                const auto request = warp_pending_requests[pending_head + lane_idx];
                const auto entry_idx = static_cast<int>(request);
                const auto global_idx = static_cast<int>(request >> 32);
                const auto peer_local_idx = global_idx - peer_global_idx_begin;
                const auto intra_peer_rank_idx = peer_local_idx / num_entries_per_rank;
                const auto local_entry_idx = peer_local_idx % num_entries_per_rank;
                const auto src_byte_offset = engram_t::get_storage_byte_offset(layer_idx, local_entry_idx);
                const auto dst_byte_offset = engram_t::get_recv_byte_offset(
                    layer_idx, entry_idx, kNumMaxTokens);
                record_request(layer_idx, issue_layout.rdma_peer_idx, entry_idx,
                               intra_peer_rank_idx, src_byte_offset, dst_byte_offset);
            }
        };

        for (int token_tile_idx = issue_layout.peer_warp_idx;
             token_tile_idx < num_token_tiles;
             token_tile_idx += issue_layout.num_peer_warps) {
            const auto first_entry_idx = token_tile_idx * kWarpSize * kNumEntriesPerToken;

            for (int round_idx = 0; round_idx < kNumEntriesPerToken; ++round_idx) {
                const auto entry_idx = first_entry_idx + round_idx * kWarpSize + lane_idx;
                const auto global_idx = entry_idx < num_entries
                    ? __ldg(indices + layer_idx * num_entries + entry_idx)
                    : -1;
                const auto belongs_to_peer =
                    global_idx >= peer_global_idx_begin && global_idx < peer_global_idx_end;
                const auto peer_mask = __ballot_sync(0xffffffffu, belongs_to_peer);

                if (belongs_to_peer) {
                    const auto lower_lanes_mask = (1u << lane_idx) - 1;
                    const auto pending_idx =
                        (pending_head + num_pending_requests +
                         __popc(peer_mask & lower_lanes_mask)) &
                        (2 * kWarpSize - 1);
                    warp_pending_requests[pending_idx] =
                        (static_cast<int64_t>(global_idx) << 32) | entry_idx;
                }
                __syncwarp();
                num_pending_requests += __popc(peer_mask);

                if (num_pending_requests > kWarpSize) {
                    issue_batch(kWarpSize);
                    pending_head ^= kWarpSize;
                    num_pending_requests -= kWarpSize;
                }
            }
        }

        if (num_pending_requests > 0) {
            issue_batch(num_pending_requests, true);
            __syncwarp();
        }
        __syncwarp();
    }
}

// ------------------- Part E: SF 拷贝（逐行抽取 copy_sf_entry，get_sym_ptr → 本地分片桩）---------
// 复刻 impls/engram/engram_fetch.cuh:208-254 的 SF warp 主体；把
// `gin.get_sym_ptr<ncclTeamTagLsa>(sf_shards, sf_shard_idx)` 换成
// `sf_shards + sf_shard_idx * shard_stride`（本地分片数组桩）。
template <int kNumSMs, int kNumSFWarpsPerSM, int kNumIssueWarpsPerSM, typename engram_t,
          int kNumSFPacks, int kNumMaxTokens>
__global__ void __launch_bounds__((kNumSFWarpsPerSM + kNumIssueWarpsPerSM) * 32, 1)
k_sf_copy(int* indices, sf_pack_t* fetched_sf, const sf_pack_t* sf_shards,
          long long shard_stride, int num_tokens,
          int sf_token_stride, int sf_hidden_stride, int sf_layer_stride) {
    constexpr int kWarpSize = 32;

    if (threadIdx.x / kWarpSize < kNumIssueWarpsPerSM) return;   // 只有 SF warp 参与
    const auto warp_idx = static_cast<int>(threadIdx.x) / kWarpSize;
    const auto lane_idx = static_cast<int>(threadIdx.x) % kWarpSize;
    const auto sm_idx = static_cast<int>(blockIdx.x);

    constexpr int kNumPacksPerVec = sizeof(uint64_t) / sizeof(sf_pack_t);
    constexpr int kNumVecs = kNumSFPacks / kNumPacksPerVec;

    const auto sf_warp_idx = sm_idx * kNumSFWarpsPerSM + warp_idx - kNumIssueWarpsPerSM;
    const auto num_entries = num_tokens * engram_t::kNumEntriesPerToken;
    const auto num_sf_warps = kNumSMs * kNumSFWarpsPerSM;

    const auto copy_sf_entry = [&](const int layer_idx, const int entry_idx) {
        const auto global_idx = __ldg(indices + layer_idx * num_entries + entry_idx);
        if (global_idx < 0) return;

        const auto sf_row_idx = engram_t::get_sf_row_idx(layer_idx, global_idx);
        const auto sf_shard_idx = engram_t::get_sf_shard_idx(layer_idx, global_idx);
        const auto* sf_shard = sf_shards + static_cast<long long>(sf_shard_idx) * shard_stride;
        const auto* src = reinterpret_cast<const uint64_t*>(sf_shard + sf_row_idx * kNumSFPacks);

        uint64_t values[kNumVecs];
        #pragma unroll
        for (int i = 0; i < kNumVecs; ++ i)
            values[i] = __ldg(src + i);

        auto* dst = fetched_sf + layer_idx * sf_layer_stride +
            (entry_idx / engram_t::kNumEntriesPerToken) * sf_token_stride +
            (entry_idx % engram_t::kNumEntriesPerToken) * kNumSFPacks * sf_hidden_stride;
        #pragma unroll
        for (int i = 0; i < kNumVecs; ++ i) {
            #pragma unroll
            for (int j = 0; j < kNumPacksPerVec; ++ j)
                dst[(i * kNumPacksPerVec + j) * sf_hidden_stride] =
                    reinterpret_cast<const sf_pack_t*>(&values[i])[j];
        }
    };

    #pragma unroll
    for (int layer_idx = 0; layer_idx < engram_t::kNumLayers; ++layer_idx) {
        const auto num_token_tiles = math::ceil_div(num_tokens, kWarpSize);
        for (int i = sf_warp_idx; i < engram_t::kNumEntriesPerToken * num_token_tiles; i += num_sf_warps) {
            const auto entry_slot_idx = i / num_token_tiles;
            const auto token_tile_idx = i % num_token_tiles;
            const auto token_idx = token_tile_idx * kWarpSize + lane_idx;
            if (token_idx < num_tokens)
                copy_sf_entry(layer_idx, token_idx * engram_t::kNumEntriesPerToken + entry_slot_idx);
        }
    }
}

}  // namespace deep_ep::engram

using namespace deep_ep;

// ============================ 配置 ============================
// 布局：4 ranks，2 SF shards，hidden 64B，每 token 2 个 entry，层 entry 数 {4, 6}
using LAYOUT = layout::EngramLayout<4, 2, 64, 2, 4, 6>;
constexpr int kNumRanks = 4;
constexpr int kNumSFShards = 2;
constexpr int kNumRDMAPeers = 2;      // num_ranks / gin_min_stride = 4 / 2
constexpr int kNumRanksPerPeer = 2;   // gin_min_stride
constexpr int kNumQPs = 2;
constexpr int kNumIssueWarpsPerSM = 4;
constexpr int kNumSFWarpsPerSM = 4;
constexpr int kNumSMs = 1;
constexpr int kNumMaxTokens = 64;
constexpr int kNumSFPacks = 4;
static_assert(kNumRDMAPeers * kNumRanksPerPeer == kNumRanks, "rank count");

int main(int argc, char** argv) {
    const char* dump_path = argc > 1 ? argv[1] : "/tmp/deepep_engram_dev.txt";

    printf("=== Part A: layout::EngramLayout 布局算术 ===\n");
    {
        const int total_entries = 4 + 6;
        int64_t* d_out; long long* d_storage; long long* d_recv; int* d_row; int* d_shard;
        CK(cudaMalloc(&d_out, sizeof(int64_t) * 8));
        CK(cudaMalloc(&d_storage, sizeof(long long) * total_entries));
        CK(cudaMalloc(&d_recv, sizeof(long long) * total_entries));
        CK(cudaMalloc(&d_row, sizeof(int) * 2048));
        CK(cudaMalloc(&d_shard, sizeof(int) * 2048));
        int* d_g; CK(cudaMalloc(&d_g, sizeof(int) * 2048));
        deep_ep::engram::k_layout<LAYOUT><<<1, 32>>>(d_out, d_storage, d_recv, d_row, d_shard, d_g, 2, 2, kNumMaxTokens, kNumRanks);
        CK(cudaDeviceSynchronize());
        int64_t h_out[8]; long long h_storage[16], h_recv[16]; int h_row[2048], h_shard[2048], h_g[2048];
        CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_storage, d_storage, sizeof(long long) * total_entries, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_recv, d_recv, sizeof(long long) * total_entries, cudaMemcpyDeviceToHost));
        const int q = static_cast<int>(h_out[1]);
        CK(cudaMemcpy(h_row, d_row, sizeof(int) * q, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_shard, d_shard, sizeof(int) * q, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_g, d_g, sizeof(int) * q, cudaMemcpyDeviceToHost));
        printf("  total_entries=%lld\n", (long long)h_out[0]);
        for (int i = 0; i < total_entries; ++i)
            dump("LA " + std::to_string(h_storage[i]) + " " + std::to_string(h_recv[i]));
        dump("LN " + std::to_string(h_out[0]));
        for (int i = 0; i < q; ++i)
            dump("LS " + std::to_string(h_row[i]) + " " + std::to_string(h_shard[i]) + " " + std::to_string(h_g[i]));
        dump("LQ " + std::to_string(q));
        check(h_out[0] == 10, "get_num_prefix_entries+last 求和 = 4+6 = 10");
        check(h_storage[0] == 0 && h_storage[4] == 4 * 64, "layer0→layer1 存储偏移按 entry 数连续");
        check(h_recv[1] - h_recv[0] == 64, "接收偏移步长 = hidden bytes");
        cudaFree(d_out); cudaFree(d_storage); cudaFree(d_recv); cudaFree(d_row); cudaFree(d_shard); cudaFree(d_g);
    }

    printf("=== Part B: layout::EngramWorkspace 偏移 ===\n");
    {
        int64_t* d; CK(cudaMalloc(&d, sizeof(int64_t) * 8));
        deep_ep::engram::k_workspace<<<1, 32>>>(d);
        CK(cudaDeviceSynchronize());
        int64_t h[8]; CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
        printf("  sizeof=%lld barrier@%lld issue_counters@%lld ctr(3,5)@%lld layers=%lld ctx=%lld\n",
               (long long)h[0], (long long)h[1], (long long)h[2], (long long)h[3],
               (long long)h[4], (long long)h[5]);
        dump("WA " + std::to_string(h[0]));
        dump("WB " + std::to_string(h[1]));
        dump("WC " + std::to_string(h[2]));
        dump("WD " + std::to_string(h[3]));
        dump("WE " + std::to_string(h[4]));
        dump("WF " + std::to_string(h[5]));
        // ctr(layer, pc) = barrier_off(64) + (layer*kNumMaxPeerContexts + pc)*4
        const long long exp = 64 + (3LL * h[5] + 5) * 4;
        check(h[3] == exp, "get_issue_counter_ptr 偏移 = align(barrier,64) + (layer*2048+pc)*4");
        check(h[0] == 64 + 128LL * 2048 * 4, "sizeof = BarrierSignals(64) + 128*2048*4");
        cudaFree(d);
    }

    printf("=== Part C: EngramIssueWarpLayout 映射 ===\n");
    {
        const int kNumIssueWarps = kNumSMs * kNumIssueWarpsPerSM;
        int *d_out, *d_nt; CK(cudaMalloc(&d_out, sizeof(int) * kNumIssueWarps * 8));
        CK(cudaMalloc(&d_nt, sizeof(int)));
        deep_ep::engram::k_issue_layout<kNumRDMAPeers, kNumQPs, kNumIssueWarps>
            <<<1, 128>>>(d_out, d_nt, kNumMaxTokens);
        CK(cudaDeviceSynchronize());
        int h[kNumIssueWarps * 8], nt; CK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&nt, d_nt, sizeof(nt), cudaMemcpyDeviceToHost));
        dump("CN " + std::to_string(nt) + " " + std::to_string(kNumIssueWarps) + " " +
             std::to_string(kNumRDMAPeers) + " " + std::to_string(kNumQPs));
        for (int i = 0; i < kNumIssueWarps; ++i)
            dump("CA " + std::to_string(h[i * 8 + 0]) + " " + std::to_string(h[i * 8 + 1]) + " " +
                 std::to_string(h[i * 8 + 2]) + " " + std::to_string(h[i * 8 + 3]) + " " +
                 std::to_string(h[i * 8 + 4]) + " " + std::to_string(h[i * 8 + 5]) + " " +
                 std::to_string(h[i * 8 + 6]) + " " + std::to_string(h[i * 8 + 7]));
        cudaFree(d_out); cudaFree(d_nt);
    }

    printf("=== Part D: issue 循环请求生成（gin.get → recorder）===\n");
    {
        const int num_tokens = kNumMaxTokens;
        const int num_entries = num_tokens * LAYOUT::kNumEntriesPerToken;
        std::vector<int> h_indices(LAYOUT::kNumLayers * num_entries);
        // 构造：一部分 -1，一部分落在 peer0 span，一部分落在 peer1 span
        for (int l = 0; l < LAYOUT::kNumLayers; ++l) {
            const int epr = LAYOUT::get_num_entries(l);
            const int span = kNumRanksPerPeer * epr;
            for (int i = 0; i < num_entries; ++i) {
                int v;
                const int m = (i + l) % 6;
                if (m == 0) v = -1;                                   // OOB
                else if (m <= 2) v = (i * 3) % span;                  // peer 0 span
                else if (m <= 4) v = span + ((i * 5) % span);         // peer 1 span
                else v = (i * 7) % (kNumRanks * epr);         // 任意 rank（仍属某 peer）
                h_indices[l * num_entries + i] = v;
            }
        }
        int* d_indices; CK(cudaMalloc(&d_indices, sizeof(int) * h_indices.size()));
        CK(cudaMemcpy(d_indices, h_indices.data(), sizeof(int) * h_indices.size(), cudaMemcpyHostToDevice));
        EngramRequest* d_req; int* d_cnt;
        const int cap = 1 << 20;
        CK(cudaMalloc(&d_req, sizeof(EngramRequest) * cap));
        CK(cudaMalloc(&d_cnt, sizeof(int)));
        CK(cudaMemset(d_cnt, 0, sizeof(int)));
        CK(cudaMemcpyToSymbol(g_requests, &d_req, sizeof(void*)));
        CK(cudaMemcpyToSymbol(g_num_requests, &d_cnt, sizeof(void*)));
        deep_ep::engram::k_issue_loop<kNumSMs, kNumQPs, kNumIssueWarpsPerSM, kNumMaxTokens,
                                      kNumRDMAPeers, kNumRanksPerPeer, LAYOUT>
            <<<kNumSMs, kNumIssueWarpsPerSM * 32>>>(d_indices, num_tokens);
        CK(cudaDeviceSynchronize());
        int cnt; CK(cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<EngramRequest> reqs(cnt);
        CK(cudaMemcpy(reqs.data(), d_req, sizeof(EngramRequest) * cnt, cudaMemcpyDeviceToHost));
        printf("  recorded requests = %d\n", cnt);
        dump("DN " + std::to_string(num_tokens) + " " + std::to_string(LAYOUT::kNumLayers) + " " +
             std::to_string(LAYOUT::kNumEntriesPerToken) + " " + std::to_string(kNumRanksPerPeer) + " " +
             std::to_string(kNumRanks));
        for (int i = 0; i < LAYOUT::kNumLayers; ++i)
            dump("DL " + std::to_string(LAYOUT::get_num_entries(i)));
        for (const auto& r : reqs)
            dump("DR " + std::to_string(r.layer) + " " + std::to_string(r.rdma_peer) + " " +
                 std::to_string(r.entry_idx) + " " + std::to_string(r.window) + " " +
                 std::to_string(r.src_off) + " " + std::to_string(r.dst_off));
        for (int i = 0; i < num_entries; ++i)
            dump("DI " + std::to_string(h_indices[i]) + " " + std::to_string(h_indices[num_entries + i]));
        cudaFree(d_indices); cudaFree(d_req); cudaFree(d_cnt);
    }

    printf("=== Part E: SF 拷贝（copy_sf_entry, get_sym_ptr → 本地分片桩）===\n");
    {
        const int num_tokens = kNumMaxTokens;
        const int num_layers = LAYOUT::kNumLayers;
        const int num_entries = num_tokens * LAYOUT::kNumEntriesPerToken;
        const int kNumSFPacks = ::kNumSFPacks;

        // SF 源：每 shard rows = kNumSFOwnersPerShard * 所有层 entry 总数 行，逐行 kNumSFPacks 个 pack
        int total_entries_all = 0;
        for (int l = 0; l < num_layers; ++l) total_entries_all += LAYOUT::get_num_entries(l);
        const int rows_per_shard = LAYOUT::kNumSFOwnersPerShard * total_entries_all;
        const long long shard_stride = static_cast<long long>(rows_per_shard) * kNumSFPacks;
        std::vector<sf_pack_t> h_shards(kNumSFShards * shard_stride);
        for (size_t i = 0; i < h_shards.size(); ++i) h_shards[i].ue8m0x4 = static_cast<int>(i * 2654435761u);
        sf_pack_t* d_shards; CK(cudaMalloc(&d_shards, sizeof(sf_pack_t) * h_shards.size()));
        CK(cudaMemcpy(d_shards, h_shards.data(), sizeof(sf_pack_t) * h_shards.size(), cudaMemcpyHostToDevice));
        int* d_indices; CK(cudaMalloc(&d_indices, sizeof(int) * num_layers * num_entries));
        std::vector<int> hi(num_layers * num_entries);
        for (int l = 0; l < num_layers; ++l) {
            const int epr = LAYOUT::get_num_entries(l);
            for (int i = 0; i < num_entries; ++i)
                hi[l * num_entries + i] = (i * 13 + l * 7) % (kNumRanks * epr);
        }
        hi[3] = -1;
        CK(cudaMemcpy(d_indices, hi.data(), sizeof(int) * num_layers * num_entries, cudaMemcpyHostToDevice));
        dump("ES " + std::to_string(kNumSFShards) + " " + std::to_string(shard_stride) + " " +
             std::to_string(kNumSFPacks) + " " + std::to_string(total_entries_all) + " " +
             std::to_string(LAYOUT::kNumSFOwnersPerShard));
        for (size_t i = 0; i < h_shards.size(); ++i)
            dump("EV " + std::to_string(h_shards[i].ue8m0x4));
        for (int l = 0; l < num_layers; ++l) {
            for (int i = 0; i < num_entries; ++i)
                dump("EI " + std::to_string(l) + " " + std::to_string(i) + " " + std::to_string(hi[l * num_entries + i]));
        }

        for (int variant = 0; variant < 2; ++variant) {
            const bool tma = variant == 0;
            int sf_token_stride, sf_hidden_stride, sf_layer_stride;
            if (tma) {
                sf_token_stride = 1;
                sf_hidden_stride = (num_tokens + deep_ep::kNumAlignedSFPacks - 1) / deep_ep::kNumAlignedSFPacks * deep_ep::kNumAlignedSFPacks;
            } else {
                sf_token_stride = LAYOUT::kNumEntriesPerToken * kNumSFPacks;
                sf_hidden_stride = 1;
            }
            sf_layer_stride = tma ? LAYOUT::kNumEntriesPerToken * kNumSFPacks * sf_hidden_stride
                                  : LAYOUT::kNumEntriesPerToken * kNumSFPacks * num_tokens;
            const int alloc = num_layers * sf_layer_stride;
            sf_pack_t* d_out; CK(cudaMalloc(&d_out, sizeof(sf_pack_t) * alloc));
            CK(cudaMemset(d_out, 0, sizeof(sf_pack_t) * alloc));
            deep_ep::engram::k_sf_copy<kNumSMs, kNumSFWarpsPerSM, kNumIssueWarpsPerSM,
                                       LAYOUT, kNumSFPacks, kNumMaxTokens>
                <<<kNumSMs, (kNumSFWarpsPerSM + kNumIssueWarpsPerSM) * 32>>>(
                    d_indices, d_out, d_shards, shard_stride, num_tokens,
                    sf_token_stride, sf_hidden_stride, sf_layer_stride);
            CK(cudaDeviceSynchronize());
            std::vector<sf_pack_t> h_out(alloc);
            CK(cudaMemcpy(h_out.data(), d_out, sizeof(sf_pack_t) * alloc, cudaMemcpyDeviceToHost));
            dump(std::string("EN ") + (tma ? "0" : "1") + " " + std::to_string(sf_token_stride) + " " +
                 std::to_string(sf_hidden_stride) + " " + std::to_string(sf_layer_stride) + " " +
                 std::to_string(kNumSFPacks) + " " + std::to_string(shard_stride) + " " + std::to_string(rows_per_shard));
            char buf[64];
            for (int i = 0; i < alloc; ++i) {
                snprintf(buf, sizeof(buf), "%d", h_out[i].ue8m0x4);
                dump(std::string("EO ") + buf);
            }
            cudaFree(d_out);
        }
        cudaFree(d_shards); cudaFree(d_indices);
    }

    FILE* f = fopen(dump_path, "w");
    if (!f) { printf("cannot open dump %s\n", dump_path); return 1; }
    fwrite(g_dump.data(), 1, g_dump.size(), f);
    fclose(f);
    printf("\n===== device: PASS=%d FAIL=%d =====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
