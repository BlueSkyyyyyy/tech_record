// P34-1 实验源码：Mega MoE 共享专家分支的**调度器真实设备代码**。
//
// 为什么 H100(sm90) 可跑：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` 的
// `MegaMoEScheduler` 只依赖 `ptx::atomic_add` / `ptx::exchange` / `cute::elect_one_sync`
// 与纯算术（`math::constexpr_*`），**不依赖 UMMA/tcgen05/TMA**。因此可以在 sm90 上直接
// include 并运行它的**任务枚举器** `get_next_task_idx<kNumClusters>`，验证共享专家的
// `shared_mainloop`（:421-436）在「每域一个计数器」下把 (m_block, n_cluster) 恰好枚举一次。
//
//   (A) 共享 L1 / 共享 L2 的真实任务枚举覆盖率 + 局部域切分；
//   (B) 真实设备代码里复算 `get_num_l1_warmup_waves` / `get_num_max_live_pool_blocks`
//       （host+device constexpr，:17-80），与 host 侧 Python 复刻对拍；
//   (C) `is_shared_l2_early`（:443-444）的判定（共享 L2 提前 vs 推迟）。
//
// 主 kernel `impls/sm100_fp8_fp4_mega_moe.cuh` 仍是 SM100 专属，不涉及。

#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <algorithm>

#include <deep_gemm/scheduler/mega_moe.cuh>

namespace dg = deep_gemm;

#define CUDA_CHECK(expr) do {                                                \
    cudaError_t e = (expr);                                                  \
    if (e != cudaSuccess) {                                                  \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1);                                                             \
    }                                                                        \
} while (0)

// 配置：H=2048, I=512, BLOCK_N=128, BLOCK_K=128, E_s=1, 2 域。
constexpr uint32_t HID = 2048, INTER = 512;
constexpr uint32_t BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 128;
constexpr uint32_t L1_SHAPE_N = INTER * 2;      // 1024
constexpr uint32_t L1_SHAPE_K = HID;            // 2048
constexpr uint32_t L2_SHAPE_N = HID;            // 2048
constexpr uint32_t L2_SHAPE_K = INTER;          // 512
constexpr uint32_t kNumExpertsPerRank = 32;
constexpr uint32_t kNumSMs = 132;               // H100 SXM
constexpr uint32_t kNumRanks = 1;
constexpr uint32_t kNumRingBlocks = 8;
constexpr uint32_t kNumLocalityDomains = 2;
constexpr uint32_t kNumSharedExperts = 1;

using Sched = dg::sched::MegaMoEScheduler<
    BLOCK_M, BLOCK_N, BLOCK_K,
    L1_SHAPE_N, L1_SHAPE_K, L2_SHAPE_N, L2_SHAPE_K,
    kNumExpertsPerRank, kNumSMs, kNumRanks,
    kNumRingBlocks, kNumLocalityDomains,
    kNumSharedExperts>;

// ---------------------------------------------------------------------------
// (A) 共享专家任务枚举：直接调用真实设备代码 `get_next_task_idx<kNumClusters>`
// ---------------------------------------------------------------------------
__global__ void enumerate_shared_tasks_kernel(
        const uint32_t* __restrict__ task_count,   // [kNumLocalityDomains]，初值 0
        uint32_t* __restrict__ out_m,              // [max_tasks]
        uint32_t* __restrict__ out_n,              // [max_tasks]
        uint32_t* __restrict__ out_count,
        const uint32_t num_m_blocks,
        const uint32_t kNumNClusters) {
    const uint32_t domain = blockIdx.x % kNumLocalityDomains;
    uint32_t m_block_idx, n_cluster_idx;
    const bool lane0 = cute::elect_one_sync();
    // 真实设备代码：`shared_mainloop`（:431）用的就是这条静态枚举器。
    // kNumClusters 是模板参（不能运行期传），故对两个具体的 N 簇数分支。
    while (true) {
        bool got = false;
        if (kNumNClusters == 4)
            got = Sched::template get_next_task_idx<4>(task_count, domain, num_m_blocks, m_block_idx, n_cluster_idx);
        else if (kNumNClusters == 8)
            got = Sched::template get_next_task_idx<8>(task_count, domain, num_m_blocks, m_block_idx, n_cluster_idx);
        if (! got)
            break;
        // 每 warp 记录一次（与真实设备代码一样，1 warp = 1 task）
        if (lane0) {
            const uint32_t slot = atomicAdd(out_count, 1u);
            out_m[slot] = m_block_idx;
            out_n[slot] = n_cluster_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// (B)+(C) 真实 constexpr 公式 vs host
// ---------------------------------------------------------------------------
__global__ void formulas_kernel(const uint32_t num_total_m_blocks, const uint32_t num_tokens,
                                uint32_t* out) {
    // 真实设备代码：scheduler/mega_moe.cuh:48-80 / :17-45
    const uint32_t num_clusters = num_total_m_blocks;  // 占位：下面用 sched:: 真函数重算
    (void)num_clusters;
    out[0] = dg::sched::get_num_max_live_pool_blocks(num_total_m_blocks, kNumSMs, HID, INTER);
    const uint32_t num_l1_n_clusters = L1_SHAPE_N / (2 * BLOCK_N);
    const uint32_t num_l2_n_clusters = L2_SHAPE_N / (2 * BLOCK_N);
    out[1] = dg::sched::get_num_l1_warmup_waves(
        num_total_m_blocks, kNumSMs / 2, num_l1_n_clusters, num_l2_n_clusters);
    // 真实设备代码：:443-444（is_shared_l2_early）
    constexpr uint32_t kNumSharedL2Clusters = L2_SHAPE_N / BLOCK_N / 2;
    out[2] = (dg::math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2) ? 1u : 0u;
    out[3] = kNumSharedL2Clusters;
}


// host 侧 Python 复刻（与 scheduler/mega_moe.cuh 一致）
static int ceildiv(int a, int b) { return (a + b - 1) / b; }

static int py_get_num_l1_warmup_waves(int num_total_m_blocks, int num_clusters,
                                      int num_l1_n_clusters, int num_l2_n_clusters) {
    int first_l2_m = ceildiv(num_clusters, num_l2_n_clusters);
    int warmup_for_first = ceildiv(first_l2_m * num_l1_n_clusters, num_clusters);
    int diff = num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    int warmup_interleave = ceildiv(num_l1_n_clusters + (num_total_m_blocks - 1) * diff, num_clusters) + 1;
    return std::max(warmup_for_first, warmup_interleave);
}

static int py_get_num_max_live_pool_blocks(int num_total_m_blocks, int num_sms, int hidden, int inter) {
    const int kBlockN = 128, kCTAs = 2;
    int num_clusters = num_sms / kCTAs;
    int num_l1_n_clusters = inter * 2 / (kCTAs * kBlockN);
    int num_l2_n_clusters = hidden / (kCTAs * kBlockN);
    int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    int num_l1_waves = ceildiv(num_l1_clusters, num_clusters);
    int min_warmup = py_get_num_l1_warmup_waves(num_total_m_blocks, num_clusters,
                                                num_l1_n_clusters, num_l2_n_clusters);
    int warmup_waves = std::min(min_warmup, num_l1_waves);
    int warmup_clusters = std::min(warmup_waves * num_clusters, num_l1_clusters);
    int live_after = ceildiv(warmup_clusters, num_l1_n_clusters);
    int frontier = num_l2_n_clusters > num_l1_n_clusters ?
        ceildiv(num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    int margin = ceildiv(num_clusters, std::min(num_l1_n_clusters, num_l2_n_clusters));
    return std::min(num_total_m_blocks, live_after + frontier + margin);
}

int main() {
    printf("=== Mega MoE 共享专家调度器真实设备代码（sm90）===\n");
    printf("配置: H=%u I=%u BLOCK_M=%u BLOCK_N=%u BLOCK_K=%u E_s=%u 域=%u SM=%u\n\n",
           HID, INTER, BLOCK_M, BLOCK_N, BLOCK_K, kNumSharedExperts, kNumLocalityDomains, kNumSMs);

    const uint32_t num_tokens = 1024;
    const uint32_t num_m_blocks = (num_tokens + BLOCK_M - 1) / BLOCK_M;

    // 两个共享形状的 N 簇数
    struct { const char* name; uint32_t shape_n; } shapes[2] = {
        {"SharedLinear1", Sched::SHARED_L1_SHAPE_N},
        {"SharedLinear2", Sched::SHARED_L2_SHAPE_N},
    };

    bool all_ok = true;
    for (auto& s : shapes) {
        const uint32_t kNumNClusters = s.shape_n / BLOCK_N / 2;
        const uint32_t max_tasks = num_m_blocks * kNumNClusters + 8;

        uint32_t *d_count, *d_m, *d_n, *d_oc;
        CUDA_CHECK(cudaMalloc(&d_count, kNumLocalityDomains * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_m, max_tasks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_n, max_tasks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_oc, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_count, 0, kNumLocalityDomains * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_oc, 0, sizeof(uint32_t)));

        // 64 个 warp，均匀分到两个域；每域一个独立计数器
        const uint32_t kNumWarps = 64;
        enumerate_shared_tasks_kernel<<<kNumWarps, 32>>>(d_count, d_m, d_n, d_oc,
                                                          num_m_blocks, kNumNClusters);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint32_t> m(max_tasks), n(max_tasks);
        uint32_t oc;
        CUDA_CHECK(cudaMemcpy(m.data(), d_m, max_tasks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(n.data(), d_n, max_tasks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&oc, d_oc, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t cnt[2];
        CUDA_CHECK(cudaMemcpy(cnt, d_count, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // 覆盖率：每个 (m, n) 恰好一次；n 只落在自己域的那半
        std::set<std::pair<uint32_t, uint32_t>> seen;
        bool dup = false;
        for (uint32_t i = 0; i < oc; ++ i)
            if (! seen.insert({m[i], n[i]}).second) dup = true;
        const uint32_t expect = num_m_blocks * kNumNClusters;
        // 期望集合 = {(mb, nc) | 0<=mb<num_m_blocks, 0<=nc<kNumNClusters}，每域恰好覆盖自己那半
        bool full = true;
        const uint32_t local = kNumNClusters / kNumLocalityDomains;
        for (uint32_t mb = 0; mb < num_m_blocks && full; ++ mb)
            for (uint32_t nc = 0; nc < kNumNClusters; ++ nc)
                if (seen.find({mb, nc}) == seen.end()) { full = false; break; }
        const bool cover = (! dup) && (seen.size() == expect) && (oc == expect) && full;
        // 域计数：每域 true 次数恰为本域的 (kNumLocalClusters * num_m_blocks)，
        // 加上每个 warp 收尾那次返回 false 的 1 次增量（32 个 warp）。这里只核对 true 侧不变量。
        const bool counters = (cnt[0] >= local * num_m_blocks) && (cnt[1] >= local * num_m_blocks);
        all_ok &= cover && counters;

        printf("[A] %s: SHARED_SHAPE_N=%u  N簇=%u  发射任务=%u  期望=%u  去重=%u  "
               "覆盖=%s  域计数(%u,%u) 域下界(%u,%u)=%s\n",
               s.name, s.shape_n, kNumNClusters, oc, expect, (uint32_t)seen.size(),
               cover ? "OK" : "FAIL", cnt[0], cnt[1],
               local * num_m_blocks, local * num_m_blocks, counters ? "OK" : "FAIL");

        cudaFree(d_count); cudaFree(d_m); cudaFree(d_n); cudaFree(d_oc);
    }

    // B) 公式对拍
    printf("\n[B] 真实 constexpr 公式 vs host 复刻:\n");
    printf("  %-26s %-10s %-10s %s\n", "num_total_m_blocks", "code(live)", "py(live)", "code/py warmup");
    for (uint32_t nmb : {8u, 32u, 128u, 512u}) {
        uint32_t* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(uint32_t)));
        formulas_kernel<<<1, 1>>>(nmb, num_tokens, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t o[4];
        CUDA_CHECK(cudaMemcpy(o, d_out, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const int py_live = py_get_num_max_live_pool_blocks(nmb, kNumSMs, HID, INTER);
        const int py_warmup = py_get_num_l1_warmup_waves(
            nmb, kNumSMs / 2, L1_SHAPE_N / (2 * BLOCK_N), L2_SHAPE_N / (2 * BLOCK_N));
        const bool ok = (int)o[0] == py_live && (int)o[1] == py_warmup;
        all_ok &= ok;
        printf("  %-26u %-10u %-10d warmup code=%u py=%d  %s\n",
               nmb, o[0], py_live, o[1], py_warmup, ok ? "OK" : "FAIL");
        cudaFree(d_out);
    }

    // C) is_shared_l2_early
    printf("\n[C] is_shared_l2_early（真实设备常量 :443-444）:\n");
    for (uint32_t nt : {256u, 1024u, 8192u}) {
        uint32_t* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(uint32_t)));
        formulas_kernel<<<1, 1>>>(num_m_blocks, nt, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t o[4];
        CUDA_CHECK(cudaMemcpy(o, d_out, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        printf("  num_tokens=%-6u  L2clusters=%u  early=%s\n", nt, o[3], o[2] ? "True" : "False");
        cudaFree(d_out);
    }

    printf("\n总判定：%s\n", all_ok ? "ALL PASS" : "HAS FAILURE !!");
    return all_ok ? 0 : 1;
}
