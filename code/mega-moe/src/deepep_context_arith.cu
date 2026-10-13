// DeepEP `comm::Context` 的 host 几何算术复刻（P130-1）。
//
// 直接 include DeepEP 真实头文件（`deep_ep/common/compiled.cuh` 的常量、`math.cuh` 的
// `advance_ptr`），把 `csrc/kernels/comm/context.cpp` 里那几段**纯算术**逐条重写：
//   1. 对齐/上限常量（kNumAllocationAlignmentBytes 等）
//   2. 物理域 / 逻辑域分解（context.cpp:166-183）
//   3. GIN signal 预算 ginSignalCount = num_ranks + 4（context.cpp:140）与连接类型选择（:141-146）
//   4. 对称缓冲的字节偏移布局：workspace / buffer / RDMA storage windows（context.cpp:200-213）
//
// 只用到 __host__ constexpr 与 advance_ptr（`__host__ __device__`），不需要 GPU；
// 与 Python 侧独立复刻对拍时，本程序输出 KEY=VALUE 供解析。

#include <cstdio>
#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>

using namespace deep_ep;

static void print_domain(int num_ranks, int num_nvl_ranks, bool hybrid, int rank_idx) {
    const int num_rdma_ranks = num_ranks / num_nvl_ranks;
    const int nvl_rank_idx = rank_idx % num_nvl_ranks;
    const int rdma_rank_idx = rank_idx / num_nvl_ranks;
    const bool identity_ok = (rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);

    int num_scaleout_ranks, num_scaleup_ranks, scaleout_rank_idx, scaleup_rank_idx;
    if (hybrid) {
        num_scaleout_ranks = num_rdma_ranks; num_scaleup_ranks = num_nvl_ranks;
        scaleout_rank_idx = rdma_rank_idx;   scaleup_rank_idx = nvl_rank_idx;
    } else {
        num_scaleout_ranks = 1; num_scaleup_ranks = num_ranks;
        scaleout_rank_idx = 0;  scaleup_rank_idx = rank_idx;
    }
    const bool is_scaleup_nvlink = (num_scaleup_ranks == num_nvl_ranks);
    const int gin_signal_count = num_ranks + 2 * 2;

    printf("DOMAIN ranks=%d nvl=%d hybrid=%d rank=%d "
           "num_rdma=%d nvl_idx=%d rdma_idx=%d identity=%d "
           "scaleout=%d scaleup=%d scaleout_idx=%d scaleup_idx=%d "
           "is_scaleup_nvlink=%d gin_signal_count=%d\n",
           num_ranks, num_nvl_ranks, (int) hybrid, rank_idx,
           num_rdma_ranks, nvl_rank_idx, rdma_rank_idx, (int) identity_ok,
           num_scaleout_ranks, num_scaleup_ranks, scaleout_rank_idx, scaleup_rank_idx,
           (int) is_scaleup_nvlink, gin_signal_count);
}

int main() {
    printf("CONST kNumTMAAlignmentBytes=%d\n", kNumTMAAlignmentBytes);
    printf("CONST kNumRDMAAlignmentBytes=%d\n", kNumRDMAAlignmentBytes);
    printf("CONST kNumAllocationAlignmentBytes=%d\n", kNumAllocationAlignmentBytes);
    printf("CONST kNumMaxRanks=%d\n", kNumMaxRanks);
    printf("CONST kNumMaxQPs=%d\n", kNumMaxQPs);
    printf("CONST kDefaultQPDepth=%d\n", kDefaultQPDepth);
    printf("CONST sizeof_topk_idx_t=%zu\n", sizeof(topk_idx_t));

    // advance_ptr：与 context.cpp:200-213 同构的串行加法（无空洞）
    {
        char base[8192];
        void* p = base;
        void* workspace = math::advance_ptr<void>(p, 0);              // context.cpp:200 (offset 0)
        void* buffer = math::advance_ptr<void>(workspace, 1024);      // :201 buffer = workspace + num_workspace_bytes
        void* storage_base = math::advance_ptr<void>(p, 4096);        // :208 base + num_sym_bytes
        int stride = 4, rank_idx = 6; int64_t storage_bytes = 256;
        void* local_storage = math::advance_ptr<void>(
            storage_base, (int64_t)(rank_idx % stride) * storage_bytes);  // :209
        printf("PTR workspace_off=%td\n", (char*) workspace - base);
        printf("PTR buffer_off=%td\n", (char*) buffer - base);
        printf("PTR storage_base_off=%td\n", (char*) storage_base - base);
        printf("PTR local_storage_off=%td\n", (char*) local_storage - base);
        for (int i = 0; i < stride; ++ i)
            printf("PTR storage_window_%d_off=%td\n", i,
                   (char*) math::advance_ptr<void>(storage_base, (int64_t) i * storage_bytes) - base);
    }

    // 域分解覆盖：物理域 (rank/nvl) × hybrid/非-hybrid × 若干 rank
    const int cases[][2] = {{8, 8}, {16, 8}, {16, 16}, {64, 8}, {256, 8}, {132, 4}};
    for (auto& c : cases) {
        int num_ranks = c[0], num_nvl = c[1];
        for (int rank : {0, 1, num_nvl - 1, num_nvl, num_ranks - 1})
            for (bool h : {true, false})
                print_domain(num_ranks, num_nvl, h, rank);
    }
    return 0;
}
