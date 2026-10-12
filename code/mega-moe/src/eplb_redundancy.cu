// P43-1 实验设备代码：DeepEP EPLB 冗余专家「权重预取 / 梯度归约」的负载均衡切块。
//
// 为什么 H100(sm90) 可跑：
//   * build_tasks（EPLBSharedMemoryLayout）与 EPWeightIterator 都**不依赖 NCCL GIN**：
//     build_tasks 只用 cub::BlockScan + smem；EPWeightIterator 只在一个地方调 gin.get_sym_ptr，
//     其余是纯整数切块逻辑。故把这两段**逐行原样抽取**（见下方 VERBATIM 标注），
//     并把 get_sym_ptr 换成一个语义等价的对称指针映射桩（offsets[dst]+ptr，与 DeepGEMM SymBuffer 一致）。
//   * 真正的 lb_prefetch_weights / lb_reduce_grads kernel（impls/ep/*.cuh）用 ncclDevComm / ncclGin，
//     本机 NCCL 2.29.7 头不提供 GIN 设备 API → 仅源码分析，不在本文件编译。
//
// 编译/运行见 harness/exp_eplb_redundancy.sh。

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

#include <cub/block/block_scan.cuh>

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <random>
#include <set>
#include <tuple>
#include <vector>

// ncclTeamTagLsa 在真实代码里来自 <nccl_device.h>；此处只作为模板标签，给个空壳。
struct ncclTeamTagLsa {};

// ===================== VERBATIM: layout/ep/eplb.cuh:14-19 =====================
namespace deep_ep::layout {

struct EPLBTask {
    int local_expert_idx;
    int dst_nvl_rank_idx;
    int redundant_expert_idx;
};
// =============================================================================

// ===================== VERBATIM: layout/ep/eplb.cuh:21-67 =====================
struct alignas(kNumTMAAlignmentBytes) EPLBSharedMemoryLayout {
    static constexpr int kNumBytes = 32 * 1024;
    static constexpr int kNumMaxTasks = (kNumBytes - sizeof(int)) / sizeof(EPLBTask);

    int num_tasks;
    EPLBTask tasks[kNumMaxTasks];

    template <int kNumRanks, int kNumLocalExperts, int kNumRedundantExperts, int kNumThreads, int kNumWorkspaceBytes>
    __forceinline__ __device__ void build_tasks(const int* redundancy_mapping, const int& rank_idx, const int& thread_idx,
                                               void* smem_workspace_ptr) {
        EP_STATIC_ASSERT(kNumLocalExperts > 0, "Invalid number of local experts");

        // Reuse TMA storage for the block scan
        using BlockScan = cub::BlockScan<int, kNumThreads>;
        EP_STATIC_ASSERT(sizeof(typename BlockScan::TempStorage) <= kNumWorkspaceBytes, "Insufficient shared-memory workspace");
        auto& smem_workspace = *static_cast<typename BlockScan::TempStorage*>(smem_workspace_ptr);

        // Compact local expert exchanges in mapping order
        int num_found_tasks = 0;
        for (int base = 0; base < kNumRanks * kNumRedundantExperts; base += kNumThreads) {
            const int i = base + thread_idx;
            const int expert_idx = i < kNumRanks * kNumRedundantExperts ? redundancy_mapping[i] : -1;
            const int is_local = expert_idx >= 0 and expert_idx / kNumLocalExperts == rank_idx;
            int task_idx, num_iter_found_tasks;
            BlockScan(smem_workspace).ExclusiveSum(is_local, task_idx, num_iter_found_tasks);
            if (is_local) {
                task_idx += num_found_tasks;
                EP_DEVICE_ASSERT(task_idx < kNumMaxTasks);
                tasks[task_idx] = {
                    .local_expert_idx = expert_idx % kNumLocalExperts,
                    .dst_nvl_rank_idx = i / kNumRedundantExperts,
                    .redundant_expert_idx = i % kNumRedundantExperts
                };
            }
            num_found_tasks += num_iter_found_tasks;
            __syncthreads();
        }

        // Publish task count
        if (ptx::get_warp_idx() == 0 and ptx::elect_one_sync())
            num_tasks = num_found_tasks;
        __syncthreads();
    }
};
// =============================================================================

// ===================== VERBATIM: layout/ep/eplb.cuh:69-87 =====================
static constexpr int kNumMaxWeightEntries = 16;

struct EPWeight {
    void* redundant_expert_weights;
    void* expert_weights;
    int64_t num_bytes_per_expert;
};

struct EPWeightList {
    EPWeight weights[kNumMaxWeightEntries] = {};

    __forceinline__ __device__ __host__ EPWeight& operator[](const int& weight_idx) {
        return weights[weight_idx];
    }

    __forceinline__ __device__ __host__ const EPWeight& operator[](const int& weight_idx) const {
        return weights[weight_idx];
    }
};
// =============================================================================

}  // namespace deep_ep::layout

// 桩：等价于 comm::NCCLGin::get_sym_ptr<ncclTeamTagLsa>（NCcl LSA 对称指针映射，
// 即 DeepGEMM layout/sym_buffer.cuh 的 offsets[dst]+ptr）。offsets 用字节偏移。
namespace deep_ep::comm {
struct NCCLGin {
    const int64_t* rank_offsets;
    __host__ __device__ __forceinline__ explicit NCCLGin(const int64_t* offsets): rank_offsets(offsets) {}

    template <typename team_t = void, typename dtype_t = void*>
    __device__ __forceinline__ dtype_t* get_sym_ptr(dtype_t* ptr, const int& dst_rank_idx) const {
        const int64_t mapped = rank_offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return reinterpret_cast<dtype_t*>(mapped);
    }
};
}  // namespace deep_ep::comm

// ===================== VERBATIM: layout/ep/eplb.cuh:89-140 =====================
// 说明：唯一改动是 `comm::NCCLGin` 已是上面的等价桩（成员/签名不变）。
namespace deep_ep::layout {

// Claim chunks in weight, chunk, task order
template <int kNumWeights, int kNumBytesPerChunk>
struct EPWeightIterator {
    const comm::NCCLGin& gin;
    const EPLBSharedMemoryLayout& smem_layout;
    const EPWeight* weights;
    const int num_tasks;
    int* chunk_idx_counter;

    // States, valid after a successful `next()`
    int num_chunk_bytes;
    void *src_ptr, *dst_ptr;

    __forceinline__ __device__ EPWeightIterator(const comm::NCCLGin& gin, const EPLBSharedMemoryLayout& smem_layout,
                                                const EPWeight* weights,
                                                int* chunk_idx_counter):
        gin(gin), smem_layout(smem_layout), weights(weights), num_tasks(smem_layout.num_tasks),
        chunk_idx_counter(chunk_idx_counter) {
    }

    __forceinline__ __device__ bool next() {
        // Claim a chunk
        int chunk_idx = atomicAdd(chunk_idx_counter, 1) - num_skipped_chunks;

        // Advance weight monotonically
        while (weight_idx < kNumWeights) {
            const int num_chunks = static_cast<int>(math::ceil_div<int64_t>(weights[weight_idx].num_bytes_per_expert, kNumBytesPerChunk)) * num_tasks;
            if (chunk_idx < num_chunks)
                break;

            chunk_idx -= num_chunks;
            num_skipped_chunks += num_chunks;
            weight_idx ++;
        }
        if (weight_idx == kNumWeights)
            return false;

        // Locate task and resolve pointers
        const auto& [redundant_expert_weights, expert_weights, num_bytes_per_expert] = weights[weight_idx];
        const auto& [local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx] = smem_layout.tasks[chunk_idx % num_tasks];
        const int64_t chunk_offset = static_cast<int64_t>(chunk_idx / num_tasks) * kNumBytesPerChunk;
        num_chunk_bytes = static_cast<int>(min(static_cast<int64_t>(kNumBytesPerChunk), num_bytes_per_expert - chunk_offset));
        src_ptr = math::advance_ptr(expert_weights, local_expert_idx * num_bytes_per_expert + chunk_offset);
        dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(math::advance_ptr(redundant_expert_weights,
            redundant_expert_idx * num_bytes_per_expert + chunk_offset), dst_nvl_rank_idx);
        return true;
    }

private:
    int weight_idx = 0, num_skipped_chunks = 0;
};
// =============================================================================

}  // namespace deep_ep::layout

using deep_ep::layout::EPLBTask;
using deep_ep::layout::EPLBSharedMemoryLayout;
using deep_ep::layout::EPWeight;
using deep_ep::layout::EPWeightIterator;
using deep_ep::comm::NCCLGin;

#define CUDA_CHECK(x)                                                      \
    do {                                                                   \
        const cudaError_t e = (x);                                         \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------- Part A: build_tasks

template <int NR, int NL, int NRD, int NT>
__global__ void build_tasks_kernel(const int* mapping, int rank_idx, EPLBTask* out, int* out_num) {
    constexpr int kWorkspaceBytes = 8192;
    extern __shared__ __align__(deep_ep::kNumTMAAlignmentBytes) int8_t smem[];
    auto& layout = *reinterpret_cast<EPLBSharedMemoryLayout*>(smem);
    auto* workspace = smem + sizeof(EPLBSharedMemoryLayout);
    layout.template build_tasks<NR, NL, NRD, NT, kWorkspaceBytes>(mapping, rank_idx, threadIdx.x, workspace);
    __syncthreads();
    if (threadIdx.x == 0) {
        *out_num = layout.num_tasks;
        for (int i = 0; i < layout.num_tasks; ++i) out[i] = layout.tasks[i];
    }
}

struct PartAConfig {
    int num_ranks, num_local, num_redundant, num_threads, rank_idx;
    std::vector<int> mapping;
};

std::vector<EPLBTask> ref_build_tasks(const PartAConfig& c) {
    std::vector<EPLBTask> out;
    for (int i = 0; i < c.num_ranks * c.num_redundant; ++i) {
        const int e = c.mapping[i];
        if (e >= 0 && e / c.num_local == c.rank_idx)
            out.push_back({e % c.num_local, i / c.num_redundant, i % c.num_redundant});
    }
    return out;
}

template <int NR, int NL, int NRD>
int run_partA_one(const PartAConfig& c, int& pass, int& fail) {
    int* d_mapping = nullptr;
    EPLBTask* d_out = nullptr;
    int* d_num = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mapping, sizeof(int) * c.mapping.size()));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(EPLBTask) * EPLBSharedMemoryLayout::kNumMaxTasks));
    CUDA_CHECK(cudaMalloc(&d_num, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mapping, c.mapping.data(), sizeof(int) * c.mapping.size(), cudaMemcpyHostToDevice));

    const int num_tasks = NR * NRD;
    const int smem = sizeof(EPLBSharedMemoryLayout) + 8192;
    switch (c.num_threads) {
        case 128: build_tasks_kernel<NR, NL, NRD, 128><<<1, 128, smem>>>(d_mapping, c.rank_idx, d_out, d_num); break;
        case 256: build_tasks_kernel<NR, NL, NRD, 256><<<1, 256, smem>>>(d_mapping, c.rank_idx, d_out, d_num); break;
        case 512: build_tasks_kernel<NR, NL, NRD, 512><<<1, 512, smem>>>(d_mapping, c.rank_idx, d_out, d_num); break;
        default: printf("bad threads\n"); return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int num = 0;
    std::vector<EPLBTask> got(EPLBSharedMemoryLayout::kNumMaxTasks);
    CUDA_CHECK(cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, sizeof(EPLBTask) * num, cudaMemcpyDeviceToHost));
    got.resize(num);
    const auto ref = ref_build_tasks(c);
    bool ok = (got.size() == ref.size());
    for (size_t i = 0; ok && i < ref.size(); ++i)
        ok = got[i].local_expert_idx == ref[i].local_expert_idx &&
             got[i].dst_nvl_rank_idx == ref[i].dst_nvl_rank_idx &&
             got[i].redundant_expert_idx == ref[i].redundant_expert_idx;
    printf("  A ranks=%d local=%d red=%d threads=%d rank=%d : tasks(code)=%d ref=%zu %s\n",
           c.num_ranks, c.num_local, c.num_redundant, c.num_threads, c.rank_idx,
           num, ref.size(), ok ? "OK" : "MISMATCH");
    if (!ok) {
        for (size_t i = 0; i < std::max(got.size(), ref.size()); ++i) {
            if (i >= got.size() || i >= ref.size() ||
                got[i].local_expert_idx != ref[i].local_expert_idx ||
                got[i].dst_nvl_rank_idx != ref[i].dst_nvl_rank_idx ||
                got[i].redundant_expert_idx != ref[i].redundant_expert_idx) {
                printf("    first diff @%zu\n", i);
                break;
            }
        }
    }
    ok ? ++pass : ++fail;
    cudaFree(d_mapping); cudaFree(d_out); cudaFree(d_num);
    return 0;
}

int run_partA() {
    printf("=== Part A: EPLBSharedMemoryLayout::build_tasks（真实设备代码，逐行抽取）===\n");
    int pass = 0, fail = 0;
    std::mt19937 rng(1234);

    // 用例 1：8 ranks × 4 local × 16 redundant，映射稠密（含 -1），多个 rank
    for (int rank = 0; rank < 8; rank += 3) {
        PartAConfig c{8, 4, 16, 256, rank, {}};
        c.mapping.resize(8 * 16);
        std::uniform_int_distribution<int> d(-1, 8 * 4 - 1);
        for (auto& v : c.mapping) v = d(rng);
        for (int nt : {128, 256, 512}) { c.num_threads = nt; if (run_partA_one<8, 4, 16>(c, pass, fail)) return 1; }
    }
    // 用例 2：映射长度 > 线程数（8×40=320 > 128），且大量 -1
    for (int rank = 0; rank < 8; rank += 2) {
        PartAConfig c{8, 4, 40, 128, rank, {}};
        c.mapping.resize(8 * 40, -1);
        std::uniform_int_distribution<int> d(0, 8 * 4 - 1);
        for (auto& v : c.mapping) if ((rng() & 3) == 0) v = d(rng);
        if (run_partA_one<8, 4, 40>(c, pass, fail)) return 1;
    }
    // 用例 3：全部为空 / 全部本地
    {
        PartAConfig c{4, 8, 8, 256, 2, std::vector<int>(4 * 8, -1)};
        if (run_partA_one<4, 8, 8>(c, pass, fail)) return 1;
        for (auto& v : c.mapping) v = 2 * 8 + 1;  // rank 2 的专家 id
        if (run_partA_one<4, 8, 8>(c, pass, fail)) return 1;
    }
    // 用例 4：16 ranks × 2 local × 32 redundant，rank 15（末 rank）
    {
        PartAConfig c{16, 2, 32, 512, 15, {}};
        c.mapping.resize(16 * 32);
        std::uniform_int_distribution<int> d(-1, 16 * 2 - 1);
        for (auto& v : c.mapping) v = d(rng);
        if (run_partA_one<16, 2, 32>(c, pass, fail)) return 1;
    }
    printf("Part A: %d PASS / %d FAIL\n\n", pass, fail);
    return fail;
}

// ---------------------------------------------------------------- Part B: EPWeightIterator

struct ChunkRec {
    int64_t src_off;  // 相对 expert_weights 基址
    int64_t dst_off;  // 相对 redundant_expert_weights 基址（含 rank 对称偏移）
    int num_bytes;
};

template <int NW>
__global__ void iterator_kernel(NCCLGin gin, const EPLBSharedMemoryLayout* smem_layout,
                                EPWeight* weights, int* counter,
                                ChunkRec* recs, int* num_recs) {
    if (deep_ep::ptx::elect_one_sync()) {
        EPWeightIterator<NW, 1024> it(gin, *smem_layout, weights, counter);
        while (it.next()) {
            const int r = atomicAdd(num_recs, 1);
            recs[r] = {reinterpret_cast<int64_t>(it.src_ptr),
                       reinterpret_cast<int64_t>(it.dst_ptr),
                       it.num_chunk_bytes};
        }
    }
}

int run_partB() {
    printf("=== Part B: EPWeightIterator::next 切块（真实设备代码，逐行抽取 + 对称指针桩）===\n");
    int fail = 0;

    // 配置：3 个权重，字节数刻意取非 chunk 整数倍以触发尾部 clamp
    constexpr int CH = 1024;
    const int num_tasks = 7;
    std::vector<EPLBTask> host_tasks(num_tasks);
    for (int i = 0; i < num_tasks; ++i)
        host_tasks[i] = {i % 3, i % 4, i};  // local_expert, dst_rank, redundant

    const std::vector<int64_t> bytes = {4096 + 500, 2048, 1024 * 3 + 7};  // 非整数倍
    const int num_weights = 3;

    // 主机侧引用：枚举全局 chunk 序号 → (weight, task, chunk_offset, num_bytes) → 偏移
    struct RefRec { int64_t src_off, dst_off; int num_bytes; };
    std::vector<RefRec> ref;
    {
        // 与设备端一致：权重 w 的 expert/redundant 基址各偏移 w*8192；
        // dst 再叠加对称偏移 rank*4096。这里只算偏移、不解引用。
        for (int w = 0; w < num_weights; ++w) {
            const int64_t B = bytes[w];
            const int64_t num_chunks = (B + CH - 1) / CH * num_tasks;
            for (int64_t c = 0; c < num_chunks; ++c) {
                const auto& t = host_tasks[c % num_tasks];
                const int64_t off = (c / num_tasks) * CH;
                const int nb = (int)std::min<int64_t>(CH, B - off);
                ref.push_back({w * 8192 + t.local_expert_idx * B + off,
                               w * 8192 + t.redundant_expert_idx * B + off + (int64_t)t.dst_nvl_rank_idx * 4096,
                               nb});
            }
        }
    }
    const int total = (int)ref.size();

    // 设备侧：为了让 src/dst 偏移可比较，把两个 buffer 分别放在已知的 0 基址上（用 cudaMalloc 返回值做基址）。
    int64_t *d_expert = nullptr, *d_red = nullptr;
    CUDA_CHECK(cudaMalloc(&d_expert, 16 * 1024));
    CUDA_CHECK(cudaMalloc(&d_red, 16 * 1024));
    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    ChunkRec* d_recs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recs, sizeof(ChunkRec) * (total + 64)));
    int* d_num = nullptr;
    CUDA_CHECK(cudaMalloc(&d_num, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_num, 0, sizeof(int)));

    // rank 对称偏移：rank r 的缓冲区比本 rank 高 r*4KiB（仅用于校验映射生效）
    int64_t* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int64_t) * 4));
    {
        std::vector<int64_t> offs(4);
        for (int r = 0; r < 4; ++r) offs[r] = (int64_t)r * 4096;
        CUDA_CHECK(cudaMemcpy(d_offsets, offs.data(), sizeof(int64_t) * 4, cudaMemcpyHostToDevice));
    }

    // weights：expert_weights[0..num_weights-1] 指向 d_expert（不同权重间用 B 偏移区分，故共用基址 + 权重序号偏移）
    std::vector<EPWeight> h_weights(num_weights);
    for (int w = 0; w < num_weights; ++w) {
        h_weights[w].expert_weights = reinterpret_cast<void*>(reinterpret_cast<int64_t>(d_expert) + w * 8192);
        h_weights[w].redundant_expert_weights = reinterpret_cast<void*>(reinterpret_cast<int64_t>(d_red) + w * 8192);
        h_weights[w].num_bytes_per_expert = bytes[w];
    }
    EPWeight* d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weights, sizeof(EPWeight) * num_weights));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), sizeof(EPWeight) * num_weights, cudaMemcpyHostToDevice));

    EPLBSharedMemoryLayout smem_layout{};
    smem_layout.num_tasks = num_tasks;
    for (int i = 0; i < num_tasks; ++i) smem_layout.tasks[i] = host_tasks[i];
    EPLBSharedMemoryLayout* d_layout = nullptr;
    CUDA_CHECK(cudaMalloc(&d_layout, sizeof(EPLBSharedMemoryLayout)));
    CUDA_CHECK(cudaMemcpy(d_layout, &smem_layout, sizeof(EPLBSharedMemoryLayout), cudaMemcpyHostToDevice));

    NCCLGin gin(d_offsets);
    // 4 个 warp 并发认领（模拟真实 kernel 的多 warp 竞争）
    iterator_kernel<num_weights><<<1, 128>>>(gin, d_layout, d_weights, d_counter, d_recs, d_num);
    CUDA_CHECK(cudaDeviceSynchronize());

    int got_num = 0, counter = 0;
    CUDA_CHECK(cudaMemcpy(&got_num, d_num, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<ChunkRec> got(got_num);
    CUDA_CHECK(cudaMemcpy(got.data(), d_recs, sizeof(ChunkRec) * got_num, cudaMemcpyDeviceToHost));

    printf("  配置: num_weights=%d num_tasks=%d bytes={%lld,%lld,%lld} chunk=%d\n",
           num_weights, num_tasks, (long long)bytes[0], (long long)bytes[1], (long long)bytes[2], CH);
    printf("  参考 chunk 总数=%d  设备认领=%d  全局 counter=%d\n", total, got_num, counter);
    bool ok = (got_num == total);

    // 用 (src_off,dst_off,num_bytes) 三元组做多重集比较（去掉各自基址 + host 占位）
    const int64_t expert_base = reinterpret_cast<int64_t>(d_expert);
    const int64_t red_base = reinterpret_cast<int64_t>(d_red);
    auto norm = [&](const ChunkRec& r, bool to_ref) {
        if (to_ref) return std::make_tuple(r.src_off, r.dst_off, r.num_bytes);
        return std::make_tuple(r.src_off - expert_base, r.dst_off - red_base, r.num_bytes);
    };
    std::multiset<std::tuple<int64_t, int64_t, int>> a, b;
    for (const auto& r : got) a.insert(norm(r, false));
    for (const auto& r : ref) b.insert(norm({r.src_off, r.dst_off, r.num_bytes}, true));
    if (a != b) {
        ok = false;
        printf("  多重集不一致：code=%zu ref=%zu\n", a.size(), b.size());
        // 打印前几个差异
        std::vector<std::tuple<int64_t, int64_t, int>> only_a, only_b;
        for (const auto& x : a) only_a.push_back(x);
        for (const auto& x : b) only_b.push_back(x);
        std::printf("    (示例) code[0]=(%lld,%lld,%d) ref[0]=(%lld,%lld,%d)\n",
                    (long long)std::get<0>(only_a[0]), (long long)std::get<1>(only_a[0]), std::get<2>(only_a[0]),
                    (long long)std::get<0>(only_b[0]), (long long)std::get<1>(only_b[0]), std::get<2>(only_b[0]));
    }
    // rank 对称偏移生效：dst_off - red_base 必须含 redundant_idx*B + rank*4096 项
    printf("  src/dst 偏移逐条与主机参考一致: %s\n", ok ? "YES" : "NO");
    // 尾部 clamp：找出 bytes 非 CH 整数倍的权重最后一块应 < CH
    printf("  (bytes[0]%%CH=%lld, 应有 %d 字节的尾块)\n",
           (long long)(bytes[0] % CH), (int)(bytes[0] % CH));
    fail += ok ? 0 : 1;

    cudaFree(d_layout); cudaFree(d_expert); cudaFree(d_red); cudaFree(d_counter); cudaFree(d_recs);
    cudaFree(d_num); cudaFree(d_offsets); cudaFree(d_weights);
    printf("Part B: %s\n\n", ok ? "PASS" : "FAIL");
    return fail;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device=%s sm_%d%d\n", prop.name, prop.major, prop.minor);

    int fail = 0;
    fail += run_partA();
    fail += run_partB();
    printf("TOTAL: %s\n", fail == 0 ? "ALL PASS" : "SOME FAILED");
    return fail;
}
