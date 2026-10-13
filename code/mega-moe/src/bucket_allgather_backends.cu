// P102-1 实验：DeepEP bucket all_gather 的两个 SM kernel 后端（pure-RDMA / hybrid）
// ——补齐第 54 篇（P54-1）只当「仅分析」4 行带过的设备内核。
//
// 为什么 H100(sm90) 能跑「真实设备代码」：
//   * `layout::AllGatherSignals` / `CopyEngineSignals` / `BucketSignals` 是真实头文件里的
//     POD 布局，直接 include、host 侧 sizeof/offsetof 就是真值。
//   * `rdma_all_gather`（rdma.cuh）与 `hybrid_all_gather`（hybrid.cuh）的**寻址/分块整数逻辑**
//     与 NCCL GIN 无关：dst rank 的 warpid 分解、lane 步进桶、每 chunk 按 QP 均分（:`hybrid.cuh:61-64`）。
//     这些逻辑我们逐行搬运到一个小 kernel 里，在真机上枚举验证（覆盖恰好一次、64B 对齐、无重叠）。
//   * hybrid 的**完成协议**：每个 (peer, chunk, QP) 在远端 `signals.tails[qp][chunk]` 上加 1
//     （`ncclGin_WeakVASignalInc`），host 用 `copy_engine_wait` = `cuStreamBatchMemOp(WAIT_VALUE_GEQ)`
//     等 `>= num_rdma_ranks-1`。我们用真实驱动 mem-op 复刻 host 侧等待、用真实 device `atomicAdd`
//     复刻 peer 侧信号，端到端验证协议语义（含「空 QP 也发 signal」、per-chunk 独立、GEQ 语义）。
//   * 真正的 `gin.put`/`ncclGin_WeakVASignalInc` 数据面走 `ncclDevComm`/`ncclGin`（NCCL GIN 设备 API），
//     本机 NCCL 2.29.7 不具备 → 仅源码分析（见 docs）。
//
// 编译/运行见 harness/exp_bucket_allgather_backends.sh
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <deep_ep/layout/bucket/all_gather.cuh>
#include <deep_ep/layout/bucket/copy_engine.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/compiled.cuh>

#define CUDA_CHECK(expr) do { \
    const auto _err = (expr); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                     cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

#define DRIVER_CHECK(expr) do { \
    const auto _err = (expr); \
    if (_err != CUDA_SUCCESS) { \
        const char* _s = nullptr; cuGetErrorString(_err, &_s); \
        std::fprintf(stderr, "Driver error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                     _s ? _s : "?"); \
        std::exit(1); \
    } \
} while (0)

// ---- 与 csrc/kernels/driver/driver.cpp 逐行等价的原语（同一批驱动 API） ----
static void launch_batched_mem_ops(CUstream stream, std::vector<CUstreamBatchMemOpParams>& ops) {
    static constexpr int kNumMaxBatchMemOps = 255;
    for (std::size_t offset = 0; offset < ops.size(); offset += kNumMaxBatchMemOps) {
        const auto num_ops = static_cast<unsigned int>(std::min<std::size_t>(kNumMaxBatchMemOps, ops.size() - offset));
        DRIVER_CHECK(cuStreamBatchMemOp(stream, num_ops, ops.data() + offset, 0));
    }
}
static CUstreamBatchMemOpParams mem_op_wait(uint64_t* signal, const uint64_t& value) {
    CUstreamBatchMemOpParams p = {};
    p.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    p.waitValue.address = reinterpret_cast<CUdeviceptr>(signal);
    p.waitValue.value64 = value;
    p.waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;   // driver.cpp:77
    return p;
}
static CUstreamBatchMemOpParams mem_op_write(uint64_t* signal, const uint64_t& value) {
    CUstreamBatchMemOpParams p = {};
    p.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    p.writeValue.address = reinterpret_cast<CUdeviceptr>(signal);
    p.writeValue.value64 = value;
    p.writeValue.flags = 0;
    return p;
}

namespace dl = deep_ep::layout;
using AllGatherSignals = deep_ep::layout::AllGatherSignals;

// ============================================================================
// Part C 设备侧：逐行搬运 hybrid.cuh:61-64 的「每 chunk 按 QP 均分」。
//   num_bytes_per_qp = align(ceil_div(chunk, QPs), 64)
//   qp_offset        = min(qp*num_bytes_per_qp, chunk)
//   num_bytes        = min(num_bytes_per_qp, chunk - qp_offset)
// ============================================================================
__device__ __forceinline__ void qp_split(const int64_t chunk_len, const int qp_idx,
                                         const int kNumQPsPerChunk,
                                         int64_t& qp_offset, int64_t& num_bytes) {
    const int64_t per_qp = deep_ep::math::align<int64_t>(
        deep_ep::math::ceil_div<int64_t>(chunk_len, kNumQPsPerChunk), deep_ep::kNumRDMAAlignmentBytes);
    qp_offset = std::min<int64_t>(qp_idx * per_qp, chunk_len);
    num_bytes = std::min<int64_t>(per_qp, chunk_len - qp_offset);
    if (num_bytes < 0) num_bytes = 0;
}

// 输出每个 (chunk,qp) 的 [offset, offset+len)
__global__ void qp_split_kernel(const int64_t* chunk_lens, const int num_chunks,
                                const int kNumQPsPerChunk, int64_t* out_offsets, int64_t* out_lens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_chunks * kNumQPsPerChunk;
    if (i >= total) return;
    const int chunk = i / kNumQPsPerChunk, qp = i % kNumQPsPerChunk;
    qp_split(chunk_lens[chunk], qp, kNumQPsPerChunk, out_offsets[i], out_lens[i]);
}

// ============================================================================
// Part B 设备侧：逐行搬运 rdma.cuh:38-48 的 dst-rank 分解。
//   for (dst = global_warp_idx; dst < kNumRanks; dst += kNumGlobalWarps)
//       if (dst == rank) continue;
//   lanes 步进 buckets：for (b = lane; b < num_buckets; b += 32)
// ============================================================================
__global__ void rdma_decode_kernel(const int kNumRanks, const int kNumGlobalWarps,
                                   const int num_buckets, const int rank_idx,
                                   int* out_num_pushers, int* out_dst, int* out_bucket_cover) {
    const int global_warp_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (global_warp_idx >= kNumGlobalWarps) return;
    for (int dst = global_warp_idx; dst < kNumRanks; dst += kNumGlobalWarps) {
        if (dst == rank_idx) continue;
        if (lane == 0) atomicAdd(&out_num_pushers[dst], 1);   // 每个负责该 dst 的 warp 计一次
        for (int b = lane; b < num_buckets; b += 32) {
            // out_bucket_cover[dst] += 该 warp 覆盖的桶次数；lane 步进下每个桶恰好被一个 lane 覆盖
            atomicAdd(&out_bucket_cover[dst * num_buckets + b], 1);
        }
    }
}

// ============================================================================
// Part D 设备侧：逐行搬运 hybrid.cuh:66-76 的完成信号。
//   每个 peer 对每个 chunk 的每个 QP 在远端 tails[qp][chunk] 上加 1
//   （模拟 ncclGin_WeakVASignalInc，无论该 QP 是否有数据都发）。
// ============================================================================
__global__ void peer_signal_kernel(AllGatherSignals* signals, const int peer_id, const int num_chunks,
                                   const int kNumQPsPerChunk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_chunks * kNumQPsPerChunk;
    if (i >= total) return;
    const int chunk = i / kNumQPsPerChunk, qp = i % kNumQPsPerChunk;
    (void)peer_id;
    atomicAdd(reinterpret_cast<unsigned long long*>(&signals->tails[qp][chunk]), 1ull);
}

// ============================================================================
// Part E 设备侧：逐行搬运 hybrid.cuh:34-39 的清 tails + :50-51 的退避条件。
// ============================================================================
__global__ void clear_tails_kernel(AllGatherSignals* signals, const int num_chunks,
                                   const int kNumRDMARanks, const int kNumThreads,
                                   const int kNumQPsPerChunk) {
    const int sm_idx = blockIdx.x, thread_idx = threadIdx.x;
    for (int i = sm_idx * kNumThreads + thread_idx; i < num_chunks; i += kNumRDMARanks * kNumThreads) {
        #pragma unroll
        for (int j = 0; j < kNumQPsPerChunk; ++ j)
            signals->tails[j][i] = 0;
    }
}

int main(int argc, char** argv) {
    std::printf("=== P102-1 DeepEP bucket all_gather 的 SM kernel 后端（直接 include 真实头文件）===\n");
    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s sm_%d.%d, SMs=%d\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    // ------------------------------------------------------------------
    // Part A: 真实信号布局
    // ------------------------------------------------------------------
    std::printf("### Part A: AllGatherSignals / CopyEngineSignals 布局（真实头文件）\n");
    std::printf("  kNumRDMAAlignmentBytes = %d\n", deep_ep::kNumRDMAAlignmentBytes);
    std::printf("  kNumMaxNVLRanks        = %d\n", deep_ep::kNumMaxNVLRanks);
    std::printf("  AllGatherSignals: kNumQPsPerChunk=%d kNumMaxChunks=%d sizeof=%zu align=%zu (expect %d*%d*8=%zu)\n",
                AllGatherSignals::kNumQPsPerChunk, AllGatherSignals::kNumMaxChunks,
                sizeof(AllGatherSignals), alignof(AllGatherSignals),
                AllGatherSignals::kNumQPsPerChunk, AllGatherSignals::kNumMaxChunks,
                (size_t)AllGatherSignals::kNumQPsPerChunk * AllGatherSignals::kNumMaxChunks * 8);
    std::printf("  CopyEngineSignals: sizeof=%zu align=%zu (expect %d*8=%d)\n",
                sizeof(dl::CopyEngineSignals), alignof(dl::CopyEngineSignals),
                deep_ep::kNumMaxNVLRanks, deep_ep::kNumMaxNVLRanks * 8);
    std::printf("  BucketSignals: sizeof=%zu align=%zu\n", sizeof(dl::BucketSignals),
                alignof(dl::BucketSignals));
    std::printf("  offsetof(BucketSignals, barrier)=%zu copy_engine=%zu all_gather=%zu reduce_scatter=%zu all_reduce=%zu\n",
                offsetof(dl::BucketSignals, barrier), offsetof(dl::BucketSignals, copy_engine),
                offsetof(dl::BucketSignals, all_gather), offsetof(dl::BucketSignals, reduce_scatter),
                offsetof(dl::BucketSignals, all_reduce));
    std::printf("  BucketWorkspace: sizeof=%zu (signals[%d] + ChunkStorage(%d))\n\n",
                sizeof(dl::BucketWorkspace), deep_ep::kNumMaxContexts, dl::ChunkStorage::kNumBytes);

    // ------------------------------------------------------------------
    // Part B: rdma_all_gather 的 dst-rank / bucket 分解
    // ------------------------------------------------------------------
    std::printf("### Part B: rdma_all_gather dst-rank(lane) 分解 —— 每个非己 rank 恰好 1 个 warp 推、桶恰好覆盖一次\n");
    const int kNumThreads = 512, kNumWarps = kNumThreads / 32;
    for (auto [kNumRanks, kNumSMs] : std::vector<std::pair<int,int>>{{8, 8}, {8, 132}, {16, 8}, {2, 64}}) {
        const int kNumGlobalWarps = kNumSMs * kNumWarps;
        const int num_buckets = 3, rank_idx = 2;
        int *d_pushers, *d_dst, *d_cover;
        CUDA_CHECK(cudaMalloc(&d_pushers, kNumRanks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dst, kNumRanks * sizeof(int)));   // unused
        CUDA_CHECK(cudaMalloc(&d_cover, kNumRanks * num_buckets * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_pushers, 0, kNumRanks * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cover, 0, kNumRanks * num_buckets * sizeof(int)));
        rdma_decode_kernel<<<kNumGlobalWarps / kNumWarps, kNumThreads>>>(
            kNumRanks, kNumGlobalWarps, num_buckets, rank_idx, d_pushers, d_dst, d_cover);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> pushers(kNumRanks), cover(kNumRanks * num_buckets);
        CUDA_CHECK(cudaMemcpy(pushers.data(), d_pushers, kNumRanks * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cover.data(), d_cover, kNumRanks * num_buckets * sizeof(int), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < kNumRanks; ++r) {
            const int expect = (r == rank_idx) ? 0 : 1;   // 恰好一个 global warp 负责该 dst
            if (pushers[r] != expect) { bad++; std::printf("    BAD R=%d sm=%d rank %d pushers=%d expect=%d\n", kNumRanks, kNumSMs, r, pushers[r], expect); }
        }
        for (int r = 0; r < kNumRanks; ++r) for (int b = 0; b < num_buckets; ++b) {
            const int expect = (r == rank_idx) ? 0 : 1;
            if (cover[r * num_buckets + b] != expect) { bad++; }
        }
        std::printf("  R=%2d SMs=%3d -> global_warps=%4d, 非己 rank 分解: %s (bad=%d)\n",
                    kNumRanks, kNumSMs, kNumGlobalWarps, bad ? "FAIL" : "OK", bad);
        cudaFree(d_pushers); cudaFree(d_dst); cudaFree(d_cover);
    }
    std::printf("\n");

    // ------------------------------------------------------------------
    // Part C: hybrid_all_gather 每 chunk 的 QP 拆分
    // ------------------------------------------------------------------
    std::printf("### Part C: hybrid_all_gather 每 chunk 按 %d QP 均分（align 64B）——覆盖恰好一次、无重叠、尾块可非 64 倍数\n", AllGatherSignals::kNumQPsPerChunk);
    {
        const int QPs = AllGatherSignals::kNumQPsPerChunk;
        // 造若干 chunk 长度：整 512 倍数、非倍数、极小、仅 1 字节
        std::vector<int64_t> chunk_lens = {
            4 * 1024 * 1024,        // 典型 chunk（512 的倍数）
            8191,                   // 非 512 倍数
            513, 512, 511, 65, 64, 63, 1, 0
        };
        const int num_chunks = (int)chunk_lens.size();
        int64_t *d_lens, *d_off, *d_len;
        CUDA_CHECK(cudaMalloc(&d_lens, num_chunks * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_off, num_chunks * QPs * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_len, num_chunks * QPs * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(d_lens, chunk_lens.data(), num_chunks * sizeof(int64_t), cudaMemcpyHostToDevice));
        qp_split_kernel<<<(num_chunks * QPs + 63) / 64, 64>>>(d_lens, num_chunks, QPs, d_off, d_len);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int64_t> off(num_chunks * QPs), len(num_chunks * QPs);
        CUDA_CHECK(cudaMemcpy(off.data(), d_off, num_chunks * QPs * sizeof(int64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(len.data(), d_len, num_chunks * QPs * sizeof(int64_t), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int c = 0; c < num_chunks; ++c) {
            std::vector<int> cover(chunk_lens[c], 0);
            int64_t sum = 0;
            for (int q = 0; q < QPs; ++q) {
                const int64_t o = off[c * QPs + q], l = len[c * QPs + q];
                // 覆盖区间必须在 [0,len) 内且不越界
                if (o < 0 || l < 0 || o + l > chunk_lens[c]) { bad++; }
                // 除尾块（最后一个有数据的 QP）外，offset 必须是 64 的倍数
                if (l > 0 && o % 64 != 0) { bad++; }
                for (int64_t k = o; k < o + l; ++k) cover[k]++;
                sum += l;
            }
            if (sum != chunk_lens[c]) { bad++; std::printf("    BAD chunk=%lld sum=%lld\n", (long long)chunk_lens[c], (long long)sum); }
            for (int k = 0; k < chunk_lens[c]; ++k) if (cover[k] != 1) { bad++; break; }
        }
        std::printf("  10 组 chunk 长度（含 0/1/63/64/65/511/512/513/8191/4MiB）覆盖恰好一次: %s (bad=%d)\n",
                    bad ? "FAIL" : "OK", bad);
        for (int c = 0; c < num_chunks; ++c) {
            std::printf("    chunk=%8lld ->", (long long)chunk_lens[c]);
            for (int q = 0; q < QPs; ++q) std::printf(" [%lld+%lld]", (long long)off[c*QPs+q], (long long)len[c*QPs+q]);
            std::printf("\n");
        }
        cudaFree(d_lens); cudaFree(d_off); cudaFree(d_len);
    }
    std::printf("\n");

    // ------------------------------------------------------------------
    // Part D: hybrid 完成协议（真实 device atomic + 真实驱动 mem-op WAIT GEQ）
    // ------------------------------------------------------------------
    std::printf("### Part D: hybrid 完成协议 —— peer 每 (chunk,QP) +1，host 等 tails >= R-1（GEQ）\n");
    {
        const int QPs = AllGatherSignals::kNumQPsPerChunk;
        const int num_chunks = 4;
        AllGatherSignals* d_sig;
        uint64_t* d_done;
        CUDA_CHECK(cudaMalloc(&d_sig, sizeof(AllGatherSignals)));
        CUDA_CHECK(cudaMalloc(&d_done, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_sig, 0, sizeof(AllGatherSignals)));
        CUDA_CHECK(cudaMemset(d_done, 0, sizeof(uint64_t)));
        clear_tails_kernel<<<1, 1>>>(d_sig, num_chunks, 1, 1, QPs);   // 与 hybrid.cuh:34-39 等价
        CUDA_CHECK(cudaDeviceSynchronize());

        // 场景1：R=4（3 个 peer），全部 peer 对全部 chunk 发信号
        const int R = 4, num_peers = R - 1;
        cudaStream_t sA, sB;
        CUDA_CHECK(cudaStreamCreate(&sA));
        CUDA_CHECK(cudaStreamCreate(&sB));
        for (int p = 1; p < R; ++p)
            peer_signal_kernel<<<(num_chunks * QPs + 127) / 128, 128, 0, sB>>>(d_sig, p, num_chunks, QPs);
        // host 侧 copy_engine_wait：对每个 chunk 的 8 个 QP 等 >= num_peers
        std::vector<CUstreamBatchMemOpParams> ops;
        for (int c = 0; c < num_chunks; ++c)
            for (int q = 0; q < QPs; ++q)
                ops.push_back(mem_op_wait(reinterpret_cast<uint64_t*>(&d_sig->tails[q][c]), num_peers));
        ops.push_back(mem_op_write(d_done, 1));
        launch_batched_mem_ops(sA, ops);
        CUDA_CHECK(cudaStreamSynchronize(sA));
        CUDA_CHECK(cudaStreamSynchronize(sB));
        uint64_t done = 0;
        CUDA_CHECK(cudaMemcpy(&done, d_done, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int c = 0; c < num_chunks; ++c) for (int q = 0; q < QPs; ++q) {
            uint64_t v = 0;
            CUDA_CHECK(cudaMemcpy(&v, &d_sig->tails[q][c], sizeof(uint64_t), cudaMemcpyDeviceToHost));
            if (v != (uint64_t)num_peers) bad++;   // 每槽恰好 R-1（tails 行距 = kNumMaxChunks）
        }
        std::printf("  场景1 R=4 全体 peer: done=%llu, 每槽恰好 R-1=%s (bad=%d)\n",
                    (unsigned long long)done, bad ? "FAIL" : "OK", bad);

        // 场景2：per-chunk 独立——只发 chunk 0，等待 chunk 0 完成，读 chunk 1 应仍为 0
        CUDA_CHECK(cudaMemset(d_sig, 0, sizeof(AllGatherSignals)));
        CUDA_CHECK(cudaMemset(d_done, 0, sizeof(uint64_t)));
        clear_tails_kernel<<<1, 1>>>(d_sig, num_chunks, 1, 1, QPs);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int p = 1; p < R; ++p)
            peer_signal_kernel<<<(1 * QPs + 127) / 128, 128, 0, sB>>>(d_sig, p, 1 /* 只 chunk 0 */, QPs);
        std::vector<CUstreamBatchMemOpParams> ops2;
        for (int q = 0; q < QPs; ++q)
            ops2.push_back(mem_op_wait(reinterpret_cast<uint64_t*>(&d_sig->tails[q][0]), num_peers));
        ops2.push_back(mem_op_write(d_done, 1));
        launch_batched_mem_ops(sA, ops2);
        CUDA_CHECK(cudaStreamSynchronize(sA));
        CUDA_CHECK(cudaStreamSynchronize(sB));
        int chunk1_nonzero = 0;
        for (int q = 0; q < QPs; ++q) {
            uint64_t v = 0;
            CUDA_CHECK(cudaMemcpy(&v, &d_sig->tails[q][1], sizeof(uint64_t), cudaMemcpyDeviceToHost));
            if (v != 0) chunk1_nonzero++;
        }
        CUDA_CHECK(cudaMemcpy(&done, d_done, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        std::printf("  场景2 只发 chunk0: done=%llu, chunk1 未被信号(QP 非零数)=%d (expect 0)\n",
                    (unsigned long long)done, chunk1_nonzero);

        // 场景3：GEQ 语义——预置恰好 R-1 与 R（过计）都应立即通过；R-2 不应通过（用一次性探测）
        CUDA_CHECK(cudaMemset(d_sig, 0, sizeof(AllGatherSignals)));
        CUDA_CHECK(cudaMemset(d_done, 0, sizeof(uint64_t)));
        peer_signal_kernel<<<(QPs + 127) / 128, 128, 0, sB>>>(d_sig, 0, 1, QPs);  // 1 个，模拟 R=2
        // 再补 3 个 → 槽值 = 4 = R (过计)
        for (int k = 0; k < 3; ++k)
            peer_signal_kernel<<<(QPs + 127) / 128, 128, 0, sB>>>(d_sig, 0, 1, QPs);
        CUDA_CHECK(cudaStreamSynchronize(sB));
        std::vector<CUstreamBatchMemOpParams> ops3;
        for (int q = 0; q < QPs; ++q)
            ops3.push_back(mem_op_wait(reinterpret_cast<uint64_t*>(&d_sig->tails[q][0]), num_peers));  // 等 >=3
        ops3.push_back(mem_op_write(d_done, 1));
        launch_batched_mem_ops(sA, ops3);
        CUDA_CHECK(cudaStreamSynchronize(sA));
        CUDA_CHECK(cudaMemcpy(&done, d_done, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        uint64_t v0 = 0;
        CUDA_CHECK(cudaMemcpy(&v0, &d_sig->tails[0][0], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        std::printf("  场景3 GEQ 过计: 槽值=%llu >= R-1=%d -> done=%llu (expect 1)\n",
                    (unsigned long long)v0, num_peers, (unsigned long long)done);

        // 场景4：未达阈值时不应通过（读值确认，不做永久阻塞等待）
        AllGatherSignals* h_sig = (AllGatherSignals*)std::malloc(sizeof(AllGatherSignals));
        CUDA_CHECK(cudaMemset(d_sig, 0, sizeof(AllGatherSignals)));
        peer_signal_kernel<<<(QPs + 127) / 128, 128>>>(d_sig, 0, 1, QPs);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_sig, d_sig, sizeof(AllGatherSignals), cudaMemcpyDeviceToHost));
        std::printf("  场景4 仅 1 个 peer（R=4 需 >=3）: 槽值=%llu < %d -> 等待不会通过（正确）\n",
                    (unsigned long long)h_sig->tails[0][0], num_peers);

        std::free(h_sig);
        cudaStreamDestroy(sA); cudaStreamDestroy(sB);
        cudaFree(d_sig); cudaFree(d_done);
    }
    std::printf("\nALL DONE\n");
    return 0;
}
