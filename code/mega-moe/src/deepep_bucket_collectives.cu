// P62-1 实验：DeepEP bucket 的 all_reduce / reduce_scatter 里**架构无关**的那部分，
// 在 H100(sm90) 上**直接 include 真实头文件、编译运行真实设备代码**。
//
// 为什么 H100(sm90) 可跑（详见 docs/72）：
//   * 六个 kernel（all_reduce/{nvlink,rdma,hybrid} + reduce_scatter/{nvlink,rdma,hybrid}）的**数据面**
//     走 NCCL 设备 API（`ncclDevComm_t` / `ncclGin` / `multimem.*`）与 SM100 的 `cvt.rs`，本机
//     NCCL 2.29.7 不提供 GIN 设备 API、sm90 ptxas 也不支持 `cvt.rs` → 那部分**仅源码分析**。
//   * 但下面这些**只踩 deep_ep/common + layout/bucket**，纯整数/位运算/`fadd4`/`fmul4`：
//       Part A  信号与工作区布局（sizeof / static_assert 预算）
//       Part B  `ChunkIterator` 覆盖 + 两种**分块→(rank,slot,generation) 映射**（all_reduce 与
//               reduce_scatter 完全不同）——直接调用真实 `layout::ChunkIterator`
//       Part C  归约语义：reduce_scatter 的**反序环累加**（与测试 rtol=0 的参考同序，可逐位对拍）
//               与 all_reduce 的环求和 + scale（`ptx::fadd4`/`fmul4` 真实设备代码）
//       Part D  bf16 reduce-scatter 的**随机舍入哈希**（architecture-independent，逐位与 tilelang
//               参考一致；真正的 `cvt.rs` 指令是 SM100 专属 → 仅分析）
//   故本文件把设备代码跑出来，dump 成文本；`harness/exp_deepep_bucket_collectives.py` 解析并
//   用**独立 host 实现**逐项对拍。
//
//   nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
//     -I<DeepEP>/deep_ep/include -I/usr/local/cuda/include/cccl \
//     deepep_bucket_collectives.cu -o /tmp/dbc && /tmp/dbc <dump>
//
// 源码坐标（~/github/DeepEP@93eb6eb，v2.5.0）：
//   layout/bucket/workspace.cuh, all_reduce.cuh, reduce_scatter.cuh, chunk.cuh
//   comm/bucket.cuh, common/compiled.cuh, common/ptx.cuh, common/utils.cuh
//   impls/bucket/all_reduce/{nvlink,rdma,hybrid}.cuh, reduce_scatter/{nvlink,rdma,hybrid}.cuh
//   csrc/kernels/bucket/{all_reduce,reduce_scatter}.hpp
//   third-party/tilelang_ops/stochastic_rounding.py

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <deep_ep/comm/bucket.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

using namespace deep_ep;

static FILE* g_out = nullptr;
#define EMIT(...) do { fprintf(g_out, __VA_ARGS__); } while (0)

// ---------------------------------------------------------------------------
// Part A: 布局几何
// ---------------------------------------------------------------------------
static void part_a_geometry() {
    EMIT("GEOM Bucket sizeof %zu off_offset %zu off_num_bytes %zu\n",
         sizeof(comm::Bucket), offsetof(comm::Bucket, offset), offsetof(comm::Bucket, num_bytes));
    EMIT("GEOM BucketList sizeof %zu kNumMaxBuckets %d\n",
         sizeof(comm::BucketList), comm::kNumMaxBuckets);
    EMIT("GEOM BarrierSignals sizeof %zu off_counter %zu off_signals %zu\n",
         sizeof(layout::BarrierSignals), offsetof(layout::BarrierSignals, nvl_barrier_counter),
         offsetof(layout::BarrierSignals, nvl_barrier_signals));
    EMIT("GEOM CopyEngineSignals sizeof %zu\n", sizeof(layout::CopyEngineSignals));
    EMIT("GEOM AllGatherSignals sizeof %zu kNumQPsPerChunk %d kNumMaxChunks %d\n",
         sizeof(layout::AllGatherSignals), layout::AllGatherSignals::kNumQPsPerChunk,
         layout::AllGatherSignals::kNumMaxChunks);
    EMIT("GEOM ReduceScatterSignals sizeof %zu kNumTotalSlots %d\n",
         sizeof(layout::ReduceScatterSignals), layout::ReduceScatterSignals::kNumTotalSlots);
    EMIT("GEOM AllReduceSignals sizeof %zu kNumTotalSlots %d\n",
         sizeof(layout::AllReduceSignals), layout::AllReduceSignals::kNumTotalSlots);
    EMIT("GEOM BucketSignals sizeof %zu\n", sizeof(layout::BucketSignals));
    EMIT("GEOM BucketWorkspace sizeof %zu off_signals %zu off_chunk_storage %zu\n",
         sizeof(layout::BucketWorkspace), offsetof(layout::BucketWorkspace, signals),
         offsetof(layout::BucketWorkspace, chunk_storage));
    EMIT("GEOM ChunkStorage sizeof %zu kNumBytes %d\n",
         sizeof(layout::ChunkStorage), layout::ChunkStorage::kNumBytes);
    EMIT("GEOM kNumMaxContexts %d kNumMaxSignalBytes %d kNumRDMAAlignmentBytes %d kNumMaxRDMARanks %d\n",
         kNumMaxContexts, kNumMaxSignalBytes, kNumRDMAAlignmentBytes, kNumMaxRDMARanks);
}

// ---------------------------------------------------------------------------
// Part B1: ChunkIterator 覆盖（真实设备代码）
// ---------------------------------------------------------------------------
constexpr int kMaxItersPerBlock = 64;
template <int kChunkBytes, int kStride>
__global__ void chunk_iter_kernel(const int num_buckets, const comm::BucketList buckets,
                                  int* out_ci, int64_t* out_off, int* out_bytes, int* out_cnt) {
    auto it = layout::ChunkIterator<kChunkBytes, kStride>(num_buckets, buckets, blockIdx.x);
    const int base = blockIdx.x * kMaxItersPerBlock;
    int n = 0;
    while (it.next()) {
        out_ci[base + n] = it.chunk_idx;
        out_off[base + n] = it.chunk_offset_in_bucket;
        out_bytes[base + n] = it.num_chunk_bytes;
        ++ n;
    }
    out_cnt[blockIdx.x] = n;
}

// ---------------------------------------------------------------------------
// Part B2/B3: 两种分块映射（直接照抄 kernel 里的取址公式）
// ---------------------------------------------------------------------------
// all_reduce issue（all_reduce/rdma.cuh:96-102）：
//   dst = ci % R; shard = ci / R; slot = shard % S; gen = shard / S
__global__ void ar_map_kernel(const int R, const int S, const int N,
                              int* dst, int* slot, int* gen) {
    const int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= N) return;
    const int chunk_idx_in_shard = ci / R;
    dst[ci] = ci % R;
    slot[ci] = chunk_idx_in_shard % S;
    gen[ci] = chunk_idx_in_shard / S;
}

// reduce_scatter issue（reduce_scatter/rdma.cuh:100-101）：
//   slot = ci % S; gen = ci / S
__global__ void rs_map_kernel(const int S, const int N, int* slot, int* gen) {
    const int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= N) return;
    slot[ci] = ci % S;
    gen[ci] = ci / S;
}

// ---------------------------------------------------------------------------
// Part B4: ReduceScatterSignals 的信用环寻址（get_head_ptr / get_tail_ptr，真实设备方法）
//   reduce_scatter.cuh:16-23：head = heads + rank*S + slot，tail = tails + rank*S + slot
// ---------------------------------------------------------------------------
template <int kSlots>
__global__ void rs_signal_probe_kernel(layout::ReduceScatterSignals* sig, const int R,
                                       int64_t* head_off, int64_t* tail_off) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * 3) return;
    const int rank = i / 3, slot = i % 3;
    const auto base = reinterpret_cast<int8_t*>(sig);
    head_off[i] = reinterpret_cast<int8_t*>(sig->get_head_ptr<kSlots>(rank, slot)) - base;
    tail_off[i] = reinterpret_cast<int8_t*>(sig->get_tail_ptr<kSlots>(rank, slot)) - base;
}

// ---------------------------------------------------------------------------
// Part C1: reduce_scatter 的**反序环累加**（与 test_reduce_scatter.py:84-87 的参考同序）
//   输出 shard r 的结果 = scale * ( x_rank[shard r] + x_{rank-1}[shard r] + ... )
//   kernel 侧：reduce_scatter/rdma.cuh:145(dst_ptr=本 rank 本地先行) + :175-176 反序 peer；
//              scaled only on last batch :195-201；fadd4 :191。
// ---------------------------------------------------------------------------
__global__ void rs_sum_kernel(const int R, const int num_vec, const float* in, float* out, const float scale) {
    // in 布局： [rank][num_vec] float4，out 布局 [rank][num_vec] float4
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec >= num_vec) return;
    const int rank = blockIdx.y;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    float4 v = in4[rank * num_vec + vec];
    for (int i = 1; i < R; ++ i) {
        const int p = (rank - i + 2 * R) % R;
        v = ptx::fadd4(v, in4[p * num_vec + vec]);
    }
    v = ptx::fmul4(v, scale);
    out4[rank * num_vec + vec] = v;
}

// ---------------------------------------------------------------------------
// Part C2: all_reduce（环求和）—— nvlink 的 surrogate：把 multimem_ld_reduce_add 换成显式 R 路读 + fadd4
//   nvlink all_reduce: nvlink.cuh:44-50（iterate_ld_st + multimem）+ 可选 scale
//   这里用真实 iterate_ld_st 做网格步进分块，process 函数做 R 路求和 + scale。
// ---------------------------------------------------------------------------
template <int kNumSMs, int kNumThreads>
__global__ void ar_surrogate_kernel(const int R, const int64_t num_vec, const float4* in, float4* out,
                                    const float scale) {
    const int global_thread_idx = blockIdx.x * kNumThreads + threadIdx.x;
    // in 布局： [rank][num_vec] float4；每 rank 的整份相同（全量）
    auto ld = [&](const float4* ptr, const int& lhs = 1, const int& rhs = 0) {
        // multimem_ld_reduce_add 的语义 surrogate：对 R 个 peer 的同地址求和
        float4 a = ptx::ld_with_gt_pred(ptr, lhs, rhs);
        for (int i = 1; i < R; ++ i)
            a = ptx::fadd4(a, ptx::ld_with_gt_pred(ptr + i * num_vec, lhs, rhs));
        return a;
    };
    auto st = [](float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {
        ptx::st_with_gt_pred(ptr, value, lhs, rhs);
    };
    utils::iterate_ld_st<kNumSMs, kNumThreads, 8>(
        global_thread_idx, num_vec, in, out, ld, st,
        [&](const float4& v) { return ptx::fmul4(v, scale); });
}

// ---------------------------------------------------------------------------
// Part D: bf16 reduce-scatter 的随机舍入哈希（reduce_scatter/rdma.cuh:401-412）
//   h  = x*0x5671D42B + y*0x9995E499 + z*0xACE1B8A5 + w*0xE153538D + global_float4_index
//   h ^= h>>23; h *= 0x7FEB352D; h ^= h>>16;
//   h1 = h*0x846CA68B; h1 ^= h1>>11;
//   h2 = h*0xD35A2D97; h2 ^= h2>>11;
// 与 tilelang 参考 stochastic_rounding.py:53-65 同一公式（global index = dst_rank*quads_per_rank + quad）。
// ---------------------------------------------------------------------------
__global__ void bf16_hash_kernel(const int N, const float4* in, uint32_t* h1, uint32_t* h2) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= N) return;
    const float4 value = in[q];
    const uint4 bits = reinterpret_cast<const uint4&>(value);
    uint32_t h = bits.x * 0x5671D42Bu + bits.y * 0x9995E499u +
                 bits.z * 0xACE1B8A5u + bits.w * 0xE153538Du +
                 static_cast<uint32_t>(q);
    h ^= h >> 23;
    h *= 0x7FEB352Du;
    h ^= h >> 16;
    uint32_t a = h * 0x846CA68Bu; a ^= a >> 11;
    uint32_t b = h * 0xD35A2D97u; b ^= b >> 11;
    h1[q] = a;
    h2[q] = b;
}

// ---------------------------------------------------------------------------
static void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    cudaMalloc(&p, bytes);
    return p;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/dbc_dump.txt";
    g_out = fopen(path, "w");
    if (!g_out) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    EMIT("DeepEP bucket all_reduce / reduce_scatter —— H100(sm90a) 真实设备代码 dump\n");
    EMIT("DeepEP commit 93eb6eb\n");

    part_a_geometry();

    // ---- Part B1: ChunkIterator 覆盖 ----
    {
        constexpr int kChunk = 64 * 1024;
        comm::BucketList buckets;  // 按值作 kernel 参数传入（param 空间拷贝）
        const int64_t sizes[3] = {200000, 65536, 131072};
        int64_t off = 0;
        for (int i = 0; i < 3; ++ i) { buckets[i].offset = off; buckets[i].num_bytes = sizes[i]; off += sizes[i]; }

        const int kStride = 4;
        const int kBuf = kStride * kMaxItersPerBlock;
        int* d_ci = (int*)dev_alloc(sizeof(int) * kBuf);
        int64_t* d_off = (int64_t*)dev_alloc(sizeof(int64_t) * kBuf);
        int* d_bytes = (int*)dev_alloc(sizeof(int) * kBuf);
        int* d_cnt = (int*)dev_alloc(sizeof(int) * kStride);
        chunk_iter_kernel<kChunk, kStride><<<kStride, 1>>>(3, buckets, d_ci, d_off, d_bytes, d_cnt);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "chunk kernel err: %s\n", cudaGetErrorString(err)); return 2; }
        int* ci = (int*)malloc(sizeof(int) * kBuf);
        int* bytes = (int*)malloc(sizeof(int) * kBuf);
        int64_t* o = (int64_t*)malloc(sizeof(int64_t) * kBuf);
        int cnt[kStride];
        cudaMemcpy(ci, d_ci, sizeof(int) * kBuf, cudaMemcpyDeviceToHost);
        cudaMemcpy(o, d_off, sizeof(int64_t) * kBuf, cudaMemcpyDeviceToHost);
        cudaMemcpy(bytes, d_bytes, sizeof(int) * kBuf, cudaMemcpyDeviceToHost);
        cudaMemcpy(cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost);
        EMIT("CHUNK kChunkBytes %d kStride %d num_buckets 3 sizes", kChunk, kStride);
        for (int i = 0; i < 3; ++ i) EMIT(" %lld", (long long)sizes[i]);
        EMIT("\n");
        int total = 0;
        for (int s = 0; s < kStride; ++ s) {
            EMIT("CHUNKSTRIDE %d cnt %d\n", s, cnt[s]);
            for (int n = 0; n < cnt[s]; ++ n) {
                const int idx = s * kMaxItersPerBlock + n;
                EMIT("CHUNK %d %d %lld %d\n", s, ci[idx], (long long)o[idx], bytes[idx]);
            }
            total += cnt[s];
        }
        EMIT("CHUNKTOTAL %d\n", total);
    }

    // ---- Part B2/B3: 两种映射 ----
    {
        const int R = 4;
        const int S_ar = layout::ChunkStorage::kNumBytes / (64 * 1024) / R;   // all_reduce: chunk 64KB
        const int S_rs = layout::ChunkStorage::kNumBytes / (32 * 1024) / R;   // reduce_scatter: chunk 32KB
        const int N = 2 * R * S_ar;  // 覆盖 gen 0/1 与 slot 回绕
        int* dst = (int*)dev_alloc(sizeof(int) * N);
        int* slot = (int*)dev_alloc(sizeof(int) * N);
        int* gen = (int*)dev_alloc(sizeof(int) * N);
        ar_map_kernel<<<(N + 255) / 256, 256>>>(R, S_ar, N, dst, slot, gen);
        cudaDeviceSynchronize();
        int* h_dst = (int*)malloc(sizeof(int) * N);
        int* h_slot = (int*)malloc(sizeof(int) * N);
        int* h_gen = (int*)malloc(sizeof(int) * N);
        cudaMemcpy(h_dst, dst, sizeof(int) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_slot, slot, sizeof(int) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gen, gen, sizeof(int) * N, cudaMemcpyDeviceToHost);
        EMIT("ARMAP R %d S %d N %d\n", R, S_ar, N);
        for (int i = 0; i < N; ++ i) EMIT("ARMAP %d %d %d %d\n", i, h_dst[i], h_slot[i], h_gen[i]);

        const int N2 = 2 * R * S_rs;
        int* d2 = (int*)dev_alloc(sizeof(int) * N2);
        int* g2 = (int*)dev_alloc(sizeof(int) * N2);
        rs_map_kernel<<<(N2 + 255) / 256, 256>>>(S_rs, N2, d2, g2);
        cudaDeviceSynchronize();
        int* hs2 = (int*)malloc(sizeof(int) * N2);
        int* hg2 = (int*)malloc(sizeof(int) * N2);
        cudaMemcpy(hs2, d2, sizeof(int) * N2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hg2, g2, sizeof(int) * N2, cudaMemcpyDeviceToHost);
        EMIT("RSMAP R %d S %d N %d\n", R, S_rs, N2);
        for (int i = 0; i < N2; ++ i) EMIT("RSMAP %d %d %d\n", i, hs2[i], hg2[i]);
    }

    // ---- Part B4: 信用环信号寻址 ----
    {
        constexpr int kSlots = 64;
        layout::ReduceScatterSignals* d_sig = (layout::ReduceScatterSignals*)dev_alloc(sizeof(layout::ReduceScatterSignals));
        const int R = 4;
        int64_t* d_ho = (int64_t*)dev_alloc(sizeof(int64_t) * R * 3);
        int64_t* d_to = (int64_t*)dev_alloc(sizeof(int64_t) * R * 3);
        rs_signal_probe_kernel<kSlots><<<1, 32>>>(d_sig, R, d_ho, d_to);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "signal probe err: %s\n", cudaGetErrorString(err)); return 2; }
        int64_t ho[R * 3], to[R * 3];
        cudaMemcpy(ho, d_ho, sizeof(ho), cudaMemcpyDeviceToHost);
        cudaMemcpy(to, d_to, sizeof(to), cudaMemcpyDeviceToHost);
        EMIT("RSSIG kSlots %d R %d\n", kSlots, R);
        for (int i = 0; i < R * 3; ++ i) EMIT("RSSIG %d %lld %lld\n", i, (long long)ho[i], (long long)to[i]);
    }

    // ---- Part C1: reduce_scatter 反序环累加（逐位）----
    {
        const int R = 4, L = 4096;  // L = 一片（每 rank 的 shard），float 数；须 4 的倍数
        const float scale = 0.25f;
        const size_t n = (size_t)R * L;
        float* h_in = (float*)malloc(sizeof(float) * n);
        // 用非二进制有理数，保证多项 fp32 相加**不可结合**（否则「顺序重要」测不出来）
        for (size_t i = 0; i < n; ++ i) {
            const uint32_t r = (uint32_t)(i * 1103515245u + 12345u);
            h_in[i] = (float)(r % 1000000u) / 1000000.0f * 100.0f - 50.0f;
        }
        float* d_in = (float*)dev_alloc(sizeof(float) * n);
        float* d_out = (float*)dev_alloc(sizeof(float) * n);
        cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);
        const int num_vec = L / 4;
        dim3 grid((num_vec + 255) / 256, R);
        rs_sum_kernel<<<grid, 256>>>(R, num_vec, d_in, d_out, scale);
        cudaDeviceSynchronize();
        float* h_out = (float*)malloc(sizeof(float) * n);
        cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);
        EMIT("RSSUM R %d L %d scale %g\n", R, L, scale);
        for (size_t i = 0; i < n; ++ i)
            EMIT("RSSUM %zu %08x\n", i, *reinterpret_cast<uint32_t*>(&h_out[i]));
        // 也把输入 dump 出去供 Python 对拍
        for (size_t i = 0; i < n; ++ i)
            EMIT("RSSUMIN %zu %08x\n", i, *reinterpret_cast<uint32_t*>(&h_in[i]));
    }

    // ---- Part C2: all_reduce 环求和 surrogate（iterate_ld_st）----
    {
        const int R = 8, kNumSMs = 4, kNumThreads = 256;
        const int64_t num_vec = 4096 / 4;  // float4 个数
        const float scale = 0.5f;
        const size_t n = (size_t)R * num_vec * 4;
        float* h_in = (float*)malloc(sizeof(float) * n);
        for (size_t i = 0; i < n; ++ i) {
            const uint32_t r = (uint32_t)(i * 22695477u + 1u);
            h_in[i] = (float)(r % 1000003u) / 1000003.0f * 200.0f - 100.0f;  // 非二进制有理数
        }
        float4* d_in = (float4*)dev_alloc(sizeof(float4) * R * num_vec);
        float4* d_out = (float4*)dev_alloc(sizeof(float4) * num_vec);
        cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);
        ar_surrogate_kernel<kNumSMs, kNumThreads><<<kNumSMs, kNumThreads>>>(R, num_vec, d_in, d_out, scale);
        cudaDeviceSynchronize();
        float* h_out = (float*)malloc(sizeof(float) * num_vec * 4);
        cudaMemcpy(h_out, d_out, sizeof(float) * num_vec * 4, cudaMemcpyDeviceToHost);
        EMIT("ARSUM R %d num_vec %lld scale %g\n", R, (long long)num_vec, scale);
        for (size_t i = 0; i < (size_t)num_vec * 4; ++ i)
            EMIT("ARSUM %zu %08x\n", i, *reinterpret_cast<uint32_t*>(&h_out[i]));
        for (size_t i = 0; i < n; ++ i)
            EMIT("ARSUMIN %zu %08x\n", i, *reinterpret_cast<uint32_t*>(&h_in[i]));
    }

    // ---- Part D: bf16 随机舍入哈希（逐位）----
    {
        const int N = 4096;
        float4* d_in = (float4*)dev_alloc(sizeof(float4) * N);
        float4* h_in = (float4*)malloc(sizeof(float4) * N);
        for (int i = 0; i < N; ++ i) {
            uint32_t* b = reinterpret_cast<uint32_t*>(&h_in[i]);
            b[0] = 0x3f800000u + (i * 2654435761u % 1000);
            b[1] = 0xbf800000u ^ (i * 40503u);
            b[2] = 0x40490fdbu + (i * 7);
            b[3] = (i * 2166136261u) | 0x3f000000u;
        }
        cudaMemcpy(d_in, h_in, sizeof(float4) * N, cudaMemcpyHostToDevice);
        uint32_t* d1 = (uint32_t*)dev_alloc(sizeof(uint32_t) * N);
        uint32_t* d2 = (uint32_t*)dev_alloc(sizeof(uint32_t) * N);
        bf16_hash_kernel<<<(N + 255) / 256, 256>>>(N, d_in, d1, d2);
        cudaDeviceSynchronize();
        uint32_t* h1 = (uint32_t*)malloc(sizeof(uint32_t) * N);
        uint32_t* h2 = (uint32_t*)malloc(sizeof(uint32_t) * N);
        cudaMemcpy(h1, d1, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2, d2, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
        EMIT("BF16H N %d\n", N);
        for (int i = 0; i < N; ++ i) {
            const uint32_t* b = reinterpret_cast<const uint32_t*>(&h_in[i]);
            EMIT("BF16H %d %08x %08x %08x %08x %08x %08x\n",
                 i, b[0], b[1], b[2], b[3], h1[i], h2[i]);
        }
    }

    fclose(g_out);
    printf("dump written to %s\n", path);
    return 0;
}
