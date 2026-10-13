// P54-1 实验：DeepEP bucket 集合通信的 copy-engine 路径与分块层。
//
// 为什么 H100(sm90) 可跑：
//   * bucket 的 all-gather（纯 NVLink 腿）**不用任何 SM kernel**，而是用硬件 copy engine 驱动：
//       - `deep_ep::driver::copy_engine_push`  = `cudaMemcpyBatchAsync(...)`（CUDA 12.8+，本机 13.2）
//       - `deep_ep::driver::copy_engine_sync`  = `cuStreamBatchMemOp` 的 WRITE_VALUE_64 + WAIT_VALUE_64
//       - `deep_ep::driver::copy_engine_wait`  = WAIT_VALUE_GEQ
//     这些是**驱动/运行时 API，与 SM 架构无关**，故本机真机直接跑真实代码路径。
//   * `layout::ChunkIterator`（分块布局）只依赖整数运算，直接 include 真实头文件跑真实设备代码。
//   * hybrid all-gather 的 chunk 尺寸自选择、`get_theoretical_num_sms` 带宽模型是纯 host 算术，
//     由 Python 独立复刻 + 与真实源码交叉核对。
//   * 真正跨 rank 的 all_reduce/reduce_scatter 用 `ncclDevComm`/`ncclGin`/`multimem`，本机 NCCL 2.29.7
//     不提供 GIN 设备 API → 仅源码分析（本文件只验证 copy-engine 腿 + 分块布局）。
//
// 多 rank 模拟：在一个 GPU 上用 R 条 stream + R 组对称缓冲/信号，忠实复刻 DeepEP 的
// 「write 我的槽到每个 peer + wait 每个 peer 到我」握手协议（GPU 侧 mem-op，非 host 阻塞）。
//
// 编译/运行见 harness/exp_deepep_bucket.sh
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <deep_ep/layout/bucket/chunk.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
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

// ---- 与 csrc/kernels/driver/driver.cpp 逐行等价的原语 ----
// 注：这里为了不引入 JIT/lazy_driver，直接用驱动 API（与 driver.cpp 调用同一批 API）。

static void launch_batched_mem_ops(CUstream stream, std::vector<CUstreamBatchMemOpParams>& ops) {
    static constexpr int kNumMaxBatchMemOps = 255;
    for (std::size_t offset = 0; offset < ops.size(); offset += kNumMaxBatchMemOps) {
        const auto num_ops = static_cast<unsigned int>(
            std::min<std::size_t>(kNumMaxBatchMemOps, ops.size() - offset));
        DRIVER_CHECK(cuStreamBatchMemOp(stream, num_ops, ops.data() + offset, 0));
    }
}

static CUstreamBatchMemOpParams create_mem_op(
    uint64_t* signal, const uint64_t& value, const CUstreamBatchMemOpType& type,
    const CUstreamWaitValue_flags& wait_flag = CU_STREAM_WAIT_VALUE_EQ) {
    CUstreamBatchMemOpParams params = {};
    if (type == CU_STREAM_MEM_OP_WRITE_VALUE_64) {
        params.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        params.writeValue.address = reinterpret_cast<CUdeviceptr>(signal);
        params.writeValue.value64 = value;
        params.writeValue.flags = 0;
    } else {
        params.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        params.waitValue.address = reinterpret_cast<CUdeviceptr>(signal);
        params.waitValue.value64 = value;
        params.waitValue.flags = wait_flag;
    }
    return params;
}

// copy_engine_push：等价于 driver.cpp:64-77
static void copy_engine_push(CUstream stream,
                             const std::vector<void*>& dsts,
                             const std::vector<void*>& srcs,
                             const std::vector<size_t>& sizes) {
    if (dsts.empty()) return;
    cudaMemcpyAttributes attrs = {};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(),
                                    &attrs, &attr_idx, 1, stream));
}

// copy_engine_sync：等价于 driver.cpp:79-95
static void copy_engine_sync(CUstream stream, const uint64_t& seq_idx,
                             const std::vector<uint64_t*>& write_signals,
                             const std::vector<uint64_t*>& wait_signals) {
    std::vector<CUstreamBatchMemOpParams> ops(write_signals.size() + wait_signals.size());
    for (std::size_t i = 0; i < write_signals.size(); ++i)
        ops[i] = create_mem_op(write_signals[i], seq_idx, CU_STREAM_MEM_OP_WRITE_VALUE_64);
    for (std::size_t i = 0; i < wait_signals.size(); ++i)
        ops[write_signals.size() + i] = create_mem_op(
            wait_signals[i], seq_idx, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ);
    launch_batched_mem_ops(stream, ops);
}

// copy_engine_wait：等价于 driver.cpp:97-104
static void copy_engine_wait(CUstream stream, const std::vector<uint64_t*>& signals,
                             const uint64_t& target) {
    std::vector<CUstreamBatchMemOpParams> ops(signals.size());
    for (std::size_t i = 0; i < signals.size(); ++i)
        ops[i] = create_mem_op(signals[i], target, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ);
    launch_batched_mem_ops(stream, ops);
}

static uint64_t increase_seq_idx(CUstream stream, uint64_t& seq_idx) {
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) { std::fprintf(stderr, "capture unsupported\n"); std::exit(1); }
    return ++seq_idx;
}

// ---------------------------------------------------------------

static std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap; va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    ok ? ++g_pass : ++g_fail;
}

// Part 1: cudaMemcpyBatchAsync 语义
static void part1(CUstream stream) {
    std::printf("\n=== Part 1: copy_engine_push = cudaMemcpyBatchAsync（driver.cpp:64-77）===\n");
    constexpr int kNum = 64;
    constexpr int kBytes = 4096;
    uint8_t *src = nullptr, *dst = nullptr;
    CUDA_CHECK(cudaMalloc(&src, kNum * kBytes));
    CUDA_CHECK(cudaMalloc(&dst, kNum * kBytes));
    CUDA_CHECK(cudaMemset(dst, 0, kNum * kBytes));
    std::vector<uint8_t> host(kNum * kBytes);
    for (int i = 0; i < kNum * kBytes; ++i) host[i] = static_cast<uint8_t>(i * 37 + 11);
    CUDA_CHECK(cudaMemcpy(src, host.data(), host.size(), cudaMemcpyHostToDevice));

    std::vector<void*> dsts(kNum), srcs(kNum);
    std::vector<size_t> sizes(kNum, kBytes);
    for (int i = 0; i < kNum; ++i) { dsts[i] = dst + i * kBytes; srcs[i] = src + i * kBytes; }
    copy_engine_push(stream, dsts, srcs, sizes);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(kNum * kBytes);
    CUDA_CHECK(cudaMemcpy(got.data(), dst, got.size(), cudaMemcpyDeviceToHost));
    check(got == host, fmt("%d 个 D2D 批拷贝逐字节一致（cudaMemcpyFlagPreferOverlapWithCompute）", kNum));

    // sizes 可为每段不同，且允许非 4B 对齐（cudaMemcpyBatchAsync 无 4B 约束，与 cudaMemcpy 同）
    std::vector<size_t> sizes2 = {1, 2, 3, 7, 4096, 100};
    std::vector<void*> d2, s2;
    for (std::size_t i = 0; i < sizes2.size(); ++i) { d2.push_back(dst + i * 8192); s2.push_back(src + i * 8192); }
    CUDA_CHECK(cudaMemset(dst, 0, kNum * kBytes));
    copy_engine_push(stream, d2, s2, sizes2);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = true;
    for (std::size_t i = 0; i < sizes2.size(); ++i) {
        std::vector<uint8_t> a(sizes2[i]), b(sizes2[i]);
        CUDA_CHECK(cudaMemcpy(a.data(), (uint8_t*)d2[i], sizes2[i], cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), (uint8_t*)s2[i], sizes2[i], cudaMemcpyDeviceToHost));
        if (a != b) ok = false;
    }
    check(ok, "任意长度/任意对齐的段（{1,2,3,7,4096,100}）逐字节一致");

    // 空列表 no-op
    copy_engine_push(stream, {}, {}, {});
    CUDA_CHECK(cudaDeviceSynchronize());
    check(true, "空拷贝列表为 no-op（driver.cpp:66）");

    cudaFree(src); cudaFree(dst);
}

// Part 2: cuStreamBatchMemOp write/wait 顺序 + 255 批拆分
static void part2(CUstream stream) {
    std::printf("\n=== Part 2: copy_engine_sync/wait = cuStreamBatchMemOp（driver.cpp:79-104）===\n");
    uint64_t* signal = nullptr; uint64_t* data = nullptr;
    CUDA_CHECK(cudaMalloc(&signal, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&data, 4096));
    CUDA_CHECK(cudaMemset(signal, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(data, 0, 4096));

    // 顺序：写值 7 → memcpy 填 0xAB → wait>=7 → memcpy 填 0xCD
    // 第二个 memcpy 必须在 wait 之后；我们用 host 侧读回验证最终为 0xCD 且 signal=7
    {
        std::vector<uint64_t*> w = {signal}, wt = {signal};
        copy_engine_sync(stream, 7, w, {});           // 只写不等待
        uint8_t pat = 0xAB;
        // 用一次 device memset 代替 memcpy（等价作用）
        CUDA_CHECK(cudaMemsetAsync(data, pat, 4096, stream));
        copy_engine_wait(stream, wt, 7);              // 等到 signal>=7
        CUDA_CHECK(cudaMemsetAsync(data, 0xCD, 4096, stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t sv = 0; CUDA_CHECK(cudaMemcpy(&sv, signal, 8, cudaMemcpyDeviceToHost));
        std::vector<uint8_t> buf(4096); CUDA_CHECK(cudaMemcpy(buf.data(), data, 4096, cudaMemcpyDeviceToHost));
        check(sv == 7 && buf[0] == 0xCD && buf[4095] == 0xCD, fmt("WRITE_VALUE_64 后 WAIT_VALUE_GEQ 生效（signal=%llu）", (unsigned long long)sv));
    }

    // GEQ 语义：signal=100，wait>=50 应立即过；wait>=200 会挂 → 用 EQ vs GEQ 对照（EQ 用值 100）
    {
        CUDA_CHECK(cudaMemset(signal, 0, 8));
        std::vector<uint64_t*> w = {signal};
        copy_engine_sync(stream, 100, w, {});
        std::vector<uint64_t*> wt = {signal};
        copy_engine_wait(stream, wt, 50);   // GEQ 50，通过
        CUDA_CHECK(cudaDeviceSynchronize());
        check(true, "WAIT_VALUE_GEQ：100 >= 50 立即通过");
    }

    // 255 批拆分：构造 600 个写操作（6 个 slot × 100 值），验证全部落地
    {
        constexpr int kSlots = 600;
        uint64_t* slots = nullptr;
        CUDA_CHECK(cudaMalloc(&slots, kSlots * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(slots, 0, kSlots * sizeof(uint64_t)));
        std::vector<uint64_t*> w(kSlots);
        for (int i = 0; i < kSlots; ++i) w[i] = slots + i;
        // 直接调底层 launch_batched_mem_ops，触发 offset 分批
        std::vector<CUstreamBatchMemOpParams> ops(kSlots);
        for (int i = 0; i < kSlots; ++i)
            ops[i] = create_mem_op(slots + i, static_cast<uint64_t>(i + 1), CU_STREAM_MEM_OP_WRITE_VALUE_64);
        launch_batched_mem_ops(stream, ops);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint64_t> host(kSlots);
        CUDA_CHECK(cudaMemcpy(host.data(), slots, kSlots * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < kSlots; ++i) if (host[i] != static_cast<uint64_t>(i + 1)) ok = false;
        check(ok, fmt("%d 个 mem-op 超过 kNumMaxBatchMemOps(255) 被正确分批（driver.cpp:42-49）", kSlots));
        cudaFree(slots);
    }

    // 2D 屏障：两 stream 交替，验证 cross-stream 信号握手（模拟多 rank）
    {
        CUDA_CHECK(cudaMemset(signal, 0, 8));
        cudaStream_t s0, s1;
        CUDA_CHECK(cudaStreamCreate(&s0)); CUDA_CHECK(cudaStreamCreate(&s1));
        // s0 写 1，s1 等 1；s1 写 2，s0 等 2
        std::vector<uint64_t*> w1 = {signal};
        copy_engine_sync(s0, 1, w1, {});
        std::vector<uint64_t*> wt1 = {signal};
        // s1 先等待 1（会阻塞在 GPU 侧直到 s0 写完）
        copy_engine_wait(s1, wt1, 1);
        copy_engine_sync(s1, 2, w1, {});
        copy_engine_wait(s0, wt1, 2);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t sv = 0; CUDA_CHECK(cudaMemcpy(&sv, signal, 8, cudaMemcpyDeviceToHost));
        check(sv == 2, "两条 stream 的 WRITE/WAIT 交替握手（GPU 侧阻塞，非 host）");
        cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    }

    cudaFree(signal); cudaFree(data);
}

// Part 3: 纯 NVLink all-gather（host 描述符构造 all_gather.hpp:62-108 + copy-engine 序列）
// 单 GPU、R 个逻辑 rank：R 个对称缓冲 + R 个对称信号，R 条 stream 并发。
static void part3(int R, int64_t shard_bytes, const char* dump_path, bool write_dump) {
    std::printf("\n=== Part 3: 纯 NVLink all-gather（copy engine，R=%d）===\n", R);

    std::vector<CUstream> streams(R);
    std::vector<uint8_t*> buffers(R);
    std::vector<uint64_t*> signals(R);
    for (int r = 0; r < R; ++r) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&buffers[r], static_cast<size_t>(R) * shard_bytes));
        CUDA_CHECK(cudaMemset(buffers[r], 0, static_cast<size_t>(R) * shard_bytes));
        CUDA_CHECK(cudaMalloc(&signals[r], R * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(signals[r], 0, R * sizeof(uint64_t)));
    }
    // 每个 rank 的本地 shard = 全缓冲的第 r 段，填可辨识 pattern
    for (int r = 0; r < R; ++r) {
        std::vector<uint8_t> shard(shard_bytes);
        for (int64_t i = 0; i < shard_bytes; ++i) shard[i] = static_cast<uint8_t>(r * 53 + (i % 251));
        CUDA_CHECK(cudaMemcpy(buffers[r] + r * shard_bytes, shard.data(), shard_bytes, cudaMemcpyHostToDevice));
    }

    // seq_idx 是 per-context 单调计数器（BucketBuffer::seq_indices, bucket.hpp:46 / driver.cpp:14-19）。
    // 真实多 rank 里每个 rank 各自持有一个 counter，故这里给每个逻辑 rank 一份。
    std::vector<uint64_t> seq_idx(R, 0);

    // 逐行复刻 all_gather.hpp:62-108 的 host 描述符构造
    for (int r = 0; r < R; ++r) {
        CUstream stream = streams[r];
        std::vector<uint64_t*> write_signals, wait_signals;
        for (int rank_offset = 1; rank_offset < R; ++rank_offset) {
            const int peer = (r + rank_offset) % R;
            // write_signals[peer] = get_sym_ptr(signals.base, peer) + my_rank  （我在 peer 上的槽）
            write_signals.push_back(signals[peer] + r);
            // wait_signals[peer]  = signals.base + peer                 （peer 在我上的槽）
            wait_signals.push_back(signals[r] + peer);
        }
        std::vector<void*> dst_ptrs, src_ptrs;
        std::vector<size_t> sizes;
        const int64_t shard_offset = r * shard_bytes;                // offset + nvl_rank_idx * num_bytes
        for (int rank_offset = 1; rank_offset < R; ++rank_offset) {
            const int peer = (r + rank_offset) % R;
            dst_ptrs.push_back(buffers[peer] + shard_offset);        // get_sym_ptr(buffer+shard, peer)
            src_ptrs.push_back(buffers[r] + shard_offset);
            sizes.push_back(shard_bytes);
        }
        // 等待到达 → push → 再次同步（all_gather.hpp:106-108）
        copy_engine_sync(stream, increase_seq_idx(stream, seq_idx[r]), write_signals, wait_signals);
        copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
        copy_engine_sync(stream, increase_seq_idx(stream, seq_idx[r]), write_signals, wait_signals);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证：每个 rank 的缓冲现在含所有 rank 的 shard
    bool ok = true;
    for (int r = 0; r < R; ++r) {
        std::vector<uint8_t> buf(static_cast<size_t>(R) * shard_bytes);
        CUDA_CHECK(cudaMemcpy(buf.data(), buffers[r], buf.size(), cudaMemcpyDeviceToHost));
        for (int s = 0; s < R && ok; ++s)
            for (int64_t i = 0; i < shard_bytes; ++i)
                if (buf[s * shard_bytes + i] != static_cast<uint8_t>(s * 53 + (i % 251))) { ok = false; break; }
    }
    check(ok, fmt("R=%d shard=%lldKiB：每个 rank 的缓冲含全部 %d 个 shard，逐字节一致", R, (long long)(shard_bytes / 1024), R));

    if (write_dump) {
        FILE* f = std::fopen(dump_path, "w");
        if (f) {
            std::fprintf(f, "R %d\nshard_bytes %lld\nseq_idx %llu\nresult %s\n",
                         R, (long long)shard_bytes, (unsigned long long)seq_idx[R - 1], ok ? "OK" : "BAD");
            std::fclose(f);
        }
    }

    for (int r = 0; r < R; ++r) {
        cudaFree(buffers[r]); cudaFree(signals[r]); cudaStreamDestroy(streams[r]);
    }
}

// Part 5: 真·多 GPU（每 rank 一块 GPU）纯 NVLink all-gather —— copy engine + peer 内存
static void part5(int R, int64_t shard_bytes, const char* dump_path) {
    std::printf("\n=== Part 5: 真·%d-GPU all-gather（copy engine 跨 NVLink peer，all_gather.hpp:62-108）===\n", R);
    int ndev = 0; CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < R) { std::printf("  [SKIP] 设备数 %d < R %d\n", ndev, R); return; }

    std::vector<uint8_t*> buffers(R, nullptr);
    std::vector<uint64_t*> signals(R, nullptr);
    std::vector<cudaStream_t> streams(R, nullptr);
    // 分配：每个 rank 在自己的 GPU 上
    for (int r = 0; r < R; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&buffers[r], static_cast<size_t>(R) * shard_bytes));
        CUDA_CHECK(cudaMemset(buffers[r], 0, static_cast<size_t>(R) * shard_bytes));
        CUDA_CHECK(cudaMalloc(&signals[r], R * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(signals[r], 0, R * sizeof(uint64_t)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
        std::vector<uint8_t> shard(shard_bytes);
        for (int64_t i = 0; i < shard_bytes; ++i) shard[i] = static_cast<uint8_t>(r * 53 + (i % 251));
        CUDA_CHECK(cudaMemcpy(buffers[r] + r * shard_bytes, shard.data(), shard_bytes, cudaMemcpyHostToDevice));
    }
    // 开启全互联 peer access
    for (int a = 0; a < R; ++a) for (int b = 0; b < R; ++b) {
        if (a == b) continue;
        CUDA_CHECK(cudaSetDevice(a));
        int can = 0; CUDA_CHECK(cudaDeviceCanAccessPeer(&can, a, b));
        if (can) { cudaError_t e = cudaDeviceEnablePeerAccess(b, 0); (void)e; }  // already-enabled 无害
    }
    // 每个 rank 的 seq 计数器（真实部署中 per process，各自从 0 起）
    std::vector<uint64_t> seq(R, 0);
    for (int r = 0; r < R; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUstream stream = streams[r];
        std::vector<uint64_t*> write_signals, wait_signals;
        for (int off = 1; off < R; ++off) {
            const int peer = (r + off) % R;
            write_signals.push_back(signals[peer] + r);
            wait_signals.push_back(signals[r] + peer);
        }
        std::vector<void*> dst_ptrs, src_ptrs; std::vector<size_t> sizes;
        const int64_t shard_offset = r * shard_bytes;
        for (int off = 1; off < R; ++off) {
            const int peer = (r + off) % R;
            dst_ptrs.push_back(buffers[peer] + shard_offset);
            src_ptrs.push_back(buffers[r] + shard_offset);
            sizes.push_back(shard_bytes);
        }
        copy_engine_sync(stream, increase_seq_idx(stream, seq[r]), write_signals, wait_signals);
        copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
        copy_engine_sync(stream, increase_seq_idx(stream, seq[r]), write_signals, wait_signals);
    }
    for (int r = 0; r < R; ++r) { CUDA_CHECK(cudaSetDevice(r)); CUDA_CHECK(cudaDeviceSynchronize()); }

    bool ok = true;
    for (int r = 0; r < R && ok; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        std::vector<uint8_t> buf(static_cast<size_t>(R) * shard_bytes);
        CUDA_CHECK(cudaMemcpy(buf.data(), buffers[r], buf.size(), cudaMemcpyDeviceToHost));
        for (int s = 0; s < R && ok; ++s)
            for (int64_t i = 0; i < shard_bytes; ++i)
                if (buf[s * shard_bytes + i] != static_cast<uint8_t>(s * 53 + (i % 251))) { ok = false; break; }
    }
    check(ok, fmt("真 %d-GPU：每块 GPU 的缓冲都含全部 %d 个 shard，逐字节一致（跨 NVLink peer）", R, R));

    if (dump_path) {
        FILE* f = std::fopen(dump_path, "w");
        if (f) { std::fprintf(f, "R %d\nshard_bytes %lld\nresult %s\n", R, (long long)shard_bytes, ok ? "OK" : "BAD"); std::fclose(f); }
    }
    for (int r = 0; r < R; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        cudaFree(buffers[r]); cudaFree(signals[r]); cudaStreamDestroy(streams[r]);
    }
    CUDA_CHECK(cudaSetDevice(0));
}

// Part D: 真实 sizeof / offsetof（直接 include 真实布局头文件）
static void part_layout(const char* dump_path) {
    using namespace deep_ep::layout;
    std::printf("\n=== Part D(真实编译): bucket 布局 sizeof/offsetof（layout/bucket/*.cuh）===\n");
    const size_t sz_bucket = sizeof(deep_ep::comm::Bucket);
    const size_t off_num = offsetof(deep_ep::comm::Bucket, num_bytes);
    const size_t sz_barrier = sizeof(BarrierSignals);
    const size_t sz_ce = sizeof(CopyEngineSignals);
    const size_t sz_ag = sizeof(AllGatherSignals);
    const size_t sz_ar = sizeof(AllReduceSignals);
    const size_t sz_rs = sizeof(ReduceScatterSignals);
    const size_t sz_bs = sizeof(BucketSignals);
    const size_t sz_bw = sizeof(BucketWorkspace);
    const size_t sz_chunk = sizeof(ChunkStorage);
    std::printf("  Bucket=%zu (offset num_bytes=%zu)\n", sz_bucket, off_num);
    std::printf("  BarrierSignals=%zu CopyEngineSignals=%zu AllGatherSignals=%zu\n", sz_barrier, sz_ce, sz_ag);
    std::printf("  AllReduceSignals=%zu ReduceScatterSignals=%zu\n", sz_ar, sz_rs);
    std::printf("  BucketSignals=%zu  BucketWorkspace=%zu  ChunkStorage=%zu\n", sz_bs, sz_bw, sz_chunk);
    check(sz_bucket == 16 and off_num == 8, "真实 sizeof(Bucket)=16, offsetof(num_bytes)=8（comm/bucket.cuh:25-27）");
    check(sz_barrier == 64, "真实 sizeof(BarrierSignals)=64");
    check(sz_ce == 8192, "真实 sizeof(CopyEngineSignals)=8192");
    check(sz_ag == 512 * 1024, "真实 sizeof(AllGatherSignals)=512KiB");
    check(sz_ar == 128 * 1024, "真实 sizeof(AllReduceSignals)=128KiB");
    check(sz_rs == 128 * 1024, "真实 sizeof(ReduceScatterSignals)=128KiB");
    check(sz_bs == sz_barrier + sz_ce + sz_ag + sz_ar + sz_rs, "BucketSignals 无内部 padding（成员皆 64B 对齐）");
    check(sz_chunk == 256 * 1024 * 1024, "真实 sizeof(ChunkStorage)=256MiB");
    if (dump_path) {
        FILE* f = std::fopen(dump_path, "w");
        if (f) {
            std::fprintf(f, "Bucket %zu\nBucket_num_bytes_off %zu\nBarrierSignals %zu\nCopyEngineSignals %zu\n"
                            "AllGatherSignals %zu\nAllReduceSignals %zu\nReduceScatterSignals %zu\n"
                            "BucketSignals %zu\nBucketWorkspace %zu\nChunkStorage %zu\n",
                         sz_bucket, off_num, sz_barrier, sz_ce, sz_ag, sz_ar, sz_rs, sz_bs, sz_bw, sz_chunk);
            std::fclose(f);
        }
    }
}

// Part 4: 真实 ChunkIterator（layout/bucket/chunk.cuh，真实设备代码）
template <int kStride>
__global__ void chunk_iter_kernel(const int num_buckets, const int64_t* bucket_bytes,
                                  int* visited, long long* byte_sum, int* partial_count,
                                  int* num_total_chunks_out) {
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    if (start >= kStride) return;
    deep_ep::comm::BucketList list;
    for (int i = 0; i < num_buckets; ++i) list[i] = {0, bucket_bytes[i]};
    deep_ep::layout::ChunkIterator<8192, kStride> it(num_buckets, list, start);
    if (start == 0) *num_total_chunks_out = it.num_total_chunks;
    while (it.next()) {
        atomicAdd(&visited[it.chunk_idx], 1);
        atomicAdd(reinterpret_cast<unsigned long long*>(byte_sum), static_cast<unsigned long long>(it.num_chunk_bytes));
        if (it.num_chunk_bytes != 8192) atomicAdd(partial_count, 1);
    }
}

template <int kStride>
static void run_chunk_case(const char* label, const std::vector<int64_t>& buckets_bytes) {
    const int nb = static_cast<int>(buckets_bytes.size());
    int64_t* d_bytes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bytes, nb * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_bytes, buckets_bytes.data(), nb * sizeof(int64_t), cudaMemcpyHostToDevice));
    // 计算总块数
    int64_t total = 0;
    for (auto b : buckets_bytes) total += (b + 8191) / 8192;
    int* visited = nullptr; long long* byte_sum = nullptr; int* partial = nullptr; int* ntc = nullptr;
    CUDA_CHECK(cudaMalloc(&visited, total * sizeof(int)));
    CUDA_CHECK(cudaMemset(visited, 0, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&byte_sum, sizeof(long long)));
    CUDA_CHECK(cudaMemset(byte_sum, 0, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&partial, sizeof(int)));
    CUDA_CHECK(cudaMemset(partial, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ntc, sizeof(int)));
    chunk_iter_kernel<kStride><<<1, kStride>>>(nb, d_bytes, visited, byte_sum, partial, ntc);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> hvis(total); CUDA_CHECK(cudaMemcpy(hvis.data(), visited, total * sizeof(int), cudaMemcpyDeviceToHost));
    long long hsum = 0; CUDA_CHECK(cudaMemcpy(&hsum, byte_sum, sizeof(long long), cudaMemcpyDeviceToHost));
    int hpart = 0; CUDA_CHECK(cudaMemcpy(&hpart, partial, sizeof(int), cudaMemcpyDeviceToHost));
    int hntc = 0; CUDA_CHECK(cudaMemcpy(&hntc, ntc, sizeof(int), cudaMemcpyDeviceToHost));
    bool all_once = true;
    for (int i = 0; i < total; ++i) if (hvis[i] != 1) all_once = false;
    int64_t expect_sum = 0; for (auto b : buckets_bytes) expect_sum += b;
    int expect_partial = 0; for (auto b : buckets_bytes) if (b % 8192 != 0) expect_partial++;
    check(all_once && hntc == total && hsum == expect_sum && hpart == expect_partial,
          fmt("%s stride=%d nb=%d：每块恰好访问一次（%lld 块）、字节和=%lld、部分尾块=%d",
                      label, kStride, nb, (long long)total, hsum, hpart));
    cudaFree(d_bytes); cudaFree(visited); cudaFree(byte_sum); cudaFree(partial); cudaFree(ntc);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dump = argc > 1 ? argv[1] : "/tmp/deepep_bucket_ce.txt";
    std::printf("DeepEP bucket 集合通信的 copy-engine 路径与分块层实验 —— H100 (sm90a)，真实代码路径\n");
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device=%s sm=%d.%d\n", prop.name, prop.major, prop.minor);

    // 子模式：只跑某一组 Part 3 配置（以便每组独立进程 + timeout，避免单 GPU 模拟互锁拖垮整轮）
    if (argc >= 4 and std::string(argv[1]) == "--part3") {
        const int R = std::atoi(argv[2]);
        const int64_t shard = std::atoll(argv[3]);
        const char* d = argc >= 5 ? argv[4] : "/tmp/deepep_bucket_ce_p3.txt";
        part3(R, shard, d, true);
        std::printf("=== Part3 汇总：PASS=%d FAIL=%d ===\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
    // 子模式：真·多 GPU all-gather
    if (argc >= 4 and std::string(argv[1]) == "--multigpu") {
        const int R = std::atoi(argv[2]);
        const int64_t shard = std::atoll(argv[3]);
        const char* d = argc >= 5 ? argv[4] : "/tmp/deepep_bucket_ce_mg.txt";
        part5(R, shard, d);
        std::printf("=== Part5 汇总：PASS=%d FAIL=%d ===\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    part_layout("/tmp/deepep_bucket_layout.txt");

    cudaStream_t main_stream; CUDA_CHECK(cudaStreamCreate(&main_stream));
    part1(main_stream);
    part2(main_stream);
    CUDA_CHECK(cudaStreamDestroy(main_stream));

    // Part 4: ChunkIterator（真实设备代码）
    std::printf("\n=== Part 4: ChunkIterator 分块覆盖（layout/bucket/chunk.cuh:58-102，真实设备代码）===\n");
    run_chunk_case<1>("单遍", {8192 * 3 + 123, 8192 * 2});
    run_chunk_case<2>("stride=2", {8192 * 3 + 123, 8192 * 2});
    run_chunk_case<4>("stride=4", {1, 8192 * 5 - 7, 8192, 8192 * 2 + 1});
    run_chunk_case<8>("stride=8", {8192 * 17 + 5, 8192, 8192 * 7 - 1});
    run_chunk_case<128>("stride=128", {8192 * 200 + 33});

    std::printf("\n=== 汇总（Part1/2/4）：PASS=%d FAIL=%d ===\n", g_pass, g_fail);
    {
        FILE* f = std::fopen(dump, "w");
        if (f) { std::fprintf(f, "parts124 PASS %d FAIL %d\nresult %s\n", g_pass, g_fail, g_fail == 0 ? "OK" : "BAD"); std::fclose(f); }
    }
    std::printf("dump -> %s\n", dump);
    return g_fail == 0 ? 0 : 1;
}
