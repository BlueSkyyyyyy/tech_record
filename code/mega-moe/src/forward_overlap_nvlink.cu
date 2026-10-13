// P134-1 实验：Mega MoE **前向 step 的通信-计算重叠**——正对照（SM 预算分区）。
//
// 为什么做这个：
//   `harness/exp_forward_overlap.py`（同一篇）用**真实 grouped GEMM × NCCL a2a** 量得：把 NCCL a2a 发在
//   侧流、grouped GEMM 发在主**流，overlap_factor ≈ 1.0（几乎不重叠，见 .out.txt）——因为 NCCL 的 a2a
//   kernel 会占满 SM，把计算 kernel 挤出去。这正是 Mega MoE **不用**「NCCL 侧流 a2a」、而把 dispatch/
//   combine 做进**同一个 kernel**、按 warp 解耦（第 2/3/9 篇）的原因。
//
//   但「通信-计算重叠」本身在 H100 上**能**成立——前提是给通信一个**小 SM 预算**。本文件是那个正对照：
//   在单进程里用 8 卡 NVLink 做一个**限块数**的 peer scatter（模拟 dispatch/combine 的远端写），
//   与一个**计算受限** kernel 在两个流上并发，扫描通信块数 `B`，量出 overlap_factor 随 SM 预算的变化。
//   它对应 DeepEP 的 `get_theoretical_num_sms`（第 63 篇：通信只许占一小撮 SM）与第 65 篇的
//   「限 SM 的 copy ∥ matmul 才重叠」。
//
// 本文件只用于实验，不进入 kernel 实现。
//
// 编译：nvcc -arch=sm_90 -O3 -std=c++17 -pthread src/forward_overlap_nvlink.cu -o /tmp/forward_overlap_nvlink
// 运行：/tmp/forward_overlap_nvlink   （输出见 harness/exp_forward_overlap.out.txt）

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e), __FILE__, __LINE__);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// --------------------------------------------------------------------------------------
// 通信：每块卡把自己的 send 按「目的 rank」切块，写到各 peer 的 recv（NVLink 远端写）。
// 模拟 balanced a2a：每块卡向 world-1 个 peer 各写 blk 字节。
// --------------------------------------------------------------------------------------
__global__ void scatter_kernel(const uint4* __restrict__ send, uint4* const* __restrict__ recvs,
                               int me, int world, size_t blk_u4) {
    const size_t total = (size_t)(world - 1) * blk_u4;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int pi = (int)(idx / blk_u4);          // 0..world-2 第几个远端 peer
        const int j = (pi < me) ? pi : pi + 1;       // 实际 peer 号（跳过自己）
        const size_t off = idx - (size_t)pi * blk_u4;
        // 我把「发给 rank j 的那一块」写到 rank j 的 recv 里「来自 rank me 的那一块」
        recvs[j][(size_t)me * blk_u4 + off] = send[(size_t)j * blk_u4 + off];
    }
}

// --------------------------------------------------------------------------------------
// 计算：计算受限的 FMA 燃烧 kernel（占满 SM）。
// --------------------------------------------------------------------------------------
__global__ void burn_kernel(float* __restrict__ out, int iters) {
    float a = (float)threadIdx.x * 1e-3f + (float)blockIdx.x;
    float acc = 0.5f;
    #pragma unroll 8
    for (int i = 0; i < iters; ++i)
        acc = fmaf(acc, 1.0000001f, a);
    if (acc == 0.1234567f)                        // 永不成立，防 DCE
        out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// --------------------------------------------------------------------------------------
struct DevState {
    int dev;
    uint4* send = nullptr;
    uint4* recv = nullptr;
    uint4** d_ptr = nullptr;   // 设备端保存所有 peer 的 recv 指针
    float* burn_out = nullptr;
    cudaStream_t s_comm, s_comp;
    cudaEvent_t e0, e1;
};

static constexpr int kWorld = 8;
static constexpr size_t kRealBytes = 88ull * 1024 * 1024;          // 每 rank 的 dispatch+combine 字节
static constexpr size_t kBlkBytes = kRealBytes / kWorld;           // 每 peer 一块

int main() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    std::printf("visible GPUs = %d\n", ndev);
    if (ndev < kWorld) {
        std::printf("need >= %d GPUs; abort\n", kWorld);
        return 1;
    }
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    std::printf("SM count = %d, per-rank comm bytes = %.1f MB (%.1f MB/peer)\n",
                sm_count, kRealBytes / 1048576.0, kBlkBytes / 1048576.0);

    const size_t sbuf_u4 = (kWorld * kBlkBytes) / 16;
    const size_t blk_u4 = kBlkBytes / 16;

    std::vector<DevState> S(kWorld);
    // ---- 分配（单线程顺序）----
    for (int d = 0; d < kWorld; ++d) {
        S[d].dev = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&S[d].send, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].recv, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].burn_out, 4096 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&S[d].d_ptr), kWorld * sizeof(uint4*)));
        CUDA_CHECK(cudaMemset(S[d].send, 0x3f, kWorld * kBlkBytes));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_comm));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_comp));
        CUDA_CHECK(cudaEventCreate(&S[d].e0));
        CUDA_CHECK(cudaEventCreate(&S[d].e1));
    }
    // ---- 开启 P2P + 填 peer 指针表 ----
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        std::vector<uint4*> host_ptr(kWorld, nullptr);
        for (int j = 0; j < kWorld; ++j) {
            if (j == d)
                continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, d, j));
            if (can)
                CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
        }
        for (int j = 0; j < kWorld; ++j)
            host_ptr[j] = S[j].recv;
        CUDA_CHECK(cudaMemcpy(S[d].d_ptr, host_ptr.data(), kWorld * sizeof(uint4*),
                              cudaMemcpyHostToDevice));
    }

    // ---- 计时：把 SM 预算在通信与计算之间分区（B 个 block 给通信、余下 * 每 SM 常数 给计算）----
    const int kBlocksPerSM = 4;
    auto run_comm = [&](int d, int blocks) {
        scatter_kernel<<<blocks, 256, 0, S[d].s_comm>>>(S[d].send, S[d].d_ptr, d, kWorld, blk_u4);
    };
    // 固定**总计算量** W = (sm_count*kBlocksPerSM) * 40000 * 256 次 FMA，
    // 网格越小、每线程迭代越多 → t_comp ∝ 1/grid（模拟固定 FLOP 的 GEMM 随 SM 预算缩放）。
    const long long kTotalFMA = (long long)sm_count * kBlocksPerSM * 40000 * 256;
    auto run_burn = [&](int d, int grid) {
        int iters = (int)(kTotalFMA / ((long long)grid * 256));
        burn_kernel<<<grid, 256, 0, S[d].s_comp>>>(S[d].burn_out, iters);
    };

    struct Result { double t_comm, t_comp, t_serial, t_overlap; };
    auto measure = [&](int cblocks, int gblocks, int iters) -> Result {
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            run_comm(d, cblocks);
            run_burn(d, gblocks);
        }
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); }
        double tc = 0, tb = 0, ts = 0, to = 0;
        {   // comm alone
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_comm); }
            for (int it = 0; it < iters; ++it)
                for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); run_comm(d, cblocks); }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e1, S[d].s_comm); }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms, S[d].e0, S[d].e1); tc += ms; }
            tc = tc / kWorld / iters / 1e3;
        }
        {   // compute alone at the partitioned budget
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_comp); }
            for (int it = 0; it < iters; ++it)
                for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); run_burn(d, gblocks); }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e1, S[d].s_comp); }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms, S[d].e0, S[d].e1); tb += ms; }
            tb = tb / kWorld / iters / 1e3;
        }
        {   // serial（真串行）：先跑完 iters 次 comm，再让 s_comp 等，再跑完 iters 次 compute
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_comm); }
            for (int it = 0; it < iters; ++it)
                for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); run_comm(d, cblocks); }
            for (int d = 0; d < kWorld; ++d) {
                CUDA_CHECK(cudaSetDevice(d));
                cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                cudaEventRecord(ev, S[d].s_comm);
                cudaStreamWaitEvent(S[d].s_comp, ev, 0);
                cudaEventDestroy(ev);
            }
            for (int it = 0; it < iters; ++it)
                for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); run_burn(d, gblocks); }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e1, S[d].s_comp); }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms, S[d].e0, S[d].e1); ts += ms; }
            ts = ts / kWorld / iters / 1e3;
        }
        {   // overlap：两条流并发（先发通信、再发计算），末端 s_comp 等 s_comm 再记 e1
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_comm); }
            for (int it = 0; it < iters; ++it)
                for (int d = 0; d < kWorld; ++d) {
                    CUDA_CHECK(cudaSetDevice(d));
                    run_comm(d, cblocks);
                    run_burn(d, gblocks);
                }
            for (int d = 0; d < kWorld; ++d) {
                CUDA_CHECK(cudaSetDevice(d));
                cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                cudaEventRecord(ev, S[d].s_comm);
                cudaStreamWaitEvent(S[d].s_comp, ev, 0);
                cudaEventRecord(S[d].e1, S[d].s_comp);
                cudaEventDestroy(ev);
            }
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms, S[d].e0, S[d].e1); to += ms; }
            to = to / kWorld / iters / 1e3;
        }
        return {tc, tb, ts, to};
    };

    // 参考：全 SM 计算腿
    double t_burn_full = 0;
    {
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_comp); }
        for (int it = 0; it < 20; ++it)
            for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); run_burn(d, sm_count * kBlocksPerSM); }
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e1, S[d].s_comp); }
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); float ms; cudaEventElapsedTime(&ms, S[d].e0, S[d].e1); t_burn_full += ms; }
        t_burn_full = t_burn_full / kWorld / 20 / 1e3;
    }
    std::printf("compute leg alone at full SM (%d blocks) = %.1f us\n", sm_count * kBlocksPerSM,
                t_burn_full * 1e6);

    std::printf("\n%-7s %9s %11s %10s %10s %10s %8s %10s\n",
                "B_comm", "t_comm_us", "t_comp_us", "serial_us", "ovlp_us", "ideal_us", "ovf", "comm_GB/s");
    const int iters = 30;
    for (int B : {4, 8, 16, 32, 48, 64, 96, 120}) {
        int gblocks = (sm_count - B) * kBlocksPerSM;
        Result r = measure(B, gblocks, iters);
        double bytes = (double)(kWorld - 1) * kBlkBytes;
        double ideal = std::max(r.t_comm, r.t_comp);
        double ovf = r.t_serial / r.t_overlap;
        std::printf("%-7d %9.1f %11.1f %10.1f %10.1f %10.1f %8.3f %10.1f\n",
                    B, r.t_comm * 1e6, r.t_comp * 1e6, r.t_serial * 1e6, r.t_overlap * 1e6,
                    ideal * 1e6, ovf, bytes / r.t_comm / 1e9);
    }

    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaFree(S[d].send);
        cudaFree(S[d].recv);
        cudaFree(S[d].burn_out);
        cudaFree(S[d].d_ptr);
        cudaStreamDestroy(S[d].s_comm);
        cudaStreamDestroy(S[d].s_comp);
        cudaEventDestroy(S[d].e0);
        cudaEventDestroy(S[d].e1);
    }
    return 0;
}
