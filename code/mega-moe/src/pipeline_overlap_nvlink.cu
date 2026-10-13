// P155-1 实验：跨 micro-batch 流水线的**正对照**——给通信一个 SM 预算，重叠立刻成立。
//
// 为什么做这个：
//   `harness/exp_pipeline_overlap.py`（同一篇）用**真实 grouped GEMM × NCCL a2a** 的 micro-batch 流水线
//   量到：即使把 `dispatch_{i+1}` 排到 `compute_i` 的时间窗里，NCCL 通信也只被藏住一小部分
//   （ovf 远低于 ideal）——因为 NCCL 的 a2a 会占满 SM、并与 GEMM 互抢 L2。这正是第 144 篇
//   「侧流 NCCL 不重叠」在**流水线维度**的复现。
//
//   但流水线重叠本身**能**成立——前提同样是给通信一个**小 SM 预算**（第 63 篇
//   `get_theoretical_num_sms`、第 144 篇正对照）。本文件在单进程 8 卡上建一条**真正的
//   D_i → K_i → C_i 依赖链**（dispatch → compute → combine），把通信腿做成**限块数**的 NVLink
//   peer scatter，扫通信块数 `B` 与 micro-batch 数 `M`，量 pipeline / 分组串行 的 overlap factor。
//
// 本文件只用于实验，不进入 kernel 实现。
//
// 编译：nvcc -arch=sm_90 -O3 -std=c++17 -pthread src/pipeline_overlap_nvlink.cu -o /tmp/pipeline_overlap_nvlink
// 运行：/tmp/pipeline_overlap_nvlink   （输出见 harness/exp_pipeline_overlap.out.txt）

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <utility>
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

// 通信：把本 rank 的一段发送写进各 peer 的 recv（NVLink 远端写）。
__global__ void scatter_kernel(const uint4* __restrict__ send, uint4* const* __restrict__ recvs,
                               int me, int world, size_t blk_u4, size_t base_u4) {
    const size_t total = (size_t)(world - 1) * blk_u4;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int pi = (int)(idx / blk_u4);
        const int j = (pi < me) ? pi : pi + 1;
        const size_t off = idx - (size_t)pi * blk_u4;
        recvs[j][(size_t)me * blk_u4 + off] = send[(size_t)j * blk_u4 + off + base_u4];
    }
}

// 计算：计算受限的 FMA 燃烧内核。
__global__ void burn_kernel(float* __restrict__ out, int iters) {
    float a = (float)threadIdx.x * 1e-3f + (float)blockIdx.x;
    float acc = 0.5f;
    #pragma unroll 8
    for (int i = 0; i < iters; ++i)
        acc = fmaf(acc, 1.0000001f, a);
    if (acc == 0.1234567f)
        out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

static constexpr int kWorld = 8;
static constexpr size_t kCommBytesPerRank = 176ull * 1024 * 1024;   // ≈ 8 micro-batch × 22MB（与 py 侧 M=8 同量级）
static constexpr size_t kBlkBytes = kCommBytesPerRank / kWorld;     // 每 peer 总块
static constexpr int kMicroList[] = {1, 2, 4, 8};
static constexpr int kBlockList[] = {8, 16, 32, 48, 64, 96, 120};

struct DevState {
    uint4* send = nullptr;
    uint4* recv = nullptr;
    uint4** d_ptr = nullptr;
    float* burn_out = nullptr;
    cudaStream_t s_comm, s_comp, s_timing;
    cudaEvent_t e0, e1;
};

int main() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    std::printf("visible GPUs = %d\n", ndev);
    if (ndev < kWorld) { std::printf("need >= %d GPUs; abort\n", kWorld); return 1; }
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    std::printf("SM count = %d, comm bytes/rank = %.1f MB (%.1f MB/peer)\n",
                sm_count, kCommBytesPerRank / 1048576.0, kBlkBytes / 1048576.0);

    const size_t blk_u4 = kBlkBytes / 16;

    std::vector<DevState> S(kWorld);
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&S[d].send, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].recv, kWorld * kBlkBytes));
        CUDA_CHECK(cudaMalloc(&S[d].burn_out, 4096 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&S[d].d_ptr), kWorld * sizeof(uint4*)));
        CUDA_CHECK(cudaMemset(S[d].send, 0x3f, kWorld * kBlkBytes));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_comm));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_comp));
        CUDA_CHECK(cudaStreamCreate(&S[d].s_timing));
        CUDA_CHECK(cudaEventCreate(&S[d].e0));
        CUDA_CHECK(cudaEventCreate(&S[d].e1));
    }
    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        std::vector<uint4*> host_ptr(kWorld, nullptr);
        for (int j = 0; j < kWorld; ++j) {
            if (j == d) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, d, j));
            if (can) CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
        }
        for (int j = 0; j < kWorld; ++j) host_ptr[j] = S[j].recv;
        CUDA_CHECK(cudaMemcpy(S[d].d_ptr, host_ptr.data(), kWorld * sizeof(uint4*),
                              cudaMemcpyHostToDevice));
    }

    const int kBlocksPerSM = 4;
    // 固定**总计算量**，网格越小每线程迭代越多 → t_comp ∝ 1/grid。
    const long long kTotalFMA = (long long)sm_count * kBlocksPerSM * 40000 * 256;

    auto enqueue = [&](int d, int micro, int mode, int cblocks, int gblocks) {
        CUDA_CHECK(cudaSetDevice(d));
        DevState& s = S[d];
        const size_t pm = blk_u4 / micro;
        const int fma_per_micro = (int)(kTotalFMA / ((long long)std::max(gblocks, 1) * 256) / micro);

        auto D_i = [&](cudaStream_t st, int i) {
            if (cblocks > 0)
                scatter_kernel<<<cblocks, 256, 0, st>>>(s.send, s.d_ptr, d, kWorld, pm,
                                                        (size_t)i * pm * kWorld);
        };
        auto K_i = [&](cudaStream_t st, int i) {
            (void)i;
            if (gblocks > 0) burn_kernel<<<gblocks, 256, 0, st>>>(s.burn_out, fma_per_micro);
        };

        if (mode == 0) {                       // serial_micro：D_i,K_i,C_i 逐 micro 串行（单流）
            for (int i = 0; i < micro; ++i) { D_i(s.s_timing, i); K_i(s.s_timing, i); D_i(s.s_timing, i); }
        } else if (mode == 1) {                // serial_grouped：全 D、全 K、全 C（单流，无重叠）
            for (int i = 0; i < micro; ++i) D_i(s.s_timing, i);
            for (int i = 0; i < micro; ++i) K_i(s.s_timing, i);
            for (int i = 0; i < micro; ++i) D_i(s.s_timing, i);
        } else {                               // pipeline：D→K→C 用事件跨流解耦
            // comm 流先排 D_0..D_{M-1}，每个后面记一个事件
            std::vector<cudaEvent_t> ed(micro), ek(micro);
            for (int i = 0; i < micro; ++i) {
                D_i(s.s_comm, i);
                CUDA_CHECK(cudaEventCreateWithFlags(&ed[i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(ed[i], s.s_comm));
            }
            // comp 流：K_i 等 D_i（comp 流顺序保证 K_i 串行）
            for (int i = 0; i < micro; ++i) {
                CUDA_CHECK(cudaStreamWaitEvent(s.s_comp, ed[i], 0));
                K_i(s.s_comp, i);
                CUDA_CHECK(cudaEventCreateWithFlags(&ek[i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(ek[i], s.s_comp));
            }
            // comm 流回填 C_i：C_i 等 K_i
            for (int i = 0; i < micro; ++i) {
                CUDA_CHECK(cudaStreamWaitEvent(s.s_comm, ek[i], 0));
                D_i(s.s_comm, i);
            }
            for (int i = 0; i < micro; ++i) { cudaEventDestroy(ed[i]); cudaEventDestroy(ek[i]); }
        }
    };

    auto sync_all = [&]() {
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize()); }
    };

    auto timed_once = [&](int cblocks, int gblocks, int micro, int iters, int mode) -> double {
        sync_all();
        for (int d = 0; d < kWorld; ++d) { CUDA_CHECK(cudaSetDevice(d)); cudaEventRecord(S[d].e0, S[d].s_timing); }
        for (int it = 0; it < iters; ++it)
            for (int d = 0; d < kWorld; ++d) enqueue(d, micro, mode, cblocks, gblocks);
        // 让 timing 流等 comm/comp 两条流的尾部，再记 e1
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            cudaEvent_t ec, ep;
            CUDA_CHECK(cudaEventCreateWithFlags(&ec, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ep, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(ec, S[d].s_comm));
            CUDA_CHECK(cudaEventRecord(ep, S[d].s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(S[d].s_timing, ec, 0));
            CUDA_CHECK(cudaStreamWaitEvent(S[d].s_timing, ep, 0));
            CUDA_CHECK(cudaEventRecord(S[d].e1, S[d].s_timing));
            cudaEventDestroy(ec); cudaEventDestroy(ep);
        }
        sync_all();
        double t = 0;
        for (int d = 0; d < kWorld; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, S[d].e0, S[d].e1)); t += ms;
        }
        return t / kWorld / iters / 1e3;
    };

    // 重复 3 次取中位数：8 卡共享节点上会有别的自驱任务抢卡造成离群尖峰。
    auto timed = [&](int cblocks, int gblocks, int micro, int iters, int mode) -> double {
        double v[3] = {timed_once(cblocks, gblocks, micro, iters, mode),
                       timed_once(cblocks, gblocks, micro, iters, mode),
                       timed_once(cblocks, gblocks, micro, iters, mode)};
        if (v[0] > v[1]) std::swap(v[0], v[1]);
        if (v[1] > v[2]) std::swap(v[1], v[2]);
        if (v[0] > v[1]) std::swap(v[0], v[1]);
        return v[1];
    };

    std::printf("\n# NVLink 流水线正对照：固定总计算量、限通信块数 B（comm 176MB/rank）\n");
    std::printf("%-4s %-6s %10s %10s %11s %11s %9s %9s %9s\n",
                "M", "B_comm", "t_comm_us", "t_comp_us", "t_grouped", "t_pipe", "ovf_grp", "ovf_idel", "comm_GB/s");

    for (int micro : kMicroList) {
        for (int B : kBlockList) {
            int gblocks = (sm_count - B) * kBlocksPerSM;
            if (gblocks < kBlocksPerSM) continue;
            const int iters = 20;
            double t_comm = timed(B, 0, micro, iters, 1);
            double t_comp = timed(0, gblocks, micro, iters, 1);
            double t_grouped = timed(B, gblocks, micro, iters, 1);
            double t_pipe = timed(B, gblocks, micro, iters, 2);
            double ideal = std::max(t_comm, t_comp);
            double ovf_ideal = ideal > 0 ? (t_comm + t_comp) / ideal : 0.0;
            double bytes = (double)(kWorld - 1) * kBlkBytes;
            std::printf("%-4d %-6d %10.1f %10.1f %11.1f %11.1f %9.3f %9.3f %9.1f\n",
                        micro, B, t_comm * 1e6, t_comp * 1e6, t_grouped * 1e6, t_pipe * 1e6,
                        t_pipe > 0 ? t_grouped / t_pipe : 0.0, ovf_ideal,
                        t_comm > 0 ? bytes / t_comm / 1e9 : 0.0);
        }
    }

    for (int d = 0; d < kWorld; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaFree(S[d].send); cudaFree(S[d].recv); cudaFree(S[d].burn_out); cudaFree(S[d].d_ptr);
        cudaStreamDestroy(S[d].s_comm); cudaStreamDestroy(S[d].s_comp); cudaStreamDestroy(S[d].s_timing);
        cudaEventDestroy(S[d].e0); cudaEventDestroy(S[d].e1);
    }
    return 0;
}
