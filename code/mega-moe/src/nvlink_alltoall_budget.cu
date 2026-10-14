// P182-1 实验：通信腿在「多对多（all-to-all）」下的每 SM 带宽预算
//   —— 第 181 篇只量了「一条 NVLink P2P」（rank0 <-> rank1）在 8 个 SM 处饱和于 ~372 GB/s，
//   但真实 EP（dispatch/combine）里一个 rank 要和**其余 7 个 rank 同时**通信。问题：这条
//   「一条链路 ~8 SM 封顶」的结论，在多对多下是该乘以 7（56 SM 才够），还是仍封顶在
//   一个 rank 的**总** NVLink 出口（~478 GB/s），几个 SM 就打满？
//
// 背景（DeepEP SM 模型）：`get_theoretical_num_sms()`（deep_ep/buffers/ep.py:431-541）把
//   每 token 的字节数按**每 SM 流量**分解、再除以每 SM 带宽常量（envs.py:193-204 硬编码
//   180/45），据此决定 dispatch/combine 开几个 SM。它隐含假设「每 SM 带宽恒定」，等价于
//   「每 SM 的通信吞吐随 peer 数线性叠加」。本实验直接量这个假设在多对多下是否成立。
//
// 本实验在 H100(sm90) 上纯 CUDA/CUDA-runtime 实现（不碰 DeepEP/NCCL 设备 API），
//   使用 8 卡 UVA peer 访问（NV18 全互联）。为了避免占用被别的任务吃满的 GPU0，
//   driver 用 CUDA_VISIBLE_DEVICES 暴露一组空闲卡，rank0 固定为其中第 0 张。
//   Part A —— 单 peer 基线（复刻第 181 篇 Part B）：rank0 <-> 1 个 peer，扫 grid；
//   Part B —— 多 peer「每块一个 peer」：block 轮流绑到不同 peer，扫 grid，量聚合带宽；
//   Part C —— 多 peer「每块轮询所有 peer」：每个 block 依次读/写全部 peer（striped）；
//   Part D —— **全部 rank 同时**做 all-to-all（每张卡各起一份 kernel、各自引自己的 peer），
//             量「整个节点满负荷」时每个 rank 的实际带宽。
//
// 编译/运行见 harness/exp_nvlink_alltoall_budget.sh。目标卡代号敏感：一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>

#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// ---------------------------------------------------------------------------
// 远端（NVLink/UVA）读：block 轮流绑到一个 peer，读该 peer 里自己那一段
//   allptr[dev] 是第 dev 张卡的缓冲区基址；rank 自己的可见序号为 rank；
//   第 b 个 block 的 peer = (rank+1 + b % npeer) % ndev，段内偏移 = (b / npeer) * chunk
// ---------------------------------------------------------------------------
// UNROLL：每线程每次迭代发 U 个独立的 __ldcs/__stcs，给远端访存足够的 MLP，
//   逼近 TMA 深流水线的形态（单发出、无限飞的读会被远端延迟卡死在 ~15 GB/s/SM）。
template <int U>
__global__ void k_read_peer(const float4* const* __restrict__ allptr, int ndev, int rank,
                            int npeer, size_t chunk, float* __restrict__ sink) {
    const int dev = (rank + 1 + (int)(blockIdx.x % npeer)) % ndev;
    const float4* __restrict__ in = allptr[dev];
    const size_t base = (size_t)(blockIdx.x / npeer) * chunk;
    const size_t stride = blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    size_t i = threadIdx.x;
    for (; i + (U - 1) * stride < chunk; i += U * stride) {
        float4 a[U];
#pragma unroll
        for (int u = 0; u < U; ++u) a[u] = __ldcs(in + base + i + u * stride);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            acc.x += a[u].x; acc.y += a[u].y; acc.z += a[u].z; acc.w += a[u].w;
        }
    }
    for (; i < chunk; i += stride) {
        float4 a = __ldcs(in + base + i);
        acc.x += a.x; acc.y += a.y; acc.z += a.z; acc.w += a.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == -12345.678f) sink[blockIdx.x] = s;
}

template <int U>
__global__ void k_write_peer(float4* const* __restrict__ allptr, int ndev, int rank,
                             int npeer, size_t chunk) {
    const int dev = (rank + 1 + (int)(blockIdx.x % npeer)) % ndev;
    float4* __restrict__ out = allptr[dev];
    const size_t base = (size_t)(blockIdx.x / npeer) * chunk;
    const size_t stride = blockDim.x;
    const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    size_t i = threadIdx.x;
    for (; i + (U - 1) * stride < chunk; i += U * stride) {
#pragma unroll
        for (int u = 0; u < U; ++u) __stcs(out + base + i + u * stride, v);
    }
    for (; i < chunk; i += stride)
        __stcs(out + base + i, v);
}

// ---------------------------------------------------------------------------
// 远端读：每个 block 依次读「全部」peer（striped），偏移 = blockIdx.x * chunk
// ---------------------------------------------------------------------------
template <int U>
__global__ void k_read_striped(const float4* const* __restrict__ allptr, int ndev, int rank,
                               int npeer, size_t chunk, float* __restrict__ sink) {
    const size_t base = (size_t)blockIdx.x * chunk;
    const size_t stride = blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < npeer; ++p) {
        const int dev = (rank + 1 + p) % ndev;
        const float4* __restrict__ in = allptr[dev];
        size_t i = threadIdx.x;
        for (; i + (U - 1) * stride < chunk; i += U * stride) {
            float4 a[U];
#pragma unroll
            for (int u = 0; u < U; ++u) a[u] = __ldcs(in + base + i + u * stride);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                acc.x += a[u].x; acc.y += a[u].y; acc.z += a[u].z; acc.w += a[u].w;
            }
        }
        for (; i < chunk; i += stride) {
            float4 a = __ldcs(in + base + i);
            acc.x += a.x; acc.y += a.y; acc.z += a.z; acc.w += a.w;
        }
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == -12345.678f) sink[blockIdx.x] = s;
}

template <int U>
__global__ void k_write_striped(float4* const* __restrict__ allptr, int ndev, int rank,
                                int npeer, size_t chunk) {
    const size_t base = (size_t)blockIdx.x * chunk;
    const size_t stride = blockDim.x;
    const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (int p = 0; p < npeer; ++p) {
        const int dev = (rank + 1 + p) % ndev;
        float4* __restrict__ out = allptr[dev];
        size_t i = threadIdx.x;
        for (; i + (U - 1) * stride < chunk; i += U * stride) {
#pragma unroll
            for (int u = 0; u < U; ++u) __stcs(out + base + i + u * stride, v);
        }
        for (; i < chunk; i += stride)
            __stcs(out + base + i, v);
    }
}

static float timeit(std::function<void()> fn, int reps) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a));
        fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms);
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return best;
}

int main() {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const int NSM = prop.multiProcessorCount;
    printf("DEVICE name=%s sm=%d ndev=%d\n", prop.name, NSM, ndev);

    const size_t PVEC = (size_t)1 << 24;  // 每个 peer 256 MiB（2^24 个 float4）
    const size_t PBYTES = PVEC * 16;

    // ---- 每张卡一块 buffer，并开双向 peer access ----
    std::vector<float4*> base(ndev, nullptr);
    std::vector<float*>  sink(ndev, nullptr);
    for (int d = 0; d < ndev; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&base[d], PBYTES));
        CK(cudaMemset(base[d], 1, PBYTES));
        CK(cudaMalloc(&sink[d], 4096));
    }
    int peer_ok = 0;
    for (int d = 0; d < ndev; ++d) {
        CK(cudaSetDevice(d));
        for (int e = 0; e < ndev; ++e) {
            if (d == e) continue;
            cudaError_t err = cudaDeviceEnablePeerAccess(e, 0);
            if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) peer_ok++;
        }
    }
    printf("PEER enable_pairs_ok=%d (expect %d)\n", peer_ok, ndev * (ndev - 1));

    // ---- 每张卡一份「全部 base 指针」的设备数组 ----
    std::vector<const float4**> d_all(ndev, nullptr);
    std::vector<float4**>       d_all_w(ndev, nullptr);
    for (int d = 0; d < ndev; ++d) {
        std::vector<const float4*> h(ndev);
        for (int e = 0; e < ndev; ++e) h[e] = base[e];
        CK(cudaSetDevice(d));
        CK(cudaMalloc((void**)&d_all[d], sizeof(const float4*) * ndev));
        CK(cudaMemcpy((void*)d_all[d], h.data(), sizeof(const float4*) * ndev, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**)&d_all_w[d], sizeof(float4*) * ndev));
        CK(cudaMemcpy((void*)d_all_w[d], h.data(), sizeof(float4*) * ndev, cudaMemcpyHostToDevice));
    }

    const int RANK = 0;            // 可见序号 0 的卡作为发起 rank
    const int NPEER = ndev - 1;    // 其余卡为 peer

    std::vector<int> grids = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    std::sort(grids.begin(), grids.end());
    grids.erase(std::unique(grids.begin(), grids.end()), grids.end());

    CK(cudaSetDevice(RANK));

    // ------------------- Part A: 单 peer 基线（rank0 <-> 1 peer） -------------------
    printf("\n== Part A: single-peer NVLink read/write (rank0<->peer) vs #SM ==\n");
    for (int N : grids) {
        if (N > NSM) continue;
        size_t chunk = PVEC / N;
        size_t moved = (size_t)N * chunk * 16;
        float mr = timeit([&] { k_read_peer<8><<<N, 1024>>>(d_all[RANK], ndev, RANK, 1, chunk, sink[RANK]); }, 5);
        float mw = timeit([&] { k_write_peer<8><<<N, 1024>>>(d_all_w[RANK], ndev, RANK, 1, chunk); }, 5);
        printf("A_P1 N=%d readGBps=%.2f writeGBps=%.2f perSMread=%.2f\n",
               N, (double)moved / (mr / 1e3) / 1e9, (double)moved / (mw / 1e3) / 1e9,
               (double)moved / (mr / 1e3) / 1e9 / N);
    }

    // ------------------- Part B: 多 peer，每块一个 peer -------------------
    printf("\n== Part B: multi-peer, block-per-peer, all %d peers ==\n", NPEER);
    for (int N : grids) {
        if (N > NSM) continue;
        int slots = (N + NPEER - 1) / NPEER;      // 每个 peer 被分到的 block 数
        size_t chunk = PVEC / std::max(slots, 1);
        size_t moved = (size_t)N * chunk * 16;
        float mr = timeit([&] { k_read_peer<8><<<N, 1024>>>(d_all[RANK], ndev, RANK, NPEER, chunk, sink[RANK]); }, 5);
        float mw = timeit([&] { k_write_peer<8><<<N, 1024>>>(d_all_w[RANK], ndev, RANK, NPEER, chunk); }, 5);
        printf("B_MP N=%d readGBps=%.2f writeGBps=%.2f perSMread=%.2f\n",
               N, (double)moved / (mr / 1e3) / 1e9, (double)moved / (mw / 1e3) / 1e9,
               (double)moved / (mr / 1e3) / 1e9 / N);
    }

    // ------------------- Part C: 多 peer，每块轮询所有 peer（striped） -------------------
    printf("\n== Part C: multi-peer, striped per block ==\n");
    for (int N : grids) {
        if (N > NSM) continue;
        size_t chunk = PVEC / N;
        size_t moved = (size_t)N * NPEER * chunk * 16;
        float mr = timeit([&] { k_read_striped<8><<<N, 1024>>>(d_all[RANK], ndev, RANK, NPEER, chunk, sink[RANK]); }, 5);
        float mw = timeit([&] { k_write_striped<8><<<N, 1024>>>(d_all_w[RANK], ndev, RANK, NPEER, chunk); }, 5);
        printf("C_STR N=%d readGBps=%.2f writeGBps=%.2f perSMread=%.2f\n",
               N, (double)moved / (mr / 1e3) / 1e9, (double)moved / (mw / 1e3) / 1e9,
               (double)moved / (mr / 1e3) / 1e9 / N);
    }

    // ------------------- Part D: 全部 rank 同时 all-to-all -------------------
    printf("\n== Part D: simultaneous all-rank all-to-all (%d ranks) ==\n", ndev);
    {
        std::vector<cudaStream_t> st(ndev);
        std::vector<cudaEvent_t> ev0(ndev), ev1(ndev);
        for (int d = 0; d < ndev; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaStreamCreate(&st[d]));
            CK(cudaEventCreate(&ev0[d])); CK(cudaEventCreate(&ev1[d]));
        }
        for (int N : {8, 16, 32, 64, 132}) {
            if (N > NSM) continue;
            int slots = (N + NPEER - 1) / NPEER;
            size_t chunk = PVEC / std::max(slots, 1);
            size_t moved = (size_t)N * chunk * 16;   // 每个 rank 搬的字节
            // 预热
            for (int d = 0; d < ndev; ++d) {
                CK(cudaSetDevice(d));
                k_read_peer<8><<<N, 1024, 0, st[d]>>>(d_all[d], ndev, d, NPEER, chunk, sink[d]);
            }
            for (int d = 0; d < ndev; ++d) CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize());
            // 计时：所有 rank 同起同停，取各自 event 耗时里的最大值（最慢 rank）
            for (int d = 0; d < ndev; ++d) {
                CK(cudaSetDevice(d));
                CK(cudaEventRecord(ev0[d], st[d]));
                k_read_peer<8><<<N, 1024, 0, st[d]>>>(d_all[d], ndev, d, NPEER, chunk, sink[d]);
                CK(cudaEventRecord(ev1[d], st[d]));
            }
            double worst = 0, best = 1e30, sum = 0;
            for (int d = 0; d < ndev; ++d) {
                CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d]));
                float ms; CK(cudaEventElapsedTime(&ms, ev0[d], ev1[d]));
                double g = (double)moved / (ms / 1e3) / 1e9;
                worst = std::max(worst, (double)ms); best = std::min(best, g); sum += g;
            }
            double agg = (double)ndev * moved / (worst / 1e3) / 1e9;
            printf("D_ALL N=%d perRankWorstGBps=%.2f perRankMin=%.2f meanPerRank=%.2f nodeAggGBps=%.2f\n",
                   N, (double)moved / (worst / 1e3) / 1e9, best, sum / ndev, agg);
        }
        for (int d = 0; d < ndev; ++d) { CK(cudaSetDevice(d)); cudaStreamDestroy(st[d]); }
    }

    // ------------------- Part E: 固定 grid，扫 peer 数 -------------------
    // 核心提问：同样多的 SM，把 peer 数从 1 加到 6，聚合带宽会不会线性涨？
    printf("\n== Part E: fixed grid=64, sweep #peers (read) ==\n");
    for (int np = 1; np <= NPEER; ++np) {
        const int N = 64;
        int slots = (N + np - 1) / np;
        size_t chunk = PVEC / std::max(slots, 1);
        size_t moved = (size_t)N * chunk * 16;
        float mr = timeit([&] { k_read_peer<8><<<N, 1024>>>(d_all[RANK], ndev, RANK, np, chunk, sink[RANK]); }, 5);
        double g = (double)moved / (mr / 1e3) / 1e9;
        printf("E_SWEEP npeer=%d grid=64 readGBps=%.2f perSMread=%.2f\n", np, g, g / N);
    }

    printf("\nDONE\n");
    return 0;
}
