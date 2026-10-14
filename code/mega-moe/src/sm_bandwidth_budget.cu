// P181-1 实验：通信腿的「每 SM 带宽」预算 —— 校验 DeepEP 自动 SM 模型里被钉死的两个常量
//   `get_sm_read_gbs()=180` / `get_sm_write_gbs()=45`（deep_ep/utils/envs.py:193-204），
//   并量出「一个 SM 能拉多少 HBM / NVLink 带宽、几个 SM 就把 HBM/NVLink 打满」。
//
// 背景：DeepEP 用 `get_theoretical_num_sms()`（deep_ep/buffers/ep.py:431-541）决定 dispatch/combine
//   每个 kernel 开多少 SM：它把「每 token 的 HBM 读/写 + NVLink/RDMA 流量」分解成**每 SM 的字节数**，
//   再除以每 SM 的 HBM 读/写带宽（:524-528），取四候选最大、×1.3、align4、max(4,·)（:530-532）。
//   也就是说，**通信腿该开几个 SM，完全由这两个每 SM 带宽常量决定**。而它们是**硬编码的、带 TODO
//   「use architecture-specific values」的**（envs.py:195/202）。第 63 篇（docs/73）拆了这层模型却
//   只**打印**了这两个常量、从没在真机上量过它们到底对不对；第 180 篇（docs/190）画了**计算腿**的
//   SM 预算曲线，本篇补上其**对偶**：给通信让出的 SM 到底买到多少带宽。
//
// 本实验在 H100(sm90) 上纯 CUDA/CUDA-runtime 实现（不碰 DeepEP/NCCL 设备 API）：
//   Part A —— HBM：扫 grid = N（1 block/SM，1024 线程，16B 向量化，streaming 提示），量
//             read / write / copy 三条腿的**聚合带宽**与**每 SM 带宽**，找「每 SM 线性」失效的拐点；
//             并单独标定「单 SM 峰值」（扫 blockDim / UNROLL），看 180 / 45 是否可达。
//   Part B —— NVLink P2P：8 卡 UVA peer，在 rank0 上跑 grid=N 从 rank1 读 / 写 rank1，
//             量「几个 SM 打满一条 NVLink」。
//
// 编译/运行见 harness/exp_sm_bandwidth_budget.sh。目标卡代号敏感：一律用「目标卡」指代。

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
// HBM 三条腿：grid = N（N <= SM 数 ⇒ 1 block/SM）
// ---------------------------------------------------------------------------
template <int UNROLL>
__global__ void k_read(const float4* __restrict__ in, size_t n, float* __restrict__ sink) {
    const size_t base = (size_t)blockIdx.x * n;
    const size_t stride = blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    size_t i = threadIdx.x;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        float4 a = __ldcs(in + base + i);
        float4 b = __ldcs(in + base + i + stride);
        float4 c = __ldcs(in + base + i + 2 * stride);
        float4 d = __ldcs(in + base + i + 3 * stride);
        acc.x += a.x + b.x + c.x + d.x;
        acc.y += a.y + b.y + c.y + d.y;
        acc.z += a.z + b.z + c.z + d.z;
        acc.w += a.w + b.w + c.w + d.w;
    }
    for (; i < n; i += stride) {
        float4 a = __ldcs(in + base + i);
        acc.x += a.x; acc.y += a.y; acc.z += a.z; acc.w += a.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == -12345.678f) sink[blockIdx.x] = s;
}

template <int UNROLL>
__global__ void k_write(float4* __restrict__ out, size_t n) {
    const size_t base = (size_t)blockIdx.x * n;
    const size_t stride = blockDim.x;
    const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    size_t i = threadIdx.x;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        __stcs(out + base + i, v);
        __stcs(out + base + i + stride, v);
        __stcs(out + base + i + 2 * stride, v);
        __stcs(out + base + i + 3 * stride, v);
    }
    for (; i < n; i += stride)
        __stcs(out + base + i, v);
}

template <int UNROLL>
__global__ void k_copy(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    const size_t base = (size_t)blockIdx.x * n;
    const size_t stride = blockDim.x;
    size_t i = threadIdx.x;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        __stcs(out + base + i, __ldcs(in + base + i));
        __stcs(out + base + i + stride, __ldcs(in + base + i + stride));
        __stcs(out + base + i + 2 * stride, __ldcs(in + base + i + 2 * stride));
        __stcs(out + base + i + 3 * stride, __ldcs(in + base + i + 3 * stride));
    }
    for (; i < n; i += stride)
        __stcs(out + base + i, __ldcs(in + base + i));
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
    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    printf("DEVICE name=%s sm=%d clock=%.0fMHz bus=%dbit L2=%dKB\n",
           prop.name, NSM, clock_khz / 1e3, prop.memoryBusWidth, prop.l2CacheSize / 1024);
    printf("NVLINK_DETECT ndev=%d\n", ndev);

    // 2 GiB 缓冲区（16B 向量化 = 2^27 个 float4）
    const size_t VEC = (size_t)1 << 27;
    const size_t BYTES = VEC * 16;
    float4 *d_in, *d_out;
    float* d_sink;
    CK(cudaMalloc(&d_in, BYTES));
    CK(cudaMalloc(&d_out, BYTES));
    CK(cudaMalloc(&d_sink, 4096));
    CK(cudaMemset(d_in, 1, BYTES));

    std::vector<int> grids = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 132};
    // 去重 + 只保留 <= NSM
    std::sort(grids.begin(), grids.end());
    grids.erase(std::unique(grids.begin(), grids.end()), grids.end());

    // ------------------- Part A1: 每 SM 读带宽（1024 线程） -------------------
    printf("\n== Part A1: HBM read vs #SM (block=1024, UNROLL=4) ==\n");
    for (int N : grids) {
        if (N > NSM) continue;
        size_t usable = VEC - (VEC % N);
        size_t n = usable / N;
        float ms = timeit([&] { k_read<4><<<N, 1024>>>(d_in, n, d_sink); }, 5);
        double gbs = (double)usable * 16.0 / (ms / 1e3) / 1e9;
        printf("A1_READ N=%d ms=%.4f GBps=%.2f perSM=%.3f\n", N, ms, gbs, gbs / N);
    }

    // ------------------- Part A2: 每 SM 写带宽 -------------------
    printf("\n== Part A2: HBM write vs #SM (block=1024, UNROLL=4) ==\n");
    for (int N : grids) {
        if (N > NSM) continue;
        size_t usable = VEC - (VEC % N);
        size_t n = usable / N;
        float ms = timeit([&] { k_write<4><<<N, 1024>>>(d_out, n); }, 5);
        double gbs = (double)usable * 16.0 / (ms / 1e3) / 1e9;
        printf("A2_WRITE N=%d ms=%.4f GBps=%.2f perSM=%.3f\n", N, ms, gbs, gbs / N);
    }

    // ------------------- Part A3: 每 SM copy（读+写） -------------------
    printf("\n== Part A3: HBM copy vs #SM (block=1024, UNROLL=4) ==\n");
    for (int N : grids) {
        if (N > NSM) continue;
        size_t usable = VEC - (VEC % N);
        size_t n = usable / N;
        float ms = timeit([&] { k_copy<4><<<N, 1024>>>(d_in, d_out, n); }, 5);
        double gbs = (double)usable * 16.0 / (ms / 1e3) / 1e9;      // 单程字节/时长
        double eff = 2.0 * (double)usable * 16.0 / (ms / 1e3) / 1e9;  // 读+写合计
        printf("A3_COPY N=%d ms=%.4f GBps=%.2f eff2x=%.2f perSM=%.3f\n", N, ms, gbs, eff, gbs / N);
    }

    // ------------------- Part A4: 刻度（cudaMemset / D2D memcpy 上限） -------------------
    printf("\n== Part A4: 参考上限（库路径） ==\n");
    {
        float ms = timeit([&] { CK(cudaMemset(d_out, 7, BYTES)); }, 5);
        printf("A4_MEMSET GBps=%.2f\n", (double)BYTES / (ms / 1e3) / 1e9);
    }
    {
        float ms = timeit([&] { CK(cudaMemcpy(d_out, d_in, BYTES, cudaMemcpyDeviceToDevice)); }, 5);
        printf("A4_MEMCPYD2D GBps=%.2f (单程) eff2x=%.2f\n",
               (double)BYTES / (ms / 1e3) / 1e9, 2.0 * (double)BYTES / (ms / 1e3) / 1e9);
    }

    // ------------------- Part A5: 单 SM 峰值 vs blockDim / UNROLL -------------------
    printf("\n== Part A5: single-SM peak (N=1) vs blockDim/UNROLL ==\n");
    for (int bd : {128, 256, 512, 1024}) {
        size_t usable = VEC;
        size_t n = usable;
        float mr = timeit([&] { k_read<4><<<1, bd>>>(d_in, n, d_sink); }, 5);
        double gr = (double)usable * 16.0 / (mr / 1e3) / 1e9;
        float mw = timeit([&] { k_write<4><<<1, bd>>>(d_out, n); }, 5);
        double gw = (double)usable * 16.0 / (mw / 1e3) / 1e9;
        printf("A5_SINGLESM bd=%d readGBps=%.2f writeGBps=%.2f\n", bd, gr, gw);
    }
    // 单 SM copy 的 blockDim 扫描（读+写都吃 LSU）
    for (int bd : {256, 512, 1024}) {
        size_t n = VEC;
        float mc = timeit([&] { k_copy<4><<<1, bd>>>(d_in, d_out, n); }, 5);
        printf("A5_SINGLESM_COPY bd=%d eff2xGBps=%.2f\n", bd, 2.0 * (double)n * 16.0 / (mc / 1e3) / 1e9);
    }

    // ------------------- Part B: NVLink P2P read/write vs #SM -------------------
    printf("\n== Part B: NVLink P2P (rank0 <-> rank1) read/write vs #SM ==\n");
    if (ndev >= 2) {
        float4* p0 = nullptr; float4* p1 = nullptr;
        CK(cudaSetDevice(0)); CK(cudaMalloc(&p0, BYTES));
        CK(cudaSetDevice(1)); CK(cudaMalloc(&p1, BYTES)); CK(cudaMemset(p1, 1, BYTES));
        cudaError_t e0 = cudaSetDevice(0), e1 = cudaSuccess;
        (void)e0;
        e1 = cudaDeviceEnablePeerAccess(1, 0);
        if (e1 != cudaSuccess) {
            printf("B_NOPEER enable=1 from 0 failed: %s\n", cudaGetErrorString(e1));
        } else {
            // 用 P2P 可访问范围：固定 256 MiB（够大，且两端都放得下）
            const size_t PVEC = (size_t)1 << 24;  // 256 MiB
            int can = 0; CK(cudaDeviceCanAccessPeer(&can, 0, 1));
            printf("B_PEER canAccess=%d\n", can);
            for (int N : grids) {
                if (N > NSM) continue;
                size_t usable = PVEC - (PVEC % N);
                size_t n = usable / N;
                float mr = timeit([&] { k_read<4><<<N, 1024>>>(p1, n, d_sink); }, 5);
                double gr = (double)usable * 16.0 / (mr / 1e3) / 1e9;
                float mw = timeit([&] { k_write<4><<<N, 1024>>>(p1, n); }, 5);
                double gw = (double)usable * 16.0 / (mw / 1e3) / 1e9;
                printf("B_P2P N=%d readGBps=%.2f writeGBps=%.2f\n", N, gr, gw);
            }
        }
        CK(cudaSetDevice(0)); cudaFree(p0);
        CK(cudaSetDevice(1)); cudaFree(p1);
        CK(cudaSetDevice(0));
    } else {
        printf("B_NOPEER ndev<2\n");
    }

    printf("\nDONE\n");
    return 0;
}
