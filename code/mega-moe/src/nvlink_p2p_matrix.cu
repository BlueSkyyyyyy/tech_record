// P109-1 实验：H100 上量化「全互联 NVLink」的 P2P 拷贝带宽矩阵。
//
// 为什么做这个：DeepEP 的 `check_nvlink_connections`（`deep_ep/utils/envs.py:146-181`）只用
// **NVML** 的 `nvmlDeviceGetP2PStatus(..., NVML_P2P_CAPS_INDEX_NVLINK)` 判定「某对 GPU 之间
// 有没有 NVLink」，它返回的是一个离散的**能力状态**（OK / 不可用），并不给带宽。而 DeepEP 的
// NVLink 域（`num_nvlink_ranks`）、对称内存 combine、`nvlink_barrier` 全都建立在「节点内所有
// GPU 两两都能通过 NVLink 直接互访」这个假设上。本实验给出这个假设的**量化版本**：
// 用 CUDA 的 `cudaMemcpyPeerAsync` 量 8×8 的有向 P2P 带宽，坐实 H100 SXM 是**满互联**
// （每对 GPU 都走 NVLink，而不是 PCIe/NVLink 混合或经 CPU 绕行）。
//
// 本文件只用于实验，不进入 kernel 实现。
//
// 编译：nvcc -arch=sm_90 -O3 -std=c++17 src/nvlink_p2p_matrix.cu -o /tmp/nvlink_p2p_matrix
// 运行：/tmp/nvlink_p2p_matrix [size_mib] [iters]
//   默认 64 MiB / 30 iters。输出见 harness/exp_deepep_nvlink_p2p.out.txt。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            std::printf("[CUDA_ERROR] %s -> %s\n", #expr, cudaGetErrorString(_e)); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

int main(int argc, char** argv) {
    int size_mib = argc > 1 ? std::atoi(argv[1]) : 64;
    int iters = argc > 2 ? std::atoi(argv[2]) : 30;
    size_t bytes = (size_t)size_mib << 20;

    int num_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_dev));
    std::printf("num_devices=%d  size=%d MiB  iters=%d\n", num_dev, size_mib, iters);

    // 设备名 + 同/跨 NUMA 提示（用 cudaDeviceGetAttribute 拿 pci bus id 不够，直接打印名字）
    for (int i = 0; i < num_dev; ++i) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        std::printf("  dev%d: %s  cc=%d.%d  pci_bus=%02x\n",
                    i, p.name, p.major, p.minor, p.pciBusID);
    }

    // 1. 建立每对设备的 peer access（i 能访问 j），并报告 canAccessPeer。
    std::printf("\n== cudaDeviceCanAccessPeer 与 cudaDeviceEnablePeerAccess ==\n");
    int can[num_dev][num_dev];
    for (int i = 0; i < num_dev; ++i)
        for (int j = 0; j < num_dev; ++j) can[i][j] = 0;
    for (int i = 0; i < num_dev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < num_dev; ++j) {
            if (i == j) continue;
            int c = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&c, i, j));
            can[i][j] = c;
            if (c) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                // 已启用会返回 cudaErrorPeerAccessAlreadyEnabled，忽略
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    std::printf("  enable peer %d->%d failed: %s\n", i, j, cudaGetErrorString(e));
                }
            }
        }
    }
    std::printf("canAccessPeer matrix (row=src i, col=dst j):\n     ");
    for (int j = 0; j < num_dev; ++j) std::printf("%4d", j);
    std::printf("\n");
    for (int i = 0; i < num_dev; ++i) {
        std::printf("src%-2d", i);
        for (int j = 0; j < num_dev; ++j) std::printf("%4d", i == j ? -1 : can[i][j]);
        std::printf("\n");
    }

    // 2. 分配每设备的源缓冲（每个设备一份），供任意 i->j 拷贝使用。
    void* buf[num_dev];
    cudaStream_t stream[num_dev];
    for (int i = 0; i < num_dev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&buf[i], bytes));
        CUDA_CHECK(cudaMemset(buf[i], 0x3f, bytes));
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    // 3. 有向 P2P 带宽矩阵：src=i -> dst=j（在设备 j 的流上做 peer copy 并计时）。
    auto bench_peer = [&](int dst, int src) -> double {
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaSetDevice(dst));
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        // warmup
        for (int k = 0; k < 3; ++k) {
            CUDA_CHECK(cudaMemcpyPeerAsync(buf[dst], dst, buf[src], src, bytes, stream[dst]));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream[dst]));
        CUDA_CHECK(cudaEventRecord(e0, stream[dst]));
        for (int k = 0; k < iters; ++k) {
            CUDA_CHECK(cudaMemcpyPeerAsync(buf[dst], dst, buf[src], src, bytes, stream[dst]));
        }
        CUDA_CHECK(cudaEventRecord(e1, stream[dst]));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        double sec = ms / 1e3;
        return (double)bytes * iters / sec / 1e9;  // GB/s (SI, 1e9)
    };

    std::printf("\n== directed P2P bandwidth (GB/s, SI) src=row -> dst=col ==\n     ");
    for (int j = 0; j < num_dev; ++j) std::printf("%8d", j);
    std::printf("   (GB/s)\n");
    double bw[num_dev][num_dev];
    for (int i = 0; i < num_dev; ++i)
        for (int j = 0; j < num_dev; ++j) bw[i][j] = 0.0;
    for (int i = 0; i < num_dev; ++i) {
        std::printf("src%-2d", i);
        for (int j = 0; j < num_dev; ++j) {
            if (i == j) {
                std::printf("%8s", "-");
                continue;
            }
            double g = bench_peer(j, i);
            bw[i][j] = g;
            std::printf("%8.1f", g);
        }
        std::printf("\n");
    }

    // 4. 同设备 D2D 拷贝带宽（基线）。
    std::printf("\n== same-device D2D baseline ==\n");
    for (int i = 0; i < num_dev; ++i) {
        void* tmp = nullptr;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&tmp, bytes));
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        for (int k = 0; k < 3; ++k)
            CUDA_CHECK(cudaMemcpyAsync(tmp, buf[i], bytes, cudaMemcpyDeviceToDevice, stream[i]));
        CUDA_CHECK(cudaStreamSynchronize(stream[i]));
        CUDA_CHECK(cudaEventRecord(e0, stream[i]));
        for (int k = 0; k < iters; ++k)
            CUDA_CHECK(cudaMemcpyAsync(tmp, buf[i], bytes, cudaMemcpyDeviceToDevice, stream[i]));
        CUDA_CHECK(cudaEventRecord(e1, stream[i]));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("  dev%d d2d: %.1f GB/s\n", i, (double)bytes * iters / (ms / 1e3) / 1e9);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(tmp);
    }

    // 5. 汇总：min/max/mean P2P。
    double mn = 1e18, mx = 0.0, sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < num_dev; ++i)
        for (int j = 0; j < num_dev; ++j)
            if (i != j) { mn = bw[i][j] < mn ? bw[i][j] : mn; mx = bw[i][j] > mx ? bw[i][j] : mx; sum += bw[i][j]; ++cnt; }
    std::printf("\n== summary ==\np2p_gbps: min=%.1f max=%.1f mean=%.1f  (pairs=%d)\n",
                mn, mx, sum / cnt, cnt);
    return 0;
}
