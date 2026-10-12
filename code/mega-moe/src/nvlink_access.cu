// P3-4 实验：H100 上量化「本地显存 vs NVLink 远端显存」的访问成本。
//
// 为什么做这个：Mega MoE（SM100 专属）的两段通信都跨 rank 走 NVLink：
//   1. dispatch：dispatch warp 用 `tma_load_1d` 从**远端 rank** 的 `input_token_buffer`
//      把 token 拉回本地（kernel:552-576），并且窥探远端 `get_src_token_topk_idx_ptr`
//      （`:531`）、远端 topk weight（`:579-581`）、逐 32 元素搬远端 SF（`:586-599`）；
//   2. combine：L2 epilogue 把 BF16 结果经 `sym_buffer.map(dst_ptr, dst_rank_idx)`
//      **直写**到远端 rank 的 `combine_token_buffer`（`:1346-1351`），最后 combine 段
//      再用 `tma_load_1d` 拉回 top-k 贡献做 fp32 归约（`:1454-1460`）。
//
// 这些远端访问的**延迟**远高于本地，而带宽受 NVLink 限制，所以 kernel 把它们切成
// chunk、和计算重叠（P3-2）。此外 SM100 的 `locality domain`（`csrc/runtime/locality_domain.hpp`、
// `csrc/apis/locality_domain.hpp`）用 **pointer-chase 测延迟**给每个 SM 打「离哪块显存更近」的标签，
// 调度器据此把 N 轴任务分域（`scheduler/mega_moe.cuh:344-370`），让 SM 处理**本地域**的数据。
//
// 本机 H100 是**单 die 显存**（没有 SM100 那种双 locality domain 的 HBM），但可以用
// **8 卡 NVLink 拓扑**做最近的类比：同一进程里从 GPU0 访问 GPU0（本地）、GPU1（同 NUMA socket）、
// GPU4（跨 NUMA socket）的显存，量出「越远越慢」的梯度。这解释了两件事：
//   a. 为什么 dispatch 的远端 pull 要分 chunk、要尽早 issue 以掩盖延迟；
//   b. 为什么 SM100 要费劲做 locality-domain 亲和（处理本地域数据的 SM 延迟最低）。
//
// 本文件只用于实验，不进入 kernel 实现。
//
// 编译：nvcc -arch=sm_90 -O3 -std=c++17 src/nvlink_access.cu -o /tmp/nvlink_access
// 运行：/tmp/nvlink_access   （输出见 harness/exp_nvlink_access.out.txt）

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ---- 参数 ----------------------------------------------------------------
static constexpr uint32_t kNumLines = 1u << 20;                 // 1M 行
static constexpr uint32_t kLineU32 = 32;                        // 128B / 4B
static constexpr uint32_t kChainStride = 7;                     // 与 2^20 互质 -> 遍历所有行
static constexpr size_t kBufBytes = size_t(kNumLines) * kLineU32 * 4;  // 128 MiB

// 用 pointer-chase 走一遍整条链：每个 hop 的地址依赖上一个 load 的结果，
// 因此 hop 之间严格串行，量到的就是一次（本地或远端）访存的裸延迟。
// NOTES: `.cv`（volatile，绕过 L1/L2 复用）确保每个 hop 真的落到 HBM / NVLink，
// 而不是被 cache 命中——这与 SM100 locality-domain 探针的意图一致（它们用 `.cg`，
// 但那里两块显存都是同型号 HBM，只比「home die」；这里跨卡必须绕 cache）。
__global__ __launch_bounds__(1, 1)
void chase_kernel(const uint32_t* buf, uint32_t hops, uint64_t* out) {
    if (threadIdx.x != 0)
        return;
    uint32_t idx = __ldcv(buf);                 // 起点：line0 word0 -> stride*32
    uint64_t start = clock64();
    for (uint32_t i = 0; i < hops; ++i)
        idx = __ldcv(buf + idx);
    uint64_t cycles = clock64() - start;
    *out = (idx == 0xffffffffu) ? 0ull : cycles;
}

__global__ void init_chain_kernel(uint32_t* buf, uint32_t nlines) {
    uint32_t l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nlines)
        return;
    buf[l * 32 + 0] = ((l + kChainStride) % nlines) * 32;
}

// 流式读带宽：本地 vs 远端。用 `.cg` 只走 L2，避免 L1 影响；buffer 大小 > L2（50MB）。
__global__ void stream_read_kernel(const uint4* p, size_t n_u4, unsigned long long* sink) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    unsigned long long s = 0;
    for (; i < n_u4; i += stride) {
        uint4 v = __ldcg(p + i);
        s += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    if (s == 0x123456789abcdefull)
        atomicAdd(sink, s);
}

// 流式写带宽：对应 combine 段 L2 epilogue 经 `sym_buffer.map` 直写远端 combine buffer
// （kernel:1346-1351）。这里每线程写连续的 16B（float4），与 combine 的写法一致。
__global__ void stream_write_kernel(uint4* p, size_t n_u4, uint4 val) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < n_u4; i += stride)
        __stcg(p + i, val);
}

static uint64_t run_chase(int run_dev, const uint32_t* ptr, uint32_t hops, int iters) {
    uint64_t* d_out = nullptr;
    CUDA_CHECK(cudaSetDevice(run_dev));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint64_t)));
    // 预热
    chase_kernel<<<1, 1>>>(ptr, 1u << 16, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t best = ~0ull;
    for (int it = 0; it < iters; ++it) {
        chase_kernel<<<1, 1>>>(ptr, hops, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t c = 0;
        CUDA_CHECK(cudaMemcpy(&c, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (c != 0 && c < best)
            best = c;
    }
    cudaFree(d_out);
    return best / hops;  // cycles per hop
}

static double run_stream(int run_dev, const uint4* ptr, size_t bytes, int iters) {
    const size_t n_u4 = bytes / 16;
    unsigned long long* sink = nullptr;
    CUDA_CHECK(cudaSetDevice(run_dev));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(unsigned long long)));
    dim3 block(256), grid(2048);
    double best_ms = 1e9;
    for (int it = 0; it < iters; ++it) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        stream_read_kernel<<<grid, block>>>(ptr, n_u4, sink);
        cudaEventRecord(e1);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        if (ms < best_ms)
            best_ms = ms;
    }
    cudaFree(sink);
    return bytes / (best_ms / 1e3) / 1e9;  // GB/s
}

static double run_stream_write(int run_dev, uint4* ptr, size_t bytes, int iters) {
    const size_t n_u4 = bytes / 16;
    CUDA_CHECK(cudaSetDevice(run_dev));
    dim3 block(256), grid(2048);
    uint4 val = make_uint4(0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u);
    double best_ms = 1e9;
    for (int it = 0; it < iters; ++it) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        stream_write_kernel<<<grid, block>>>(ptr, n_u4, val);
        cudaEventRecord(e1);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        if (ms < best_ms)
            best_ms = ms;
    }
    return bytes / (best_ms / 1e3) / 1e9;  // GB/s
}

int main() {
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    std::printf("visible GPUs = %d\n", dev_count);
    if (dev_count < 5) {
        std::printf("need >=5 GPUs (0 local, 1 same-socket peer, 4 cross-socket peer); abort\n");
        return 1;
    }

    const int dev_local = 0, dev_near = 1, dev_far = 4;
    const int owning[3] = {dev_local, dev_near, dev_far};
    const char* names[3] = {"local  (GPU0->GPU0)", "peer   (GPU0->GPU1, same socket)",
                            "peer   (GPU0->GPU4, cross socket)"};

    // 时钟频率（kHz），用于 cycles -> ns
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev_local));
    const double ns_per_cycle = 1e6 / clock_khz;
    std::printf("clock rate = %.3f GHz (ns/cycle = %.4f)\n",
                clock_khz / 1e6, ns_per_cycle);

    // 在每块卡的显存上分配并初始化一条 pointer-chase 链
    uint32_t* buf[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaSetDevice(owning[i]));
        CUDA_CHECK(cudaMalloc(&buf[i], kBufBytes));
        init_chain_kernel<<<(kNumLines + 255) / 256, 256>>>(buf[i], kNumLines);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 开启 GPU0 -> {GPU1, GPU4} 的 P2P 访问
    CUDA_CHECK(cudaSetDevice(dev_local));
    int can_near = 0, can_far = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_near, dev_local, dev_near));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_far, dev_local, dev_far));
    std::printf("P2P access from GPU0: ->GPU%d=%d ->GPU%d=%d\n",
                dev_near, can_near, dev_far, can_far);
    if (can_near)
        CUDA_CHECK(cudaDeviceEnablePeerAccess(dev_near, 0));
    if (can_far)
        CUDA_CHECK(cudaDeviceEnablePeerAccess(dev_far, 0));

    // ---- 1. 依赖加载延迟（pointer chase, .cv）----------------------------
    std::printf("\n[dependent-load latency] chain = %u lines x 128B = %.0f MiB, .cv (bypass caches)\n",
                kNumLines, kBufBytes / 1048576.0);
    double local_cyc = 0.0;
    for (int i = 0; i < 3; ++i) {
        uint64_t cyc = run_chase(dev_local, buf[i], kNumLines, 3);
        double ns = cyc * ns_per_cycle;
        if (i == 0)
            local_cyc = (double)cyc;
        std::printf("  %-38s raw=%llu cyc/hop  cyc/hop=%6llu  ~%7.1f ns  (x%.2f vs local)\n",
                    names[i], (unsigned long long)cyc, (unsigned long long)cyc, ns,
                    local_cyc > 0 ? cyc / local_cyc : 1.0);
    }

    // ---- 2. 流式读带宽（本地 vs 远端）-----------------------------------
    std::printf("\n[streaming read bandwidth] %.0f MiB, grid=2048x256\n", kBufBytes / 1048576.0);
    for (int i = 0; i < 3; ++i) {
        double gbps = run_stream(dev_local, reinterpret_cast<const uint4*>(buf[i]), kBufBytes, 20);
        std::printf("  %-38s %7.1f GB/s\n", names[i], gbps);
    }

    // ---- 3. 流式写带宽（本地 vs 远端）-----------------------------------
    std::printf("\n[streaming write bandwidth] %.0f MiB, grid=2048x256\n", kBufBytes / 1048576.0);
    for (int i = 0; i < 3; ++i) {
        double gbps = run_stream_write(dev_local, reinterpret_cast<uint4*>(buf[i]), kBufBytes, 20);
        std::printf("  %-38s %7.1f GB/s\n", names[i], gbps);
    }

    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaSetDevice(owning[i]));
        cudaFree(buf[i]);
    }
    return 0;
}
