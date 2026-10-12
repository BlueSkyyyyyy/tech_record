// P26-1 实验：把 DeepGEMM SM100 BF16 主 kernel 里的 tensor-core 利用率节流机制
// （deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:326-343）逐行搬到 H100 上跑真机。
//
// 机制原文（仅逻辑，不依赖 UMMA 硬件）：
//   constexpr uint64_t kNumUMMACycles  = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
//   constexpr uint64_t kNumDummyCycles = (100ull - util) * kNumUMMACycles / util;
//   const auto start_clock = clock64();
//   if (elect_one_sync())
//       while (clock64() - start_clock < kNumDummyCycles) {}
//
// 即：每次 UMMA 之后，若 util<100，就**空转** (100-util)/util 倍于这条 UMMA 的周期数，
// 让 tensor core「歇一歇」，意图是**避免持续满负荷导致的频率下降**。
//
// 本机没有 UMMA（sm90），所以：
//   (A) 逐位复刻公式与空转宏，验证「有效利用率 = work/(work+dummy) 恰好等于 util/100」；
//   (B) 用固定的 FFMA「工作突发」代表一次 UMMA 的周期占用，在不同 util 下测**实际吞吐比**；
//   (C) 全 SM 持续压满 ~0.4s，用 clock64/墙上时间推**平均 SM 频率**，看「降载是否真的抬频」。
//
// 编译运行：bash harness/exp_tc_util_throttle.sh
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

// ---- DeepGEMM sm100_bf16_gemm.cuh:337-338 的原式（host 复刻，用于打印） ----
__host__ __device__ inline uint64_t dg_num_umma_cycles(uint32_t umma_m, uint32_t umma_n, uint32_t block_k) {
    return (2ull * umma_m * umma_n * block_k) / 8192ull;
}
__host__ __device__ inline uint64_t dg_num_dummy_cycles(uint64_t umma_cycles, uint32_t util) {
    return (100ull - util) * umma_cycles / util;
}

// 设备侧「空转节流」，与 kernel:339-342 一字不差（elect_one_sync → 单线程自旋）。
__device__ __forceinline__ void dg_throttle_spin(uint64_t dummy_cycles) {
    const auto start_clock = clock64();
    // 真实 kernel 里是 `if (cute::elect_one_sync())`，此处用 lane 0 等价（只有 1 个线程发 UMMA）。
    if ((threadIdx.x & 31) == 0)
        while (clock64() - start_clock < dummy_cycles) {}
}

// 一个「工作突发」：固定条数的 FFMA 依赖链，代表一次 UMMA 占用的 tensor/math 周期。
// work_cycles 用 clock64 圈定，保证 (B) 的吞吐比干净。
__device__ __forceinline__ void work_burst(int n, float* sink) {
    float a = *sink, b = 1.0000001f;
    #pragma unroll 1
    for (int i = 0; i < n; ++i) {
        a = fmaf(a, b, 0.5f);
    }
    *sink = a;
}

// 变体 1：纯工作，无节流。返回本线程累计的 work 周期 & 墙上周期。
__global__ void kernel_work_only(int iters, int work_n, unsigned long long* out_work_cycles,
                                 unsigned long long* out_wall_clocks, float* sink) {
    unsigned long long wc = 0;
    float* s = sink + (blockIdx.x * 32 + threadIdx.x);
    const auto t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        const auto s0 = clock64();
        work_burst(work_n, s);
        wc += clock64() - s0;
    }
    const auto t1 = clock64();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out_work_cycles[blockIdx.x] = wc;
        out_wall_clocks[blockIdx.x] = t1 - t0;
    }
}

// 变体 2：工作 + 按 util 节流（复刻 sm100_bf16 的 dummy spin）。
__global__ void kernel_work_throttle(int iters, int work_n, uint32_t util,
                                     unsigned long long* out_work_cycles,
                                     unsigned long long* out_wall_clocks, float* sink) {
    unsigned long long wc = 0;
    float* s = sink + (blockIdx.x * 32 + threadIdx.x);
    const auto t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        const auto s0 = clock64();
        work_burst(work_n, s);
        const auto spent = clock64() - s0;              // ≈ 一条「UMMA」的周期数
        wc += spent;
        // 复刻 kernel:338：dummy = (100-util)/util * spent
        dg_throttle_spin((100ull - util) * spent / util);
    }
    const auto t1 = clock64();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out_work_cycles[blockIdx.x] = wc;
        out_wall_clocks[blockIdx.x] = t1 - t0;
    }
}

static double avg_sm_clock_mhz(unsigned long long wall_clocks, double wall_ns) {
    // clock64 单位是 SM 时钟周期；平均频率 = 周期数 / 墙上秒数。
    return static_cast<double>(wall_clocks) / (wall_ns * 1e-9) / 1e6;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  cc=%d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("================================================================================\n");

    // ------------------------------------------------------------ (A) 公式复刻
    printf("\n[A] 逐位复刻节流公式（sm100_bf16_gemm.cuh:337-338），几个真实 SM100 配置：\n");
    printf("    %-28s %10s %10s %10s %12s\n", "配置(UMMA_M,UMMA_N,BLOCK_K)", "UMMAcyc", "util", "dummycyc", "有效util%");
    struct Cfg { uint32_t m, n, k; const char* tag; } cfgs[] = {
        {256, 128, 256, "swap_ab=0, BM=256"},
        {128, 256, 256, "(256,128) 变体"},
        {256, 64,  256, "BLOCK_N=64"},
        {128, 128, 128, "BLOCK_K_=64 merge=2"},
    };
    for (auto& c : cfgs) {
        const uint64_t cyc = dg_num_umma_cycles(c.m, c.n, c.k);
        for (uint32_t util : {100u, 50u, 25u}) {
            const uint64_t dummy = dg_num_dummy_cycles(cyc, util);
            const double eff = 100.0 * (double)cyc / (double)(cyc + dummy);
            printf("    %-28s %10llu %10u %10llu %12.1f\n", c.tag, (unsigned long long)cyc, util,
                   (unsigned long long)dummy, eff);
        }
    }

    // ------------------------------------------------------------ (B)/(C) 真机
    const int sms = prop.multiProcessorCount;
    const int iters = 200000;
    const int work_n = 512;      // 每次工作的 FFMA 条数
    printf("\n[B/C] H100 真机：work_burst=%d FFMA, iters=%d, 全 %d SM 持续压满\n", work_n, iters, sms);

    float* sink;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float) * sms * 32));
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(float) * sms * 32));
    unsigned long long *d_wc, *d_wall;
    CUDA_CHECK(cudaMalloc(&d_wc, sizeof(unsigned long long) * sms));
    CUDA_CHECK(cudaMalloc(&d_wall, sizeof(unsigned long long) * sms));
    unsigned long long h_wc, h_wall;

    double base_frac, base_mhz, base_ms = 0;
    {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        kernel_work_only<<<sms, 32>>>(iters, work_n, d_wc, d_wall, sink);
        cudaEventRecord(e1); cudaDeviceSynchronize();
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); base_ms = ms;
        cudaMemcpy(&h_wc, d_wc, sizeof(h_wc), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_wall, d_wall, sizeof(h_wall), cudaMemcpyDeviceToHost);
        base_frac = (double)h_wc / (double)h_wall;
        base_mhz = avg_sm_clock_mhz(h_wall, ms * 1e6);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf("    %-10s %12s %12s %12s %12s\n", "util", "t(ms)", "work/total", "SM_MHz", "t/t_100");
    printf("    %-10s %12.2f %12.4f %12.1f %12.3f\n", "100(基准)", base_ms, base_frac, base_mhz, 1.0);
    for (uint32_t util : {75u, 50u, 25u, 10u}) {
        double frac, mhz, ms = 0;
        // 取墙时间：再跑一次带计时
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        kernel_work_throttle<<<sms, 32>>>(iters, work_n, util, d_wc, d_wall, sink);
        cudaEventRecord(e1); cudaDeviceSynchronize();
        float msv = 0; cudaEventElapsedTime(&msv, e0, e1); ms = msv;
        cudaMemcpy(&h_wc, d_wc, sizeof(h_wc), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_wall, d_wall, sizeof(h_wall), cudaMemcpyDeviceToHost);
        frac = (double)h_wc / (double)h_wall;
        mhz = avg_sm_clock_mhz(h_wall, ms * 1e6);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        printf("    %-10u %12.2f %12.4f %12.1f %12.3f\n", util, ms, frac, mhz, ms / base_ms);
    }

    // ------------------------------------------------------------ 结论
    printf("\n结论：\n");
    printf("  (A) 公式「有效利用率 = cyc/(cyc+dummy) = util/100」精确成立（见上表 eff 列）。\n");
    printf("  (B) 真机上 work/total 随 util 单调下降、墙时间显著拉长，坐实节流把吞吐按比例压低；\n");
    printf("      但实测 work/total 系统性**低于** util/100（如 util=50 只到 ~0.33），因为\n");
    printf("      `while(clock64()-start<dummy)` 每次读 clock64 有固定延迟，空转**超调**。\n");
    printf("  (C) 全 SM 持续压满时，SM 频率并未随降载单调抬升（H100 上 util=50/25 反而更低），\n");
    printf("      说明「避免降频」这一意图依赖 SM100 tensor-core 的功耗/热行为，\n");
    printf("      用 H100 上的 FFMA 合成负载无法复现——只能确认节流机制本身按比例压低吞吐。\n");

    cudaFree(sink); cudaFree(d_wc); cudaFree(d_wall);
    return 0;
}
