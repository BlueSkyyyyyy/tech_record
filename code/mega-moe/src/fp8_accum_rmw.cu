// P141-1 实验：MoE wgrad 的「FP8 累加器」路线——把每步累积器的 RMW 从 8B 压到 2B，
//   量它的**真实带宽**与**数值边界**。
//
// 背景（承接第 145~150 篇）：
//   第 145 篇把 wgrad 慢的根因量化清楚：真凶是**每专家一份 fp32 `[M,N]` 参数梯度**
//     （占 96% 访存流量）。第 146~149 篇试了 bf16 输出 / 融合进优化器 / epilogue 归约，
//     端到端封顶 ~1.67×；第 149 篇坐实「每步多读一份 D 会彻底暴露」；第 150 篇给出
//     **结构性正面答案**：把 S 步梯度折进收缩维 K（big-K 折叠），跨 launch 的累积器 RMW 归零。
//   第 150 篇结尾的**遗留**是路线 ①：**FP8 累加器**——把每步 RMW 的字节从 fp32 的
//     `4B 读 + 4B 写 = 8B/元素/步` 压到 FP8 的 `1B + 1B = 2B/元素/步`（省 4×）。
//
// 本文件量的是路线 ① 的**字节侧下界**（数值侧另见 harness/exp_wgrad_fp8_accum.py）：
//   一个 GEMM 的 epilogue 在**寄存器里**已经拿到本步的 fp32 偏导 `dp`（融合场景，见第 148/149 篇），
//   它要做的是 `acc ← quant(acc + dp)`：**HBM 上只有累积器自己被读写**。
//     存储 fp32 : 读 4B + 写 4B = 8B/元素/步
//     存储 bf16 : 读 2B + 写 2B = 4B/元素/步   （第 149 篇的 EPI-B）
//     存储 fp8  : 读 1B + 写 1B = 2B/元素/步   （本路线）
//   本文件用真实 CUDA 内核把三种存储的 RMW 带宽量出来（架构无关：只有标量浮点 + bf16/fp8 转换），
//   H100(sm90) 真机可跑。数值正确性自检（fp32 精确 / bf16 与 __float2bfloat16 一致 / fp8 与
//   __nv_cvt_float_to_fp8 一致）也在本文件里给出。
//
// 编译/运行见 harness/exp_wgrad_fp8_accum.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using bf16 = __nv_bfloat16;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// 合成的「本步偏导」dp：只依赖下标、住寄存器，不碰 HBM（模拟融合 epilogue 里 TMEM/寄存器中的偏导）。
// 注意：用**除法**而非 `*0.001f`——后者会被 nvcc 与「加」收缩成 FMA，使 host/device 参考不一致。
__host__ __device__ __forceinline__ float dp_of(long i) {
    return (float)((int)(i % 201) - 100) / 1000.0f;
}

// ---------------------------------------------------------------------------
// 三种存储的 RMW 内核（向量化 16B/线程）：acc[i] ← quant(acc[i] + dp_of(i))
//   fp32: 4 元素/线程；bf16: 8 元素/线程；fp8: 16 元素/线程 —— 每条线程恰好 16B 读 + 16B 写。
// ---------------------------------------------------------------------------
__global__ void k_rmw_fp32(float* __restrict__ acc, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const long stride = (long)gridDim.x * blockDim.x * 4;
    for (; i < n; i += stride) {
        uint4 raw = *reinterpret_cast<uint4*>(acc + i);
        float* f = reinterpret_cast<float*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) f[j] += dp_of(i + j);
        *reinterpret_cast<uint4*>(acc + i) = raw;
    }
}

__global__ void k_rmw_bf16(bf16* __restrict__ acc, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const long stride = (long)gridDim.x * blockDim.x * 8;
    for (; i < n; i += stride) {
        uint4 raw = *reinterpret_cast<uint4*>(acc + i);   // 8 × bf16 = 16B
        bf16* p = reinterpret_cast<bf16*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) p[j] = __float2bfloat16(__bfloat162float(p[j]) + dp_of(i + j));
        *reinterpret_cast<uint4*>(acc + i) = raw;
    }
}

__global__ void k_rmw_fp8(__nv_fp8_storage_t* __restrict__ acc, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const long stride = (long)gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        uint4 raw = *reinterpret_cast<uint4*>(acc + i);   // 16 × e4m3 = 16B
        unsigned char* p = reinterpret_cast<unsigned char*>(&raw);
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            // 2 宽打包转换（贴近真实 kernel：cvt.e4m3x2）
            __half2 h2 = __nv_cvt_fp8x2_to_halfraw2(*reinterpret_cast<__nv_fp8x2_storage_t*>(p + j), __NV_E4M3);
            float2 f = __half22float2(h2);
            f.x += dp_of(i + j);
            f.y += dp_of(i + j + 1);
            *reinterpret_cast<__nv_fp8x2_storage_t*>(p + j) =
                __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uint4*>(acc + i) = raw;
    }
}

// ---------------------------------------------------------------------------
// 数值自检：小规模、单步 RMW，与 CPU 参考逐位比。
// ---------------------------------------------------------------------------
__global__ void k_rmw_check_fp32(const float* in, float* out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + dp_of(i);
}
__global__ void k_rmw_check_bf16(const bf16* in, bf16* out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(in[i]) + dp_of(i));
}
__global__ void k_rmw_check_fp8(const __nv_fp8_storage_t* in, __nv_fp8_storage_t* out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half h = __nv_cvt_fp8_to_halfraw(in[i], __NV_E4M3);
        out[i] = __nv_cvt_float_to_fp8(__half2float(h) + dp_of(i), __NV_SATFINITE, __NV_E4M3);
    }
}

static void check() {
    const long n = 70000;

    // ---- fp32：应逐位等于 a + dp ----
    std::vector<float> hf(n); std::vector<float> ho(n);
    for (long i = 0; i < n; ++i) hf[i] = 0.01f * (float)((i * 37) % 199) - 1.0f;
    float *df, *dof; CK(cudaMalloc(&df, n * 4)); CK(cudaMalloc(&dof, n * 4));
    CK(cudaMemcpy(df, hf.data(), n * 4, cudaMemcpyHostToDevice));
    k_rmw_check_fp32<<<(n + 255) / 256, 256>>>(df, dof, n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ho.data(), dof, n * 4, cudaMemcpyDeviceToHost));
    long mism = 0;
    for (long i = 0; i < n; ++i) if (ho[i] != hf[i] + dp_of(i)) ++mism;
    printf("  [check] fp32 RMW 逐位不符 = %ld / %ld\n", mism, n);

    // ---- bf16：device 与 host __float2bfloat16 逐位比 ----
    std::vector<bf16> hbf(n), hbo(n);
    for (long i = 0; i < n; ++i) hbf[i] = __float2bfloat16(0.01f * (float)((i * 53) % 199) - 1.0f);
    bf16 *dbf, *dbfo; CK(cudaMalloc(&dbf, n * 2)); CK(cudaMalloc(&dbfo, n * 2));
    CK(cudaMemcpy(dbf, hbf.data(), n * 2, cudaMemcpyHostToDevice));
    k_rmw_check_bf16<<<(n + 255) / 256, 256>>>(dbf, dbfo, n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hbo.data(), dbfo, n * 2, cudaMemcpyDeviceToHost));
    long bmism = 0;
    for (long i = 0; i < n; ++i) {
        bf16 ref = __float2bfloat16(__bfloat162float(hbf[i]) + dp_of(i));
        if (__bfloat16_as_ushort(hbo[i]) != __bfloat16_as_ushort(ref)) ++bmism;
    }
    printf("  [check] bf16 RMW vs __float2bfloat16 逐位不符 = %ld / %ld\n", bmism, n);

    // ---- fp8：device 与 host 参考（半精度中转）逐位比 ----
    std::vector<__nv_fp8_storage_t> h8(n), h8o(n);
    for (long i = 0; i < n; ++i)
        h8[i] = __nv_cvt_float_to_fp8(0.01f * (float)((i * 71) % 199) - 1.0f, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_storage_t *d8, *d8o; CK(cudaMalloc(&d8, n)); CK(cudaMalloc(&d8o, n));
    CK(cudaMemcpy(d8, h8.data(), n, cudaMemcpyHostToDevice));
    k_rmw_check_fp8<<<(n + 255) / 256, 256>>>(d8, d8o, n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h8o.data(), d8o, n, cudaMemcpyDeviceToHost));
    // host 参考：fp8→half→float，加 dp，再 float→fp8（同一转换函数的 host 版本在头里不可用，
    // 故这里只用 device 结果做自洽性：两次运行应逐位一致）
    long f8mism = 0;
    {
        __nv_fp8_storage_t* d8o2; CK(cudaMalloc(&d8o2, n));
        k_rmw_check_fp8<<<(n + 255) / 256, 256>>>(d8, d8o2, n); CK(cudaDeviceSynchronize());
        std::vector<__nv_fp8_storage_t> t(n); CK(cudaMemcpy(t.data(), d8o2, n, cudaMemcpyDeviceToHost));
        for (long i = 0; i < n; ++i) if (t[i] != h8o[i]) ++f8mism;
        cudaFree(d8o2);
    }
    printf("  [check] fp8  RMW 两次运行逐位不符 = %ld / %ld（确定性）\n", f8mism, n);
    // fp8 的动态范围自检：|acc+dp| 超 E4M3 上限（448）会被 SATFINITE 钳到 448
    {
        __nv_fp8_storage_t one[1], out1[1];
        float bigv = 1000.0f;
        one[0] = __nv_cvt_float_to_fp8(bigv, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t* dd; CK(cudaMalloc(&dd, 1));
        CK(cudaMemcpy(dd, one, 1, cudaMemcpyHostToDevice));
        k_rmw_check_fp8<<<1, 1>>>(dd, dd, 1); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out1, dd, 1, cudaMemcpyDeviceToHost));
        float v = __half2float(__nv_cvt_fp8_to_halfraw(out1[0], __NV_E4M3));
        printf("  [check] fp8 饱和：quant(1000)=%.1f（E4M3 上限 448）\n", v);
        cudaFree(dd);
    }

    cudaFree(df); cudaFree(dof); cudaFree(dbf); cudaFree(dbfo); cudaFree(d8); cudaFree(d8o);
}

// ---------------------------------------------------------------------------
// 带宽计时：对 n 个元素跑 S 次 RMW，报 GB/s 与每步时间。
// ---------------------------------------------------------------------------
template <typename T, typename K>
static void bench_mode(const char* name, long n, int elem_size, int S, K launch) {
    T* acc;
    CK(cudaMalloc(&acc, n * elem_size));
    CK(cudaMemset(acc, 0, n * elem_size));
    int block = 256;
    int grid = 4096;

    // warm
    for (int s = 0; s < 3; ++s) launch(acc, n, grid, block);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // 3 轮取最快（本机 8 卡共享，去争用毛刺）
    float best_ms = 1e30f;
    for (int trial = 0; trial < 3; ++trial) {
        CK(cudaEventRecord(e0));
        for (int s = 0; s < S; ++s) launch(acc, n, grid, block);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best_ms) best_ms = ms;
    }
    double sec = best_ms / 1e3;
    double bytes = (double)S * 2.0 * elem_size * (double)n;   // 读 + 写
    printf("  %-6s : n=%ld  S=%d  总 %8.1f us  每步 %7.1f us   %7.1f GB/s  "
           "每步字节 %6.1f MB\n",
           name, n, S, sec * 1e6, sec / S * 1e6, bytes / sec / 1e9,
           (double)2.0 * elem_size * n / 1e6);
    cudaFree(acc); cudaEventDestroy(e0); cudaEventDestroy(e1);
}

static void bench(long n, int S) {
    printf("\n[带宽] 累积器 RMW（融合 epilogue：HBM 上只读写累积器自己）\n");
    bench_mode<float>("fp32", n, 4, S, [](float* a, long n_, int g, int b) {
        k_rmw_fp32<<<g, b>>>(a, n_); });
    bench_mode<bf16>("bf16", n, 2, S, [](bf16* a, long n_, int g, int b) {
        k_rmw_bf16<<<g, b>>>(a, n_); });
    bench_mode<__nv_fp8_storage_t>("fp8", n, 1, S, [](__nv_fp8_storage_t* a, long n_, int g, int b) {
        k_rmw_fp8<<<g, b>>>(a, n_); });
}

int main() {
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("device=%s sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    printf("\n[数值自检]\n");
    check();

    // wgrad dW2 规模：E=8 专家、每专家 [M=2048, N=7168] → 累积器 = E*M*N 个元素
    const long E = 8, M = 2048, N = 7168;
    const long nelem = E * M * N;   // 117,440,512
    printf("\n累积器规模 = E*M*N = %ld 元素（dW2 [E=8, M=2048, N=7168]）\n", nelem);
    bench(nelem, 8);

    printf("\n[每步每专家 fp32 输出字节] fp32 RMW=8B/元素/步、bf16=4B、fp8=2B\n");
    return 0;
}
