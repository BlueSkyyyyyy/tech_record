// setmaxnreg_rebalance.cu — MegaKernel 专题 P18-1 小实验：
//   **megakernel 寄存器足迹的「可绕开性」：setmaxnreg warp 级寄存器再平衡**
//
// 背景（承接 P14-1，`docs/04-experiments.md` §8）：
//   MPK §7 (论文快照行 1740–1748) 指出 mega-kernel 的唯一不可分页足迹是 per-thread 寄存器，
//   它被固定为「各 task 类型的最大值」。P14-1 已量化：把一个重 task 编进单 kernel 后，轻访存
//   task 的带宽损失 1.33×–2.29×（占用率被拉低）。P14-1 §8.5/§8.6 留下尾巴：生产案例（DeepGEMM
//   mega 家族）用手工 **warp 角色 + setmaxnreg 式动态寄存器重配置** 在工程上绕开这一项，
//   但本专题「未测 setmaxnreg 动态重配置（只在 docs/02 资料层面）」。本实验补上这一格。
//
// 机制（Hopper sm90a）：
//   `setmaxnreg.inc/dec.sync.aligned.u32 N` 让**一个 warpgroup（128 线程）**动态增减自己的寄存器
//   分配（N ∈ [24,256]，8 的倍数）。配合 `__launch_bounds__(T, minBlocks)`——它给 kernel 一个
//   **较低的「入口」寄存器数**（= 65536/(T*minBlocks)），从而算出更高的 blocks/SM 占用率；
//   运行时重 task 的 warpgroup `inc` 到高位、其余 warpgroup `dec` 到低位，**在同一 block 内换手**，
//   ptxas 因此在编译期不再给重路径 spill。这正是 CUTLASS/DeepGEMM warp-specialized kernel 的做法
//   （`docs/02` §1.2/§1.3；线上 MPK `tasks/hopper/utils.cuh` 的 `wg_increase/decrease_regs`
//   包 setmaxnreg，但在 `linear_hopper.cuh` 里被注释掉，见 `docs/07` §2）。
//
// 对照（同一「每 block：读整段 + 做 heavy 计算」的融合工作负载）：
//   - separate     ：两次 launch（light_kernel + heavy_kernel），各自按自己占用率跑（kernel-per-operator 上界）
//   - fused_naive  ：单 kernel，所有线程都背 heavy 的寄存器（P14-1 的朴素融合，occupancy 被拉低）
//   - fused_ws0    ：单 kernel，warp-specialized（WG0 heavy / WG1 light），**不调** setmaxnreg（会 spill）
//   - fused_ws1    ：单 kernel，warp-specialized + **setmaxnreg**（WG0 inc / WG1 dec）
//
// 编译（setmaxnreg 只在 sm90a 上可用）：
//   nvcc -O3 -gencode arch=compute_90a,code=sm_90a setmaxnreg_rebalance.cu -o setmaxnreg_rebalance
// 用法：./setmaxnreg_rebalance [iters] [n_light_M] [G]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

static const int T = 256;             // 每 block 线程数
static const unsigned INC_REG = 232;  // 重 warpgroup 目标寄存器（≤ sm90 上限）
static const unsigned DEC_REG = 24;   // 轻 warpgroup 让出的目标（≥ setmaxnreg 下限）

__device__ __forceinline__ void smn_inc_232() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(INC_REG));
}
__device__ __forceinline__ void smn_dec_24() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(DEC_REG));
}

// 重计算路径：H 个常驻寄存器、长 FMA 链，把 H 个寄存器全部保持 live。
template <int H>
__device__ __forceinline__ float heavy_path(float* __restrict__ scratch, int reps) {
  float r[H];
#pragma unroll
  for (int k = 0; k < H; ++k) r[k] = (float)(threadIdx.x + k) * 1e-6f + 1.0f;
  for (int rr = 0; rr < reps; ++rr) {
#pragma unroll
    for (int k = 0; k < H; ++k) r[k] = fmaf(r[k], 1.0000001f, 1e-7f);
  }
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < H; ++k) t += r[k];
  if (threadIdx.x == 0) scratch[blockIdx.x] = t;  // 防优化
  return t;
}

// 轻访存路径（全 block 线程）：block 内所有线程按 blockDim.x 跨步。
__device__ __forceinline__ float light_all(const float* __restrict__ in, long long b,
                                           long long e) {
  float s = 0.f;
  for (long long i = b + threadIdx.x; i < e; i += blockDim.x) s += in[i];
  return s;
}

// 轻访存路径（仅一个 warpgroup，128 线程）：用 group 内 local lane 跨步 —— 覆盖同一段 [b,e)。
__device__ __forceinline__ float light_wg(const float* __restrict__ in, long long b,
                                          long long e) {
  float s = 0.f;
  int lane = threadIdx.x & 127;  // warpgroup 内 local lane
  for (long long i = b + lane; i < e; i += 128) s += in[i];
  return s;
}

__device__ __forceinline__ float block_reduce(float s) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  return s;
}

// ---- 参照 kernel ----
__global__ void light_kernel(const float* __restrict__ in, float* __restrict__ out,
                             long long n) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg, e = b + seg;
  float s = light_all(in, b, e);
  s = block_reduce(s);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

template <int H>
__global__ void heavy_kernel(float* __restrict__ scratch, float* __restrict__ out,
                             long long n, int reps) {
  float s = heavy_path<H>(scratch, reps);
  s = block_reduce(s);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// ---- 朴素融合（P14-1 形态）：所有线程都背 heavy<H> 的寄存器 ----
// heavy=0 时只走轻路径，但寄存器已被抬到 max（用于 Section B 隔离「寄存器足迹」）。
template <int H>
__global__ void fused_naive_kernel(const float* __restrict__ in, float* __restrict__ out,
                                   float* __restrict__ scratch, long long n, int heavy,
                                   int reps) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg, e = b + seg;
  float s = light_all(in, b, e);
  if (heavy) s += heavy_path<H>(scratch, reps);
  s = block_reduce(s);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// ---- warp-specialized 融合：WG0 做 heavy，WG1 做 light；可选 setmaxnreg ----
// USE=0：不调 setmaxnreg（编译期重路径会 spill）；USE=1：WG0 inc / WG1 dec。
// `__launch_bounds__(T,MINB)` 把「入口」寄存器压到 65536/(T*MINB)，占用率由它决定。
template <int H, int MINB, int USE>
__global__ void __launch_bounds__(T, MINB)
fused_ws_kernel(const float* __restrict__ in, float* __restrict__ out,
                float* __restrict__ scratch, long long n, int reps) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg, e = b + seg;
  int wg = threadIdx.x >> 7;
  float s = 0.f;
  if (wg == 0) {                 // 重 warpgroup
    if (USE) smn_inc_232();
    s += heavy_path<H>(scratch, reps);
  } else {                       // 轻 warpgroup：覆盖整段
    if (USE) smn_dec_24();
    s += light_wg(in, b, e);
  }
  s = block_reduce(s);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// 只用 WG1（128 线程）做轻访存的 kernel —— 用于校验 light_wg 读取的数据与 light_kernel 一致。
__global__ void light_wg_kernel(const float* __restrict__ in, float* __restrict__ out,
                                long long n) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg, e = b + seg;
  float s = 0.f;
  if ((threadIdx.x >> 7) == 1) s = light_wg(in, b, e);
  s = block_reduce(s);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

template <typename K>
static void report_attrs(const char* name, K kernel) {
  cudaFuncAttributes a;
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int nb = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, T, 0));
  printf("%-26s | %5d | %9zu | %9d | %8d\n", name, a.numRegs, a.localSizeBytes, nb,
         nb * T / 32);
}

// ---- 计时包装 ----
template <typename Launch>
static float timeit(int iters, Launch launch) {
  launch();  // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) launch();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 30;
  const long long n = (argc > 2) ? (long long)atoi(argv[2]) << 20
                                 : (long long)1 << 27;  // 128Mi float = 512MB
  const int G = (argc > 3) ? atoi(argv[3]) : 528;
  const int REPS = 200;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d, regs/SM=%d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount, prop.regsPerMultiprocessor);
  printf("grid G=%d, T=%d, n=%lld floats (%.0f MB), heavy reps=%d, iters=%d\n",
         G, T, n, n * 4.0 / 1e6, REPS, iters);
  printf("setmaxnreg: WG0 inc->%u, WG1 dec->%u ; launch_bounds minBlocks=2\n\n",
         INC_REG, DEC_REG);

  float* d_in = nullptr;
  float* d_out = nullptr;
  float* d_scratch = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)G * 4096 * sizeof(float)));
  {
    float* h = (float*)malloc(n * sizeof(float));
    for (long long i = 0; i < n; ++i) h[i] = (float)((i % 1024) + 1) * 1e-3f;
    CUDA_CHECK(cudaMemcpy(d_in, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
  }
  const double bytes = n * 4.0;

  // ---- Section A：寄存器 / 溢出 / 占用率 ----
  printf("== Section A: 寄存器 / 溢出 / 占用率 (T=%d) ==\n", T);
  printf("%-26s | %5s | %9s | %9s | %8s\n", "kernel", "regs", "localB", "blocks/SM",
         "warps/SM");
  printf("---------------------------+-------+-----------+-----------+----------\n");
  report_attrs("light_kernel", light_kernel);
  report_attrs("heavy_kernel<128>", heavy_kernel<128>);
  report_attrs("heavy_kernel<192>", heavy_kernel<192>);
  report_attrs("heavy_kernel<232>", heavy_kernel<232>);
  report_attrs("fused_naive<128>", fused_naive_kernel<128>);
  report_attrs("fused_naive<192>", fused_naive_kernel<192>);
  report_attrs("fused_naive<232>", fused_naive_kernel<232>);
  report_attrs("ws<128,lb2,U0>", fused_ws_kernel<128, 2, 0>);
  report_attrs("ws<128,lb2,U1>", fused_ws_kernel<128, 2, 1>);
  report_attrs("ws<192,lb2,U1>", fused_ws_kernel<192, 2, 1>);
  report_attrs("ws<232,lb2,U1>", fused_ws_kernel<232, 2, 1>);

  // ---- Section B：轻访存带宽 —— 隔离「寄存器足迹」（复现 P14-1 §8-B） ----
  printf("\n== Section B: 轻访存带宽 —— fused_naive<H>(heavy=0) vs light_kernel ==\n");
  printf("%6s | %11s | %11s | %9s | %8s\n", "H", "light ms", "fused ms", "GB/s",
         "比率");
  printf("-------+-------------+-------------+-----------+---------\n");
  {
    float base = timeit(iters, [&] { light_kernel<<<G, T>>>(d_in, d_out, n); });
    printf("%6s | %11.4f | %11.4f | %9.1f | %7.2fx\n", "base", base, base,
           bytes / (base * 1e-3) / 1e9, 1.0);
    auto row = [&](int H, auto kern, const char* tag) {
      float ms = timeit(iters, [&] {
        kern<<<G, T>>>(d_in, d_out, d_scratch, n, 0, REPS);
      });
      printf("%6d | %11.4f | %11.4f | %9.1f | %7.2fx  %s\n", H, base, ms,
             bytes / (ms * 1e-3) / 1e9, ms / base, tag);
    };
    row(64, fused_naive_kernel<64>, "naive");
    row(128, fused_naive_kernel<128>, "naive");
    row(192, fused_naive_kernel<192>, "naive");
    row(232, fused_naive_kernel<232>, "naive");
  }

  // ---- Section C：融合工作负载总时间（读整段 + heavy 计算） ----
  //   工作负载对齐：每 block 读整段 [b,e) + 做「256 线程 × REPS」量的 heavy 计算。
  //   naive：256 线程各 heavy REPS 次；ws：128 个 heavy 线程各 2*REPS 次（同总 FLOPs）。
  printf("\n== Section C: 融合工作负载（读整段 + heavy）总时间 ==\n");
  printf("%6s | %11s | %11s | %11s | %11s | %8s %8s\n", "H", "separate", "naive",
         "ws(U0)", "ws(U1)", "U1/sep", "U1/nv");
  printf("-------+-------------+-------------+-------------+-------------+--------+--------\n");
  auto crow = [&]<int H>(auto naive_kern, auto ws0, auto ws1) {
    float sep = timeit(iters, [&] {
      light_kernel<<<G, T>>>(d_in, d_out, n);
      heavy_kernel<H><<<G, T>>>(d_scratch, d_out, n, REPS);
    });
    float nv = timeit(iters, [&] {
      naive_kern<<<G, T>>>(d_in, d_out, d_scratch, n, 1, REPS);
    });
    float w0 = timeit(iters, [&] {
      ws0<<<G, T>>>(d_in, d_out, d_scratch, n, 2 * REPS);
    });
    float w1 = timeit(iters, [&] {
      ws1<<<G, T>>>(d_in, d_out, d_scratch, n, 2 * REPS);
    });
    printf("%6d | %11.4f | %11.4f | %11.4f | %11.4f | %7.2fx %7.2fx\n", H, sep, nv,
           w0, w1, w1 / sep, w1 / nv);
  };
  crow.operator()<128>(fused_naive_kernel<128>, fused_ws_kernel<128, 2, 0>,
                       fused_ws_kernel<128, 2, 1>);
  crow.operator()<192>(fused_naive_kernel<192>, fused_ws_kernel<192, 2, 0>,
                       fused_ws_kernel<192, 2, 1>);
  crow.operator()<232>(fused_naive_kernel<232>, fused_ws_kernel<232, 2, 0>,
                       fused_ws_kernel<232, 2, 1>);

  printf("\n>> 读法：\n");
  printf(">> A: fused_naive<H> 的 numRegs=max(light,heavy<H>)、occupancy 随最重 task 下降（P14-1）。\n");
  printf(">>    ws<H,lb2,U1> 的 numRegs 被 __launch_bounds__ 压到 128（2 block/SM），\n");
  printf(">>    但重路径无 spill —— setmaxnreg 让 WG0 运行时 inc 到 232 而占用率仍按 128 算。\n");
  printf(">> B: 同样的轻访存，naive 融合因占用率低而掉带宽（复现 P14-1 §8-B）。\n");
  printf(">> C: 融合工作负载总时间 —— ws(U1) 相对 naive 的加速即 setmaxnreg 回收的寄存器足迹代价；\n");
  printf(">>    ws(U0) 是否更慢 = 不调 setmaxnreg 时重路径 spill 的代价。\n");

  // ---- 正确性：light_wg（WG1 读取）与 light_kernel（全线程读取）应一致 ----
  {
    float a = 0.f, b = 0.f;
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    light_kernel<<<G, T>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&a, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    light_wg_kernel<<<G, T>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&b, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    double rel = (a != 0) ? (double)(b - a) / (double)a : 0.0;
    printf("\n>> correctness: light_kernel=%.4f light_wg(WG1)=%.4f rel=%.3e\n", a, b, rel);

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    fused_ws_kernel<128, 2, 1><<<G, T>>>(d_in, d_out, d_scratch, n, 2 * REPS);
    CUDA_CHECK(cudaDeviceSynchronize());
    float o = 0;
    CUDA_CHECK(cudaMemcpy(&o, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf(">> correctness: ws<128,U1> fused out=%.4f (finite=%d)\n", o,
           (o == o) && (o < 1e30f && o > -1e30f));
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scratch));
  return 0;
}
