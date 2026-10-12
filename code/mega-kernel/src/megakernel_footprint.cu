// megakernel_footprint.cu — MegaKernel 专题 P14-1 小实验：
//   **megakernel 的资源足迹代价**：寄存器被固定为「各 task 类型所需的最大值」
//
// 目的（目标卡 H100 / sm90 实测）：
//   MPK §7 Discussion「Resource footprint」（论文快照行 1740–1748）指出，mega-kernel 的常见顾虑
//   是「单个 kernel 必须在每个 SM 上预留所有被融合算子的合并资源」。MPK 用**时分复用**（同一 SM 依次
//   执行不同 task，行 1742）化解了大部分：shared memory 用 **paged abstraction**（§5.3，行 1313–1325）
//   按 task 粒度申请/释放，因此不按最大值常驻。**唯一的例外是 per-thread 寄存器**：
//     "The main exception is per-thread register usage, which is fixed for the mega-kernel
//      at the maximum required across task types."（行 1748）
//   也就是说，一旦把一个「寄存器需求很重」的 task 和一个「很轻」的 task 编进同一个 megakernel，
//   编译出的 kernel 寄存器数 = 两者最大值，**所有 task（包括轻的）都按重 task 的占用率运行**，
//   轻 task 的驻留 warp 数被拉低，访存延迟藏不住，吞吐下降。这是被融合算子异构性带来的真实代价，
//   此前只有资料分析（docs/01 §5.8），本实验把它**量化出来**。
//
// 做法：
//   - `light_kernel`：只有访存路径（grid-stride 求和），寄存器轻、访存受限，占用率越高带宽越高。
//   - `heavy_kernel<H>`：只有计算路径，H 个常驻寄存器做长 FMA 链，寄存器重（H 越大越重）。
//   - `fused_light_kernel<H>`：**同一个 kernel** 内含「轻访存路径 + 重计算路径（运行时 heavy 开关）」，
//     模拟 megakernel 把两类 task 编进一个 kernel。运行时只走轻路径（heavy=0），但寄存器数已被
//     重路径抬到 max —— 这样「同样的轻工作，唯一变量是 kernel 里有没有重 task」。
//   - `fused_two_kernel<H>`：每 block 依次做「轻访存 + 重计算」两个阶段，模拟 MPK 的**时分复用**
//     （同一 SM 先跑轻 task 再跑重 task）；对照 `light_kernel` + `heavy_kernel<H>` 两次 launch
//     （kernel-per-operator，各自按自己的寄存器占用率跑）。
//
// 三个 Section：
//   A 寄存器/占用率表：light / heavy<H> / fused<H> 的 numRegs、溢出、blocks/SM、warps/SM。
//   B 轻路径代价扫描：同样的轻访存工作，`fused_light_kernel<H>`（heavy=0）vs `light_kernel`，
//     随 H 增大，带宽/时间怎么变 —— 隔离出「寄存器足迹」这一项。
//   C 时分复用场景：`fused_two_kernel<H>`（单 kernel，轻+重）vs 两次 launch（light+heavy），
//     即 megakernel vs kernel-per-operator 的真实工作负载对比。
//
// 编译：nvcc -O3 -arch=sm_90 megakernel_footprint.cu -o megakernel_footprint
// 用法：./megakernel_footprint [iters] [n_light_M]

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

static const int T = 256;  // 每 block 线程数

// 重计算路径：H 个常驻寄存器，长 FMA 链，把 H 个寄存器全部保持 live。
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

// 轻访存路径：grid-stride 求和，block 内归约。
__device__ __forceinline__ float light_path(const float* __restrict__ in,
                                            long long seg_begin, long long seg_end) {
  float s = 0.f;
  for (long long i = seg_begin + threadIdx.x; i < seg_end; i += blockDim.x) s += in[i];
  return s;
}

__device__ __forceinline__ float block_reduce(float s, float* __restrict__ out) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  return s;
}

// 只有轻路径的基准 kernel（不含任何重 task 代码）。
__global__ void light_kernel(const float* __restrict__ in, float* __restrict__ out,
                             long long n) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg;
  long long e = b + seg;
  float s = light_path(in, b, e);
  s = block_reduce(s, out);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// 只有重路径的 kernel。
template <int H>
__global__ void heavy_kernel(float* __restrict__ scratch, float* __restrict__ out,
                             long long n, int reps) {
  float s = heavy_path<H>(scratch, reps);
  s = block_reduce(s, out);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// 融合 kernel：同一个 kernel 内含轻、重两条路径。运行时 heavy=0 只走轻路径，
// 但寄存器数已被重路径抬到 max —— 用来隔离「寄存器足迹」一项。
template <int H>
__global__ void fused_light_kernel(const float* __restrict__ in,
                                   float* __restrict__ out, float* __restrict__ scratch,
                                   long long n, int heavy, int reps) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg;
  long long e = b + seg;
  float s = light_path(in, b, e);
  if (heavy) s += heavy_path<H>(scratch, reps);
  s = block_reduce(s, out);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// 时分复用 kernel：每 block 依次做「轻访存 + 重计算」两阶段（模拟 MPK §7 行 1742 的时分复用）。
template <int H>
__global__ void fused_two_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, float* __restrict__ scratch,
                                 long long n, int reps) {
  long long seg = n / gridDim.x;
  long long b = (long long)blockIdx.x * seg;
  long long e = b + seg;
  float s = light_path(in, b, e);
  s += heavy_path<H>(scratch, reps);
  s = block_reduce(s, out);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

template <typename K>
static void report_attrs(const char* name, K kernel) {
  cudaFuncAttributes a;
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int nb = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, T, 0));
  printf("%-22s | %6d | %8zu | %11d | %10d\n", name, a.numRegs,
         a.localSizeBytes, nb, nb * T / 32);
}

static float time_light_base(int iters, int G, long long n, const float* d_in,
                             float* d_out) {
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  light_kernel<<<G, T>>>(d_in, d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) light_kernel<<<G, T>>>(d_in, d_out, n);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

template <typename K>
static float time_light(int iters, int G, long long n, K kernel, const float* d_in,
                        float* d_out, float* d_scratch, int heavy, int reps) {
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  kernel<<<G, T>>>(d_in, d_out, d_scratch, n, heavy, reps);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) kernel<<<G, T>>>(d_in, d_out, d_scratch, n, heavy, reps);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

template <int H>
static float time_heavy(int iters, int G, long long n, const float* d_in,
                        float* d_out, float* d_scratch, int reps) {
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  heavy_kernel<H><<<G, T>>>(d_scratch, d_out, n, reps);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) heavy_kernel<H><<<G, T>>>(d_scratch, d_out, n, reps);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

// 两次 launch（kernel-per-operator）：先 light 再 heavy，各自按自己的占用率跑。
template <int H>
static float time_separate(int iters, int G, long long n, const float* d_in,
                           float* d_out, float* d_scratch, int reps) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int r = 0; r < 2; ++r) {  // warmup
    light_kernel<<<G, T>>>(d_in, d_out, n);
    heavy_kernel<H><<<G, T>>>(d_scratch, d_out, n, reps);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) {
    light_kernel<<<G, T>>>(d_in, d_out, n);
    heavy_kernel<H><<<G, T>>>(d_scratch, d_out, n, reps);
  }
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

// 单 kernel 时分复用：light+heavy 在一个 kernel 里。
template <int H>
static float time_fused_two(int iters, int G, long long n, const float* d_in,
                            float* d_out, float* d_scratch, int reps) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int r = 0; r < 2; ++r) fused_two_kernel<H><<<G, T>>>(d_in, d_out, d_scratch, n, reps);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    fused_two_kernel<H><<<G, T>>>(d_in, d_out, d_scratch, n, reps);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

// 占用率探测用的平凡 kernel（只为读 occupancy API，不做实际工作）。
__global__ void smem_probe(float* __restrict__ out) {
  extern __shared__ float s[];
  if (threadIdx.x == 0 && blockIdx.x == 0xffffffffu) out[0] = s[0];
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 30;
  const long long n = (argc > 2) ? (long long)atoi(argv[2]) << 20
                                 : (long long)1 << 27;  // 默认 128Mi float = 512MB
  const int G = (argc > 3) ? atoi(argv[3]) : 528;  // 4 block/SM @132 SM，让占用率真正起作用
  const int REPS = 200;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d, regs/SM=%d, smem/SM=%zu B\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor);
  printf("grid G=%d, blockDim T=%d, light n=%lld floats (%.0f MB), "
         "heavy reps=%d, iters=%d\n\n",
         G, T, n, n * 4.0 / 1e6, REPS, iters);

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
  CUDA_CHECK(cudaMemset(d_scratch, 0, (size_t)G * 4096 * sizeof(float)));

  const double bytes = n * 4.0;

  // ---- 参照：纯 light 基准 ----
  float base_ms = time_light_base(iters, G, n, d_in, d_out);
  printf("== reference ==\n");
  printf("light_kernel (全占用): %.4f ms  ->  %.1f GB/s (HBM read)\n\n",
         base_ms, bytes / (base_ms * 1e-3) / 1e9);

  // ---- Section A：寄存器 / 占用率表 ----
  printf("== Section A: 寄存器 / 占用率 (T=%d) ==\n", T);
  printf("%-22s | %6s | %8s | %11s | %10s\n", "kernel", "regs", "localB",
         "blocks/SM", "warps/SM");
  printf("-----------------------+--------+----------+-------------+-----------\n");
  report_attrs("light_kernel", light_kernel);
  report_attrs("heavy_kernel<32>", heavy_kernel<32>);
  report_attrs("heavy_kernel<64>", heavy_kernel<64>);
  report_attrs("heavy_kernel<128>", heavy_kernel<128>);
  report_attrs("heavy_kernel<192>", heavy_kernel<192>);
  report_attrs("fused_light<32>", fused_light_kernel<32>);
  report_attrs("fused_light<64>", fused_light_kernel<64>);
  report_attrs("fused_light<128>", fused_light_kernel<128>);
  report_attrs("fused_light<192>", fused_light_kernel<192>);

  // ---- Section B：轻路径代价扫描（同一轻工作，kernel 里带不同重 task） ----
  printf("\n== Section B: 轻访存路径的代价 —— fused<H>(heavy=0) vs light_kernel ==\n");
  printf("%6s | %11s | %11s | %9s | %9s | %9s\n", "H", "light ms", "fused ms",
         "GB/s", "比率", "vs基线");
  printf("-------+-------------+-------------+-----------+-----------+----------\n");
  printf("%6s | %11.4f | %11.4f | %9.1f | %8.2fx | %8.2fx\n", "base", base_ms,
         base_ms, bytes / (base_ms * 1e-3) / 1e9, 1.0, 1.0);
#define BROW(Hv)                                                               \
  {                                                                            \
    float fm = time_light(iters, G, n, fused_light_kernel<Hv>, d_in, d_out,    \
                          d_scratch, 0, REPS);                                 \
    printf("%6d | %11.4f | %11.4f | %9.1f | %8.2fx | %8.2fx\n", Hv, base_ms,  \
           fm, bytes / (fm * 1e-3) / 1e9, fm / base_ms, fm / base_ms);         \
  }
  BROW(16)
  BROW(32)
  BROW(48)
  BROW(64)
  BROW(96)
  BROW(128)
  BROW(144)
  BROW(192)
#undef BROW

  // ---- Section C：时分复用场景（megakernel vs 两次 launch） ----
  printf("\n== Section C: 轻+重时分复用 —— 单 kernel(fused_two) vs 两次 launch ==\n");
  printf("%6s | %12s | %12s | %9s | %9s\n", "H", "separate ms", "fused ms",
         "比率", "heavy ms");
  printf("-------+--------------+--------------+-----------+-----------\n");
#define CROW(Hv)                                                               \
  {                                                                            \
    float sep = time_separate<Hv>(iters, G, n, d_in, d_out, d_scratch, REPS);  \
    float fus = time_fused_two<Hv>(iters, G, n, d_in, d_out, d_scratch, REPS); \
    float hm = time_heavy<Hv>(iters, G, n, d_in, d_out, d_scratch, REPS);      \
    printf("%6d | %12.4f | %12.4f | %8.2fx | %9.4f\n", Hv, sep, fus,          \
           fus / sep, hm);                                                     \
  }
  CROW(32)
  CROW(64)
  CROW(128)
  CROW(192)
#undef CROW

  // ---- Section D：shared memory 足迹（对照：MPK 用 paged SMEM 规避，寄存器无法规避） ----
  printf("\n== Section D: shared memory 足迹的占用率影响（SMEM/SM=%zu B） ==\n",
         prop.sharedMemPerMultiprocessor);
  printf("%10s | %11s | %11s | %11s\n", "smem/block", "smem/blk", "blocks/SM",
         "smem used/SM");
  printf("-----------+-------------+-------------+------------\n");
  CUDA_CHECK(cudaFuncSetAttribute(smem_probe,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  227 * 1024));
  for (size_t kb : {(size_t)0, (size_t)8, (size_t)16, (size_t)32, (size_t)64,
                    (size_t)100, (size_t)160, (size_t)220}) {
    size_t sh = kb * 1024;
    int nb = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, smem_probe, T, sh));
    printf("%7zu KB | %11zu | %11d | %8zu KB\n", kb, sh, nb, kb * (size_t)nb);
  }

  printf("\n>> 读法：\n");
  printf(">> A: fused<H> 的 numRegs = max(light, heavy<H>)，即 megakernel 的 per-thread 寄存器\n");
  printf(">>    被固定为各 task 最大值（MPK §7 行 1748）；occupancy 随最重 task 下降。\n");
  printf(">> B: 同样的轻访存工作，只要 kernel 里编进了重 task，轻 task 的带宽就下降 —— 隔离出\n");
  printf(">>    寄存器足迹这一项的代价（占用率低 -> 驻留 warp 少 -> 藏不住 HBM 延迟）。\n");
  printf(">> C: 时分复用（每 block 轻后重）下单 kernel vs 两次 launch：单 kernel 的轻阶段被重 task\n");
  printf(">>    的寄存器拖累，而两次 launch 让 light 按自己的高占用率跑。差值即 megakernel 此项代价。\n");
  printf(">> D: SMEM 足迹同样会压占用率（若按各 task 最大值常驻）；MPK 用 paged SMEM(§5.3 行 1313-1325)\n");
  printf(">>    按 task 粒度申请/释放来规避，**唯独寄存器不能规避**（§7 行 1748）。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scratch));
  return 0;
}
