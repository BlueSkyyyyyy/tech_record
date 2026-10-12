// two_layer_footprint.cu — MegaKernel 专题 P17-1 小实验：
//   **megakernel「双层足迹」的叠加**：寄存器足迹（P14-1）× 代码足迹（P16-1）编进同一个 kernel
//
// 目的（目标卡 H100 / sm90 实测）：
//   P14-1（§8）量化了 megakernel 的**寄存器足迹**——per-thread 寄存器被固定为各 task 类型的最大值
//   （MPK §7 行 1748 的「唯一例外」），把轻 task 的访存带宽拉低 1.33×–2.29×（占用率 32→24→8 warp/SM）。
//   P16-1（§10）量化了**代码足迹**——每个 task 是一个 CUDA device function（MPK §4.2 行 1138 /
//   §7 行 1750），融合的 distinct task 类型越多，被执行代码工作集越大，超出 L1 指令缓存后取指 miss，
//   P=256 时慢 2.18×（no_instruction 停顿 0.63%→59.19%）。
//   两节的「局限」都明确写着**未把两者编进同一个 kernel 测叠加**（§8.6 / §10.6）。真实 megakernel 里
//   这两笔代价**同时都要付**：一个常驻 kernel 里既有寄存器很重的 task，又有许多互异的 task body。
//   本实验把二者放进**同一个 kernel**，量化它们是否（以及如何）叠加。
//
// 做法：
//   一个 kernel `mega<P,H>` 同时包含三条路径，运行时用 `do_mem` / `do_comp` 开关隔离：
//     - **访存路径**（occupancy 敏感，对应 P14-1 的轻 task）：对本 block 分到的段做 grid-stride 求和，
//       读一个远大于 L2 的数组，占用率被拉低就直接掉带宽。
//     - **代码路径**（I-cache 敏感，对应 P16-1）：每 block 以 block 粒度从一张 **P 项函数指针表**里
//       间接调用 `body<0..P-1>`（P 个 SASS 互异的 device function），做满 `steps` 步 —— 一条执行流
//       扫过的代码工作集 ≈ P 个 body。
//     - **重寄存器路径**：`heavy_path<H>` 保持 H 个常驻寄存器（H=0 时该分支被 `if constexpr` 丢弃，
//       kernel 不含重代码），一旦 H>0，整 kernel 的 numRegs 就被抬到 max(轻, H)。
//   P 与 H 都是**编译期**参数，故 `mega<P,H>` 被执行代码集合与寄存器数都精确可控，**工作量与 P、H 无关**。
//
// 三个 Section：
//   A 寄存器 / 占用率表：证明 `mega<P,H>` 的 numRegs = max(代码路径, heavy<H>)，且与 P 无关。
//   B1 访存路径隔离（do_comp=0）：同一段访存，随 H 抬寄存器看带宽掉多少 —— 复现 P14-1 的寄存器足迹。
//   B2 代码路径隔离（do_mem=0）：同工作量，随 P 增大看吞吐掉多少 —— 复现 P16-1 的代码足迹；并看
//      在**低占用率**（H 大）下这个代价是否被放大（取指停顿更藏不住）。
//   B3 叠加（do_mem=do_comp=1）：2×3 网格，比较「两个边际代价之积 / 之和」与实测的总代价，判定
//      双层足迹是**独立相乘、相加、还是相互放大**。
//
// 编译：nvcc -O3 -arch=sm_90 two_layer_footprint.cu -o two_layer_footprint
// 用法：./two_layer_footprint [iters] [n_mem_M] [steps] [device]

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
static const int R = 64;   // 每个 body 的 FMA 次数（名义计算量，与 P 无关）

using fn_ptr = float (*)(float);

// 代码互异的 task body：同样 R 次 FMA + 各自不同的立即数 -> 每个实例都是一段独立 SASS（~1 KB）。
template <int I>
__device__ __noinline__ float body(float x) {
  float a = fmaf(x, 1.0f, 1e-6f * (float)(I + 1));
#pragma unroll
  for (int k = 0; k < R; ++k) a = fmaf(a, 1.0000001f, 1e-7f);
  return a + 1e-9f * (float)(I + 1);
}

// 重寄存器路径：H 个常驻寄存器做长 FMA 链，全部保持 live（P14-1 的 heavy_path）。
// H==0 时该路径被完全丢弃（if constexpr），kernel 不含重代码。
template <int H>
__device__ __forceinline__ float heavy_path(float* __restrict__ scratch, int reps) {
  if constexpr (H == 0) {
    (void)scratch;
    (void)reps;
    return 0.f;
  } else {
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
}

// 编译期递归：只实例化 body<0..N-1> 填进函数指针表。
template <int N>
__device__ void fill_tab(fn_ptr* tab) {
  tab[N - 1] = &body<N - 1>;
  fill_tab<N - 1>(tab);
}
template <>
__device__ void fill_tab<0>(fn_ptr* tab) {}

template <int P>
__global__ void init_tab(fn_ptr* tab) {
  fill_tab<P>(tab);
}

// 融合 kernel：同一 kernel 内含「访存路径 + P 个互异 body 的代码路径 + 重寄存器路径」。
// do_mem / do_comp 运行时开关用于隔离；重寄存器路径只在 H>0 时编入。
template <int P, int H>
__global__ void mega(const float* __restrict__ in, float* __restrict__ out,
                     float* __restrict__ scratch, fn_ptr* __restrict__ tab,
                     long long n_mem, long long W, int steps, int reps,
                     int do_mem, int do_comp) {
  float s = 0.f;

  // 访存路径（occupancy 敏感）
  if (do_mem) {
    long long seg = n_mem / gridDim.x;
    long long b = (long long)blockIdx.x * seg;
    long long e = b + seg;
    for (long long i = b + threadIdx.x; i < e; i += blockDim.x) s += in[i];
  }

  // 代码路径（I-cache 敏感）：block 粒度选 body，同 block 内无 divergence。
  if (do_comp) {
    long long base = (long long)blockIdx.x * steps;
    for (int u = 0; u < steps; ++u) {
      long long sid = base + u;
      long long idx = sid * blockDim.x + threadIdx.x;
      if (idx >= W) break;
      float x = in[idx];
      unsigned h = (unsigned)(sid * 2654435761u) ^ ((unsigned)blockIdx.x * 2246822519u);
      int type = (int)(h % (unsigned)P);
      s += tab[type](x);
    }
  }

  // 重寄存器路径（只在 H>0 时编入；实际计算量极小，只为把 numRegs 抬到 H）
  if constexpr (H > 0) {
    if (reps > 0) s += heavy_path<H>(scratch, reps);
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

static fn_ptr* g_tab = nullptr;
static const int PMAX = 256;

template <int P>
static void setup() {
  init_tab<P><<<1, 1>>>(g_tab);
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <int P, int H>
static float time_case(int iters, int G, long long n_mem, long long W, int steps,
                       int reps, int do_mem, int do_comp, const float* d_in,
                       float* d_out, float* d_scratch) {
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  mega<P, H><<<G, T>>>(d_in, d_out, d_scratch, g_tab, n_mem, W, steps, reps, do_mem,
                       do_comp);
  CUDA_CHECK(cudaDeviceSynchronize());  // warmup
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    mega<P, H><<<G, T>>>(d_in, d_out, d_scratch, g_tab, n_mem, W, steps, reps, do_mem,
                         do_comp);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

template <typename K>
static void report_attrs(const char* name, K kernel) {
  cudaFuncAttributes a;
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int nb = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, T, 0));
  printf("%-20s | %6d | %8zu | %11d | %10d\n", name, a.numRegs, a.localSizeBytes, nb,
         nb * T / 32);
}

// ncu 单项剖析用：只 launch 一次。
template <int P, int H>
static void profile_once(int G, long long n_mem, long long W, int steps, int reps,
                         int do_mem, int do_comp, const float* d_in, float* d_out,
                         float* d_scratch) {
  setup<P>();
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  mega<P, H><<<G, T>>>(d_in, d_out, d_scratch, g_tab, n_mem, W, steps, reps, do_mem,
                       do_comp);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// 打印一行并返回该行耗时，同时记录 ratio 基准。
#define CASE_ROW(Pv, Hv, dm, dc, base_ms, note)                                 \
  {                                                                             \
    float ms = time_case<Pv, Hv>(iters, G, n_mem, W, steps, REPS, dm, dc, d_in, \
                                 d_out, d_scratch);                             \
    printf("%3d | %3d | %11.4f | %10.3fx   %s\n", Pv, Hv, ms,                   \
           ms / (base_ms), note);                                               \
  }

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 30;
  const long long n_mem =
      (argc > 2) ? ((long long)atoi(argv[2]) << 20) : (long long)1 << 27;  // 128Mi float=512MB
  const int steps = (argc > 3) ? atoi(argv[3]) : 512;
  const int dev = (argc > 4) ? atoi(argv[4]) : 3;  // 默认放在空闲的 device 3
  const int REPS = 8;

  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  const int SM = prop.multiProcessorCount;
  const int G = SM * 4;  // 4 block/SM @132 SM = 528（与 P14-1 同，让占用率真正起作用）
  const long long W = (long long)G * steps * T;  // 代码路径元素数

  printf("== device ==\n%s, sm_%d%d, SMs=%d, regs/SM=%d, L2=%.1f MB\n", prop.name,
         prop.major, prop.minor, SM, prop.regsPerMultiprocessor,
         prop.l2CacheSize / 1048576.0);
  printf("blockDim T=%d, R=%d FMA/body, grid G=%d (%d/SM), steps/block=%d, "
         "n_mem=%lld floats (%.0f MB), W=%lld floats, heavy reps=%d, iters=%d\n\n",
         T, R, G, 4, steps, n_mem, n_mem * 4.0 / 1e6, W, REPS, iters);

  CUDA_CHECK(cudaMalloc(&g_tab, PMAX * sizeof(fn_ptr)));

  float* d_in = nullptr;
  float* d_out = nullptr;
  float* d_scratch = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n_mem * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)G * 4096 * sizeof(float)));
  {
    float* h = (float*)malloc(n_mem * sizeof(float));
    for (long long i = 0; i < n_mem; ++i) h[i] = (float)((i % 1024) + 1) * 1e-3f;
    CUDA_CHECK(cudaMemcpy(d_in, h, n_mem * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
  }
  CUDA_CHECK(cudaMemset(d_scratch, 0, (size_t)G * 4096 * sizeof(float)));

  // ---- ncu 单项剖析模式：argv[5]=P, argv[6]=H, argv[7]=do_mem, argv[8]=do_comp ----
  if (argc > 5) {
    const int pv = atoi(argv[5]);
    const int hv = atoi(argv[6]);
    const int dm = (argc > 7) ? atoi(argv[7]) : 1;
    const int dc = (argc > 8) ? atoi(argv[8]) : 1;
    if (pv == 1 && hv == 0) profile_once<1, 0>(G, n_mem, W, steps, REPS, dm, dc, d_in, d_out, d_scratch);
    else if (pv == 1 && hv == 128) profile_once<1, 128>(G, n_mem, W, steps, REPS, dm, dc, d_in, d_out, d_scratch);
    else if (pv == 256 && hv == 0) profile_once<256, 0>(G, n_mem, W, steps, REPS, dm, dc, d_in, d_out, d_scratch);
    else if (pv == 256 && hv == 128) profile_once<256, 128>(G, n_mem, W, steps, REPS, dm, dc, d_in, d_out, d_scratch);
    else printf("unsupported (P,H)=(%d,%d)\n", pv, hv);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(g_tab));
    return 0;
  }

  // ---- 基准与 Section A：寄存器 / 占用率 ----
  setup<1>();
  setup<16>();
  setup<256>();
  printf("== Section A: 寄存器 / 占用率 (T=%d) —— 证明 numRegs = max(代码路径, heavy<H>) ==\n", T);
  printf("%-20s | %6s | %8s | %11s | %10s\n", "kernel", "regs", "localB", "blocks/SM",
         "warps/SM");
  printf("---------------------+--------+----------+-------------+-----------\n");
  report_attrs("mega<1,0>", mega<1, 0>);
  report_attrs("mega<1,64>", mega<1, 64>);
  report_attrs("mega<1,128>", mega<1, 128>);
  report_attrs("mega<256,0>", mega<256, 0>);
  report_attrs("mega<256,64>", mega<256, 64>);
  report_attrs("mega<256,128>", mega<256, 128>);

  // 基准：P=1,H=0, 两条路径全开
  float base_ms = time_case<1, 0>(iters, G, n_mem, W, steps, REPS, 1, 1, d_in, d_out, d_scratch);
  printf("\n== reference ==\nmega<1,0> (全路径, 无足迹代价): %.4f ms\n", base_ms);

  // ---- Section B1：访存路径隔离（do_comp=0）——寄存器足迹 ----
  float mem0 = time_case<1, 0>(iters, G, n_mem, W, steps, REPS, 1, 0, d_in, d_out, d_scratch);
  printf("\n== Section B1: 访存路径隔离 (do_comp=0) —— 寄存器足迹（P14-1 复现） ==\n");
  printf("%3s | %3s | %11s | %10s   %s\n", "P", "H", "ms", "vs base", "note");
  printf("----+----+-------------+-----------\n");
  printf("%3d | %3d | %11.4f | %10.3fx   mem-only baseline %.1f GB/s\n", 1, 0, mem0, 1.0,
         n_mem * 4.0 / (mem0 * 1e-3) / 1e9);
#define MROW(Pv, Hv)                                                            \
  {                                                                             \
    float ms = time_case<Pv, Hv>(iters, G, n_mem, W, steps, REPS, 1, 0, d_in,   \
                                 d_out, d_scratch);                             \
    printf("%3d | %3d | %11.4f | %10.3fx   %.1f GB/s\n", Pv, Hv, ms, ms / mem0, \
           n_mem * 4.0 / (ms * 1e-3) / 1e9);                                    \
  }
  MROW(1, 64)
  MROW(1, 128)
  MROW(256, 128)
#undef MROW

  // ---- Section B2：代码路径隔离（do_mem=0）——代码足迹 + 与占用率的交互 ----
  float comp0 = time_case<1, 0>(iters, G, n_mem, W, steps, REPS, 0, 1, d_in, d_out, d_scratch);
  printf("\n== Section B2: 代码路径隔离 (do_mem=0) —— 代码足迹（P16-1 复现）及低占用率下的放大 ==\n");
  printf("%3s | %3s | %11s | %10s   %s\n", "P", "H", "ms", "vs base", "note");
  printf("----+----+-------------+-----------\n");
  printf("%3d | %3d | %11.4f | %10.3fx   comp-only baseline %.1f GFLOP/s\n", 1, 0, comp0,
         1.0, (double)W * R * 2.0 / (comp0 * 1e-3) / 1e9);
#define CROW(Pv, Hv)                                                            \
  {                                                                             \
    float ms = time_case<Pv, Hv>(iters, G, n_mem, W, steps, REPS, 0, 1, d_in,   \
                                 d_out, d_scratch);                             \
    printf("%3d | %3d | %11.4f | %10.3fx   %.1f GFLOP/s\n", Pv, Hv, ms,         \
           ms / comp0, (double)W * R * 2.0 / (ms * 1e-3) / 1e9);                \
  }
  CROW(1, 64)
  CROW(1, 128)
  CROW(16, 0)
  CROW(256, 0)
  CROW(16, 128)
  CROW(256, 128)
#undef CROW

  // ---- Section B3：叠加（do_mem=do_comp=1） ----
  printf("\n== Section B3: 叠加 (访存+代码全开) —— 双层足迹同时付 ==\n");
  printf("%3s | %3s | %11s | %10s   %s\n", "P", "H", "ms", "vs base", "note");
  printf("----+----+-------------+-----------\n");
  printf("%3d | %3d | %11.4f | %10.3fx   base (P=1,H=0)\n", 1, 0, base_ms, 1.0);
  CASE_ROW(1, 64, 1, 1, base_ms, "reg only")
  CASE_ROW(1, 128, 1, 1, base_ms, "reg only")
  CASE_ROW(16, 0, 1, 1, base_ms, "code only")
  CASE_ROW(256, 0, 1, 1, base_ms, "code only")
  CASE_ROW(16, 128, 1, 1, base_ms, "both")
  CASE_ROW(256, 128, 1, 1, base_ms, "both")

  printf("\n>> 读法：\n");
  printf(">> 定义边际代价：m_code = comp(P=256,H=0)/comp(P=1,H=0)（代码足迹）；\n");
  printf(">>              m_reg  = mem(P=1,H=128)/mem(P=1,H=0)（寄存器足迹，作用在访存路径）。\n");
  printf(">> B3 的 both 行若 ≈ m_code×m_reg 的加权组合 -> 两笔代价近似独立叠加；\n");
  printf(">> 若 B2 里同一个 P 在 H=128（低占用率）下比 H=0 更慢 -> 低占用率放大了取指停顿（交互项）。\n");
  printf(">> 核心问题：把重寄存器 task 与大量互异 task body 编进同一 kernel，代价是否只是简单相加。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scratch));
  CUDA_CHECK(cudaFree(g_tab));
  return 0;
}
