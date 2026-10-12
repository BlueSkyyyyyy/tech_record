// code_footprint.cu — MegaKernel 专题 P16-1 小实验：
//   **megakernel 的代码足迹代价**：整图融合 = 许多「per-task device 函数」挤进一个 kernel 镜像
//
// 目的（目标卡 H100 / sm90 实测）：
//   MPK 把每个 task 编译成一个 **CUDA device function**（论文 §4.2 行 1138：以 thread block 为单位
//   复用 Mirage superoptimizer 生成 CUDA 实现），整张 tGraph 的 task 都进同一个常驻 kernel 镜像。
//   §7 Discussion 也点明「Each MPK task is a CUDA device function … per-task code generation is the
//   only hardware-specific component」（快照行 1750–1764）。于是**融合的算子/task 类型越多，单个 kernel
//   镜像里的代码就越多**。寄存器足迹 P14-1 已量化（MPK §7 行 1748 的「唯一例外」）；但**指令足迹**
//   （instruction footprint / L1 instruction cache 压力）是另一条、且同样不能分页的代价：
//   某个 SM 在一段时间里若在多个 task body 之间来回执行（同 SM 上并发的 block 各跑不同 task，或同一
//   block 依次跑很多 task），被执行的代码工作集超过 L1i 容量，I-fetch 就会 miss，出现取指停顿。
//   这与「寄存器被抬到各 task 最大值」是两条独立的融合代价。
//
// 做法：
//   - 生成 P 个**代码互异**的 task body：`body<I>` 是 `__device__ __noinline__` 模板函数，
//     每个做同样 R 次 FMA（名义计算量相同），但带各自的立即数常量，保证 SASS 互不折叠（代码量 ~1.6 KB/个）。
//   - `mega<P>` 是同一个 kernel：每个 block 以 **block 粒度**（无 warp divergence）按
//     `hash(uid, blockIdx) % P` 从函数指针表里选一个 body 调用，做满 `units_per_block` 次。
//     **总工作量与 P 无关**（每 unit 一次 body 调用、R 次 FMA）；唯一变量是 **P**，即被执行代码的工作集大小。
//   - 分发是**一次间接 call**（函数指针表），其开销与 P 无关；P=1 时表里只有一个 body（代码极小），
//     作为「不做融合 / 只有一个 task 类型」的基准。
//   - `fill_tab<P>` 用编译期递归只实例化 `body<0..P-1>`，因此每个 `mega<P>` 实际会被执行的代码集合
//     严格是前 P 个 body —— P 就是「融合了多少种 task 类型」。
//
// 三个 Section：
//   A 代码足迹扫描：固定总工作量，扫 P=1…256，看吞吐/耗时怎么随被执行代码工作集增长。
//   B 并发度依赖：固定 P=256，扫同 SM 并发 block 数 B（1/2/4/8），看「同时跑多个不同 task」是否加剧取指压力。
//   C ncu 归因：对 P=1 与 P=256 采 instruction-fetch / branch-resolving 停顿占比，把 I-cache 项分离出来。
//
// 编译：nvcc -O3 -arch=sm_90 code_footprint.cu -o code_footprint
// 用法：./code_footprint [iters] [units_per_block] [B]

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

// 代码互异的 task body。相同 R 次 FMA + 各不相同的立即数 -> 每个实例都是一段独立 SASS。
template <int I>
__device__ __noinline__ float body(float x) {
  float a = fmaf(x, 1.0f, 1e-6f * (float)(I + 1));
#pragma unroll
  for (int k = 0; k < R; ++k) a = fmaf(a, 1.0000001f, 1e-7f);
  return a + 1e-9f * (float)(I + 1);
}

// 编译期递归：只实例化 body<0..N-1>，把它们填进函数指针表。
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

// 融合 kernel：一个 kernel 内含 P 个互异 task body，运行时按 block 选类型调用。
// 每个 block 处理 `steps` 个「步」，每步全 block T 个线程各处理一个互异元素、但**整块走同一个
// body**（类型只由 step/block 决定，不含 threadIdx）——因此 block 内无 warp divergence。
// 总工作量恒定（每元素一次 body 调用、R 次 FMA）；唯一变量 P = 被执行代码工作集。
template <int P>
__global__ void mega(const float* __restrict__ in, float* __restrict__ out,
                     fn_ptr* __restrict__ tab, long long W, int steps,
                     int sticky) {
  long long base = (long long)blockIdx.x * steps;
  // sticky：每个 block 固定用 blockIdx % P 这一个 body（每条执行流的代码工作集恒为 1 个 body）。
  // 否则：类型随步号轮转（一条执行流会扫过 P 个 body）——这正是「融合很多 task 类型」的形态。
  int sticky_type = (int)((unsigned)blockIdx.x % (unsigned)P);
  float acc = 0.f;
  for (int u = 0; u < steps; ++u) {
    long long sid = base + u;                       // 全 block 共享的步号
    long long idx = sid * T + threadIdx.x;          // 该线程负责的元素
    if (idx >= W) break;
    float x = in[idx];
    // 类型不含 threadIdx：同一 block 内所有线程走同一 body（无 divergence）。
    unsigned h = (unsigned)(sid * 2654435761u) ^ ((unsigned)blockIdx.x * 2246822519u);
    int type = sticky ? sticky_type : (int)(h % (unsigned)P);
    acc += tab[type](x);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

static fn_ptr* g_tab = nullptr;
static const int PMAX = 256;

template <int P>
static void setup() {
  init_tab<P><<<1, 1>>>(g_tab);
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <int P>
static float run_once(int G, long long W, int steps, int sticky, const float* d_in,
                      float* d_out) {
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  mega<P><<<G, T>>>(d_in, d_out, g_tab, W, steps, sticky);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  mega<P><<<G, T>>>(d_in, d_out, g_tab, W, steps, sticky);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms;
}

template <int P>
static float timed(int iters, int G, long long W, int steps, int sticky,
                   const float* d_in, float* d_out) {
  setup<P>();
  run_once<P>(G, W, steps, sticky, d_in, d_out);  // warmup
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    mega<P><<<G, T>>>(d_in, d_out, g_tab, W, steps, sticky);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

// ncu 单项剖析用：只 launch 一次 mega<P>。
template <int P>
static void profile_once(int G, long long W, int steps, int sticky,
                         const float* d_in, float* d_out) {
  setup<P>();
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  mega<P><<<G, T>>>(d_in, d_out, g_tab, W, steps, sticky);
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <int P>
static float timed_s(int iters, int G, long long W, int steps, const float* d_in,
                     float* d_out) {
  return timed<P>(iters, G, W, steps, 0, d_in, d_out);
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 30;
  const int upb = (argc > 2) ? atoi(argv[2]) : 1024;  // 每 block 的步数
  const int B = (argc > 3) ? atoi(argv[3]) : 8;        // 每 SM 并发 block 数

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int SM = prop.multiProcessorCount;
  printf("== device ==\n%s, sm_%d%d, SMs=%d, regs/SM=%d\n", prop.name, prop.major,
         prop.minor, SM, prop.regsPerMultiprocessor);

  CUDA_CHECK(cudaMalloc(&g_tab, PMAX * sizeof(fn_ptr)));

  const int G = SM * B;
  const long long W = (long long)G * upb * T;  // 元素数

  printf("blockDim T=%d, R=%d FMA/body, grid G=%d (%d/SM), steps/block=%d, "
         "total_elements=%lld, iters=%d\n\n",
         T, R, G, B, upb, W, iters);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, W * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  {
    float* h = (float*)malloc(W * sizeof(float));
    for (long long i = 0; i < W; ++i) h[i] = (float)((i % 997) + 1) * 1e-4f;
    CUDA_CHECK(cudaMemcpy(d_in, h, W * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
  }

  const double flops = (double)W * (double)R * 2.0;  // 每元素一次 body=R 次 FMA=2R flop

  // ---- ncu 单项剖析模式：argv[4]=P，argv[5]=sticky(可选) 时只 launch 一次并退出 ----
  if (argc > 4) {
    const int pv = atoi(argv[4]);
    const int st = (argc > 5) ? atoi(argv[5]) : 0;
    switch (pv) {
      case 1: profile_once<1>(G, W, upb, st, d_in, d_out); break;
      case 16: profile_once<16>(G, W, upb, st, d_in, d_out); break;
      case 64: profile_once<64>(G, W, upb, st, d_in, d_out); break;
      case 256: profile_once<256>(G, W, upb, st, d_in, d_out); break;
      default: printf("unsupported P=%d (use 1/16/64/256)\n", pv); break;
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(g_tab));
    return 0;
  }

  // ---- Section A：代码足迹扫描（固定总工作量，扫 P） ----
  printf("== Section A: 代码足迹扫描（固定总工作量，扫被执行 task body 数 P） ==\n");
  printf("%6s | %12s | %12s | %11s\n", "P", "ms", "GFLOP/s", "vs P=1");
  printf("-------+--------------+--------------+------------\n");
  float base_ms = -1.f;
#define AROW(Pv)                                                               \
  {                                                                            \
    float ms = timed_s<Pv>(iters, G, W, upb, d_in, d_out);                     \
    if (base_ms < 0) base_ms = ms;                                             \
    printf("%6d | %12.4f | %12.1f | %10.3fx\n", Pv, ms,                        \
           flops / (ms * 1e-3) / 1e9, ms / base_ms);                           \
  }
  AROW(1)
  AROW(2)
  AROW(4)
  AROW(8)
  AROW(16)
  AROW(32)
  AROW(64)
  AROW(128)
  AROW(256)
#undef AROW

  // ---- Section B：隔离「代码工作集」——轮转(spread) vs 固定(sticky)，同 P、同工作量 ----
  //   两种模式都经同一张 P 项函数指针表做间接调用（分支预测目标数相同）、总工作量相同；
  //   唯一差别是**每条执行流扫过的 body 数**：spread≈P，sticky=1。若 sticky 明显快，
  //   代价就来自「执行流的时间代码工作集」而非「分发表有多大」。
  printf("\n== Section B: 代码工作集隔离（同 P、同工作量；spread=每流扫 P 个 body, "
         "sticky=每流固定 1 个） ==\n");
  printf("%6s | %12s | %12s | %12s | %9s\n", "P", "spread ms", "sticky ms",
         "sticky GF/s", "spread/sticky");
  printf("-------+--------------+--------------+--------------+----------\n");
  float b1 = timed<1>(iters, G, W, upb, 0, d_in, d_out);
  float b1s = timed<1>(iters, G, W, upb, 1, d_in, d_out);
  printf("%6d | %12.4f | %12.4f | %12.1f | %8.3fx\n", 1, b1, b1s,
         flops / (b1s * 1e-3) / 1e9, b1 / b1s);
#define BROW(Pv)                                                               \
  {                                                                            \
    float sp = timed<Pv>(iters, G, W, upb, 0, d_in, d_out);                    \
    float st = timed<Pv>(iters, G, W, upb, 1, d_in, d_out);                    \
    printf("%6d | %12.4f | %12.4f | %12.1f | %8.3fx\n", Pv, sp, st,           \
           flops / (st * 1e-3) / 1e9, sp / st);                                \
  }
  BROW(16)
  BROW(64)
  BROW(256)
#undef BROW

  printf("\n>> 读法：\n");
  printf(">> A: 总 FMA 量恒定，唯一变量是「融合进这个 kernel 的 task 类型数 P」。P=1 是单个 task\n");
  printf(">>    类型（未融合）；P 增大即被执行的代码工作集增大。若耗时随 P 上升，即为**代码足迹/\n");
  printf(">>    L1 指令缓存**代价（融合的 task 类型越多，取指越容易 miss）。\n");
  printf(">> B: 同 P、同工作量，spread(每执行流扫 P 个 body) vs sticky(每流固定 1 个 body)：\n");
  printf(">>    两者分发表大小/间接调用数相同，差别只在**时间代码工作集**。spread/sticky 即该代价。\n");
  printf(">> C(ncu 另跑): 对 P=1/P=256 采 instruction-fetch / branch-resolving 停顿占比归因。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(g_tab));
  return 0;
}
