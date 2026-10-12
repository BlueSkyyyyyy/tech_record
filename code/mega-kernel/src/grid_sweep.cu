// grid_sweep.cu — MegaKernel 专题 P10-1：persistent kernel 网格大小扫描 + 多 block 协调成本
//
// 目的（目标卡 H100 / sm90 实测）：
//   P5-1（docs/04 §1）用固定 grid=#SM 的 persistent kernel 度量了 launch 开销，留下
//   「persistent 的 grid=#SM=132 固定，未扫网格大小」（§1.6）；P5-2/P9-1 又反复用
//   cooperative 全 grid 屏障做设备内协调。本节把这两点补齐，回答 megakernel 的两类
//   资源配置问题：
//
//   Section A（吞吐随网格的标度）：固定总工作量（N 个独立单元，每单元 K 次自旋），
//     扫 persistent kernel 的 grid 大小 G。看：grid 太小→并行度不足；增大到 ~#SM 后
//     是否饱和；G > #SM（每 SM 多个常驻 block）是否还能再涨。给出 speedup(G)/G=1。
//
//   Section B（过配/摊薄）：固定较小的 N（单元数），扫跨过 G=N 的网格大小。看 G>N 后
//     大量 block 空转，时间不再下降——即「grid 过大则每 block 单元太少、调度摊薄」。
//
//   Section C（多 block 协调成本）：用 cooperative groups 的 grid.sync() 做 B 次全 grid
//     屏障，扫网格大小 G。度量单次屏障耗时 ns/barrier 如何随 G 增长。生产 megakernel
//     （DeepGEMM/MPK）之所以用**去中心化 event queue / 异步原子 ring barrier** 而非全
//     grid 硬屏障，正是为了不让协调成本随 CTA 数线性膨胀——本节给出这条成本曲线。
//
// 编译：nvcc -O3 -arch=sm_90 -rdc=true grid_sweep.cu -o grid_sweep
// 用法：./grid_sweep [iters]   （iters 默认 200）

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// 一个「工作单元」的设备侧工作：LCG 自旋 K 次 + 每 block 一次 atomicAdd。
__device__ __forceinline__ void do_unit(int* sink, int K) {
  int acc = threadIdx.x * 2654435761u;
  for (int i = 0; i < K; ++i) acc = acc * 1664525 + 1013904223;
  if (acc == -123456789) sink[1] = acc;  // 阻止编译器删掉自旋
  if (threadIdx.x == 0) atomicAdd(sink, 1);
}

// persistent worker：一次 launch，grid-stride 消费 N 个独立单元。
__global__ void persist_units(int* sink, long N, int K) {
  for (long u = blockIdx.x; u < N; u += gridDim.x) do_unit(sink, K);
}

// cooperative persistent kernel：做 B 次全 grid 屏障。
__global__ void barrier_kernel(int* sink, int B) {
  cg::grid_group g = cg::this_grid();
  int acc = threadIdx.x;
  for (int i = 0; i < B; ++i) {
    acc += i;
    g.sync();
  }
  if (acc == -123456789) sink[1] = acc;
  if (threadIdx.x == 0) atomicAdd(sink, 1);
}

static float bench_units(long N, int K, int grid, int threads, int iters,
                         int* sink, cudaStream_t s) {
  for (int w = 0; w < 3; ++w) persist_units<<<grid, threads, 0, s>>>(sink, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < iters; ++r)
    persist_units<<<grid, threads, 0, s>>>(sink, N, K);
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

static float bench_barrier(int B, int grid, int threads, int iters, int* sink,
                           cudaStream_t s) {
  void* args[] = {(void*)&sink, (void*)&B};
  for (int w = 0; w < 3; ++w)
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(grid),
                                           dim3(threads), args, 0, s));
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < iters; ++r)
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(grid),
                                           dim3(threads), args, 0, s));
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 200;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int nsm = prop.multiProcessorCount;
  int coop = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));

  const int threads = 256;
  printf("== device ==\n%s, sm_%d%d, SMs=%d, cooperativeLaunch=%d\n", prop.name,
         prop.major, prop.minor, nsm, coop);
  printf("iters=%d, threads/block=%d\n\n", iters, threads);

  int* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int) * 2));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(int) * 2));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // cooperative 上限（barrier_kernel 固定角色，用占用率 API 查）
  int maxblk = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk,
                                                           barrier_kernel, threads, 0));
  int coop_grid = maxblk * nsm;
  printf("cooperative barrier kernel: maxBlocks/SM=%d -> grid<=%d blocks x %d threads\n\n",
         maxblk, coop_grid, threads);

  // 网格大小扫描点（覆盖 <#SM、=#SM、>#SM）
  const int grids[] = {1, 2, 4, 8, 16, 32, 64, 128, 132, 192, 256, 384, 528, 768, 1056};
  const int ng = (int)(sizeof(grids) / sizeof(grids[0]));

  // -------------------------------------------------------------------------
  // Section A：固定总工作量，扫 grid —— 并行度标度
  //   N=8192 个独立单元，每单元 K=2048 次自旋；总设备侧工作与 grid 无关。
  // -------------------------------------------------------------------------
  printf("== Section A: fixed work, sweep grid (N=8192 units, K=2048) ==\n");
  printf("%6s | %11s | %10s | %11s | %9s | %12s\n", "grid", "time ms", "speedup",
         "units/blk", "ns/unit", "ns/unit/blk");
  printf("-------+-------------+------------+-------------+-----------+-------------\n");
  const long NA = 8192;
  const int KA = 2048;
  float tA1 = 0;
  for (int gi = 0; gi < ng; ++gi) {
    int G = grids[gi];
    float t = bench_units(NA, KA, G, threads, iters, sink, s);
    if (gi == 0) tA1 = t;
    long units_per_blk = NA / G;  // G<=N 于本区间
    double ns_per_unit = t * 1e6 / (double)NA;             // 墙钟每单元
    double ns_per_unit_blk = t * 1e6 / (double)units_per_blk;  // 每 block 每单元
    printf("%6d | %11.4f | %9.2fx | %11.1f | %9.1f | %12.1f\n", G, t, tA1 / t,
           (double)NA / G, ns_per_unit, ns_per_unit_blk);
  }

  // -------------------------------------------------------------------------
  // Section B：小 N，扫 grid 跨过 N —— 过配/摊薄
  //   N=64 个单元；grid 从 1 扫到 528（远超 N）。G>N 后大量 block 空转。
  // -------------------------------------------------------------------------
  printf("\n== Section B: small N, sweep grid across N (N=64 units, K=2048) ==\n");
  printf("%6s | %11s | %10s | %9s | %11s\n", "grid", "time ms", "speedup",
         "units/blk", "active blks");
  printf("-------+-------------+------------+-----------+------------\n");
  const long NB = 64;
  const int KB = 2048;
  float tB1 = 0;
  for (int gi = 0; gi < ng; ++gi) {
    int G = grids[gi];
    float t = bench_units(NB, KB, G, threads, iters, sink, s);
    if (gi == 0) tB1 = t;
    int active = (G < NB) ? G : (int)NB;
    printf("%6d | %11.4f | %9.2fx | %9.1f | %11d\n", G, t, tB1 / t,
           (double)NB / G, active);
  }

  // -------------------------------------------------------------------------
  // Section C：cooperative 全 grid 屏障成本随网格的标度
  //   每轮做 B=200 次 grid.sync()，报 ns/barrier。
  // -------------------------------------------------------------------------
  printf("\n== Section C: cooperative grid.sync() cost vs grid (B=200 barriers) ==\n");
  printf("%6s | %11s | %12s | %12s\n", "grid", "time ms", "ns/barrier",
         "ns/barrier/blk");
  printf("-------+-------------+--------------+--------------\n");
  const int B = 200;
  for (int gi = 0; gi < ng; ++gi) {
    int G = grids[gi];
    if (G > coop_grid) continue;
    float t = bench_barrier(B, G, threads, iters, sink, s);
    double ns_per = t * 1e6 / B;
    printf("%6d | %11.4f | %12.1f | %12.4f\n", G, t, ns_per, ns_per / G);
  }

  // 正确性/工作量核对：persist_units 应把 sink[0] 增加 N 次（不管 grid 大小）。
  {
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(int) * 2));
    long NT = 4096;
    persist_units<<<1, threads, 0, s>>>(sink, NT, 8);
    persist_units<<<132, threads, 0, s>>>(sink, NT, 8);
    persist_units<<<528, threads, 0, s>>>(sink, NT, 8);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h = 0;
    CUDA_CHECK(cudaMemcpy(&h, sink, sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nwork check (grid=1/132/528, each N=%ld): sink[0]=%d (expect %ld)\n",
           NT, h, 3 * NT);
  }

  printf("\n>> 读法：Section A 的 speedup 随 G 上升到 ~#SM 附近趋饱和，G>#SM 再增\n");
  printf(">> 收益有限（每 SM 多常驻 block 只能吃占用率余量）；Section B 里 G>N\n");
  printf(">> 后大量 block 空转、时间不再降（过配）。Section C 的 ns/barrier\n");
  printf(">> 随 grid 增长——全 grid 硬屏障的成本随 CTA 数膨胀，这正是生产\n");
  printf(">> megakernel 用去中心化 event queue / 异步原子 ring barrier 的动机。\n");
  return 0;
}
