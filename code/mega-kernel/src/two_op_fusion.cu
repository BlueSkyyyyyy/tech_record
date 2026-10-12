// two_op_fusion.cu — MegaKernel 专题 P5-2 小实验：两算子融合成单 persistent kernel
//
// 目的（目标卡 H100 / sm90 实测）：
//   构造一对有数据依赖的逐元素算子
//       op1: y = f(x)          （producer）
//       op2: z = g(y)          （consumer，全图依赖 op1）
//   比较四种把它们跑到 GPU 上的策略，把 megakernel 的两类收益**分开量化**：
//
//     (1) eager2   ：k_op1 写 y 到 HBM，再 launch k_op2 读 y 写 z。
//                    2 次 host launch + 2 次设备 grid 启动；中间张量 y 落 HBM。
//     (2) graph2   ：把上面两个 kernel 捕获成一张 CUDA Graph，一次 replay。
//                    消掉 host 侧 2 次提交，但**不消**设备侧 2 次 grid 启动，
//                    也**不消**中间张量 y 的 HBM 往返，且两 kernel 间仍是完整 barrier。
//     (3) barrier  ：单个 persistent kernel（grid=#SM），kernel 内做两阶段——
//                    phase1 全员写 y 到 HBM，grid.sync()，phase2 全员读 y 写 z。
//                    消掉第 2 次 launch/grid 启动（损失二），但**仍保留** y 的 HBM 往返。
//                    → 与 (1) 之差 = launch 收益；与 (2) 对照 = 设备侧 grid 启动的差别。
//     (4) fused    ：单个 persistent kernel，逐元素把 op2(op1(x)) 一步算完，
//                    中间结果只活在寄存器里，**不落 HBM**。
//                    → 与 (3) 之差 = 中间张量 HBM 往返的收益（损失三）。
//
//   三种「1 次 launch」策略的设备侧计算量完全等价（同样的 op1/op2/元素数），
//   所以 (1)vs(3)、(3)vs(4) 的差就是各层收益的干净度量。
//
//   预期：本实验选便宜算子 + 大数组，使 kernel 处于 **HBM 带宽主导** 区间，
//   此时 fused 的显存流量是 8 B/元素（读 x 写 z），而三种经 HBM 中间量的策略是
//   16 B/元素（读 x 写 y 读 y 写 z），故 fused 应接近 **2×** 于 bandwidth-bound 基线。
//
// 编译：nvcc -O3 -arch=sm_90 -rdc=true two_op_fusion.cu -o two_op_fusion
// 用法：./two_op_fusion [iters]   （iters 默认 50）

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
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

// ---- 两个逐元素算子（便宜、带宽主导；float4 向量化） ----
__device__ __forceinline__ float op1f(float v) { return fmaf(v, 1.5f, 0.25f); }
__device__ __forceinline__ float op2f(float v) { return sqrtf(fabsf(v)) + 0.5f * v; }

__device__ __forceinline__ float4 op1v(float4 v) {
  v.x = op1f(v.x); v.y = op1f(v.y); v.z = op1f(v.z); v.w = op1f(v.w);
  return v;
}
__device__ __forceinline__ float4 op2v(float4 v) {
  v.x = op2f(v.x); v.y = op2f(v.y); v.z = op2f(v.z); v.w = op2f(v.w);
  return v;
}

// ---- (1) 两个独立 kernel：op1 写 y 到 HBM ----
__global__ void k_op1(const float4* __restrict__ x, float4* __restrict__ y,
                      long n4) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) y[i] = op1v(x[i]);
}

// ---- (1) 第二个独立 kernel：op2 读 y 写 z ----
__global__ void k_op2(const float4* __restrict__ y, float4* __restrict__ z,
                      long n4) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) z[i] = op2v(y[i]);
}

// ---- (3) 单 persistent kernel + grid barrier，中间量仍走 HBM ----
__global__ void k_barrier(const float4* __restrict__ x, float4* __restrict__ y,
                          float4* __restrict__ z, long n4) {
  cg::grid_group g = cg::this_grid();
  long stride = (long)gridDim.x * blockDim.x;
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  for (; i < n4; i += stride) y[i] = op1v(x[i]);
  g.sync();  // 全 grid 屏障：phase2 必须看到所有 phase1 的 y
  i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  for (; i < n4; i += stride) z[i] = op2v(y[i]);
}

// ---- (4) 单 persistent kernel，两算子融合在寄存器里，不落 HBM ----
__global__ void k_fused(const float4* __restrict__ x, float4* __restrict__ z,
                        long n4) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) z[i] = op2v(op1v(x[i]));
}

// ---- 计时工具：对同一 stream 上的一次「策略」重复 iters 次取均值 ----
static float bench(int iters, const std::function<void(cudaStream_t)>& f,
                   cudaStream_t s) {
  for (int w = 0; w < 3; ++w) f(s);  // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < iters; ++r) f(s);
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 50;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int nsm = prop.multiProcessorCount;

  // 确认 device-side cooperative launch 可用（k_barrier 需要）
  int coop = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));

  const int threads = 256;
  printf("== device ==\n%s, sm_%d%d, SMs=%d, cooperativeLaunch=%d\n", prop.name,
         prop.major, prop.minor, nsm, coop);
  printf("iters=%d, threads/block=%d\n\n", iters, threads);

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // k_barrier 的最大常驻 block 数（cooperative launch 上限）
  int maxblk = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, k_barrier,
                                                          threads, 0));
  int coop_grid = maxblk * nsm;
  printf("cooperative: maxBlocks/SM=%d -> grid=%d blocks x %d threads\n\n",
         maxblk, coop_grid, threads);

  // 普通 kernel（op1/op2/fused）用标准 grid-stride，grid 取较宽以占满 SM
  const int wide_grid = nsm * 16;

  printf("== two-op fusion: op1 y=f(x), op2 z=g(y), float4 ==\n");
  printf("%10s | %9s | %9s | %9s | %9s | %7s | %7s | %7s\n", "M (elem)",
         "eager2", "graph2", "barrier", "fused", "e/g", "b/e", "f/b");
  printf("----------- | --------- | --------- | --------- | --------- | ------- | ------- | -------\n");
  printf("%10s | %9s | %9s | %9s | %9s | %7s | %7s | %7s\n", "", "ms", "ms", "ms",
         "ms", "", "", "");
  printf("----------- | --------- | --------- | --------- | --------- | ------- | ------- | -------\n");

  const long melem[] = {16L << 20, 64L << 20, 256L << 20};
  const int nm = 3;

  long  rep_M = 0;
  double gbs_eager = 0, gbs_graph = 0, gbs_bar = 0, gbs_fused = 0;

  for (int mi = 0; mi < nm; ++mi) {
    long M = melem[mi];
    long n4 = M / 4;
    float *x, *y, *z;
    CUDA_CHECK(cudaMalloc(&x, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&z, M * sizeof(float)));
    // 初始化 x（用 memset 保证确定，op 对值不敏感）
    CUDA_CHECK(cudaMemset(x, 0, M * sizeof(float)));

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    float4* z4 = reinterpret_cast<float4*>(z);

    // (1) eager2：两个 kernel 串行（同 stream 天然依赖）
    float te = bench(iters, [&](cudaStream_t st) {
      k_op1<<<wide_grid, threads, 0, st>>>(x4, y4, n4);
      k_op2<<<wide_grid, threads, 0, st>>>(y4, z4, n4);
    }, s);

    // (2) graph2：捕获同一对 kernel，一次 replay
    cudaGraph_t graph;
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    k_op1<<<wide_grid, threads, 0, s>>>(x4, y4, n4);
    k_op2<<<wide_grid, threads, 0, s>>>(y4, z4, n4);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    float tg = bench(iters, [&](cudaStream_t st) {
      CUDA_CHECK(cudaGraphLaunch(gexec, st));
    }, s);
    CUDA_CHECK(cudaGraphExecDestroy(gexec));

    // (3) barrier：单 cooperative persistent kernel，中间量走 HBM
    float tb = bench(iters, [&](cudaStream_t st) {
      void* args[] = {(void*)&x4, (void*)&y4, (void*)&z4, (void*)&n4};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_barrier,
          dim3(coop_grid), dim3(threads), args, 0, st));
    }, s);

    // (4) fused：单 persistent kernel，中间量不落 HBM
    float tf = bench(iters, [&](cudaStream_t st) {
      k_fused<<<wide_grid, threads, 0, st>>>(x4, z4, n4);
    }, s);

    // 表观显存流量（GB）：经 HBM 中间量 = 16 B/elem，fused = 8 B/elem
    double gb_hbm = (double)M * 16.0 / 1e9;
    double gb_fus = (double)M * 8.0 / 1e9;
    printf("%10ld | %9.4f | %9.4f | %9.4f | %9.4f | %6.2fx | %6.2fx | %6.2fx\n",
           M, te, tg, tb, tf, te / tg, tb / te, tf / tb);

    if (mi == nm - 1) {  // 记录最大规模一行供带宽换算
      rep_M = M;
      gbs_eager = gb_hbm / (te / 1e3) / 1.0;  // GB / s
      gbs_graph = gb_hbm / (tg / 1e3);
      gbs_bar   = gb_hbm / (tb / 1e3);
      gbs_fused = gb_fus / (tf / 1e3);
    }

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z));
  }

  printf("\n== apparent bandwidth @ M=%ld elems (GB/s, 越大越接近 HBM 峰值) ==\n",
         rep_M);
  printf("eager2=%.0f  graph2=%.0f  barrier=%.0f (16 B/elem) | fused=%.0f (8 B/elem)\n",
         gbs_eager, gbs_graph, gbs_bar, gbs_fused);

  // ---- 正确性核对：fused z 应与 eager2 z 一致 ----
  {
    long M = 1L << 20, n4 = M / 4;
    float *x, *y, *z1, *z2;
    CUDA_CHECK(cudaMalloc(&x, M * 4));
    CUDA_CHECK(cudaMalloc(&y, M * 4));
    CUDA_CHECK(cudaMalloc(&z1, M * 4));
    CUDA_CHECK(cudaMalloc(&z2, M * 4));
    float* hx = (float*)malloc(M * 4);
    for (long i = 0; i < M; ++i) hx[i] = (float)(i % 977) * 0.001f - 0.5f;
    CUDA_CHECK(cudaMemcpy(x, hx, M * 4, cudaMemcpyHostToDevice));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    k_op1<<<wide_grid, threads>>>(x4, reinterpret_cast<float4*>(y), n4);
    k_op2<<<wide_grid, threads>>>(reinterpret_cast<float4*>(y),
                                  reinterpret_cast<float4*>(z1), n4);
    k_fused<<<wide_grid, threads>>>(x4, reinterpret_cast<float4*>(z2), n4);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* h1 = (float*)malloc(M * 4);
    float* h2 = (float*)malloc(M * 4);
    CUDA_CHECK(cudaMemcpy(h1, z1, M * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2, z2, M * 4, cudaMemcpyDeviceToHost));
    double maxd = 0;
    for (long i = 0; i < M; ++i) {
      double d = fabs((double)h1[i] - (double)h2[i]);
      if (d > maxd) maxd = d;
    }
    printf("\ncorrectness: max|z_two - z_fused| = %.3e over %ld elems (expect ~0)\n",
           maxd, M);
    free(hx); free(h1); free(h2);
    CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z1)); CUDA_CHECK(cudaFree(z2));
  }

  printf("\n>> 读法：\n");
  printf(">>  e/g  : graph 相对 eager 的收益（只消 host 侧提交，设备侧仍 2 次 grid 启动）。\n");
  printf(">>  b/e  : barrier 相对 eager（单 launch + kernel 内 grid barrier，仍走 HBM 中间量）\n");
  printf(">>          → 这一档度量「消掉第 2 次 launch/grid 启动」的收益（损失二）。\n");
  printf(">>  f/b  : fused 相对 barrier（同样单 launch，但中间量不落 HBM）\n");
  printf(">>          → 这一档度量「消掉中间张量 HBM 往返」的收益（损失三）。\n");
  printf(">>  本实验带宽主导，故 f/b 应接近 2×（16 B/elem vs 8 B/elem）。\n");

  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
