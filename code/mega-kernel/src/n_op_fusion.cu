// n_op_fusion.cu — MegaKernel 专题 P9-1 小实验：N 个依赖算子的链式融合
//
// 目的（目标卡 H100 / sm90 实测）：
//   把 P5-2 的「两算子融合」从 N=2 推广到任意 N，构造一条 N 个逐元素算子的依赖链
//       a0 = x
//       a_i = op_i(a_{i-1})   (i = 1..N)
//   比较四种策略，同时观察两类损失随 N / M 的变化：
//
//     (1) eagerN   ：N 个独立 kernel 逐个 launch（ping-pong 缓冲经 HBM 传递中间量）。
//                    N 次 host launch + N 次设备 grid 启动；中间量 8N B/元素。
//     (2) graphN   ：把 N 个 kernel 捕获成一张 CUDA Graph，一次 replay。
//                    消 host 侧 N 次提交，但设备侧 N 次 grid 启动与中间量 HBM 往返不变。
//     (3) persistN ：单个 persistent kernel（grid=#SM），kernel 内 N 个阶段 + 阶段间
//                    grid.sync()，中间量仍走 HBM。消掉 N-1 次 launch/grid 启动（损失二），
//                    但显存流量仍是 8N B/元素（损失三不变）。
//     (4) fusedN   ：单个 persistent kernel，每 block 把自己的 tile 连续过完 N 个算子，
//                    中间结果只活在本线程寄存器里，**不落 HBM**。
//                    显存流量恒为 8 B/元素（读 x 写 z），与 N 无关。
//
//   因为所有策略设备侧算术完全相同、显存流量模型清晰，可直接对照：
//     - 带宽主导（大 M）：eagerN/persistN 流量都是 8N B/元素，fusedN 是 8 B/元素，
//       故 fusedN 相对前两者应趋近 **N×**；速度比随 N 线性增长。
//     - launch 主导（小 M、大 N）：eagerN 额外 N 次 launch 显形，graphN 只消 host 提交，
//       persistN/fusedN 消掉设备侧 grid 启动，比值由 launch 开销而非流量决定。
//
// 与 P5-1 / P5-2 的关系：
//   - P5-1：N 个**独立**单元，单 persistent kernel 消 launch（Section A/B）。
//   - P5-2：**2 个依赖**算子，分开度量损失二 / 损失三。
//   - 本节：**N 个依赖**算子，看两类损失随 N 的标度（loss 随 N 增长），
//     并把「小 M + 大 N」这一 MPK 报告的场景（293 launch/token）落到实处。
//
// 编译：nvcc -O3 -arch=sm_90 -rdc=true n_op_fusion.cu -o n_op_fusion
// 用法：./n_op_fusion [iters]   （iters 默认 100）

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

// ---- 一组便宜、有界的逐元素算子（按 op 序号循环使用；保持 |值| O(1) 防溢出） ----
__device__ __forceinline__ float apply_op(int op, float v) {
  switch (op & 3) {
    case 0: return fmaf(v, 0.5f, 0.25f);
    case 1: return fabsf(v) * 0.5f + 0.1f;
    case 2: return sqrtf(fabsf(v));
    default: return fmaf(v, -0.25f, 0.5f);
  }
}
__device__ __forceinline__ float4 apply_v(int op, float4 v) {
  v.x = apply_op(op, v.x);
  v.y = apply_op(op, v.y);
  v.z = apply_op(op, v.z);
  v.w = apply_op(op, v.w);
  return v;
}

// ---- (1)(2) 单个阶段 kernel：一个算子，in -> out（中间量落 HBM） ----
__global__ void k_stage(const float4* __restrict__ in, float4* __restrict__ out,
                        long n4, int op) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) out[i] = apply_v(op, in[i]);
}

// ---- (3) 单 persistent kernel + 阶段间 grid barrier，中间量仍走 HBM ----
// in: x；buf[0]/buf[1]: ping-pong 中间缓冲；out: z。阶段 s 写：
//   s < N-1  -> buf[s & 1]
//   s == N-1 -> z
// （N==1 时直接 x->z。）
__global__ void k_persist(const float4* __restrict__ x, float4* __restrict__ a,
                          float4* __restrict__ b, float4* __restrict__ z,
                          long n4, int N) {
  cg::grid_group g = cg::this_grid();
  long stride = (long)gridDim.x * blockDim.x;
  const float4* in = x;
  for (int s = 0; s < N; ++s) {
    float4* out = (s == N - 1) ? z : ((s & 1) ? b : a);
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    for (; i < n4; i += stride) out[i] = apply_v(s, in[i]);
    if (s != N - 1) g.sync();  // 下一阶段必须看到本轮全部输出
    in = out;
  }
}

// ---- (4) 单 persistent kernel，N 个算子全在寄存器里连过，不落 HBM ----
__global__ void k_fused(const float4* __restrict__ x, float4* __restrict__ z,
                        long n4, int N) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 v = x[i];
    for (int s = 0; s < N; ++s) v = apply_v(s, v);
    z[i] = v;
  }
}

// ---- 计时 ----
static float bench(int iters, const std::function<void(cudaStream_t)>& f,
                   cudaStream_t s) {
  for (int w = 0; w < 3; ++w) f(s);
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

struct Row {
  long M;
  int N;
  float te, tg, tp, tf;
};

// 跑一组 (M, N) 的四种策略，返回结果
static Row run_case(long M, int N, int iters, int wide_grid, int coop_grid,
                    int threads) {
  long n4 = M / 4;
  float *x, *a, *b, *z;
  CUDA_CHECK(cudaMalloc(&x, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&a, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&z, M * sizeof(float)));
  CUDA_CHECK(cudaMemset(x, 0, M * sizeof(float)));

  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* a4 = reinterpret_cast<float4*>(a);
  float4* b4 = reinterpret_cast<float4*>(b);
  float4* z4 = reinterpret_cast<float4*>(z);

  auto launch_chain = [&](cudaStream_t st) {
    const float4* in = x4;
    for (int s = 0; s < N; ++s) {
      float4* out = (s == N - 1) ? z4 : ((s & 1) ? b4 : a4);
      k_stage<<<wide_grid, threads, 0, st>>>(in, out, n4, s);
      in = out;
    }
  };

  // (1) eagerN
  float te = bench(iters, launch_chain, 0);

  // (2) graphN
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreate(&cs));
  cudaGraph_t graph;
  cudaGraphExec_t gexec;
  CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
  launch_chain(cs);
  CUDA_CHECK(cudaStreamEndCapture(cs, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  float tg = bench(iters, [&](cudaStream_t st) {
    CUDA_CHECK(cudaGraphLaunch(gexec, st));
  }, 0);
  CUDA_CHECK(cudaGraphExecDestroy(gexec));
  CUDA_CHECK(cudaStreamDestroy(cs));

  // (3) persistN
  float tp = bench(iters, [&](cudaStream_t st) {
    void* args[] = {(void*)&x4, (void*)&a4, (void*)&b4, (void*)&z4,
                    (void*)&n4, (void*)&N};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_persist, dim3(coop_grid),
                                           dim3(threads), args, 0, st));
  }, 0);

  // (4) fusedN
  float tf = bench(iters, [&](cudaStream_t st) {
    k_fused<<<wide_grid, threads, 0, st>>>(x4, z4, n4, N);
  }, 0);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(z));
  return Row{M, N, te, tg, tp, tf};
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 100;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int nsm = prop.multiProcessorCount;
  int coop = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));

  const int threads = 256;
  printf("== device ==\n%s, sm_%d%d, SMs=%d, cooperativeLaunch=%d\n", prop.name,
         prop.major, prop.minor, nsm, coop);
  printf("iters=%d, threads/block=%d\n\n", iters, threads);

  int maxblk = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, k_persist,
                                                           threads, 0));
  int coop_grid = maxblk * nsm;
  printf("cooperative: maxBlocks/SM=%d -> grid=%d blocks x %d threads\n",
         maxblk, coop_grid, threads);
  const int wide_grid = nsm * 16;

  // 流量模型：经 HBM 中间量 = 8N B/元素；fused = 8 B/元素
  auto print_hdr = [](const char* t) {
    printf("\n== %s ==\n", t);
    printf("%9s | %5s | %9s | %9s | %9s | %9s | %6s | %8s | %7s\n", "M(elem)", "N",
           "eager ms", "graph ms", "persist ms", "fused ms", "e/g", "e/persist",
           "e/fused");
    printf("--------- | ----- | --------- | --------- | --------- | --------- | ------ | -------- | -------\n");
  };
  auto print_row = [](const Row& r) {
    printf("%9ld | %5d | %9.4f | %9.4f | %9.4f | %9.4f | %5.2fx | %7.2fx | %6.2fx\n",
           r.M, r.N, r.te, r.tg, r.tp, r.tf, r.te / r.tg, r.te / r.tp,
           r.te / r.tf);
  };

  // ---- Section A：固定 N=8，扫 M（launch 主导 -> 带宽主导 的过渡） ----
  print_hdr("Section A: N=8 fixed, sweep M");
  {
    const long ms[] = {16L << 10, 64L << 10, 256L << 10, 1L << 20, 4L << 20,
                       16L << 20};
    for (int i = 0; i < 6; ++i) print_row(run_case(ms[i], 8, iters, wide_grid,
                                                   coop_grid, threads));
  }

  // ---- Section B：固定大 M=16Mi，扫 N（带宽主导：fused 相对 eager 应 ~N×） ----
  print_hdr("Section B: M=16Mi fixed (bandwidth-bound), sweep N");
  {
    const int ns[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 6; ++i) print_row(run_case(16L << 20, ns[i], iters,
                                                   wide_grid, coop_grid,
                                                   threads));
  }

  // ---- Section C：固定小 M=16Ki，扫 N（launch 主导：多算子 + 轻负载） ----
  print_hdr("Section C: M=16Ki fixed (launch-bound), sweep N");
  {
    const int ns[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 6; ++i) print_row(run_case(16L << 10, ns[i], iters,
                                                   wide_grid, coop_grid,
                                                   threads));
  }

  // ---- 正确性：fusedN == eagerN（同样算子序列，寄存器 vs 经 HBM） ----
  {
    const long M = 1L << 20;
    const int N = 16;
    long n4 = M / 4;
    float *x, *a, *b, *z1, *z2;
    CUDA_CHECK(cudaMalloc(&x, M * 4));
    CUDA_CHECK(cudaMalloc(&a, M * 4));
    CUDA_CHECK(cudaMalloc(&b, M * 4));
    CUDA_CHECK(cudaMalloc(&z1, M * 4));
    CUDA_CHECK(cudaMalloc(&z2, M * 4));
    float* hx = (float*)malloc(M * 4);
    for (long i = 0; i < M; ++i) hx[i] = (float)(i % 977) * 0.001f - 0.5f;
    CUDA_CHECK(cudaMemcpy(x, hx, M * 4, cudaMemcpyHostToDevice));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* in = x4;
    for (int s = 0; s < N; ++s) {
      float4* out = (s == N - 1) ? reinterpret_cast<float4*>(z1)
                                 : ((s & 1) ? reinterpret_cast<float4*>(b)
                                            : reinterpret_cast<float4*>(a));
      k_stage<<<wide_grid, threads>>>(in, out, n4, s);
      in = out;
    }
    k_fused<<<wide_grid, threads>>>(x4, reinterpret_cast<float4*>(z2), n4, N);
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
    printf("\ncorrectness: max|z_eagerN - z_fusedN| = %.3e over %ld elems "
           "(N=%d, expect ~0)\n",
           maxd, M, N);
    free(hx); free(h1); free(h2);
    CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(z1)); CUDA_CHECK(cudaFree(z2));
  }

  printf("\n>> 读法：\n");
  printf(">>  流量模型：eagerN/graphN/persistN = 8N B/元素（中间量落 HBM）；fusedN = 8 B/元素。\n");
  printf(">>  e/persist 度量「消掉 N-1 次 launch/grid 启动」（损失二，设备侧）。\n");
  printf(">>  e/fused 度量损失二 + 「中间量不落 HBM」（损失三）之和。\n");
  printf(">>  Section B 带宽主导：e/fused 应随 N 线性增长（趋近 N×）。\n");
  printf(">>  Section C launch 主导：小 M + 大 N 下 eagerN 的 N 次 launch 显形。\n");

  return 0;
}
