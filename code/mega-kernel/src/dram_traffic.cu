// dram_traffic.cu — MegaKernel 专题 P24-1 小实验：直接测量「损失三」消掉的 HBM 字节
//
// 目的（目标卡 H100 / sm90 实测）：
//   P5-2 只用「表观带宽 × 假设字节数」间接推断中间张量的 HBM 往返（损失三），
//   从未直接量过真实 DRAM 流量。本实验用 ncu 的 dram__bytes_read/write.sum
//   把「kernel-per-operator 每多一个算子就多一趟全张量 HBM 往返」测成硬数字：
//
//   构造 Q 个逐元素算子串成的依赖链  y_q = f(y_{q-1}),  y_0 = x：
//     (1) eagerQ  ：Q 个独立 kernel（kernel-per-operator）。每步都读入张量、写出张量，
//                   中间量全部落 HBM。DRAM 流量 ≈ Q × (读 N + 写 N)。
//     (2) graphQ  ：把上面 Q 个 kernel 捕获成一张 CUDA Graph、一次 replay。
//                   消掉 host 侧 Q 次提交，但设备侧 Q 次 grid 启动 + 中间量 HBM 往返不变。
//     (3) barrier ：单个 persistent kernel，逐算子 grid.sync()，中间量仍写/读 HBM scratch。
//                   消掉 Q 次 launch，但 **不消** 中间量的 HBM 往返。
//     (4) fused   ：单个 persistent kernel，逐元素把 Q 个算子一步算完，
//                   中间结果只活在寄存器里 → DRAM 流量恒为「读 x + 写 z」。
//
//   预期（关键标度律）：
//     eager/barrier 的 DRAM 字节 ≈ 2Q × N × 4 B；fused ≈ 2 × N × 4 B。
//     故 **fused 相对 eager 的 HBM 字节优势 ≈ Q×**（随算子数线性放大），
//     这就是「损失三随算子数标度」的直接字节证据（对照 P9-1 的时间标度）。
//
// 编译：nvcc -O3 -arch=sm_90 -rdc=true src/dram_traffic.cu -o src/dram_traffic
// 用法：
//   ./dram_traffic bench [iters]            # 计时表（Q=1,2,4,8）
//   ./dram_traffic one <eager|graph|barrier|fused> <Q>   # 单次运行，供 ncu 采样

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

// ---- 单个逐元素算子（便宜、带宽主导；float4 向量化） ----
__device__ __forceinline__ float dt_op(float v) { return fmaf(v, 1.5f, 0.25f); }
__device__ __forceinline__ float4 dt_op4(float4 v) {
  v.x = dt_op(v.x); v.y = dt_op(v.y); v.z = dt_op(v.z); v.w = dt_op(v.w);
  return v;
}

// 初始化 kernel（命名不以 dt_ 开头，便于 ncu 用 -k regex:dt_ 只抓目标 kernel）
__global__ void warm_fill(float4* a, long n4) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float base = (float)(i % 977) * 0.001f - 0.5f;
    a[i] = make_float4(base, base + 0.1f, base + 0.2f, base + 0.3f);
  }
}

// ---- (1)/(2) 一个「步」= 一个算子：out = f(in) ----
__global__ void dt_step(const float4* __restrict__ in, float4* __restrict__ out,
                        long n4) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) out[i] = dt_op4(in[i]);
}

// ---- (3) 单 persistent kernel + 逐算子 grid barrier，中间量仍走 HBM scratch ----
__global__ void dt_barrier(const float4* __restrict__ x, float4* __restrict__ s0,
                           float4* __restrict__ s1, float4* __restrict__ z,
                           int Q, long n4) {
  cg::grid_group g = cg::this_grid();
  long stride = (long)gridDim.x * blockDim.x;
  const float4* in = x;
  for (int q = 0; q < Q; ++q) {
    float4* out = (q == Q - 1) ? z : ((q % 2 == 0) ? s0 : s1);
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    for (; i < n4; i += stride) out[i] = dt_op4(in[i]);
    g.sync();           // phase 边界：下一步必须看到全 grid 写出的 out
    in = out;
  }
}

// ---- (4) 单 persistent kernel，Q 个算子融合进寄存器，不落 HBM ----
__global__ void dt_fused(const float4* __restrict__ x, float4* __restrict__ z,
                         int Q, long n4) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 v = x[i];
    for (int q = 0; q < Q; ++q) v = dt_op4(v);
    z[i] = v;
  }
}

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

// 用两个 scratch buffer 做 Q 步 ping-pong：buf0=s0(偶数步输出)、buf1=s1(奇数步)。
// 第 q 步读 (q==0? x : buf[(q-1)%2])、写 (q==Q-1? z : buf[q%2])。
static void host_eager_chain(const float4* x, float4* s0, float4* s1, float4* z,
                             int Q, long n4, int grid, int threads,
                             cudaStream_t st) {
  float4* buf[2] = {s0, s1};
  for (int q = 0; q < Q; ++q) {
    const float4* in = (q == 0) ? x : buf[(q - 1) % 2];
    float4* out = (q == Q - 1) ? z : buf[q % 2];
    dt_step<<<grid, threads, 0, st>>>(in, out, n4);
  }
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "bench";

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int nsm = prop.multiProcessorCount;
  const int threads = 256;

  if (strcmp(mode, "one") == 0) {
    // 单次运行：供 ncu 采样。用法 ./dram_traffic one <strategy> <Q>
    const char* strat = (argc > 2) ? argv[2] : "eager";
    int Q = (argc > 3) ? atoi(argv[3]) : 2;
    long M = 64L << 20;          // 256 MB/数组 ≫ 50 MB L2
    long n4 = M / 4;
    float *x, *s0, *s1, *z;
    CUDA_CHECK(cudaMalloc(&x, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s0, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s1, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&z, M * sizeof(float)));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* s04 = reinterpret_cast<float4*>(s0);
    float4* s14 = reinterpret_cast<float4*>(s1);
    float4* z4 = reinterpret_cast<float4*>(z);

    int maxblk = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk,
                                                             dt_barrier, threads, 0));
    int coop_grid = maxblk * nsm;
    int wide_grid = nsm * 16;

    warm_fill<<<wide_grid, threads>>>(const_cast<float4*>(x4), n4);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    if (strcmp(strat, "eager") == 0) {
      host_eager_chain(x4, s04, s14, z4, Q, n4, wide_grid, threads, s);
    } else if (strcmp(strat, "graph") == 0) {
      cudaGraph_t graph; cudaGraphExec_t gexec;
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      host_eager_chain(x4, s04, s14, z4, Q, n4, wide_grid, threads, s);
      CUDA_CHECK(cudaStreamEndCapture(s, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
      CUDA_CHECK(cudaGraphDestroy(graph));
      CUDA_CHECK(cudaGraphLaunch(gexec, s));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGraphExecDestroy(gexec));
    } else if (strcmp(strat, "barrier") == 0) {
      void* args[] = {(void*)&x4, (void*)&s04, (void*)&s14, (void*)&z4,
                      (void*)&Q, (void*)&n4};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)dt_barrier,
          dim3(coop_grid), dim3(threads), args, 0, s));
    } else if (strcmp(strat, "fused") == 0) {
      dt_fused<<<wide_grid, threads, 0, s>>>(x4, z4, Q, n4);
    } else {
      printf("unknown strategy %s\n", strat); return 2;
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(s0));
    CUDA_CHECK(cudaFree(s1)); CUDA_CHECK(cudaFree(z));
    return 0;
  }

  // ---- bench 模式 ----
  int iters = (argc > 2) ? atoi(argv[2]) : 30;

  int coop = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d, cooperativeLaunch=%d\n", prop.name,
         prop.major, prop.minor, nsm, coop);
  printf("iters=%d, threads/block=%d, N=64Mi elem (256 MB/array, >50MB L2)\n\n",
         iters, threads);

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  int maxblk = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, dt_barrier,
                                                          threads, 0));
  int coop_grid = maxblk * nsm;
  int wide_grid = nsm * 16;
  printf("grid: step/fused wide=%d, barrier coop=%d (maxBlocks/SM=%d)\n\n",
         wide_grid, coop_grid, maxblk);

  long M = 64L << 20, n4 = M / 4;
  float *x, *s0, *s1, *z;
  CUDA_CHECK(cudaMalloc(&x, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&s0, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&s1, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&z, M * sizeof(float)));
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* s04 = reinterpret_cast<float4*>(s0);
  float4* s14 = reinterpret_cast<float4*>(s1);
  float4* z4 = reinterpret_cast<float4*>(z);
  warm_fill<<<wide_grid, threads>>>(const_cast<float4*>(x4), n4);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("== Q-op chain fusion: y_q = f(y_{q-1}), f(v)=fma(v,1.5,0.25), float4 ==\n");
  printf("%4s | %9s | %9s | %9s | %9s | %7s | %7s | %7s\n", "Q", "eagerQ",
         "graphQ", "barrier", "fused", "e/g", "b/e", "f/b");
  printf("-----+-----------+-----------+-----------+-----------+---------+---------+--------\n");
  printf("%4s | %9s | %9s | %9s | %9s | %7s | %7s | %7s\n", "", "ms", "ms", "ms",
         "ms", "", "", "");
  printf("-----+-----------+-----------+-----------+-----------+---------+---------+--------\n");

  const int qs[] = {1, 2, 4, 8};
  for (int qi = 0; qi < 4; ++qi) {
    int Q = qs[qi];
    float te = bench(iters, [&](cudaStream_t st) {
      host_eager_chain(x4, s04, s14, z4, Q, n4, wide_grid, threads, st);
    }, s);

    cudaGraph_t graph; cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    host_eager_chain(x4, s04, s14, z4, Q, n4, wide_grid, threads, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    float tg = bench(iters, [&](cudaStream_t st) {
      CUDA_CHECK(cudaGraphLaunch(gexec, st));
    }, s);
    CUDA_CHECK(cudaGraphExecDestroy(gexec));

    float tb = bench(iters, [&](cudaStream_t st) {
      void* args[] = {(void*)&x4, (void*)&s04, (void*)&s14, (void*)&z4,
                      (void*)&Q, (void*)&n4};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)dt_barrier,
          dim3(coop_grid), dim3(threads), args, 0, st));
    }, s);

    float tf = bench(iters, [&](cudaStream_t st) {
      dt_fused<<<wide_grid, threads, 0, st>>>(x4, z4, Q, n4);
    }, s);

    printf("%4d | %9.4f | %9.4f | %9.4f | %9.4f | %6.2fx | %6.2fx | %6.2fx\n",
           Q, te, tg, tb, tf, te / tg, tb / te, tf / tb);
  }

  // ---- 正确性核对：fused z 应与 eagerQ z 一致 ----
  {
    long M2 = 1L << 20, n42 = M2 / 4;
    float *xa, *sa, *sb, *z1, *z2;
    CUDA_CHECK(cudaMalloc(&xa, M2 * 4)); CUDA_CHECK(cudaMalloc(&sa, M2 * 4));
    CUDA_CHECK(cudaMalloc(&sb, M2 * 4)); CUDA_CHECK(cudaMalloc(&z1, M2 * 4));
    CUDA_CHECK(cudaMalloc(&z2, M2 * 4));
    float* hx = (float*)malloc(M2 * 4);
    for (long i = 0; i < M2; ++i) hx[i] = (float)(i % 977) * 0.001f - 0.5f;
    CUDA_CHECK(cudaMemcpy(xa, hx, M2 * 4, cudaMemcpyHostToDevice));
    const float4* xa4 = reinterpret_cast<const float4*>(xa);
    for (int Q = 1; Q <= 8; Q *= 2) {
      host_eager_chain(xa4, reinterpret_cast<float4*>(sa),
                       reinterpret_cast<float4*>(sb),
                       reinterpret_cast<float4*>(z1), Q, n42, wide_grid, threads, s);
      dt_fused<<<wide_grid, threads, 0, s>>>(xa4, reinterpret_cast<float4*>(z2), Q, n42);
      CUDA_CHECK(cudaDeviceSynchronize());
      float* h1 = (float*)malloc(M2 * 4); float* h2 = (float*)malloc(M2 * 4);
      CUDA_CHECK(cudaMemcpy(h1, z1, M2 * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h2, z2, M2 * 4, cudaMemcpyDeviceToHost));
      double maxd = 0;
      for (long i = 0; i < M2; ++i) {
        double d = fabs((double)h1[i] - (double)h2[i]);
        if (d > maxd) maxd = d;
      }
      printf("correctness Q=%d: max|z_eager - z_fused| = %.3e (expect 0)\n", Q, maxd);
      free(h1); free(h2);
    }
    free(hx);
    CUDA_CHECK(cudaFree(xa)); CUDA_CHECK(cudaFree(sa)); CUDA_CHECK(cudaFree(sb));
    CUDA_CHECK(cudaFree(z1)); CUDA_CHECK(cudaFree(z2));
  }

  printf("\n>> 读法：\n");
  printf(">>  b/e : barrier 相对 eagerQ（单 launch，仍走 HBM 中间量）→ 损失二（launch）。\n");
  printf(">>  f/b : fused 相对 barrier（同样单 launch，中间量不落 HBM）→ 损失三（HBM 往返）。\n");
  printf(">>  Q 每加 1，eager/barrier 多一趟全张量 HBM 往返；fused 恒 8 B/elem。\n");
  printf(">>  故时间优势 f/b 应随 Q 增大而放大（HBM 字节比 ≈ Q）——真实字节见 ncu 采样。\n");

  CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(s0));
  CUDA_CHECK(cudaFree(s1)); CUDA_CHECK(cudaFree(z));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
