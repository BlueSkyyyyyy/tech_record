// in_kernel_pipeline.cu — MegaKernel 专题 P13-1 小实验：
//   **单 kernel 内跨 task 软件流水**：把「损失一（kernel barrier 挡住软件流水）」实测出来
//
// 目的（目标卡 H100 / sm90 实测）：
//   P5-2（docs/04 §2）把两个算子融进单 persistent kernel，但两段之间用的是硬同步，**没有做
//   跨 tile 软件流水**（§2.6/§5.6 明确列为未覆盖项）。而 MPK §5.3（快照行 1326–1363）的核心机制
//   正是「worker 做完一个 task 后不等全局 barrier 就开始下一个 task 的访存阶段」，§6.6 报告
//   cross-task software pipelining 带来 1.2–1.3×；线上实现的 `Kstages` 双重流水见 P11
//   （`linear_hopper.cuh`，快照 Kstages=3/6）。本实验在**同一个 kernel** 内量化这条机制。
//
//   构造一条「访存阶段 -> 计算阶段」依赖链（计算依赖刚载入的数据），每个 block 常驻处理 chunks 个
//   chunk，每个 chunk = 先 cp.async 载入 T*E 个 float 到共享内存、再用 R 轮 FMA 消费。用 K 级软件
//   流水控制「提前发射几段拷贝」：
//     K=1 ：issue copy s -> wait -> compute s（严格阶段串行，即「kernel barrier 后的两个 kernel」）
//     K>1 ：预取 K-1 段，边算 chunk s 边让 chunk s+1..s+K-1 的拷贝在飞（深度 K-1 软件流水）
//   两模式指令路径完全相同（同一 cp.async 路径），唯一差别是**流水深度**，所以 K=1→K 的比值
//   就是**软件流水能拿回的损失一**。扫描 K 也直接对应 MPK/DeepGEMM 为何选多层 `Kstages`。
//
//   同时给纯访存（R=0）与纯计算（comp_only）参照，用来谈「访存/计算重叠」的理论上界。
//
// 结构：
//   每个 block 常驻，负责独立输入区 region = blockIdx * chunks * T * E 个 float；
//   输入总大小 = grid * chunks * T * E * 4 B，远大于 L2（50MB），故访存基本打 HBM。
//
// 编译：nvcc -O3 -arch=sm_90 in_kernel_pipeline.cu -o in_kernel_pipeline
// 用法：./in_kernel_pipeline [iters] [T] [G]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#ifndef KMAX
#define KMAX 8
#endif

// 每线程载入 E 个 float（E 是 4 的倍数，用 E/4 条 16B cp.async）
template <int E>
__device__ __forceinline__ void async_load(float* dst, const float* src) {
#pragma unroll
  for (int q = 0; q < E / 4; ++q) {
    __pipeline_memcpy_async(dst + 4 * q, src + 4 * q, 16);
  }
}

// K 级软件流水：K=1 即严格串行；K 最大 KMAX。每 block 常驻，chunks 个 chunk。
template <int E>
__global__ void pipe_kernel(const float* __restrict__ in, float* __restrict__ out,
                            int chunks, int R, int K) {
  extern __shared__ float smem[];
  const int T = blockDim.x;
  const int tid = threadIdx.x;
  const long long region = (long long)blockIdx.x * chunks * T * E;

  float acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.f;

  const int inflight = K - 1;  // 预取段数
  int issued = 0;

  // 序言：先发射前 K-1 段拷贝
  for (; issued < inflight && issued < chunks; ++issued) {
    async_load<E>(smem + (size_t)(issued % K) * T * E + tid * E,
                  in + region + (long long)issued * T * E + tid * E);
    __pipeline_commit();
  }
  for (int s = 0; s < chunks; ++s) {
    int nxt = s + inflight;
    if (nxt < chunks) {
      async_load<E>(smem + (size_t)(nxt % K) * T * E + tid * E,
                    in + region + (long long)nxt * T * E + tid * E);
      __pipeline_commit();
      ++issued;
    }
    // 等 chunk s 就绪：只允许更新的 (issued-s-1) 段仍在飞
    int prior = issued - s - 1;
    __pipeline_wait_prior(prior);
    __syncthreads();
    float* cur = smem + (size_t)(s % K) * T * E;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      float x = cur[tid * E + e];
#pragma unroll 4
      for (int r = 0; r < R; ++r) x = fmaf(x, 1.0000001f, 1e-7f);
      acc[e] += x;
    }
    __syncthreads();
  }

  float s = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) s += acc[e];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((tid & 31) == 0) atomicAdd(out, s);
}

// 纯计算参照：每个线程做 chunks*E*R 次 FMA（E 条独立链），不读全局。
template <int E>
__global__ void comp_only(float* __restrict__ out, int chunks, int R) {
  float acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = threadIdx.x * 1e-6f + e * 1e-3f + 1.0f;
  for (int s = 0; s < chunks; ++s) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      float x = acc[e];
#pragma unroll 4
      for (int r = 0; r < R; ++r) x = fmaf(x, 1.0000001f, 1e-7f);
      acc[e] = x;
    }
  }
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) s += acc[e];
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}

template <int E>
static float time_pipe(int iters, int G, int T, int chunks, int R, int K,
                       float* d_in, float* d_out, float* h_out) {
  size_t shmem = (size_t)K * T * E * sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(pipe_kernel<E>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)shmem));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  pipe_kernel<E><<<G, T, shmem>>>(d_in, d_out, chunks, R, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    pipe_kernel<E><<<G, T, shmem>>>(d_in, d_out, chunks, R, K);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

template <int E>
static float time_comp(int iters, int G, int T, int chunks, int R,
                       float* d_out) {
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  comp_only<E><<<G, T>>>(d_out, chunks, R);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) comp_only<E><<<G, T>>>(d_out, chunks, R);
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
  const int T = (argc > 2) ? atoi(argv[2]) : 256;
  const int E = 8;  // 每线程 8 个 float
  const int G = (argc > 3) ? atoi(argv[3]) : 132;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int clk_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d, clock=%.0f MHz\n", prop.name,
         prop.major, prop.minor, prop.multiProcessorCount, clk_khz / 1000.0);
  printf("grid G=%d, blockDim T=%d, E=%d, iters=%d\n\n", G, T, E, iters);

  const int CMAX = 256;
  const size_t total = (size_t)G * CMAX * T * E;
  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  {
    float* h = (float*)malloc(total * sizeof(float));
    for (size_t i = 0; i < total; ++i) h[i] = (float)((i % 1024) + 1) * 1e-3f;
    CUDA_CHECK(cudaMemcpy(d_in, h, total * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
  }

  float h_out = 0;
  auto bytes = [&](int chunks) {
    return (double)G * chunks * T * E * sizeof(float);
  };

  // ---- 参照：纯访存（R=0, K=8）与纯计算（扫 R）----
  printf("== reference ==\n");
  const int CREF = 256;
  float mem_ms = time_pipe<E>(iters, G, T, CREF, 0, KMAX, d_in, d_out, &h_out);
  printf("mem-only  (chunks=%d, R=0, K=%d): %9.4f ms   %.1f GB/s (HBM read)\n",
         CREF, KMAX, mem_ms, bytes(CREF) / (mem_ms * 1e-3) / 1e9);
  printf("%6s | %11s | %12s | %9s\n", "R", "comp ms", "us/chunk", "GFLOPs");
  printf("-------+-------------+--------------+----------\n");
  for (int R : {16, 32, 64, 128, 256}) {
    float cms = time_comp<E>(iters, G, T, CREF, R, d_out);
    double flops = 2.0 * (double)G * CREF * T * E * R;
    printf("%6d | %11.4f | %12.3f | %7.1f\n", R, cms,
           cms * 1e3 / CREF, flops / (cms * 1e-3) / 1e12);
  }

  // ---- Section A：R=64，扫流水深度 K ----
  printf("\n== Section A: R=64, chunks=256, sweep pipeline depth K ==\n");
  printf("%3s | %11s | %11s | %8s | %10s\n", "K", "serial-equiv ms",
         "pipeline ms", "speedup", "us/chunk");
  printf("----+-------------+-------------+----------+-----------\n");
  for (int K : {1, 2, 3, 4, 6, 8}) {
    float tp = time_pipe<E>(iters, G, T, CREF, 64, K, d_in, d_out, &h_out);
    float t1 = time_pipe<E>(iters, G, T, CREF, 64, 1, d_in, d_out, &h_out);
    printf("%3d | %11.4f | %11.4f | %7.2fx | %9.3f\n", K, t1, tp, t1 / tp,
           tp * 1e3 / CREF);
  }

  // ---- Section B：K=4，扫计算强度 R ----
  printf("\n== Section B: K=4, chunks=256, sweep R (compute intensity) ==\n");
  printf("%6s | %11s | %11s | %8s | %12s | %9s\n", "R", "K=1 ms", "K=4 ms",
         "speedup", "overlap ideal", "sp/ideal");
  printf("-------+-------------+-------------+----------+"
         "--------------+----------\n");
  float mem_p = time_pipe<E>(iters, G, T, CREF, 0, 4, d_in, d_out, &h_out);
  for (int R : {0, 16, 32, 64, 128, 256}) {
    float t1 = time_pipe<E>(iters, G, T, CREF, R, 1, d_in, d_out, &h_out);
    float tk = time_pipe<E>(iters, G, T, CREF, R, 4, d_in, d_out, &h_out);
    // 单 task 理论流水上界：把「理想深层访存」与「计算」做 max（K 足够深时）
    float cms = (R == 0) ? 0.f : time_comp<E>(iters, G, T, CREF, R, d_out);
    double ideal = (mem_p + cms) / (mem_p > cms ? mem_p : cms);
    printf("%6d | %11.4f | %11.4f | %7.2fx | %11.2fx | %8.2f%s\n", R, t1, tk,
           t1 / tk, ideal, (R && ideal > 0) ? (t1 / tk) / ideal : 0.0,
           (R ? "x" : "-"));
  }

  // ---- Section C：R=64、K=4，扫 chunks S ----
  printf("\n== Section C: R=64, K=4, sweep chunks S ==\n");
  printf("%7s | %11s | %11s | %8s\n", "chunks", "K=1 ms", "K=4 ms", "speedup");
  printf("--------+-------------+-------------+----------\n");
  for (int S : {16, 32, 64, 128, 256}) {
    float t1 = time_pipe<E>(iters, G, T, S, 64, 1, d_in, d_out, &h_out);
    float tk = time_pipe<E>(iters, G, T, S, 64, 4, d_in, d_out, &h_out);
    printf("%7d | %11.4f | %11.4f | %7.2fx\n", S, t1, tk, t1 / tk);
  }

  printf("\n>> 读法：\n");
  printf(">> K=1 = 每 chunk「等拷贝完再算」，两段严格串行（= kernel barrier 后的两个 kernel）；\n");
  printf(">> K>1 = 边算 chunk s 边预取后面 K-1 段拷贝，把访存藏进计算。\n");
  printf(">> Section A：深度 1->2 增益最大（K=1 把每段 HBM 延迟都暴露出来），随 K 加深趋饱和。\n");
  printf(">> Section B：R=0（纯访存）时 K 也能藏「相邻拷贝之间的延迟」；R 越大访存占比越小、\n");
  printf(">>   可重叠越多，speedup 趋近上界 (Tmem+Tcomp)/max(Tmem,Tcomp)。\n");
  printf(">> Section C：chunks 越多越接近稳态，speedup 收敛（每 block 只剩一次首延迟）。\n");
  printf(">> 含义：损失一（kernel barrier 挡住软件流水）在单 kernel 内**可回收**，回收比例由\n");
  printf(">>   访存延迟/计算时长之比决定——这解释了 MPK/DeepGEMM 为何用多层 Kstages（P11）。\n");

  printf("\ncorrectness: mem-only checksum = %.1f (仅确认计算未被优化掉)\n", h_out);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
