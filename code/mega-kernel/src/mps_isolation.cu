// P39-1: 多租户共存隔离的第二条路线 —— MPS（Multi-Process Service）
// 目标卡 H100 / sm90。
//
// 动机：P38-1 用 Hopper Green Contexts 把整机 SM「按张数」分区，给共存负载硬隔离。
//   §27.6 明确留了「未测 MPS 作为另一条隔离路线」。MPS 是**跨进程**共享一个 CUDA context
//   的服务：没有 MPS 时，不同进程的 kernel 不能在 GPU 上真正并发（context 按 kernel 粒度
//   时分/抢占）；有了 MPS 才允许跨进程并发，并可用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
//   给每个 client 设「线程容量上限」（≈ 占用率上限，而非 SM 张数）。
//
// 本程序是**跨进程**实验的被测体（单 GPU 上跑两个进程）：
//   `mps_isolation persist <MB> <rounds> <marker>` —— 整机持久化、访存为主的内核；
//       起内核后写 marker 文件（供脚本判「已在跑」），结束打印自身吞吐。
//   `mps_isolation victim  <MB> <passes>`        —— 小内核，用 CUDA event 计时（含被饿死等待），
//       打印 min/median（ms）。
// 驱动脚本 harness/mps_isolation.sh 负责起/停 MPS daemon 与编排对照组。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("[FAIL] %s -> %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

// 整机持久化、访存为主：rounds 轮 grid-stride 求和（只读，带宽主导）。
__global__ void persist_kernel(const float* __restrict__ buf, long n, int rounds, float* sink) {
  float acc = 0.f;
  for (int r = 0; r < rounds; ++r) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
      acc += buf[i];
    }
  }
  if (acc == -123456.f) sink[0] = acc;   // 防优化
}

// victim：passes 轮 grid-stride 求和，末线程写 sink（防优化）。
__global__ void victim_kernel(const float* __restrict__ buf, long n, int passes, float* sink) {
  float acc = 0.f;
  for (int p = 0; p < passes; ++p) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
      acc += buf[i];
    }
  }
  if (acc == -123456.f) sink[0] = acc;   // 防优化
}

static int mode_persist(long MB, int rounds, const char* marker) {
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  long n = MB * 1024L * 1024L / 4;
  float *buf, *sink;
  CK(cudaMalloc(&buf, n * sizeof(float)));
  CK(cudaMemset(buf, 1, n * sizeof(float)));
  CK(cudaMalloc(&sink, sizeof(float)));
  int bps = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, persist_kernel, 256, 0));
  int Gp = bps * prop.multiProcessorCount;
  printf("PERSIST dev=%s SMs=%d blk/SM=%d grid=%d MB=%ld rounds=%d\n",
         prop.name, prop.multiProcessorCount, bps, Gp, MB, rounds);
  fflush(stdout);

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  CK(cudaEventRecord(e0, s));
  persist_kernel<<<Gp, 256, 0, s>>>(buf, n, rounds, sink);
  CK(cudaGetLastError());
  CK(cudaEventRecord(e1, s));

  if (marker) { FILE* f = fopen(marker, "w"); if (f) { fprintf(f, "%d\n", Gp); fclose(f); } }

  CK(cudaStreamSynchronize(s));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  double gbps = (double)n * 4 * rounds / (ms / 1e3) / 1e9;
  printf("PERSIST_DONE ms=%.3f gbps=%.1f\n", ms, gbps);
  cudaFree(buf); cudaFree(sink); cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
  return 0;
}

static int mode_victim(long MB, int passes, int capBlocks) {
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  long n = MB * 1024L * 1024L / 4;
  float *buf, *sink;
  CK(cudaMalloc(&buf, n * sizeof(float)));
  CK(cudaMemset(buf, 1, n * sizeof(float)));
  CK(cudaMalloc(&sink, sizeof(float)));
  int bps = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, victim_kernel, 256, 0));
  int Gv = bps * prop.multiProcessorCount;
  if (capBlocks > 0) Gv = capBlocks;
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  auto once = [&](double* out) -> int {
    CK(cudaEventRecord(e0, s));
    victim_kernel<<<Gv, 256, 0, s>>>(buf, n, passes, sink);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1, s));
    // 有界等待：被整机 persistent 饿死时 5s 超时（返回 -1）
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    double t0 = ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      clock_gettime(CLOCK_MONOTONIC, &ts);
      if (ts.tv_sec * 1e3 + ts.tv_nsec / 1e6 - t0 > 5000.0) { *out = -1; return 1; }
    }
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    *out = ms; return 0;
  };

  for (int i = 0; i < 5; i++) { double x; int r = once(&x); if (r) { printf("VICTIM dev=%s grid=%d TIMEOUT(first)\n", prop.name, Gv); return 0; } }
  double mn = 1e30, arr[7];
  for (int i = 0; i < 7; i++) { double x; int r = once(&x); if (r) { printf("VICTIM dev=%s grid=%d TIMEOUT\n", prop.name, Gv); return 0; } arr[i] = x; if (x < mn) mn = x; }
  // 中位数
  for (int i = 0; i < 7; i++) for (int j = i + 1; j < 7; j++) if (arr[j] < arr[i]) { double t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
  double md = arr[3];
  printf("VICTIM dev=%s grid=%d passes=%d min=%.3f median=%.3f ms\n", prop.name, Gv, passes, mn, md);
  cudaFree(buf); cudaFree(sink); cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: %s {persist|victim} MB [param] [marker] [capBlocks]\n", argv[0]); return 1; }
  const char* mode = argv[1];
  long MB = atol(argv[2]);
  if (!strcmp(mode, "persist")) {
    int rounds = (argc > 3) ? atoi(argv[3]) : 20000;
    const char* marker = (argc > 4) ? argv[4] : nullptr;
    return mode_persist(MB, rounds, marker);
  } else if (!strcmp(mode, "victim")) {
    int passes = (argc > 3) ? atoi(argv[3]) : 20;
    int cap = (argc > 5) ? atoi(argv[5]) : 0;
    return mode_victim(MB, passes, cap);
  }
  printf("unknown mode %s\n", mode);
  return 1;
}
