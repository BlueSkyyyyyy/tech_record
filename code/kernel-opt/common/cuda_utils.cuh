// kernel-opt 系列公共工具：设备信息、错误检查、CUDA event 计时、带宽/算力换算。
//
// 只依赖 CUDA runtime，header-only，直接 #include 即可。
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// 错误检查
// ---------------------------------------------------------------------------
inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[CUDA ERROR] %s at %s:%d -> %s\n", expr, file, line,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// ---------------------------------------------------------------------------
// 设备信息
// ---------------------------------------------------------------------------
struct DeviceInfo {
  char name[256];
  int sms;                     // SM 数量
  double clock_ghz;            // SM 主频 (GHz)
  double mem_bw_gbps;          // 显存理论带宽 (GB/s)
  double mem_gb;               // 显存容量 (GB)
  int max_threads_per_sm;
  int regs_per_sm;
  size_t shared_per_sm;        // 每 SM 共享内存 (byte)
};

inline DeviceInfo device_info(int dev = 0) {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  DeviceInfo d{};
  std::snprintf(d.name, sizeof(d.name), "%s", p.name);
  d.sms = p.multiProcessorCount;
  // CUDA 13 起 cudaDeviceProp 不再直接带 clockRate / memoryClockRate，改用属性查询。
  int clock_khz = 0, mem_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
  d.clock_ghz = clock_khz / 1e6;
  d.mem_bw_gbps = 2.0 * mem_clock_khz * (p.memoryBusWidth / 8.0) / 1e6;  // DDR * width
  d.mem_gb = p.totalGlobalMem / 1e9;
  d.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
  d.regs_per_sm = p.regsPerMultiprocessor;
  d.shared_per_sm = p.sharedMemPerMultiprocessor;
  return d;
}

inline void print_device_info(const DeviceInfo& d) {
  std::printf("Device : %s\n", d.name);
  std::printf("  SMs                : %d\n", d.sms);
  std::printf("  clock              : %.3f GHz\n", d.clock_ghz);
  std::printf("  HBM bandwidth(peak): %.1f GB/s\n", d.mem_bw_gbps);
  std::printf("  memory             : %.1f GB\n", d.mem_gb);
  std::printf("  max threads/SM     : %d  regs/SM: %d  smem/SM: %zu B\n",
              d.max_threads_per_sm, d.regs_per_sm, d.shared_per_sm);
}

// ---------------------------------------------------------------------------
// CUDA event 计时
// ---------------------------------------------------------------------------
class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  void stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, s));
    CUDA_CHECK(cudaEventSynchronize(stop_));
  }
  float ms() const {
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, start_, stop_));
    return t;
  }

 private:
  cudaEvent_t start_{}, stop_{};
};

// 对可调用对象 f 重复计时：先 warmup 次，再 iters 次，返回平均耗时 (ms)。
// 注意：测到的是「这一串调用的总墙钟 / iters」，包含 launch 与 host 派发开销；
// 单次 kernel 纯 device 时间请用 ncu 或 torch profiler（见系列第 03 篇）。
template <class F>
double bench_ms(F&& f, int warmup = 10, int iters = 50) {
  for (int i = 0; i < warmup; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.start();
  for (int i = 0; i < iters; ++i) f();
  t.stop();
  return t.ms() / static_cast<double>(iters);
}

// ---------------------------------------------------------------------------
// 带宽 / 算力换算与打印
// ---------------------------------------------------------------------------
inline double to_gbps(double bytes, double ms) {
  return bytes * 1000.0 / ms / 1e9;  // bytes / (ms/1000) / 1e9
}
inline double to_tflops(double flops, double ms) {
  return flops * 1000.0 / ms / 1e12;
}

// 打印一行正式结果：名称 / 耗时 / 有效带宽 / 带宽占比。
inline void report_mem(const char* name, double ms, double bytes, const DeviceInfo& d) {
  double gbps = to_gbps(bytes, ms);
  std::printf("%-28s %8.4f ms  %8.1f GB/s  (%5.1f%% of peak)\n", name, ms, gbps,
              100.0 * gbps / d.mem_bw_gbps);
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
template <typename T>
void fill_random(T* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    p[i] = static_cast<T>(1.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX));
}

inline int div_up(int a, int b) { return (a + b - 1) / b; }
