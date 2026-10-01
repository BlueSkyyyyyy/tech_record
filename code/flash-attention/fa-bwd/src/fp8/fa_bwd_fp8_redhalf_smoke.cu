// fa_bwd_fp8_redhalf_smoke.cu —— O114 前置冒烟：验证「把 dK/dV 的跨 CTA 归约从 fp32
//   `red.global.add.f32` 收窄到 fp16 `red.global.add.f16`（元素宽度减半）在本卡的
//   (a) 指令是否原生 RED（而非 CAS 循环）、(b) L2 扇区数是否减半、(c) 吞吐是否不劣。
//
// 背景：fp8 反向默认 main 的 L2 墙 = dK/dV 跨 CTA `red`（S4096 ~114.5M 扇区 = L2 的 ~80%）。
//   ROADMAP「阻塞」已判「red 由工作划分决定、归约机制（atomic/TMA）无关」，但那条结论是在
//   **元素宽度恒为 fp32** 的前提下——把被归约的元素从 fp32 收窄到 fp16，每个元素写入字节
//   减半 ⇒ 扇区数应减半（与机制无关）。本冒烟先钉死硬件是否支持「原生、扇区减半、不慢」的
//   fp16 原子加，再决定是否接进 `fp8_mma_body`（下一轮）。
//
// 用法：scripts/run.sh src/fp8/fa_bwd_fp8_redhalf_smoke.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

template <int NINST>
__global__ void red_f32_kernel(float2* out, int cols) {
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int base = (blockIdx.x * blockDim.x + w * 32) * 2;
#pragma unroll
  for (int i = 0; i < NINST; ++i) {
    long long e = (long long)base + (long long)lane * 2 + (long long)i * 64 * gridDim.x;
    int col = (int)(e % cols);
    int row = (int)(e / cols);
    float2 v = make_float2((float)lane * 1e-3f, (float)lane * 2e-3f);
    atomicAdd(reinterpret_cast<float2*>(out + (long long)row * (cols / 2) + col / 2), v);
  }
}

template <int NINST>
__global__ void red_f16_kernel(__half2* out, int cols) {
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int base = (blockIdx.x * blockDim.x + w * 32) * 2;
#pragma unroll
  for (int i = 0; i < NINST; ++i) {
    long long e = (long long)base + (long long)lane * 2 + (long long)i * 64 * gridDim.x;
    int col = (int)(e % cols);
    int row = (int)(e / cols);
    __half2 v = __floats2half2_rn((float)lane * 1e-3f, (float)lane * 2e-3f);
    atomicAdd(out + (long long)row * (cols / 2) + col / 2, v);
  }
}

// 直接用 PTX `red.global.add.noftz.f16x2`（无返回、走 RED 路径）——CUDA 的
//   `atomicAdd(__half2*)` 头实现会退化成带返回的 `ATOM`（读改写），本变体强制 RED。
__device__ __forceinline__ void red_add_f16x2(void* p, __half2 v) {
  unsigned packed = *reinterpret_cast<unsigned*>(&v);
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(p), "r"(packed) : "memory");
}

template <int NINST>
__global__ void red_f16red_kernel(__half2* out, int cols) {
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int base = (blockIdx.x * blockDim.x + w * 32) * 2;
#pragma unroll
  for (int i = 0; i < NINST; ++i) {
    long long e = (long long)base + (long long)lane * 2 + (long long)i * 64 * gridDim.x;
    int col = (int)(e % cols);
    int row = (int)(e / cols);
    __half2 v = __floats2half2_rn((float)lane * 1e-3f, (float)lane * 2.0e-3f);
    red_add_f16x2(out + (long long)row * (cols / 2) + col / 2, v);
  }
}

// ---- 模仿 `mma.m16n8` 累加器片段：warp 内 lane = 4*g + c2，g=0..7（8 个不同行，行距 = rowstride），
//      c2=0..3（每行连续 4 个 lane，列 = c2*2）。fp32 float2 与 fp16 f16x2 各写 2 元素/lane。
//      这正是真实 dK/dV epilogue 的访问模式 ⇒ 用 ncu 可验证：此模式下一次 warp 请求固定吃
//      **8 个 32B 扇区**（8 行各一个），与元素宽度无关 ⇒ fp16 只把每扇区填充率从满降到半，
//      **扇区数不减**。这解释了为什么真实 main kernel 的 `lts op_red` 一字不变。
template <int NINST>
__global__ void red_scatter_f32_kernel(float* out, int rowstride) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c2 = lane & 3;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
#pragma unroll
  for (int i = 0; i < NINST; ++i) {
    long long row = ((long long)wid * NINST + i) * 8 + g;   // 每请求覆盖 8 行（各 1 扇区）
    float2 v = make_float2((float)g * 1e-3f, (float)c2 * 1e-3f);
    atomicAdd(reinterpret_cast<float2*>(out + row * rowstride + c2 * 2), v);
  }
}
template <int NINST>
__global__ void red_scatter_f16_kernel(__half* out, int rowstride) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, c2 = lane & 3;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
#pragma unroll
  for (int i = 0; i < NINST; ++i) {
    long long row = ((long long)wid * NINST + i) * 8 + g;
    __half2 v = __floats2half2_rn((float)g * 1e-3f, (float)c2 * 1e-3f);
    red_add_f16x2(out + row * rowstride + c2 * 2, v);
  }
}

int main(int argc, char** argv) {
  const int NINST = 64;
  const int threads = 256;
  const int blocks = 4 * 132;
  const int cols = 131072;
  const long long rows = (long long)blocks * threads * 2 * NINST / cols + 2;
  const long long n = rows * cols;
  printf("smoke: blocks=%d threads=%d NINST=%d cols=%d rows=%lld elems=%.2fM\n",
         blocks, threads, NINST, cols, rows, n / 1e6);

  float2* d_f32; __half2* d_f16;
  CK(cudaMalloc(&d_f32, n * sizeof(float)));
  CK(cudaMalloc(&d_f16, n * sizeof(__half)));
  CK(cudaMemset(d_f32, 0, n * sizeof(float)));
  CK(cudaMemset(d_f16, 0, n * sizeof(__half)));
  int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s sm_%d%d\n", prop.name, prop.major, prop.minor);

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double elems = (double)blocks * threads * NINST * 2;
  auto bench = [&](auto kernel, auto* buf, const char* tag, int wsize, int iters) -> float {
    for (int i = 0; i < 3; ++i) kernel<<<blocks, threads>>>(buf, cols);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) kernel<<<blocks, threads>>>(buf, cols);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
    printf("[%s] %.4f ms  elems=%.1fM  write=%.3f GB  eff=%.2f TB/s\n", tag, ms, elems / 1e6,
           elems * wsize / 1e9, elems * wsize / (ms * 1e-3) / 1e12);
    return ms;
  };
  float m32 = bench(red_f32_kernel<NINST>, d_f32, "red.f32x2", 4, 50);
  float m16 = bench(red_f16_kernel<NINST>, d_f16, "red.f16x2(atom)", 2, 50);
  float m16r = bench(red_f16red_kernel<NINST>, d_f16, "red.f16x2(red)", 2, 50);
  printf("f16(atom)/f32 = %.3f | f16(red)/f32 = %.3f\n", m16 / m32, m16r / m32);

  // ---- 模仿 mma 片段的 row-scatter 模式（真实 dK/dV epilogue）----
  const int rs = 64;                        // 每行 64 元素
  const long long nrows = (long long)(blocks * threads / 32) * NINST * 8;
  printf("scatter: nrows=%lld rowstride=%d (每请求 8 行⇒预期 8 扇区/请求，与宽度无关)\n", nrows, rs);
  float* s_f32; __half* s_f16;
  CK(cudaMalloc(&s_f32, nrows * rs * sizeof(float)));
  CK(cudaMalloc(&s_f16, nrows * rs * sizeof(__half)));
  CK(cudaMemset(s_f32, 0, nrows * rs * sizeof(float)));
  CK(cudaMemset(s_f16, 0, nrows * rs * sizeof(__half)));
  auto bench2 = [&](auto kernel, auto* buf, const char* tag, int wsize, int iters) -> float {
    for (int i = 0; i < 3; ++i) kernel<<<blocks, threads>>>(buf, rs);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) kernel<<<blocks, threads>>>(buf, rs);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
    printf("[%s] %.4f ms  write=%.3f GB eff=%.2f TB/s\n", tag, ms, elems * wsize / 1e9,
           elems * wsize / (ms * 1e-3) / 1e12);
    return ms;
  };
  float s32 = bench2(red_scatter_f32_kernel<NINST>, s_f32, "scatter.f32x2", 4, 50);
  float s16 = bench2(red_scatter_f16_kernel<NINST>, s_f16, "scatter.f16x2", 2, 50);
  printf("scatter f16/f32 time ratio = %.3f (字节减半但扇区数同⇒时间应≈1.0)\n", s16 / s32);
  CK(cudaFree(s_f32)); CK(cudaFree(s_f16));

  CK(cudaFree(d_f32)); CK(cudaFree(d_f16));
  printf("PASS\n");
  return 0;
}
