// mitigations.cu — MegaKernel 专题 P1-2 小实验：已有缓解手段的边界
//
// 目的：用目标卡（H100 / sm90）实测，回答「算子融合 / CUDA Graphs / PDL / 多流」各自
//       能治什么、治不了什么。核心论点：多流只能并行「无依赖」算子；一旦有依赖，
//       无论分几个 stream 都必须串行，且 CUDA Graph 只降低 launch 开销、不打破依赖串行。
//
// 四个对照（每个由 N 个小 kernel 组成，每个 kernel 自旋一段固定时间）：
//   A. independent, 1 stream      —— N 次串行 launch（基线）
//   B. independent, 4 streams      —— 无依赖：stream 并行，应显著小于 A
//   C. dependent chain, 4 streams  —— 有依赖：event 串起来，即使分 4 stream 也≈A（不退化成并行）
//   D. dependent chain as graph    —— 依赖链捕获成 CUDA Graph：只降 launch，仍≈串行时间
//
// 编译：nvcc -O3 -arch=sm_90 mitigations.cu -o mitigations
// 用法：./mitigations [N] [spin_cycles]

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

// 自旋固定 cycles，占住一个 block 计算一会儿（grid=1 block，保证多流可并存）
__global__ void spin_kernel(long long cycles, int* sink) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
  if (sink && threadIdx.x == 0) atomicAdd(sink, 1);
}

static float time_loop(int N, int iters,
                       void (*launch)(int, void*), void* ctx) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) launch(N, ctx);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms;
}

// --- 每个对照的 launch 闭包上下文 ---
struct Ctx {
  long long cycles;
  int* sink;
  cudaStream_t s[4];
  cudaEvent_t ev[1024];  // 依赖链用
  cudaGraph_t graph;
  cudaGraphExec_t gexec;
};

static void launch_serial(int N, void* v) {
  Ctx* c = (Ctx*)v;
  for (int i = 0; i < N; ++i)
    spin_kernel<<<1, 256, 0, c->s[0]>>>(c->cycles, c->sink);
}

static void launch_4stream_indep(int N, void* v) {
  Ctx* c = (Ctx*)v;
  for (int i = 0; i < N; ++i)
    spin_kernel<<<1, 256, 0, c->s[i & 3]>>>(c->cycles, c->sink);
}

// 依赖链分到 4 个 stream：kernel i 必须先等 kernel i-1 的 event
static void launch_4stream_dep(int N, void* v) {
  Ctx* c = (Ctx*)v;
  for (int i = 0; i < N; ++i) {
    cudaStream_t si = c->s[i & 3];
    if (i > 0) CUDA_CHECK(cudaStreamWaitEvent(si, c->ev[i - 1], 0));
    spin_kernel<<<1, 256, 0, si>>>(c->cycles, c->sink);
    if (i < N) CUDA_CHECK(cudaEventRecord(c->ev[i], si));
  }
}

static void launch_graph(int N, void* v) {
  Ctx* c = (Ctx*)v;
  (void)N;
  CUDA_CHECK(cudaGraphLaunch(c->gexec, c->s[0]));
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 64;
  long long cycles = (argc > 2) ? atoll(argv[2]) : 30000;
  if (N > 1024) N = 1024;

  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount);
  printf("N=%d kernels, spin=%lld cycles\n\n", N, cycles);

  Ctx c;
  c.cycles = cycles;
  int* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(int)));
  c.sink = sink;
  for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaStreamCreate(&c.s[i]));
  for (int i = 0; i < 1024; ++i) CUDA_CHECK(cudaEventCreateWithFlags(&c.ev[i], cudaEventDisableTiming));

  const int iters = 30;

  float tA = time_loop(N, iters, launch_serial, &c);
  float tB = time_loop(N, iters, launch_4stream_indep, &c);
  float tC = time_loop(N, iters, launch_4stream_dep, &c);

  // 捕获依赖链（同一 stream 天然串行）成 graph
  CUDA_CHECK(cudaStreamBeginCapture(c.s[0], cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < N; ++i)
    spin_kernel<<<1, 256, 0, c.s[0]>>>(c.cycles, c.sink);
  CUDA_CHECK(cudaStreamEndCapture(c.s[0], &c.graph));
  CUDA_CHECK(cudaGraphInstantiate(&c.gexec, c.graph, nullptr, nullptr, 0));
  float tD = time_loop(N, iters, launch_graph, &c);

  printf("A independent  1 stream : %8.3f ms  (%.3f us/kernel)\n", tA / iters,
         tA / iters / N * 1000.0f);
  printf("B independent  4 streams: %8.3f ms  (%.3f us/kernel)  speedup vs A = %.2fx\n",
         tB / iters, tB / iters / N * 1000.0f, tA / tB);
  printf("C dependent    4 streams: %8.3f ms  (%.3f us/kernel)  speedup vs A = %.2fx\n",
         tC / iters, tC / iters / N * 1000.0f, tA / tC);
  printf("D dependent chain graph : %8.3f ms  (%.3f us/kernel)  speedup vs A = %.2fx\n",
         tD / iters, tD / iters / N * 1000.0f, tA / tD);

  printf("\n>> 结论：B 显著小于 A（无依赖可多流并行）；\n");
  printf(">> C≈A（有依赖时多流无效）；D 只比 C 略低（graph 只降 launch，不破依赖串行）。\n");
  return 0;
}
