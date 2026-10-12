// probe_env.cu — MegaKernel 专题 P0-3 环境探测
//
// 目的：在目标卡（H100 / sm90）上确认本专题后续实验所需的三项能力是否可编译、可运行：
//   (1) persistent kernel + cooperative grid sync（P5 的「单持久化 kernel」基础）
//   (2) CUDA Graph capture/replay（P4-1 / P5-1 的 launch 开销对照）
//   (3) PDL = Programmatic Dependent Launch（P4-2 / P5-3 的跨 kernel 重叠）
//
// 只做「能不能跑 + 极简计时」，量产实验留给 P5。编译：
//   nvcc -O3 -arch=sm_90 -rdc=true probe_env.cu -o probe_env
// （-rdc=true 是为了 cooperative groups 的 grid sync 在部分工具链下更稳；本机无需也可，保留无妨）
//
// 用法：./probe_env [graph_iters]

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

// ---------------------------------------------------------------------------
// (1) persistent kernel：一个持久化 kernel，grid = #SM，内部做多轮 grid sync
// ---------------------------------------------------------------------------
__global__ void persistent_kernel(int* counter, int iters) {
  cg::grid_group g = cg::this_grid();
  int nblocks = gridDim.x;
  for (int it = 0; it < iters; ++it) {
    // 每个 block 做一点固定工作（访存 + 整数运算）
    int acc = 0;
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) acc += i ^ it;
    if (acc == -1) counter[blockIdx.x] = acc;  // 防止被优化掉
    g.sync();                                   // 跨 block 屏障（persistent 语义）
  }
  if (threadIdx.x == 0) atomicAdd(counter + nblocks, 1);
}

__global__ void trivial_kernel(int* counter, int iters) {
  int acc = 0;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) acc += i ^ iters;
  if (acc == -1) counter[blockIdx.x] = acc;
  if (threadIdx.x == 0) atomicAdd(counter + gridDim.x, 1);
}

// ---------------------------------------------------------------------------
// (3) PDL kernels：driver kernel + dependent kernel，后者先等依赖再干活
// ---------------------------------------------------------------------------
__global__ void pdl_producer(int* flag) {
  // 生产者写完数据后，允许后续 kernel 提前启动其「前导」部分
  if (threadIdx.x == 0) atomicAdd(flag, 1);
  cudaTriggerProgrammaticLaunchCompletion();  // 通知可提前 launch 下游
}

__global__ void pdl_consumer(int* out) {
  cudaGridDependencySynchronize();  // 等待上游全部完成（保证读到正确数据）
  if (threadIdx.x == 0) atomicAdd(out + 1, 1);
}

// ---------------------------------------------------------------------------
static bool device_supports_coop(int dev, int* max_blocks_per_sm) {
  int coop = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
  int v = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &v, persistent_kernel, 128, 0));
  *max_blocks_per_sm = v;
  return coop != 0;
}

static void probe_device_info() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  printf("== device ==\n");
  printf("name                : %s\n", p.name);
  printf("compute capability  : %d.%d\n", p.major, p.minor);
  printf("SM count            : %d\n", p.multiProcessorCount);
  printf("shared mem / block  : %zu bytes\n", p.sharedMemPerBlock);
  printf("shared mem / SM     : %zu bytes\n", p.sharedMemPerMultiprocessor);
  printf("regs / SM           : %d\n", p.regsPerMultiprocessor);
  printf("driver / runtime CUDA: %d / %d\n", driver, runtime);
  printf("cooperativeLaunch   : %s\n",
         p.cooperativeLaunch ? "YES" : "NO");
  printf("memoryPoolsSupported: %s\n",
         p.memoryPoolsSupported ? "YES" : "NO");
  printf("\n");
}

static void probe_persistent() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int max_bpsm = 0;
  bool coop = device_supports_coop(dev, &max_bpsm);
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int nsm = p.multiProcessorCount;

  printf("== (1) persistent kernel + cooperative grid sync ==\n");
  printf("cooperativeLaunch supported : %s\n", coop ? "YES" : "NO");
  printf("persistent_kernel occupancy  : %d blocks/SM @128 threads\n", max_bpsm);
  if (!coop) {
    printf("RESULT: UNSUPPORTED (skip)\n\n");
    return;
  }

  const int iters = 200;
  int* d_counter = nullptr;
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int) * (nsm + 2)));
  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int) * (nsm + 2)));

  int grid = nsm;  // 每 SM 一个 worker block（MPK 的 worker 语义）
  void* args[] = {&d_counter, (void*)&iters};

  // warmup
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)persistent_kernel,
                                         dim3(grid), dim3(128), args, 0, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(a));
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)persistent_kernel,
                                         dim3(grid), dim3(128), args, 0, 0));
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

  // 对照：同样数量的 trivial kernel 串行 launch（无 grid barrier）
  CUDA_CHECK(cudaEventRecord(a));
  for (int it = 0; it < iters; ++it)
    trivial_kernel<<<grid, 128>>>(d_counter, it);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms_trivial = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_trivial, a, b));

  printf("persistent kernel (grid=%d, %d iters, grid sync each): %.3f ms\n",
         grid, iters, ms);
  printf("trivial kernels   (%d separate launches, no sync):     %.3f ms\n",
         iters, ms_trivial);
  printf("RESULT: SUPPORTED & RAN\n\n");

  CUDA_CHECK(cudaFree(d_counter));
}

static void probe_cuda_graph(int iters) {
  printf("== (2) CUDA Graph capture/replay ==\n");
  const int grid = 128, threads = 128;
  int* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));

  int nk = 32;  // 图中 32 个小 kernel

  // 非 graph：串行 launch nk 个 kernel，重复 iters 次
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    for (int k = 0; k < nk; ++k) trivial_kernel<<<grid, threads>>>(d, k);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms_plain = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_plain, a, b));

  // 捕获成 graph（用独立 stream；legacy default stream 在 global 模式下不可捕获）
  cudaStream_t cap_stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&cap_stream));
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeThreadLocal));
  for (int k = 0; k < nk; ++k) trivial_kernel<<<grid, threads, 0, cap_stream>>>(d, k);
  CUDA_CHECK(cudaStreamEndCapture(cap_stream, &graph));
  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) CUDA_CHECK(cudaGraphLaunch(exec, 0));
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms_graph = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_graph, a, b));

  printf("graph nodes                  : %d kernels\n", nk);
  printf("plain  %d x %d launches      : %.3f ms\n", iters, nk, ms_plain);
  printf("graph  %d replays            : %.3f ms\n", iters, ms_graph);
  printf("speedup (plain/graph)        : %.2fx\n", ms_plain / ms_graph);
  printf("RESULT: SUPPORTED & RAN\n\n");

  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(cap_stream));
  CUDA_CHECK(cudaFree(d));
}

static const char* pdl_support_string() {
  // PDL 需要 sm90+ 与 CUDA >= 11.8；运行时以实际 launch 是否成功为准
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  if (p.major < 9) return "NO (needs sm90+)";
  return "likely YES (sm90+, CUDA 13.2)";
}

static void probe_pdl() {
  printf("== (3) PDL (Programmatic Dependent Launch) ==\n");
  printf("precheck                     : %s\n", pdl_support_string());

  int* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int) * 2));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int) * 2));

  // 构造带 PDL 属性的 launch
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = 0;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  cudaError_t e1 = cudaLaunchKernelEx(&cfg, pdl_producer, d);
  cudaError_t e2 = cudaLaunchKernelEx(&cfg, pdl_consumer, d);
  cudaError_t e3 = cudaDeviceSynchronize();

  printf("launch producer (PDL)        : %s\n",
         e1 == cudaSuccess ? "ok" : cudaGetErrorString(e1));
  printf("launch consumer (PDL)        : %s\n",
         e2 == cudaSuccess ? "ok" : cudaGetErrorString(e2));
  printf("device sync                  : %s\n",
         e3 == cudaSuccess ? "ok" : cudaGetErrorString(e3));

  if (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess) {
    int h[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(h, d, sizeof(int) * 2, cudaMemcpyDeviceToHost));
    printf("flags after run              : producer=%d consumer=%d\n", h[0], h[1]);
    printf("RESULT: SUPPORTED & RAN (both kernels completed)\n\n");
  } else {
    printf("RESULT: UNSUPPORTED / FAILED\n\n");
  }
  CUDA_CHECK(cudaFree(d));
}

int main(int argc, char** argv) {
  int iters = 200;
  if (argc > 1) iters = std::atoi(argv[1]);
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  probe_device_info();
  probe_persistent();
  probe_cuda_graph(iters);
  probe_pdl();
  printf("== done ==\n");
  return 0;
}
