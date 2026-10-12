// cuda_graph.cu — MegaKernel 专题 P4-1 小实验：CUDA Graphs 消除了什么、没消除什么
//
// 目的（目标卡 H100 / sm90 实测）：
//   CUDA Graphs 把一串 kernel 捕获成图、之后用「一次图提交」低开销重放，直接治 launch 开销。
//   本实验量化：
//     (1) eager 逐步 launch 的 host 侧开销随 kernel 数线性增长；
//     (2) graph 重放把 host 侧开销压成「一次 cudaGraphLaunch」，per-kernel 成本被摊薄；
//     (3) graph 的固定成本：build/instantiate 一次，单次 cudaGraphLaunch 的固定延迟；
//     (4) graph 仍保留依赖语义：依赖链在 graph 里照样串行（用时间验证）。
//
//   注意：本实验只测 launch 开销，不评价 graph 内依赖的正确性/并行机会——后者见
//   harness/mitigations.out.txt 的 D 对照（依赖链 graph 1.05×，只降 launch 不破串行）。
//
// 编译：nvcc -O3 -arch=sm_90 cuda_graph.cu -o cuda_graph
// 用法：./cuda_graph [iters]

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

// 近乎空、但要保证不被编译器优化掉的 kernel；固定写一个 sink，单 block。
__global__ void tiny_kernel(int* sink) {
  if (threadIdx.x == 0) atomicAdd(sink, 1);
}

// 依赖链：同一 stream 上 N 个 tiny_kernel 顺序 launch，天然串行。
static void eager_chain(int N, int* sink, cudaStream_t s) {
  for (int i = 0; i < N; ++i) tiny_kernel<<<1, 32, 0, s>>>(sink);
}

static float time_eager_chain(int N, int iters, int* sink, cudaStream_t s) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) eager_chain(N, sink, s);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

static float time_graph_replay(int iters, cudaGraphExec_t gexec, cudaStream_t s) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) CUDA_CHECK(cudaGraphLaunch(gexec, s));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(b));
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
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount);
  printf("iters=%d\n\n", iters);

  int* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(int)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // 让 host 与 device 尽量解耦统计：先用一次高开销迭代热身。
  eager_chain(64, sink, s);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("%6s | %12s | %12s | %10s | %10s | %8s\n", "N", "eager ms/it",
         "graph ms/it", "eager ns/k", "graph ns/k", "speedup");
  printf("-------+--------------+--------------+------------+------------+---------\n");

  const int sizes[] = {4, 16, 64, 256, 1024};
  for (int si = 0; si < 5; ++si) {
    int N = sizes[si];
    // 捕获 N 个 tiny_kernel 的依赖链（同一 stream，天然串行）。
    cudaGraph_t graph;
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    eager_chain(N, sink, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));

    float te = time_eager_chain(N, iters, sink, s);
    float tg = time_graph_replay(iters, gexec, s);

    printf("%6d | %12.4f | %12.4f | %10.1f | %10.1f | %7.2fx\n", N, te, tg,
           te / N * 1e6, tg / N * 1e6, te / tg);

    CUDA_CHECK(cudaGraphExecDestroy(gexec));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }

  // 单节点 graph 的固定重放成本（一次 cudaGraphLaunch 的基线）。
  {
    cudaGraph_t graph;
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    tiny_kernel<<<1, 32, 0, s>>>(sink);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    float tg1 = time_graph_replay(iters, gexec, s);
    printf("\n单节点 graph 重放固定成本: %.4f ms/it (%.1f ns)  <- cudaGraphLaunch 的 host 侧固定延迟\n",
           tg1, tg1 * 1e6);
    CUDA_CHECK(cudaGraphExecDestroy(gexec));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }

  // build + instantiate 的一次性成本。
  {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    const int reps = 20;
    for (int r = 0; r < reps; ++r) {
      cudaGraph_t graph;
      cudaGraphExec_t gexec;
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      eager_chain(64, sink, s);
      CUDA_CHECK(cudaStreamEndCapture(s, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
      CUDA_CHECK(cudaGraphExecDestroy(gexec));
      CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    printf("64 节点 graph capture+instantiate 一次性成本: %.3f ms (%.1f us/次)\n",
           ms / reps, ms / reps * 1000.0f);
  }

  printf("\n>> 读法：eager 的 per-kernel host 开销随 N 基本线性；graph 重放把整图压成一次\n");
  printf(">> cudaGraphLaunch（固定 ~数 us），per-kernel 被摊到接近 0。这就是 CUDA Graphs\n");
  printf(">> 治 launch 开销的机理。但它不改变图内依赖——依赖边仍在 device 上被串行执行。\n");
  return 0;
}
