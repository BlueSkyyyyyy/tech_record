// launch_overhead.cu — MegaKernel 专题 P5-1 小实验：N 次 launch 的三种执行策略对照
//
// 目的（目标卡 H100 / sm90 实测）：
//   对「N 个独立、等量的微小工作单元」，比较三种把它们跑到 GPU 上的策略：
//     (A) eager    ：N 个独立 kernel 逐个 launch（依赖链，同一 stream 天然串行）；
//     (B) graph    ：把上面 N 个 kernel 捕获成一张 CUDA Graph，之后一次 cudaGraphLaunch 重放；
//     (C) persistent：单个常驻 kernel，grid = #SM，每个 block 用 grid-stride 循环处理 N 个单元
//                     （即 MPK 的 worker 语义：一次 launch、kernel 内循环消费任务）。
//
//   三者做**完全相同的设备侧工作**（每单元一次 LCG 自旋 + 一次 atomicAdd 计数），
//   所以差异只来自「launch / 调度开销」以及「设备侧 grid 启动次数」。
//
//   两个横切：
//     - Section A（K=0）：单元几乎无计算 → 纯 launch 开销主导，看谁把 launch 开销消掉。
//     - Section B（N=256）：逐步加大每单元计算量 K → 找 launch 开销被计算淹没的交叉点，
//       说明 persistent/megakernel 的价值是「在 launch-bound 区间把开销清零」，
//       一旦 device 侧计算主导，三种策略趋同。
//
// 编译：nvcc -O3 -arch=sm_90 launch_overhead.cu -o launch_overhead
// 用法：./launch_overhead [iters]   （iters 默认 200）

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

// 每个「工作单元」的设备侧工作：LCG 自旋 K 次 + 一次 atomicAdd 计数。
// 对 tiny_kernel 是 1 个 block 做一次；对 persistent 是每个 block 每轮做一次。
__device__ __forceinline__ void do_unit(int* sink, int K) {
  int acc = threadIdx.x * 2654435761u;
  for (int i = 0; i < K; ++i) acc = acc * 1664525 + 1013904223;
  if (acc == -123456789) sink[1] = acc;  // 依赖 acc，阻止编译器删掉自旋
  if (threadIdx.x == 0) atomicAdd(sink, 1);
}

__global__ void tiny_kernel(int* sink, int K) { do_unit(sink, K); }

// persistent worker：一次 launch，grid-stride 消费 N 个单元，全部在 kernel 内完成。
__global__ void persistent_units(int* sink, int N, int K) {
  for (int u = blockIdx.x; u < N; u += gridDim.x) do_unit(sink, K);
}

static void eager_chain(int N, int K, int* sink, cudaStream_t s) {
  for (int i = 0; i < N; ++i) tiny_kernel<<<1, 32, 0, s>>>(sink, K);
}

static float time_eager(int N, int K, int iters, int* sink, cudaStream_t s) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) eager_chain(N, K, sink, s);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

static float time_graph(int iters, cudaGraphExec_t g, cudaStream_t s) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) CUDA_CHECK(cudaGraphLaunch(g, s));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

static float time_persistent(int N, int K, int grid, int iters, int* sink,
                             cudaStream_t s) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  // warmup
  persistent_units<<<grid, 32, 0, s>>>(sink, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    persistent_units<<<grid, 32, 0, s>>>(sink, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

// 构造 N 节点的 graph（依赖链）并返回 exec。
static cudaGraphExec_t build_chain_graph(int N, int K, int* sink,
                                         cudaStream_t s) {
  cudaGraph_t graph;
  cudaGraphExec_t gexec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  eager_chain(N, K, sink, s);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return gexec;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 200;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int nsm = prop.multiProcessorCount;
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, nsm);
  printf("iters=%d, persistent grid=%d blocks x 32 threads\n\n", iters, nsm);

  int* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int) * 2));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(int) * 2));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  eager_chain(64, 0, sink, s);
  CUDA_CHECK(cudaDeviceSynchronize());

  // -------------------------------------------------------------------------
  // Section A：K=0，纯 launch 开销主导，扫 N
  // -------------------------------------------------------------------------
  printf("== Section A: per-unit work K=0 (launch-bound) ==\n");
  printf("%6s | %11s | %11s | %11s | %9s | %9s | %8s | %8s\n", "N", "eager ms",
         "graph ms", "persist ms", "eager ns/u", "persist ns/u", "e/g", "e/p");
  printf("-------+-------------+-------------+-------------+-----------+-----------+----------+---------\n");

  const int sizes[] = {4, 16, 64, 256, 1024, 4096};
  for (int si = 0; si < 6; ++si) {
    int N = sizes[si];
    int K = 0;
    cudaGraphExec_t g = build_chain_graph(N, K, sink, s);
    float te = time_eager(N, K, iters, sink, s);
    float tg = time_graph(iters, g, s);
    float tp = time_persistent(N, K, nsm, iters, sink, s);
    printf("%6d | %11.4f | %11.4f | %11.4f | %9.1f | %9.1f | %7.2fx | %7.2fx\n",
           N, te, tg, tp, te / N * 1e6, tp / N * 1e6, te / tg, te / tp);
    CUDA_CHECK(cudaGraphExecDestroy(g));
  }

  // -------------------------------------------------------------------------
  // Section B：固定 N=256，扫每单元计算量 K，找 launch-bound → compute-bound 交叉。
  //   为把「launch 开销」和「并行度」两个因素分开，persistent 给两个网格：
  //     - persistG ：grid=#SM，132 个 block 并行消费 N 个单元（真 megakernel 并行度）；
  //     - persist1 ：grid=1，单 block 串行跑完 N 个单元 —— 设备侧工作与 eager
  //                  （N 个 1-block kernel 串行）**完全等价**，因而二者之差只剩 launch 开销。
  //   K 增大到计算主导时，eager 与 persist1 应收敛（证明差额本就是 launch 开销），
  //   而 persistG 因并行度保持领先。
  // -------------------------------------------------------------------------
  printf("\n== Section B: N=256, per-unit work K sweep (crossover) ==\n");
  printf("%10s | %11s | %11s | %11s | %11s | %8s | %8s | %8s\n", "K (per unit)",
         "eager ms", "graph ms", "persist1 ms", "persistG ms", "e/g", "e/p1",
         "e/pG");
  printf("-----------+-------------+-------------+-------------+-------------+----------+---------+--------\n");

  const int N = 256;
  const int ks[] = {0, 64, 256, 1024, 4096, 16384};
  for (int ki = 0; ki < 6; ++ki) {
    int K = ks[ki];
    cudaGraphExec_t g = build_chain_graph(N, K, sink, s);
    float te = time_eager(N, K, iters, sink, s);
    float tg = time_graph(iters, g, s);
    float tp1 = time_persistent(N, K, 1, iters, sink, s);
    float tpG = time_persistent(N, K, nsm, iters, sink, s);
    printf("%10d | %11.4f | %11.4f | %11.4f | %11.4f | %7.2fx | %6.2fx | %6.2fx\n",
           K, te, tg, tp1, tpG, te / tg, te / tp1, te / tpG);
    CUDA_CHECK(cudaGraphExecDestroy(g));
  }

  // 正确性/工作量核对：三种策略都应把 sink[0] 增加 N 次。
  {
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(int) * 2));
    int NN = 256;
    eager_chain(NN, 8, sink, s);
    persistent_units<<<1, 32, 0, s>>>(sink, NN, 8);
    persistent_units<<<nsm, 32, 0, s>>>(sink, NN, 8);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h = 0;
    CUDA_CHECK(cudaMemcpy(&h, sink, sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nwork check (eager + persist1 + persistG, each N=%d): sink[0]=%d (expect %d)\n",
           NN, h, 3 * NN);
  }

  printf("\n>> 读法：K=0 时 eager 的 per-unit 时间≈host launch 间隔(~1.8us)，\n");
  printf(">> graph 把 N 次 launch 压成一次提交；persistent 直接=一次 launch 内跑完，\n");
  printf(">> 连设备侧 N 次 grid 启动都省掉。Section B 里 persist1(单 block 串行) 与\n");
  printf(">> eager 设备侧工作等价，K 增大时二者收敛→证明差额本就是 launch 开销；\n");
  printf(">> persistG 靠 132 block 并行度继续领先。megakernel 只治 launch-bound\n");
  printf(">> 区间，纯计算主导且并行度已饱和时收益自然消失。\n");
  return 0;
}
