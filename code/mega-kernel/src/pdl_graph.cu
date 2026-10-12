// pdl_graph.cu — MegaKernel 专题 P19-1 小实验：PDL 与 CUDA Graph 叠加（能不能相乘？）
//
// 目的（目标卡 H100 / sm90 实测）：
//   CUDA Graph（P4-1）治的是**损失二（launch 开销）**，但它**不改变依赖串行**（依赖链仅 1.05×）；
//   PDL（P4-2/P5-3）治的是**损失一的一个特例**——相邻 kernel 重叠 min(上游 tail, 下游 prologue)。
//   两者机制正交，`docs/03` §2.3 说「PDL 与 CUDA Graphs 可叠加」，但 `docs/04` §6.6 明确记着
//   **本专题未实测该叠加**。本实验把 K 个 kernel 串成同一条 PDL 链，对四种组合公平对照：
//       (a) eager_nopdl : K 次 host launch，无 PDL         —— 退化基线
//       (b) eager_pdl   : K 次 host launch，k>0 带 PDL 属性 —— P4-2/P5-3 的已知形态
//       (c) graph_nopdl : 把 (a) capture 成图、replay        —— 只有 Graph
//       (d) graph_pdl   : 把 (b) capture 成图、replay        —— Graph × PDL 叠加
//   回答三个问题：
//     Q1  **CUDA Graph capture 是否保留 PDL 语义？**——graph_pdl 相对 graph_nopdl 是否仍有
//          每条链边 min(tail, prologue) 的重叠（用正确性计数证明排序仍被 PDL 保证）。
//     Q2  **两者是否「相乘」？**——graph_pdl 相对 eager_nopdl 的加速是否 ≈ (Graph 收益)×(PDL 收益)。
//     Q3  **谁封顶谁？**——当 K 很大、launch 开销被 Graph 摊薄后，PDL 的 min(tail,prologue) 上限还在不在。
//
// 链结构（每个 kernel 的 time 轴）与 pdl_pipeline.cu 完全一致：
//     [prologue 自旋 pl] -> gridDependencySynchronize -> [临界段 crit: 校验上游 seq, 写 seq]
//     -> triggerProgrammaticLaunchCompletion -> [tail 自旋 tl]
//   正确性：expect 为编译期烘进 kernel 的哨兵值；capture 后每次 replay 参数固定。
//     seq 初值 0；若 PDL 排序被破坏，下游会读到 0 != expect → fail 每轮递增；
//     若排序保证，下游恒读到 expect、fail 恒 0。故 fail==0 是「PDL 在 Graph 里仍生效」的硬证据。
//
// 编译：nvcc -O3 -arch=sm_90 pdl_graph.cu -o pdl_graph
// 用法：./pdl_graph [iters] [Kmax] [prologue] [crit] [tail] [tail2] [device]

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

// 忙等约 n 个 SM clock 周期（所有线程各自自旋）。
__device__ __forceinline__ void spin_cycles(long long n) {
  long long s = clock64();
  while (clock64() - s < n) {
  }
}

// 链上的第 idx 个 kernel。expect 为上游写入 seq 的哨兵值。
__global__ void chain_kernel(int* seq, int idx, int* fail, int expect,
                             long long prologue, long long crit, long long tail) {
  spin_cycles(prologue);
  cudaGridDependencySynchronize();  // 等到上游 kernel 放行（idx=0 时为 no-op）
  if (threadIdx.x == 0) {
    if (idx > 0 && seq[idx] != expect) atomicAdd(fail, 1);  // 校验上游数据可见
    spin_cycles(crit);
    seq[idx + 1] = expect;  // 把本 kernel 的「结果」交给下游
  }
  __syncthreads();
  cudaTriggerProgrammaticLaunchCompletion();  // 放行下游 prologue 与本 kernel tail 重叠
  spin_cycles(tail);
}

// 一个 launch 序列 = K 个 chain_kernel（k>0 时可选带 PDL 属性）。
static void launch_chain(cudaStream_t s, int K, bool use_pdl, int* seq, int* fail,
                         int expect, long long prologue, long long crit,
                         long long tail, const cudaLaunchAttribute* attrs) {
  for (int k = 0; k < K; ++k) {
    if (use_pdl && k > 0) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(1);
      cfg.blockDim = dim3(128);
      cfg.dynamicSmemBytes = 0;
      cfg.stream = s;
      cfg.attrs = const_cast<cudaLaunchAttribute*>(attrs);
      cfg.numAttrs = 1;
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, chain_kernel, seq, k, fail, expect,
                                    prologue, crit, tail));
    } else {
      chain_kernel<<<1, 128, 0, s>>>(seq, k, fail, expect, prologue, crit, tail);
    }
  }
}

// 建一张 K-node 图（capture 方式）。use_pdl 决定被 capture 的 launch 是否带 PDL 属性。
static cudaGraphExec_t build_graph(cudaStream_t s, int K, bool use_pdl, int* seq,
                                   int* fail, int expect, long long prologue,
                                   long long crit, long long tail,
                                   const cudaLaunchAttribute* attrs) {
  cudaGraph_t g;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  launch_chain(s, K, use_pdl, seq, fail, expect, prologue, crit, tail, attrs);
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  return ge;
}

// 计时：strategy 0=eager_nopdl 1=eager_pdl 2=graph_nopdl 3=graph_pdl。返回每轮平均 ms。
static float time_case(int strategy, int iters, int K, int* seq, int* fail,
                       int expect, long long prologue, long long crit,
                       long long tail, const cudaLaunchAttribute* attrs,
                       int* out_fail) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  cudaGraphExec_t ge = nullptr;
  if (strategy >= 2) {
    // capture 前把 seq 清零：若 PDL 排序坏，下游会读到 0。
    CUDA_CHECK(cudaMemset(seq, 0, sizeof(int) * (K + 2)));
    CUDA_CHECK(cudaMemset(fail, 0, sizeof(int)));
    ge = build_graph(s, K, strategy == 3, seq, fail, expect, prologue, crit,
                     tail, attrs);
  }

  // warmup + 计时。
  if (strategy >= 2) {
    CUDA_CHECK(cudaGraphLaunch(ge, s));
  } else {
    launch_chain(s, K, strategy == 1, seq, fail, expect, prologue, crit, tail,
                 attrs);
  }
  CUDA_CHECK(cudaStreamSynchronize(s));

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < iters; ++r) {
    if (strategy >= 2) {
      CUDA_CHECK(cudaGraphLaunch(ge, s));
    } else {
      launch_chain(s, K, strategy == 1, seq, fail, expect, prologue, crit, tail,
                   attrs);
    }
  }
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

  CUDA_CHECK(cudaMemcpy(out_fail, fail, sizeof(int), cudaMemcpyDeviceToHost));
  if (ge) CUDA_CHECK(cudaGraphExecDestroy(ge));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

static void run_case(const char* name, int iters, int Kmax, long long prologue,
                     long long crit, long long tail) {
  int* seq = nullptr;
  int* fail = nullptr;
  CUDA_CHECK(cudaMalloc(&seq, sizeof(int) * (Kmax + 2)));
  CUDA_CHECK(cudaMalloc(&fail, sizeof(int)));
  const int expect = 12345;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;

  printf("\n== %s ==  prologue=%lld crit=%lld tail=%lld (SM cycles)\n", name,
         prologue, crit, tail);
  long long ov = prologue < tail ? prologue : tail;

  printf("%4s | %10s | %10s | %10s | %10s | %7s | %7s | %7s | %s\n", "K",
         "eager ms", "ePDL ms", "graph ms", "gPDL ms", "e/ePDL", "gPDL/g",
         "e/gPDL", "fail");
  printf("-----+------------+------------+------------+------------+---------+"
         "---------+---------+------\n");

  for (int K = 1; K <= Kmax; K *= 2) {
    int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    CUDA_CHECK(cudaMemset(seq, 0, sizeof(int) * (Kmax + 2)));
    float t_e = time_case(0, iters, K, seq, fail, expect, prologue, crit, tail,
                          attrs, &f0);
    CUDA_CHECK(cudaMemset(seq, 0, sizeof(int) * (Kmax + 2)));
    float t_ep = time_case(1, iters, K, seq, fail, expect, prologue, crit, tail,
                           attrs, &f1);
    float t_g = time_case(2, iters, K, seq, fail, expect, prologue, crit, tail,
                          attrs, &f2);
    float t_gp = time_case(3, iters, K, seq, fail, expect, prologue, crit, tail,
                           attrs, &f3);
    printf("%4d | %10.4f | %10.4f | %10.4f | %10.4f | %6.2fx | %6.2fx | %6.2fx "
           "| %d/%d/%d/%d\n",
           K, t_e, t_ep, t_g, t_gp, t_e / t_ep, t_g / t_gp, t_e / t_gp, f0, f1,
           f2, f3);
  }
  printf("fail 列 = eager/ePDL/graph/gPDL 的原子计数之和（0 = 下游都读到上游哨兵 %d）\n",
         expect);
  printf("预测：eager 每 kernel = span；PDL 每条链边重叠 min(prologue,tail)=%lld，"
         "稳态每 kernel = span-%lld。\n",
         ov, ov);
  CUDA_CHECK(cudaFree(seq));
  CUDA_CHECK(cudaFree(fail));
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 30;
  int Kmax = (argc > 2) ? atoi(argv[2]) : 16;
  long long prologue = (argc > 3) ? atoll(argv[3]) : 200000;
  long long crit = (argc > 4) ? atoll(argv[4]) : 2000;
  long long tail = (argc > 5) ? atoll(argv[5]) : 200000;
  long long tail2 = (argc > 6) ? atoll(argv[6]) : 600000;
  int dev = (argc > 7) ? atoi(argv[7]) : 0;

  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("== device ==\n%s, sm_%d%d, SMs=%d (device %d)\n", prop.name,
         prop.major, prop.minor, prop.multiProcessorCount, dev);
  printf("iters=%d Kmax=%d\n", iters, Kmax);

  run_case("A: tail == prologue (overlap 完全可放行)", iters, Kmax, prologue,
           crit, tail);
  run_case("B: tail = 3x prologue (overlap 被 prologue 封顶)", iters, Kmax,
           prologue, crit, tail2);
  // C 段：把单 kernel 的段缩短到与 launch 开销同量级（span ~9k 周期 ≈ 4.6us，
  //        launch ~1.7us/kernel），才能同时看到 Graph 治损失二、PDL 治损失一。
  run_case("C: 小 span (launch 开销可感) tail == prologue", 100, 64, 4000, 1000,
           4000);

  printf("\n>> 读法：eager/ePDL=无 Graph；graph/gPDL=Graph replay。\n");
  printf(">> Q1 若 graph→gPDL 有加速且 gPDL 的 fail==0，则 CUDA Graph 保留了 PDL 语义；\n");
  printf(">> Q2 若 e/gPDL ≈ (e/graph)×(ePDL/e)，则 Graph 与 PDL 近似「相乘」互补；\n");
  printf(">> Q3 Graph 消的是损失二（launch），PDL 消的是损失一特例 min(tail,prologue)，二者上限独立。\n");
  return 0;
}
