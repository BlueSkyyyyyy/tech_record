// graph_conditional.cu — MegaKernel 专题 P36-1 小实验：
//   CUDA Graph 的**条件/循环节点**（conditional graph nodes：IF / WHILE）能否让**一张图**表达
//   「数据相关 / 可变行程」的控制流？——对照 MPK「每 batch size 一张静态图」的局限
//   （`docs/01` §5.8 局限 5；MPK 把「多轮服务」做成运行层的 `iteration` 计数，见 `docs/06` §4）。
//
// 背景（目标卡 H100 / sm90 实测）：
//   - P4-1（§1）/P19-1（§13）/P35-1（§24）已把 CUDA Graph 的「静态图 / PDL 边 / 显式构图」测遍，
//     结论一致：图是**静态 DAG**、节点拓扑固定。MPK 同样受此限（每 batch size 一张图）。
//   - 官方 CUDA Programming Guide §4.2.4「Conditional Graph Nodes」给出的 IF/WHILE/SWITCH 节点
//     （`driver_types.h:3559-3561` 的 `cudaGraphCondTypeIf/While/Switch`；`cudaGraphSetConditional`
//     设备函数 `cuda_device_runtime_api.h:480`；handle `cudaGraphConditionalHandleCreate`
//     `cuda_runtime_api.h:13362`）允许**在设备侧求值循环/分支条件**，且**body 图在创建节点时
//     由驱动分配**（`driver_types.h` 的 `cudaConditionalNodeParams.phGraph_out`，CUDA-owned array）。
//
// 本实验问三件事：
//   (A) 行程 N 可变的循环：`eager`（host 循环 N 次 launch）/ `unroll`（N 个串联 kernel 节点的静态图，
//       每个 N 需重新 instantiate）/ `cwhile`（**一个 WHILE 条件节点**，body 一个 kernel；**同一个
//       graphExec 服务任意 N**）/ `kinternal`（**单 kernel 内 for 循环 N 次** = 持久化内核微缩版）。
//       比较每次调用的端到端延迟与「每行程」摊薄成本，并核对数值。
//   (B) 控制开销与实例化成本：work=0（每行程只做一次条件求值 + 计数）扫 N，量 cwhile 的**每行程
//       设备内条件求值**成本 vs unroll 的**每节点**成本 vs eager 的**每次 launch**；并量
//       `instantiate` 时间随 N 的标度（unroll 线性、cwhile 常数）。
//   (C) 数据相关分支：IF 条件节点的条件由**上游 kernel 从数据算出**，同一张图对两个数据 case
//       走不同 body；证明「静态图 + 设备侧条件」可覆盖 MPK 需要按数据改图的那类分支。
//
// 所有策略的算子表达式与累加顺序一致 → `acc[]` 应逐位一致（max|Δ|=0）作为正确性交叉校验。
//
// 编译：nvcc -O3 -arch=sm_90 graph_conditional.cu -o graph_conditional
// 用法：./graph_conditional [iters] [Nmax] [G] [T] [work]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
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

// ---- 迭代体 kernel：每 block 每线程对 acc[i] 做 work 轮 FMA；thread0 计数并（可选）设条件。 ----
__global__ void iter_kernel(unsigned long long handle, int use_cond, int* iterc,
                            const int* N, float* acc, int work) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = acc[i];
  for (int w = 0; w < work; ++w) a = fmaf(a, 1.0000001f, 0.5f);
  acc[i] = a;
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int v = ++(*iterc);
    if (use_cond) cudaGraphSetConditional(handle, (v < *N) ? 1u : 0u);
  }
}

// ---- 复位 kernel：每次图 launch 开始时把行程计数清零（条件循环需要）----
__global__ void reset_kernel(int* iterc) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *iterc = 0;
}

// ---- 单 kernel 内 for 循环 N 次（持久化内核微缩版：所有行程复用同一批 block）----
__global__ void internal_loop_kernel(float* acc, int work, int N, int* iterc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = acc[i];
  for (int it = 0; it < N; ++it)
    for (int w = 0; w < work; ++w) a = fmaf(a, 1.0000001f, 0.5f);
  acc[i] = a;
  if (threadIdx.x == 0 && blockIdx.x == 0) *iterc = N;
}

// ---- 上游谓词 kernel（IF 用）：从数据算出条件并写 handle ----
__global__ void pred_kernel(unsigned long long handle, const float* x, int thr) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    cudaGraphSetConditional(handle, (x[0] > (float)thr) ? 1u : 0u);
}

// ---- 分支标记 kernel：body 被执行时置 sentinel ----
__global__ void mark_kernel(int* flag, int v) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *flag = v;
}

// ============================ 图构造 ============================

// 静态展开图：N 个 iter_kernel 串联。
static cudaGraphExec_t build_unroll(int G, int T, int N, int* iterc,
                                    float* acc, int work) {
  cudaGraph_t gr;
  CUDA_CHECK(cudaGraphCreate(&gr, 0));
  int use_cond = 0;
  const int* Nnull = nullptr;
  unsigned long long hdummy = 0;
  void* args[] = {&hdummy, &use_cond, &iterc, &Nnull, &acc, &work};
  cudaGraphNode_t prev = nullptr;
  for (int k = 0; k < N; ++k) {
    cudaGraphNodeParams np = {};
    np.type = cudaGraphNodeTypeKernel;
    np.kernel.func = (void*)iter_kernel;
    np.kernel.gridDim = dim3(G);
    np.kernel.blockDim = dim3(T);
    np.kernel.kernelParams = args;
    cudaGraphNode_t nd;
    CUDA_CHECK(cudaGraphAddNode(&nd, gr, prev ? &prev : nullptr, nullptr,
                                prev ? 1 : 0, &np));
    prev = nd;
  }
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  CUDA_CHECK(cudaGraphDestroy(gr));
  return ge;
}

// 条件循环图：reset -> WHILE 节点，body 一个 iter_kernel（use_cond=1）。同一 exec 服务任意 N。
static cudaGraphExec_t build_while(int G, int T, int* iterc, const int* NdevValue,
                                   float* acc, int work,
                                   cudaGraphConditionalHandle* handle_out) {
  // kernel 参数需要在该 graphExec 存活期间一直有效；用函数静态存储保存设备端 N 指针。
  static const int* s_Ndev = nullptr;
  s_Ndev = NdevValue;
  cudaGraph_t gr;
  CUDA_CHECK(cudaGraphCreate(&gr, 0));
  cudaGraphConditionalHandle h;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &h, gr, /*defaultLaunchValue=*/1, cudaGraphCondAssignDefault));
  *handle_out = h;

  // reset 节点
  cudaGraphNodeParams rp = {};
  rp.type = cudaGraphNodeTypeKernel;
  rp.kernel.func = (void*)reset_kernel;
  rp.kernel.gridDim = dim3(1);
  rp.kernel.blockDim = dim3(1);
  void* rargs[] = {&iterc};
  rp.kernel.kernelParams = rargs;
  cudaGraphNode_t resetNode;
  CUDA_CHECK(cudaGraphAddNode(&resetNode, gr, nullptr, nullptr, 0, &rp));

  // WHILE 条件节点（依赖 reset）
  cudaGraphNodeParams cp = {};
  cp.type = cudaGraphNodeTypeConditional;
  cp.conditional.handle = h;
  cp.conditional.type = cudaGraphCondTypeWhile;
  cp.conditional.size = 1;
  cp.conditional.phGraph_out = nullptr;  // 驱动分配 CUDA-owned 数组
  cudaGraphNode_t condNode;
  CUDA_CHECK(cudaGraphAddNode(&condNode, gr, &resetNode, nullptr, 1, &cp));
  cudaGraph_t body = cp.conditional.phGraph_out[0];  // 读回 body 图

  int use_cond = 1;
  void* args[] = {&h, &use_cond, &iterc, &s_Ndev, &acc, &work};
  cudaGraphNodeParams kp = {};
  kp.type = cudaGraphNodeTypeKernel;
  kp.kernel.func = (void*)iter_kernel;
  kp.kernel.gridDim = dim3(G);
  kp.kernel.blockDim = dim3(T);
  kp.kernel.kernelParams = args;
  cudaGraphNode_t bodyKernel;
  CUDA_CHECK(cudaGraphAddNode(&bodyKernel, body, nullptr, nullptr, 0, &kp));

  cudaGraphExec_t ge;
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  CUDA_CHECK(cudaGraphDestroy(gr));
  return ge;
}

// IF 图：pred_kernel -> IF 节点（body = mark_kernel）。predicate 由数据算出。
static cudaGraphExec_t build_if(int* flag, const float* x, int thr,
                                cudaGraphConditionalHandle* handle_out) {
  cudaGraph_t gr;
  CUDA_CHECK(cudaGraphCreate(&gr, 0));
  cudaGraphConditionalHandle h;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, gr, 0, 0));
  *handle_out = h;

  cudaGraphNodeParams pp = {};
  pp.type = cudaGraphNodeTypeKernel;
  pp.kernel.func = (void*)pred_kernel;
  pp.kernel.gridDim = dim3(1);
  pp.kernel.blockDim = dim3(1);
  void* pargs[] = {&h, &x, &thr};
  pp.kernel.kernelParams = pargs;
  cudaGraphNode_t predNode;
  CUDA_CHECK(cudaGraphAddNode(&predNode, gr, nullptr, nullptr, 0, &pp));

  cudaGraphNodeParams cp = {};
  cp.type = cudaGraphNodeTypeConditional;
  cp.conditional.handle = h;
  cp.conditional.type = cudaGraphCondTypeIf;
  cp.conditional.size = 1;
  cp.conditional.phGraph_out = nullptr;
  cudaGraphNode_t condNode;
  CUDA_CHECK(cudaGraphAddNode(&condNode, gr, &predNode, nullptr, 1, &cp));
  cudaGraph_t body = cp.conditional.phGraph_out[0];

  int v = 12345;
  cudaGraphNodeParams kp = {};
  kp.type = cudaGraphNodeTypeKernel;
  kp.kernel.func = (void*)mark_kernel;
  kp.kernel.gridDim = dim3(1);
  kp.kernel.blockDim = dim3(1);
  void* kargs[] = {&flag, &v};
  kp.kernel.kernelParams = kargs;
  cudaGraphNode_t bodyKernel;
  CUDA_CHECK(cudaGraphAddNode(&bodyKernel, body, nullptr, nullptr, 0, &kp));

  cudaGraphExec_t ge;
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  CUDA_CHECK(cudaGraphDestroy(gr));
  return ge;
}

// ============================ 计时 / 分发 ============================

struct Ctx {
  cudaStream_t s;
  cudaGraphExec_t ge;  // 非空 -> launch 图
  int eagerN;          // >0 -> eager 循环 launch kernel
  int G, T, work;
  int Nhost;           // 主机端 N（kinternal 用）
  int* iterc;
  int* N;              // 设备端 N（eager 用，仅作 kernel 参数占位）
  float* acc;
};

static void launch_dispatch(void* p) {
  Ctx* c = (Ctx*)p;
  if (c->ge) {
    CUDA_CHECK(cudaGraphLaunch(c->ge, c->s));
  } else if (c->eagerN > 0) {
    for (int k = 0; k < c->eagerN; ++k)
      iter_kernel<<<c->G, c->T, 0, c->s>>>(0, 0, c->iterc, nullptr, c->acc,
                                           c->work);
  } else {
    internal_loop_kernel<<<c->G, c->T, 0, c->s>>>(c->acc, c->work, c->Nhost,
                                                  c->iterc);
  }
}

static double time_calls(cudaStream_t s, int K, void (*launch)(void*), void* ctx) {
  for (int w = 0; w < 5; ++w) launch(ctx);
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < K; ++r) launch(ctx);
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return (double)ms * 1000.0 / K;  // µs / call
}

static double host_us(std::chrono::steady_clock::time_point t0,
                      std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 50;
  int Nmax = argc > 2 ? atoi(argv[2]) : 64;
  int G = argc > 3 ? atoi(argv[3]) : 32;
  int T = argc > 4 ? atoi(argv[4]) : 256;
  int work = argc > 5 ? atoi(argv[5]) : 64;

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount);
  printf("iters=%d  Nmax=%d  G=%d  T=%d  work=%d (FMA/线程/行程)\n", iters, Nmax,
         G, T, work);
  printf("打点：每策略同一 stream 上 K=%d 次调用的 CUDA event 端到端差，报 µs/调用。\n",
         iters);

  int M = G * T;
  float* acc;
  int* iterc;
  int* Ndev;
  CUDA_CHECK(cudaMalloc(&acc, sizeof(float) * M));
  CUDA_CHECK(cudaMalloc(&iterc, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ndev, sizeof(int)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  float* h = (float*)malloc(sizeof(float) * M);
  for (int i = 0; i < M; ++i) h[i] = 1.0f;
  double ref_sum = 0;
  {  // 参考：eager N=64, work 之后的 acc
    CUDA_CHECK(cudaMemcpy(acc, h, sizeof(float) * M, cudaMemcpyHostToDevice));
    for (int k = 0; k < 64; ++k)
      iter_kernel<<<G, T, 0, s>>>(0, 0, iterc, nullptr, acc, work);
    CUDA_CHECK(cudaStreamSynchronize(s));
    float* hr = (float*)malloc(sizeof(float) * M);
    CUDA_CHECK(cudaMemcpy(hr, acc, sizeof(float) * M, cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; ++i) ref_sum += hr[i];
    free(hr);
  }
  printf("参考（eager N=64, work=%d）：acc sum=%.6f\n\n", work, ref_sum);

  // ---------------- Section A：N 扫描，四策略 ----------------
  printf("=== Section A: 行程 N 扫描（work=%d，G=%d，T=%d）===\n", work, G, T);
  printf("策略：eager=N 次 host launch | unroll=N 节点静态图 | cwhile=1 个 WHILE 节点（同一 exec）| kinternal=单 kernel 内循环\n");
  printf("%6s | %14s | %14s | %14s | %14s | %12s\n", "N", "eager(us)",
         "unroll(us)", "cwhile(us)", "kinternal(us)", "cwhile-it/acc");
  printf("-------+----------------+----------------+----------------+----------------+--------------\n");
  for (int N = 1; N <= Nmax; N *= 2) {
    CUDA_CHECK(cudaMemcpy(Ndev, &N, sizeof(int), cudaMemcpyHostToDevice));

    Ctx ce = {s, nullptr, N, G, T, work, N, iterc, Ndev, acc};
    double t_eager = time_calls(s, iters, launch_dispatch, &ce);

    cudaGraphExec_t gu = build_unroll(G, T, N, iterc, acc, work);
    Ctx cu = {s, gu, 0, G, T, work, N, iterc, Ndev, acc};
    double t_unroll = time_calls(s, iters, launch_dispatch, &cu);
    CUDA_CHECK(cudaGraphExecDestroy(gu));

    cudaGraphConditionalHandle hw;
    cudaGraphExec_t gw = build_while(G, T, iterc, Ndev, acc, work, &hw);
    Ctx cw = {s, gw, 0, G, T, work, N, iterc, Ndev, acc};
    double t_cwhile = time_calls(s, iters, launch_dispatch, &cw);

    Ctx ck = {s, nullptr, 0, G, T, work, N, iterc, Ndev, acc};
    double t_kint = time_calls(s, iters, launch_dispatch, &ck);

    // 正确性：干净跑一次 cwhile 到 N，读回行程与 acc sum
    CUDA_CHECK(cudaMemcpy(acc, h, sizeof(float) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGraphLaunch(gw, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    int gotN = -1;
    CUDA_CHECK(cudaMemcpy(&gotN, iterc, sizeof(int), cudaMemcpyDeviceToHost));
    float* hr = (float*)malloc(sizeof(float) * M);
    CUDA_CHECK(cudaMemcpy(hr, acc, sizeof(float) * M, cudaMemcpyDeviceToHost));
    double ssum = 0;
    for (int i = 0; i < M; ++i) ssum += hr[i];
    free(hr);
    CUDA_CHECK(cudaGraphExecDestroy(gw));

    printf("%6d | %14.3f | %14.3f | %14.3f | %14.3f | %d/%.4f\n", N, t_eager,
           t_unroll, t_cwhile, t_kint, gotN, ssum);
  }

  // ---------------- Section B：控制开销（work=0）与实例化成本 ----------------
  printf("\n=== Section B: 控制开销（work=0，纯条件求值/计数/调度）===\n");
  printf("%6s | %14s | %14s | %14s | %14s\n", "N", "eager(us)", "unroll(us)",
         "cwhile(us)", "kinternal(us)");
  printf("-------+----------------+----------------+----------------+----------------\n");
  for (int N = 1; N <= Nmax; N *= 2) {
    CUDA_CHECK(cudaMemcpy(Ndev, &N, sizeof(int), cudaMemcpyHostToDevice));
    Ctx ce = {s, nullptr, N, G, T, 0, N, iterc, Ndev, acc};
    double t_e = time_calls(s, iters, launch_dispatch, &ce);
    cudaGraphExec_t gu = build_unroll(G, T, N, iterc, acc, 0);
    Ctx cu = {s, gu, 0, G, T, 0, N, iterc, Ndev, acc};
    double t_u = time_calls(s, iters, launch_dispatch, &cu);
    cudaGraphExecDestroy(gu);
    cudaGraphConditionalHandle hw;
    cudaGraphExec_t gw = build_while(G, T, iterc, Ndev, acc, 0, &hw);
    Ctx cw = {s, gw, 0, G, T, 0, N, iterc, Ndev, acc};
    double t_w = time_calls(s, iters, launch_dispatch, &cw);
    cudaGraphExecDestroy(gw);
    Ctx ck = {s, nullptr, 0, G, T, 0, N, iterc, Ndev, acc};
    double t_k = time_calls(s, iters, launch_dispatch, &ck);
    printf("%6d | %14.3f | %14.3f | %14.3f | %14.3f\n", N, t_e, t_u, t_w, t_k);
  }

  printf("\n--- 实例化成本（host 时间）：unroll(N 节点) vs cwhile(2 节点) ---\n");
  printf("%6s | %14s | %14s\n", "N", "unroll(us)", "cwhile(us)");
  printf("-------+----------------+----------------\n");
  for (int N = 1; N <= Nmax; N *= 2) {
    cudaGraph_t gr;
    CUDA_CHECK(cudaGraphCreate(&gr, 0));
    int use_cond = 0;
    const int* Nnull = nullptr;
    unsigned long long hdummy = 0;
  void* args[] = {&hdummy, &use_cond, &iterc, &Nnull, &acc, &work};
    cudaGraphNode_t prev = nullptr;
    for (int k = 0; k < N; ++k) {
      cudaGraphNodeParams np = {};
      np.type = cudaGraphNodeTypeKernel;
      np.kernel.func = (void*)iter_kernel;
      np.kernel.gridDim = dim3(G);
      np.kernel.blockDim = dim3(T);
      np.kernel.kernelParams = args;
      cudaGraphNode_t nd;
      CUDA_CHECK(cudaGraphAddNode(&nd, gr, prev ? &prev : nullptr, nullptr,
                                  prev ? 1 : 0, &np));
      prev = nd;
    }
    auto t0 = std::chrono::steady_clock::now();
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
    auto t1 = std::chrono::steady_clock::now();
    double unroll_us = host_us(t0, t1);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(gr);

    cudaGraphConditionalHandle hw;
    auto t2 = std::chrono::steady_clock::now();
    cudaGraphExec_t gw = build_while(G, T, iterc, Ndev, acc, work, &hw);
    auto t3 = std::chrono::steady_clock::now();
    double cwhile_us = host_us(t2, t3);
    cudaGraphExecDestroy(gw);
    printf("%6d | %14.1f | %14.1f\n", N, unroll_us, cwhile_us);
  }

  // ---------------- Section C：数据相关分支（IF 条件节点） ----------------
  printf("\n=== Section C: 数据相关分支（IF 节点，条件由上游 kernel 从数据算出）===\n");
  printf("pred_kernel 读 x[0]，>thr 则写 handle=1 触发 body(mark=12345)；否则条件 0、body 不执行。\n");
  printf("同一张图（同一 graphExec）对两个数据 case 各 launch 一次。\n");
  {
    float* x;
    int* flag;
    CUDA_CHECK(cudaMalloc(&x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flag, sizeof(int)));
    cudaGraphConditionalHandle hif;
    cudaGraphExec_t gi = build_if(flag, x, /*thr=*/10, &hif);
    for (int case_i = 0; case_i < 2; ++case_i) {
      float xv = case_i == 0 ? 100.0f : 3.0f;
      CUDA_CHECK(cudaMemcpy(x, &xv, sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(flag, 0, sizeof(int)));
      CUDA_CHECK(cudaGraphLaunch(gi, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      int fv = -1;
      CUDA_CHECK(cudaMemcpy(&fv, flag, sizeof(int), cudaMemcpyDeviceToHost));
      printf("  case%d x[0]=%.1f -> flag=%d (%s)\n", case_i, xv, fv,
             fv == 12345 ? "body 执行" : "body 跳过(条件=0)");
    }
    cudaGraphExecDestroy(gi);
    cudaFree(x);
    cudaFree(flag);
  }

  // ---------------- Section D：正确性交叉校验（四策略 acc 逐位一致） ----------------
  printf("\n=== Section D: 正确性交叉校验（N=64，四策略对 acc 逐位对比）===\n");
  {
    int Nc = 64;
    CUDA_CHECK(cudaMemcpy(Ndev, &Nc, sizeof(int), cudaMemcpyHostToDevice));
    float* base = (float*)malloc(sizeof(float) * M);
    float* cur = (float*)malloc(sizeof(float) * M);
    // eager 基准
    CUDA_CHECK(cudaMemcpy(acc, h, sizeof(float) * M, cudaMemcpyHostToDevice));
    for (int k = 0; k < Nc; ++k)
      iter_kernel<<<G, T, 0, s>>>(0, 0, iterc, nullptr, acc, work);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(base, acc, sizeof(float) * M, cudaMemcpyDeviceToHost));

    struct { const char* name; int kind; } cfgs[] = {
        {"eager", 0}, {"unroll", 1}, {"cwhile", 2}, {"kinternal", 3}};
    for (auto& cf : cfgs) {
      CUDA_CHECK(cudaMemcpy(acc, h, sizeof(float) * M, cudaMemcpyHostToDevice));
      if (cf.kind == 0) {
        for (int k = 0; k < Nc; ++k)
          iter_kernel<<<G, T, 0, s>>>(0, 0, iterc, nullptr, acc, work);
        CUDA_CHECK(cudaStreamSynchronize(s));
      } else if (cf.kind == 1) {
        cudaGraphExec_t gu = build_unroll(G, T, Nc, iterc, acc, work);
        CUDA_CHECK(cudaGraphLaunch(gu, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        cudaGraphExecDestroy(gu);
      } else if (cf.kind == 2) {
        cudaGraphConditionalHandle hw;
        cudaGraphExec_t gw = build_while(G, T, iterc, Ndev, acc, work, &hw);
        CUDA_CHECK(cudaGraphLaunch(gw, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        cudaGraphExecDestroy(gw);
      } else {
        internal_loop_kernel<<<G, T, 0, s>>>(acc, work, Nc, iterc);
        CUDA_CHECK(cudaStreamSynchronize(s));
      }
      CUDA_CHECK(cudaMemcpy(cur, acc, sizeof(float) * M, cudaMemcpyDeviceToHost));
      double maxd = 0, maxrel = 0;
      for (int i = 0; i < M; ++i) {
        double d = cur[i] - base[i];
        if (d < 0) d = -d;
        if (d > maxd) maxd = d;
        double r = base[i] != 0 ? d / (base[i] < 0 ? -base[i] : base[i]) : d;
        if (r > maxrel) maxrel = r;
      }
      printf("  %-10s max|Δ|=%.3e  max rel|Δ|=%.3e\n", cf.name, maxd, maxrel);
    }
    free(base);
    free(cur);
  }

  printf("\n>> 读法：\n");
  printf(">> A: 行程 N 变大时 unroll/cwhile/kinternal 相对 eager 省的是 **launch**；\n");
  printf(">>    cwhile 每行程多一次**设备内条件求值 + body 图重新分发**；\n");
  printf(">>    kinternal（持久化内核）是下限——同一批 block 内循环，无图/无 node 分发。\n");
  printf(">> B: work=0 时每行程成本 = 纯控制；instantiate 时间 unroll 随 N 线性、cwhile 常数。\n");
  printf(">> C: 同一张静态图可由**设备侧数据**决定分支——CUDA Graph 对 MPK\n");
  printf(">>    「按数据/batch 改图」局限的一条（受限）出路。\n");

  cudaFree(acc);
  cudaFree(iterc);
  cudaFree(Ndev);
  cudaStreamDestroy(s);
  free(h);
  return 0;
}
