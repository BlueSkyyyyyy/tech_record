// graph_programmatic.cu — MegaKernel 专题 P35-1 小实验：
//   显式构图（cudaGraphAddKernelNode + cudaGraphAddDependencies 的 PROGRAMMATIC 边）能否把
//   下游 kernel 的放行点前移到「上游完成之前」？与 stream 上的 PDL / capture 图对照。
//
// 背景（目标卡 H100 / sm90 实测）：
//   - P19-1（`src/pdl_graph.cu`）证明 **capture 方式**构造的图完整保留 PDL 语义（gPDL/g 逐行等于
//     e/ePDL），但 §13.6 明留「未测**显式构图** API（`cudaGraphAddKernelNode` +
//     `cudaGraphKernelNodeSetAttribute` 的 PDL 属性）以及 graph 内部的 programmatic edge」。
//   - P32-1（`src/programmatic_event.cu`）测出：**stream 上** `cudaLaunchAttributeProgrammaticEvent`
//     的提前 trigger **不放行** cross-stream `cudaStreamWaitEvent` 依赖（只在完成点放行），
//     唯一能提前放行的是 LaunchCompletionEvent；并留 §21.6「未测 CUDA Graph 的 PROGRAMMATIC 边」
//     （`CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC`，`cuda.h:2055–2065` / `driver_types.h:3708–3761`）。
//   本节把这两条尾巴一起补上：**图的 programmatic 边把「进度」这一概念真正做进了图**——
//   `cudaGraphEdgeData.from_port` = `cudaGraphKernelNodePortProgrammatic`（上游 programmatic 端口）
//   或 `cudaGraphKernelNodePortLaunchCompletion`（全体 block 启动端口），`type` = Programmatic，
//   下游 kernel 用 `cudaGridDependencySynchronize()` 等该端口。
//
// 模型（一个图，两个 kernel 节点 A -> B；`globaltimer` 打点，ns）：
//   A（上游，G 个 block）：开始打 g[0] -> prologue(+i*stagger) -> [可选 cudaTriggerProgrammatic-
//                          LaunchCompletion] -> tail -> 结束打 g[1]
//   B（下游，1 个 block）：入口打 g[2] -> prologue -> cudaGridDependencySynchronize -> 打 g[3]
//                          -> crit -> atomicAdd(g[4])（正确性哨兵）
//   观测量：A_total = g[1]-g[0]；sched = g[2]-g[0]（B 何时被**调度**拿到 SM）；
//          rel = g[3]-g[0]（B 的 gridDependencySync 何时**返回**，恒等上游完成）。
//          done = g[4]（应 == iters）。关键量是 sched。
//
// 边档（同一 A/B workload 公平对照）：
//   block      : 普通边（from_port=Default, type=Default）—— 完成即放行（地板）
//   prog_nt    : programmatic 边（from_port=Programmatic），A **不** trigger —— 端口在终止时激活
//   prog       : programmatic 边，A 在 prologue 后 trigger —— 端口在上游 tail 之前激活
//   lce        : programmatic 边（from_port=LaunchCompletion）—— 全体 block 启动即激活
//
// 编译：nvcc -O3 -arch=sm_90 graph_programmatic.cu -o graph_programmatic
// 用法：./graph_programmatic [iters] [Kmax] [pro_cycles] [tail_cycles] [stagger] [device]

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

// 全局一致的纳秒时钟（跨 SM 可比）。clock64 是 per-SM，不适合跨 block 比时间。
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void spin_cycles(long long n) {
  long long s = clock64();
  while (clock64() - s < n) {
  }
}

// 上游 A：prologue(+i*stagger) -> [可选 trigger] -> tail。
__global__ void up_kernel(long long* g, long long pro, long long tail,
                          int do_trigger, long long stagger) {
  long long mypro = pro + (long long)blockIdx.x * stagger;
  if (threadIdx.x == 0 && blockIdx.x == 0) g[0] = gtimer();
  spin_cycles(mypro);
  if (do_trigger) {
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
  }
  spin_cycles(tail);
  if (threadIdx.x == 0 && blockIdx.x == 0) g[1] = gtimer();
}

// 下游 B：入口打点 -> prologue -> gridDependencySync -> 放行打点 -> crit -> 哨兵。
__global__ void down_kernel(long long* g, long long prol, long long crit) {
  if (threadIdx.x == 0) g[2] = gtimer();
  spin_cycles(prol);
  cudaGridDependencySynchronize();
  if (threadIdx.x == 0) g[3] = gtimer();
  spin_cycles(crit);
  if (threadIdx.x == 0) atomicAdd((int*)&g[4], 1);
}

// 边档。
enum { E_BLOCK = 0, E_PROG_NT, E_PROG, E_LCE };

static const char* edge_name(int e) {
  switch (e) {
    case E_BLOCK: return "block";
    case E_PROG_NT: return "prog_nt";
    case E_PROG: return "prog";
    case E_LCE: return "lce";
  }
  return "?";
}

struct Result {
  double a_total_ns;
  double sched_ns;
  double rel_ns;
  int done;
};

// 构造显式图：A -> B，边按 edge 档。capture=kCapture 用 stream capture 构造（对照）。
static cudaGraphExec_t build_explicit(long long* g, int G, int edge,
                                      long long pro, long long tail,
                                      long long stagger, long long prol,
                                      long long crit) {
  cudaGraph_t gr;
  CUDA_CHECK(cudaGraphCreate(&gr, 0));

  int do_trigger = (edge == E_PROG) ? 1 : 0;

  // ---- 节点 A ----
  cudaGraphNode_t nA;
  long long proL = pro, tailL = tail, stagL = stagger;
  int trigL = do_trigger;
  void* argsA[] = {&g, &proL, &tailL, &trigL, &stagL};
  cudaKernelNodeParams pa = {};
  pa.func = (void*)up_kernel;
  pa.gridDim = dim3(G);
  pa.blockDim = dim3(128);
  pa.sharedMemBytes = 0;
  pa.kernelParams = argsA;
  CUDA_CHECK(cudaGraphAddKernelNode(&nA, gr, nullptr, 0, &pa));

  // ---- 节点 B ----
  cudaGraphNode_t nB;
  long long prolL = prol, critL = crit;
  void* argsB[] = {&g, &prolL, &critL};
  cudaKernelNodeParams pb = {};
  pb.func = (void*)down_kernel;
  pb.gridDim = dim3(1);
  pb.blockDim = dim3(128);
  pb.sharedMemBytes = 0;
  pb.kernelParams = argsB;
  CUDA_CHECK(cudaGraphAddKernelNode(&nB, gr, nullptr, 0, &pb));

  // PDL 序列化属性是 launch 级（`driver_types.h:3986` "Valid for launches"，**不在** graph node
  // 可用列表里）；对显式图，尝试把它设到节点上，看驱动是否接受。预期被拒 → 端口只由 programmatic
  // 边本身提供。只打印一次，避免刷屏。
  static bool attr_warned = false;
  cudaKernelNodeAttrValue v = {};
  v.programmaticStreamSerializationAllowed = 1;
  cudaError_t ae = cudaGraphKernelNodeSetAttribute(
      nB, cudaLaunchAttributeProgrammaticStreamSerialization, &v);
  if (ae != cudaSuccess) {
    if (!attr_warned) {
      printf("  [note] 节点级 PDL 序列化属性被驱动拒绝（%s）——端口只由 programmatic 边提供\n",
             cudaGetErrorString(ae));
      attr_warned = true;
    }
    cudaGetLastError();
  }

  // ---- 边 ----
  cudaGraphEdgeData ed = {};
  if (edge == E_BLOCK) {
    ed.from_port = cudaGraphKernelNodePortDefault;
    ed.type = cudaGraphDependencyTypeDefault;
  } else if (edge == E_LCE) {
    ed.from_port = cudaGraphKernelNodePortLaunchCompletion;
    ed.type = cudaGraphDependencyTypeProgrammatic;
  } else {  // PROG / PROG_NT
    ed.from_port = cudaGraphKernelNodePortProgrammatic;
    ed.type = cudaGraphDependencyTypeProgrammatic;
  }
  ed.to_port = 0;
  CUDA_CHECK(cudaGraphAddDependencies(gr, &nA, &nB, &ed, 1));

  cudaGraphExec_t ge;
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(gr));
  return ge;
}

// capture 构造（对照）：A 带 ProgrammaticEvent 事件属性 + B 带 PDL 序列化属性，串在同一条 stream。
static cudaGraphExec_t build_capture(long long* g, int G, long long pro,
                                     long long tail, long long stagger,
                                     long long prol, long long crit) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  cudaEvent_t evX;
  CUDA_CHECK(cudaEventCreateWithFlags(&evX, cudaEventDisableTiming));

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));

  cudaLaunchConfig_t ca = {};
  ca.gridDim = dim3(G);
  ca.blockDim = dim3(128);
  ca.stream = s;
  cudaLaunchAttribute aa[1] = {};
  aa[0].id = cudaLaunchAttributeProgrammaticEvent;
  aa[0].val.programmaticEvent.event = evX;
  aa[0].val.programmaticEvent.flags = 0;
  aa[0].val.programmaticEvent.triggerAtBlockStart = 0;
  ca.attrs = aa;
  ca.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&ca, up_kernel, g, pro, tail, 1, stagger));

  cudaLaunchConfig_t cb = {};
  cb.gridDim = dim3(1);
  cb.blockDim = dim3(128);
  cb.stream = s;
  cudaLaunchAttribute ab[1] = {};
  ab[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  ab[0].val.programmaticStreamSerializationAllowed = 1;
  cb.attrs = ab;
  cb.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cb, down_kernel, g, prol, crit));

  cudaGraph_t gr;
  CUDA_CHECK(cudaStreamEndCapture(s, &gr));
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(gr));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaEventDestroy(evX));
  return ge;
}

// 跑一个图；返回 g[] 打点（取最后一轮）。done 累加校验。
static Result run_graph(cudaGraphExec_t ge, long long* g, int iters) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  CUDA_CHECK(cudaMemset(g, 0, sizeof(long long) * 8));
  for (int r = 0; r < iters; ++r) CUDA_CHECK(cudaGraphLaunch(ge, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  long long h[8] = {0};
  CUDA_CHECK(cudaMemcpy(h, g, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaStreamDestroy(s));

  Result r;
  r.a_total_ns = (double)(h[1] - h[0]);
  r.sched_ns = (double)(h[2] - h[0]);
  r.rel_ns = (double)(h[3] - h[0]);
  r.done = (int)h[4];
  return r;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 30;
  int Kmax = (argc > 2) ? atoi(argv[2]) : 32;
  long long pro = (argc > 3) ? atoll(argv[3]) : 100000;
  long long tail = (argc > 4) ? atoll(argv[4]) : 100000;
  long long stagger = (argc > 5) ? atoll(argv[5]) : 20000;
  int dev = (argc > 6) ? atoi(argv[6]) : 0;

  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int clk_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  if (clk_khz <= 0) clk_khz = 1980000;
  double cyc_us = 1000.0 / (double)clk_khz;  // SM cycles -> us
  printf("== device ==\n%s, sm_%d%d, SMs=%d (device %d), clock=%.3f GHz\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount, dev,
         clk_khz / 1e6);
  printf("iters=%d  pro=%lld  tail=%lld  stagger=%lld cycles  (A_total_proj=%.1f us)\n",
         iters, pro, tail, stagger, (pro + tail) * cyc_us);
  printf("打点用 %%globaltimer（全局一致 ns）；sched=B 被调度时刻；rel=B 的 "
         "cudaGridDependencySynchronize 返回时刻（=放行点）。\n");

  long long* g = nullptr;
  CUDA_CHECK(cudaMalloc(&g, sizeof(long long) * 8));

  // ---- Section A：G=1，四档边（同一 A/B workload） ----
  printf("\n=== Section A: G=1，显式构图，四档边 ===\n");
  printf("A = pro(%lld) -> [trigger?] -> tail(%lld)；B = prologue(%lld) -> sync -> crit(%lld)\n",
         pro, tail, tail, tail);
  printf("%8s | %10s | %10s | %10s | %8s | %8s\n", "edge", "A_total(us)",
         "sched(us)", "rel(us)", "rel/A", "done");
  printf("---------+------------+------------+------------+----------+---------\n");
  {
    const int edges[4] = {E_BLOCK, E_PROG_NT, E_PROG, E_LCE};
    for (int i = 0; i < 4; ++i) {
      cudaGraphExec_t ge =
          build_explicit(g, 1, edges[i], pro, tail, 0, tail, tail);
      Result r = run_graph(ge, g, iters);
      printf("%8s | %10.2f | %10.2f | %10.2f | %8.3f | %8d\n",
             edge_name(edges[i]), r.a_total_ns / 1000.0, r.sched_ns / 1000.0,
             r.rel_ns / 1000.0, r.rel_ns / r.a_total_ns, r.done);
      CUDA_CHECK(cudaGraphExecDestroy(ge));
    }
  }

  // ---- Section B：多 block + stagger，放行点随 G 的标度 ----
  printf("\n=== Section B: 多 block（G，block i prologue += i*stagger），prog vs lce vs block ===\n");
  printf("prog：端口在**所有 block trigger** 时激活；lce：所有 block 启动即激活。\n");
  printf("%5s | %12s | %10s | %10s | %10s | %10s | %10s\n", "G",
         "(G-1)*stag(us)", "block A(us)", "prog_sched", "prog_rel",
         "lce_rel", "prog_done");
  printf("------+--------------+------------+------------+------------+------------+-----------\n");
  for (int G = 1; G <= Kmax; G *= 2) {
    cudaGraphExec_t gb = build_explicit(g, G, E_BLOCK, pro, tail, stagger,
                                        tail, tail);
    Result rb = run_graph(gb, g, iters);
    cudaGraphExec_t gp = build_explicit(g, G, E_PROG, pro, tail, stagger,
                                        tail, tail);
    Result rp = run_graph(gp, g, iters);
    cudaGraphExec_t gl = build_explicit(g, G, E_LCE, pro, tail, stagger,
                                        tail, tail);
    Result rl = run_graph(gl, g, iters);
    printf("%5d | %12.2f | %10.2f | %10.2f | %10.2f | %10.2f | %10d\n", G,
           (G - 1) * stagger * cyc_us, rb.a_total_ns / 1000.0,
           rp.sched_ns / 1000.0, rp.rel_ns / 1000.0, rl.rel_ns / 1000.0,
           rp.done);
    CUDA_CHECK(cudaGraphExecDestroy(gb));
    CUDA_CHECK(cudaGraphExecDestroy(gp));
    CUDA_CHECK(cudaGraphExecDestroy(gl));
  }

  // ---- Section C：capture 图 vs 显式图（同一 programmatic 依赖） ----
  printf("\n=== Section C: capture 图（P19-1 形态）vs 显式 programmatic 边 ===\n");
  printf("同一条 A(trigger)->B 依赖，两种构图方式的放行点对比（G=1）。\n");
  printf("%-24s | %10s | %10s | %10s | %8s | %8s\n", "构图", "A_total(us)",
         "sched(us)", "rel(us)", "rel/A", "done");
  printf("-------------------------+------------+------------+------------+----------+---------\n");
  {
    cudaGraphExec_t gc = build_capture(g, 1, pro, tail, 0, tail, tail);
    Result rc = run_graph(gc, g, iters);
    printf("%-24s | %10.2f | %10.2f | %10.2f | %8.3f | %8d\n", "capture (PDL attr)",
           rc.a_total_ns / 1000.0, rc.sched_ns / 1000.0, rc.rel_ns / 1000.0,
           rc.rel_ns / rc.a_total_ns, rc.done);
    CUDA_CHECK(cudaGraphExecDestroy(gc));
    cudaGraphExec_t ge = build_explicit(g, 1, E_PROG, pro, tail, 0, tail, tail);
    Result re = run_graph(ge, g, iters);
    printf("%-24s | %10.2f | %10.2f | %10.2f | %8.3f | %8d\n", "explicit PROGRAMMATIC",
           re.a_total_ns / 1000.0, re.sched_ns / 1000.0, re.rel_ns / 1000.0,
           re.rel_ns / re.a_total_ns, re.done);
    CUDA_CHECK(cudaGraphExecDestroy(ge));
  }

  printf("\n>> 读法：关键量是 **sched**（下游何时被调度拿到 SM），不是 rel。\n");
  printf(">> block：sched≈A_total（完成即放行，下游 prologue 不能与上游 tail 重叠）；\n");
  printf(">> prog_nt：A 不 trigger -> 端口在终止时激活，sched≈A_total；\n");
  printf(">> prog：A 在 prologue 后 trigger -> sched≈pro << A_total——**图的 programmatic 边\n");
  printf(">>   把下游的调度点前移到了上游完成之前**（下游 prologue 与上游 tail 重叠），\n");
  printf(">>   这正是 stream 上 cudaStreamWaitEvent 做不到的（§21）。\n");
  printf(">> lce：全体 block 启动即放行，sched≈0。\n");
  printf(">> 但 rel（下游 cudaGridDependencySynchronize 返回）恒 ≈ A_total——端口只决定\n");
  printf(">>   下游**何时被调度**，不改变 gridDependencySync 等**上游完成**的语义（与 §21 一致）。\n");
  printf(">> done 应恒 == iters。\n");

  CUDA_CHECK(cudaFree(g));
  return 0;
}
