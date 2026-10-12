// programmatic_event.cu — MegaKernel 专题 P32-1 小实验：
//   cudaLaunchAttributeProgrammaticEvent / LaunchCompletionEvent 能把「事件可见性」提前到
//   kernel 完成之前的哪一点？
//
// 背景（目标卡 H100 / sm90 实测）：
//   P5-3/P12-1/P19-1 已经把 PDL（Programmatic Dependent Launch）测透：下游 kernel 靠
//   cudaGridDependencySynchronize() 越过上游 grid 依赖，重叠量 = min(tail, prologue)。
//   但 PDL 里「放行信号」只在**上游 grid 退出/全体触发**时发生。CUDA 还提供两种把
//   **一个 cudaEvent 的触发点前移**的 launch attribute（`driver_types.h`：
//   cudaLaunchAttributeProgrammaticEvent=7、cudaLaunchAttributeLaunchCompletionEvent=12）：
//     - ProgrammaticEvent：事件在「所有 block 都调用过 cudaTriggerProgrammaticLaunchCompletion()
//       （PTX griddepcontrol.launch_dependents）」后触发；也可用 triggerAtBlockStart=1 让每个
//       block 在**启动时**自动触发。
//     - LaunchCompletionEvent：在所有 block **开始**执行时触发（best effort）。
//   这恰好是 megakernel 最关心的问题：**「中间进度」能否被别的执行体更早看到**——
//   MPK 用 in-kernel event counter + 原子队列把依赖粒度压到 task（`docs/06` §3），
//   本节量的则是 kernel 级原语能把可见性提前多少（0 / prologue / 完成）。
//
// 模型：A 一个 kernel = [prologue 自旋（可逐 block 错峰）] -> [可选 trigger] -> [tail 自旋]；
//       C 一个消费 kernel 放在另一条 stream，靠 `cudaStreamWaitEvent(C, evX)` 等某个事件。
//       用两个**计时事件** evA0（A 之前 record）与 evC1（C 之后 record）测
//       「A 启动 → C 被放行开始」的 GPU 时间轴间隔 = **放行信号的时间位置**。
//
// 四种事件档（公平对照，同一 A 工作量）：
//   seq    : 无事件，A 后直接在同一 stream 排 C（=A 完成才跑）—— 参考地板
//   post   : 普通 cudaEventRecord(evX, streamA)（A 完成后触发）—— 基线
//   early  : ProgrammaticEvent，A 在 prologue 后调 trigger（事件在两段之间触发）
//   none   : ProgrammaticEvent，A 不调 trigger（事件在 grid 退出前自动触发）
//   bstart : ProgrammaticEvent + triggerAtBlockStart=1（每个 block 启动即触发）
//   lce    : LaunchCompletionEvent（所有 block 开始执行即触发）
//
// 编译：nvcc -O3 -arch=sm_90 programmatic_event.cu -o programmatic_event
// 用法：./programmatic_event [iters] [pro_cycles] [tail_cycles] [step_cycles]

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

// 忙等约 n 个 SM clock 周期（每线程各自自旋，不依赖 block 内对齐）。
__device__ __forceinline__ void spin_cycles(long long n) {
  long long s = clock64();
  while (clock64() - s < n) {
  }
}

// A：prologue -> [可选 trigger] -> tail。
//   stagger：逐 block 错峰量——block i 的 prologue 延长 i*stagger 个周期（测「所有 block 触发」语义）。
//   do_trigger：prologue 后是否调用 cudaTriggerProgrammaticLaunchCompletion()（early 档）。
__global__ void ev_kernel(long long pro, long long tail, int do_trigger,
                          long long stagger) {
  long long mypro = pro + (long long)blockIdx.x * stagger;
  spin_cycles(mypro);
  if (do_trigger) {
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
  }
  spin_cycles(tail);
}

// C：极短消费 kernel——被放行后立刻开始、写一个哨兵（正确性用）。
__global__ void dep_kernel(int* out) {
  if (threadIdx.x == 0) atomicAdd(out, 1);
}

// B：PDL 下游——prologue 后调 cudaGridDependencySynchronize()（等上游 programmatic port 放行），
//    再跑临界段。用于验证「programmatic 提前 trigger 是设备侧 port」。
__global__ void pdl_consumer(long long prol, long long crit, int* out) {
  spin_cycles(prol);
  cudaGridDependencySynchronize();
  spin_cycles(crit);
  if (threadIdx.x == 0) atomicAdd(out, 1);
}

enum EvMode { M_SEQ = 0, M_POST, M_EARLY, M_NONE, M_BSTART, M_LCE };
static const char* mode_name(int m) {
  switch (m) {
    case M_SEQ: return "seq";
    case M_POST: return "post";
    case M_EARLY: return "early";
    case M_NONE: return "none";
    case M_BSTART: return "bstart";
    case M_LCE: return "lce";
  }
  return "?";
}

// 返回「A 启动 → C 被放行开始」的平均 GPU 时间（µs），同时回填两个参考：
//   a_total：A 整段耗时（µs，post 档下 evX 触发即 A 完成）；dep：C 被真正执行次数（正确性）。
static float run_case(int mode, int G, long long pro, long long tail,
                      long long stagger, int iters, float* a_total, int* dep) {
  cudaStream_t sA, sC;
  CUDA_CHECK(cudaStreamCreate(&sA));
  CUDA_CHECK(cudaStreamCreate(&sC));
  cudaEvent_t evA0, evC1, evAend;
  CUDA_CHECK(cudaEventCreate(&evA0));
  CUDA_CHECK(cudaEventCreate(&evC1));
  CUDA_CHECK(cudaEventCreate(&evAend));

  int* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, sizeof(int)));
  CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));

  int do_trigger = (mode == M_EARLY) ? 1 : 0;
  dim3 grid(G), block(128);

  double sum = 0.0, sum_a = 0.0;
  for (int it = 0; it < iters; ++it) {
    cudaEvent_t evX = nullptr;
    if (mode != M_SEQ) {
      // programmatic / LCE 事件必须 disable timing（cuda.h 明确要求）。
      CUDA_CHECK(cudaEventCreateWithFlags(&evX, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(evA0, sA));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = sA;
    cudaLaunchAttribute attrs[1] = {};
    if (mode == M_EARLY || mode == M_NONE) {
      attrs[0].id = cudaLaunchAttributeProgrammaticEvent;
      attrs[0].val.programmaticEvent.event = evX;
      attrs[0].val.programmaticEvent.flags = 0;
      attrs[0].val.programmaticEvent.triggerAtBlockStart = 0;
      cfg.attrs = attrs;
      cfg.numAttrs = 1;
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, ev_kernel, pro, tail, do_trigger, stagger));
    } else if (mode == M_BSTART) {
      attrs[0].id = cudaLaunchAttributeProgrammaticEvent;
      attrs[0].val.programmaticEvent.event = evX;
      attrs[0].val.programmaticEvent.flags = 0;
      attrs[0].val.programmaticEvent.triggerAtBlockStart = 1;
      cfg.attrs = attrs;
      cfg.numAttrs = 1;
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, ev_kernel, pro, tail, 0, stagger));
    } else if (mode == M_LCE) {
      attrs[0].id = cudaLaunchAttributeLaunchCompletionEvent;
      attrs[0].val.launchCompletionEvent.event = evX;
      attrs[0].val.launchCompletionEvent.flags = 0;
      cfg.attrs = attrs;
      cfg.numAttrs = 1;
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, ev_kernel, pro, tail, 0, stagger));
    } else {
      ev_kernel<<<grid, block, 0, sA>>>(pro, tail, 0, stagger);
    }

    if (mode == M_SEQ) {
      // 无事件：同一 stream 直接排 C（A 完成后才执行）。
      dep_kernel<<<1, 32, 0, sA>>>(out);
      CUDA_CHECK(cudaEventRecord(evC1, sA));
      CUDA_CHECK(cudaEventSynchronize(evC1));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, evA0, evC1));
      sum += ms * 1000.0;
    } else {
      if (mode == M_POST) {
        // 基线：A 完成后普通 record。
        CUDA_CHECK(cudaEventRecord(evX, sA));
      }
      CUDA_CHECK(cudaStreamWaitEvent(sC, evX, 0));
      dep_kernel<<<1, 32, 0, sC>>>(out);
      CUDA_CHECK(cudaEventRecord(evC1, sC));
      CUDA_CHECK(cudaEventSynchronize(evC1));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, evA0, evC1));
      sum += ms * 1000.0;
    }

    // A 整段耗时（post 档下其事件触发点即完成点；其余档也用同一 measure，仅作参考）。
    CUDA_CHECK(cudaEventRecord(evAend, sA));
    CUDA_CHECK(cudaEventSynchronize(evAend));
    float msa = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msa, evA0, evAend));
    sum_a += msa * 1000.0;

    if (evX) {
      CUDA_CHECK(cudaStreamSynchronize(sA));
      CUDA_CHECK(cudaEventDestroy(evX));
    }
  }

  int hdep = 0;
  CUDA_CHECK(cudaMemcpy(&hdep, out, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaStreamDestroy(sA));
  CUDA_CHECK(cudaStreamDestroy(sC));
  CUDA_CHECK(cudaEventDestroy(evA0));
  CUDA_CHECK(cudaEventDestroy(evC1));
  CUDA_CHECK(cudaEventDestroy(evAend));
  *a_total = (float)(sum_a / iters);
  *dep = hdep;
  return (float)(sum / iters);
}

// 同 stream PDL：A（pro -> [可选 trigger] -> tail）紧跟 B（prol -> gridDependencySync -> crit）。
// 返回 stream 总耗时（µs）。attach_ev：是否给 A 挂 ProgrammaticEvent 事件属性。
static float run_stream_case(int do_trigger, int attach_ev, long long pro,
                             long long tail, long long prol, long long crit,
                             int iters) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  int* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, sizeof(int)));
  CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));
  double sum = 0;
  for (int it = 0; it < iters; ++it) {
    cudaEvent_t evX = nullptr;
    if (attach_ev) CUDA_CHECK(cudaEventCreateWithFlags(&evX, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(e0, s));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.stream = s;
    cudaLaunchAttribute at[1] = {};
    if (attach_ev) {
      at[0].id = cudaLaunchAttributeProgrammaticEvent;
      at[0].val.programmaticEvent.event = evX;
      cfg.attrs = at;
      cfg.numAttrs = 1;
    }
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, ev_kernel, pro, tail, do_trigger, 0LL));
    // 下游 B：programmatic serialization allowed，其 gridDependencySync 等上游 port 放行。
    cudaLaunchConfig_t c2 = {};
    c2.gridDim = dim3(1);
    c2.blockDim = dim3(128);
    c2.stream = s;
    cudaLaunchAttribute ca[1] = {};
    ca[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    ca[0].val.programmaticStreamSerializationAllowed = 1;
    c2.attrs = ca;
    c2.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&c2, pdl_consumer, prol, crit, out));
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    sum += ms * 1000.0;
    if (evX) CUDA_CHECK(cudaEventDestroy(evX));
  }
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaStreamDestroy(s));
  return (float)(sum / iters);
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 30;
  long long pro = (argc > 2) ? atoll(argv[2]) : 100000;
  long long tail = (argc > 3) ? atoll(argv[3]) : 100000;
  long long step = (argc > 4) ? atoll(argv[4]) : 20000;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int clk_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  if (clk_khz <= 0) clk_khz = 1980000;  // H100 boost 兜底
  double clk_hz = (double)clk_khz * 1000.0;
  printf("== device ==\n%s, sm_%d%d, SMs=%d, clock=%.3f GHz\n", prop.name,
         prop.major, prop.minor, prop.multiProcessorCount, clk_hz / 1e9);
  printf("iters=%d  pro=%lld  tail=%lld  step=%lld cycles\n", iters, pro, tail,
         step);

  const int modes[6] = {M_SEQ, M_POST, M_EARLY, M_NONE, M_BSTART, M_LCE};

  // ---- Section A：单 block，扫事件档——放行信号在时间轴上的位置 ----
  printf("\n=== Section A: 单 block（G=1），event 触发点 vs A 时间轴 ===\n");
  printf("A = prologue(%lld cyc) -> [trigger] -> tail(%lld cyc)，C 在另一 stream 等 evX\n",
         pro, tail);
  printf("%8s | %14s | %14s | %14s | %8s\n", "mode", "A_start->C(us)",
         "A_total(us)", "C-Ahead(us)", "dep");
  printf("---------+----------------+----------------+----------------+---------\n");
  float ref_c = 0;
  for (int i = 0; i < 6; ++i) {
    float at = 0;
    int dep = 0;
    float t = run_case(modes[i], 1, pro, tail, 0, iters, &at, &dep);
    if (modes[i] == M_POST) ref_c = t;
    printf("%8s | %14.2f | %14.2f | %14.2f | %8d\n", mode_name(modes[i]), t,
           at, ref_c - t, dep);
  }

  // ---- Section B：多 block，stagger 错峰——cross-stream 放行点随 G 怎么变 ----
  printf("\n=== Section B: 多 block 错峰（block i 的 prologue 加 i*step）===\n");
  printf("step=%.2f us，tail=%.2f us；若事件在完成点放行，A_start->C ≈ (G-1)*step+pro+tail\n",
         step / clk_hz * 1000000.0, tail / clk_hz * 1000000.0);
  printf("%5s | %10s | %14s | %14s | %14s | %14s\n", "G", "(G-1)*step(us)",
         "early(us)", "bstart(us)", "post(us)", "lce(us)");
  printf("------+------------+----------------+----------------+----------------+----------------\n");
  for (int G : {1, 2, 4, 8, 16, 32, 64, 128}) {
    float at = 0;
    int dep = 0;
    float t_early = run_case(M_EARLY, G, pro, tail, step, iters, &at, &dep);
    float t_bs = run_case(M_BSTART, G, pro, tail, step, iters, &at, &dep);
    float t_post = run_case(M_POST, G, pro, tail, step, iters, &at, &dep);
    float t_lce = run_case(M_LCE, G, pro, tail, step, iters, &at, &dep);
    printf("%5d | %10.2f | %14.2f | %14.2f | %14.2f | %14.2f\n", G,
           (G - 1) * step / clk_hz * 1000000.0, t_early, t_bs, t_post, t_lce);
  }

  // ---- Section C：同 stream PDL——trigger 的作用是「允许下游 grid 提前被调度」 ----
  long long prol = tail;  // 下游 prologue，取与上游 tail 等长以便观测重叠
  long long crit = tail;  // 下游 sync 之后的临界段
  printf("\n=== Section C: 同 stream PDL，A=pro(%lld)->[可选 trigger]->tail(%lld)，"
         "B=prol(%lld)->gridDependencySync->crit(%lld)===\n",
         pro, tail, prol, crit);
  printf("B 用 cudaLaunchAttributeProgrammaticStreamSerialization；看 stream 总耗时：\n");
  printf("  无 trigger  -> B 不能被提前调度，B 整段排在 A 之后\n");
  printf("  有 trigger  -> B 的 prologue 与 A 的 tail 重叠（sync 仍等 A 完成，只重叠 prologue）\n");
  printf("%-46s | %12s\n", "case", "stream total(us)");
  printf("-----------------------------------------------+--------------\n");
  float c_noT = run_stream_case(0, 0, pro, tail, prol, crit, iters);
  float c_early = run_stream_case(1, 0, pro, tail, prol, crit, iters);
  float c_early_ev = run_stream_case(1, 1, pro, tail, prol, crit, iters);
  printf("%-46s | %12.2f\n", "A 不调 trigger", c_noT);
  printf("%-46s | %12.2f\n", "A 调 early trigger（无事件属性）", c_early);
  printf("%-46s | %12.2f\n", "A 调 early trigger + ProgrammaticEvent 属性", c_early_ev);
  printf("期望：无 trigger ≈ pro+tail+prol+crit=%.1f；有 trigger ≈ pro+max(tail,prol)+crit=%.1f\n",
         (pro + tail + prol + crit) / clk_hz * 1000000.0,
         (pro + (tail > prol ? tail : prol) + crit) / clk_hz * 1000000.0);
  printf("结论：trigger 的作用是**允许下游 grid 提前被调度**（省下 min(tail,prol) 的重叠），\n");
  printf("      但下游的 cudaGridDependencySynchronize() 仍等上游**完成**——trigger 不放行 sync；\n");
  printf("      挂不挂 ProgrammaticEvent 事件属性结果相同（设备端口由 trigger 本身提供，事件属性\n");
  printf("      只是把同一端口额外记录到 cudaEvent，供图 PROGRAMMATIC 边使用）。\n");

  printf("\n>> 读法：A_start->C 越小 = 下游越早被放行（cross-stream cudaStreamWaitEvent）。\n");
  printf(">> seq/post ≈ A 整段 = 完成点放行（基线）；\n");
  printf(">> programmatic（early/none/bstart）经 cross-stream wait **也落在完成点**——\n");
  printf(">>   ProgrammaticEvent 的提前 trigger 不放行 stream/host 依赖（见正文，与 CUDA 文档\n");
  printf(">>   '依赖方不保证在 trigger 时刻观察到放行' 一致）；它是 PDL 的『可提前调度』port，\n");
  printf(">>   供设备内 cudaGridDependencySynchronize() 与图 PROGRAMMATIC 边使用（见 Section C）。\n");
  printf(">> 只有 lce（LaunchCompletionEvent）真正提前：所有 block 一启动即放行 ~= 0；\n");
  printf(">> Section B 里 lce 恒 ~0 不随 G 变，而 early/bstart/post 随 (G-1)*step 线性增长。\n");
  return 0;
}
