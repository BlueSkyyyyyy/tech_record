// pdl.cu — MegaKernel 专题 P4-2 小实验：PDL（Programmatic Dependent Launch）能重叠多少
//
// 目的（目标卡 H100 / sm90 实测）：
//   PDL 让同一 stream 上的相邻 kernel **部分重叠**——下游 kernel 可以提前被硬件启动，
//   先执行自己的「前导（prologue）」代码，再调用 cudaGridDependencySynchronize() 等待上游
//   完成（或上游 cudaTriggerProgrammaticLaunchCompletion() 放行）后才读上游数据。
//   MPK 论文说 PDL「能缓解 kernel barrier，但要重写 kernel 结构与控制流」（快照 §1 行 78–84）。
//   本实验量化这句话：
//     (1) 无 PDL：下游必须等上游**整段**跑完才启动（串行，total ≈ P + C）；
//     (2) 有 PDL + 下游有 prologue：下游 prologue 与上游 tail 重叠（total ≈ max 而非 sum）；
//     (3) 有 PDL 但下游**没有** prologue：几乎无收益（收益来自「可提前做的活」）。
//
// 模型：producer 先做 pre 段，调用 trigger 放行，再做 tail 段；
//       consumer 先做 prologue 段（不碰上游数据），再 grid-dependency-sync，最后读 flag。
//
// 编译：nvcc -O3 -arch=sm_90 pdl.cu -o pdl
// 用法：./pdl [iters] [pre_cycles] [tail_cycles] [prologue_cycles]

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

// 忙等约 n 个 SM clock 周期（只看 block 0/thread 0，其余线程由 __syncthreads 对齐）。
__device__ __forceinline__ void spin_cycles(long long n) {
  long long s = clock64();
  while (clock64() - s < n) {
  }
}

// producer：pre 段 -> 数据写 memory -> trigger 放行 -> tail 段
__global__ void pdl_producer(int* flag, long long pre, long long tail) {
  spin_cycles(pre);
  if (threadIdx.x == 0) *flag = 1;
  __syncthreads();
  cudaTriggerProgrammaticLaunchCompletion();  // 全 grid 触发后，下游可越过 grid 依赖
  spin_cycles(tail);
}

// consumer：prologue 段（不依赖上游）-> 等上游 -> 读数据
__global__ void pdl_consumer(int* flag, int* out, long long prologue) {
  spin_cycles(prologue);
  cudaGridDependencySynchronize();  // PDL 语义：等上游 grid 放行/完成
  if (threadIdx.x == 0 && *flag == 1) atomicAdd(out, 1);
}

static float time_pair(int iters, bool use_pdl, long long pre, long long tail,
                       long long prologue, int* flag, int* out) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = use_pdl ? 1 : 0;

  auto launch_once = [&]() {
    // producer 无论基线都用普通 launch（trigger 本身是 no-op 语义，基线下游不会提前启动）
    pdl_producer<<<1, 128, 0, s>>>(flag, pre, tail);
    if (use_pdl) {
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, pdl_consumer, flag, out, prologue));
    } else {
      pdl_consumer<<<1, 128, 0, s>>>(flag, out, prologue);
    }
  };

  launch_once();
  CUDA_CHECK(cudaDeviceSynchronize());  // warmup
  CUDA_CHECK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r) launch_once();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 500;
  long long pre = (argc > 2) ? atoll(argv[2]) : 200000;
  long long tail = (argc > 3) ? atoll(argv[3]) : 200000;
  long long prologue = (argc > 4) ? atoll(argv[4]) : 200000;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount);
  printf("iters=%d  pre=%lld tail=%lld prologue=%lld (SM clock cycles)\n\n",
         iters, pre, tail, prologue);

  int* flag = nullptr;
  int* out = nullptr;
  CUDA_CHECK(cudaMalloc(&flag, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out, sizeof(int)));

  // 公平对照：对同一 prologue 值，分别测 no-PDL 与 PDL（工作量相同）。
  const long long prol[] = {0, prologue / 2, prologue, 2 * prologue};
  printf("%10s | %12s | %12s | %8s | %10s\n", "prologue", "no-PDL ms/it",
         "PDL ms/it", "speedup", "overlap(cyc)");
  printf("-----------+--------------+--------------+----------+------------\n");
  int h_out = 0;
  for (int i = 0; i < 4; ++i) {
    long long pl = prol[i];
    CUDA_CHECK(cudaMemset(flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));
    float t_no = time_pair(iters, false, pre, tail, pl, flag, out);

    CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));
    float t_pdl = time_pair(iters, true, pre, tail, pl, flag, out);

    // 理论重叠量：被省下的时间 ≈ min(tail, prologue)，换算回周期数（用 PDL 的单轮 ms 标定）
    CUDA_CHECK(cudaMemcpy(&h_out, out, sizeof(int), cudaMemcpyDeviceToHost));

    printf("%10lld | %12.4f | %12.4f | %7.2fx | %10lld\n", pl, t_no, t_pdl,
           t_no / t_pdl, (long long)(pl < tail ? pl : tail));
  }

  printf("\nchecksum (consumer 成功读到 flag 的累计次数，末轮值): %d\n", h_out);

  printf("\n>> 读法：no-PDL 时下游严格等上游整段跑完，total≈pre+tail+prologue；\n");
  printf(">> PDL 时下游 prologue 与上游 tail 并行，total≈pre+max(tail,prologue)；\n");
  printf(">> prologue 越大收益越大（上限 min(tail,prologue) 被省下）；\n");
  printf(">> prologue=0 时 speedup≈1——PDL 不是万能加速，只重叠「下游本可提前做的活」。\n");

  CUDA_CHECK(cudaFree(flag));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
