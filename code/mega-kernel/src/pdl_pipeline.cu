// pdl_pipeline.cu — MegaKernel 专题 P5-3 小实验：PDL 串成「流水线」后能省多少
//
// 目的（目标卡 H100 / sm90 实测）：
//   P4-2（src/pdl.cu）只量了**一对** kernel 的 PDL 重叠：重叠量 = min(上游 tail, 下游 prologue)，
//   收益上限 1.5×。本实验把 K 个 kernel 用 PDL 串成一条**链**，回答：
//     (1) 无 PDL 时，K 个 kernel 严格串行，总时间 ∝ K·(prologue+crit+tail)；
//   (2) 有 PDL 时，第 i 个 kernel 的 prologue 与第 i-1 个 kernel 的 tail 重叠，
//         **每条链边的重叠量 = min(prologue, tail)**（与 P4-2 单对结论一致），
//         故稳态每 kernel 周期 = (prologue+crit+tail) − min(prologue, tail)；
//     (3) 于是链越长，speedup 越接近 (prologue+crit+tail)/(prologue+crit+tail−min(prologue,tail))，
//         **由 min(prologue,tail) 决定上限、不会随 K 无限叠加**——tail≫prologue 时收益被 prologue 封顶。
//   这解释了为什么 PDL 只是 kernel 级流水：它把「相邻一对」重叠掉 min(prologue,tail)，
//   多条链边并不累积成更细的流水，也不是 MPK 的跨算子 SM 级软件流水（§5.3 行 1326–1363）。
//
// 链结构（每个 kernel 的 time 轴）：
//     [prologue 自旋 pl] -> cudaGridDependencySynchronize -> [临界段: 校验上游 seq, 写 seq, 自旋 crit]
//     -> cudaTriggerProgrammaticLaunchCompletion -> [tail 自旋 tl]
//   idx=0 无上游；idx>=1 的 kernel 用 cudaLaunchKernelEx 带 programmatic 属性启动。
//
// 编译：nvcc -O3 -arch=sm_90 pdl_pipeline.cu -o pdl_pipeline
// 用法：./pdl_pipeline [iters] [Kmax] [prologue] [crit] [tail] [tl2]

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

// 链上的第 idx 个 kernel。seq 长度 = K+1；每轮用 gen 做代次，避免跨迭代的假通过。
__global__ void chain_kernel(int* seq, int gen, int idx, int* fail,
                             long long prologue, long long crit, long long tail) {
  spin_cycles(prologue);
  cudaGridDependencySynchronize();  // 等到上游 kernel 放行（idx=0 时为 no-op）
  if (threadIdx.x == 0) {
    if (idx > 0 && seq[idx] != gen) atomicAdd(fail, 1);  // 校验上游数据可见
    spin_cycles(crit);
    seq[idx + 1] = gen;                                  // 把本 kernel 的「结果」交给下游
  }
  __syncthreads();
  cudaTriggerProgrammaticLaunchCompletion();  // 放行下游，使其 prologue 可与本 kernel tail 重叠
  spin_cycles(tail);
}

// 跑一条 K-kernel 链，返回每轮平均 ms。gen 由调用方维护。
static float time_chain(int iters, int K, bool use_pdl, long long prologue,
                        long long crit, long long tail, int* seq, int* fail,
                        long long* out_gen, int* out_fail) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;

  int gen = *out_gen;
  auto launch_chain = [&]() {
    for (int k = 0; k < K; ++k) {
      if (use_pdl && k > 0) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(1);
        cfg.blockDim = dim3(128);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = s;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, chain_kernel, seq, gen, k, fail,
                                      prologue, crit, tail));
      } else {
        chain_kernel<<<1, 128, 0, s>>>(seq, gen, k, fail, prologue, crit, tail);
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(s));  // 每轮收尾，避免轮与轮之间重叠
  };

  gen++;
  launch_chain();  // warmup
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < iters; ++r) {
    gen++;
    launch_chain();
  }
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

  *out_gen = gen;
  CUDA_CHECK(cudaMemcpy(out_fail, fail, sizeof(int), cudaMemcpyDeviceToHost));
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
  long long gen = 0;
  int fails = 0;

  printf("\n== %s ==  prologue=%lld crit=%lld tail=%lld (SM cycles)\n", name,
         prologue, crit, tail);
  long long span = prologue + crit + tail;
  long long ov = prologue < tail ? prologue : tail;  // 每条链边的重叠量 = min(prologue, tail)
  printf("%4s | %11s | %11s | %8s | %13s | %13s | %13s\n", "K", "no-PDL ms",
         "PDL ms", "speedup", "no-PDL/K ms", "PDL/K ms", "pred PDL/K");
  printf("-----+-------------+-------------+----------+---------------+----------"
         "-----+--------------\n");

  for (int K = 1; K <= Kmax; K *= 2) {
    CUDA_CHECK(cudaMemset(seq, 0, sizeof(int) * (Kmax + 2)));
    CUDA_CHECK(cudaMemset(fail, 0, sizeof(int)));
    int f1 = 0;
    float t_no = time_chain(iters, K, false, prologue, crit, tail, seq, fail,
                            &gen, &f1);

    CUDA_CHECK(cudaMemset(seq, 0, sizeof(int) * (Kmax + 2)));
    CUDA_CHECK(cudaMemset(fail, 0, sizeof(int)));
    int f2 = 0;
    float t_pdl = time_chain(iters, K, true, prologue, crit, tail, seq, fail,
                             &gen, &f2);
    fails += f1 + f2;

    // 预测：稳态每 kernel 周期 = span - min(prologue, tail)（每条链边的重叠）。
    // 用 no-PDL 的每 kernel ms 标定周期→ms（no-PDL/K = span 周期）。
    double pred_pdl_K =
        t_no / K * (double)(span - ov) / (double)span;
    printf("%4d | %11.4f | %11.4f | %7.2fx | %13.4f | %13.4f | %13.4f\n", K,
           t_no, t_pdl, t_no / t_pdl, t_no / K, t_pdl / K, pred_pdl_K);
  }
  printf("correctness: fail counters summed over all K/iters = %d "
         "(0 = 所有下游都正确读到上游 seq)\n", fails);

  // 链路（K=Kmax）的末尾值：seq[K] 应等于最后一次 gen
  int h_last = 0;
  CUDA_CHECK(cudaMemcpy(&h_last, seq + Kmax, sizeof(int), cudaMemcpyDeviceToHost));
  printf("tail check: seq[%d]=%lld (last gen=%lld, 两者相符则链路完整)\n", Kmax,
         (long long)h_last, gen);

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

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount);
  printf("iters=%d Kmax=%d\n", iters, Kmax);

  run_case("A: tail == prologue", iters, Kmax, prologue, crit, tail);
  run_case("B: tail = 3x prologue", iters, Kmax, prologue, crit, tail2);

  printf("\n>> 读法：no-PDL 每 kernel = prologue+crit+tail（严格串行）；\n");
  printf(">> PDL 每条链边重叠 min(prologue, tail)，稳态每 kernel = span - min(prologue, tail)，\n");
  printf(">> 故 K 大时 speedup 趋 span/(span-min)；A 组(pl=tail) →~2×，B 组(tail=3×pl) →~1.33×。\n");
  printf(">> 关键：收益由 min(prologue,tail) 封顶，**不会随 K 无限叠加**——PDL 是 kernel 级流水，\n");
  printf(">> 链上仍有 host 逐个 launch、中间量仍可落 HBM，非 MPK 单 kernel 内 SM 级软件流水（§5.3 行 1326–1363）。\n");
  return 0;
}
