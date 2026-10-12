// pdl_multiblock.cu — MegaKernel 专题 P12-1 小实验：多 block PDL 的 occupancy 资源约束
//
// 目的（目标卡 H100 / sm90 实测）：
//   P4-2 与 P5-3 都用 **grid=1** 的单个小 kernel 造流，得到「PDL 重叠量 = min(上游 tail, 下游 prologue)」。
//   但真实算子的 grid 是**上百个 block**。P5-3 §3.6、P10-1 §5.5 都留了同一个尾巴：
//     **PDL 后继 grid 的 block 何时真正拿到 SM 资源？多 block / 占满 SM 时重叠还剩多少？**
//   本实验把一对 producer/consumer 从 grid=1 扩到多 block，量化「占用率」对 PDL 重叠的约束：
//
//   (1) PDL 的硬件语义：下游 grid 可在上游触发后**提前被 launch**，但下游 block 仍要等某个 SM
//       有空闲槽位才能跑它的 prologue。上游 grid 越接近占满 occupancy，下游能落脚的余地越小，
//       且它与上游 block 共享 SM 发射/L1，prologue 本身也被拖慢——**有效重叠缩水**。
//   (2) 因此 PDL 的收益不只取决于「下游有多少 prologue」，还取决于「上游给不给 occupancy 余量」：
//       它把一部分重叠机会从 kernel 边界转移到了 **occupancy 边界**上。
//
// 结构（一对 kernel，可多 block）：
//   producer: [pre 自旋] -> thread0 写 flag[0]=gen -> trigger -> [tail 自旋] -> exit
//   consumer: [prologue 自旋] -> gridDependencySynchronize -> 校验 flag[0]==gen
//   下游 gridDependencySync 之前的那段 prologue 才是能与上游 tail 重叠的部分。
//
// 编译：nvcc -O3 -arch=sm_90 pdl_multiblock.cu -o pdl_multiblock
// 用法：./pdl_multiblock [iters] [blockDim] [pre] [tail] [prologue] [crit]

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

// producer（上游）：pre 段 -> 写 flag -> 放行下游 -> tail 段。
__global__ void mb_producer(int* flag, int gen, long long pre, long long tail) {
  spin_cycles(pre);
  if (threadIdx.x == 0) flag[0] = gen;
  __syncthreads();
  cudaTriggerProgrammaticLaunchCompletion();  // 全 grid 触发后下游可越过 grid 依赖
  spin_cycles(tail);
}

// consumer（下游）：prologue 段（可与上游 tail 重叠）-> 等上游 -> 只由 block0/thread0 读 flag。
__global__ void mb_consumer(const int* flag, int gen, int* out, long long prologue) {
  spin_cycles(prologue);
  cudaGridDependencySynchronize();
  if (threadIdx.x == 0 && blockIdx.x == 0 && flag[0] == gen) atomicAdd(out, 1);
}

// 跑一对 (Gu, Gd) 的 kernel，返回每轮平均 ms，并把正确性写入 out（期望 = iters）。
static float time_pair(int iters, bool use_pdl, int Gu, int Gd, int B,
                       long long pre, long long tail, long long prologue,
                       int* flag, int* out) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(Gd);
  cfg.blockDim = dim3(B);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = use_pdl ? 1 : 0;

  int gen = 0;
  auto launch_once = [&]() {
    mb_producer<<<Gu, B, 0, s>>>(flag, gen, pre, tail);
    if (use_pdl) {
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, mb_consumer, flag, gen, out, prologue));
    } else {
      mb_consumer<<<Gd, B, 0, s>>>(flag, gen, out, prologue);
    }
  };

  ++gen;
  launch_once();
  CUDA_CHECK(cudaDeviceSynchronize());  // warmup
  CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(int), s));  // 清掉 warmup 的计数

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < iters; ++r) {
    ++gen;
    launch_once();
  }
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

// 一轮「配对」测量：同一 (Gu,Gd) 工作量下测 no-PDL 与 PDL。
// 有效重叠 = (t_no - t_pdl) / (no-PDL 每轮 ms/cycle)，ms/cycle 用 no-PDL 的跨度周期标定。
static void pair_point(int iters, int Gu, int Gd, int B, long long pre,
                       long long tail, long long prologue, int* flag, int* out,
                       float* t_no_out, float* t_pdl_out, float* sp_out,
                       long long* eff_ov_out, int* fail_out) {
  int h_out = 0;
  CUDA_CHECK(cudaMemset(flag, 0, sizeof(int)));
  float t_no = time_pair(iters, false, Gu, Gd, B, pre, tail, prologue, flag, out);
  CUDA_CHECK(cudaMemcpy(&h_out, out, sizeof(int), cudaMemcpyDeviceToHost));
  int f_no = (h_out == iters) ? 0 : 1;

  float t_pdl = time_pair(iters, true, Gu, Gd, B, pre, tail, prologue, flag, out);
  CUDA_CHECK(cudaMemcpy(&h_out, out, sizeof(int), cudaMemcpyDeviceToHost));
  int f_pdl = (h_out == iters) ? 0 : 1;

  double span_cyc = (double)(pre + tail + prologue);
  double ms_per_cyc = t_no / span_cyc;
  long long eff_ov = (long long)((t_no - t_pdl) / ms_per_cyc);

  *t_no_out = t_no;
  *t_pdl_out = t_pdl;
  *sp_out = t_no / t_pdl;
  *eff_ov_out = eff_ov;
  *fail_out = f_no + f_pdl;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 20;
  int B = (argc > 2) ? atoi(argv[2]) : 256;
  long long pre = (argc > 3) ? atoll(argv[3]) : 80000;
  long long tail = (argc > 4) ? atoll(argv[4]) : 80000;
  long long prologue = (argc > 5) ? atoll(argv[5]) : 80000;
  long long crit = (argc > 6) ? atoll(argv[6]) : 2000;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int sm = prop.multiProcessorCount;
  int rblk_p = 0, rblk_c = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rblk_p, mb_producer, B, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rblk_c, mb_consumer, B, 0));
  int R = rblk_p < rblk_c ? rblk_p : rblk_c;  // 两 kernel 都只用最简资源，取较小者
  int cap = R * sm;

  printf("== device ==\n%s, sm_%d%d, SMs=%d\n", prop.name, prop.major,
         prop.minor, sm);
  printf("blockDim=%d  blocks/SM(producer)=%d (consumer)=%d  ->  占用上限 R*#SM = %d\n",
         B, rblk_p, rblk_c, cap);
  printf("iters=%d  pre=%lld tail=%lld prologue=%lld crit=%lld (SM cycles), "
         "理想重叠 = min(tail,prologue) = %lld cyc\n\n",
         iters, pre, tail, prologue, crit,
         tail < prologue ? tail : prologue);

  int* flag = nullptr;
  int* out = nullptr;
  CUDA_CHECK(cudaMalloc(&flag, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out, sizeof(int)));

  int fails = 0;
  long long ideal = tail < prologue ? tail : prologue;

  // ---- Section A：固定消费端 Gd=1，扫生产端 Gu —— 看「上游占用率」如何吃掉重叠。
  printf("== Section A: Gd=1, sweep producer grid Gu ==\n");
  printf("%8s | %9s | %11s | %11s | %8s | %13s | %9s\n", "Gu", "Gu/cap",
         "no-PDL ms", "PDL ms", "speedup", "eff overlap", "ov/ideal");
  printf("---------+-----------+-------------+-------------+----------+"
         "---------------+----------\n");
  const int Gu_list[] = {1, 2, 4, 8, 16, 32, 64, 132, 264, 528, 1056, 1584};
  for (int i = 0; i < (int)(sizeof(Gu_list) / sizeof(Gu_list[0])); ++i) {
    int Gu = Gu_list[i];
    if (Gu > cap) break;
    float t_no, t_pdl, sp;
    long long eov;
    int f = 0;
    pair_point(iters, Gu, 1, B, pre, tail, prologue, flag, out, &t_no, &t_pdl,
               &sp, &eov, &f);
    fails += f;
    printf("%8d | %8.2fx | %11.4f | %11.4f | %7.2fx | %13lld | %8.2fx\n", Gu,
           (double)Gu / cap, t_no, t_pdl, sp, eov, (double)eov / ideal);
  }

  // ---- Section B：固定生产端 Gu=cap/2，扫消费端 Gd —— 看「下游 grid 大小」是否重要。
  int Gu_b = cap / 2;
  printf("\n== Section B: Gu=%d (=cap/2), sweep consumer grid Gd (空闲槽位=%d) ==\n",
         Gu_b, cap - Gu_b);
  printf("%8s | %9s | %11s | %11s | %8s | %13s | %9s\n", "Gd", "Gd/free",
         "no-PDL ms", "PDL ms", "speedup", "eff overlap", "ov/ideal");
  printf("---------+-----------+-------------+-------------+----------+"
         "---------------+----------\n");
  const int Gd_list[] = {1, 2, 4, 8, 16, 32, 64, 132, 264, 528, 1056};
  for (int i = 0; i < (int)(sizeof(Gd_list) / sizeof(Gd_list[0])); ++i) {
    int Gd = Gd_list[i];
    if (Gu_b + Gd > cap) break;
    float t_no, t_pdl, sp;
    long long eov;
    int f = 0;
    pair_point(iters, Gu_b, Gd, B, pre, tail, prologue, flag, out, &t_no, &t_pdl,
               &sp, &eov, &f);
    fails += f;
    printf("%8d | %8.2fx | %11.4f | %11.4f | %7.2fx | %13lld | %8.2fx\n", Gd,
           (double)Gd / (cap - Gu_b), t_no, t_pdl, sp, eov,
           (double)eov / ideal);
  }

  // ---- Section C：上游占满 Gu=cap，扫下游 prologue —— 看满占用下重叠的「天花板」。
  printf("\n== Section C: 上游占满 Gu=cap=%d, Gd=1, sweep prologue ==\n", cap);
  printf("%10s | %9s | %11s | %11s | %8s | %13s | %9s\n", "prologue",
         "pl/tail", "no-PDL ms", "PDL ms", "speedup", "eff overlap", "ov/ideal");
  printf("-----------+-----------+-------------+-------------+----------+"
         "---------------+----------\n");
  const long long pl_list[] = {0, 20000, 40000, 80000, 160000, 320000};
  for (int i = 0; i < (int)(sizeof(pl_list) / sizeof(pl_list[0])); ++i) {
    long long pl = pl_list[i];
    long long idl = tail < pl ? tail : pl;
    float t_no, t_pdl, sp;
    long long eov;
    int f = 0;
    pair_point(iters, cap, 1, B, pre, tail, pl, flag, out, &t_no, &t_pdl, &sp,
               &eov, &f);
    fails += f;
    printf("%10lld | %8.2fx | %11.4f | %11.4f | %7.2fx | %13lld | %8.2f%s\n", pl,
           (double)pl / tail, t_no, t_pdl, sp, eov, idl ? (double)eov / idl : 0.0,
           idl ? "x" : "-");
  }

  printf("\ncorrectness: fail counters summed over all points = %d "
         "(0 = 每轮 consumer 都读到 producer 的当前 gen)\n", fails);

  printf("\n>> 读法：\n");
  printf(">> Section A：Gd=1 时，上游 Gu 远低于 cap 时重叠 ≈ 理想 min(tail,prologue)；\n");
  printf(">>   Gu 逼近 cap（上游占满 occupancy）时有效重叠明显缩水——收益被 ***上游占用率*** 吃掉。\n");
  printf(">> Section B：Gu=cap/2 固定、下游 Gd 从 1 扫到空闲槽位数，重叠几乎不变：\n");
  printf(">>   只要 总占用 (Gu+Gd) ≤ cap，下游 grid 多大都不影响重叠。约束在「上游给不给余量」。\n");
  printf(">> Section C：上游占满 cap 时，重叠不随 prologue 无限增长，而是饱和在一个天花板\n");
  printf(">>   （约理想值的 ~0.7×）——下游 block 与满负荷上游争 SM，prologue 也得慢慢跑。\n");
  printf(">> 含义：PDL 帮助的是「上游没有占满 SM」的情形；生产 megakernel 恰恰相反——\n");
  printf(">>   它用一个常驻 kernel **占满全部 SM**（见 docs/04 §5、MPK §5.1），所以在单 kernel 内\n");
  printf(">>   用去中心化 event queue 做 task 级重叠，而不是靠 PDL 跨 kernel 挤 SM 槽位。\n");

  CUDA_CHECK(cudaFree(flag));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
