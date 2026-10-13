// P38-1: megakernel 不能独占整机时 —— 持久化内核对共存负载的干扰 & Green Contexts SM 分区隔离
// 目标卡 H100 / sm90。CUDA 13.2 runtime API (cudaDevSmResourceSplit / cudaGreenCtxCreate).
//
// 动机：megakernel = 单次 launch 占用整机（MPK §5.1 固定 worker/scheduler 的 SM 分配）。
//   但真实部署（serving）要与其它负载共享 GPU。本节量化：
//     (A) 整机 persistent kernel 会「饿死」同机 kernel（其 block 拿不到 SM 槽位）；
//     (B) Hopper Green Contexts 的 SM 分区能否给共存负载硬隔离；
//     (C) 流优先级不能替代分区（运行中的 block 不被抢占）；
//     (D) 分区的代价：persistent 自身吞吐随分到的 SM 数缩放。
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("[FAIL] %s -> %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

static double now_ms() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// 整机 persistent kernel：网格常驻，rounds 轮 grid-stride 求和（访存为主）。
// block0/thread0 置 started；结束写 start/end globaltimer。
__global__ void persist_kernel(const float* __restrict__ buf, long n, int rounds,
                               volatile unsigned int* started,
                               unsigned long long* times) {
  float acc = 0.f;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    times[0] = gtimer();
    *started = 1u;
  }
  for (int r = 0; r < rounds; ++r) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
      acc += buf[i];
    }
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0 && acc == -123456.f) times[1] = 0; // 防优化
  if (blockIdx.x == 0 && threadIdx.x == 0) times[1] = gtimer();
}

// victim：多 block grid-stride 求和，最后一个完成的 block 置 done（mapped pinned）。
__global__ void victim_kernel(const float* __restrict__ buf, long n,
                              unsigned int* counter, volatile unsigned int* done,
                              int Gv) {
  float acc = 0.f;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x * blockDim.x) {
    acc += buf[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (acc == -123456.f) printf("x"); // 防优化（几乎不可能命中）
    unsigned int c = atomicAdd(counter, 1u);
    if (c == (unsigned int)Gv - 1u) { __threadfence_system(); *done = 1u; }
  }
}

struct Gc {
  cudaDevResource smr_full;
  cudaExecutionContext_t ctx[2] = {nullptr, nullptr};
  cudaStream_t stream[2] = {nullptr, nullptr};
  unsigned int sm[2] = {0, 0};
};

// 把整机 SM 资源按 smA 切成两组，建两个 green context（A=smA, B=其余，用 backfill）。
static int make_gc(int dev, unsigned int smA, Gc* g) {
  CK(cudaDeviceGetDevResource(dev, &g->smr_full, cudaDevResourceTypeSm));
  unsigned int total = g->smr_full.sm.smCount;
  unsigned int align = g->smr_full.sm.smCoscheduledAlignment ? g->smr_full.sm.smCoscheduledAlignment : 1;
  if (smA == 0 || smA >= total) smA = (total / 2 / align) * align;
  cudaDevSmResourceGroupParams params[2];
  memset(params, 0, sizeof(params));
  params[0].smCount = smA; params[0].coscheduledSmCount = align;
  params[1].smCount = 0; params[1].coscheduledSmCount = align;
  params[1].flags = cudaDevSmResourceGroupBackfill;
  cudaDevResource groups[2];
  cudaDevResource remainder; memset(&remainder, 0, sizeof(remainder));
  cudaError_t se = cudaDevSmResourceSplit(groups, 2, &g->smr_full, &remainder, 0, params);
  if (se != cudaSuccess) { printf("[FAIL] split smA=%u -> %s\n", smA, cudaGetErrorString(se)); return 1; }
  for (int i = 0; i < 2; i++) {
    cudaDevResourceDesc_t desc;
    CK(cudaDevResourceGenerateDesc(&desc, &groups[i], 1));
    CK(cudaGreenCtxCreate(&g->ctx[i], desc, dev, 0));
    CK(cudaExecutionCtxStreamCreate(&g->stream[i], g->ctx[i], cudaStreamNonBlocking, 0));
    cudaDevResource got;
    CK(cudaExecutionCtxGetDevResource(g->ctx[i], &got, cudaDevResourceTypeSm));
    g->sm[i] = got.sm.smCount;
  }
  printf("  [gc] total=%u align=%u -> A=%u SM, B=%u SM\n", total, align, g->sm[0], g->sm[1]);
  return 0;
}
static void destroy_gc(Gc* g) {
  for (int i = 0; i < 2; i++) if (g->ctx[i]) cudaExecutionCtxDestroy(g->ctx[i]);
}

int main(int argc, char** argv) {
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  long MB = (argc > 1) ? atol(argv[1]) : 256;
  int rounds = (argc > 2) ? atoi(argv[2]) : 64;
  printf("=== P38-1 persistent kernel 共存干扰 / Green Contexts 分区 ===\n");
  printf("device=%s sm_%d%d SMs=%d  buf=%ldMB rounds=%d\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount, MB, rounds);

  long n = MB * 1024L * 1024L / 4;
  long nv = (MB / 4) * 1024L * 1024L / 4; // victim = 1/4 缓冲
  float *buf;
  CK(cudaMalloc(&buf, n * sizeof(float)));
  CK(cudaMemset(buf, 1, n * sizeof(float)));

  unsigned int *counter;
  CK(cudaMalloc(&counter, sizeof(unsigned int)));
  unsigned long long *times;
  CK(cudaMalloc(&times, 2 * sizeof(unsigned long long)));

  volatile unsigned int *started_map, *done_map;
  CK(cudaHostAlloc((void**)&started_map, sizeof(unsigned int), cudaHostAllocMapped));
  CK(cudaHostAlloc((void**)&done_map, sizeof(unsigned int), cudaHostAllocMapped));

  int maxBlocks; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, victim_kernel, 256, 0));
  int persistBlocks; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&persistBlocks, persist_kernel, 256, 0));
  int cap = persistBlocks * prop.multiProcessorCount;
  printf("occupancy: victim %d blk/SM, persist %d blk/SM -> 整机 cap=%d blocks\n",
         maxBlocks, persistBlocks, persistBlocks * prop.multiProcessorCount);

  cudaStream_t sp, sv;
  CK(cudaStreamCreate(&sp)); CK(cudaStreamCreate(&sv));

  const int Gv = 264; // victim grid

  // 单次 victim：reset done → launch → host 轮询 done（时间上限 3s）
  auto run_once = [&](cudaStream_t s, int Gv_, double* out_ms) -> int {
    *done_map = 0;
    CK(cudaMemsetAsync(counter, 0, sizeof(unsigned int), s));
    double t0 = now_ms();
    victim_kernel<<<Gv_, 256, 0, s>>>(buf, nv, counter, done_map, Gv_);
    while (*done_map == 0) {
      if (now_ms() - t0 > 3000.0) { *out_ms = -1; return 1; }
    }
    *out_ms = now_ms() - t0;
    return 0;
  };
  // 预热 + 取 min（host 墙钟含 launch 提交，量级读）
  auto run_victim = [&](cudaStream_t s, double* out_ms) -> int {
    for (int i = 0; i < 5; i++) { double x; if (run_once(s, Gv, &x)) return 1; }
    double best = 1e30;
    for (int i = 0; i < 5; i++) { double x; if (run_once(s, Gv, &x)) return 1; if (x < best) best = x; }
    *out_ms = best; return 0;
  };
  auto wait_started = [&](double limit_ms) -> bool {
    double t0 = now_ms();
    while (*started_map == 0) { if (now_ms() - t0 > limit_ms) return false; }
    return true;
  };
  // 干扰测量：每次重新起一个 persistent（占满其网格），随后 launch 一次 victim 并计时，
  // 重复 3 次取 min（只有「persistent 还在跑」的那次才可能被饿死）。
  auto measure_under = [&](cudaStream_t spersist, cudaStream_t victim, cudaExecutionContext_t ctx,
                           int Gp, double* out_ms, double* out_persist_ms) -> int {
    double best = 1e30; int any_to = 0; double pbest = 1e30;
    for (int k = 0; k < 3; k++) {
      *started_map = 0;
      CK(cudaMemsetAsync(times, 0, 2 * sizeof(unsigned long long), spersist));
      persist_kernel<<<Gp, 256, 0, spersist>>>(buf, n, rounds, started_map, times);
      if (!wait_started(2000.0)) return 1;
      double x = 0; int to = run_once(victim, Gv, &x);
      if (ctx) CK(cudaExecutionCtxSynchronize(ctx)); else CK(cudaStreamSynchronize(spersist));
      unsigned long long th[2]; CK(cudaMemcpy(th, times, sizeof(th), cudaMemcpyDeviceToHost));
      double pms = (th[1] - th[0]) / 1e6;
      if (pms > 0 && pms < pbest) pbest = pms;
      if (to) { any_to = 1; continue; }
      if (x < best) best = x;
    }
    *out_ms = any_to && best > 1e29 ? -1.0 : best;
    if (out_persist_ms) *out_persist_ms = pbest;
    return 0;
  };

  printf("\n[Section A] 整机 persistent 对同机 kernel 的干扰（无隔离，默认 context 两条流）\n");
  {
    double base = 0; if (run_victim(sv, &base)) return 1;
    printf("  A0 victim alone (baseline, min/5)         : %.3f ms\n", base);

    for (int Gp : {132, cap}) {
      double v2 = 0, pms = 0; if (measure_under(sp, sv, nullptr, Gp, &v2, &pms)) return 1;
      printf("  A1 Gp=%-4d（%s）%s | victim=%.3f ms", Gp, Gp >= cap ? "占满整机" : "仅 1 blk/SM",
             Gp >= cap ? "[被饿死→等 persistent 结束]" : "[有空槽位可共驻]", v2);
      printf("  (%.1fx baseline) | persist %.2f ms\n", v2 / base, pms);
    }
  }

  printf("\n[Section B] Green Contexts SM 分区隔离（A 放 persistent，B 放 victim）\n");
  {
    Gc g;
    if (make_gc(dev, 64, &g) != 0) { printf("  gc 创建失败，跳过 B\n"); }
    else {
      cudaStream_t a2;
      CK(cudaExecutionCtxStreamCreate(&a2, g.ctx[0], cudaStreamNonBlocking, 0));
      double vB = 0; if (run_victim(g.stream[1], &vB)) return 1;
      printf("  B0 victim alone in gcB(%u SM)              : %.3f ms (baseline)\n", g.sm[1], vB);

      int capA = persistBlocks * g.sm[0];
      double v1 = 0, pms = 0; if (measure_under(g.stream[0], g.stream[1], g.ctx[0], capA, &v1, &pms)) return 1;
      printf("  B1 persist@A(%u SM 占满) + victim@B(隔离)  : victim=%.3f ms (%.2fx baseline) | persist %.2f ms\n",
             g.sm[0], v1, v1 / vB, pms);

      double v2 = 0; if (measure_under(g.stream[0], a2, g.ctx[0], capA, &v2, &pms)) return 1;
      printf("  B2 persist@A + victim@A(同分区,另流,对照)  : victim=%.3f ms (%.2fx baseline) | persist %.2f ms\n",
             v2, v2 / vB, pms);
      cudaStreamDestroy(a2);
      destroy_gc(&g);
    }
  }

  printf("\n[Section C] 流优先级不能替代分区（默认 context，persist 低优 / victim 高优）\n");
  {
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    printf("  priority range: low=%d high=%d\n", lo, hi);
    cudaStream_t sp_lo, sv_hi;
    CK(cudaStreamCreateWithPriority(&sp_lo, cudaStreamNonBlocking, lo));
    CK(cudaStreamCreateWithPriority(&sv_hi, cudaStreamNonBlocking, hi));
    double vlo = 0; if (run_victim(sv_hi, &vlo)) return 1;
    printf("  C0 victim(高优) alone                       : %.3f ms\n", vlo);
    double v1 = 0, pms = 0; if (measure_under(sp_lo, sv_hi, nullptr, cap, &v1, &pms)) return 1;
    printf("  C1 persist(低优,占满) + victim(高优)        : victim=%.3f ms (%.2fx baseline) | persist %.2f ms\n",
           v1, v1 / vlo, pms);
    cudaStreamDestroy(sp_lo); cudaStreamDestroy(sv_hi);
  }

  printf("\n[Section D] 分区的代价：persistent 自身吞吐 vs 分到的 SM 数\n");
  {
    for (unsigned int smA : {32u, 64u, 96u}) {
      Gc g;
      if (make_gc(dev, smA, &g) != 0) continue;
      int Gp = persistBlocks * g.sm[0];
      CK(cudaMemsetAsync(times, 0, 2 * sizeof(unsigned long long), g.stream[0]));
      persist_kernel<<<Gp, 256, 0, g.stream[0]>>>(buf, n, rounds, started_map, times);
      CK(cudaExecutionCtxSynchronize(g.ctx[0]));
      unsigned long long th[2]; CK(cudaMemcpy(th, times, sizeof(th), cudaMemcpyDeviceToHost));
      double persist_ms = (th[1] - th[0]) / 1e6;
      double gbps = (double)n * 4 * rounds / (persist_ms / 1e3) / 1e9;
      printf("  D smA=%-4u (gcA=%u SM) grid=%-5d persist=%.2f ms  %6.1f GB/s\n",
             smA, g.sm[0], Gp, persist_ms, gbps);
      destroy_gc(&g);
    }
  }

  cudaFree(buf); cudaFree(counter); cudaFree(times);
  cudaFreeHost((void*)started_map); cudaFreeHost((void*)done_map);
  cudaStreamDestroy(sp); cudaStreamDestroy(sv);
  printf("\nOK\n");
  return 0;
}
