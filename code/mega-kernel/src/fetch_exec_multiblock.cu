// fetch_exec_multiblock.cu — MegaKernel 专题 P57-1 小实验：
//   **取 task × 吃 HBM 的执行，跨「多 block/SM」维度：小 SMEM 环换高占用率，能同时喂满数据面吗？**
//
// 动机 / 接续（补齐 §45.7 P56-1 明留的第一条尾巴）：
//   §45（P56-1）把「取 task」与「吃 HBM 的执行」推到饱和，得到两个互相矛盾的最优：
//     - **数据面**（HBM 流式读）要 **多线程**：单 block `T=256→1024` 把读带宽从 ~1.2 TB/s 抬到
//       ~2983 GB/s（H100 HBM3 ≈89%）；
//     - **控制面**（描述符 TMA 取数）要 **少线程**：`T=256→1024` 由 1791→891 GB/s（多线程只增
//       `mbar_wait` 自旋与 `__syncthreads` 宽度）。
//   单个 worker block 只有一个 `blockDim`，故「要兼顾只能 warp specialization」。但 §45 的占用率被
//   SMEM 环锁在 **1 block/SM**（`K=128,D=2` ⇒ 88 KB/block），故它自陈〔§45.7〕：
//     「若要测『多 block/SM 争 HBM』需另配小 SMEM 环，但小 K 会让控制面 TMA 低效（§41/P52），
//       两者不可兼得——留作后续」。
//
//   本节把「**小 SMEM 环 ⇒ 多 block/SM**」这条路走通，问三件事：
//     Q1（§B）**数据面饱和的旋钮到底是「每 SM 总线程数」还是「每 block 线程数」**？同样
//             `B×T=1024` 线程/SM 下，`B=1,T=1024` 与 `B=8,T=128` 的读带宽是否相同？（若相同 ⇒
//             拆多块可替代单块大 T ⇒ 给「控制面要少线程」让路）。
//     Q2（§C）**小 K（小环、高占用率）是否真的让控制面取数变慢**？（P52 §C 断言小 K 因每批一次
//             mbarrier 等待 + 两次 `__syncthreads` 的固定开销而低效——在本节的多 worker 口径复核。）
//     Q3（§D）在**多 block/SM + 小 K** 上，预取 `D=2` 是否仍把控制面藏进数据面（`p/max→1`）？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：全线程协作 16 B `cp.async` 进 SMEM **单缓冲** `cp_async_wait<0>`
//     （`persistent_kernel.cuh:1063-1078`、`:1076-1077`）；论文 §5.3「Pre-fetching task descriptions」
//     （行 1364-1368）/ paged shared memory（行 1313-1325）；`TaskDesc` 352 B（论文 §6.1 行 1365）。
//   - worker 数按物理 SM（论文 §5.1 行 1168）——但**每 SM 常驻几个 worker / 环占多少 SMEM 是可配的**
//     （§41 §E 已证 SMEM 是软约束、可换 worker 数），故「多 block/SM」是真实的部署旋钮。
//
// 机制（与 §44/§45 同骨架，只把占用率当独立维度）：
//   G = NSM·B 个常驻 worker block，每 block T 线程；全局 `NDESC` 条描述符（`DW=88` 字 = 352 B，
//   数组 ≫ L2）；全局流式数据数组 `NDESC·EDW` 个字（≫ L2）。block `b` 处理 batch `bb = b + t·G`，
//   每 batch `K` 条连续描述符；取数载体 = TMA bulk + 硬件 mbarrier，环形缓冲 D 槽
//   （SMEM = D·K·352 B）。`D=1` = 快照单缓冲；`D≥2` 预取。执行 = 控制面 `base_fold`（读描述符）
//   + **数据面读**（HBM）+ `R` 轮 LCG 计算压舱。
//   **小 `K` ⇒ 小 SMEM 环 ⇒ 多 block/SM ⇒ 高占用率**；`do_fetch=0` 关掉 TMA（只跑执行）；
//   `EDW=0` 退化为纯控制面形态。
//
// 安全：数组预写、无生产者；mbarrier 期望字节恒被 complete_tx 满足；脚本带 timeout。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/fetch_exec_multiblock.cu -o /tmp/fetch_exec_multiblock
// 用法：
//   ./fetch_exec_multiblock run <G> <NDESC> <DW> <EDW> <K> <D> <R> <reps> [lanes] [do_fetch]
//   ./fetch_exec_multiblock bench

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t _e = (x);                                                             \
    if (_e != cudaSuccess) {                                                          \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));   \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

#define MAXD 64

__host__ __device__ __forceinline__ uint32_t mix(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 硬件 mbarrier + TMA bulk 1D ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "MBW_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni MBD_%=;\n"
      "bra.uni MBW_%=;\n"
      "MBD_%=:\n"
      "}\n" ::"r"(a),
      "r"(phase));
}
__device__ __forceinline__ void bulk_1d(void* smem_ptr, const void* gmem_ptr,
                                        uint32_t bytes, uint64_t* bar) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], "
      "%2, [%3];\n" ::"r"(s),
      "l"(gmem_ptr),
      "r"(bytes),
      "r"(m)
      : "memory");
}

// 「执行」一个 batch：控制面折叠（读本批 K 条描述符，走 SMEM 环）+ **数据面**（整块连续流式读
// 本批 `K·EDW` 个字，走 HBM，块内各线程 grid-stride ⇒ coalesced）+ R 轮 LCG 计算压舱。
__device__ __forceinline__ unsigned long long exec_batch(const uint32_t* desc, int K, int DW,
                                                         const uint32_t* dbase,
                                                         int nlanes, int lane, int EDW,
                                                         int R, int bid) {
  uint32_t a = 0x9e3779b9u;
  for (int j = lane; j < K; j += nlanes)
    for (int i = 0; i < DW; ++i) a = a * 2246822519u + desc[(size_t)j * DW + i];
  uint32_t s = 0;
  const size_t totv = ((size_t)K * (size_t)EDW) >> 2;  // 16 B(uint4) 单元数
  const uint4* dv = reinterpret_cast<const uint4*>(dbase);
  for (size_t i = lane; i < totv; i += nlanes) {
    uint4 x = dv[i];
    s += x.x + x.y + x.z + x.w;
  }
  uint32_t v = a * 2654435761u + s + (uint32_t)bid;
  for (int r = 0; r < R; ++r) v = v * 1664525u + 1013904223u;
  return (unsigned long long)a + (unsigned long long)v;
}

// =====================================================================================
// 取数+执行 kernel：D 深环形缓冲，每槽 K 条描述符；载体 = TMA bulk + 硬件 mbarrier
//   do_fetch=0 => 不发/不等 TMA，纯执行（隔离执行时间，含数据面 HBM 读）
// =====================================================================================
__global__ void fetch_exec_mb_kernel(int NDESC, int DW, int EDW, int K, int D, int R,
                                     int do_fetch, const uint32_t* __restrict__ descs,
                                     const uint32_t* __restrict__ data,
                                     unsigned long long* __restrict__ sink) {
  extern __shared__ __align__(16) uint32_t smem[];
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int G = gridDim.x;
  const int nbatch = NDESC / K;
  const int bytes_per_batch = K * DW * 4;
  unsigned long long acc = 0;

  const int nt = (blockIdx.x < nbatch) ? (nbatch - blockIdx.x + G - 1) / G : 0;

  if (do_fetch) {
    if (lane == 0)
      for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
    __syncthreads();
  }

  auto issue = [&](int t) {
    int slot = t % D;
    int bb = blockIdx.x + t * G;
    const uint32_t* base = descs + (size_t)bb * K * DW;
    if (lane == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
      bulk_1d(smem + (size_t)slot * K * DW, base, (uint32_t)bytes_per_batch,
              &mbar[slot]);
    }
  };

  if (do_fetch)
    for (int t = 0; t < D && t < nt; ++t) issue(t);

  for (int t = 0; t < nt; ++t) {
    int slot = t % D;
    if (do_fetch) {
      uint32_t phase = (uint32_t)((t / D) & 1u);
      mbar_wait(&mbar[slot], phase);
    }
    const uint32_t* buf = smem + (size_t)slot * K * DW;
    int bb = blockIdx.x + t * G;
    const uint32_t* dbase = (EDW > 0) ? data + (size_t)bb * K * EDW : data;
    acc += exec_batch(buf, K, DW, dbase, nlanes, lane, EDW, R, bb);
    __syncthreads();
    if (do_fetch && t + D < nt) issue(t + D);
  }

  if (acc) atomicAdd(sink, acc);
}

// =====================================================================================
// host
// =====================================================================================
__global__ void fill_desc_kernel(uint32_t* descs, long long NDESC, int DW) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= NDESC) return;
  uint32_t* p = descs + id * DW;
  p[0] = (uint32_t)id;
  if (DW > 1) p[1] = (uint32_t)(id % 4);
  if (DW > 2) p[2] = (uint32_t)(id / 4);
  if (DW > 3) p[3] = 0u;
  for (int i = 4; i < DW; ++i) p[i] = mix((uint32_t)id, (uint32_t)i);
}
__global__ void fill_data_kernel(uint32_t* data, long long n, int seed) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  data[id] = mix((uint32_t)seed, (uint32_t)(id & 0xffffffffu)) ^ (uint32_t)(id >> 32);
}

static uint32_t* g_descs = nullptr;
static uint32_t* g_data = nullptr;
static unsigned long long* g_sink = nullptr;
static long long g_alloc = -1;
static int g_dw = -1;
static long long g_data_alloc = -1;
static int g_edw_filled = -1;

static void ensure_desc(long long NDESC, int DW) {
  if (g_alloc == NDESC && g_dw == DW) return;
  if (g_descs) cudaFree(g_descs);
  if (g_sink) cudaFree(g_sink);
  CK(cudaMalloc(&g_descs, (size_t)NDESC * DW * 4));
  CK(cudaMalloc(&g_sink, 8));
  fill_desc_kernel<<<(int)((NDESC + 255) / 256), 256>>>(g_descs, NDESC, DW);
  CK(cudaDeviceSynchronize());
  g_alloc = NDESC;
  g_dw = DW;
}

static void ensure_data(long long NDESC, int EDW) {
  long long n = (EDW > 0) ? NDESC * (long long)EDW : 1;
  if (n == g_data_alloc && EDW == g_edw_filled) return;
  if (g_data) cudaFree(g_data);
  CK(cudaMalloc(&g_data, (size_t)n * 4));
  fill_data_kernel<<<(int)((n + 255) / 256), 256>>>(g_data, n, 12345);
  CK(cudaDeviceSynchronize());
  g_data_alloc = n;
  g_edw_filled = EDW;
}

static size_t g_optin = 0;

struct Res {
  unsigned long long got;
  double ms;
  double gbps_desc;   // 仅描述符字节
  double gbps_data;   // 仅数据面字节
  double gbps_total;  // 描述符 + 数据面字节
};

static int max_blocks_per_sm(int T, size_t dyn) {
  if (dyn > 48 * 1024)
    cudaFuncSetAttribute(fetch_exec_mb_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fetch_exec_mb_kernel, T, dyn);
  if (nb <= 0 && dyn > 0) {  // 回退：按 SMEM/SM 与线程上限手算上界
    int byThr = 2048 / T;
    int bySmem = (int)(233472 / dyn);
    nb = byThr < bySmem ? byThr : bySmem;
  }
  return nb;
}

static Res run_fe(int G, long long NDESC, int DW, int EDW, int K, int D, int R, int reps,
                  int nlanes, int do_fetch, int verify) {
  Res res = {0, 0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (do_fetch && smem_bytes > g_optin) {
    printf("  EDW=%-5d K=%-4d D=%-3d R=%-6d SMEM=%6zuB  SKIP (exceeds optin)\n", EDW, K, D,
           R, smem_bytes);
    return res;
  }
  size_t req = smem_bytes;  // exec_only 也按同一环尺寸开 SMEM，保证占用率可比
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(fetch_exec_mb_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));
  ensure_desc(NDESC, DW);
  ensure_data(NDESC, EDW);

  unsigned long long got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_min = 1e30f;  // 取 min-of-reps，压低邻居/热噪声
  for (int w = 0; w < 2; ++w) {  // 2 次不计时 warmup（时钟/TLB/指令缓存）
    fetch_exec_mb_kernel<<<G, nlanes, req>>>(NDESC, DW, EDW, K, D, R, do_fetch, g_descs,
                                             g_data, g_sink);
    CK(cudaDeviceSynchronize());
  }
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    fetch_exec_mb_kernel<<<G, nlanes, req>>>(NDESC, DW, EDW, K, D, R, do_fetch, g_descs,
                                             g_data, g_sink);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < ms_min) ms_min = ms;
    CK(cudaMemcpy(&got, g_sink, 8, cudaMemcpyDeviceToHost));
  }
  res.got = got;
  res.ms = ms_min;
  double bytes_desc = (double)NDESC * DW * 4.0;
  double bytes_data = (double)NDESC * (double)EDW * 4.0;
  double sec = ms_min / 1e3;
  res.gbps_desc = bytes_desc / sec / 1e9;
  res.gbps_data = bytes_data / sec / 1e9;
  res.gbps_total = (bytes_desc + bytes_data) / sec / 1e9;
  printf("  EDW=%-5d K=%-4d D=%-3d R=%-6d G=%-6d T=%-5d fetch=%d  %8.4f ms/rep  "
         "desc=%7.1f data=%7.1f tot=%7.1f GB/s  got=%llu\n",
         EDW, K, D, R, G, nlanes, do_fetch, res.ms, res.gbps_desc, res.gbps_data,
         res.gbps_total, res.got);
  (void)verify;
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return res;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  const int NSM = prop.multiProcessorCount;
  const size_t OPTIN = prop.sharedMemPerBlockOptin;
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB  optin=%zu  SMEM/SM=%zu  maxThreads/SM=%d\n",
         dev, prop.name, NSM, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20), OPTIN,
         (size_t)prop.sharedMemPerMultiprocessor, prop.maxThreadsPerMultiProcessor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  g_optin = OPTIN;

  const int DW = 88;               // 352 B, 对齐 MPK §6.1
  const int LANES = 256;
  const long long NDESC = 393216;  // ×352 B ≈ 132 MB ≫ L2

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int G = argc > 2 ? atoi(argv[2]) : NSM;
    long long N = argc > 3 ? atoll(argv[3]) : NDESC;
    int dw = argc > 4 ? atoi(argv[4]) : DW;
    int edw = argc > 5 ? atoi(argv[5]) : 256;
    int K = argc > 6 ? atoi(argv[6]) : 128;
    int D = argc > 7 ? atoi(argv[7]) : 2;
    int R = argc > 8 ? atoi(argv[8]) : 0;
    int reps = argc > 9 ? atoi(argv[9]) : 5;
    int nlanes = argc > 10 ? atoi(argv[10]) : LANES;
    int fetch = argc > 11 ? atoi(argv[11]) : 1;
    run_fe(G, N, dw, edw, K, D, R, reps, nlanes, fetch, 1);
    return 0;
  }

  // 小 SMEM 环的主配置：K=16,D=2 ⇒ D·K·352 = 11264 B ≈ 11 KB/block ⇒ 高占用率。
  // 对照大环（§45 口径）：K=128,D=2 ⇒ 88 KB/block ⇒ 1 block/SM。
  const int KSMALL = 16, KLRG = 128, D_ = 2;
  const int SMEM_SMALL = D_ * KSMALL * DW * 4 + D_ * 8;
  const int SMEM_LRG = D_ * KLRG * DW * 4 + D_ * 8;
  printf("SMEM ring: K=%d ⇒ %d B  (maxBlocks/SM @T256=%d @T128=%d) | "
         "K=%d ⇒ %d B  (maxBlocks/SM @T256=%d)\n",
         KSMALL, SMEM_SMALL, max_blocks_per_sm(256, SMEM_SMALL), max_blocks_per_sm(128, SMEM_SMALL),
         KLRG, SMEM_LRG, max_blocks_per_sm(256, SMEM_LRG));

  printf("\n===== §A 正确性：K=%d 下 D=1..4，EDW=8，R=0，G=%d，sink 逐位一致 =====\n", KSMALL,
         NSM * 8);
  unsigned long long refA = 0;
  for (int D : {1, 2, 3, 4}) {
    Res g = run_fe(NSM * 8, 8192, DW, 8, KSMALL, D, 0, 1, 256, 1, 0);
    if (D == 1) refA = g.got;
    printf("      -> D=%d %s reference   (expect=%llu)\n", D,
           (g.got == refA) ? "==" : "!=", refA);
  }

  // B 档位：同样「每 SM 总线程数」下用不同 (B,T) 拆法；以及不同总线程数。
  struct Cfg { int B; int T; };
  const Cfg configs[] = {{1, 256}, {1, 512}, {1, 1024},
                         {2, 256}, {2, 512}, {4, 256}, {8, 128}, {8, 256}, {16, 128}};

  printf("\n===== §B 数据面(exec_only)饱和旋钮：K=%d,D=%d 扫 (B=blocks/SM, T) —— 总线程 vs 块内线程 =====\n",
         KSMALL, D_);
  printf("  %-3s %-5s %-7s %-9s %-8s %11s %11s\n", "B", "T", "G", "thr/SM", "blk/SM", "exec_only",
         "data GB/s");
  for (const Cfg& c : configs) {
    int G = NSM * c.B;
    int bps = max_blocks_per_sm(c.T, SMEM_SMALL);
    Res e = run_fe(G, NDESC, DW, 4096, KSMALL, D_, 0, 9, c.T, 0, 0);
    printf("  %-3d %-5d %-7d %-9d %-8d %11.4f %11.1f\n", c.B, c.T, G, c.B * c.T, bps, e.ms,
           e.gbps_data);
  }

  printf("\n===== §C 控制面(fetch_only, EDW=0)取数带宽：小 K vs 大 K，高占用率 vs 1 blk/SM =====\n");
  printf("  %-4s %-3s %-5s %-8s %11s %11s\n", "K", "B", "T", "blk/SM", "fetch_only", "desc GB/s");
  {
    const int Ks[] = {8, 16, 32, 128};
    for (int K : Ks) {
      size_t sm = (size_t)D_ * K * DW * 4 + D_ * 8;
      if (sm > g_optin) continue;
      // 高占用率档：取该环能达到的最大 blocks/SM（受线程上限），B 取该值
      int bT = 256;
      int bps = max_blocks_per_sm(bT, sm);
      int B = bps > 0 ? bps : 1;
      Res f = run_fe(NSM * B, NDESC, DW, 0, K, D_, 0, 9, bT, 1, 0);
      printf("  %-4d %-3d %-5d %-8d %11.4f %11.1f\n", K, B, bT, bps, f.ms, f.gbps_desc);
      // 对照：1 block/SM
      Res f1 = run_fe(NSM, NDESC, DW, 0, K, D_, 0, 9, bT, 1, 0);
      printf("  %-4d %-3d %-5d %-8d %11.4f %11.1f\n", K, 1, bT, 1, f1.ms, f1.gbps_desc);
    }
  }

  printf("\n===== §D 多 block/SM + 小 K 下预取是否仍藏住控制面：G=%d,B=8,T=256,K=%d,D=%d =====\n",
         NSM * 8, KSMALL, D_);
  printf("  %-6s %11s %11s %11s %11s |  p/max  p/(f+e)  s1/exec  p-exec\n", "EDW",
         "exec_only", "D=1(ms)", "D=2(ms)", "fetch_ref");
  {
    int G = NSM * 8, T = 256;
    Res F = run_fe(G, NDESC, DW, 0, KSMALL, D_, 0, 9, T, 1, 0);  // fetch_only reference
    double fconst = F.ms;
    for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
      Res e = run_fe(G, NDESC, DW, EDW, KSMALL, D_, 0, 9, T, 0, 0);
      Res s1 = run_fe(G, NDESC, DW, EDW, KSMALL, 1, 0, 9, T, 1, 0);
      Res p = run_fe(G, NDESC, DW, EDW, KSMALL, D_, 0, 9, T, 1, 0);
      double pmax = p.ms / (fconst > e.ms ? fconst : e.ms);
      double pfe = p.ms / (fconst + e.ms);
      printf("  %-6d %11.4f %11.4f %11.4f %11.4f | %6.3f %7.3f %8.3f %8.4f\n", EDW, e.ms,
             s1.ms, p.ms, fconst, pmax, pfe, s1.ms / e.ms, p.ms - e.ms);
    }
  }

  if (g_descs) cudaFree(g_descs);
  if (g_data) cudaFree(g_data);
  if (g_sink) cudaFree(g_sink);
  return 0;
}
