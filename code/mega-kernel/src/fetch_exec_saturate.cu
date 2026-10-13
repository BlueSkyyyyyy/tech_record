// fetch_exec_saturate.cu — MegaKernel 专题 P56-1 小实验：
//   **取 task × 吃 HBM 的 task 执行（饱和侧）：数据面把 HBM 打满时控制面还能被藏住吗？**
//
// 动机 / 接续（补齐 §44.7 P55-1 明留的第一条尾巴）：
//   §44（P55-1）把「取 task」与「吃 HBM 的执行」放进同一 worker 循环，得到 `D=3 ≈ max(fetch,exec)`
//   （取数被预取藏住、不与执行相加）。但那条结论有一个它自己都标出的**口径边界**：
//     「数据面未打满 HBM」——1 block/SM（受 SMEM 环 `D·K·352 B` 限制，见 §41/§52-1）+
//     `uint4` 流式只到 ~1.2–1.4 TB/s（描述符流单独 ~1.8 TB/s，H100 HBM3 上限更高）。
//     故 §44 是「**未饱和区间**取数仍被藏住」；**若执行把 HBM 打满，控制面字节是否转为可见争抢未测**。
//
//   本节把执行侧的 **occupancy 拉满**（缩小 SMEM 环 → 多 block/SM + 放大 grid），让数据面读**逼近
//   HBM 峰值**，再测同一组分解量：
//     Q1（§B）数据面在读带宽上是否真的**饱和**？在几 block/SM 处饱和、峰值多少 GB/s？
//     Q2（§C）**饱和后** `D=3` 是仍趋 `max(fetch,exec)`（控制面被藏）还是滑向 `fetch+exec`
//             （控制面与数据面在打满的 HBM 上**排队相加**）？
//     Q3（§D）饱和下快照的单缓冲 `D=1` 的额外代价还剩多少？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：全线程协作 16 B `cp.async` 进 SMEM **单缓冲** `cp_async_wait<0>`
//     （`persistent_kernel.cuh:1063-1078`、`:1076-1077`）；论文 §5.3「Pre-fetching task descriptions」
//     （行 1364-1368）/ paged shared memory（行 1313-1325）；`TaskDesc` 352 B（论文 §6.1 行 1365）。
//   - worker 数按物理 SM（论文 §5.1 行 1168）——但**每 SM 常驻几个 worker / 环占多少 SMEM 是可配的**
//     （§41 §E 已证 SMEM 是软约束、可换 worker 数），故「把数据面打满」是真实的部署旋钮。
//
// 机制（与 §44 同骨架，只加一个 occupancy 维度）：
//   G 个常驻 worker block；全局 `NDESC` 条描述符（`DW=88` 字 = 352 B，数组 ≫ L2）；
//   全局流式数据数组 `NDESC·EDW` 个字（≫ L2）。block `b` 处理 batch `bb = b + t·G`，每 batch `K`
//   条连续描述符；取数载体 = TMA bulk + 硬件 mbarrier，环形缓冲 D 槽（SMEM = D·K·352 B）。
//   `D=1` = 快照单缓冲；`D≥2` 预取。执行 = 控制面 `base_fold`（读描述符）+ **数据面读**（HBM）
//   + `R` 轮 LCG 计算压舱。**小 `K`（⇒ 小 SMEM 环）+ 大 `G`（`= NSM·B`）⇒ 多 block/SM ⇒ 高占用率**，
//   数据面聚合带宽随之抬高。
//   `do_fetch=0` 关掉 TMA（只跑执行，隔离数据面 HBM 时间）；`EDW=0` 退化为纯控制面形态。
//
// 安全：数组预写、无生产者；mbarrier 期望字节恒被 complete_tx 满足；脚本带 timeout。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/fetch_exec_saturate.cu -o /tmp/fetch_exec_saturate
// 用法：
//   ./fetch_exec_saturate run <G> <NDESC> <DW> <EDW> <K> <D> <R> <reps> [lanes] [do_fetch]
//   ./fetch_exec_saturate bench

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
__global__ void fetch_exec_sat_kernel(int NDESC, int DW, int EDW, int K, int D, int R,
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
    CK(cudaFuncSetAttribute(fetch_exec_sat_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));
  ensure_desc(NDESC, DW);
  ensure_data(NDESC, EDW);

  unsigned long long got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_min = 1e30f;  // 取 min-of-reps，压低邻居/热噪声
  for (int w = 0; w < 2; ++w) {  // 2 次不计时 warmup（时钟/TLB/指令缓存）
    fetch_exec_sat_kernel<<<G, nlanes, req>>>(NDESC, DW, EDW, K, D, R, do_fetch, g_descs,
                                              g_data, g_sink);
    CK(cudaDeviceSynchronize());
  }
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    fetch_exec_sat_kernel<<<G, nlanes, req>>>(NDESC, DW, EDW, K, D, R, do_fetch, g_descs,
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
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB  optin=%zu  SMEM/SM=%zu\n", dev,
         prop.name, NSM, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20), OPTIN,
         (size_t)prop.sharedMemPerMultiprocessor);
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

  // 实验配置：**大 K**（K=128 ⇒ TMA 高效，§41/P52：K 才是取 task 的主旋钮）**+ D=2**（§41 §B/
  // §42 §C：D=2 已拿满预取收益）；每个 worker block 用 **T 线程**（T 是本节的核心旋钮：数据面
  // 流式读需要线程数才喂得满 HBM，控制面 TMA 取数反而偏好少线程）。K=128,D=2 ⇒ SMEM 环
  // = 2·128·88·4 = 90112 B = 88 KB/block ⇒ ≤2 block/SM（threads 亦 ≤2）。
  const int KS = 128, DS = 2;
  const int GSAT = NSM;  // 饱和配置：G=#SM（= MPK §5.1 的 worker 数按物理 SM），T=1024
  const int TSAT = 1024;

  printf("\n===== §A 正确性：K=%d 下 D=1..4，EDW=8，R=0，G=%d，sink 逐位一致 =====\n", KS,
         GSAT);
  unsigned long long refA = 0;
  for (int D : {1, 2, 3, 4}) {
    Res g = run_fe(GSAT, 8192, DW, 8, KS, D, 0, 1, 256, 1, 0);
    if (D == 1) refA = g.got;
    printf("      -> D=%d %s reference   (expect=%llu)\n", D,
           (g.got == refA) ? "==" : "!=", refA);
  }

  printf("\n===== §B1 数据面(exec_only)读带宽随线程数饱和：K=%d,D=%d,G=%d 扫 (EDW,T) =====\n",
         KS, DS, GSAT);
  printf("  %-6s %-6s %11s %11s\n", "EDW", "T", "exec_only", "data GB/s");
  for (int EDW : {256, 1024, 4096}) {
    for (int T : {256, 512, 1024}) {
      Res e = run_fe(GSAT, NDESC, DW, EDW, KS, DS, 0, 9, T, 0, 0);
      printf("  %-6d %-6d %11.4f %11.1f\n", EDW, T, e.ms, e.gbps_data);
    }
  }

  printf("\n===== §B2 控制面(fetch_only, EDW=0)取数带宽随线程数**下降**：K=%d,D=%d,G=%d =====\n",
         KS, DS, GSAT);
  printf("  %-6s %11s %11s\n", "T", "fetch_only", "desc GB/s");
  Res FETCHT[3];
  {
    int i = 0;
    for (int T : {256, 512, 1024}) {
      FETCHT[i] = run_fe(GSAT, NDESC, DW, 0, KS, DS, 0, 9, T, 1, 0);
      printf("  %-6d %11.4f %11.1f\n", T, FETCHT[i].ms, FETCHT[i].gbps_desc);
      ++i;
    }
  }
  double fconst = FETCHT[2].ms;  // 与 §C/§D 的饱和配置 T=1024 对齐

  printf("\n===== §C 饱和态(G=%d,T=%d) 扫 EDW：D=%d 仍趋 max(fetch,exec) 还是相加？ =====\n",
         GSAT, TSAT, DS);
  printf("  %-6s %11s %11s %11s %11s |  p/max  p/(f+e)  s1/exec  p-exec\n", "EDW",
         "exec_only", "D=1(ms)", "D=2(ms)", "fetch_ref");
  for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
    Res e = run_fe(GSAT, NDESC, DW, EDW, KS, DS, 0, 9, TSAT, 0, 0);   // exec_only
    Res s1 = run_fe(GSAT, NDESC, DW, EDW, KS, 1, 0, 9, TSAT, 1, 0);   // D=1 单缓冲
    Res p = run_fe(GSAT, NDESC, DW, EDW, KS, DS, 0, 9, TSAT, 1, 0);   // D=2 预取
    double pmax = p.ms / (fconst > e.ms ? fconst : e.ms);
    double pfe = p.ms / (fconst + e.ms);
    printf("  %-6d %11.4f %11.4f %11.4f %11.4f | %6.3f %7.3f %8.3f %8.4f\n", EDW, e.ms,
           s1.ms, p.ms, fconst, pmax, pfe, s1.ms / e.ms, p.ms - e.ms);
  }

  printf("\n===== §D 饱和态单缓冲(D=1)的额外代价：D=1/D=2 与绝对缝，扫 EDW =====\n");
  printf("  %-6s %11s %11s %11s | D1/D2  D1-exec  D2-exec\n", "EDW", "exec_only", "D=1(ms)",
         "D=2(ms)");
  for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
    Res e = run_fe(GSAT, NDESC, DW, EDW, KS, DS, 0, 9, TSAT, 0, 0);
    Res s1 = run_fe(GSAT, NDESC, DW, EDW, KS, 1, 0, 9, TSAT, 1, 0);
    Res s2 = run_fe(GSAT, NDESC, DW, EDW, KS, DS, 0, 9, TSAT, 1, 0);
    printf("  %-6d %11.4f %11.4f %11.4f | %6.3f %8.4f %8.4f\n", EDW, e.ms, s1.ms, s2.ms,
           s1.ms / s2.ms, s1.ms - e.ms, s2.ms - e.ms);
  }

  if (g_descs) cudaFree(g_descs);
  if (g_data) cudaFree(g_data);
  if (g_sink) cudaFree(g_sink);
  return 0;
}
