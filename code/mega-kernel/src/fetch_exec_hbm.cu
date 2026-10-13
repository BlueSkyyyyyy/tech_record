// fetch_exec_hbm.cu — MegaKernel 专题 P55-1 小实验：
//   **取 task 与「吃 HBM 的」task 执行争带宽：描述符取数还能被藏住吗？**
//
// 动机 / 接续（补齐 §43.7 P54-1 明留的第一条尾巴）：
//   §43（P54-1）第一次把「取 task」与「执行 task」放进同一个 worker 循环，得到决定性结论
//   `D=3 ≈ max(fetch_only, exec_only)`（取数被**完全**藏进执行）。但那一条成立的前提是：
//   **`exec_task` 是纯 ALU 的串行 LCG「计算压舱」，完全不吃 HBM**（§43.7 自己点名：「真实 task
//   （GEMM/RMSNorm）会同时吃 HBM/TC，届时『取数与执行是否争带宽』需另测」）。
//
//   本机模型换成**数据面吃 HBM 的执行**：每条描述符除了「控制面折叠」（读描述符本身，走 SMEM 环），
//   还要从一块 ≫L2 的流式数组里读 `EDW` 个字（**数据面 HBM 流量**）。于是「取 task」的 TMA bulk
//   （控制面，源也是 HBM）与「执行 task」的数据面读**共用同一条 HBM 带宽**。三个问题：
//     Q1（§A）正确性：取+执行环在多 worker 下 checksum 逐位一致？
//     Q2（§B）**固定计算、扫数据面 EDW**：`D=3` 是趋 `max(fetch,exec)`（§43，藏住）还是
//             `fetch+exec`（取数与执行在带宽上**排队相加**）？
//     Q3（§C）**固定数据面、扫计算 R**：从「执行内存受限」（`D=3=fetch+exec`）到「执行计算受限」
//             （`D=3=max`）的**交叉点**在哪？即「取数被藏住」到底需要 task 有多重？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：全线程协作 16 B `cp.async` 进 SMEM **单缓冲** `cp_async_wait<0>`
//     （`persistent_kernel.cuh:1063-1078`、`:1076-1077`）；论文 §5.3「Pre-fetching task descriptions」
//     （行 1364-1368）/ paged shared memory（行 1313-1325）；`TaskDesc` 352 B（论文 §6.1 行 1365）。
//   - 真实 task 一定同时吃 HBM（GEMM 读 A/B、RMSNorm 读整行）——本节把这条「数据面」加进来。
//
// 机制：
//   G 个常驻 worker block；全局 `NDESC` 条描述符（`DW=88` 字 = 352 B，数组 ≈ 132 MB ≫ L2）；
//   另有全局流式数据数组 `NDESC·EDW` 个字（`EDW` = 每条 task 的数据面字数，数组 ≫ L2）。
//   block `b` 处理 batch `bb = b + t·G`，每 batch `K` 条连续描述符；取数载体 = TMA bulk + 硬件
//   mbarrier，环形缓冲 D 槽（SMEM = D·K·352 B，与 `EDW` 无关）。`D=1` = 快照单缓冲；`D≥2` 预取。
//   执行 = 控制面 `base_fold`（读描述符）+ **数据面读 `EDW` 个字**（HBM）+ `R` 轮 LCG 计算压舱。
//   `do_fetch=0` 关掉 TMA（只跑执行，隔离数据面 HBM 时间）；`EDW=0` 退化为 §43 的纯计算形态。
//
// 安全：数组预写、无生产者；mbarrier 期望字节恒被 complete_tx 满足；脚本带 timeout。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/fetch_exec_hbm.cu -o /tmp/fetch_exec_hbm
// 用法：
//   ./fetch_exec_hbm run <G> <NDESC> <DW> <EDW> <K> <D> <R> <reps> [lanes] [do_fetch]
//   ./fetch_exec_hbm bench

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

// 「执行」一条 task：
//   (1) 控制面折叠 base_fold：读描述符 DW 个字（走 SMEM 环，不是 HBM 主流量）；
//   (2) **数据面**：从 data 流式读 EDW 个字（HBM），求和；
//   (3) R 轮串行 LCG 计算压舱（R=0 即纯数据面/控制面形态）。
// task_id = bb*K+j 决定数据面切片 data[task_id*EDW + e)。
// 「执行」一个 batch：控制面折叠（读本批 K 条描述符，走 SMEM 环）+ **数据面**（整块连续流式读
// 本批 `K·EDW` 个字，走 HBM，块内各线程 grid-stride ⇒ coalesced 高带宽）+ R 轮 LCG 计算压舱。
// 每个 batch 的数据面区域连续（data + bb·K·EDW），故是真实 task 从 HBM 拉数据的形态。
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
__global__ void fetch_exec_hbm_kernel(int NDESC, int DW, int EDW, int K, int D, int R,
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
  double gbps_desc;  // 仅描述符字节
  double gbps_total; // 描述符 + 数据面字节
};

static Res run_fe(int G, long long NDESC, int DW, int EDW, int K, int D, int R, int reps,
                  int nlanes, int do_fetch, int verify) {
  Res res = {0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (do_fetch && smem_bytes > g_optin) {
    printf("  EDW=%-5d K=%-4d D=%-3d R=%-6d SMEM=%6zuB  SKIP (exceeds optin)\n", EDW, K, D,
           R, smem_bytes);
    return res;
  }
  size_t req = do_fetch ? smem_bytes : (size_t)K * DW * 4 + 16;
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(fetch_exec_hbm_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));
  ensure_desc(NDESC, DW);
  ensure_data(NDESC, EDW);

  unsigned long long got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_min = 1e30f;  // 取 min-of-reps，压低邻居/热噪声
  for (int w = 0; w < 2; ++w) {  // 2 次不计时 warmup（时钟/TLB/指令缓存）
    fetch_exec_hbm_kernel<<<G, nlanes, req>>>(NDESC, DW, EDW, K, D, R, do_fetch, g_descs,
                                              g_data, g_sink);
    CK(cudaDeviceSynchronize());
  }
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    fetch_exec_hbm_kernel<<<G, nlanes, req>>>(NDESC, DW, EDW, K, D, R, do_fetch, g_descs,
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
  res.gbps_total = (bytes_desc + bytes_data) / sec / 1e9;
  printf("  EDW=%-5d K=%-4d D=%-3d R=%-6d fetch=%d  %8.4f ms/rep  desc=%7.1f GB/s  "
         "tot=%7.1f GB/s  got=%llu\n",
         EDW, K, D, R, do_fetch, res.ms, res.gbps_desc, res.gbps_total, res.got);
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
    int D = argc > 7 ? atoi(argv[7]) : 3;
    int R = argc > 8 ? atoi(argv[8]) : 0;
    int reps = argc > 9 ? atoi(argv[9]) : 3;
    int nlanes = argc > 10 ? atoi(argv[10]) : LANES;
    int fetch = argc > 11 ? atoi(argv[11]) : 1;
    run_fe(G, N, dw, edw, K, D, R, reps, nlanes, fetch, 1);
    return 0;
  }

  printf("\n===== §A 正确性：D=1..4，EDW=8，R=0，多 worker，sink 逐位一致 =====\n");
  unsigned long long refA = 0;
  for (int D : {1, 2, 3, 4}) {
    Res g = run_fe(NSM, 8192, DW, 8, 32, D, 0, 1, LANES, 1, 0);
    if (D == 1) refA = g.got;
    printf("      -> D=%d %s reference   (expect=%llu)\n", D,
           (g.got == refA) ? "==" : "!=", refA);
  }

  // 纯描述符取数的独立参照（EDW=0，R=0，D=3）：cross-EDW 恒定
  printf("\n===== [参考] 纯描述符 fetch_only（EDW=0, R=0, D=3）=====\n");
  Res FETCH = run_fe(NSM, NDESC, DW, 0, 128, 3, 0, 5, LANES, 1, 0);
  double fconst = FETCH.ms;

  printf("\n===== §B 固定纯数据面(R=0)、扫 EDW：D=3 趋 max(fetch,exec) 还是 fetch+exec？ =====\n");
  printf("  %-6s %11s %11s %11s %11s |  p/max  p/(f+e)  s/exec  D3-exec\n", "EDW",
         "exec_only", "D=1(ms)", "D=3(ms)", "fetch_ref");
  for (int EDW : {0, 16, 64, 256, 1024}) {
    Res e = run_fe(NSM, NDESC, DW, EDW, 128, 1, 0, 5, LANES, 0, 0);  // exec_only（无 TMA）
    Res s = run_fe(NSM, NDESC, DW, EDW, 128, 1, 0, 5, LANES, 1, 0);  // D=1 serial
    Res p = run_fe(NSM, NDESC, DW, EDW, 128, 3, 0, 5, LANES, 1, 0);  // D=3 prefetch
    double pmax = p.ms / (fconst > e.ms ? fconst : e.ms);
    double pfe = p.ms / (fconst + e.ms);
    printf("  %-6d %11.4f %11.4f %11.4f %11.4f | %6.3f %7.3f %7.3f %8.4f\n", EDW, e.ms,
           s.ms, p.ms, fconst, pmax, pfe, s.ms / e.ms, p.ms - e.ms);
  }

  printf("\n===== §C 固定数据面 EDW=256、扫计算 R：从内存受限到计算受限的交叉 =====\n");
  printf("  %-8s %11s %11s %11s |  p/max  p/(f+e)  s/exec  D3-exec\n", "R", "exec_only",
         "D=1(ms)", "D=3(ms)");
  for (int R : {0, 32, 128, 512, 2048, 8192, 32768}) {
    Res e = run_fe(NSM, NDESC, DW, 256, 128, 1, R, 5, LANES, 0, 0);
    Res s = run_fe(NSM, NDESC, DW, 256, 128, 1, R, 5, LANES, 1, 0);
    Res p = run_fe(NSM, NDESC, DW, 256, 128, 3, R, 5, LANES, 1, 0);
    double pmax = p.ms / (fconst > e.ms ? fconst : e.ms);
    double pfe = p.ms / (fconst + e.ms);
    printf("  %-8d %11.4f %11.4f %11.4f | %6.3f %7.3f %7.3f %8.4f\n", R, e.ms, s.ms, p.ms,
           pmax, pfe, s.ms / e.ms, p.ms - e.ms);
  }

  printf("\n===== §D 取 task 字节的『加价』：D=3-exec 的绝对差 vs fetch 字节/HBM =====\n");
  printf("  %-6s %14s %14s %14s\n", "EDW", "D=3-exec(ms)", "exec_only(ms)", "D3-exec/exec");
  for (int EDW : {0, 16, 64, 256, 1024}) {
    Res e = run_fe(NSM, NDESC, DW, EDW, 128, 1, 0, 5, LANES, 0, 0);
    Res p = run_fe(NSM, NDESC, DW, EDW, 128, 3, 0, 5, LANES, 1, 0);
    double extra = p.ms - e.ms;
    printf("  %-6d %14.4f %14.4f %14.3f\n", EDW, extra, e.ms, extra / e.ms);
  }

  if (g_descs) cudaFree(g_descs);
  if (g_data) cudaFree(g_data);
  if (g_sink) cudaFree(g_sink);
  return 0;
}
