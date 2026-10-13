// fetch_exec_overlap.cu — MegaKernel 专题 P54-1 小实验：
//   **worker「取 task」与「执行 task」的流水：描述符的取数成本能不能被 task 执行藏住？**
//
// 动机 / 接续：
//   §39–§42（P50-1..P53-1）把「worker 取 task」的**载体**（寄存器 / 16 B cp.async / TMA bulk）
//   和**缓冲预算**（批量 K × 环深 D）量透了，但**全都在「只取数、不执行 task」的口径**下：
//   消费者只做一次 `fold_words`（异或折叠）当哨兵。于是留下一个从未被直接测过的整合问题——
//   **取 task 的时间在真实 worker 循环里是不是关键路径？加预取（D≥2）能不能把它藏进 task 执行里？**
//
// 为什么这对 megakernel 重要：
//   - MPK 快照的 worker 取 task 就是**单缓冲** `cp_async_wait<0>`（`persistent_kernel.cuh:1063-1078`、
//     `:1076-1077`）——即「取完一批、等它到齐，再执行」。若取数不便宜，它就在每条 task 之间的
//     **串行缝**里。MPK §5.3 专门列了「Pre-fetching task descriptions」（快照行 1364-1368）正是
//     要把这道缝藏掉；本节第一次用**真实 task 执行负载**量出这道缝的大小与可藏性。
//   - 这与 P13-1（单 task 内 `cp.async` 载激活 → FMA 消费的流水）、P23-1（跨层权重 tile 预取流水）
//     是**同一套软件流水思想**，但对象不同：P13-1 流的是 task 内部数据、P23-1 流的是权重，
//     本节流的是**控制面**（task 描述符）。三者合起来才是 MPK §5.3 cross-task software pipelining
//     的完整画像。
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：全线程协作 16 B `cp.async` 进 SMEM **单缓冲** `cp_async_wait<0>`
//     （`persistent_kernel.cuh:1063-1078`）；`load_smem` = `cp.async.ca.shared.global.L2::128B`
//     （`tasks/common/copy_sm80.cuh:44-57`）。
//   - 论文 §5.3「Pre-fetching task descriptions」（行 1364-1368）/ paged shared memory（行 1313-1325）；
//     `TaskDesc` 352 B（论文 §6.1 行 1365）。
//
// 核心问题：
//   Q1（§A）把「取 + 执行」的环在多 worker 下跑通，checksum 与静态参考一致（正确性）？
//   Q2（§B）**扫 task 计算强度 R**：把「单缓冲串行（D=1，= 快照）」与「双/三缓冲预取（D≥2）」
//          公平对照——预取收益随 R 怎么变？
//   Q3（§C）扫环深 D：收益是否也像 §41/§42 那样 `1→2` 拿满、再深无益？
//   Q4（§D）**决定性分解**：单独量 `fetch_only`（R=0）与 `exec_only`（do_fetch=0，只跑执行），
//          看 `D=3` 究竟趋近 `max(fetch,exec)`（真交叠）还是 `fetch+exec`（没交叠）。
//
// 机制：
//   G 个常驻 worker block；全局 `NDESC` 条自描述描述符（`DW=88` 字 = 352 B，数组 ≫ L2）。
//   block `b` 处理自己的 batch 序列 `bb = b + t·G`，每 batch `K` 条连续描述符。
//   取数载体 = **TMA bulk + 硬件 mbarrier**（§40/§41 的最优载体）：lane0 发一条 bulk 把整批
//   `K·352 B` 搬进槽；环形缓冲 = D 个槽（SMEM = `D·K·352 B`）。**D=1 即"取完再执行、不预取"**，
//   正是快照单缓冲的等价物；**D≥2** 令 batch `t+D` 的取数与 batch `t` 的执行重叠。
//   `do_fetch=0` 关掉所有 TMA issue/wait（在已在 SMEM 的字节上纯执行），用来隔离「执行时间」。
//   拿到描述符后**执行真实 task**：每条描述符做一次数据相关折叠（`base_fold`，DW 次乘加）+
//   `R` 轮串行 LCG「计算压舱」（`exec_task`），把 K 条 task 的结果求和累加。R 是唯一调用来量
//   「执行时间」的旋钮（R=0 退化为 §41 的只取数形态）。
//
// 安全：数组预写、无生产者；无死循环（mbarrier 期望字节恒被 complete_tx 满足）。脚本带 timeout。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/fetch_exec_overlap.cu -o /tmp/fetch_exec_overlap
// 用法：
//   ./fetch_exec_overlap run <G> <NDESC> <DW> <K> <D> <R> <reps> [lanes] [do_fetch]
//   ./fetch_exec_overlap bench

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

// ---- 硬件 mbarrier + TMA bulk 1D（barrier.cuh / tma_2d.cuh 的形，搬到「取 task」层） ----
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

// 「执行」一条 task：数据相关折叠（base_fold，DW 次乘加）+ R 轮串行 LCG 计算压舱。
// 返回 a + v：R=0 时 = a + (a*C+12345)，仍数据相关（host 可廉价复算）；R>0 时 R 轮 LCG
// 的真实耗时即「task 执行时间」，编译器必须保留（结果被用）。
__device__ __forceinline__ unsigned long long exec_task(const uint32_t* p, int DW, int R) {
  uint32_t a = 0x9e3779b9u;
  for (int i = 0; i < DW; ++i) a = a * 2246822519u + p[i];
  uint32_t v = a * 2654435761u + 12345u;
  for (int r = 0; r < R; ++r) v = v * 1664525u + 1013904223u;
  return (unsigned long long)a + (unsigned long long)v;
}

// =====================================================================================
// 取数+执行 kernel：D 深环形缓冲，每槽 K 条描述符；载体 = TMA bulk + 硬件 mbarrier
//   block b 的 batch 序列：bb = blockIdx.x + t*G，t = 0..nt-1
//   D=1  => 取完再执行（串行，= 快照单缓冲）；D>=2 => 执行 batch t 时预取 batch t+D
//   do_fetch=0 => 不发/不等 TMA，纯执行（隔离执行时间）
// =====================================================================================
__global__ void fetch_exec_kernel(int NDESC, int DW, int K, int D, int R, int do_fetch,
                                  const uint32_t* __restrict__ descs,
                                  unsigned long long* __restrict__ sink) {
  extern __shared__ __align__(16) uint32_t smem[];
  // mbarrier 数组放在动态 SMEM 尾部（buffer 之后 8B 对齐），避免静态 SMEM 与 opt-in 上限叠加溢出
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

  // 序言：把前 min(D, nt) 批全部发出去（填满环）。D=1 时只发第 0 批。
  if (do_fetch)
    for (int t = 0; t < D && t < nt; ++t) issue(t);

  for (int t = 0; t < nt; ++t) {
    int slot = t % D;
    if (do_fetch) {
      uint32_t phase = (uint32_t)((t / D) & 1u);
      mbar_wait(&mbar[slot], phase);  // 等本批到齐
    }
    const uint32_t* buf = smem + (size_t)slot * K * DW;
    for (int j = lane; j < K; j += nlanes)  // 执行本批 K 条 task
      acc += exec_task(buf + (size_t)j * DW, DW, R);
    __syncthreads();  // 全线程执行完该槽后，才复发下一批覆盖它
    if (do_fetch && t + D < nt) issue(t + D);  // 预取/推进下一批（D=1 即取完再发）
  }

  if (acc) atomicAdd(sink, acc);
}

// =====================================================================================
// host
// =====================================================================================
__global__ void fill_kernel(uint32_t* descs, long long NDESC, int DW) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= NDESC) return;
  uint32_t* p = descs + id * DW;
  p[0] = (uint32_t)id;
  if (DW > 1) p[1] = (uint32_t)(id % 4);
  if (DW > 2) p[2] = (uint32_t)(id / 4);
  if (DW > 3) p[3] = 0u;
  for (int i = 4; i < DW; ++i) p[i] = mix((uint32_t)id, (uint32_t)i);
}

// R=0 时的参考值和（host 复算 exec_task 的 base_fold 部分）
static unsigned long long host_expected(long long NDESC, int DW) {
  unsigned long long s = 0;
  for (long long id = 0; id < NDESC; ++id) {
    uint32_t p[1024];
    int n = DW > 1024 ? 1024 : DW;
    p[0] = (uint32_t)id;
    if (n > 1) p[1] = (uint32_t)(id % 4);
    if (n > 2) p[2] = (uint32_t)(id / 4);
    if (n > 3) p[3] = 0u;
    for (int i = 4; i < n; ++i) p[i] = mix((uint32_t)id, (uint32_t)i);
    uint32_t a = 0x9e3779b9u;
    for (int i = 0; i < n; ++i) a = a * 2246822519u + p[i];
    uint32_t v = a * 2654435761u + 12345u;
    s += (unsigned long long)a + (unsigned long long)v;
  }
  return s;
}

static uint32_t* g_descs = nullptr;
static unsigned long long* g_sink = nullptr;
static long long g_alloc = -1;
static int g_dw = -1;
static unsigned long long g_exp = 0;  // 仅 R=0 有效

static void ensure_buf(long long NDESC, int DW) {
  if (g_alloc == NDESC && g_dw == DW) return;
  if (g_descs) cudaFree(g_descs);
  if (g_sink) cudaFree(g_sink);
  CK(cudaMalloc(&g_descs, (size_t)NDESC * DW * 4));
  CK(cudaMalloc(&g_sink, 8));
  fill_kernel<<<(int)((NDESC + 255) / 256), 256>>>(g_descs, NDESC, DW);
  CK(cudaDeviceSynchronize());
  g_alloc = NDESC;
  g_dw = DW;
  g_exp = host_expected(NDESC, DW);
}

static size_t g_optin = 0;

struct Res {
  unsigned long long got;
  double ms;        // per rep
  double ns_batch;  // aggregate throughput per batch
  double gbps;
};

static Res run_fe(int G, long long NDESC, int DW, int K, int D, int R, int reps,
                  int nlanes, int do_fetch, int verify) {
  Res res = {0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (do_fetch && smem_bytes > g_optin) {
    printf("  K=%-4d D=%-4d R=%-6d SMEM=%6zuB  SKIP (exceeds optin)\n", K, D, R,
           smem_bytes);
    return res;
  }
  size_t req = do_fetch ? smem_bytes : (size_t)K * DW * 4 + 16;  // do_fetch=0 也要够放一个槽
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(fetch_exec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)req));
  ensure_buf(NDESC, DW);

  unsigned long long got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    fetch_exec_kernel<<<G, nlanes, req>>>(NDESC, DW, K, D, R, do_fetch, g_descs, g_sink);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms_accum += ms;
    CK(cudaMemcpy(&got, g_sink, 8, cudaMemcpyDeviceToHost));
  }
  res.got = got;
  res.ms = ms_accum / reps;
  res.ns_batch = (res.ms / 1e3) / (double)(NDESC / K) * 1e9;
  res.gbps = (double)NDESC * (double)DW * 4.0 * (double)reps / (ms_accum / 1e3) / 1e9;
  if (verify && do_fetch && R == 0) {
    int bad = (got == g_exp) ? 0 : 1;
    printf("  K=%-4d D=%-4d R=%-6d SMEM=%6zuB  bad=%d  ~%8.3f ns/batch  %7.1f GB/s  "
           "(%.3f ms/rep)\n",
           K, D, R, smem_bytes, bad, res.ns_batch, res.gbps, res.ms);
  } else {
    printf("  K=%-4d D=%-4d R=%-6d fetch=%d  ~%8.3f ns/batch  %7.1f GB/s  "
           "(%.3f ms/rep)\n",
           K, D, R, do_fetch, res.ns_batch, res.gbps, res.ms);
  }
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
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB  optin=%zu  SMEM/SM=%zu\n",
         dev, prop.name, NSM, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20),
         OPTIN, (size_t)prop.sharedMemPerMultiprocessor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  g_optin = OPTIN;

  const int DW = 88;               // 352 B, 对齐 MPK §6.1
  const int LANES = 128;
  const long long NDESC = 393216;  // ×352 B ≈ 132 MB ≫ L2

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int G = argc > 2 ? atoi(argv[2]) : NSM;
    long long N = argc > 3 ? atoll(argv[3]) : NDESC;
    int dw = argc > 4 ? atoi(argv[4]) : DW;
    int K = argc > 5 ? atoi(argv[5]) : 128;
    int D = argc > 6 ? atoi(argv[6]) : 2;
    int R = argc > 7 ? atoi(argv[7]) : 1024;
    int reps = argc > 8 ? atoi(argv[8]) : 3;
    int nlanes = argc > 9 ? atoi(argv[9]) : LANES;
    int fetch = argc > 10 ? atoi(argv[10]) : 1;
    run_fe(G, N, dw, K, D, R, reps, nlanes, fetch, 1);
    return 0;
  }

  printf("\n===== §A 正确性：D=1..4 多 worker，R=0 与 host 参考逐位一致（小 NDESC）=====\n");
  for (int D : {1, 2, 3, 4})
    run_fe(NSM, 8192, DW, 32, D, 0, 1, LANES, 1, 1);

  printf("\n===== §A2 正确性：R=1024 下 D=1..4 的 sink 逐位相同（大 NDESC，交叉校验）=====\n");
  unsigned long long ref = 0;
  for (int D : {1, 2, 3, 4}) {
    Res g = run_fe(NSM, NDESC, DW, 128, D, 1024, 2, LANES, 1, 0);
    if (D == 1) ref = g.got;
    printf("      -> D=%d %s reference\n", D, (g.got == ref) ? "==" : "!=");
  }

  printf("\n===== §B 扫 task 计算强度 R：串行(D=1) vs 预取(D=3)，K=128, G=NSM =====\n");
  printf("  %-7s %13s %13s %8s %13s\n", "R", "D=1 ns/batch", "D=3 ns/batch", "speedup",
         "abs_save(ms)");
  const int Rlist[] = {0, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 262144};
  for (int i = 0; i < (int)(sizeof(Rlist) / sizeof(Rlist[0])); ++i) {
    int R = Rlist[i];
    Res a = run_fe(NSM, NDESC, DW, 128, 1, R, 3, LANES, 1, 0);
    Res b = run_fe(NSM, NDESC, DW, 128, 3, R, 3, LANES, 1, 0);
    printf("  %-7d %13.3f %13.3f %8.3f %13.4f\n", R, a.ns_batch, b.ns_batch,
           a.ns_batch / b.ns_batch, a.ms - b.ms);
  }

  printf("\n===== §C 环深 D 扫描：K=128, R=1024, G=NSM（收益是否 D=1→2 拿满？）=====\n");
  for (int D : {1, 2, 3, 4, 5})
    run_fe(NSM, NDESC, DW, 128, D, 1024, 3, LANES, 1, 0);

  printf("\n===== §D 决定性分解：fetch_only(R=0) / exec_only(do_fetch=0) / D=1 / D=3 =====\n");
  printf("  %-7s %11s %11s %11s %11s | D=1 vs f+e   D=3 vs max\n", "R", "fetch(ms)",
         "exec(ms)", "D=1(ms)", "D=3(ms)");
  for (int R : {1024, 4096, 16384, 65536, 262144}) {
    Res f = run_fe(NSM, NDESC, DW, 128, 3, 0, 3, LANES, 1, 0);       // fetch_only
    Res e = run_fe(NSM, NDESC, DW, 128, 1, R, 3, LANES, 0, 0);       // exec_only
    Res s = run_fe(NSM, NDESC, DW, 128, 1, R, 3, LANES, 1, 0);       // D=1 serial
    Res p = run_fe(NSM, NDESC, DW, 128, 3, R, 3, LANES, 1, 0);       // D=3 prefetch
    printf("  %-7d %11.4f %11.4f %11.4f %11.4f | s/(f+e)=%.3f  p/max=%.3f\n", R, f.ms,
           e.ms, s.ms, p.ms, s.ms / (f.ms + e.ms), p.ms / (f.ms > e.ms ? f.ms : e.ms));
  }

  printf("\n===== §E 批量 K 扫描：R=1024, D=1 vs D=3, G=NSM =====\n");
  for (int K : {32, 64, 128, 256}) {
    Res a = run_fe(NSM, NDESC, DW, K, 1, 1024, 3, LANES, 1, 0);
    Res b = run_fe(NSM, NDESC, DW, K, 3, 1024, 3, LANES, 1, 0);
    printf("      K=%d speedup D1/D3 = %.3f\n", K, a.ns_batch / b.ns_batch);
  }

  if (g_descs) cudaFree(g_descs);
  if (g_sink) cudaFree(g_sink);
  return 0;
}
