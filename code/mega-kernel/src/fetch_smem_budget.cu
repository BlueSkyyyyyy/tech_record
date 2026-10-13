// fetch_smem_budget.cu — MegaKernel 专题 P52-1 小实验：
//   **worker「取 task」的 SMEM 预算：批量 K × 环形深度 D 的联合约束，以及它与 task 执行争 SMEM 的占用率代价。**
//
// 动机 / 接续：
//   §39（P50-1，src/cpasync_desc_fetch.cu）量了单 block 的四条取数路径；§40（P51-1，
//   src/multi_worker_desc_fetch.cu）把它扩到 G 个 worker，并加了两条 TMA bulk + 硬件 mbarrier 路径，
//   结论：真实 352 B 描述符下 **TMA bulk 双缓冲（`bulk_pf`）独一档（1263 GB/s，比寄存器读快 3.75×）**，
//   取数是 per-worker **延迟绑定**、带宽靠堆 worker。但 §40.8 明确留了三条尾巴，本节点名第一条：
//     「未测：**`K` 与 SMEM 容量的联合约束**、`cp.async` 释放的寄存器对下游 task 执行的实际收益、
//       与 DeepGEMM SM100 路径对跑（SM100 专属，本机不具备）。」
//   本节把 **`K` × SMEM** 这条做成实测：把「双缓冲」推广成 **D 深环形缓冲**（每槽 K 条描述符），
//   于是「描述符暂存占多少 SMEM」= `D · K · 352 B`，K 与 D 是一对可调的联合参数；总 SMEM 受
//   H100 每 block opt-in 上限（232448 B）与每 SM 容量（233472 B）约束。
//
// 为什么这对 megakernel 重要：
//   - MPK 把整图编进单 kernel，worker 取 task 的描述符缓冲占用的是**同一块 SMEM**——MPK §5.3 用
//     **paged shared memory**（快照行 1313-1325）在「描述符暂存 / 跨 task 软件流水的激活缓冲 / task
//     内部（K/N 双重流水）」之间分页复用。取 task 多吃一页 SMEM，task 执行就少一页。
//   - 所以「取 task 的 SMEM 预算」不是一个孤立旋钮：它直接换 worker 的**占用率**（每 SM 能放几个 worker）、
//     进而换 aggregate 带宽（§40 §B 已证带宽 ∝ worker 数）。这正是 P14-1/P16-1/P17-1「资源足迹成本侧」
//     在**取 task 通路**上的对应物。
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：全线程协作 16 B `cp.async` 进 SMEM **单缓冲** `cp_async_wait<0>`
//     （`persistent_kernel.cuh:1063-1078`）；`load_smem` = `cp.async.ca.shared.global.L2::128B`
//     （`tasks/common/copy_sm80.cuh:44-57`）。
//   - 硬件 mbarrier + TMA bulk 形态在 task 单体内部（`tasks/hopper/tma_2d.cuh:119`、
//     `barrier.cuh:42/59/75`）；本节把 `bulk`+mbarrier 搬到「取 task」层，并推广为 D 深环形。
//   - `TaskDesc` 352 B（论文 §6.1 行 1365）；论文 §5.3「Pre-fetching task descriptions」（行 1364-1368）；
//     论文 §5.3 paged shared memory（行 1313-1325）。
//
// 核心问题：
//   Q1（§A）环形缓冲（任意 K、D）在多 worker 下 checksum 是否一致（正确性）？
//   Q2（§B）固定 K，扫**环形深度 D**：D 从 1（无预取）加深到多少才饱和？§40 只测了 D=1/2。
//   Q3（§C）**固定 SMEM 预算 S，扫 (K, D) 组合**（全部 `K·D·352 ≈ S`）：同等 SMEM 下是「大批量浅环」
//          还是「小批量深环」更优？（本节 headline = K×SMEM 联合约束）
//   Q4（§D）SMEM 上限：总 SMEM 逼近 opt-in 上限（232448 B）时吞吐/正确性如何？
//   Q5（§E）**与 task 执行争 SMEM**：每 block 的取 task SMEM 预算 S 决定每 SM 能放几个 worker
//           （占用率），量 aggregate 带宽随 S 的下降——把「取 task 的 SMEM」换成「worker 数」的账。
//
// 机制：
//   G 个常驻消费者 block；全局 `NDESC` 条自描述描述符（每条 DW 字，DW=88 → 352 B，数组 ≫ L2）。
//   每个 block 处理自己的 batch 序列 `bb = blockIdx.x + t·G`（每 batch K 条连续描述符）。
//   取数载体 = **TMA bulk + 硬件 mbarrier**（§40 的最优载体）：单线程 lane0 发一条
//   `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` 把**整批** K·DW·4 字节搬进 SMEM 槽；
//   环形缓冲 = D 个槽（SMEM 总字节 = D·K·DW·4），槽 `slot = t mod D`，其第 `t/D` 次使用用
//   `try_wait.parity` 的奇偶相位等待（mbarrier 计数=1，expect_tx+complete_tx 一次翻相）。
//   消费（fold）后 `__syncthreads()` 再复发下一批进同槽（先消费、后覆盖，无覆盖竞态）。
//
// 安全：数组预写、无生产者；无死循环（mbarrier 期望字节恒被 complete_tx 满足；asm 写错会 bad≠0 或
//   挂起，脚本带 timeout）。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/fetch_smem_budget.cu -o /tmp/fetch_smem_budget
// 用法：
//   ./fetch_smem_budget run <G> <NDESC> <DW> <K> <D> <reps> [lanes]
//   ./fetch_smem_budget bench

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

#define MAXD 1024

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

__device__ __forceinline__ uint32_t fold_words(const uint32_t* p, int DW) {
  uint32_t x = 0;
  for (int i = 0; i < DW; ++i) x ^= p[i];
  return x;
}

// =====================================================================================
// 取数 kernel：D 深环形缓冲，每槽 K 条描述符；载体 = TMA bulk + 硬件 mbarrier
//   block b 的 batch 序列：bb = blockIdx.x + t*G，t = 0..nt-1
// =====================================================================================
__global__ void ring_fetch_kernel(int NDESC, int DW, int K, int D, int reps,
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

  if (lane == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

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

  for (int rep = 0; rep < reps; ++rep) {
    // 序言：把前 min(D, nt) 批全部发出去（填满环）
    for (int t = 0; t < D && t < nt; ++t) issue(t);
    for (int t = 0; t < nt; ++t) {
      int slot = t % D;
      uint32_t phase = (uint32_t)((t / D) & 1u);
      mbar_wait(&mbar[slot], phase);
      const uint32_t* buf = smem + (size_t)slot * K * DW;
      for (int j = lane; j < K; j += nlanes) acc += fold_words(buf + (size_t)j * DW, DW);
      __syncthreads();  // 全线程读完该槽后，才复发下一批覆盖它
      if (t + D < nt) issue(t + D);
    }
    __syncthreads();
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

static unsigned long long host_expected(long long NDESC, int DW) {
  unsigned long long s = 0;
  for (long long id = 0; id < NDESC; ++id) {
    uint32_t x = (uint32_t)id;
    if (DW > 1) x ^= (uint32_t)(id % 4);
    if (DW > 2) x ^= (uint32_t)(id / 4);
    if (DW > 3) x ^= 0u;
    for (int i = 4; i < DW; ++i) x ^= mix((uint32_t)id, (uint32_t)i);
    s += x;
  }
  return s;
}

static uint32_t* g_descs = nullptr;
static unsigned long long* g_sink = nullptr;
static long long g_alloc = -1;
static int g_dw = -1;
static unsigned long long g_exp = 0;

static void ensure_buf(long long NDESC, int DW) {
  if (g_alloc == NDESC && g_dw == DW) return;  // 精确命中才复用（g_exp 与 NDESC 绑定）
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

// smem_bytes 由 K,D,DW 决定；返回 bad
static int run_ring(int G, long long NDESC, int DW, int K, int D, int reps,
                    int nlanes) {
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;  // kernel 按整 batch 处理，checksum 也按截断后的算
  const size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (smem_bytes > g_optin) {
    printf("  ring K=%-4d D=%-4d SMEM=%6zuB  SKIP (exceeds optin %zu)\n", K, D,
           smem_bytes, g_optin);
    return 0;
  }
  if (smem_bytes > 48 * 1024)
    CK(cudaFuncSetAttribute(ring_fetch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem_bytes));
  ensure_buf(NDESC, DW);

  unsigned long long got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    ring_fetch_kernel<<<G, nlanes, smem_bytes>>>(NDESC, DW, K, D, 1, g_descs, g_sink);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms_accum += ms;
    CK(cudaMemcpy(&got, g_sink, 8, cudaMemcpyDeviceToHost));
  }
  double items = (double)NDESC * (double)reps;
  double ns_per_item = (ms_accum / 1e3) / items * 1e9;
  double gbps = (double)NDESC * (double)DW * 4.0 * (double)reps / (ms_accum / 1e3) / 1e9;
  int bad = (got == g_exp) ? 0 : 1;
  printf("  ring K=%-4d D=%-4d SMEM=%6zuB(%5.1fKB) G=%-4d lanes=%-4d  bad=%d  "
         "~%8.3f ns/item  %7.1f GB/s  (%.3f ms/rep)\n",
         K, D, smem_bytes, smem_bytes / 1024.0, G, nlanes, bad, ns_per_item, gbps,
         ms_accum / reps);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return bad;
}

static int occ_blocks(int K, int D, int DW, int nlanes) {
  size_t smem = (size_t)D * K * DW * 4 + (size_t)D * 8;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(ring_fetch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, ring_fetch_kernel, nlanes,
                                                (int)smem);
  return n;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  const int NSM = prop.multiProcessorCount;
  const size_t OPTIN = prop.sharedMemPerBlockOptin;
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB  SMEM/block(req)=%zu "
         "optin=%zu  SMEM/SM=%zu\n",
         dev, prop.name, NSM, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20),
         (size_t)prop.sharedMemPerBlock, OPTIN, (size_t)prop.sharedMemPerMultiprocessor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  g_optin = OPTIN;

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int G = argc > 2 ? atoi(argv[2]) : NSM;
    long long NDESC = argc > 3 ? atoll(argv[3]) : 700000;
    int DW = argc > 4 ? atoi(argv[4]) : 88;
    int K = argc > 5 ? atoi(argv[5]) : 8;
    int D = argc > 6 ? atoi(argv[6]) : 2;
    int reps = argc > 7 ? atoi(argv[7]) : 5;
    int nlanes = argc > 8 ? atoi(argv[8]) : 128;
    run_ring(G, NDESC, DW, K, D, reps, nlanes);
    return 0;
  }

  const int DW = 88;  // 352 B, 对齐 MPK §6.1
  const long long BYTES = 256LL << 20;
  // 取 128 的整数倍，令 K∈{1,2,4,…,128} 都能整除（截断为空操作，checksum 口径一致）
  const long long NDESC = (BYTES / (DW * 4)) / 128 * 128;

  printf("\n===== §A 正确性：环形缓冲（多 worker）checksum 一致（DW=88=352B, G=NSM, lanes=128） =====\n");
  for (int K : {1, 4, 16, 64})
    for (int D : {1, 2, 4, 8}) run_ring(NSM, NDESC, DW, K, D, 2, 128);

  printf("\n===== §B 环形深度 D 扫描（K=8 固定, DW=88, G=NSM, lanes=128, 256MB） =====\n");
  for (int D : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}) run_ring(NSM, NDESC, DW, 8, D, 5, 128);

  printf("\n===== §C K×SMEM 联合约束：固定 SMEM 预算 S 扫 (K,D)（DW=88, G=NSM, lanes=128） =====\n");
  for (long long S : {(long long)64 << 10, (long long)128 << 10, (long long)192 << 10}) {
    printf("  --- S=%lld B (%.1f KB) ---\n", S, S / 1024.0);
    for (int K : {1, 2, 4, 8, 16, 32, 64, 128}) {
      int D = (int)(S / ((long long)K * DW * 4));
      if (D < 1) D = 1;
      if (D > MAXD) D = MAXD;
      run_ring(NSM, NDESC, DW, K, D, 5, 128);
    }
  }

  printf("\n===== §D SMEM 上限：总 SMEM 逼近 opt-in（K=1, 扫 D；DW=88, G=NSM, lanes=128） =====\n");
  for (int D : {64, 128, 256, 384, 512, 600, 640}) run_ring(NSM, NDESC, DW, 1, D, 5, 128);

  printf("\n===== §E 与 task 执行争 SMEM：每 block SMEM 预算 S → 占用率 → aggregate 带宽（DW=88, lanes=128, 256MB） =====\n");
  for (int K : {1, 2, 4, 8, 16, 32, 64, 128}) {
    int D = 2;
    size_t smem = (size_t)D * K * DW * 4 + (size_t)D * 8;
    if (smem > OPTIN) continue;
    int bps = occ_blocks(K, D, DW, 128);
    int G = NSM * (bps > 0 ? bps : 1);
    printf("  [K=%d D=%d SMEM=%.1fKB] occ=%d block/SM -> G=%d\n", K, D,
           smem / 1024.0, bps, G);
    run_ring(G, NDESC, DW, K, D, 5, 128);
  }

  if (g_descs) cudaFree(g_descs);
  if (g_sink) cudaFree(g_sink);
  return 0;
}
