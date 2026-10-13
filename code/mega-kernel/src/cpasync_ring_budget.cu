// cpasync_ring_budget.cu — MegaKernel 专题 P53-1 小实验：
//   **worker「取 task」深环的第三种载体（16 B cp.async）与 §41 的 TMA bulk 对跑；以及 cp.async
//     「不占寄存器」对下游 task 执行的实际收益。**
//
// 动机 / 接续：
//   §41（P52-1，src/fetch_smem_budget.cu）把取 task 的「双缓冲」推广成 D 深环形缓冲（每槽 K 条描述符），
//   结论：**深环 D≥2 无益、批量 K 才是主旋钮**（大 K + D=2 最优，K=128/D=2 达 2667 GB/s），取 task 的
//   SMEM 是软约束（换 worker 数、总带宽守恒）。但载体固定为 TMA bulk + mbarrier。§41.8 明留两条尾巴：
//     「载体固定为 TMA bulk + mbarrier，**未同时跑 16 B `cp.async` 环**（其预取机制 `wait_group` 是
//       软件计数，深环写法见 §40.8 的 drain 坑）；**未测 `cp.async` 释放的寄存器对下游 task 执行的
//       实际收益**（§40.8 第二条，仍余）。」
//   本节点名这两条做成实测。
//
// 为什么这对 megakernel 重要：
//   - MPK worker 取 task 的真实载体就是 **16 B `cp.async` 进 SMEM 单缓冲**（`persistent_kernel.cuh:1063-1078`
//     `cp_async_wait<0>`）——所以「cp.async 深环」才是快照的**原生**形态，而 §41 的 bulk 是「换更好的载体」。
//     本节回答：原生 cp.async 能不能靠深环追平 bulk？
//   - `cp.async` 与寄存器读的本质区别：**数据 gmem→SMEM 不经过寄存器**。所以它同时是「换载体」和
//     「释放寄存器」——后者对 megakernel 尤其重要，因为 per-thread 寄存器是整个 kernel 的**不可分页**
//     足迹（P14-1/§8、P17-1/§11、MPK §7 行 1748），取 task 若占寄存器就抬高 worker 的寄存器高水位。
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - worker 取 task：全线程协作 16 B `cp.async` 进 SMEM 单缓冲 `cp_async_wait<0>`
//     （`persistent_kernel.cuh:1063-1078`）；`load_smem` = `cp.async.ca.shared.global.L2::128B`
//     （`tasks/common/copy_sm80.cuh:44-57`）。
//   - `TaskDesc` 352 B（论文 §6.1 行 1365）；论文 §5.3 pre-fetching（行 1364-1368）/ paged SMEM（行 1313-1325）。
//
// 核心问题：
//   Q1（§A）cp.async D 深环在任意 (K,D) 多 worker 下 checksum 是否一致？
//   Q2（§B）同为 D 深环，**cp.async vs TMA bulk** 的 K×D 平面对照：原生 cp.async 能追平 bulk 吗？
//   Q3（§C）cp.async 的软件 `wait_group` 深环对**环深 D** 的敏感性与 bulk 的硬件 mbarrier 有何不同？
//   Q4（§D）**取 task 载体对下游 task 的寄存器足迹**：cp.async（不占寄存器）vs vec4（经寄存器），
//         扫下游 task 的计算寄存器需求 H，量 numRegs/占用率/端到端时间。
//
// 机制：
//   - bulk 环（复刻 §41）：单线程 lane0 发 `cp.async.bulk` 整批 K·DW·4 B 进 SMEM 槽，硬件 mbarrier
//     `complete_tx` 记字节、`try_wait.parity` 等完成；运行时 D。
//   - cp.async 环（本节新增）：每 lane 发自己的 16 B `cp.async.cg`，整批 `commit_group`，消费前
//     `wait_group(D-1)`（序言填满 D 槽后，等待到「最多 D-1 组在飞」即保证第 t 组完成）；尾批改用
//     `wait_group 0` 兜底（否则漏等，正是 §40.8 记录的 drain 坑）。因 `wait_group` 需**立即数**，环深 D
//     走模板实例化。
//
// 安全：数组预写、无生产者；脚本带 timeout；asm 写错会 bad≠0 或挂起。
//
// 编译：
//   nvcc -O3 -arch=sm_90a src/cpasync_ring_budget.cu -o /tmp/cpasync_ring_budget
// 用法：
//   ./cpasync_ring_budget run <carrier:bulk|ca> <G> <NDESC> <DW> <K> <D> <reps> [lanes]
//   ./cpasync_ring_budget bench

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

// ---- 硬件 mbarrier + TMA bulk 1D（与 §41 同） ----
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

// ---- 16 B cp.async（copy_sm80.cuh 的形，搬到「取 task」层） ----
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gmem_ptr) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ uint32_t fold_words(const uint32_t* p, int DW) {
  uint32_t x = 0;
  for (int i = 0; i < DW; ++i) x ^= p[i];
  return x;
}

// =====================================================================================
// (1) bulk 环（复刻 §41 的 ring_fetch_kernel，运行时 D）
// =====================================================================================
__global__ void ring_bulk_kernel(int NDESC, int DW, int K, int D,
                                 const uint32_t* __restrict__ descs,
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

  for (int t = 0; t < D && t < nt; ++t) issue(t);
  for (int t = 0; t < nt; ++t) {
    int slot = t % D;
    uint32_t phase = (uint32_t)((t / D) & 1u);
    mbar_wait(&mbar[slot], phase);
    const uint32_t* buf = smem + (size_t)slot * K * DW;
    for (int j = lane; j < K; j += nlanes) acc += fold_words(buf + (size_t)j * DW, DW);
    __syncthreads();
    if (t + D < nt) issue(t + D);
  }
  if (acc) atomicAdd(sink, acc);
}

// =====================================================================================
// (2) cp.async 环（模板 D，以便 wait_group 用立即数）
// =====================================================================================
template <int D>
__global__ void ring_ca_kernel(int NDESC, int DW, int K,
                               const uint32_t* __restrict__ descs,
                               unsigned long long* __restrict__ sink) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int G = gridDim.x;
  const int nbatch = NDESC / K;
  const int words_per_batch = K * DW;
  const int chunks_per_batch = words_per_batch / 4;  // 16 B = 4 字，DW=88 可整除
  unsigned long long acc = 0;
  const int nt = (blockIdx.x < nbatch) ? (nbatch - blockIdx.x + G - 1) / G : 0;

  auto issue = [&](int t) {
    int slot = t % D;
    int bb = blockIdx.x + t * G;
    const char* base = (const char*)(descs + (size_t)bb * words_per_batch);
    char* dst = (char*)(smem + (size_t)slot * words_per_batch);
    for (int c = lane; c < chunks_per_batch; c += nlanes)
      cp_async16(dst + (size_t)c * 16, base + (size_t)c * 16);
    cp_async_commit();
  };

  for (int t = 0; t < D && t < nt; ++t) issue(t);
  for (int t = 0; t < nt; ++t) {
    int slot = t % D;
    if (t + D <= nt)
      cp_async_wait<D - 1>();  // 保证第 t 组完成（其后最多 D-1 组在飞）
    else
      cp_async_wait<0>();      // 尾批兜底（§40.8 drain 坑）
    __syncthreads();           // 等全 block 各组都到齐
    const uint32_t* buf = smem + (size_t)slot * words_per_batch;
    for (int j = lane; j < K; j += nlanes) acc += fold_words(buf + (size_t)j * DW, DW);
    __syncthreads();           // 全线程读完该槽后，才复发下一批覆盖它
    if (t + D < nt) issue(t + D);
  }
  if (acc) atomicAdd(sink, acc);
}

// =====================================================================================
// (3) 取 task + 下游 task：载体对「下游 task 寄存器足迹」的影响
//   MODE 0 = vec4 寄存器读；MODE 1 = 16 B cp.async（不占寄存器）
//   H = 下游 task 的常驻寄存器需求（长 FMA 链）
// =====================================================================================
template <int H>
__device__ __forceinline__ float heavy_path(float* __restrict__ scratch, int reps) {
  constexpr int HN = (H > 0) ? H : 1;
  float r[HN];
#pragma unroll
  for (int k = 0; k < HN; ++k) r[k] = (float)(threadIdx.x + k) * 1e-6f + 1.0f;
  for (int rr = 0; rr < reps; ++rr) {
#pragma unroll
    for (int k = 0; k < HN; ++k) r[k] = fmaf(r[k], 1.0000001f, 1e-7f);
  }
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < HN; ++k) t += r[k];
  if (H == 0) t = 0.f;
  if (threadIdx.x == 0) scratch[blockIdx.x] = t;
  return t;
}

// 载体 0：vec4 寄存器读（把一批 K 条描述符读进寄存器折叠成 seed，再跑下游 task）
template <int H>
__global__ void fetch_task_reg(int NDESC, int DW, int K, int reps,
                               const uint32_t* __restrict__ descs,
                               float* __restrict__ scratch, float* __restrict__ out) {
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int G = gridDim.x;
  const int nbatch = NDESC / K;
  const int words = K * DW;
  const int nt = (blockIdx.x < nbatch) ? (nbatch - blockIdx.x + G - 1) / G : 0;
  float seed = 0.f;
  for (int t = 0; t < nt; ++t) {
    int bb = blockIdx.x + t * G;
    const uint32_t* base = descs + (size_t)bb * words;
    uint32_t a = 0;
    // 每 lane 以 16 B 为单位（uint4）读自己那份，折叠
    for (int c = lane * 4; c < words; c += nlanes * 4) {
      uint4 v = *reinterpret_cast<const uint4*>(base + c);
      a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    seed += (float)(a & 0xffffu) * 1e-3f;
  }
  float s = seed;
  if (H > 0) s += heavy_path<H>(scratch, reps);
  if ((lane & 31) == 0) atomicAdd(out, s);
}

// 载体 1：16 B cp.async 进 SMEM（数据不经寄存器），再从 SMEM 折叠，再跑下游 task
template <int H>
__global__ void fetch_task_ca(int NDESC, int DW, int K, int reps,
                              const uint32_t* __restrict__ descs,
                              float* __restrict__ scratch, float* __restrict__ out) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int G = gridDim.x;
  const int nbatch = NDESC / K;
  const int words = K * DW;
  const int chunks = words / 4;
  const int nt = (blockIdx.x < nbatch) ? (nbatch - blockIdx.x + G - 1) / G : 0;
  float seed = 0.f;
  for (int t = 0; t < nt; ++t) {
    int bb = blockIdx.x + t * G;
    const char* base = (const char*)(descs + (size_t)bb * words);
    char* dst = (char*)smem;
    for (int c = lane; c < chunks; c += nlanes) cp_async16(dst + (size_t)c * 16, base + (size_t)c * 16);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    uint32_t a = 0;
    for (int j = lane; j < K; j += nlanes) a ^= fold_words(smem + (size_t)j * DW, DW);
    __syncthreads();
    seed += (float)(a & 0xffffu) * 1e-3f;
  }
  float s = seed;
  if (H > 0) s += heavy_path<H>(scratch, reps);
  if ((lane & 31) == 0) atomicAdd(out, s);
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

// ---- 模板 D 分发 ----
typedef void (*ca_kern_t)(int, int, int, const uint32_t*, unsigned long long*);

static ca_kern_t ca_dispatch(int D, int* out_D) {
  switch (D) {
#define CA_CASE(d) case d: *out_D = d; return ring_ca_kernel<d>;
  CA_CASE(1) CA_CASE(2) CA_CASE(3) CA_CASE(4) CA_CASE(5) CA_CASE(6) CA_CASE(7)
  CA_CASE(8) CA_CASE(9) CA_CASE(10) CA_CASE(11) CA_CASE(12) CA_CASE(13) CA_CASE(14)
  CA_CASE(15) CA_CASE(16) CA_CASE(17) CA_CASE(18) CA_CASE(19) CA_CASE(20) CA_CASE(21)
  CA_CASE(22) CA_CASE(23) CA_CASE(24) CA_CASE(25) CA_CASE(26) CA_CASE(27) CA_CASE(28)
  CA_CASE(29) CA_CASE(30) CA_CASE(31) CA_CASE(32)
#undef CA_CASE
  default: *out_D = 0; return nullptr;
  }
}

static size_t g_optin = 0;

// carrier: 0=bulk, 1=ca
static int run_ring(int carrier, int G, long long NDESC, int DW, int K, int D,
                    int reps, int nlanes, int quiet) {
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  const size_t smem_bytes = (size_t)D * K * DW * 4 + ((carrier == 0) ? (size_t)D * 8 : 0);
  if (smem_bytes > g_optin) {
    if (!quiet)
      printf("  %s K=%-4d D=%-4d SMEM=%6zuB  SKIP (exceeds optin %zu)\n",
             carrier ? "ca  " : "bulk", K, D, smem_bytes, g_optin);
    return 0;
  }
  if (smem_bytes > 48 * 1024) {
    if (carrier == 0)
      CK(cudaFuncSetAttribute(ring_bulk_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    else {
      int Dd = 0; ca_kern_t k = ca_dispatch(D, &Dd);
      if (!k) { printf("  unsupported D=%d\n", D); return 0; }
      CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    }
  }
  ensure_buf(NDESC, DW);

  unsigned long long got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float ms_accum = 0.f;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    if (carrier == 0)
      ring_bulk_kernel<<<G, nlanes, smem_bytes>>>(NDESC, DW, K, D, g_descs, g_sink);
    else {
      int Dd = 0; ca_kern_t k = ca_dispatch(D, &Dd);
      k<<<G, nlanes, smem_bytes>>>(NDESC, DW, K, g_descs, g_sink);
    }
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
  if (!quiet)
    printf("  %s K=%-4d D=%-4d SMEM=%6zuB(%5.1fKB) G=%-4d lanes=%-4d  bad=%d  "
           "~%8.3f ns/item  %7.1f GB/s  (%.3f ms/rep)\n",
           carrier ? "ca  " : "bulk", K, D, smem_bytes, smem_bytes / 1024.0, G, nlanes,
           bad, ns_per_item, gbps, ms_accum / reps);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return bad;
}

// ---- §D：载体对下游 task 的寄存器足迹 ----
template <typename KF>
static void attrs(const char* name, KF kernel, int nlanes, int smem) {
  cudaFuncAttributes a;
  CK(cudaFuncGetAttributes(&a, kernel));
  int nb = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, nlanes, smem));
  printf("  %-26s | %5d | %8zu | %8d | %8d\n", name, a.numRegs, a.localSizeBytes,
         nb, nb * nlanes / 32);
}

template <int H>
static void reg_task_row(int G, long long NDESC, int DW, int K, int reps, int nlanes,
                         const float* scratch, float* out, int iters) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int r = 0; r < 2; ++r) fetch_task_reg<H><<<G, nlanes>>>(NDESC, DW, K, reps, g_descs,
                                                               (float*)scratch, out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    fetch_task_reg<H><<<G, nlanes>>>(NDESC, DW, K, reps, g_descs, (float*)scratch, out);
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  printf("  H=%-4d reg  : %8.4f ms\n", H, ms / iters);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
}

template <int H>
static void ca_task_row(int G, long long NDESC, int DW, int K, int reps, int nlanes,
                        int smem, const float* scratch, float* out, int iters) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int r = 0; r < 2; ++r) fetch_task_ca<H><<<G, nlanes, smem>>>(NDESC, DW, K, reps,
                                                                    g_descs, (float*)scratch, out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for (int r = 0; r < iters; ++r)
    fetch_task_ca<H><<<G, nlanes, smem>>>(NDESC, DW, K, reps, g_descs, (float*)scratch, out);
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  printf("  H=%-4d ca   : %8.4f ms\n", H, ms / iters);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
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
    const char* car = argc > 2 ? argv[2] : "ca";
    int G = argc > 3 ? atoi(argv[3]) : NSM;
    long long NDESC = argc > 4 ? atoll(argv[4]) : 700000;
    int DW = argc > 5 ? atoi(argv[5]) : 88;
    int K = argc > 6 ? atoi(argv[6]) : 8;
    int D = argc > 7 ? atoi(argv[7]) : 2;
    int reps = argc > 8 ? atoi(argv[8]) : 5;
    int nlanes = argc > 9 ? atoi(argv[9]) : 128;
    run_ring(strcmp(car, "bulk") == 0 ? 0 : 1, G, NDESC, DW, K, D, reps, nlanes, 0);
    return 0;
  }

  const int DW = 88;  // 352 B, 对齐 MPK §6.1
  const long long BYTES = 256LL << 20;
  const long long NDESC = (BYTES / (DW * 4)) / 128 * 128;
  const int LANES = 128;

  printf("\n===== §A 正确性：cp.async D 深环（多 worker）checksum 一致（DW=88=352B, G=NSM, lanes=128） =====\n");
  for (int K : {1, 4, 16, 64})
    for (int D : {1, 2, 4, 8}) run_ring(1, NSM, NDESC, DW, K, D, 2, LANES, 0);

  printf("\n===== §B 载体对照 K×SMEM 联合约束：cp.async vs TMA bulk（DW=88, G=NSM, lanes=128） =====\n");
  for (long long S : {(long long)64 << 10, (long long)128 << 10}) {
    printf("  --- S=%lld B (%.1f KB) ---\n", S, S / 1024.0);
    for (int K : {1, 8, 32, 64, 128}) {
      int D = (int)(S / ((long long)K * DW * 4));
      if (D < 1) D = 1;
      if (D > 32) D = 32;
      run_ring(0, NSM, NDESC, DW, K, D, 5, LANES, 0);
      run_ring(1, NSM, NDESC, DW, K, D, 5, LANES, 0);
    }
  }

  printf("\n===== §C 环深 D 扫描：cp.async vs bulk（K=8, DW=88, G=NSM, lanes=128） =====\n");
  for (int D : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}) {
    run_ring(0, NSM, NDESC, DW, 8, D, 5, LANES, 0);
    run_ring(1, NSM, NDESC, DW, 8, D, 5, LANES, 0);
  }

  printf("\n===== §D 取 task 载体对下游 task 的寄存器足迹（G=528=4 blk/SM, lanes=128, reps=200） =====\n");
  {
    const int G = 528, reps = 200, iters = 30, K = 8;
    const int smem = K * DW * 4;
    CK(cudaFuncSetAttribute(fetch_task_ca<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CK(cudaFuncSetAttribute(fetch_task_ca<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CK(cudaFuncSetAttribute(fetch_task_ca<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CK(cudaFuncSetAttribute(fetch_task_ca<192>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    printf("  %-26s | %5s | %8s | %8s | %8s\n", "kernel", "regs", "localB", "blk/SM", "warps/SM");
    printf("  ---------------------------+-------+----------+----------+---------\n");
    attrs("fetch_task_reg<0>", fetch_task_reg<0>, LANES, 0);
    attrs("fetch_task_ca<0> ", fetch_task_ca<0>, LANES, smem);
    attrs("fetch_task_reg<32>", fetch_task_reg<32>, LANES, 0);
    attrs("fetch_task_ca<32> ", fetch_task_ca<32>, LANES, smem);
    attrs("fetch_task_reg<128>", fetch_task_reg<128>, LANES, 0);
    attrs("fetch_task_ca<128> ", fetch_task_ca<128>, LANES, smem);
    attrs("fetch_task_reg<192>", fetch_task_reg<192>, LANES, 0);
    attrs("fetch_task_ca<192> ", fetch_task_ca<192>, LANES, smem);

    float* scratch = nullptr;
    float* out = nullptr;
    CK(cudaMalloc(&scratch, (size_t)G * 64));
    CK(cudaMalloc(&out, sizeof(float)));
    printf("\n  --- 端到端时间（fetch K=8 描述符 + 下游 task H） ---\n");
#define DROW(Hv)                                                              \
  reg_task_row<Hv>(G, NDESC, DW, K, reps, LANES, scratch, out, iters);        \
  ca_task_row<Hv>(G, NDESC, DW, K, reps, LANES, smem, scratch, out, iters);
    DROW(0)
    DROW(8)
    DROW(16)
    DROW(32)
    DROW(64)
    DROW(128)
    DROW(192)
#undef DROW
    cudaFree(scratch);
    cudaFree(out);
  }

  if (g_descs) cudaFree(g_descs);
  if (g_sink) cudaFree(g_sink);
  return 0;
}
