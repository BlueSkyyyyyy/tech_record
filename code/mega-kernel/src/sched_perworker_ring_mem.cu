// ============================================================================
// P71-1 (§59)：补掉 §58.10 明留的尾巴——「`CW` 是纯 ALU 延迟，只拉长 task、
//             不产生真实 HBM 流量；真实 task 的『重』是访存+计算混合，`PS*`-vs-`CW`
//             曲线的形状可信、精确拐点只对本微基准成立」。
//
// 背景（接 §47–§58）：
//   - §58 把真实 MPK 形态「少 producer(scheduler) × 多 per-worker 环」实现并实测，
//     headline 是：喂饱 G 个 worker 所需的 producer 数 `PS*` 随 task 计算重量 `CW`
//     单调下降（CW=0 → PS*≈G；CW=2^18 → PS*≈4，对上 MPK 的 4:128=1:32）。
//   - 但 `CW` 只是 worker 循环里的**纯 ALU 延迟**（`b=b*a+c` 反复），它把 task
//     拉长、却不产生任何 HBM 流量；真实 task（GEMM/attention/RMSNorm）的「重」是
//     **访存与计算混合**。本节给 worker 加一条**真实 HBM 数据面**（每 batch 流式读
//     `MD` 个 uint4），把 §58 的 `PS*`-vs-重量曲线从「纯 ALU」推进到「访存主导」，
//     回答三个问题：
//       (1) 数据面读**不改数值**（逐位对 host 参考）；
//       (2) `PS*` 随**访存重量 `MD`** 的标度——访存型 task 是否同样「越重越少 producer」？
//       (3) 数据面与 producer 描述符写**争不争 HBM**（HBM 饱和时加 producer 是否反而降吞吐）？
//
// 唯一新变量 = worker 的**访存重量** `MD`（每 batch 从 `g_data` 流式读 MD 个 uint4，
//   用 `__ldcs` 绕 L2），描述符环协议、per-worker 环拓扑、producer 轮转全与 §58 相同；
//   `CW` 保留作「纯 ALU」对照。数据面读的值确定（data_word），故 `sink` 可与 host 参考
//   逐位比（MD>0 时仍判 OK/WRONG，而非 §58 CW>0 的 RUN）。
//
// 骨架与 §57/§58 同：producer 写描述符到 global ring（标量 store、release 发布）、
//   worker 用 TMA bulk load 搬进 SMEM 折叠 + 可选 HBM 数据面读；sink 与 host 参考逐位比；
//   clock64 守门狗 + abort 兜死锁。
//
// 用法：./sched_perworker_ring_mem run <PS> <G> <NB> <DW> <K> <D> <CAPp> <reps> [CW] [MD]
//       ./sched_perworker_ring_mem bench          # 本节全部 Section
// 目标卡 H100 / sm90a。安全：脚本带 timeout；死锁配置由守门狗兜底返回。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t _e = (x);                                                            \
    if (_e != cudaSuccess) {                                                         \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

#define MAXT 256
#define MAXD 64

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
__host__ __device__ __forceinline__ uint32_t gen_word(uint32_t gid, int i) {
  if (i == 0) return gid;
  if (i == 1) return gid % 4u;
  if (i == 2) return gid / 4u;
  if (i == 3) return 0u;
  return mix32(gid, (uint32_t)i);
}
__host__ __device__ __forceinline__ unsigned long long fold_desc(uint32_t gid, int DW) {
  uint32_t a = 0x9e3779b9u;
  for (int i = 0; i < DW; ++i) a = a * 2246822519u + gen_word(gid, i);
  return (unsigned long long)a;
}
// 数据面每个 uint4 的 4 个分量（uidx = 全局 batch*MD + m）
__host__ __device__ __forceinline__ uint32_t data_word(unsigned long long uidx, int j) {
  return mix32((uint32_t)(uidx) ^ ((uint32_t)(uidx >> 32) * 2654435761u),
               (uint32_t)j * 101u + 7u);
}

// ---- 内存序原语（对齐 MPK 队列协议；快照 mpk_atoms.cuh:17-96）----
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_release_u64(unsigned long long* p,
                                                                   unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v)
               : "memory");
  return old;
}

// ---- 硬件 mbarrier + TMA bulk load（worker 取 task）----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes)
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
__device__ __forceinline__ void bulk_load_1d(void* smem_ptr, const void* gmem_ptr,
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
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// ============================================================================
// kernel（单常驻：block 0..PS-1 = producer/scheduler，block PS..PS+G-1 = worker）
//   mode：0 = static（无生产者，worker 只读 host 预填数组，给参考）
//         1 = per-worker 环：worker w 拥有环 w，producer owner(w)=w/wpp 填环
//   PS 个 producer 每个拥有连续一段 worker 环，段内轮转填。
//   MD  MD>0 时 worker 每 task 额外从 g_data 流式读 MD 个 uint4（真实 HBM 数据面）。
// ============================================================================
__global__ void spr_mem_kernel(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode,
                               int CW, long long MD, const uint32_t* __restrict__ g_static,
                               uint32_t* __restrict__ ring, unsigned long long* head,
                               unsigned long long* tail, unsigned long long* sink, int* abort,
                               int* dbg_done, const uint4* __restrict__ g_data) {
  const int tid = threadIdx.x;
  const int nlanes = blockDim.x;
  extern __shared__ char smem_raw[];
  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_raw);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int bytes_per_batch = K * DW * 4;

  const int chunk = (NB + G - 1) / G;
  const int nb = NB;

  auto count_of = [&](int w) -> int {
    int b0 = w * chunk;
    if (b0 >= nb) return 0;
    int c = nb - b0;
    return (c < chunk) ? c : chunk;
  };

  // ---------------- producer / scheduler：block 0..PS-1 ----------------
  if (mode != 0 && blockIdx.x < PS) {
    const int s = blockIdx.x;
    const int wpp = (G + PS - 1) / PS;
    int r0 = s * wpp;
    int r1 = (s + 1) * wpp;
    if (r1 > G) r1 = G;
    int maxr = 0;
    for (int w = r0; w < r1; ++w) {
      int c = count_of(w);
      if (c > maxr) maxr = c;
    }
    // 段内轮转填：每轮给每条未填完的环写一批 bl=r。
    for (int r = 0; r < maxr; ++r) {
      for (int w = r0; w < r1; ++w) {
        int c = count_of(w);
        if (r >= c) continue;
        const int slot = r % CAPp;
        if (tid == 0) {
          long long t0 = clock64();
          unsigned long long need = (unsigned long long)(r + 1);
          unsigned long long t = ld_acquire_u64(tail + w);
          while (t + (unsigned long long)CAPp < need) {
            if (*abort) break;
            if (clock64() - t0 > 4000000000LL) { *abort = 1; break; }
            __nanosleep(200);
            t = ld_acquire_u64(tail + w);
          }
        }
        __syncthreads();
        if (*abort) return;
        const int total = K * DW;
        uint32_t* dst = (uint32_t*)ring + (size_t)((size_t)w * CAPp + slot) * K * DW;
        const int bb = w * chunk + r;  // 该环第 r 批 = 全局批 w*chunk+r
        for (int i = tid; i < total; i += nlanes) {
          int d = i / DW, wi = i - d * DW;
          uint32_t gid = (uint32_t)((size_t)bb * K + d);
          dst[i] = gen_word(gid, wi);
        }
        __syncthreads();
        if (tid == 0) {
          fence_proxy_async_global();
          st_release_u64(head + w, (unsigned long long)(r + 1));
        }
        __syncthreads();
      }
    }
    if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
    return;
  }

  // ---------------- worker：block PS..PS+G-1 ----------------
  const int w = (mode != 0) ? (blockIdx.x - PS) : blockIdx.x;
  const int nt = (w < G) ? count_of(w) : 0;
  const int b0w = w * chunk;

  if (tid == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  unsigned long long acc = 0;

  auto wait_ready = [&](int bl) -> bool {
    if (mode == 0) return true;
    long long t0 = clock64();
    unsigned long long need = (unsigned long long)(bl + 1);
    unsigned long long h = ld_acquire_u64(head + w);
    while (h < need) {
      if (abort && *abort) return false;
      if (clock64() - t0 > 2000000000LL) { if (abort) *abort = 1; return false; }
      __nanosleep(100);
      h = ld_acquire_u64(head + w);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    if (!wait_ready(t)) return false;
    const uint32_t* src;
    if (mode == 0)
      src = g_static + (size_t)(b0w + t) * K * DW;
    else
      src = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
    if (tid == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
      bulk_load_1d(smem + (size_t)slot * K * DW, src, (uint32_t)bytes_per_batch, &mbar[slot]);
    }
    return true;
  };

  bool ok = true;
  for (int t = 0; t < D && t < nt; ++t)
    if (!issue(t)) { ok = false; break; }

  for (int t = 0; t < nt && ok; ++t) {
    int slot = t % D;
    uint32_t phase = (uint32_t)((t / D) & 1u);
    mbar_wait(&mbar[slot], phase);
    const uint32_t* buf = smem + (size_t)slot * K * DW;
    unsigned long long part = 0;
    for (int j = tid; j < K; j += nlanes) {
      uint32_t a = 0x9e3779b9u;
      for (int i = 0; i < DW; ++i) a = a * 2246822519u + buf[(size_t)j * DW + i];
      part += (unsigned long long)a;
    }
    if (MD > 0) {  // 真实 HBM 数据面：每 batch 流式读 MD 个 uint4（__ldcs 绕 L2）
      const int bb = b0w + t;
      const uint4* dp = g_data + (size_t)bb * (size_t)MD;
      unsigned long long dpart = 0;
      for (long long m = tid; m < MD; m += nlanes) {
        uint4 v = __ldcs(dp + m);
        dpart += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
      }
      part += dpart;
    }
    if (CW > 0) {  // 纯 ALU 延迟（模拟计算型 task，不改 part）
      uint32_t b = (uint32_t)part + (uint32_t)(tid * 2654435761u);
      for (int c = 0; c < CW; ++c) b = b * 1664525u + 1013904223u;
      if (b == 0x1234567u) part += b;
    }
    __shared__ unsigned long long red[MAXT];
    red[tid] = part;
    __syncthreads();
    if (tid == 0) {
      unsigned long long ssum = 0;
      for (int i = 0; i < nlanes; ++i) ssum += red[i];
      acc += ssum;
    }
    __syncthreads();
    if (mode != 0 && tid == 0) atom_add_release_u64(tail + w, (unsigned long long)1);
    if (t + D < nt)
      if (!issue(t + D)) { ok = false; break; }
  }

  if (acc && tid == 0) atomicAdd(sink, acc);
  if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
}

// ============================================================================
// host
// ============================================================================
__global__ void fill_static_kernel(uint32_t* d, long long nb, int K, int DW) {
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long tot = nb * K;
  if (gid >= tot) return;
  uint32_t* p = d + gid * DW;
  for (int i = 0; i < DW; ++i) p[i] = gen_word((uint32_t)gid, i);
}
// 填 g_data 并累加 host 参考（每 uint4 的 x^y^z^w），用原子加回 dref。
__global__ void fill_data_kernel(uint4* d, long long nb, long long MD,
                                 unsigned long long* dref) {
  long long uidx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long tot = nb * MD;
  if (uidx >= tot) return;
  unsigned long long u = (unsigned long long)uidx;
  uint4 v;
  v.x = data_word(u, 0);
  v.y = data_word(u, 1);
  v.z = data_word(u, 2);
  v.w = data_word(u, 3);
  d[uidx] = v;
  atomicAdd(dref, (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w));
}

struct Res {
  double ms;
  unsigned long long sink;
  int aborted;
  int done;
};

static uint32_t* g_static = nullptr;
static uint32_t* g_ring = nullptr;
static unsigned long long* g_head = nullptr;
static unsigned long long* g_tail = nullptr;
static unsigned long long* g_sink = nullptr;
static int* g_abort = nullptr;
static int* g_done = nullptr;
static uint4* g_data = nullptr;
static unsigned long long* g_dref = nullptr;
static long long g_static_cap = 0, g_ring_cap = 0, g_ht_cap = 0, g_data_cap = 0;

static void ensure(long long static_words, long long ring_words, int G, long long data_u4) {
  if (static_words > g_static_cap) {
    if (g_static) CK(cudaFree(g_static));
    CK(cudaMalloc(&g_static, static_words * 4));
    g_static_cap = static_words;
  }
  if (ring_words > g_ring_cap) {
    if (g_ring) CK(cudaFree(g_ring));
    CK(cudaMalloc(&g_ring, ring_words * 4));
    g_ring_cap = ring_words;
  }
  if (G > g_ht_cap) {
    if (g_head) CK(cudaFree(g_head));
    if (g_tail) CK(cudaFree(g_tail));
    CK(cudaMalloc(&g_head, (size_t)G * 8));
    CK(cudaMalloc(&g_tail, (size_t)G * 8));
    g_ht_cap = G;
  }
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_abort) CK(cudaMalloc(&g_abort, 4));
  if (!g_done) CK(cudaMalloc(&g_done, 4));
  if (!g_dref) CK(cudaMalloc(&g_dref, 8));
}

static unsigned long long host_ref_desc(int NB, int K, int DW) {
  unsigned long long s = 0;
  for (long long g = 0; g < (long long)NB * K; ++g) s += fold_desc((uint32_t)g, DW);
  return s;
}

static int g_CW = 0;   // worker 每 batch 的「计算重量」（纯 ALU 延迟）
static long long g_MD = 0;  // worker 每 batch 的「访存重量」（uint4 数）

// 返回 (descriptor 参考) + (数据面参考)，数据面参考由 fill_data_kernel 原子累加
static unsigned long long prepare_data(long long NB, long long MD) {
  if (MD <= 0) return 0;
  if (!g_dref) CK(cudaMalloc(&g_dref, 8));
  long long need_u4 = NB * MD;
  if (need_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)need_u4 * 16));
    g_data_cap = need_u4;
  }
  CK(cudaMemset(g_dref, 0, 8));
  long long tot = NB * MD;
  fill_data_kernel<<<(int)((tot + 255) / 256), 256>>>(g_data, NB, MD, g_dref);
  unsigned long long dref = 0;
  CK(cudaMemcpy(&dref, g_dref, 8, cudaMemcpyDeviceToHost));
  return dref;
}

static Res run(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode, int reps,
               int T) {
  static bool attr_set = false;
  if (!attr_set) {
    CK(cudaFuncSetAttribute(spr_mem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            200 * 1024));
    attr_set = true;
  }
  const long long desc_words = (long long)NB * K * DW;
  const long long ring_words = (long long)G * CAPp * K * DW;
  ensure(desc_words, ring_words, G, (long long)NB * g_MD);
  if (mode == 0)
    fill_static_kernel<<<(int)(((long long)NB * K + 255) / 256), 256>>>(g_static, NB, K, DW);
  CK(cudaMemset(g_head, 0, G * 8));
  CK(cudaMemset(g_tail, 0, G * 8));
  CK(cudaMemset(g_sink, 0, 8));
  CK(cudaMemset(g_abort, 0, 4));
  CK(cudaMemset(g_done, 0, 4));

  int grid = (mode == 0) ? G : PS + G;
  size_t req = (size_t)D * K * DW * 4 + (size_t)D * 8 + 64;
  unsigned long long sink = 0;
  double best = 1e30;
  int aborted = 0, done = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaMemset(g_done, 0, 4));
    CK(cudaMemset(g_head, 0, G * 8));
    CK(cudaMemset(g_tail, 0, G * 8));
    CK(cudaEventRecord(e0));
    spr_mem_kernel<<<grid, T, req>>>(PS, G, NB, DW, K, D, CAPp, mode, g_CW, g_MD, g_static,
                                     g_ring, g_head, g_tail, g_sink, g_abort, g_done, g_data);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) best = ms;
    CK(cudaMemcpy(&aborted, g_abort, 4, cudaMemcpyDeviceToHost));
    if (aborted) break;
  }
  CK(cudaMemcpy(&sink, g_sink, 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&done, g_done, 4, cudaMemcpyDeviceToHost));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  Res res;
  res.ms = best;
  res.sink = sink;
  res.aborted = aborted;
  res.done = done;
  return res;
}

static void run_line(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode,
                     int reps, unsigned long long ref, int T) {
  Res r = run(PS, G, NB, DW, K, D, CAPp, mode, reps, T);
  const char* verdict;
  if (r.aborted)
    verdict = "ABORT";
  else if (mode == 0)
    verdict = "REF";
  else if (g_CW > 0)
    verdict = "RUN";  // CW>0：sink 语义含 ALU 干扰路径，仅查不死锁
  else if (r.sink == ref)
    verdict = "OK";
  else
    verdict = "WRONG";
  double desc_gbs = (double)((long long)NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  double task_gbs = (double)((long long)NB * g_MD * 16) / (r.ms * 1e-3) / 1e9;
  int wpp = (G + PS - 1) / PS;
  printf("  PS=%-3d G=%-3d wpp=%-3d NB=%-5d K=%-3d DW=%-3d D=%-2d CAPp=%-4d CW=%-6d "
         "MD=%-7lld grid=%-4d -> %-6s ms=%.4f descGB/s=%.1f taskGB/s=%.1f\n",
         PS, G, wpp, NB, K, DW, D, CAPp, g_CW, g_MD, PS + G, verdict, r.ms, desc_gbs,
         task_gbs);
}

static void bench() {
  const int DW = 88;  // 352 B，对齐 MPK §6.1
  const int K = 32;
  const int T = 256;

  printf("########################################################################\n");
  printf("# §59 真实 HBM 数据面下的 producer:worker 配比：PS* 随「访存重量 MD」的标度\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，T=%d，prefix 协议，grid=PS+G\n", DW, K, T);
  printf("########################################################################\n");

  const int G = 100;
  const int Ps[] = {1, 2, 4, 8, 16, 32};
  const int NP = 6;

  // ===== §A 正确性：数据面读不改数值（逐位对 host 参考）=====
  {
    const int NB = 512;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref_desc(NB, K, DW);
    printf("\n===== §A 正确性（宽松环 CAPp=chunk=%d，PS=4，D=2）：MD 不计入 vs 计入 =====\n",
           chunk);
    unsigned long long mds[] = {0, 1024, 8192};
    for (int i = 0; i < 3; ++i) {
      g_MD = mds[i];
      unsigned long long dref = prepare_data(NB, g_MD);
      run_line(4, G, NB, DW, K, 2, chunk, 1, 3, ref + dref, T);
    }
    g_MD = 0;
  }

  // ===== §B headline：PS* 随「访存重量 MD」（CW=0，纯访存型 task）=====
  {
    const int NB = 1000;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref_desc(NB, K, DW);
    long long mds[] = {0, 8192, 32768, 131072};  // 0 / 128KB / 512KB / 2MB per batch
    printf("\n===== §B headline：访存重量 MD × producer 数 PS（G=%d，CW=0，D=2，NB=%d）=====\n",
           G, NB);
    for (int ci = 0; ci < 4; ++ci) {
      g_MD = mds[ci];
      unsigned long long dref = prepare_data(NB, g_MD);
      printf("  --- MD=%lld (%.1f KB/batch) ---\n", g_MD, (double)g_MD * 16 / 1024.0);
      for (int i = 0; i < NP; ++i)
        run_line(Ps[i], G, NB, DW, K, 2, chunk, 1, 3, ref + dref, T);
    }
    g_MD = 0;
  }

  // ===== §C 对照：PS* 随「计算重量 CW」（复现 §58，MD=0）=====
  {
    const int NB = 1000;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref_desc(NB, K, DW);
    int CWs[] = {0, 65536, 262144};
    printf("\n===== §C 对照：计算重量 CW × PS（G=%d，MD=0，D=2，NB=%d，复现 §58）=====\n", G,
           NB);
    for (int ci = 0; ci < 3; ++ci) {
      g_CW = CWs[ci];
      g_MD = 0;
      printf("  --- CW=%d ---\n", g_CW);
      for (int i = 0; i < NP; ++i)
        run_line(Ps[i], G, NB, DW, K, 2, chunk, 1, 3, ref, T);
    }
    g_CW = 0;
  }

  // ===== §D 混合：CW=65536 × MD 扫描（两者谁主导 PS*）=====
  {
    const int NB = 1000;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref_desc(NB, K, DW);
    long long mds[] = {0, 8192, 32768};
    printf("\n===== §D 混合：CW=65536 × MD 扫描（G=%d，D=2，NB=%d）=====\n", G, NB);
    g_CW = 65536;
    for (int ci = 0; ci < 3; ++ci) {
      g_MD = mds[ci];
      printf("  --- CW=%d MD=%lld ---\n", g_CW, g_MD);
      for (int i = 0; i < NP; ++i)
        run_line(Ps[i], G, NB, DW, K, 2, chunk, 1, 3, ref, T);
    }
    g_CW = 0;
    g_MD = 0;
  }

  // ===== §E HBM 争用：MD=131072 下扫 PS，看 achieved taskGB/s 是否随 PS 变化 =====
  {
    const int NB = 1000;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref_desc(NB, K, DW);
    g_MD = 131072;
    unsigned long long dref = prepare_data(NB, g_MD);
    printf("\n===== §E HBM 争用：MD=131072 × PS（看 achieved taskGB/s，NB=%d）=====\n", NB);
    for (int i = 0; i < NP; ++i)
      run_line(Ps[i], G, NB, DW, K, 2, chunk, 1, 3, ref + dref, T);
    g_MD = 0;
  }

  printf("\n########## §59 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 10) { printf("usage: run PS G NB DW K D CAPp reps [CW] [MD]\n"); return 1; }
    int PS = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), CAPp = atoi(argv[8]), reps = atoi(argv[9]);
    if (argc >= 11) g_CW = atoi(argv[10]);
    if (argc >= 12) g_MD = atoll(argv[11]);
    unsigned long long ref = host_ref_desc(NB, K, DW) + prepare_data(NB, g_MD);
    run_line(PS, G, NB, DW, K, D, CAPp, 1, reps, ref, 256);
    return 0;
  }
  printf("usage: sched_perworker_ring_mem {bench|run ...}\n");
  return 1;
}
