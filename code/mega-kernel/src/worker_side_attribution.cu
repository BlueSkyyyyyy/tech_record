// ============================================================================
// P76-1 (§64)：worker 侧成本的「复杂度阶梯 × 环数 G」归因
//
// 背景（接 §62 / §63）：
//   - §62 在**完整端到端骨架**（producer 写 per-worker 环 + worker TMA 取 task + SMEM 折叠 +
//     HBM 数据面）上发现：固定 NB=2048、固定 PS=8 时，端到端 taskGB/s 随 worker 环数 G 增大
//     **反而下降**（2058@G=528 → 1768@G=1040），并把成本归因到「少 producer 轮转服务多
//     per-worker 环的扇出」（§62.7）。
//   - §63 把 **producer 写端单独抽出来**，证明 producer 服务成本是**纯每批**、**与环数无关**
//     （`wpp=1→64` 吞吐持平），且加回 **plain-load consumer** 也只加一笔固定开销、不随环数
//     放大——**§62.7 的「producer 环扇出」被隔离证伪**，§62 的 `G` 依赖只能住在 **worker 侧**。
//   - 但 §63.9 明留尾巴：「consumer 是 **plain-load**（非 TMA/非折叠/非数据面），故『加 consumer
//     仍平』只能证『环数本身无成本』，**不能**直接证明 §62 的 `G` 依赖来自 TMA 路径——那需要
//     把 TMA+折叠+数据面按 §62 加回」。
//
// 本节把 **worker 侧**放上一条**复杂度阶梯**，在**同一二进制、同一占用率**下扫 `G`，逐层加回
//   worker 机制，把 §62 的 `G` 依赖钉到**哪一层 worker 机制**上：
//     wmode 0 = plain ：worker 直接 `ld` 读环到寄存器 + 折叠（= §63 的 consumer，无 TMA/无 SMEM）；
//     wmode 1 = tma   ：worker **TMA bulk 取 task 进 SMEM** + 从 SMEM 折叠（= 加回 MPK 真载体）；
//     wmode 2 = tma+d ：再加 **HBM 数据面**（`__ldcs` 流式读 MD 个 uint4）= §62 的 full；
//     wmode 3 = data  ：无 producer/无环/无 TMA，worker 只做 HBM 数据面（占用率参照/§62 mode2）。
//   占用率固定 `T=256,MINB=8,U=4`（8 blk/SM，= §62 §D 配置），故 `G` 是唯一变量。
//   两条扫描：(i) 固定总工作量 `NB`（`chunk=⌈NB/G⌉` 随 G 变小）；(ii) `NB=4·G`（chunk 恒定=4，
//   分离「块数 G」与「每环批数 chunk」）——直接回答 §62/§63 遗留的「G 依赖来自哪一层」。
//   tma/tma+d 的 SMEM 每批 `K·DW` 字 + mbarrier，占用率由 `__launch_bounds__(256,8)` + opt-in
//   SMEM（~22 KB/blk × 8 ≤ 228 KB/SM）保证与 plain 同档。
//
// 用法：./worker_side_attribution bench
//       ./worker_side_attribution run <PS> <G> <NB> <DW> <K> <D> <reps> <wmode> [MD] [U] [T] [MINB]
// 目标卡 H100 / sm90a。安全：脚本带 timeout；设备侧 clock64 死锁守门狗。
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
__host__ __device__ __forceinline__ uint32_t data_word(unsigned long long uidx, int j) {
  return mix32((uint32_t)(uidx) ^ ((uint32_t)(uidx >> 32) * 2654435761u),
               (uint32_t)j * 101u + 7u);
}

// ---- 内存序原语（对齐 MPK 队列协议）----
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

// ---- 硬件 mbarrier + TMA bulk load ----
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
// kernel：block 0..PS-1 = producer（wmode 0/1/2），block 后段 = worker
//   wmode 0 = plain（无 TMA/SMEM），1 = TMA，2 = TMA+数据面，3 = 仅数据面（无 producer）
// ============================================================================
template <int U>
__device__ __forceinline__ void wa_body(int PS, int G, int NB, int DW, int K, int D, int wmode,
                                        long long MD, uint32_t* __restrict__ ring,
                                        unsigned long long* head, unsigned long long* tail,
                                        unsigned long long* sink, int* abort, int* dbg_done,
                                        const uint4* __restrict__ g_data, char* smem_raw) {
  const int tid = threadIdx.x;
  const int nlanes = blockDim.x;
  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_raw);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int bytes_per_batch = K * DW * 4;
  const int chunk = (NB + G - 1) / G;
  const int CAPp = chunk;
  const int nb = NB;

  auto count_of = [&](int w) -> int {
    int b0 = w * chunk;
    if (b0 >= nb) return 0;
    int c = nb - b0;
    return (c < chunk) ? c : chunk;
  };

  // ---------------- producer：block 0..PS-1（wmode 0/1/2）----------------
  if (wmode != 3 && blockIdx.x < PS) {
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
        uint32_t* dst = ring + (size_t)((size_t)w * CAPp + slot) * K * DW;
        const int bb = w * chunk + r;
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

  // ---------------- worker ----------------
  const int w = (wmode == 3) ? blockIdx.x : (blockIdx.x - PS);
  if (w >= G) return;
  const int nt = count_of(w);
  const int b0w = w * chunk;
  const int use_tma = (wmode == 1 || wmode == 2);

  if (use_tma) {
    if (tid == 0)
      for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
    __syncthreads();
  }

  unsigned long long acc = 0;

  auto wait_ready = [&](int bl) -> bool {
    if (wmode == 3) return true;
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
    if (!use_tma) return true;
    int slot = t % D;
    if (!wait_ready(t)) return false;
    const uint32_t* src = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
    if (tid == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
      bulk_load_1d(smem + (size_t)slot * K * DW, src, (uint32_t)bytes_per_batch, &mbar[slot]);
    }
    return true;
  };

  bool ok = true;
  if (use_tma)
    for (int t = 0; t < D && t < nt; ++t)
      if (!issue(t)) { ok = false; break; }

  __shared__ unsigned long long wred[32];
  const int nwarp = (nlanes + 31) >> 5;
  for (int t = 0; t < nt && ok; ++t) {
    unsigned long long part = 0;
    if (wmode != 3) {
      if (!wait_ready(t)) { ok = false; break; }
      const uint32_t* buf;
      if (use_tma) {
        int slot = t % D;
        uint32_t phase = (uint32_t)((t / D) & 1u);
        mbar_wait(&mbar[slot], phase);
        buf = smem + (size_t)slot * K * DW;
      } else {
        buf = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
      }
      for (int j = tid; j < K; j += nlanes) {
        uint32_t a = 0x9e3779b9u;
        for (int i = 0; i < DW; ++i) a = a * 2246822519u + buf[(size_t)j * DW + i];
        part += (unsigned long long)a;
      }
    }
    if (MD > 0) {  // 真实 HBM 数据面
      const int bb = b0w + t;
      const uint4* dp = g_data + (size_t)bb * (size_t)MD;
      unsigned long long accU[U];
      for (int u = 0; u < U; ++u) accU[u] = 0ull;
      const long long strd = (long long)nlanes * U;
      long long m = tid;
      for (; m + (long long)(U - 1) * nlanes < MD; m += strd) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
          uint4 v = __ldcs(dp + m + (long long)u * nlanes);
          accU[u] += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
        }
      }
      for (; m < MD; m += nlanes) {
        uint4 v = __ldcs(dp + m);
        accU[0] += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
      }
      unsigned long long dpart = 0;
      for (int u = 0; u < U; ++u) dpart += accU[u];
      part += dpart;
    }
    unsigned long long v = part;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((tid & 31) == 0) wred[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
      unsigned long long ssum = 0;
      for (int i = 0; i < nwarp; ++i) ssum += wred[i];
      acc += ssum;
    }
    __syncthreads();
    if (wmode != 3 && tid == 0) atom_add_release_u64(tail + w, 1ull);
    if (use_tma && t + D < nt)
      if (!issue(t + D)) { ok = false; break; }
  }

  if (acc && tid == 0) atomicAdd(sink, acc);
  if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
}

template <int U, int TB, int MINB>
__global__ void __launch_bounds__(TB, MINB) wa_kernel(
    int PS, int G, int NB, int DW, int K, int D, int wmode, long long MD,
    uint32_t* __restrict__ ring, unsigned long long* head, unsigned long long* tail,
    unsigned long long* sink, int* abort, int* dbg_done, const uint4* __restrict__ g_data) {
  extern __shared__ char smem_raw[];
  wa_body<U>(PS, G, NB, DW, K, D, wmode, MD, ring, head, tail, sink, abort, dbg_done, g_data,
             smem_raw);
}

// ============================================================================
// host
// ============================================================================
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
  int unsupported;
};

static uint32_t* g_ring = nullptr;
static unsigned long long* g_head = nullptr;
static unsigned long long* g_tail = nullptr;
static unsigned long long* g_sink = nullptr;
static int* g_abort = nullptr;
static int* g_done = nullptr;
static uint4* g_data = nullptr;
static unsigned long long* g_dref = nullptr;
static long long g_ring_cap = 0, g_ht_cap = 0, g_data_cap = 0;

static void ensure(long long ring_words, int G, long long data_u4) {
  if (ring_words > g_ring_cap) {
    if (g_ring) CK(cudaFree(g_ring));
    CK(cudaMalloc(&g_ring, (size_t)ring_words * 4));
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

static int g_U = 4;
static int g_TB = 256;
static int g_MINB = 8;
static int g_last_launch_ok = 1;

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

static void launch(int grid, int T, size_t req, int PS, int G, int NB, int DW, int K, int D,
                   int wmode, long long MD) {
#define LAUNCH_FOR(UU, TT, MB)                                                            \
  do {                                                                                    \
    static bool a_##UU##_##TT##_##MB = false;                                             \
    if (!a_##UU##_##TT##_##MB) {                                                          \
      CK(cudaFuncSetAttribute(wa_kernel<UU, TT, MB>,                                      \
                              cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));   \
      a_##UU##_##TT##_##MB = true;                                                        \
    }                                                                                     \
    wa_kernel<UU, TT, MB><<<grid, T, req>>>(PS, G, NB, DW, K, D, wmode, MD, g_ring,       \
                                            g_head, g_tail, g_sink, g_abort, g_done,      \
                                            g_data);                                      \
    g_last_launch_ok = (cudaGetLastError() == cudaSuccess);                               \
  } while (0)

#define TRY(UU, TT, MB) if (g_U == UU && g_TB == TT && g_MINB == MB) { LAUNCH_FOR(UU, TT, MB); return; }
  TRY(1, 256, 8); TRY(4, 256, 8); TRY(8, 256, 8);
  TRY(4, 512, 4); TRY(4, 1024, 2);
#undef TRY
  LAUNCH_FOR(4, 256, 8);
#undef LAUNCH_FOR
}

static Res run(int PS, int G, int NB, int DW, int K, int D, int wmode, long long MD,
               int reps, int T) {
  const long long ring_words = (long long)G * ((NB + G - 1) / G) * K * DW;
  ensure(ring_words, G, (long long)NB * MD);
  if (wmode != 3) {
    CK(cudaMemset(g_head, 0, G * 8));
    CK(cudaMemset(g_tail, 0, G * 8));
  }

  int grid = (wmode == 3) ? G : (PS + G);
  size_t req = (size_t)D * K * DW * 4 + (size_t)D * 8 + 64;
  unsigned long long sink = 0;
  double best = 1e30;
  int aborted = 0, done = 0, unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaMemset(g_done, 0, 4));
    if (wmode != 3) {
      CK(cudaMemset(g_head, 0, G * 8));
      CK(cudaMemset(g_tail, 0, G * 8));
    }
    CK(cudaEventRecord(e0));
    launch(grid, T, req, PS, G, NB, DW, K, D, wmode, MD);
    if (!g_last_launch_ok) { unsupported = 1; break; }
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
  Res res; res.ms = best; res.sink = sink; res.aborted = aborted; res.done = done;
  res.unsupported = unsupported;
  return res;
}

static const char* wmode_name(int wmode) {
  switch (wmode) {
    case 0: return "plain";
    case 1: return "tma";
    case 2: return "tma+data";
    case 3: return "data-only";
  }
  return "?";
}

static void run_line(int PS, int G, int NB, int DW, int K, int D, int wmode, long long MD,
                     int reps, unsigned long long ref, const char* tag) {
  Res r = run(PS, G, NB, DW, K, D, wmode, MD, reps, g_TB);
  const char* verdict;
  if (r.unsupported) verdict = "UNSUP";
  else if (r.aborted) verdict = "ABORT";
  else verdict = (r.sink == ref) ? "OK" : "WRONG";
  double desc_gbs = (double)((long long)NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  double task_gbs = (double)((long long)NB * MD * 16) / (r.ms * 1e-3) / 1e9;
  int chunk = (NB + G - 1) / G;
  printf("  %-9s %-10s PS=%-3d G=%-4d chunk=%-3d NB=%-4d D=%d MD=%-7lld U=%d -> %-6s "
         "ms=%.4f descGB/s=%-7.1f taskGB/s=%-7.1f\n",
         (tag ? tag : ""), wmode_name(wmode), PS, G, chunk, NB, D, MD, g_U, verdict, r.ms,
         desc_gbs, task_gbs);
}

// ============================================================================
// bench
// ============================================================================
static void bench() {
  const int DW = 88;
  const int K = 32;
  const int D = 2;
  const int PS = 8;
  const int NB = 2048;
  const long long MD = 65536;  // 1 MiB/batch -> 全图数据面总读 ≈ 2.0 GiB
  const int Gs[] = {100, 132, 264, 528, 1040};
  const int NG = 5;

  printf("########################################################################\n");
  printf("# §64 worker 侧复杂度阶梯 × 环数 G 归因（接 §62/§63）\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，D=%d，PS=%d，占用率固定 T=%d,MINB=%d,U=%d\n",
         DW, K, D, PS, g_TB, g_MINB, g_U);
  printf("# 阶梯：0=plain(寄存器读环+折叠) 1=+TMA(SMEM 取 task) 2=+HBM 数据面 3=仅数据面\n");
  printf("########################################################################\n");

  // ===== §A 正确性 =====
  {
    const int G = 100, NBc = 512;
    g_U = 4; g_TB = 256; g_MINB = 8;
    unsigned long long dref = prepare_data(NBc, 8192);
    printf("\n===== §A 正确性（G=%d，NB=%d，MD=8192）：plain/tma/tma+data/data-only =====\n",
           G, NBc);
    printf("  (host ref: desc=%llu data=%llu)\n", host_ref_desc(NBc, K, DW), dref);
    for (int wm = 0; wm <= 3; ++wm) {
      long long md = (wm == 3 || wm == 2) ? 8192 : 0;
      unsigned long long ref = (wm == 3) ? dref
                                         : host_ref_desc(NBc, K, DW) + ((md > 0) ? dref : 0);
      run_line(PS, G, NBc, DW, K, D, wm, md, 3, ref, "A");
    }
  }

  // ===== §B headline：固定 NB，扫 G（chunk 随 G 变小），三档阶梯 =====
  printf("\n===== §B 固定 NB=%d，扫 G（chunk=⌈NB/G⌉），三档阶梯：谁带来 G 依赖？=====\n", NB);
  {
    unsigned long long dref = prepare_data(NB, MD);
    unsigned long long refd = host_ref_desc(NB, K, DW);
    for (int i = 0; i < NG; ++i) {
      int G = Gs[i];
      run_line(PS, G, NB, DW, K, D, 0, 0, 3, refd, "B-plain");
      run_line(PS, G, NB, DW, K, D, 1, 0, 3, refd, "B-tma");
      run_line(PS, G, NB, DW, K, D, 0, MD, 3, refd + dref, "B-plain+d");
      run_line(PS, G, NB, DW, K, D, 2, MD, 3, refd + dref, "B-tma+d");
      run_line(0, G, NB, DW, K, D, 3, MD, 3, dref, "B-data");
    }
  }

  // ===== §C 消歧：NB=4·G（chunk 恒=4），分离「块数 G」与「每环 chunk =====
  printf("\n===== §C 消歧：NB=4·G（chunk 恒=4），扫 G——分离块数 G 与每环批数 chunk =====\n");
  {
    for (int i = 0; i < NG; ++i) {
      int G = Gs[i];
      int NBx = 4 * G;
      unsigned long long dref = prepare_data(NBx, MD);
      unsigned long long refd = host_ref_desc(NBx, K, DW);
      run_line(PS, G, NBx, DW, K, D, 0, 0, 3, refd, "C-plain");
      run_line(PS, G, NBx, DW, K, D, 1, 0, 3, refd, "C-tma");
      run_line(PS, G, NBx, DW, K, D, 0, MD, 3, refd + dref, "C-plain+d");
      run_line(PS, G, NBx, DW, K, D, 2, MD, 3, refd + dref, "C-tma+d");
      run_line(0, G, NBx, DW, K, D, 3, MD, 3, dref, "C-data");
    }
    prepare_data(NB, MD);
  }

  // ===== §D MD 扫描：固定 G=264、plain/tma/tma+data，看数据面重量 =====
  printf("\n===== §D 固定 G=264：扫数据面 MD（0→131072），plain vs tma vs tma+data =====\n");
  {
    const int G = 264;
    long long mds[] = {0, 8192, 32768, 65536, 131072};
    for (int i = 0; i < 5; ++i) {
      long long md = mds[i];
      unsigned long long dref = prepare_data(NB, md);
      unsigned long long refd = host_ref_desc(NB, K, DW);
      run_line(PS, G, NB, DW, K, D, 0, md, 3, refd + dref, "D-plain");
      run_line(PS, G, NB, DW, K, D, 1, md, 3, refd + dref, "D-tma");
      run_line(PS, G, NB, DW, K, D, 2, md, 3, refd + dref, "D-tma+d");
    }
    prepare_data(NB, MD);
  }

  // ===== §E 占用率报告 =====
  printf("\n===== §E 占用率/寄存器报告 =====\n");
  {
    cudaFuncAttributes at;
    int nb = 0;
#define REP(KER, TB, LABEL)                                                              \
    do {                                                                                 \
      CK(cudaFuncGetAttributes(&at, KER));                                               \
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, KER, TB, 22608));            \
      printf("  %-22s numRegs=%-4d localB=%-3zu -> blk/SM(T=%d,smem=22608)=%d\n", LABEL, \
             at.numRegs, at.localSizeBytes, TB, nb);                                     \
    } while (0)
    REP((wa_kernel<4, 256, 8>), 256, "U=4 T=256 MINB=8");
    REP((wa_kernel<4, 512, 4>), 512, "U=4 T=512 MINB=4");
    REP((wa_kernel<4, 1024, 2>), 1024, "U=4 T=1024 MINB=2");
#undef REP
  }

  printf("\n########## §64 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 10) {
      printf("usage: run PS G NB DW K D reps wmode [MD] [U] [T] [MINB]\n");
      return 1;
    }
    int PS = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), reps = atoi(argv[8]), wmode = atoi(argv[9]);
    long long MD = (argc >= 11) ? atoll(argv[10]) : 0;
    if (argc >= 12) g_U = atoi(argv[11]);
    if (argc >= 13) g_TB = atoi(argv[12]);
    if (argc >= 14) g_MINB = atoi(argv[13]);
    unsigned long long dref = prepare_data(NB, MD);
    unsigned long long ref = (wmode == 3) ? dref : host_ref_desc(NB, K, DW) + dref;
    run_line(PS, G, NB, DW, K, D, wmode, MD, reps, ref, "run");
    return 0;
  }
  printf("usage: worker_side_attribution {bench|run ...}\n");
  return 1;
}
