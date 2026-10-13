// ============================================================================
// P78-1 (§66)：把 §65 的 cp.async 单缓冲升级成**深环（模板化 D 槽预取）**，补掉 §65.9 的尾巴
//
// 背景（接 §42 / §64 / §65）：
//   - §65 把 MPK worker 的**原生**取 task 载体（16 B `cp.async` **单缓冲**进 SMEM，
//     `persistent_kernel.cuh:1063-1078` 的 `cp_async_wait<0>`）加进 §64 复杂度阶梯，量出
//     cp.async 的每 block prologue ≈ plain 寄存器读（−0.04…+0.005 µs/block），**不付 TMA 那笔
//     +0.3 µs/block**（TMA 专属的 `mbarrier.init×D` + `arrive.expect_tx` + 首次 bulk）。
//   - 但 §65.9 明留尾巴：「**cp.async 是单缓冲（快照形态），未做 `D` 深环预取**（§42 的
//     cp.async 深环已量『预取增益全在 D=1→2』）」——本节的载体比较是「**单缓冲 cp.async** vs
//     双缓冲 TMA」，并未回答：**cp.async 上装软件 `wait_group` 深环后，每 block prologue 会怎样？
//     深环能否进一步摊薄 prologue、或把单缓冲的 `wait_group 0` 串行缝也藏掉？**
//   - §42 在「K×SMEM 预算」平面（单 block）对跑 cp.async 深环与 TMA bulk（K≤32 cp.async 反超），
//     但那是**裸取数**平面，不是 §64/§65 的**端到端 worker 阶梯**（带 producer 环 + 数据面 + 背压）。
//
// 本节把 cp.async **深环**（模板化 `DCA` 槽，`cp.async.wait_group (DCA-1)`；尾批 `wait_group 0`
//   兜底，用 §42/§40.8 的 drain 写法）加进 §65 的同一骨架，与 plain / 单缓冲 cp.async / TMA
//   在**同一占用率**下对跑，回答：**cp.async 深环的每 block prologue 是否仍 ≈ plain？深环是否比
//   单缓冲更省？**
//   阶梯（新增两档）：
//     WMODE 0 = plain        ：worker 直接 `ld` 读环到寄存器 + 折叠（无 SMEM）；
//     WMODE 1 = tma          ：worker TMA bulk 取 task 进 SMEM（硬件 mbarrier，`D` 深环）；
//     WMODE 2 = tma+d        ：再加 HBM 数据面；                             = §64 full；
//     WMODE 3 = data         ：无 producer/环/取数，只做 HBM 数据面（占用率参照）；
//     WMODE 4 = cpasync      ：16 B `cp.async` **单缓冲**取 task（快照原生载体）；    = §65；
//     WMODE 5 = cpasync+d    ：单缓冲 cp.async + 数据面；                          = §65；
//     WMODE 6 = cpasync_ring ：16 B `cp.async` **深环**（`DCA` 槽软件流水）；      【本节新增】
//     WMODE 7 = cpasync_ring+d：深环 cp.async + 数据面。                         【本节新增】
//   占用率固定 `T=256,MINB=8,U=4`（8 blk/SM，= §62/§64/§65 配置），故 `G`/载体/`DCA` 是唯一变量。
//
// 用法：./worker_carrier_cpasync_ring bench
//       ./worker_carrier_cpasync_ring run <PS> <G> <NB> <DW> <K> <D> <reps> <WMODE> [MD] [U] [T] [MINB] [DCA]
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

// ---- 16 B cp.async（与 MPK load_smem 逐字一致的形，搬到「取 task」层） ----
__device__ __forceinline__ void load_smem(void* smem_ptr, const void* gmem_ptr) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, 16;\n" ::"r"(s),
               "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait0() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ============================================================================
// kernel：block 0..PS-1 = producer（WMODE 0/1/2/4/5/6/7），block 后段 = worker
// ============================================================================
template <int U, int DCA, int WMODE>
__device__ __forceinline__ void wcr_body(int PS, int G, int NB, int DW, int K, int D,
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

  // ---------------- producer：block 0..PS-1 ----------------
  if (WMODE != 3 && blockIdx.x < PS) {
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
  const int w = (WMODE == 3) ? blockIdx.x : (blockIdx.x - PS);
  if (w >= G) return;
  const int nt = count_of(w);
  const int b0w = w * chunk;
  constexpr bool use_tma = (WMODE == 1 || WMODE == 2);
  constexpr bool use_ca = (WMODE == 4 || WMODE == 5);
  constexpr bool use_car = (WMODE == 6 || WMODE == 7);
  const int n16 = (K * DW * 4) / 16;  // 16 B 块数

  if constexpr (use_tma) {
    if (tid == 0)
      for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
    __syncthreads();
  }

  unsigned long long acc = 0;

  auto wait_ready = [&](int bl) -> bool {
    if constexpr (WMODE == 3) return true;
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

  auto issue_tma = [&](int t) -> bool {
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
  if constexpr (use_tma)
    for (int t = 0; t < D && t < nt; ++t)
      if (!issue_tma(t)) { ok = false; break; }
  if constexpr (use_car)
    for (int t = 0; t < DCA && t < nt; ++t) {
      if (!wait_ready(t)) { ok = false; break; }
      int slot = t % DCA;
      const uint32_t* src = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
      char* dst = reinterpret_cast<char*>(smem) + (size_t)slot * K * DW * 4;
      const char* s8 = reinterpret_cast<const char*>(src);
      for (int c = tid; c < n16; c += nlanes)
        load_smem(dst + (size_t)c * 16, s8 + (size_t)c * 16);
      cp_async_commit();
    }

  __shared__ unsigned long long wred[32];
  const int nwarp = (nlanes + 31) >> 5;
  for (int t = 0; t < nt && ok; ++t) {
    unsigned long long part = 0;
    if constexpr (WMODE != 3) {
      if (!wait_ready(t)) { ok = false; break; }
      const uint32_t* buf;
      if constexpr (use_tma) {
        int slot = t % D;
        uint32_t phase = (uint32_t)((t / D) & 1u);
        mbar_wait(&mbar[slot], phase);
        buf = smem + (size_t)slot * K * DW;
      } else if constexpr (use_ca) {
        // 单缓冲 16 B cp.async：搬整批 task 进 SMEM（快照原生载体）
        const uint32_t* src0 = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
        for (int c = tid; c < n16; c += nlanes)
          load_smem(reinterpret_cast<char*>(smem) + (size_t)c * 16,
                    reinterpret_cast<const char*>(src0) + (size_t)c * 16);
        cp_async_commit();
        cp_async_wait0();
        __syncthreads();
        buf = smem;
      } else if constexpr (use_car) {
        // 深环 16 B cp.async：软件流水，`wait_group (DCA-1)`；尾批 `wait_group 0` 兜底
        int slot = t % DCA;
        if (t + DCA <= nt)
          cp_async_wait<DCA - 1>();
        else
          cp_async_wait0();
        __syncthreads();
        buf = smem + (size_t)slot * K * DW;
      } else {
        buf = ring + (size_t)((size_t)w * CAPp + (t % CAPp)) * K * DW;
      }
      for (int j = tid; j < K; j += nlanes) {
        uint32_t a = 0x9e3779b9u;
        for (int i = 0; i < DW; ++i) a = a * 2246822519u + buf[(size_t)j * DW + i];
        part += (unsigned long long)a;
      }
      // 读完整槽才能复用它发下一批
      if constexpr (use_car) __syncthreads();
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
    if (WMODE != 3 && tid == 0) atom_add_release_u64(tail + w, 1ull);
    if constexpr (use_tma)
      if (t + D < nt)
        if (!issue_tma(t + D)) { ok = false; break; }
    if constexpr (use_car)
      if (t + DCA < nt) {
        if (!wait_ready(t + DCA)) { ok = false; break; }
        int slot = (t + DCA) % DCA;
        const uint32_t* src = ring + (size_t)((size_t)w * CAPp + ((t + DCA) % CAPp)) * K * DW;
        char* dst = reinterpret_cast<char*>(smem) + (size_t)slot * K * DW * 4;
        const char* s8 = reinterpret_cast<const char*>(src);
        for (int c = tid; c < n16; c += nlanes)
          load_smem(dst + (size_t)c * 16, s8 + (size_t)c * 16);
        cp_async_commit();
      }
  }

  if (acc && tid == 0) atomicAdd(sink, acc);
  if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
}

template <int U, int TB, int MINB, int DCA, int WMODE>
__global__ void __launch_bounds__(TB, MINB) wcr_kernel(
    int PS, int G, int NB, int DW, int K, int D, long long MD,
    uint32_t* __restrict__ ring, unsigned long long* head, unsigned long long* tail,
    unsigned long long* sink, int* abort, int* dbg_done, const uint4* __restrict__ g_data) {
  extern __shared__ __align__(16) char smem_raw[];
  wcr_body<U, DCA, WMODE>(PS, G, NB, DW, K, D, MD, ring, head, tail, sink, abort, dbg_done,
                          g_data, smem_raw);
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
static int g_DCA = 2;
static int g_K = 32;
static int g_DW = 88;
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

static int g_wmode_call = 0;
static void launch(int grid, int T, size_t req, int PS, int G, int NB, int DW, int K, int D,
                   int WMODE, long long MD) {
  const int dca = (WMODE == 6 || WMODE == 7) ? g_DCA : 1;  // 载体选择 = 编译期 template
  g_wmode_call = WMODE;
#define LAUNCH_FOR(UU, TT, MB, DC, WM)                                                   \
  do {                                                                                   \
    static bool a_##UU##_##TT##_##MB##_##DC##_##WM = false;                              \
    if (!a_##UU##_##TT##_##MB##_##DC##_##WM) {                                           \
      CK(cudaFuncSetAttribute(wcr_kernel<UU, TT, MB, DC, WM>,                            \
                              cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));  \
      a_##UU##_##TT##_##MB##_##DC##_##WM = true;                                         \
    }                                                                                    \
    wcr_kernel<UU, TT, MB, DC, WM><<<grid, T, req>>>(PS, G, NB, DW, K, D, MD, g_ring,    \
                                                     g_head, g_tail, g_sink, g_abort,    \
                                                     g_done, g_data);                    \
    g_last_launch_ok = (cudaGetLastError() == cudaSuccess);                              \
  } while (0)

#define TRYF(UU, TT, MB, DC, WM)                                                          \
  if (g_U == UU && g_TB == TT && g_MINB == MB && dca == DC && g_wmode_call == WM) {       \
    LAUNCH_FOR(UU, TT, MB, DC, WM);                                                       \
    return;                                                                               \
  }
  TRYF(4, 256, 8, 1, 0); TRYF(4, 256, 8, 1, 1); TRYF(4, 256, 8, 1, 2);
  TRYF(4, 256, 8, 1, 3); TRYF(4, 256, 8, 1, 4); TRYF(4, 256, 8, 1, 5);
  TRYF(4, 256, 8, 1, 6); TRYF(4, 256, 8, 2, 6); TRYF(4, 256, 8, 4, 6); TRYF(4, 256, 8, 8, 6);
  TRYF(4, 256, 8, 1, 7); TRYF(4, 256, 8, 2, 7); TRYF(4, 256, 8, 4, 7); TRYF(4, 256, 8, 8, 7);
#undef TRYF
  LAUNCH_FOR(4, 256, 8, 1, 0);  // 兜底：未列出的组合退化为 plain（正常 bench 不会命中）
#undef LAUNCH_FOR
}

static Res run(int PS, int G, int NB, int DW, int K, int D, int WMODE, long long MD,
               int reps, int T) {
  const long long ring_words = (long long)G * ((NB + G - 1) / G) * K * DW;
  ensure(ring_words, G, (long long)NB * MD);
  if (WMODE != 3) {
    CK(cudaMemset(g_head, 0, G * 8));
    CK(cudaMemset(g_tail, 0, G * 8));
  }

  int grid = (WMODE == 3) ? G : (PS + G);
  size_t req;
  if (WMODE == 1 || WMODE == 2)
    req = (size_t)D * K * DW * 4 + (size_t)D * 8 + 64;  // TMA 深环 + mbarrier
  else if (WMODE == 6 || WMODE == 7)
    req = (size_t)g_DCA * K * DW * 4 + 64;              // cp.async 深环（DCA 槽）
  else if (WMODE == 4 || WMODE == 5)
    req = (size_t)K * DW * 4 + 64;                      // cp.async 单缓冲
  else
    req = 64;
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
    if (WMODE != 3) {
      CK(cudaMemset(g_head, 0, G * 8));
      CK(cudaMemset(g_tail, 0, G * 8));
    }
    CK(cudaEventRecord(e0));
    launch(grid, T, req, PS, G, NB, DW, K, D, WMODE, MD);
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

static const char* wmode_name(int WMODE) {
  switch (WMODE) {
    case 0: return "plain";
    case 1: return "tma";
    case 2: return "tma+data";
    case 3: return "data-only";
    case 4: return "cpasync";
    case 5: return "cpasync+d";
    case 6: return "ca-ring";
    case 7: return "ca-ring+d";
  }
  return "?";
}

static void run_line(int PS, int G, int NB, int DW, int K, int D, int WMODE, long long MD,
                     int reps, unsigned long long ref, const char* tag) {
  Res r = run(PS, G, NB, DW, K, D, WMODE, MD, reps, g_TB);
  const char* verdict;
  if (r.unsupported) verdict = "UNSUP";
  else if (r.aborted) verdict = "ABORT";
  else verdict = (r.sink == ref) ? "OK" : "WRONG";
  double desc_gbs = (double)((long long)NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  double task_gbs = (double)((long long)NB * MD * 16) / (r.ms * 1e-3) / 1e9;
  int chunk = (NB + G - 1) / G;
  printf("  %-13s %-12s PS=%-3d G=%-4d chunk=%-3d NB=%-4d D=%d DCA=%-2d MD=%-7lld U=%d -> "
         "%-6s ms=%.4f descGB/s=%-7.1f taskGB/s=%-7.1f\n",
         (tag ? tag : ""), wmode_name(WMODE), PS, G, chunk, NB, D, g_DCA, MD, g_U, verdict,
         r.ms, desc_gbs, task_gbs);
}

// ============================================================================
// bench
// ============================================================================
static void bench() {
  const int DW = g_DW;
  const int K = g_K;
  const int D = 2;
  const int PS = 8;
  const int NB = 2048;
  const long long MD = 65536;  // 1 MiB/batch
  const int Gs[] = {100, 132, 264, 528, 1040};
  const int NG = 5;

  printf("########################################################################\n");
  printf("# §66 worker 侧取 task 载体第四种：cp.async **深环**（补 §65.9 的尾巴）\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，TMA D=%d，PS=%d，占用率固定 T=%d,MINB=%d,U=%d\n",
         DW, K, D, PS, g_TB, g_MINB, g_U);
  printf("# 阶梯：0=plain 1=tma 2=tma+data 3=data-only 4=cpasync 5=cpasync+data "
         "6=ca-ring 7=ca-ring+data\n");
  printf("########################################################################\n");

  // ===== §A 正确性：环深扫描（单缓冲 + 深环 DCA=2/4/8 全对）=====
  {
    const int G = 100, NBc = 512;
    g_U = 4; g_TB = 256; g_MINB = 8;
    unsigned long long dref = prepare_data(NBc, 8192);
    printf("\n===== §A 正确性（G=%d，NB=%d，MD=8192）：单缓冲 vs 深环 DCA=2/4/8 =====\n", G, NBc);
    printf("  (host ref: desc=%llu data=%llu)\n", host_ref_desc(NBc, K, DW), dref);
    run_line(PS, G, NBc, DW, K, D, 0, 0, 3, host_ref_desc(NBc, K, DW), "A-plain");
    run_line(PS, G, NBc, DW, K, D, 4, 0, 3, host_ref_desc(NBc, K, DW), "A-ca-single");
    for (int dca : {2, 4, 8}) {
      g_DCA = dca;
      run_line(PS, G, NBc, DW, K, D, 6, 0, 3, host_ref_desc(NBc, K, DW), "A-ca-ring");
    }
    g_DCA = 2;
    run_line(PS, G, NBc, DW, K, D, 5, 8192, 3, host_ref_desc(NBc, K, DW) + dref, "A-ca-single+d");
    for (int dca : {2, 4, 8}) {
      g_DCA = dca;
      run_line(PS, G, NBc, DW, K, D, 7, 8192, 3, host_ref_desc(NBc, K, DW) + dref, "A-ca-ring+d");
    }
    g_DCA = 2;
  }

  // ===== §B headline：固定 NB，扫 G，单缓冲 cp.async vs 深环 cp.async vs plain/TMA =====
  printf("\n===== §B 固定 NB=%d，扫 G：cp.async **深环** vs 单缓冲 vs plain/TMA =====\n", NB);
  {
    unsigned long long dref = prepare_data(NB, MD);
    unsigned long long refd = host_ref_desc(NB, K, DW);
    for (int i = 0; i < NG; ++i) {
      int G = Gs[i];
      run_line(PS, G, NB, DW, K, D, 0, 0, 3, refd, "B-plain");
      run_line(PS, G, NB, DW, K, D, 1, 0, 3, refd, "B-tma");
      g_DCA = 1;
      run_line(PS, G, NB, DW, K, D, 6, 0, 3, refd, "B-caD1");
      g_DCA = 2;
      run_line(PS, G, NB, DW, K, D, 6, 0, 3, refd, "B-caD2");
      g_DCA = 4;
      run_line(PS, G, NB, DW, K, D, 6, 0, 3, refd, "B-caD4");
      g_DCA = 2;
      run_line(PS, G, NB, DW, K, D, 0, MD, 3, refd + dref, "B-plain+d");
      run_line(PS, G, NB, DW, K, D, 2, MD, 3, refd + dref, "B-tma+d");
      run_line(PS, G, NB, DW, K, D, 4, MD, 3, refd + dref, "B-caSingle+d");
      g_DCA = 2;
      run_line(PS, G, NB, DW, K, D, 6, MD, 3, refd + dref, "B-caD2+d");
      g_DCA = 4;
      run_line(PS, G, NB, DW, K, D, 6, MD, 3, refd + dref, "B-caD4+d");
      g_DCA = 2;
      run_line(0, G, NB, DW, K, D, 3, MD, 3, dref, "B-data");
    }
  }

  // ===== §C 消歧：NB=4·G（chunk 恒=4），搬出每 block prologue =====
  printf("\n===== §C 消歧：NB=4·G（chunk 恒=4），扫 G——单缓冲 vs 深环 per-block prologue =====\n");
  {
    for (int i = 0; i < NG; ++i) {
      int G = Gs[i];
      int NBx = 4 * G;
      unsigned long long dref = prepare_data(NBx, MD);
      unsigned long long refd = host_ref_desc(NBx, K, DW);
      run_line(PS, G, NBx, DW, K, D, 0, MD, 3, refd + dref, "C-plain+d");
      run_line(PS, G, NBx, DW, K, D, 2, MD, 3, refd + dref, "C-tma+d");
      run_line(PS, G, NBx, DW, K, D, 5, MD, 3, refd + dref, "C-caSingle+d");
      g_DCA = 2;
      run_line(PS, G, NBx, DW, K, D, 7, MD, 3, refd + dref, "C-caD2+d");
      g_DCA = 4;
      run_line(PS, G, NBx, DW, K, D, 7, MD, 3, refd + dref, "C-caD4+d");
      g_DCA = 2;
      run_line(0, G, NBx, DW, K, D, 3, MD, 3, dref, "C-data");
    }
    prepare_data(NB, MD);
  }

  // ===== §D MD 扫描：固定 G=264，看数据面重量下的排序 =====
  printf("\n===== §D 固定 G=264：扫数据面 MD（0→131072），plain / ca-single / ca-ring =====\n");
  {
    const int G = 264;
    long long mds[] = {0, 8192, 32768, 65536, 131072};
    for (int i = 0; i < 5; ++i) {
      long long md = mds[i];
      unsigned long long dref = prepare_data(NB, md);
      unsigned long long refd = host_ref_desc(NB, K, DW);
      run_line(PS, G, NB, DW, K, D, 0, md, 3, refd + dref, "D-plain");
      run_line(PS, G, NB, DW, K, D, 1, md, 3, refd + dref, "D-tma");
      run_line(PS, G, NB, DW, K, D, 4, md, 3, refd + dref, "D-caSingle");
      g_DCA = 2;
      run_line(PS, G, NB, DW, K, D, 6, md, 3, refd + dref, "D-caD2");
      g_DCA = 4;
      run_line(PS, G, NB, DW, K, D, 6, md, 3, refd + dref, "D-caD4");
      g_DCA = 2;
    }
    prepare_data(NB, MD);
  }

  // ===== §F chunk 扫描：固定 G=132，把 prologue 与单缓冲缝分开 =====
  printf("\n===== §F 固定 G=132、扫 chunk：单缓冲缝 vs 深环预取 =====\n");
  {
    const int G = 132;
    int chunks[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 6; ++i) {
      int chunk = chunks[i];
      int NBx = chunk * G;
      unsigned long long dref = prepare_data(NBx, MD);
      unsigned long long refd = host_ref_desc(NBx, K, DW);
      run_line(PS, G, NBx, DW, K, D, 0, MD, 3, refd + dref, "F-plain+d");
      run_line(PS, G, NBx, DW, K, D, 2, MD, 3, refd + dref, "F-tma+d");
      run_line(PS, G, NBx, DW, K, D, 5, MD, 3, refd + dref, "F-caSingle+d");
      g_DCA = 2;
      run_line(PS, G, NBx, DW, K, D, 7, MD, 3, refd + dref, "F-caD2+d");
      g_DCA = 4;
      run_line(PS, G, NBx, DW, K, D, 7, MD, 3, refd + dref, "F-caD4+d");
      g_DCA = 2;
      run_line(0, G, NBx, DW, K, D, 3, MD, 3, dref, "F-data");
    }
    prepare_data(NB, MD);
  }

  // ===== §G 环深扫描：固定 chunk=8、G=132，DCA=1→8 =====
  printf("\n===== §G 固定 chunk=8、G=132：扫环深 DCA=1..8（深环增益是否全在 1→2？）=====\n");
  {
    const int G = 132;
    int NBx = 8 * G;
    unsigned long long dref = prepare_data(NBx, MD);
    unsigned long long refd = host_ref_desc(NBx, K, DW);
    run_line(PS, G, NBx, DW, K, D, 4, MD, 3, refd + dref, "G-caSingle+d");
    for (int dca : {2, 3, 4, 6, 8}) {
      if (dca != 2 && dca != 4 && dca != 8) continue;  // 只实例化 2/4/8
      g_DCA = dca;
      run_line(PS, G, NBx, DW, K, D, 7, MD, 3, refd + dref, "G-caRing+d");
    }
    g_DCA = 2;
    run_line(0, G, NBx, DW, K, D, 3, MD, 3, dref, "G-data");
    prepare_data(NB, MD);
  }

  // ===== §E 占用率报告 =====
  printf("\n===== §E 占用率/寄存器报告 =====\n");
  {
    cudaFuncAttributes at;
    int nb = 0;
    size_t smem_tma = (size_t)D * K * DW * 4 + (size_t)D * 8 + 64;
    size_t smem_ca1 = (size_t)K * DW * 4 + 64;
    size_t smem_caD = (size_t)g_DCA * K * DW * 4 + 64;
    int ca2 = 0, ca4 = 0, ca8 = 0;
    CK(cudaFuncGetAttributes(&at, (wcr_kernel<4, 256, 8, 1, 0>)));
    printf("  numRegs=%d localB=%zu\n", at.numRegs, at.localSizeBytes);
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (wcr_kernel<4, 256, 8, 1, 1>), 256,
                                                     smem_tma));
    printf("  %-24s smem=%-6zu -> blk/SM(T=256)=%d\n", "TMA D=2", smem_tma, nb);
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (wcr_kernel<4, 256, 8, 1, 4>), 256,
                                                     smem_ca1));
    printf("  %-24s smem=%-6zu -> blk/SM(T=256)=%d\n", "ca single", smem_ca1, nb);
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ca2, (wcr_kernel<4, 256, 8, 2, 6>), 256,
                                                     smem_caD));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ca4, (wcr_kernel<4, 256, 8, 4, 6>), 256,
                                                     (size_t)4 * K * DW * 4 + 64));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ca8, (wcr_kernel<4, 256, 8, 8, 6>), 256,
                                                     (size_t)8 * K * DW * 4 + 64));
    printf("  ca ring DCA=2/4/8 smem=%zu/%zu/%zu -> blk/SM=%d/%d/%d\n", smem_caD,
           (size_t)4 * K * DW * 4 + 64, (size_t)8 * K * DW * 4 + 64, ca2, ca4, ca8);
  }

  printf("\n########## §66 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 10) {
      printf("usage: run PS G NB DW K D reps WMODE [MD] [U] [T] [MINB] [DCA]\n");
      return 1;
    }
    int PS = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), reps = atoi(argv[8]), WMODE = atoi(argv[9]);
    long long MD = (argc >= 11) ? atoll(argv[10]) : 0;
    if (argc >= 12) g_U = atoi(argv[11]);
    if (argc >= 13) g_TB = atoi(argv[12]);
    if (argc >= 14) g_MINB = atoi(argv[13]);
    if (argc >= 15) g_DCA = atoi(argv[14]);
    g_K = K; g_DW = DW;
    unsigned long long dref = prepare_data(NB, MD);
    unsigned long long ref = (WMODE == 3) ? dref : host_ref_desc(NB, K, DW) + dref;
    run_line(PS, G, NB, DW, K, D, WMODE, MD, reps, ref, "run");
    return 0;
  }
  printf("usage: worker_carrier_cpasync_ring {bench|run ...}\n");
  return 1;
}
