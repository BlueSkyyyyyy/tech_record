// ============================================================================
// P74-1 (§62)：把端到端调度器骨架（§60）推上「占用率旋钮」，并把 §60(85%) 与
//              §61(纯数据面 96%) 之间的**夹层成本**在同一二进制内分解出来。
//
// 背景（接 §60 / §61）：
//   - §60 给 §47–§59 的调度器骨架接上真实 HBM 数据面 + ILP 展开因子 U，端到端到
//     ~2840 GB/s（H100 HBM3 峰值 ~85%）。但它自陈卡在 `T=1024` **单 block/SM**
//     （50% 占用），再堆 `T×U` 受寄存器/SM 限。
//   - §61 把数据面从调度器里**解耦**成纯带宽微基准，发现缺的旋钮就是**占用率**：reg 载体
//     在 `T=1024`、**2 block/SM**（`__launch_bounds__(1024,2)`）、`U=4` 时到 ~3219 GB/s
//     = 峰值 **96%**（`docs/04` §61.3）。
//   - §61.10 明留两条尾巴：(a) 纯读微基准是**能力上界**而非端到端吞吐，与 §60 端到端的
//     差即「调度/取 task/折叠的**夹层成本**」；(b) 争用测试的 writer 是**合成写**，非真实
//     描述符/环协议。
//
// 本节把 §61 的**占用率旋钮**（多 block/SM + `__launch_bounds__`）搬回 §60 的**完整骨架**
//   （producer 写描述符到 per-worker 环 + worker TMA 取 task + 折叠 + HBM 数据面），回答：
//   Q1 端到端骨架能靠占用率从 §60 的 85% 抬到多少？离 §61 的纯数据面 96% 还差多少？
//   Q2 那截差（夹层成本）由什么构成——是按 task 计的 TMA 取描述符 + SMEM 折叠，还是
//      producer 写环的带宽争用？
//
// 关键设计：**同一个 kernel 的三档 mode**，故三档共享除「夹层」外的所有代码：
//   mode 2 = data-only：无 producer、无环、无 TMA，worker 只做 HBM 数据面读（= §61 的
//            in-binary 天花板，但用同一 dataplane 代码）；
//   mode 0 = +fold：在 mode 2 上再加「TMA 取静态描述符 → SMEM 折叠」一整套取 task 成本
//            （无 producer / 无环，隔离「每 task 的 TMA+折叠」）；
//   mode 1 = full：在 mode 0 上再加 producer 写 per-worker 环 + release 发布 + worker 等
//            head/回 tail（隔离「环协议 + producer 带宽争用」）。
//   mode2→mode0 之差不含任何 HBM 数据面变化，纯属取 task/折叠；mode0→mode1 之差是环+producer。
//
// 唯一新变量（相对 §60）：worker 的**占用率**通过 `__launch_bounds__(TB, MINB)` 模板化
//   （TB=blockDim、MINB=每 SM 最少 block 数），扫 (TB,MINB,U) 找端到端饱和点。
// 骨架/环协议/per-worker 拓扑/producer 轮转、sink 逐位校验、clock64 守门狗全部沿用 §60。
//
// 用法：./e2e_saturation run <PS> <G> <NB> <DW> <K> <D> <CAPp> <reps> [CW] [MD] [U] [mode] [TB] [MINB]
//       ./e2e_saturation bench
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

#define MAXT 1024
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
// kernel body（单常驻：block 0..PS-1 = producer，block PS..PS+G-1 = worker）
//   mode：0 = 无 producer + TMA 取静态描述符 + 折叠 + 数据面
//         1 = per-worker 环 + producer + TMA 取 task + 折叠 + 数据面（= §60 骨架）
//         2 = data-only：无 producer、无环、无 TMA/折叠，worker 只做 HBM 数据面读
// ============================================================================
template <int U>
__device__ __forceinline__ void e2e_body(int PS, int G, int NB, int DW, int K, int D, int CAPp,
                                         int mode, int CW, long long MD,
                                         const uint32_t* __restrict__ g_static,
                                         uint32_t* __restrict__ ring,
                                         unsigned long long* head, unsigned long long* tail,
                                         unsigned long long* sink, int* abort, int* dbg_done,
                                         const uint4* __restrict__ g_data, char* smem_raw) {
  const int tid = threadIdx.x;
  const int nlanes = blockDim.x;
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

  // ---------------- producer：block 0..PS-1（mode 1）----------------
  if (mode == 1 && blockIdx.x < PS) {
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
        uint32_t* dst = (uint32_t*)ring + (size_t)((size_t)w * CAPp + slot) * K * DW;
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

  // ---------------- worker：block PS..PS+G-1（或 block 0..G-1 if mode!=1）--------
  const int w = (mode == 1) ? (blockIdx.x - PS) : blockIdx.x;
  const int nt = (w < G) ? count_of(w) : 0;
  const int b0w = w * chunk;

  if (tid == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  unsigned long long acc = 0;

  auto wait_ready = [&](int bl) -> bool {
    if (mode != 1) return true;
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
  if (mode != 2)
    for (int t = 0; t < D && t < nt; ++t)
      if (!issue(t)) { ok = false; break; }

  __shared__ unsigned long long wred[32];  // 每 warp 一个部分和（blockDim<=1024 才需 <=32 槽）
  const int nwarp = (nlanes + 31) >> 5;
  for (int t = 0; t < nt && ok; ++t) {
    unsigned long long part = 0;
    if (mode != 2) {
      int slot = t % D;
      uint32_t phase = (uint32_t)((t / D) & 1u);
      mbar_wait(&mbar[slot], phase);
      const uint32_t* buf = smem + (size_t)slot * K * DW;
      for (int j = tid; j < K; j += nlanes) {
        uint32_t a = 0x9e3779b9u;
        for (int i = 0; i < DW; ++i) a = a * 2246822519u + buf[(size_t)j * DW + i];
        part += (unsigned long long)a;
      }
    }
    if (MD > 0) {  // 真实 HBM 数据面：每 batch 流式读 MD 个 uint4（__ldcs 绕 L2）
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
    if (CW > 0) {  // 纯 ALU 延迟（模拟计算型 task，不改 part）
      uint32_t b = (uint32_t)part + (uint32_t)(tid * 2654435761u);
      for (int c = 0; c < CW; ++c) b = b * 1664525u + 1013904223u;
      if (b == 0x1234567u) part += b;
    }
    // block 归约用 warp shuffle + 32 槽 shared（避免 MAXT 大小的静态 SMEM 吃掉占用率）
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
    if (mode == 1 && tid == 0) atom_add_release_u64(tail + w, (unsigned long long)1);
    if (mode != 2)
      if (t + D < nt)
        if (!issue(t + D)) { ok = false; break; }
  }

  if (acc && tid == 0) atomicAdd(sink, acc);
  if (tid == 0 && dbg_done) atomicAdd(dbg_done, 1);
}

// 三档占用率变体：MINB 经 __launch_bounds__ 把寄存器预算压到 65536/(TB*MINB)。
// TB=blockDim（须与 host 传入的 T 一致），MINB=每 SM 最少 block 数。
template <int U, int TB, int MINB>
__global__ void __launch_bounds__(TB, MINB) e2e_kernel(
    int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode, int CW, long long MD,
    const uint32_t* __restrict__ g_static, uint32_t* __restrict__ ring,
    unsigned long long* head, unsigned long long* tail, unsigned long long* sink, int* abort,
    int* dbg_done, const uint4* __restrict__ g_data) {
  extern __shared__ char smem_raw[];
  e2e_body<U>(PS, G, NB, DW, K, D, CAPp, mode, CW, MD, g_static, ring, head, tail, sink,
              abort, dbg_done, g_data, smem_raw);
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

static int g_CW = 0;
static long long g_MD = 0;
static int g_U = 1;
static int g_TB = 1024;
static int g_MINB = 1;
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
                   int CAPp, int mode) {
#define LAUNCH_FOR(UU, TT, MB)                                                          \
  do {                                                                                  \
    static bool a_##UU##_##TT##_##MB = false;                                           \
    if (!a_##UU##_##TT##_##MB) {                                                        \
      CK(cudaFuncSetAttribute(e2e_kernel<UU, TT, MB>,                                   \
                              cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));\
      a_##UU##_##TT##_##MB = true;                                                      \
    }                                                                                   \
    e2e_kernel<UU, TT, MB><<<grid, T, req>>>(PS, G, NB, DW, K, D, CAPp, mode, g_CW, g_MD,\
                                             g_static, g_ring, g_head, g_tail, g_sink,   \
                                             g_abort, g_done, g_data);                  \
    g_last_launch_ok = (cudaGetLastError() == cudaSuccess);                             \
  } while (0)

#define TRY(UU, TT, MB) if (g_U == UU && g_TB == TT && g_MINB == MB) { LAUNCH_FOR(UU, TT, MB); return; }
  TRY(1, 256, 8); TRY(4, 256, 8); TRY(8, 256, 8);
  TRY(1, 512, 2); TRY(4, 512, 2); TRY(8, 512, 2);
  TRY(1, 512, 4); TRY(4, 512, 4); TRY(8, 512, 4);
  TRY(1, 1024, 1); TRY(4, 1024, 1); TRY(8, 1024, 1);
  TRY(1, 1024, 2); TRY(4, 1024, 2); TRY(8, 1024, 2);
#undef TRY
  // fallback：未列出的组合用 (1024,1) 兜（不应发生）
  LAUNCH_FOR(1, 1024, 1);
#undef LAUNCH_FOR
}

static Res run(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode, int reps,
               int T) {
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

  int grid = (mode == 1) ? (PS + G) : G;
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
    CK(cudaMemset(g_head, 0, G * 8));
    CK(cudaMemset(g_tail, 0, G * 8));
    CK(cudaEventRecord(e0));
    launch(grid, T, req, PS, G, NB, DW, K, D, CAPp, mode);
    if (!g_last_launch_ok) {
      unsupported = 1;
      break;
    }
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
  res.unsupported = unsupported;
  return res;
}

static void run_line(int PS, int G, int NB, int DW, int K, int D, int CAPp, int mode,
                     int reps, unsigned long long ref, int T, const char* tag) {
  Res r = run(PS, G, NB, DW, K, D, CAPp, mode, reps, T);
  const char* verdict;
  if (r.unsupported)
    verdict = "UNSUP";
  else if (r.aborted)
    verdict = "ABORT";
  else if (mode == 2) {
    verdict = (r.sink == ref) ? "OK" : "WRONG";  // ref = 仅数据面
  }   else if (g_CW > 0) {
    verdict = "RUN";  // CW>0：sink 语义含 ALU 干扰，仅查不死锁
  } else if (r.sink == ref) {
    verdict = "OK";
  } else {
    verdict = "WRONG";
  }
  double desc_gbs = (double)((long long)NB * K * (long long)DW * 4) / (r.ms * 1e-3) / 1e9;
  double task_gbs = (double)((long long)NB * g_MD * 16) / (r.ms * 1e-3) / 1e9;
  int wpp = (G + PS - 1) / PS;
  printf("  %-14s PS=%-3d G=%-3d wpp=%-3d NB=%-4d mode=%d D=%-2d CAPp=%-4d CW=%-6d "
         "MD=%-7lld U=%-2d TB=%-4d MINB=%-2d -> %-6s ms=%.4f descGB/s=%.1f taskGB/s=%.1f\n",
         tag ? tag : "", PS, G, wpp, NB, mode, D, CAPp, g_CW, g_MD, g_U, g_TB, g_MINB,
         verdict, r.ms, desc_gbs, task_gbs);
}

// mode 2 的 host 参考只含数据面；mode 0 的参考只含描述符；mode 1 = 描述符+数据面。
static unsigned long long ref_for(int mode, int NB, int K, int DW, unsigned long long dref) {
  unsigned long long descr = (mode == 2) ? 0 : host_ref_desc(NB, K, DW);
  return descr + ((mode == 2) ? dref : dref);
}

// ============================================================================
// bench
// ============================================================================
static void bench() {
  const int DW = 88;
  const int K = 32;
  const int D = 2;
  const int NB = 2048;         // 批数（= task 数）
  const long long MD = 65536;  // 1 MiB/batch -> 全图数据面总读 ≈ 2.0 GiB（与 G 无关）
  const int Gs[] = {100, 132, 264, 528, 1040};
  const int NG = 5;

  printf("########################################################################\n");
  printf("# §62 端到端调度骨架的占用率扫描：§60(85%%) -> ? 与 §61(96%%) 的夹层成本分解\n");
  printf("# 目标卡 H100/sm90a，DW=%d(352B)，K=%d，D=%d，NB=%d，MD=%lld(1MiB/batch)，总读≈2GiB\n",
         DW, K, D, NB, MD);
  printf("# 占用率旋钮 = worker 块数 G（threads/SM ≈ G*T/132），而非仅 launch_bounds\n");
  printf("########################################################################\n");

  // ===== §A 正确性：三档 mode 逐位对 host 参考 =====
  {
    const int G = 100;
    const int NBc = 512;
    const int chc = (NBc + G - 1) / G;
    g_MD = 8192;
    unsigned long long dref = prepare_data(NBc, g_MD);
    unsigned long long ref = host_ref_desc(NBc, K, DW) + dref;
    printf("\n===== §A 正确性（G=%d，NB=%d，chunk=%d，MD=8192）：data-only / +fold / full =====\n",
           G, NBc, chc);
    g_U = 4; g_TB = 256; g_MINB = 8;
    run_line(4, G, NBc, DW, K, D, chc, 2, 3, dref, 256, "data-only");
    run_line(4, G, NBc, DW, K, D, chc, 0, 3, ref, 256, "+fold");
    run_line(4, G, NBc, DW, K, D, chc, 1, 3, ref, 256, "full(ring)");
    g_MD = 0; g_U = 1; g_TB = 1024; g_MINB = 1;
  }

  g_MD = MD;
  unsigned long long dref = prepare_data(NB, g_MD);
  printf("\n  （全图数据面总读 ≈ %.2f GiB，NB=%d × MD=%lld × 16B）\n",
         (double)((long long)NB * g_MD * 16) / 1073741824.0, NB, g_MD);

  // ===== §B data-only 天花板：扫 G（占用率），看同二进制内纯数据面能到多少 =====
  printf("\n===== §B data-only 天花板（mode 2，无 producer/环/TMA）：扫 G（T=256,MINB=8,U=4）=====\n");
  for (int i = 0; i < NG; ++i) {
    const int G = Gs[i];
    const int chunk = (NB + G - 1) / G;
    g_TB = 256; g_MINB = 8; g_U = 4;
    run_line(1, G, NB, DW, K, D, chunk, 2, 3, dref, 256, "ceil");
  }
  printf("  --- 对照：§60 形态（T=1024,MINB=1,U=8）的两个 G ---\n");
  for (int i = 0; i < NG; i += 2) {
    const int G = Gs[i];
    const int chunk = (NB + G - 1) / G;
    g_TB = 1024; g_MINB = 1; g_U = 8;
    run_line(1, G, NB, DW, K, D, chunk, 2, 3, dref, 1024, "ceil-1024");
  }

  // ===== §C full 骨架：同 G 扫描，看端到端能到多少 =====
  printf("\n===== §C full 骨架（mode 1，producer+环+TMA+数据面）：扫 G（T=256,MINB=8,U=4,PS=8）=====\n");
  {
    unsigned long long ref = host_ref_desc(NB, K, DW) + dref;
    for (int i = 0; i < NG; ++i) {
      const int G = Gs[i];
      const int chunk = (NB + G - 1) / G;
      g_TB = 256; g_MINB = 8; g_U = 4;
      run_line(8, G, NB, DW, K, D, chunk, 1, 3, ref, 256, "full");
    }
    printf("  --- 对照：§60 形态（T=1024,MINB=1,U=8,PS=8）的两个 G ---\n");
    for (int i = 0; i < NG; i += 2) {
      const int G = Gs[i];
      const int chunk = (NB + G - 1) / G;
      g_TB = 1024; g_MINB = 1; g_U = 8;
      run_line(8, G, NB, DW, K, D, chunk, 1, 3, ref, 1024, "full-1024");
    }
  }

  // ===== §D 夹层成本分解：data-only vs +fold vs full（同一最优占用率配置）=====
  printf("\n===== §D 夹层成本分解（G=1040，T=256，MINB=8，U=4；mode2 -> mode0 -> mode1）=====\n");
  {
    const int G = 1040;
    const int chunk = (NB + G - 1) / G;  // = 2
    unsigned long long ref = host_ref_desc(NB, K, DW) + dref;
    g_TB = 256; g_MINB = 8; g_U = 4;
    printf("  --- G=%d，chunk=%d（CAPp=%d）---\n", G, chunk, chunk);
    run_line(1, G, NB, DW, K, D, chunk, 2, 3, dref, 256, "data-only");
    run_line(1, G, NB, DW, K, D, chunk, 0, 3, ref, 256, "+fold");
    run_line(8, G, NB, DW, K, D, chunk, 1, 3, ref, 256, "full");
  }

  // ===== §E 真描述符写争用：最优占用率下扫 PS（mode1 的环+producer 写）=====
  printf("\n===== §E 最优占用率（G=1040,T=256,MINB=8,U=4）下扫 PS：真环/producer 写是否争带宽 =====\n");
  {
    const int G = 1040;
    const int chunk = (NB + G - 1) / G;
    unsigned long long ref = host_ref_desc(NB, K, DW) + dref;
    int Pss[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 6; ++i) {
      g_TB = 256; g_MINB = 8; g_U = 4;
      run_line(Pss[i], G, NB, DW, K, D, chunk, 1, 3, ref, 256, "full");
    }
  }

  // ===== §D2 在 §60 配置（G=100,T=1024,U=8）上做同一分解：对照 G 依赖 =====
  printf("\n===== §D2 夹层分解 @ §60 配置（G=100，T=1024，U=8，PS=8；mode2 -> mode0 -> mode1）=====\n");
  {
    const int G = 100;
    const int chunk = (NB + G - 1) / G;  // = 21
    unsigned long long ref = host_ref_desc(NB, K, DW) + dref;
    g_TB = 1024; g_MINB = 1; g_U = 8;
    printf("  --- G=%d，chunk=%d（CAPp=%d）---\n", G, chunk, chunk);
    run_line(1, G, NB, DW, K, D, chunk, 2, 3, dref, 1024, "data-only");
    run_line(1, G, NB, DW, K, D, chunk, 0, 3, ref, 1024, "+fold");
    run_line(8, G, NB, DW, K, D, chunk, 1, 3, ref, 1024, "full");
  }

  // ===== §G 消歧：固定 G=1040，扫 chunk（NB=G*chunk）——环数 G vs 每环 chunk 谁贵 =====
  printf("\n===== §G 消歧（G=1040,T=256,MINB=8,U=4,PS=32）：固定 G、扫 chunk=NB/G 看每 task 开销 =====\n");
  {
    const int G = 1040;
    int nbs[] = {2080, 4160, 8320};  // chunk = 2, 4, 8
    for (int i = 0; i < 3; ++i) {
      int NBx = nbs[i];
      int chunk = (NBx + G - 1) / G;
      unsigned long long ref = host_ref_desc(NBx, K, DW) + prepare_data(NBx, g_MD);
      g_TB = 256; g_MINB = 8; g_U = 4;
      run_line(32, G, NBx, DW, K, D, chunk, 1, 3, ref, 256, "full");
    }
    // 复位数据面为 bench 主 NB 的参考，供后续 §F 无影响
    dref = prepare_data(NB, g_MD);
    (void)dref;
  }

  // ===== §F 占用率/寄存器报告 =====
  printf("\n===== §F 占用率/寄存器报告（cudaOccupancyMaxActiveBlocksPerMultiprocessor）=====\n");
  {
    cudaFuncAttributes at;
    int nb = 0;
#define REP(KER, TB, LABEL)                                                              \
    do {                                                                                 \
      CK(cudaFuncGetAttributes(&at, KER));                                               \
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, KER, TB, 0));                \
      printf("  %-26s numRegs=%-4d localB=%-3zu -> blk/SM(T=%d)=%d\n", LABEL, at.numRegs, \
             at.localSizeBytes, TB, nb);                                                 \
    } while (0)
    REP((e2e_kernel<4, 1024, 1>), 1024, "U=4 T=1024 MINB=1");
    REP((e2e_kernel<4, 1024, 2>), 1024, "U=4 T=1024 MINB=2");
    REP((e2e_kernel<4, 512, 4>), 512, "U=4 T=512 MINB=4");
    REP((e2e_kernel<4, 256, 8>), 256, "U=4 T=256 MINB=8");
    REP((e2e_kernel<8, 256, 8>), 256, "U=8 T=256 MINB=8");
#undef REP
  }

  printf("\n########## §62 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 10) {
      printf("usage: run PS G NB DW K D CAPp reps [CW] [MD] [U] [mode] [TB] [MINB]\n");
      return 1;
    }
    int PS = atoi(argv[2]), G = atoi(argv[3]), NB = atoi(argv[4]), DW = atoi(argv[5]);
    int K = atoi(argv[6]), D = atoi(argv[7]), CAPp = atoi(argv[8]), reps = atoi(argv[9]);
    if (argc >= 11) g_CW = atoi(argv[10]);
    if (argc >= 12) g_MD = atoll(argv[11]);
    if (argc >= 13) g_U = atoi(argv[12]);
    int mode = (argc >= 14) ? atoi(argv[13]) : 1;
    if (argc >= 15) g_TB = atoi(argv[14]);
    if (argc >= 16) g_MINB = atoi(argv[15]);
    unsigned long long ref = ref_for(mode, NB, K, DW, prepare_data(NB, g_MD));
    run_line(PS, G, NB, DW, K, D, CAPp, mode, reps, ref, g_TB, "run");
    return 0;
  }
  printf("usage: e2e_saturation {bench|run ...}\n");
  return 1;
}
