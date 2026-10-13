// ============================================================================
// P73-1 (§61)：把「megakernel worker 的数据面」推到 HBM 真峰值，并在真峰值下复测
//              「控制面描述符写是否争 HBM」。
//
// 背景（接 §45 / §59 / §60）：
//   - §45 把数据面（纯执行）推到 ~2983 GB/s（H100 HBM3 峰值的 ~89%）——但那是
//     「纯执行、无生产者」的口径。
//   - §59 给调度器骨架接上真实 HBM 数据面，只到 ~950 GB/s（单累加器依赖链）。
//   - §60 用 ILP 展开因子 U 把那条数据面抬到 ~2840 GB/s（~85%），并在**饱和点**
//     复测「producer 描述符写不争 HBM」。但 §60.9 自陈：只到峰值 ~85%（`T=1024`
//     单 block/SM = 50% 占用、再堆 `T×U` 受寄存器/SM 限），**真 100% 峰值下的争用仍
//     留一丝未测**。
//
// 本节把「数据面带宽天花板」本身从 §60 的**调度器骨架**里抽出来单独量，回答两个问题：
//   Q1 数据面能不能到真峰值？靠什么——**占用率（多 block/SM）** 还是 **载入载体**
//      （寄存器 `__ldcs` uint4 vs **TMA bulk 进 SMEM**）？H100 的「读峰值」到底在哪？
//   Q2 在真峰值下，与生产等价的**描述符写流量**是否开始拖慢数据面？转折点在哪？
//
// 唯一新变量（相对 §60）：
//   (a) 数据面**从调度器里解耦**成纯带宽微基准（无环协议、无 TMA 取 task），故能把
//       占用率/载体这两个旋钮单独扫到极限；
//   (b) 数据面**载体**新增 **TMA bulk load → SMEM → 折叠**（MPK worker 的真实形态）
//       与寄存器 `__ldcs` 对跑；
//   (c) 争用测试用**同核并发 writer 块**（写描述符字节）在**峰值配置**下扫描写速率，
//       找「读带宽开始掉」的转折点，而非只在 ~85% 处确认「不掉」。
//
// 用法：./dataplane_peak run <NBLK> <PS> <BPu4> <T> <U> <carrier> <DC> <WU4> <reps>
//       ./dataplane_peak bench
// 目标卡 H100 / sm90a。安全：脚本带 timeout；writer 只写自有 scratch。
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

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 硬件 mbarrier + TMA bulk load（与 §58/§60 同源）----
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

// ---- 载体 0：寄存器 `__ldcs` uint4 + U 路 ILP（= §60 的数据面）----
template <int U>
__device__ __forceinline__ unsigned long long read_reg(const uint4* __restrict__ base,
                                                       long long BP, int tid, int T) {
  unsigned long long accU[U];
#pragma unroll
  for (int u = 0; u < U; ++u) accU[u] = 0ull;
  const long long strd = (long long)T * U;
  long long i = tid;
  for (; i + (long long)(U - 1) * T < BP; i += strd) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint4 v = __ldcs(base + i + (long long)u * T);
      accU[u] += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
    }
  }
  for (; i < BP; i += T) {
    uint4 v = __ldcs(base + i);
    accU[0] += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
  }
  unsigned long long a = 0;
#pragma unroll
  for (int u = 0; u < U; ++u) a += accU[u];
  return a;
}

// ---- 载体 1：TMA bulk load → SMEM 双缓冲 → 折叠（MPK worker 真实载体）----
template <int DC>
__device__ __forceinline__ unsigned long long read_tma(const uint4* __restrict__ base,
                                                       long long BP, int tid, int T,
                                                       char* smem) {
  const int D = 2;
  uint4* buf = reinterpret_cast<uint4*>(smem);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * DC * 16);
  int nchunk = (int)((BP + DC - 1) / DC);
  if (tid == 0) {
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  }
  __syncthreads();
  auto issue = [&](int c) {
    if (c >= nchunk) return;
    int slot = c & 1;
    long long off = (long long)c * DC;
    long long rem = BP - off;
    int n = (int)((rem < DC) ? rem : DC);
    int bytes = n * 16;
    if (tid == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes);
      bulk_load_1d(buf + (size_t)slot * DC, base + off, (uint32_t)bytes, &mbar[slot]);
    }
  };
  for (int c = 0; c < D && c < nchunk; ++c) issue(c);
  unsigned long long part = 0;
  for (int c = 0; c < nchunk; ++c) {
    int slot = c & 1;
    uint32_t ph = (uint32_t)((c >> 1) & 1u);
    mbar_wait(&mbar[slot], ph);
    long long off = (long long)c * DC;
    long long rem = BP - off;
    int n = (int)((rem < DC) ? rem : DC);
    const uint4* b = buf + (size_t)slot * DC;
    unsigned long long acc = 0;
#pragma unroll 4
    for (int j = tid; j < n; j += T) {
      uint4 v = b[j];
      acc += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
    }
    part += acc;
    __syncthreads();
    issue(c + D);
  }
  return part;
}

// ---- 统一 body：block < NBLK 读；block >= NBLK 写描述符（模拟 producer）----
template <int U, int CARRIER, int DC>
__device__ __forceinline__ void dp_body(int NBLK, int PS, long long BP, long long WU4,
                                        const uint4* __restrict__ data,
                                        uint4* __restrict__ wbuf,
                                        unsigned long long* __restrict__ sink,
                                        unsigned long long* red, char* smem) {
  const int tid = threadIdx.x;
  const int T = blockDim.x;
  if (blockIdx.x < NBLK) {
    const uint4* base = data + (long long)blockIdx.x * BP;
    unsigned long long part;
    if constexpr (CARRIER == 0)
      part = read_reg<U>(base, BP, tid, T);
    else
      part = read_tma<DC>(base, BP, tid, T, smem);
    // block reduce
    red[tid] = part;
    __syncthreads();
    for (int s = T / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    if (tid == 0) atomicAdd(sink, red[0]);
    return;
  }
  // writer：block NBLK..NBLK+PS-1，每个写 WU4 个 uint4（描述符字节流量模式）
  int wid = blockIdx.x - NBLK;
  if (wid < PS) {
    long long off0 = (long long)wid * WU4;
    for (long long i = tid; i < WU4; i += T) {
      uint4 v;
      v.x = mix32((uint32_t)i, wid);
      v.y = mix32(v.x, 1u);
      v.z = mix32(v.x, 2u);
      v.w = mix32(v.x, 3u);
      wbuf[off0 + i] = v;
    }
  }
}

template <int U, int CARRIER, int DC>
__global__ void dp_kernel(int NBLK, int PS, long long BP, long long WU4,
                          const uint4* __restrict__ data, uint4* __restrict__ wbuf,
                          unsigned long long* __restrict__ sink) {
  __shared__ unsigned long long red[MAXT];
  extern __shared__ char smem[];
  dp_body<U, CARRIER, DC>(NBLK, PS, BP, WU4, data, wbuf, sink, red, smem);
}

// 强制占用率（T=1024 时压到 ≤32 寄存器 → 2 block/SM）的变体
template <int U, int CARRIER, int DC>
__global__ void __launch_bounds__(1024, 2)
    dp_kernel_oc(int NBLK, int PS, long long BP, long long WU4,
                 const uint4* __restrict__ data, uint4* __restrict__ wbuf,
                 unsigned long long* __restrict__ sink) {
  __shared__ unsigned long long red[MAXT];
  extern __shared__ char smem[];
  dp_body<U, CARRIER, DC>(NBLK, PS, BP, WU4, data, wbuf, sink, red, smem);
}

// ============================================================================
// host
// ============================================================================
__global__ void fill_data_kernel(uint4* d, long long tot, unsigned long long* dref) {
  long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= tot) return;
  uint4 v;
  v.x = mix32((uint32_t)u, 0u);
  v.y = mix32((uint32_t)u, 1u);
  v.z = mix32((uint32_t)u, 2u);
  v.w = mix32((uint32_t)u, 3u);
  d[u] = v;
  atomicAdd(dref, (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w));
}

static uint4* g_data = nullptr;
static uint4* g_wbuf = nullptr;
static unsigned long long* g_sink = nullptr;
static unsigned long long* g_dref = nullptr;
static long long g_data_cap = 0, g_wbuf_cap = 0;

static void ensure(long long data_u4, long long wbuf_u4) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (wbuf_u4 > g_wbuf_cap) {
    if (g_wbuf) CK(cudaFree(g_wbuf));
    CK(cudaMalloc(&g_wbuf, (size_t)(wbuf_u4 + 1024) * 16));
    g_wbuf_cap = wbuf_u4 + 1024;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_dref) CK(cudaMalloc(&g_dref, 8));
}

static int g_U = 1, g_carrier = 0, g_DC = 0;
static int g_last_ok = 1;

static void launch(int oc, int grid, int T, size_t req, int NBLK, int PS, long long BP,
                   long long WU4) {
#define LAUNCH_U(UU, CC, DD)                                                            \
  do {                                                                                  \
    if (oc) {                                                                           \
      static bool a_##UU##_##CC##_##DD = false;                                          \
      if (CC == 1 && !a_##UU##_##CC##_##DD) {                                            \
        CK(cudaFuncSetAttribute(dp_kernel_oc<UU, CC, DD>,                                \
                                cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));\
        a_##UU##_##CC##_##DD = true;                                                     \
      }                                                                                  \
      dp_kernel_oc<UU, CC, DD><<<grid, T, req>>>(NBLK, PS, BP, WU4, g_data, g_wbuf,     \
                                                 g_sink);                                \
    } else {                                                                             \
      static bool b_##UU##_##CC##_##DD = false;                                          \
      if (CC == 1 && !b_##UU##_##CC##_##DD) {                                            \
        CK(cudaFuncSetAttribute(dp_kernel<UU, CC, DD>,                                   \
                                cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));\
        b_##UU##_##CC##_##DD = true;                                                     \
      }                                                                                  \
      dp_kernel<UU, CC, DD><<<grid, T, req>>>(NBLK, PS, BP, WU4, g_data, g_wbuf, g_sink);\
    }                                                                                    \
    g_last_ok = (cudaGetLastError() == cudaSuccess);                                     \
  } while (0)

  if (g_carrier == 0) {
    switch (g_U) {
      case 2: LAUNCH_U(2, 0, 0); break;
      case 4: LAUNCH_U(4, 0, 0); break;
      case 8: LAUNCH_U(8, 0, 0); break;
      default: LAUNCH_U(1, 0, 0); break;
    }
  } else {
    switch (g_DC) {
      case 512: LAUNCH_U(1, 1, 512); break;
      case 1024: LAUNCH_U(1, 1, 1024); break;
      case 2048: LAUNCH_U(1, 1, 2048); break;
      case 4096: LAUNCH_U(1, 1, 4096); break;
      default: LAUNCH_U(1, 1, 2048); break;
    }
  }
#undef LAUNCH_U
}

struct Res {
  double ms;
  unsigned long long sink;
  int unsupported;
};

static Res run(int NBLK, int PS, long long BP, long long WU4, int T, int reps, int oc) {
  ensure((long long)NBLK * BP, (long long)PS * WU4);
  size_t req = 0;
  if (g_carrier == 1) {
    int dc = g_DC ? g_DC : 2048;
    req = (size_t)2 * dc * 16 + (size_t)2 * 8 + 64;
  }
  int grid = NBLK + PS;
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    launch(oc, grid, T, req, NBLK, PS, BP, WU4);
    if (!g_last_ok) { unsupported = 1; break; }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) best = ms;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  Res res;
  res.ms = best;
  res.sink = 0;
  res.unsupported = unsupported;
  if (!unsupported) CK(cudaMemcpy(&res.sink, g_sink, 8, cudaMemcpyDeviceToHost));
  return res;
}

// ============================================================================
// bench
// ============================================================================
static const char* carrier_name() { return g_carrier == 0 ? "reg" : "tma"; }

static void line(const char* tag, int NBLK, int PS, long long BP, long long WU4, int T,
                 int reps, int oc, unsigned long long ref) {
  Res r = run(NBLK, PS, BP, WU4, T, reps, oc);
  double read_bytes = (double)((long long)NBLK * BP * 16);
  double write_bytes = (double)((long long)PS * WU4 * 16);
  double read_gbs = read_bytes / (r.ms * 1e-3) / 1e9;
  double write_gbs = write_bytes / (r.ms * 1e-3) / 1e9;
  const char* verdict = r.unsupported ? "UNSUP" : "OK";
  printf("  %-22s NBLK=%-4d PS=%-3d BP=%-9lld T=%-4d oc=%d U=%-2d carrier=%-3s DC=%-5d "
         "-> %-6s ms=%.4f readGB/s=%.1f writeGB/s=%.1f\n",
         tag, NBLK, PS, BP, T, oc, g_U, carrier_name(), g_DC, verdict, r.ms, read_gbs,
         write_gbs);
  (void)ref;
}

static void bench() {
  const long long TOTAL_U4 = 268435456LL;  // 4 GiB 读流量（1 uint4 = 16 B）

  printf("########################################################################\n");
  printf("# §61 数据面到真 HBM 峰值：占用率 × 载体（reg ldcs / TMA bulk）× 争用\n");
  printf("# 目标卡 H100/sm90a；每配置 = NBLK 读块 × BP uint4（总读 ≈ 4 GiB）+ PS 写块\n");
  printf("########################################################################\n");

  ensure(TOTAL_U4, 1024);
  unsigned long long ref = 0;
  CK(cudaMemset(g_dref, 0, 8));
  fill_data_kernel<<<(int)((TOTAL_U4 + 255) / 256), 256>>>(g_data, TOTAL_U4, g_dref);
  CK(cudaMemcpy(&ref, g_dref, 8, cudaMemcpyDeviceToHost));

  // ===== §A 正确性：两载体折叠逐位对 host 参考 =====
  {
    long long BP = 1 << 16;  // 1 MiB / block
    int NBLK = 16;
    long long NBsmall = (long long)NBLK * BP;  // 1M
    unsigned long long href = 0;
    for (long long u = 0; u < NBsmall; ++u) {
      uint32_t x = mix32((uint32_t)u, 0u), y = mix32((uint32_t)u, 1u);
      uint32_t z = mix32((uint32_t)u, 2u), w = mix32((uint32_t)u, 3u);
      href += (unsigned long long)(x ^ y ^ z ^ w);
    }
    printf("\n===== §A 正确性（NBLK=%d，BP=%lld=1MiB，T=256，PS=0）：两载体 vs host 参考 =====\n",
           NBLK, BP);
    g_DC = 0;
    g_carrier = 0;
    g_U = 4;
    Res rr = run(NBLK, 0, BP, 0, 256, 1, 0);
    printf("  reg  : sink=%llu ref=%llu -> %s\n", rr.sink, href,
           rr.sink == href ? "OK" : "WRONG");
    g_carrier = 1;
    g_U = 1;
    g_DC = 2048;
    Res rt = run(NBLK, 0, BP, 0, 256, 1, 0);
    printf("  tma  : sink=%llu ref=%llu -> %s\n", rt.sink, href,
           rt.sink == href ? "OK" : "WRONG");
    g_carrier = 0;
    g_DC = 0;
    g_U = 1;
  }

  // ===== §B 寄存器载体：占用率（threads/SM = B×T）扫描 =====
  printf("\n===== §B 寄存器 ldcs 载体：B block/SM × T 扫描（总读 4 GiB，PS=0）=====\n");
  g_carrier = 0;
  g_DC = 0;
  {
    int Ts[] = {256, 512, 1024};
    int Bs[] = {1, 2, 4, 8};
    int Us[] = {1, 2, 4, 8};
    for (int ti = 0; ti < 3; ++ti) {
      int T = Ts[ti];
      for (int bi = 0; bi < 4; ++bi) {
        int B = Bs[bi];
        if ((long long)B * T > 2048) continue;  // H100 上限 2048 线程/SM
        int NBLK = 132 * B;
        long long BP = TOTAL_U4 / NBLK;
        for (int ui = 0; ui < 4; ++ui) {
          g_U = Us[ui];
          char tag[32];
          snprintf(tag, sizeof(tag), "thr/SM=%d", B * T);
          line(tag, NBLK, 0, BP, 0, T, 3, 0, ref);
        }
      }
    }
    g_U = 1;
  }

  // ===== §C 强制占用率：T=1024 用 __launch_bounds__(1024,2) 压寄存器 =====
  printf("\n===== §C 强制 2 block/SM（__launch_bounds__(1024,2)）：T=1024 的 reg 载体 =====\n");
  g_carrier = 0;
  g_DC = 0;
  {
    int NBLK = 132 * 2;
    long long BP = TOTAL_U4 / NBLK;
    int Us[] = {1, 2, 4};
    for (int ui = 0; ui < 3; ++ui) {
      g_U = Us[ui];
      line("forced-2/SM", NBLK, 0, BP, 0, 1024, 3, 1, ref);
    }
    g_U = 1;
  }

  // ===== §D TMA bulk 载体：chunk DC × T 扫描 =====
  printf("\n===== §D TMA bulk 载体（SMEM 双缓冲，深度 2）：DC × T 扫描（总读 4 GiB，PS=0）=====\n");
  g_carrier = 1;
  g_U = 1;
  {
    int DCs[] = {512, 1024, 2048, 4096};
    int Ts[] = {128, 256, 512};
    for (int di = 0; di < 4; ++di) {
      for (int ti = 0; ti < 3; ++ti) {
        g_DC = DCs[di];
        int NBLK = 132;
        long long BP = TOTAL_U4 / NBLK;
        char tag[32];
        snprintf(tag, sizeof(tag), "tma-%dKB/block", (int)(BP * 16 / 1024));
        line(tag, NBLK, 0, BP, 0, Ts[ti], 3, 0, ref);
      }
    }
    g_DC = 0;
  }
  g_carrier = 0;

  // ===== §E 占用率/寄存器报告 =====
  printf("\n===== §E 占用率/寄存器报告（cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor）=====\n");
  {
    cudaFuncAttributes at;
    int nb = 0;
#define REP_FUNC(KER, LABEL)                                                          \
    do {                                                                              \
      CK(cudaFuncGetAttributes(&at, KER));                                            \
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, KER, 1024, 0));           \
      printf("  %-28s numRegs=%-4d localB=%-3zu maxThreads=%-5d -> blk/SM(T=1024)=%d\n", \
             LABEL, at.numRegs, at.localSizeBytes, at.maxThreadsPerBlock, nb);        \
    } while (0)
    REP_FUNC((dp_kernel<1, 0, 0>), "reg U=1");
    REP_FUNC((dp_kernel<2, 0, 0>), "reg U=2");
    REP_FUNC((dp_kernel<4, 0, 0>), "reg U=4");
    REP_FUNC((dp_kernel<8, 0, 0>), "reg U=8");
    REP_FUNC((dp_kernel_oc<1, 0, 0>), "reg U=1 (forced 2/SM)");
    REP_FUNC((dp_kernel_oc<4, 0, 0>), "reg U=4 (forced 2/SM)");
    REP_FUNC((dp_kernel<1, 1, 2048>), "tma DC=2048 (32KB buf)");
#undef REP_FUNC
  }

  // ===== §F 争用：在峰值配置下扫描 writer 写速率 =====
  printf("\n===== §F 争用（峰值配置 × 写块 × 写量）：数据面读带宽是否被描述符写拖慢 =====\n");
  {
    // 峰值配置（由 §B/§C/§D 选出，见 docs/04 §61）：reg U=2, T=1024。
    // NBLK=132（每 SM 1 个 reader block，留出 132 个空槽），writer 全共驻、绝不挤占
    // reader 槽位——这样写量扫描隔离的是「HBM 带宽争用」而非「SM 槽位挤占」（F3 对照）。
    g_carrier = 0;
    g_DC = 0;
    g_U = 2;
    int NBLK = 132;
    long long BP = TOTAL_U4 / NBLK;
    printf("  --- F1 写量扫描（PS=8 固定，NBLK=132 留 132 槽）：隔离 HBM 争用（reps=5）---\n");
    long long Ws[] = {16384, 65536, 262144, 1048576, 4194304, 16777216};
    for (int wi = 0; wi < 6; ++wi) {
      char tag[40];
      snprintf(tag, sizeof(tag), "WU4=%lld/block", Ws[wi]);
      line(tag, NBLK, 8, BP, Ws[wi], 1024, 5, 0, ref);
    }
    printf("  --- F2 写块数扫描（WU4=1048576/block 固定，NBLK=132，共驻，reps=5）---\n");
    int Pss[] = {0, 1, 2, 4, 8, 16, 32};
    for (int pi = 0; pi < 7; ++pi) {
      char tag[40];
      snprintf(tag, sizeof(tag), "PS=%d", Pss[pi]);
      line(tag, NBLK, Pss[pi], BP, 1048576, 1024, 5, 0, ref);
    }
    printf("  --- F3 对照：占满槽位（NBLK=264 + PS，writer 挤占 reader 槽）---\n");
    int NBLK2 = 264;
    long long BP2 = TOTAL_U4 / NBLK2;
    int Pss2[] = {0, 8, 32};
    for (int pi = 0; pi < 3; ++pi) {
      char tag[40];
      snprintf(tag, sizeof(tag), "full PS=%d", Pss2[pi]);
      line(tag, NBLK2, Pss2[pi], BP2, 1048576, 1024, 5, 1, ref);
    }
    g_U = 1;
  }

  printf("\n########## §61 完成 ##########\n");
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench")) {
    bench();
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 10) {
      printf("usage: run NBLK PS BPu4 T U carrier DC WU4 reps [oc]\n");
      return 1;
    }
    int NBLK = atoi(argv[2]), PS = atoi(argv[3]);
    long long BP = atoll(argv[4]);
    int T = atoi(argv[5]);
    g_U = atoi(argv[6]);
    g_carrier = atoi(argv[7]);
    g_DC = atoi(argv[8]);
    long long WU4 = atoll(argv[9]);
    int reps = (argc >= 11) ? atoi(argv[10]) : 3;
    int oc = (argc >= 12) ? atoi(argv[11]) : 0;
    CK(cudaMemset(g_dref, 0, 8));
    long long tot = (long long)NBLK * BP;
    fill_data_kernel<<<(int)((tot + 255) / 256), 256>>>(g_data, tot, g_dref);
    line("run", NBLK, PS, BP, WU4, T, reps, oc, 0);
    return 0;
  }
  printf("usage: dataplane_peak {bench|run ...}\n");
  return 1;
}
