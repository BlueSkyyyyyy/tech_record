// ============================================================================
// P84-1 (§72)：把 §71.7 明留的尾巴「合成访存（固定 stride 8 / 16 MiB / 无向量宽度），
//   非真实 tile；四模式各自单独跑，**未测同一 grid 内多模式共驻**」补掉——
//   在**同一常驻 grid**里，让不同 block 同时跑**不同访存模式**（异构共驻），复检 §71.7 headline 四
//   的 megakernel 论断：「**一个常驻 kernel 内 task 类型越杂、越难一套调度参数通吃**」。
//
// 背景（接 §60 / §61 / §70 / §71）：
//   - §71 已证：数据面饱和点由「每 SM 总线程 `O·T`」这个**合量**钉住（coalesced/scalar/l2res 都要
//     `O·T≈1024`），**但**同 `O·T` 下 `(T,O)` 拆分的最优**依赖访存模式**（l2res 要「多而小的
//     block」，coalesced 要「少而大的 block」，同 `O·T=1024` 差 1.47×；§71 §C）。
//   - §71.7 headline 四断言：真实 megakernel 的 task 访存模式**混杂**（GEMM tile 命中 L2/SMEM、
//     逐元素流式 HBM、gather/MoE dispatch 稀疏）→「按 O·T 铺满机器」是通用旋钮但非万能。
//   - §71.7 明文尾巴：**未把多模式放进同一 grid 共驻**（各模式此前都是**单独**整机跑）。
//
// 本节问：把**多种访存模式塞进同一个常驻 grid**（不同 block 跑不同模式），
//   ① 混合吞吐 = 各模式单独吞吐按**逻辑字节份额**的调和平均（「时间切片/无干扰」）吗？多出/少的
//      部分即**共驻干扰**（如 sparse 的 4× 物理流量抢 HBM、拖慢同一 grid 内的 coalesced 块）；
//   ② **一个静态 `(T,O)`** 能不能同时喂饱混杂模式？把「各模式各自的**最优** `(T,O)` 时间切片」
//      当 oracle，与「混合只能用**单一** `(T,O)`」的最佳值之差 = **异质性税**(heterogeneity tax)；
//   ③ 随「稀疏块占比」升高，混合**逻辑**吞吐如何塌（撞物理墙）——coalesced 块是否被稀疏块拖累。
//
// 四种模式与逻辑字节定义**完全复用 §71**（m0 coalesced / m1 sparse8x / m2 scalar4B / m3 l2res16M），
//   载体固定寄存器读 + U=4 ILP、`__maxnreg__(32)`（= §71）。**唯一新变量 = assign[]**：每 block 一个
//   模式标签（host 用误差扩散法交错填充，保证各模式**跨 SM 交织共驻**、而非先后整段执行）。
//
// 用法：./hetero_access_mix bench
//       ./hetero_access_mix run <T> <O> <reps> <r0,r1,r2,r3>   # ri = 四模式逻辑字节份额(整数)
// 目标卡 H100 / sm90a。安全：脚本带 timeout；纯读微基准，无环协议 / 无死锁。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t _e = (x);                                                            \
    if (_e != cudaSuccess) {                                                         \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

#define MAXT 1024
#define U_ILP 4
#define TOT_U4 (67108864LL)        // 逻辑总读 1 GiB（uint4 计）；所有混合份额同一逻辑字节数
#define SPARSE_STRIDE 8            // m1：每 8 个 uint4 读 1 个 → 物理 8×
#define L2N4 (1 << 20)             // m3：常驻 16 MiB = 1M uint4（< 50 MB L2）
#define NPAT 4

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 四模式读函数（与 §71 逐字一致）----
template <int U>
__device__ __forceinline__ unsigned long long read_coal(const uint4* __restrict__ base,
                                                        long long BP, int tid, int T) {
  unsigned long long acc = 0ull;
  long long i = tid;
  for (; i + (long long)(U - 1) * T < BP; i += (long long)T * U) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint4 v = __ldcs(base + i + (long long)u * T);
      acc ^= (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
    }
  }
  for (; i < BP; i += T) {
    uint4 v = __ldcs(base + i);
    acc ^= (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
  }
  return acc;
}

template <int U>
__device__ __forceinline__ unsigned long long read_sparse(const uint4* __restrict__ base,
                                                          long long BP, int tid, int T) {
  unsigned long long acc = 0ull;
  long long i = tid;
  for (; i + (long long)(U - 1) * T < BP; i += (long long)T * U) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint4 v = __ldcs(base + (i + (long long)u * T) * SPARSE_STRIDE);
      acc ^= (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
    }
  }
  for (; i < BP; i += T) {
    uint4 v = __ldcs(base + i * SPARSE_STRIDE);
    acc ^= (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
  }
  return acc;
}

template <int U>
__device__ __forceinline__ unsigned long long read_scalar(const uint32_t* __restrict__ base,
                                                          long long N32, int tid, int T) {
  unsigned long long acc = 0ull;
  long long i = tid;
  for (; i + (long long)(U - 1) * T < N32; i += (long long)T * U) {
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= (unsigned long long)__ldcs(base + i + (long long)u * T);
  }
  for (; i < N32; i += T) acc ^= (unsigned long long)__ldcs(base + i);
  return acc;
}

template <int U>
__device__ __forceinline__ unsigned long long read_l2res(const uint4* __restrict__ base,
                                                         long long BP, long long mask,
                                                         int tid, int T) {
  unsigned long long acc = 0ull;
  long long i = tid;
  for (; i + (long long)(U - 1) * T < BP; i += (long long)T * U) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint4 v = base[(i + (long long)u * T) & mask];
      acc ^= (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
    }
  }
  for (; i < BP; i += T) {
    uint4 v = base[i & mask];
    acc ^= (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
  }
  return acc;
}

template <int MAXT_>
__device__ __forceinline__ void reduce_xor(unsigned long long part, int tid, int T,
                                           unsigned long long* sink) {
  __shared__ unsigned long long red[MAXT_];
  red[tid] = part;
  __syncthreads();
  for (int s = T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] ^= red[tid + s];
    __syncthreads();
  }
  if (tid == 0 && red[0]) atomicXor(sink, red[0]);
}

// ---- 异构共驻 kernel：每 block 按 assign[blockIdx.x] 选模式（R=32 固定，与 §71 同）----
__global__ void __maxnreg__(32) pat_mix(const void* __restrict__ raw, long long BP,
                                        long long aux, const int* __restrict__ assign,
                                        unsigned long long* sink) {
  const int tid = threadIdx.x;
  const int T = blockDim.x;
  int pat = assign[blockIdx.x];
  unsigned long long part = 0ull;
  switch (pat) {
    case 0: {
      const uint4* base = (const uint4*)raw + (long long)blockIdx.x * BP;
      part = read_coal<U_ILP>(base, BP, tid, T);
      break;
    }
    case 1: {
      const uint4* base = (const uint4*)raw + (long long)blockIdx.x * BP * SPARSE_STRIDE;
      part = read_sparse<U_ILP>(base, BP, tid, T);
      break;
    }
    case 2: {
      const uint32_t* base = (const uint32_t*)raw + (long long)blockIdx.x * (BP * 4);
      part = read_scalar<U_ILP>(base, BP * 4, tid, T);
      break;
    }
    default: {
      part = read_l2res<U_ILP>((const uint4*)raw, BP, aux, tid, T);
      break;
    }
  }
  reduce_xor<MAXT>(part, tid, T, sink);
}

__global__ void fill_data_kernel(uint4* d, long long tot, unsigned long long* dref) {
  long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= tot) return;
  uint4 v;
  v.x = mix32((uint32_t)u, 0u);
  v.y = mix32((uint32_t)u, 1u);
  v.z = mix32((uint32_t)u, 2u);
  v.w = mix32((uint32_t)u, 3u);
  d[u] = v;
  atomicXor(dref, (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w));
}

// ============================================================================
// host
// ============================================================================
static uint4* g_data = nullptr;
static unsigned long long* g_sink = nullptr;
static int* g_assign = nullptr;
static long long g_data_cap = 0;
static int g_assign_cap = 0;
static int g_numSm = 0;
static int g_T = 256;

static void ensure(long long data_u4, int grid) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (grid > g_assign_cap) {
    if (g_assign) CK(cudaFree(g_assign));
    CK(cudaMalloc(&g_assign, (size_t)grid * sizeof(int)));
    g_assign_cap = grid;
  }
}

static int g_last_ok = 1;

static void launch_mix(int grid, int T, long long BP) {
  const void* raw = (const void*)g_data;
  long long aux = L2N4 - 1;
  pat_mix<<<grid, T>>>(raw, BP, aux, g_assign, g_sink);
  g_last_ok = (cudaGetLastError() == cudaSuccess);
}

// 误差扩散法：按份额份数 seq[p] 交错生成 assign[]，保证各模式跨 SM 交织共驻
static void ensure_assign(int grid) {
  if (grid > g_assign_cap) {
    if (g_assign) CK(cudaFree(g_assign));
    CK(cudaMalloc(&g_assign, (size_t)grid * sizeof(int)));
    g_assign_cap = grid;
  }
}

static void fill_assign(int grid, const int* seq, int npat) {
  long long tot = 0;
  for (int p = 0; p < npat; ++p) tot += seq[p];
  ensure_assign(grid);
  static double acc[NPAT];
  for (int p = 0; p < npat; ++p) acc[p] = 0.0;
  int* h = (int*)malloc((size_t)grid * sizeof(int));
  for (int b = 0; b < grid; ++b) {
    int pick = 0;
    double best = -1e30;
    for (int p = 0; p < npat; ++p) {
      acc[p] += (double)seq[p] / (double)tot;
      if (acc[p] > best) { best = acc[p]; pick = p; }
    }
    acc[pick] -= 1.0;
    h[b] = pick;
  }
  CK(cudaMemcpy(g_assign, h, (size_t)grid * sizeof(int), cudaMemcpyHostToDevice));
  free(h);
}

struct Res { double ms; double gbs; int unsupported; };

static Res run(int O, int reps) {
  int grid = g_numSm * O;
  long long BP = TOT_U4 / grid;
  ensure((long long)grid * BP * SPARSE_STRIDE, grid);
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    launch_mix(grid, g_T, BP);
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
  res.gbs = (double)TOT_U4 * 16.0 / (best * 1e-3) / 1e9;   // 按**逻辑**字节算 GB/s
  res.unsupported = unsupported;
  return res;
}

static Res run_single(int pat, int T, int O, int reps) {
  int seq[NPAT] = {0, 0, 0, 0};
  seq[pat] = 1;
  g_T = T;
  fill_assign(g_numSm * O, seq, NPAT);
  return run(O, reps);
}

static Res run_mix(int T, int O, const int* seq, int reps) {
  g_T = T;
  fill_assign(g_numSm * O, seq, NPAT);
  return run(O, reps);
}

static const char* PATNAME(int p) {
  switch (p) { case 0: return "coalesced"; case 1: return "sparse8x";
               case 2: return "scalar4B"; case 3: return "l2res16M"; }
  return "?";
}

static void fill_all() {
  long long tot = TOT_U4 * SPARSE_STRIDE;
  ensure(tot, 1);
  unsigned long long* dref = nullptr;
  CK(cudaMalloc(&dref, 8));
  CK(cudaMemset(dref, 0, 8));
  fill_data_kernel<<<(int)((tot + 255) / 256), 256>>>(g_data, tot, dref);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(dref));
}

// ---- host 参考（小规模）----
static unsigned long long ref_block(int pat, int b, long long BP) {
  unsigned long long r = 0;
  if (pat == 0) {
    const long long base = (long long)b * BP;
    for (long long i = 0; i < BP; ++i) {
      long long u = base + i;
      r ^= (unsigned long long)(mix32((uint32_t)u, 0u) ^ mix32((uint32_t)u, 1u) ^
                                mix32((uint32_t)u, 2u) ^ mix32((uint32_t)u, 3u));
    }
  } else if (pat == 1) {
    const long long base = (long long)b * BP * SPARSE_STRIDE;
    for (long long i = 0; i < BP; ++i) {
      long long u = base + i * SPARSE_STRIDE;
      r ^= (unsigned long long)(mix32((uint32_t)u, 0u) ^ mix32((uint32_t)u, 1u) ^
                                mix32((uint32_t)u, 2u) ^ mix32((uint32_t)u, 3u));
    }
  } else if (pat == 2) {
    const long long base = (long long)b * (BP * 4);
    for (long long p = 0; p < BP * 4; ++p)
      r ^= (unsigned long long)mix32((uint32_t)((base + p) / 4), (uint32_t)((base + p) % 4));
  } else {
    for (long long i = 0; i < BP; ++i) {
      long long ph = i & (L2N4 - 1);
      r ^= (unsigned long long)(mix32((uint32_t)ph, 0u) ^ mix32((uint32_t)ph, 1u) ^
                                mix32((uint32_t)ph, 2u) ^ mix32((uint32_t)ph, 3u));
    }
  }
  return r;
}

// ---- 由单模式 baseline 算「时间切片（无干扰）」期望逻辑带宽 ----
static double g_bw_solo[NPAT];          // 各模式单独跑（同 T,O）的逻辑 GB/s
static double g_bw_best[NPAT];          // 各模式跨配置的最优逻辑 GB/s（oracle 用）

static double expected_additive(const int* seq) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  double inv = 0.0;
  for (int p = 0; p < NPAT; ++p)
    if (seq[p] > 0) inv += (double)seq[p] / (double)tot / g_bw_solo[p];
  return inv > 0 ? 1.0 / inv : 0.0;
}

static double oracle_slice(const int* seq) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  double inv = 0.0;
  for (int p = 0; p < NPAT; ++p)
    if (seq[p] > 0) inv += (double)seq[p] / (double)tot / g_bw_best[p];
  return inv > 0 ? 1.0 / inv : 0.0;
}

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  printf("########################################################################\n");
  printf("# §72 异构访存模式**共驻**：同一常驻 grid 内不同 block 跑不同模式\n");
  printf("#   复检 §71.7 headline 四「一个常驻 kernel 内 task 类型越杂、越难一套调度参数通吃」\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 1 GiB（uint4 计）；reg U=%d；__maxnreg__(32)\n",
         g_numSm, U_ILP);
  printf("# 混合 = 误差扩散交错 assign[]；期望 = 各模式单独吞吐按逻辑字节份额的调和平均\n");
  printf("########################################################################\n");

  fill_all();

  int mainO = 4, mainT = 256;   // 主配置 O·T=1024

  // ===== §A 正确性：混合小规模对 host 参考（XOR over blocks × 各自 ref_block）=====
  {
    long long BP = 1 << 12;
    int NBLK = 16;
    int seq[NPAT] = {4, 4, 4, 4};   // 4/4/4/4
    ensure((long long)NBLK * BP * SPARSE_STRIDE, NBLK);
    // 先按同样误差扩散法生成 host 侧 assign
    int hassign[16];
    {
      long long tot = 16; static double a[NPAT] = {0, 0, 0, 0};
      for (int b = 0; b < NBLK; ++b) {
        int pick = 0; double best = -1e30;
        for (int p = 0; p < NPAT; ++p) { a[p] += (double)seq[p] / (double)tot;
          if (a[p] > best) { best = a[p]; pick = p; } }
        a[pick] -= 1.0; hassign[b] = pick;
      }
    }
    unsigned long long ref = 0;
    for (int b = 0; b < NBLK; ++b) ref ^= ref_block(hassign[b], b, BP);
    fill_assign(NBLK, seq, NPAT);
    g_T = 256;
    CK(cudaMemset(g_sink, 0, 8));
    launch_mix(NBLK, 256, BP);
    CK(cudaDeviceSynchronize());
    unsigned long long s = 0; CK(cudaMemcpy(&s, g_sink, 8, cudaMemcpyDeviceToHost));
    printf("\n===== §A 正确性（NBLK=%d，BP=%lld，assign=4/4/4/4 交错）vs host 参考 =====\n", NBLK, BP);
    printf("  mix : sink=%llu ref=%llu -> %s\n", s, ref, s == ref ? "OK" : "WRONG");
  }

  // ===== §B 单模式 baseline（同一 pat_mix kernel，all-same assign）=====
  printf("\n===== §B 单模式 baseline（pat_mix 全同 assign）：逻辑 GB/s =====\n");
  {
    const char* cfgname[3] = {"T=256,O=4 (OT=1024)", "T=1024,O=1 (OT=1024)", "T=256,O=8 (OT=2048)"};
    int Ts[3] = {256, 1024, 256};
    int Os[3] = {4, 1, 8};
    for (int p = 0; p < NPAT; ++p) {
      g_bw_best[p] = 0;
      printf("  %-11s", PATNAME(p));
      for (int c = 0; c < 3; ++c) {
        Res r = run_single(p, Ts[c], Os[c], 3);
        if (c == 0) g_bw_solo[p] = r.gbs;
        if (r.gbs > g_bw_best[p]) g_bw_best[p] = r.gbs;
        printf("  %s=%7.1f", cfgname[c], r.unsupported ? -1.0 : r.gbs);
      }
      printf("\n");
    }
  }

  // ===== §C 混合矩阵（主配置 T=256,O=4）：measured vs 期望（时间切片）=====
  printf("\n===== §C 混合（T=256,O=4，OT=1024）：measured vs 期望(时间切片) / 比 =====\n");
  printf("  %-22s %10s %10s %8s\n", "mixture(coal,sp,sc,l2)", "measured", "expected", "meas/exp");
  struct Mix { const char* nm; int seq[NPAT]; };
  Mix mixes[] = {
    {"50/50 coal+sparse",   {1, 1, 0, 0}},
    {"50/50 coal+scalar",   {1, 0, 1, 0}},
    {"50/50 coal+l2res",    {1, 0, 0, 1}},
    {"50/50 scalar+l2res",  {0, 0, 1, 1}},
    {"50/50 sparse+l2res",  {0, 1, 0, 1}},
    {"equal 25x4",          {1, 1, 1, 1}},
    {"50/25/25 coal/sp/l2", {2, 1, 0, 1}},
    {"25/50/25 coal/sp/l2", {1, 2, 0, 1}},
    {"75/25 coal+sparse",   {3, 1, 0, 0}},
  };
  int nmix = (int)(sizeof(mixes) / sizeof(mixes[0]));
  for (int m = 0; m < nmix; ++m) {
    Res r = run_mix(mainT, mainO, mixes[m].seq, 3);
    double exp = expected_additive(mixes[m].seq);
    printf("  %-22s %10.1f %10.1f %8.3f\n", mixes[m].nm,
           r.unsupported ? -1.0 : r.gbs, exp, r.unsupported ? 0.0 : r.gbs / exp);
  }

  // ===== §D (T,O) 拆分能否通吃：选几个混合扫 O·T=1024 的拆分 + OT=2048 =====
  //   每个配置用**同配置**的单模式基线算期望（消除 T/O 对单模式本身的影响）。
  printf("\n===== §D 混合扫 (T,O)：一套静态配置能否通吃多样混合？ =====\n");
  {
    int Ts[] = {128, 256, 512, 1024};
    int Os[] = {8, 4, 2, 1};
    int dmix[] = {5, 1, 0};   // 索引到 mixes[]：equal4 / coal+scalar / coal+sparse
    for (int di = 0; di < 3; ++di) {
      int mi = dmix[di];
      int* seq = mixes[mi].seq;
      printf("  --- %s ---\n", mixes[mi].nm);
      double bm = 0; int bT = 0, bO = 0;
      for (int t = 0; t < 4; ++t) {
        // 同配置单模式基线
        double inv = 0; long long tot = 0;
        for (int p = 0; p < NPAT; ++p) tot += seq[p];
        for (int p = 0; p < NPAT; ++p) {
          if (!seq[p]) continue;
          Res s = run_single(p, Ts[t], Os[t], 3);
          inv += (double)seq[p] / (double)tot / (s.gbs > 0 ? s.gbs : 1e-9);
        }
        double expc = inv > 0 ? 1.0 / inv : 0;
        Res r = run_mix(Ts[t], Os[t], seq, 3);
        printf("    OT=1024 T=%-5d O=%d -> %7.1f (expected %7.1f, %.3f)\n",
               Ts[t], Os[t], r.gbs, expc, expc > 0 ? r.gbs / expc : 0.0);
        if (r.gbs > bm) { bm = r.gbs; bT = Ts[t]; bO = Os[t]; }
      }
      double orc = oracle_slice(seq);
      printf("    -> best single (T=%d,O=%d)=%.1f ; oracle(各模式各自最优)=%.1f ; 异质性税=%.3f\n",
             bT, bO, bm, orc, orc > 0 ? bm / orc : 0.0);
    }
    printf("     [各模式跨配置最优：coal=%.1f sp=%.1f sc=%.1f l2=%.1f]\n",
           g_bw_best[0], g_bw_best[1], g_bw_best[2], g_bw_best[3]);
  }

  // ===== §E 稀疏份额扫描：coal+sparse，sparse 占比 0..100% =====
  printf("\n===== §E coal+sparse 份额扫描（T=256,O=4）：逻辑吞吐 vs 物理吞吐 =====\n");
  printf("  %-8s %10s %10s %10s %8s\n", "sp%%", "measured", "expected", "physGB/s", "meas/exp");
  {
    int shares[] = {0, 1, 2, 3, 5, 7, 9, 10};   // sparse:coal = 0:10 .. 10:0
    for (int k = 0; k < 8; ++k) {
      int seq[NPAT] = {10 - shares[k], shares[k], 0, 0};
      if (seq[0] == 0 && seq[1] == 0) continue;
      Res r = run_mix(mainT, mainO, seq, 3);
      double exp = expected_additive(seq);
      double frac = (double)shares[k] / 10.0;
      // 物理 DRAM 字节 ≈ coal 逻辑 ×(1-frac) + sparse 逻辑 ×4×frac
      //   （sparse 每 128 B 行只用 16 B ⇒ 32 B 扇区粒度 → 物理 4×，见 §71 §D ncu 实测 4.01×）
      double phys = r.gbs * ((1.0 - frac) + 4.0 * frac);
      printf("  %-8d %10.1f %10.1f %10.1f %8.3f\n", shares[k] * 10,
             r.gbs, exp, phys, r.gbs / exp);
    }
  }

  // ===== §F 结论汇总 =====
  printf("\n===== §F 结论 =====\n");
  printf("  - §C meas/exp > 1 => 混合**优于**时间切片（共驻互补）；< 1 => 共驻**干扰**（抢 HBM）。\n");
  printf("  - §D 异质性税 = 混合最佳单一 (T,O) / oracle(各模式各自最优)。<1 即『一套参数通吃』的代价。\n");
  printf("  - §E physGB/s(≈4× 稀疏流量) 应逼近 HBM 天花板 => 逻辑塌是**物理流量墙**、非额外干扰。\n");
  printf("\n########## §72 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 6) { printf("usage: run <T> <O> <reps> <r0,r1,r2,r3>\n"); return 1; }
    g_T = atoi(argv[2]);
    int O = atoi(argv[3]);
    int reps = atoi(argv[4]);
    int seq[NPAT] = {0, 0, 0, 0};
    if (sscanf(argv[5], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) < 4) {
      printf("bad seq\n"); return 1;
    }
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    Res r = run_mix(g_T, O, seq, reps);
    printf("  T=%d O=%d seq=%d,%d,%d,%d grid=%d -> %s ms=%.4f GB/s=%.1f\n",
           g_T, O, seq[0], seq[1], seq[2], seq[3], g_numSm * O,
           r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs);
    return 0;
  }
  printf("usage: hetero_access_mix {bench|run <T> <O> <reps> <r0,r1,r2,r3>}\n");
  return 1;
}
