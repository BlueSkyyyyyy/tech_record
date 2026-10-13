// ============================================================================
// P83-1 (§71)：把 §70 的尾巴「只测连续/规则的 `__ldcs` uint4 流，未扫 strided / 非对齐 /
//   L2 命中等访存模式」补掉——
//   在**固定每线程寄存器预算**（`__maxnreg__(R)`）与**固定载体**（寄存器读 + U=4 ILP）下，
//   把「数据面只认『每 SM 总线程 O·T』这一个合量」的结论拿到**四种访存模式**上复检：
//   合量关系是否模式无关？饱和点（达到峰值所需的 O·T）是否随模式移动？
//
// 背景（接 §60 / §61 / §68 / §69 / §70）：
//   - §60/§61/§72：数据面带宽天花板 ~3200 GB/s（HBM3 峰值），「ILP 与占用率是替代品」。
//   - §68/§69/§70：寄存器预算 / 占用率 O / 每块线程数 T 都是同一枚硬币的不同面；**§70 headline：
//     饱和区（`O·T≥1024`）吞吐只由『每 SM 总线程 O·T』决定，与 (T,O) 拆分、与寄存器预算无关**。
//   - §70.7 明文尾巴：**「只测连续/规则的 `__ldcs` uint4 流，未扫 strided / 非对齐 / L2 命中等
//     访存模式」**。
//
// 本节问：把访存模式作为单变量（同一 `O·T` 面、同一 R、同一 U、同一逻辑字节数），
//   ①「吞吐只由 O·T 决定」这条合量关系是否在各模式下仍成立（模式只改常数/天花板）？
//   ② 各模式的**饱和点 O·T**（吞吐逼近平台的最小 O·T）与**绝对天花板**是否不同？
//   ③ ncu 侧各模式的 DRAM 字节 / L1 sectors 是否与「逻辑 GB/s」的差异一致（归因）。
//
// 四种模式（逻辑字节数恒 = TOT_U4·16 B = 1 GiB，便于横向比「有用 GB/s」）：
//   m0 coalesced：连续 uint4 `__ldcs`（= §60/§70 基线；物理字节 = 逻辑字节）。
//   m1 sparse   ：每 8 个 uint4 只读 1 个（物理缓冲 8×，`__ldcs`）→ 每 128 B 行只用 16 B，
//                 物理流量 = 8× 逻辑，**典型非 coalesced / 稀疏访问**（逻辑 GB/s 会塌到 ~1/8 HBM）。
//   m2 scalar   ：连续但按 **4 B** `__ldcs` 读（指令数 4×、无向量宽度）→ 发射/指令面受限。
//   m3 l2res    ：连续 uint4 **普通 load**（非流式）读一个 **16 MiB 常驻缓冲**（index 掩码回绕）→
//                 中间量驻 L2、几乎不碰 HBM，测「纯在飞 load 上限」是否仍认 O·T。
//
// 载体固定：寄存器读 + U=4 路 ILP（= §60/§61/§68/§69/§70）。寄存器上限固定 `__maxnreg__(32)`
//   （保证 O·T 能上到线程面 2048；四模式共用同一 R，故 R 不是混淆变量）。
//
// 用法：./access_pattern_surface bench
//       ./access_pattern_surface run <pat> <T> <O> <reps>      # pat=0..3
// 目标卡 H100 / sm90a。安全：脚本带 timeout；纯读微基准，无环协议 / 无死锁。
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
#define U_ILP 4
#define TOT_U4 (67108864LL)        // 逻辑总读 1 GiB（uint4 计）；四模式同一逻辑字节数
#define SPARSE_STRIDE 8            // m1：每 8 个 uint4 读 1 个 → 物理 8×
#define L2N4 (1 << 20)             // m3：常驻 16 MiB = 1M uint4（< 50 MB L2）

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- m0：连续 uint4 `__ldcs`（基线）----
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

// ---- m1：稀疏（每 SPARSE_STRIDE 个 uint4 读 1 个；物理缓冲 8×）----
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

// ---- m2：连续但按 4 B `__ldcs` 读（N32 = BP*4 个 u32）----
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

// ---- m3：连续 uint4 普通 load 读常驻缓冲（index 掩码回绕到 L2N4）----
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

// ---- block 归约 + atomicXor ----
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

// ---- 模式 0..3 的 kernel（R=32 固定）----
#define DEF_PAT(PNAME, BODY)                                                              \
  __global__ void __maxnreg__(32) pat_##PNAME(const void* __restrict__ raw, long long BP, \
                                              long long aux, unsigned long long* sink) {  \
    const int tid = threadIdx.x;                                                          \
    const int T = blockDim.x;                                                             \
    BODY                                                                                  \
    reduce_xor<MAXT>(part, tid, T, sink);                                                 \
  }

DEF_PAT(m0,
  const uint4* base = (const uint4*)raw + (long long)blockIdx.x * BP;
  unsigned long long part = read_coal<U_ILP>(base, BP, tid, T);
)
DEF_PAT(m1,
  const uint4* base = (const uint4*)raw + (long long)blockIdx.x * BP * SPARSE_STRIDE;
  unsigned long long part = read_sparse<U_ILP>(base, BP, tid, T);
)
DEF_PAT(m2,
  const uint32_t* base = (const uint32_t*)raw + (long long)blockIdx.x * (BP * 4);
  unsigned long long part = read_scalar<U_ILP>(base, BP * 4, tid, T);
)
DEF_PAT(m3,
  unsigned long long part = read_l2res<U_ILP>((const uint4*)raw, BP, aux, tid, T);
)
#undef DEF_PAT

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
static long long g_data_cap = 0;
static int g_numSm = 0;
static int g_T = 256;

static void ensure(long long data_u4) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
}

static int g_last_ok = 1;
static int g_pat = 0;

static void launch_pat(int grid, int T, long long BP) {
  const void* raw = (const void*)g_data;
  long long aux = L2N4 - 1;
  switch (g_pat) {
    case 0: pat_m0<<<grid, T>>>(raw, BP, aux, g_sink); break;
    case 1: pat_m1<<<grid, T>>>(raw, BP, aux, g_sink); break;
    case 2: pat_m2<<<grid, T>>>(raw, BP, aux, g_sink); break;
    case 3: pat_m3<<<grid, T>>>(raw, BP, aux, g_sink); break;
    default: printf("  !! unknown pat=%d\n", g_pat); g_last_ok = 0; return;
  }
  g_last_ok = (cudaGetLastError() == cudaSuccess);
}

static int occ_pat(int T) {
  int nb = 0;
  cudaError_t e = cudaErrorInvalidValue;
  switch (g_pat) {
    case 0: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pat_m0, T, 0); break;
    case 1: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pat_m1, T, 0); break;
    case 2: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pat_m2, T, 0); break;
    case 3: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pat_m3, T, 0); break;
  }
  if (e != cudaSuccess) return -1;
  return nb;
}

struct Res { double ms; double gbs; int unsupported; };

static Res run(int O, int reps) {
  int grid = g_numSm * O;
  long long BP = TOT_U4 / grid;
  ensure((long long)grid * BP * SPARSE_STRIDE);   // 按最大需求（m1 物理 8×）分配
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    launch_pat(grid, g_T, BP);
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

static const char* PATNAME(int p) {
  switch (p) { case 0: return "coalesced"; case 1: return "sparse8x";
               case 2: return "scalar4B"; case 3: return "l2res16M"; }
  return "?";
}

static void fill_all() {
  long long tot = TOT_U4 * SPARSE_STRIDE;   // 最坏（m1）
  ensure(tot);
  unsigned long long* dref = nullptr;
  CK(cudaMalloc(&dref, 8));
  CK(cudaMemset(dref, 0, 8));
  fill_data_kernel<<<(int)((tot + 255) / 256), 256>>>(g_data, tot, dref);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(dref));
}

struct Point { int pat, T, O, prod; double gbs; int unsupported; };
#define MAXPTS 256
static Point g_pts[MAXPTS];
static int g_npts = 0;

static void sweep_T(int pat, int T, const int* Os, int nO, int reps) {
  g_pat = pat; g_T = T;
  printf("  %-11s T=%-5d ", PATNAME(pat), T);
  for (int k = 0; k < nO; ++k) {
    int O = Os[k];
    Res r = run(O, reps);
    printf(" O=%-2d(OT=%-4d)=%7.1f ", O, T * O, r.unsupported ? -1.0 : r.gbs);
    if (g_npts < MAXPTS) {
      Point p; p.pat = pat; p.T = T; p.O = O; p.prod = T * O;
      p.gbs = r.gbs; p.unsupported = r.unsupported; g_pts[g_npts++] = p;
    }
  }
  int lim = occ_pat(T);
  printf(" [blk/SM≤%d]\n", lim);
}

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));

  printf("########################################################################\n");
  printf("# §71 访存模式 × O·T 合量：固定 R=32/U=4，四模式（同一逻辑 1 GiB）下\n");
  printf("#   数据面是否仍只认『每 SM 总线程 O·T』？饱和点/天花板是否随模式移动？\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 1 GiB（uint4 计）；载体 reg U=%d；__maxnreg__(32)\n",
         g_numSm, U_ILP);
  printf("# m1 物理 8×（每 128 B 行用 16 B）；m3 常驻 16 MiB（< L2）；ncu 另跑\n");
  printf("########################################################################\n");

  fill_all();

  // ===== §A 正确性：每模式小规模对 host 参考 =====
  {
    long long BP = 1 << 12;
    int NBLK = 8;
    long long tot = (long long)NBLK * BP;
    printf("\n===== §A 正确性（NBLK=%d，BP=%lld，逻辑元素=%lld）：每模式 vs host 参考（XOR）=====\n",
           NBLK, BP, tot);
    // host 参考（XOR over 逻辑元素，按各模式的物理映射）
    unsigned long long ref[4] = {0, 0, 0, 0};
    for (long long i = 0; i < tot; ++i) {
      ref[0] ^= (unsigned long long)(mix32((uint32_t)i, 0u) ^ mix32((uint32_t)i, 1u) ^
                                     mix32((uint32_t)i, 2u) ^ mix32((uint32_t)i, 3u));
      long long ph = i * SPARSE_STRIDE;
      ref[1] ^= (unsigned long long)(mix32((uint32_t)ph, 0u) ^ mix32((uint32_t)ph, 1u) ^
                                     mix32((uint32_t)ph, 2u) ^ mix32((uint32_t)ph, 3u));
    }
    for (long long p = 0; p < tot * 4; ++p)
      ref[2] ^= (unsigned long long)mix32((uint32_t)(p / 4), (uint32_t)(p % 4));
    // m3 每个 block 读同一段（无 block 偏移），故总 XOR = NBLK 份单块 XOR
    for (int b = 0; b < NBLK; ++b)
      for (long long i = 0; i < BP; ++i) {
        long long ph = i & (L2N4 - 1);
        ref[3] ^= (unsigned long long)(mix32((uint32_t)ph, 0u) ^ mix32((uint32_t)ph, 1u) ^
                                       mix32((uint32_t)ph, 2u) ^ mix32((uint32_t)ph, 3u));
      }
    for (int pat = 0; pat < 4; ++pat) {
      g_pat = pat; g_T = 256;
      CK(cudaMemset(g_sink, 0, 8));
      launch_pat(NBLK, 256, BP);
      CK(cudaDeviceSynchronize());
      unsigned long long s = 0; CK(cudaMemcpy(&s, g_sink, 8, cudaMemcpyDeviceToHost));
      printf("  %-11s : sink=%llu ref=%llu -> %s\n", PATNAME(pat), s, ref[pat],
             s == ref[pat] ? "OK" : "WRONG");
    }
  }

  // ===== §B 每模式：固定 T=256，扫 O·T = 256/512/1024/2048 =====
  printf("\n===== §B 每模式固定 T=256，扫 O=1/2/4/8（O·T=256..2048）：有用 GB/s =====\n");
  {
    int Os[] = {1, 2, 4, 8};
    for (int pat = 0; pat < 4; ++pat) sweep_T(pat, 256, Os, 4, 3);
  }

  // ===== §C 每模式：固定 O·T=1024，扫 T=128/256/512/1024（O=8/4/2/1）=====
  printf("\n===== §C 每模式固定 O·T=1024，扫 T=128/256/512/1024（O=8/4/2/1）：有用 GB/s =====\n");
  {
    int Ts[] = {128, 256, 512, 1024};
    int Os[] = {8, 4, 2, 1};
    for (int pat = 0; pat < 4; ++pat) {
      for (int t = 0; t < 4; ++t) sweep_T(pat, Ts[t], &Os[t], 1, 3);
    }
  }

  // ===== §D 汇总：每模式的 O·T 标度与饱和点 =====
  printf("\n===== §D 汇总：每模式在 T=256 下的 O·T 标度（以 O·T=256 为 1.00）×，及饱和点 =====\n");
  printf("  %-11s %8s %8s %8s %8s   %-14s\n", "pattern", "OT256", "OT512", "OT1024", "OT2048", "饱和点(≈平台)");
  for (int pat = 0; pat < 4; ++pat) {
    double v[4] = {0, 0, 0, 0};
    for (int i = 0; i < g_npts; ++i)
      if (g_pts[i].pat == pat && g_pts[i].T == 256) {
        int idx = g_pts[i].prod == 256 ? 0 : g_pts[i].prod == 512 ? 1
                : g_pts[i].prod == 1024 ? 2 : g_pts[i].prod == 2048 ? 3 : -1;
        if (idx >= 0) v[idx] = g_pts[i].gbs;
      }
    double base = v[0] > 0 ? v[0] : 1.0;
    double mx = 0; for (int i = 0; i < 4; ++i) if (v[i] > mx) mx = v[i];
    // 饱和点 = 最小 O·T 使吞吐 ≥ 0.95 × 该模式行内最大值
    const char* sat = ">2048";
    if (v[0] >= 0.95 * mx) sat = "256";
    else if (v[1] >= 0.95 * mx) sat = "512";
    else if (v[2] >= 0.95 * mx) sat = "1024";
    printf("  %-11s %8.2f %8.2f %8.2f %8.2f   %-14s\n", PATNAME(pat),
           v[0] / base, v[1] / base, v[2] / base, v[3] / base, sat);
  }

  printf("\n########## §71 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 6) { printf("usage: run <pat> <T> <O> <reps>\n"); return 1; }
    g_pat = atoi(argv[2]);
    g_T = atoi(argv[3]);
    int O = atoi(argv[4]);
    int reps = atoi(argv[5]);
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    Res r = run(O, reps);
    int lim = occ_pat(g_T);
    printf("  pat=%s T=%d O=%d grid=%d blk/SM(API)=%d -> %s ms=%.4f GB/s=%.1f\n",
           PATNAME(g_pat), g_T, O, g_numSm * O, lim, r.unsupported ? "UNSUP" : "OK",
           r.ms, r.gbs);
    return 0;
  }
  printf("usage: access_pattern_surface {bench|run <pat> <T> <O> <reps>}\n");
  return 1;
}
