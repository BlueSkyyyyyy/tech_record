// ============================================================================
// P92-1 (§80)：补掉 P91-1（§79）§79.6 明留的两条尾巴——
//   (tail 1) 「`meas/E` 的 spread 混入 `E` 的拆分依赖（`scalar+l2res`/`equal4`），故比值需与
//            绝对量并读」；
//   (tail 2) 「`U=4`/`__maxnreg__(32)` 单一（每线程 ILP 与寄存器预算未当独立变量）」。
//
// 背景（接 §71–§79）：§78/§79 证实「发射受限 × 访存受限」的共驻税（`coal+scalar` 的 `meas/E`）
//   随每 SM warp 数 `Wsm=O·T` 单调收回（0.77→0.83→0.91→1.00），且与 `(T,O)` 拆分无关；但
//   §79.3 发现 `scalar+l2res`/`equal4` 的 `meas/E` 在**同一 `Wsm` 内**仍有较大 spread，而它们
//   **实测绝对吞吐几乎恒定**——spread 来自基线 `E`（同列 solo）本身随拆分变，不是税变。
//
// 本节把**两个正交问题**做出来：
//   Q1 **口径**：把「共驻税」从**比值 `meas/E`（分母随列漂移）**换成**绝对量**——
//      (a) 直接看 mixed 的绝对逻辑 GB/s 随 `Wsm` / 拆分的标度；
//      (b) 用**固定参考基线** `E_ref`（同一 `Wsm=1024/T256/O4` solo，跨列不变）归一
//          `taxRef = E_ref / mixed_abs`。判定「税只由 `Wsm` 定、与拆分无关」在哪种口径下干净。
//   Q2 **每线程 ILP `U`**：把 `U`（独立累加器 / 在飞 load 数）从固定 `4` 变成**独立变量**，
//      在**固定 `Wsm`（=1024，T256/O4）**下扫 `U∈{1,2,4,8}`，问：共驻税是否也随**每线程在飞
//      load 数**（`Wsm·U`）收回？即「税是每 SM warp 预算的函数」是否可细化为「每 SM 在飞 load 量」。
//      再测 `U` 与 `Wsm` 的**替代性**（低 `U`×高 `Wsm` vs 高 `U`×低 `Wsm`）。
//
// 仍完全复用 §72–§79 的四访存模式 / 逻辑字节 / XOR 参考 / 误差扩散 `assign[]` / SM 分配
//   (naive/bind_bal) / per-SM 原子子段游标；载体固定寄存器读 + `__maxnreg__(32)`。
//
// 用法：./hetero_sm_tax_ilp bench
//       ./hetero_sm_tax_ilp run <T> <O> <U> <map> <r0,r1,r2,r3>   # map 0=inter,1=bind,2=bind_bal
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
#define TOT_U4 (67108864LL)        // 逻辑总读 1 GiB（uint4 计）；所有混合份额同一逻辑字节数
#define SPARSE_STRIDE 8            // m1：每 8 个 uint4 读 1 个 → 物理 8×（扇区 4×）
#define L2N4 (1 << 20)             // m3：常驻 16 MiB = 1M uint4（< 50 MB L2）
#define NPAT 4
#define MAXGRID 4096
#define NU 4                       // U ∈ {1,2,4,8}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 四模式读函数（与 §71–§79 逐字一致）----
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

// ---- 统一 kernel（模板化在**每线程 ILP `U`** 上；§79 固定 U=4，本节把 U 变成独立变量）----
//   interleave（bind=0）：每 block 的区 = bb_base[blockIdx]。
//   bind（bind=1）：pat = mode_of_sm[smid]；该 SM 的 O 个 block 用 per-SM 原子游标取局部序号 local。
template <int U>
__global__ void __maxnreg__(32) pat_hetero_u(const void* __restrict__ raw, long long aux, int bind,
                                             int O,
                                             const int* __restrict__ assign,
                                             const int* __restrict__ mode_of_sm,
                                             const long long* __restrict__ bb_base,
                                             const long long* __restrict__ bb_bp,
                                             const long long* __restrict__ sm_base,
                                             const long long* __restrict__ sm_bp,
                                             int* __restrict__ sm_ctr,
                                             unsigned int* __restrict__ blk_smid,
                                             unsigned long long* sink) {
  const int tid = threadIdx.x;
  const int T = blockDim.x;
  unsigned int smid = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  blk_smid[blockIdx.x] = smid;
  int pat;
  long long base, BP;
  if (bind) {
    pat = mode_of_sm[smid];
    __shared__ int s_local;
    if (tid == 0) s_local = atomicAdd(&sm_ctr[smid], 1);
    __syncthreads();
    int local = s_local;
    long long per = sm_bp[smid];
    base = sm_base[smid] + (long long)local * per;
    BP = (local < O) ? per : 0;
  } else {
    pat = assign[blockIdx.x];
    base = bb_base[blockIdx.x];
    BP = bb_bp[blockIdx.x];
  }
  unsigned long long part = 0ull;
  switch (pat) {
    case 0: part = read_coal<U>((const uint4*)raw + base, BP, tid, T); break;
    case 1: part = read_sparse<U>((const uint4*)raw + base, BP, tid, T); break;
    case 2: part = read_scalar<U>((const uint32_t*)raw + base * 4, BP * 4, tid, T); break;
    default: part = read_l2res<U>((const uint4*)raw, BP, aux, tid, T); break;
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
static int* g_mode_of_sm = nullptr;
static int* g_sm_ctr = nullptr;
static long long* g_bb_base = nullptr;
static long long* g_bb_bp = nullptr;
static long long* g_sm_base = nullptr;
static long long* g_sm_bp = nullptr;
static unsigned int* g_blk_smid = nullptr;
static long long g_data_cap = 0;
static int g_assign_cap = 0, g_smcap = 0, g_gridcap = 0;
static int g_numSm = 0;
static int g_T = 256;
static int g_O = 1;
static int g_U = 4;

static long long g_W[NPAT], g_BP[NPAT], g_basep[NPAT], g_phys[NPAT];
static int g_nblk[NPAT];

static void ensure(long long data_u4, int grid) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_sm_ctr) CK(cudaMalloc(&g_sm_ctr, (size_t)4096 * sizeof(int)));
  if (grid > g_assign_cap) {
    if (g_assign) CK(cudaFree(g_assign));
    CK(cudaMalloc(&g_assign, (size_t)grid * sizeof(int)));
    g_assign_cap = grid;
  }
  if (g_numSm > g_smcap) {
    if (g_mode_of_sm) CK(cudaFree(g_mode_of_sm));
    CK(cudaMalloc(&g_mode_of_sm, (size_t)g_numSm * sizeof(int)));
    if (g_sm_base) CK(cudaFree(g_sm_base));
    CK(cudaMalloc(&g_sm_base, (size_t)g_numSm * sizeof(long long)));
    if (g_sm_bp) CK(cudaFree(g_sm_bp));
    CK(cudaMalloc(&g_sm_bp, (size_t)g_numSm * sizeof(long long)));
    g_smcap = g_numSm;
  }
  if (grid > g_gridcap) {
    if (g_blk_smid) CK(cudaFree(g_blk_smid));
    CK(cudaMalloc(&g_blk_smid, (size_t)grid * sizeof(unsigned int)));
    if (g_bb_base) CK(cudaFree(g_bb_base));
    CK(cudaMalloc(&g_bb_base, (size_t)grid * sizeof(long long)));
    if (g_bb_bp) CK(cudaFree(g_bb_bp));
    CK(cudaMalloc(&g_bb_bp, (size_t)grid * sizeof(long long)));
    g_gridcap = grid;
  }
}

static int g_last_ok = 1, g_cur_bind = 0;

static long long compute_layout(int O, long long* out_need) {
  long long phys_acc = 0, need = L2N4 + 1024;
  for (int p = 0; p < NPAT; ++p) {
    if (!g_nblk[p] || g_W[p] <= 0) { g_BP[p] = 0; g_basep[p] = phys_acc; g_phys[p] = 0; continue; }
    g_BP[p] = g_W[p] / g_nblk[p];
    if (g_BP[p] < 1) g_BP[p] = 1;
    g_basep[p] = phys_acc;
    long long stride = (p == 1) ? SPARSE_STRIDE : 1;
    if (p == 3) { g_phys[p] = 0; }
    else g_phys[p] = (long long)g_nblk[p] * g_BP[p] * stride;
    phys_acc += g_phys[p];
    long long endp = g_basep[p] + g_phys[p];
    if (endp > need) need = endp;
  }
  need += TOT_U4;
  *out_need = need;
  return phys_acc;
}

static void launch_hetero(int grid) {
  long long aux = L2N4 - 1;
  CK(cudaMemset(g_sm_ctr, 0, (size_t)g_numSm * sizeof(int)));
  switch (g_U) {
    case 1:
      pat_hetero_u<1><<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_O, g_assign,
                                     g_mode_of_sm, g_bb_base, g_bb_bp, g_sm_base, g_sm_bp,
                                     g_sm_ctr, g_blk_smid, g_sink);
      break;
    case 2:
      pat_hetero_u<2><<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_O, g_assign,
                                     g_mode_of_sm, g_bb_base, g_bb_bp, g_sm_base, g_sm_bp,
                                     g_sm_ctr, g_blk_smid, g_sink);
      break;
    case 4:
      pat_hetero_u<4><<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_O, g_assign,
                                     g_mode_of_sm, g_bb_base, g_bb_bp, g_sm_base, g_sm_bp,
                                     g_sm_ctr, g_blk_smid, g_sink);
      break;
    default:
      pat_hetero_u<8><<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_O, g_assign,
                                     g_mode_of_sm, g_bb_base, g_bb_bp, g_sm_base, g_sm_bp,
                                     g_sm_ctr, g_blk_smid, g_sink);
      break;
  }
  g_last_ok = (cudaGetLastError() == cudaSuccess);
}

static void fill_assign(int grid, const int* seq) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  static double acc[NPAT];
  for (int p = 0; p < NPAT; ++p) acc[p] = 0.0;
  int* h = (int*)malloc((size_t)grid * sizeof(int));
  for (int b = 0; b < grid; ++b) {
    int pick = 0; double best = -1e30;
    for (int p = 0; p < NPAT; ++p) {
      acc[p] += (double)seq[p] / (double)tot;
      if (acc[p] > best) { best = acc[p]; pick = p; }
    }
    acc[pick] -= 1.0; h[b] = pick;
  }
  CK(cudaMemcpy(g_assign, h, (size_t)grid * sizeof(int), cudaMemcpyHostToDevice));
  free(h);
}

static void fill_mode_of_sm(const int* nsm, int* h_out) {
  int* h = (h_out ? h_out : (int*)malloc((size_t)g_numSm * sizeof(int)));
  int rem[NPAT];
  for (int p = 0; p < NPAT; ++p) rem[p] = nsm[p];
  int s = 0;
  for (int i = 0; i < g_numSm; ++i) {
    int guard = 0;
    while (rem[s % NPAT] == 0 && guard < NPAT) { s++; guard++; }
    int p = s % NPAT; h[i] = p;
    if (rem[p] > 0) rem[p]--;
    s++;
  }
  if (!h_out) { CK(cudaMemcpy(g_mode_of_sm, h, (size_t)g_numSm * sizeof(int), cudaMemcpyHostToDevice)); free(h); }
}

static void setup_interleave(const int* seq, int O, int grid) {
  g_cur_bind = 0; g_O = O;
  fill_assign(grid, seq);
  int* h = (int*)malloc((size_t)grid * sizeof(int));
  CK(cudaMemcpy(h, g_assign, (size_t)grid * sizeof(int), cudaMemcpyDeviceToHost));
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) { tot += seq[p]; g_nblk[p] = 0; }
  for (int b = 0; b < grid; ++b) g_nblk[h[b]]++;
  for (int p = 0; p < NPAT; ++p)
    g_W[p] = (tot > 0) ? (long long)((double)seq[p] / (double)tot * (double)TOT_U4) : 0;
  long long need; compute_layout(O, &need);
  ensure(need, grid);
  long long rank[NPAT] = {0, 0, 0, 0};
  long long* bbase = (long long*)malloc((size_t)grid * sizeof(long long));
  long long* bbp = (long long*)malloc((size_t)grid * sizeof(long long));
  for (int b = 0; b < grid; ++b) {
    int p = h[b];
    bbase[b] = g_basep[p] + rank[p] * g_BP[p];
    bbp[b] = g_BP[p];
    rank[p]++;
  }
  CK(cudaMemcpy(g_bb_base, bbase, (size_t)grid * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g_bb_bp, bbp, (size_t)grid * sizeof(long long), cudaMemcpyHostToDevice));
  free(bbase); free(bbp); free(h);
}

static void setup_bind(const int* seq, const int* nsm, int O, int grid) {
  g_cur_bind = 1; g_O = O;
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) { tot += seq[p]; g_nblk[p] = nsm[p] * O; }
  for (int p = 0; p < NPAT; ++p)
    g_W[p] = (tot > 0) ? (long long)((double)seq[p] / (double)tot * (double)TOT_U4) : 0;
  long long need; compute_layout(O, &need);
  ensure(need, grid);
  int* hmos = (int*)malloc((size_t)g_numSm * sizeof(int));
  fill_mode_of_sm(nsm, hmos);
  CK(cudaMemcpy(g_mode_of_sm, hmos, (size_t)g_numSm * sizeof(int), cudaMemcpyHostToDevice));
  long long* sbase = (long long*)malloc((size_t)g_numSm * sizeof(long long));
  long long* sbp = (long long*)malloc((size_t)g_numSm * sizeof(long long));
  int rank[NPAT] = {0, 0, 0, 0};
  for (int s = 0; s < g_numSm; ++s) {
    int p = hmos[s];
    if (g_nblk[p] > 0) {
      sbase[s] = g_basep[p] + (long long)rank[p] * O * g_BP[p];
      sbp[s] = g_BP[p];
      rank[p]++;
    } else { sbase[s] = 0; sbp[s] = 0; }
  }
  CK(cudaMemcpy(g_sm_base, sbase, (size_t)g_numSm * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g_sm_bp, sbp, (size_t)g_numSm * sizeof(long long), cudaMemcpyHostToDevice));
  free(sbase); free(sbp); free(hmos);
}

struct Res { double ms; double gbs; int unsupported; };

static Res run(int grid, int reps) {
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    launch_hetero(grid);
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
  res.gbs = (double)TOT_U4 * 16.0 / (best * 1e-3) / 1e9;
  res.unsupported = unsupported;
  return res;
}

static const char* PATNAME(int p) {
  switch (p) { case 0: return "coalesced"; case 1: return "sparse8x";
               case 2: return "scalar4B"; case 3: return "l2res16M"; }
  return "?";
}

static double g_bw_solo[NPAT], g_r1[NPAT];

static Res run_solo(int pat, int T, int O, int U) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[pat] = 1;
  int nsm[NPAT] = {0, 0, 0, 0}; nsm[pat] = g_numSm;
  int grid = g_numSm * O;
  g_T = T; g_U = U;
  setup_bind(seq, nsm, O, grid);
  return run(grid, 3);
}

static unsigned long long ref_region(int pat, long long base, long long BP) {
  unsigned long long r = 0;
  if (pat == 0) {
    for (long long i = 0; i < BP; ++i) { long long u = base + i;
      r ^= (unsigned long long)(mix32((uint32_t)u, 0u) ^ mix32((uint32_t)u, 1u) ^
                                mix32((uint32_t)u, 2u) ^ mix32((uint32_t)u, 3u)); }
  } else if (pat == 1) {
    for (long long i = 0; i < BP; ++i) { long long u = base + i * SPARSE_STRIDE;
      r ^= (unsigned long long)(mix32((uint32_t)u, 0u) ^ mix32((uint32_t)u, 1u) ^
                                mix32((uint32_t)u, 2u) ^ mix32((uint32_t)u, 3u)); }
  } else if (pat == 2) {
    long long b32 = base * 4;
    for (long long p = 0; p < BP * 4; ++p)
      r ^= (unsigned long long)mix32((uint32_t)((b32 + p) / 4), (uint32_t)((b32 + p) % 4));
  } else {
    for (long long i = 0; i < BP; ++i) { long long ph = i & (L2N4 - 1);
      r ^= (unsigned long long)(mix32((uint32_t)ph, 0u) ^ mix32((uint32_t)ph, 1u) ^
                                mix32((uint32_t)ph, 2u) ^ mix32((uint32_t)ph, 3u)); }
  }
  return r;
}

static double expected_from(const int* seq, const double* bw) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  double inv = 0.0;
  for (int p = 0; p < NPAT; ++p)
    if (seq[p] > 0) inv += (double)seq[p] / (double)tot / bw[p];
  return inv > 0 ? 1.0 / inv : 0.0;
}

static double expected_additive(const int* seq) { return expected_from(seq, g_bw_solo); }

static void alloc_sm(const int* seq, int mode, int* nsm) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  double w[NPAT], sum = 0;
  for (int p = 0; p < NPAT; ++p) {
    if (!seq[p]) { w[p] = 0; continue; }
    double f = (double)seq[p] / (double)tot;
    w[p] = (mode == 0) ? f : f / (g_r1[p] > 0 ? g_r1[p] : 1e-9);
    sum += w[p];
  }
  int used = 0, big = 0;
  for (int p = 0; p < NPAT; ++p) {
    if (w[p] <= 0) { nsm[p] = 0; continue; }
    nsm[p] = (int)(w[p] / sum * g_numSm + 0.5);
    if (nsm[p] < 1) nsm[p] = 1;
    used += nsm[p];
    if (nsm[p] > nsm[big]) big = p;
  }
  nsm[big] += g_numSm - used;
}

static void fill_all() {
  long long tot = TOT_U4 * SPARSE_STRIDE + TOT_U4 + L2N4 + 1024;
  ensure(tot, MAXGRID);
  unsigned long long* dref = nullptr;
  CK(cudaMalloc(&dref, 8));
  CK(cudaMemset(dref, 0, 8));
  fill_data_kernel<<<(int)((tot + 255) / 256), 256>>>(g_data, tot, dref);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(dref));
}

static Res run_mix(int T, int O, int U, const int* seq, int mapping, int* nsm_out) {
  int grid = g_numSm * O;
  g_T = T; g_O = O; g_U = U;
  int nsm[NPAT] = {0, 0, 0, 0};
  if (mapping == 0) {
    setup_interleave(seq, O, grid);
  } else {
    alloc_sm(seq, mapping == 2 ? 1 : 0, nsm);
    setup_bind(seq, nsm, O, grid);
  }
  if (nsm_out) for (int p = 0; p < NPAT; ++p) nsm_out[p] = nsm[p];
  return run(grid, 3);
}

static void set_solo_col(int T, int O, int U, double* out) {
  for (int p = 0; p < NPAT; ++p) {
    Res r = run_solo(p, T, O, U);
    out[p] = r.gbs;
  }
}

struct Mix { const char* nm; int seq[NPAT]; };

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  printf("########################################################################\n");
  printf("# §80 共驻税的「口径」与「每线程 ILP」：绝对量 vs `meas/E`、`U` 作为独立旋钮（补 §79.6 尾巴，P92-1）\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 1 GiB（uint4 计）；寄存器读 + __maxnreg__(32)\n", g_numSm);
  printf("# 四访存模式：coalesced / sparse8x / scalar4B / l2res16M（与 §71–§79 逐字一致）\n");
  printf("# Q1 口径：mixed 绝对 GB/s 与固定参考归一 taxRef=E_ref/mixed（E_ref 跨列不变）vs 旧 meas/E\n");
  printf("# Q2 ILP：固定 Wsm=1024（T256/O4）扫 U∈{1,2,4,8}；再测 U×Wsm 替代性\n");
  printf("########################################################################\n");

  fill_all();

  Mix mixes[] = {
    {"equal 25x4",          {1, 1, 1, 1}},
    {"50/50 coal+scalar",   {1, 0, 1, 0}},
    {"50/50 coal+sparse",   {1, 1, 0, 0}},
    {"50/50 coal+l2res",    {1, 0, 0, 1}},
    {"50/50 scalar+l2res",  {0, 0, 1, 1}},
  };
  int nmix = (int)(sizeof(mixes) / sizeof(mixes[0]));

  // ===== §A 正确性：U=1/2/4/8，interleave 小图 + bind 全图（T256/O4）逐位对 host 参考 =====
  {
    int NBLK = 16;
    int seq[NPAT] = {4, 4, 4, 4};
    printf("\n===== §A 正确性（interleave 小图 NBLK=%d；bind 全图 T256/O4，U=1/2/4/8）=====\n", NBLK);
    g_T = 256; g_U = 4;
    setup_interleave(seq, 1, NBLK);
    CK(cudaMemset(g_sink, 0, 8));
    launch_hetero(NBLK);
    CK(cudaDeviceSynchronize());
    unsigned long long s_i = 0; CK(cudaMemcpy(&s_i, g_sink, 8, cudaMemcpyDeviceToHost));
    int* h_ass = (int*)malloc((size_t)NBLK * sizeof(int));
    long long* h_b = (long long*)malloc((size_t)NBLK * sizeof(long long));
    long long* h_bp = (long long*)malloc((size_t)NBLK * sizeof(long long));
    CK(cudaMemcpy(h_ass, g_assign, (size_t)NBLK * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_b, g_bb_base, (size_t)NBLK * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_bp, g_bb_bp, (size_t)NBLK * sizeof(long long), cudaMemcpyDeviceToHost));
    unsigned long long ref_i = 0;
    for (int b = 0; b < NBLK; ++b) ref_i ^= ref_region(h_ass[b], h_b[b], h_bp[b]);
    printf("  interleave(U=4): sink=%llu ref=%llu -> %s\n", s_i, ref_i, s_i == ref_i ? "OK" : "WRONG");
    free(h_ass); free(h_b); free(h_bp);

    int Ulist[NU] = {1, 2, 4, 8};
    for (int ui = 0; ui < NU; ++ui) {
      int U = Ulist[ui], T = 256, O = 4;
      g_T = T; g_U = U;
      int nsm[NPAT] = {0, 0, 0, 0}; alloc_sm(seq, 0, nsm);
      int grid = g_numSm * O;
      setup_bind(seq, nsm, O, grid);
      CK(cudaMemset(g_sink, 0, 8));
      launch_hetero(grid);
      if (!g_last_ok) { printf("  bind U=%d T=%4d O=%d (Wsm=%d, grid=%d): UNSUP\n", U, T, O, O * T, grid); continue; }
      CK(cudaDeviceSynchronize());
      unsigned long long s_b = 0; CK(cudaMemcpy(&s_b, g_sink, 8, cudaMemcpyDeviceToHost));
      unsigned long long ref_b = 0;
      for (int p = 0; p < NPAT; ++p)
        for (int k = 0; k < g_nblk[p]; ++k)
          ref_b ^= ref_region(p, g_basep[p] + (long long)k * g_BP[p], g_BP[p]);
      printf("  bind U=%d T=%4d O=%d (Wsm=%4d, grid=%4d): sink=%llu ref=%llu -> %s\n", U, T, O, O * T,
             grid, s_b, ref_b, s_b == ref_b ? "OK" : "WRONG");
    }
  }

  // ===== §B 单模式 baseline 随 U（固定 Wsm=1024 = T256/O4）=====
  printf("\n===== §B 单模式 baseline 随每线程 ILP `U`（固定 Wsm=1024 = T=256/O=4）：逻辑 GB/s =====\n");
  printf("  %-11s", "pattern");
  int Ulist[NU] = {1, 2, 4, 8};
  for (int ui = 0; ui < NU; ++ui) printf(" %10s", (ui == 0 ? "U=1" : ui == 1 ? "U=2" : ui == 2 ? "U=4" : "U=8"));
  printf("\n");
  double soloU[NU][NPAT];
  for (int ui = 0; ui < NU; ++ui) {
    for (int p = 0; p < NPAT; ++p) soloU[ui][p] = run_solo(p, 256, 4, Ulist[ui]).gbs;
  }
  for (int p = 0; p < NPAT; ++p) {
    printf("  %-11s", PATNAME(p));
    for (int ui = 0; ui < NU; ++ui)     printf(" %10.1f", soloU[ui][p]);
    printf("   [U=2/1=%.2f U=4/1=%.2f U=8/1=%.2f]\n",
           soloU[1][p] / soloU[0][p], soloU[2][p] / soloU[0][p], soloU[3][p] / soloU[0][p]);
  }

  // 参考基线：固定 Wsm=1024/T256/O4 @ U=4（soloU[2]，跨列不变）
  double bw_ref[NPAT];
  for (int p = 0; p < NPAT; ++p) bw_ref[p] = soloU[2][p];

  // ===== §C（headline 一，口径）固定 U=4、固定 Wsm=2048 铺全 4 个 (T,O) 拆分：复现 §79.3 的「E 漂移」=====
  printf("\n===== §C（口径）固定 U=4、固定 Wsm=2048 铺全 (T,O) 拆分：mixed 绝对量 vs meas/E vs taxRef=E_ref/mixed =====\n");
  printf("（E_ref 用固定参考列 Wsm=1024/T256/O4 的 solo，跨拆分不变；taxRef 只反映 mixed 绝对量）\n");
  struct Col { int T; int O; const char* nm; };
  Col cols[4] = {{1024, 2, "2048/T1024"}, {512, 4, "2048/T512"},
                 {256, 8, "2048/T256"},  {128, 16, "2048/T128"}};
  int ncol = 4;
  double col_solo[4][NPAT];
  for (int j = 0; j < ncol; ++j) set_solo_col(cols[j].T, cols[j].O, 4, col_solo[j]);

  printf("  %-22s %-12s %10s %10s %8s %10s %8s\n", "mixture", "split", "E_self", "mixed", "meas/E", "E_ref", "taxRef");
  for (int m = 0; m < nmix; ++m) {
    double Eref = expected_from(mixes[m].seq, bw_ref);
    for (int j = 0; j < ncol; ++j) {
      for (int p = 0; p < NPAT; ++p) { g_bw_solo[p] = col_solo[j][p]; g_r1[p] = col_solo[j][p] / (double)g_numSm; }
      double E_self = expected_additive(mixes[m].seq);
      Res rb = run_mix(cols[j].T, cols[j].O, 4, mixes[m].seq, 1, nullptr);
      printf("  %-22s %-12s %10.1f %10.1f %8.3f %10.1f %8.3f\n",
             mixes[m].nm, cols[j].nm, E_self, rb.gbs, rb.gbs / E_self, Eref, Eref / rb.gbs);
    }
    printf("\n");
  }

  // ===== §C2 口径对比：Wsm=2048 内 4 个拆分的 spread——绝对 mixed vs meas/E vs taxRef =====
  printf("\n===== §C2 口径对比（headline）：Wsm=2048 内 4 拆分的 spread——绝对 mixed / meas/E / taxRef =====\n");
  printf("  %-22s %30s %30s %18s\n", "mixture", "mixed_abs(4 splits) spread", "meas/E(4 splits) spread", "taxRef spread");
  for (int m = 0; m < nmix; ++m) {
    double Eref = expected_from(mixes[m].seq, bw_ref);
    double av[4], rt[4], tx[4];
    for (int j = 0; j < ncol; ++j) {
      for (int p = 0; p < NPAT; ++p) { g_bw_solo[p] = col_solo[j][p]; g_r1[p] = col_solo[j][p] / (double)g_numSm; }
      double E_self = expected_additive(mixes[m].seq);
      Res rb = run_mix(cols[j].T, cols[j].O, 4, mixes[m].seq, 1, nullptr);
      av[j] = rb.gbs; rt[j] = rb.gbs / E_self; tx[j] = Eref / rb.gbs;
    }
    double amn = 1e30, amx = -1e30, rmn = 1e30, rmx = -1e30, tmn = 1e30, tmx = -1e30;
    for (int j = 0; j < ncol; ++j) {
      if (av[j] < amn) amn = av[j]; if (av[j] > amx) amx = av[j];
      if (rt[j] < rmn) rmn = rt[j]; if (rt[j] > rmx) rmx = rt[j];
      if (tx[j] < tmn) tmn = tx[j]; if (tx[j] > tmx) tmx = tx[j];
    }
    printf("  %-22s [%6.0f..%6.0f] sp=%.3f     [%.3f..%.3f] sp=%.3f   [%.3f..%.3f] sp=%.3f\n",
           mixes[m].nm, amn, amx, amx / amn, rmn, rmx, rmx / rmn, tmn, tmx, tmx / tmn);
  }

  // ===== §D（headline 二，ILP）固定 Wsm=1024（T256/O4），扫 U=1/2/4/8 =====
  printf("\n===== §D（ILP）固定 Wsm=1024（T=256/O=4）扫 U∈{1,2,4,8}：mixed 绝对 GB/s 与 meas/E =====\n");
  printf("  %-22s", "mixture");
  for (int ui = 0; ui < NU; ++ui) printf(" %16s", (ui == 0 ? "U=1" : ui == 1 ? "U=2" : ui == 2 ? "U=4" : "U=8"));
  printf("\n");
  for (int m = 0; m < nmix; ++m) {
    printf("  %-22s", mixes[m].nm);
    for (int ui = 0; ui < NU; ++ui) {
      // 该 U 的 solo（固定 Wsm=1024/T256/O4）
      for (int p = 0; p < NPAT; ++p) { g_bw_solo[p] = soloU[ui][p]; g_r1[p] = soloU[ui][p] / (double)g_numSm; }
      double E_self = expected_additive(mixes[m].seq);
      Res rb = run_mix(256, 4, Ulist[ui], mixes[m].seq, 1, nullptr);
      char tmp[40]; snprintf(tmp, sizeof(tmp), "%7.1f %.3f", rb.gbs, rb.gbs / E_self);
      printf(" %16s", tmp);
    }
    printf("\n");
  }

  // ===== §E U×Wsm 替代性：同一混合 coal+scalar，比较 (U,Wsm) 组合 =====
  printf("\n===== §E U×Wsm 替代性（混合 coal+scalar，bind）：每 SM 在飞 load 量 = Wsm·U =====\n");
  struct Pair { int T; int O; int U; };
  Pair pairs[] = {{256, 2, 1}, {256, 4, 1}, {256, 2, 2}, {256, 4, 2}, {256, 8, 2},
                  {256, 2, 4}, {256, 4, 4}, {256, 8, 4}, {256, 2, 8}, {256, 4, 8}};
  int npair = (int)(sizeof(pairs) / sizeof(pairs[0]));
  int seqCS[NPAT] = {1, 0, 1, 0};
  for (int k = 0; k < npair; ++k) {
    int T = pairs[k].T, O = pairs[k].O, U = pairs[k].U;
    for (int p = 0; p < NPAT; ++p) {
      double s = 0;
      if (seqCS[p]) s = run_solo(p, T, O, U).gbs;
      g_bw_solo[p] = s; g_r1[p] = s / (double)g_numSm;
    }
    double E = expected_additive(seqCS);
    Res rb = run_mix(T, O, U, seqCS, 1, nullptr);
    printf("  T=%4d O=%d U=%d (Wsm=%4d, Wsm·U=%5d): mixed=%7.1f E=%7.1f meas/E=%.3f\n",
           T, O, U, O * T, O * T * U, rb.gbs, E, rb.gbs / E);
  }

  // ===== §F 结论 =====
  printf("\n===== §F 结论 =====\n");
  printf("  - §C/§C2: coal+scalar 的绝对 mixed 随 Wsm 单调升、meas/E 与 taxRef 同向；\n");
  printf("        scalar+l2res/equal4 若绝对 mixed 跨 Wsm 小变而 meas/E 大变 => spread 是 E 漂移（口径问题）。\n");
  printf("  - §D: 若 coal+scalar 的税随 U 收回 => 税认「每 SM 在飞 load 量 Wsm·U」；若不动 => 只认 warp 数 Wsm。\n");
  printf("  - §E: 若同 Wsm·U 的 (U,Wsm) 组合塌成同一点 => 二者是替代品（对上 §60/§68/§70）。\n");
  printf("\n########## §80 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 7) { printf("usage: run <T> <O> <U> <map> <r0,r1,r2,r3>\n"); return 1; }
    g_T = atoi(argv[2]);
    int O = atoi(argv[3]);
    g_U = atoi(argv[4]);
    int map = atoi(argv[5]);
    int seq[NPAT] = {0, 0, 0, 0};
    if (sscanf(argv[6], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) < 4) { printf("bad seq\n"); return 1; }
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    for (int p = 0; p < NPAT; ++p) {
      if (!seq[p]) continue;
      Res s = run_solo(p, g_T, O, g_U);
      g_bw_solo[p] = s.gbs; g_r1[p] = s.gbs / (double)g_numSm;
    }
    int nsm[NPAT];
    Res r = run_mix(g_T, O, g_U, seq, map, nsm);
    printf("  T=%d O=%d U=%d map=%d seq=%d,%d,%d,%d grid=%d -> %s ms=%.4f GB/s=%.1f\n",
           g_T, O, g_U, map, seq[0], seq[1], seq[2], seq[3], g_numSm * O,
           r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs);
    return 0;
  }
  printf("usage: hetero_sm_tax_ilp {bench|run <T> <O> <U> <map> <r0,r1,r2,r3>}\n");
  return 1;
}
