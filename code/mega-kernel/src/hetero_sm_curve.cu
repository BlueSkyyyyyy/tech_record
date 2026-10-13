// ============================================================================
// P86-1 (§74)：补掉 P85-1（§73）**§73.8 明文尾巴**（原话）：
//   「`bind_bal` 用**常数**每 SM 速率模型（`r1_p`），未用「SM 数→速率」曲线，故其差**部分**是模型
//    粗糙而非分区本身；未做按曲线最优配平」。
//
// 背景（接 §71 / §72 / §73）：
//   - §73 把 §72 的「按 block 交错」共驻换成「**按 SM 角色绑定**」（每 SM 只跑一种模式、工作份额不变），
//     量出：**共驻税与 `l2res` 超加和是 SM 间/全局现象**（`bind ≈ interleave`，差 ≤1.1%），
//     且 §72.8 的两条缓解方向（聚发射受限 task / 稀疏 SM 隔离）都收不回税。
//   - 但 §73 还留了一条**分区自身的**问题：**SM 分配怎么给**。它只对照了两种朴素分配——
//       naive     `bind`     ：SM 数 ∝ 逻辑份额 f_p（均分）；
//       balanced  `bind_bal` ：SM 数 ∝ f_p / r1_p（`r1_p` = §B 测得**每 SM 单独速率**）。
//     实测 `bind_bal` **反而更差**（`equal4` 0.753、`coal+l2res` 0.854）——因为 `sparse`/`l2res`
//     **低 SM 数即饱和 HBM/L2**、每 SM 速率对 SM 数**非线性**，「常数 `1/r1` 斜率」前提不成立。
//   - §73.8 据此点名：**「分区角色绑定要可用，必须按『SM 数 → 实际速率』曲线配平」**，但没做。
//
// 本节补这一刀：把 §73 的**常数速率模型**换成**实测速率曲线 + 贪心 makespan 配平**（`bind_opt`）。
//   问：① 按曲线配平能否收回 §73 里 `bind_bal` 的**分区自伤**（0.75–0.85）？→ 期望能（那部分是模型
//       粗糙）；② 能否顺带收回 **`coal+scalar` 的 ~10% 全局发射税**？→ 期望不能（§73 已证它是全局
//       现象、与 SM 归属无关）——用来划清「分区可补的（负载均衡）」与「分区补不了的（全局争用）」。
//
// 机制：三种绑定映射的工作/区分配逻辑**完全复用 §73**（`hetero_sm_bind.cu`），唯一新增是
//   **`bind_opt`（map=3）**：
//     (1) §B 先对每个模式扫一串 SM 数 `n ∈ {1,2,3,4,6,8,12,16,24,32,48,64,96,132}`，测**分区速率**
//         `rate_p(n)`（逻辑 GB/s；O=1、每 SM 一 block 独占区、走 `setup_interleave` 的按 block 指派，
//         不依赖 `%smid` 放置），存表 + **线性插值**；
//     (2) 分配 = **贪心 makespan 最小化**：给定工作 `W_p = f_p·1GiB`，`T_p(n) = W_p / rate_p(n)`。
//         从每活跃模式 `n_p=1` 起，**每轮把下一个 SM 给当前 `T_p(n_p)` 最大的那个模式**（对凹速率曲线
//         的「给最堵的队列」），直到 `Σn_p = numSM`。这天然让 `sparse`/`l2res` 在饱和后不再吃 SM。
//
// 四种模式与逻辑字节定义、载体（寄存器读 + U=4 ILP、`__maxnreg__(32)`）、三种既有映射**完全复用
//   §71/§72/§73**。**唯一新变量 = SM 分配器（常数速率 vs 实测曲线）。**
//
// 用法：./hetero_sm_curve bench
//       ./hetero_sm_curve run <T> <O> <map> <reps> <r0,r1,r2,r3>   # map: 0=inter,1=bind,2=bind_bal,3=bind_opt
// 目标卡 H100 / sm90a。安全：脚本带 timeout；纯读微基准，无环协议 / 无死锁。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
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
#define SPARSE_STRIDE 8            // m1：每 8 个 uint4 读 1 个 → 物理 8×（扇区 4×）
#define L2N4 (1 << 20)             // m3：常驻 16 MiB = 1M uint4（< 50 MB L2）
#define NPAT 4
#define MAXGRID 4096
#define NPN 14                     // 速率曲线采样点数

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 四模式读函数（与 §71/§72/§73 逐字一致）----
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

// ---- 统一 kernel（与 §73 一致）----
__global__ void __maxnreg__(32) pat_hetero(const void* __restrict__ raw, long long aux, int bind,
                                            const int* __restrict__ assign,
                                            const int* __restrict__ mode_of_sm,
                                            const long long* __restrict__ bb_base,
                                            const long long* __restrict__ bb_bp,
                                            const long long* __restrict__ sm_base,
                                            const long long* __restrict__ sm_bp,
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
    base = sm_base[smid];
    BP = sm_bp[smid];
  } else {
    pat = assign[blockIdx.x];
    base = bb_base[blockIdx.x];
    BP = bb_bp[blockIdx.x];
  }
  unsigned long long part = 0ull;
  switch (pat) {
    case 0: part = read_coal<U_ILP>((const uint4*)raw + base, BP, tid, T); break;
    case 1: part = read_sparse<U_ILP>((const uint4*)raw + base, BP, tid, T); break;
    case 2: part = read_scalar<U_ILP>((const uint32_t*)raw + base * 4, BP * 4, tid, T); break;
    default: part = read_l2res<U_ILP>((const uint4*)raw, BP, aux, tid, T); break;
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
static long long* g_bb_base = nullptr;
static long long* g_bb_bp = nullptr;
static long long* g_sm_base = nullptr;
static long long* g_sm_bp = nullptr;
static unsigned int* g_blk_smid = nullptr;
static long long g_data_cap = 0;
static int g_assign_cap = 0, g_smcap = 0, g_gridcap = 0;
static int g_numSm = 0;
static int g_T = 256;

static long long g_W[NPAT], g_BP[NPAT], g_basep[NPAT], g_phys[NPAT];
static int g_nblk[NPAT];

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
  pat_hetero<<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_assign, g_mode_of_sm,
                            g_bb_base, g_bb_bp, g_sm_base, g_sm_bp, g_blk_smid, g_sink);
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
  g_cur_bind = 0;
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
  g_cur_bind = 1;
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

static Res run_solo(int pat, int T, int O) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[pat] = 1;
  int nsm[NPAT] = {0, 0, 0, 0}; nsm[pat] = g_numSm;
  int grid = g_numSm * O;
  g_T = T;
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

static double expected_additive(const int* seq) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  double inv = 0.0;
  for (int p = 0; p < NPAT; ++p)
    if (seq[p] > 0) inv += (double)seq[p] / (double)tot / g_bw_solo[p];
  return inv > 0 ? 1.0 / inv : 0.0;
}

// 常数斜率分配：mode 0 = naive（∝ 份额）；mode 1 = balanced（∝ 份额/每SM速率）——= §73
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

// ---- §74 新增：实测速率曲线 rate_p(n) + 贪心 makespan 配平 ----
static int g_npn = 0;
static int g_raten[NPN];
static double g_rate[NPAT][NPN];

static double rate_of(int p, int n) {
  if (n <= 0) return 0.0;
  if (n <= g_raten[0]) return g_rate[p][0] * (double)n / (double)g_raten[0];   // 低端线性外推
  for (int i = 1; i < g_npn; ++i) {
    if (n <= g_raten[i]) {
      double x0 = g_raten[i - 1], x1 = g_raten[i];
      double y0 = g_rate[p][i - 1], y1 = g_rate[p][i];
      return y0 + (y1 - y0) * (n - x0) / (x1 - x0);
    }
  }
  return g_rate[p][g_npn - 1];
}

// 测一个模式在 n 个 SM（n 个 block，O=1）上的分区速率（逻辑 GB/s）
static double measure_rate(int pat, int n, int T_O1) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[pat] = 1;
  g_T = T_O1;
  setup_interleave(seq, 1, n);       // bind=0，按 block 指派模式，不依赖 %smid 放置
  int grid = n;
  Res r = run(grid, 3);
  if (r.unsupported) return 0.0;
  // 该分区实际处理 g_nblk[pat]*g_BP[pat] 个 uint4
  double act = (double)g_nblk[pat] * (double)g_BP[pat];
  return act * 16.0 / (r.ms * 1e-3) / 1e9;
}

static void build_rate_curve(int T_O1) {
  static const int samp[NPN] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 132};
  g_npn = NPN;
  for (int i = 0; i < NPN; ++i) g_raten[i] = (samp[i] <= g_numSm) ? samp[i] : g_numSm;
  // 去重（若 numSM < 132）
  int w = 0;
  for (int i = 0; i < NPN; ++i) {
    if (i == 0 || g_raten[i] != g_raten[i - 1]) g_raten[w++] = g_raten[i];
  }
  g_npn = w;
  if (g_raten[g_npn - 1] != g_numSm) g_raten[g_npn - 1] = g_numSm;
  for (int p = 0; p < NPAT; ++p)
    for (int i = 0; i < g_npn; ++i)
      g_rate[p][i] = measure_rate(p, g_raten[i], T_O1);
}

// 贪心 makespan 最小化：每轮把下一个 SM 给当前 T_p(n_p) 最大的活跃模式
static void alloc_opt(const int* seq, int* nsm) {
  long long tot = 0;
  for (int p = 0; p < NPAT; ++p) tot += seq[p];
  int active[NPAT], na = 0;
  for (int p = 0; p < NPAT; ++p) { nsm[p] = 0; if (seq[p] > 0) active[na++] = p; }
  if (na == 0) return;
  double Wb[NPAT];
  for (int p = 0; p < NPAT; ++p)
    Wb[p] = (double)seq[p] / (double)tot * (double)TOT_U4 * 16.0;
  for (int i = 0; i < na; ++i) nsm[active[i]] = 1;
  int used = na;
  while (used < g_numSm) {
    int best = -1; double bestT = -1.0;
    for (int i = 0; i < na; ++i) {
      int p = active[i];
      double r = rate_of(p, nsm[p]);
      if (r <= 0) r = 1e-9;
      double t = Wb[p] / r;
      if (t > bestT) { bestT = t; best = p; }
    }
    if (best < 0) break;
    nsm[best]++;
    used++;
  }
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

static Res run_mix(int T, int O, const int* seq, int mapping, int* nsm_out) {
  int grid = g_numSm * O;
  g_T = T;
  int nsm[NPAT] = {0, 0, 0, 0};
  if (mapping == 0) {
    setup_interleave(seq, O, grid);
  } else {
    if (mapping == 2) alloc_sm(seq, 1, nsm);
    else if (mapping == 3) alloc_opt(seq, nsm);
    else alloc_sm(seq, 0, nsm);
    setup_bind(seq, nsm, O, grid);
  }
  if (nsm_out) for (int p = 0; p < NPAT; ++p) nsm_out[p] = nsm[p];
  return run(grid, 3);
}

static const char* MAPNAME(int m) {
  switch (m) { case 0: return "interleave"; case 1: return "bind"; case 2: return "bind_bal";
               case 3: return "bind_opt"; }
  return "?";
}

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  printf("########################################################################\n");
  printf("# §74 SM 分配器：常数速率(bind/bind_bal) vs **实测『SM 数→速率』曲线**配平(bind_opt)\n");
  printf("#   补 §73.8 尾巴「bind_bal 用常数每SM速率模型，未按曲线最优配平」\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 1 GiB（uint4 计）；reg U=%d；__maxnreg__(32)\n",
         g_numSm, U_ILP);
  printf("#   问①按曲线配平能否收回 bind_bal 的分区自伤？ 问②能否收回 coal+scalar 的全局发射税？\n");
  printf("########################################################################\n");

  fill_all();

  int mainO = 1, mainT = 1024;

  // ===== §A 正确性 =====
  {
    int NBLK = 16;
    int seq[NPAT] = {4, 4, 4, 4};
    g_T = 256;
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
    printf("\n===== §A 正确性（NBLK=%d）=====\n", NBLK);
    printf("  interleave : sink=%llu ref=%llu -> %s\n", s_i, ref_i, s_i == ref_i ? "OK" : "WRONG");
    free(h_ass); free(h_b); free(h_bp);
  }

  // ===== §B solo baseline + 速率曲线 =====
  printf("\n===== §B 单模式 baseline（T=1024,O=1，全 SM 同模式）：逻辑 GB/s =====\n");
  for (int p = 0; p < NPAT; ++p) {
    Res r = run_solo(p, mainT, mainO);
    g_bw_solo[p] = r.gbs; g_r1[p] = r.gbs / (double)g_numSm;
    printf("  %-11s %8.1f GB/s   (每 SM %6.2f GB/s)\n", PATNAME(p), r.gbs, g_r1[p]);
  }

  printf("\n===== §B2 实测『SM 数→分区速率』曲线 rate_p(n)（O=1，逻辑 GB/s）=====\n");
  build_rate_curve(mainT);
  printf("  %-6s", "n");
  for (int p = 0; p < NPAT; ++p) printf(" %11s", PATNAME(p));
  printf("  | 说明\n");
  for (int i = 0; i < g_npn; ++i) {
    printf("  %-6d", g_raten[i]);
    for (int p = 0; p < NPAT; ++p) printf(" %11.1f", g_rate[p][i]);
    printf("\n");
  }
  // 每 SM 边际速率（是否随 n 下降 = 是否饱和）
  printf("  [每 SM 速率 rate_p(n)/n（GB/s/SM），随 n 下降说明该模式在低 SM 数就饱和]\n");
  for (int i = 0; i < g_npn; ++i) {
    printf("  %-6d", g_raten[i]);
    for (int p = 0; p < NPAT; ++p) printf(" %11.2f", g_rate[p][i] / g_raten[i]);
    printf("\n");
  }

  // ===== §C 四映射对照 =====
  printf("\n===== §C 混合四映射对照（T=1024,O=1）：measured GB/s 与 meas/E =====\n");
  printf("  %-22s %9s | %11s %11s %11s %11s\n",
         "mixture", "E(切片)", "interleave", "bind", "bind_bal", "bind_opt");
  struct Mix { const char* nm; int seq[NPAT]; };
  Mix mixes[] = {
    {"equal 25x4",          {1, 1, 1, 1}},
    {"50/50 coal+scalar",   {1, 0, 1, 0}},
    {"50/50 coal+sparse",   {1, 1, 0, 0}},
    {"50/50 coal+l2res",    {1, 0, 0, 1}},
    {"50/50 scalar+l2res",  {0, 0, 1, 1}},
  };
  int nmix = (int)(sizeof(mixes) / sizeof(mixes[0]));
  for (int m = 0; m < nmix; ++m) {
    double E = expected_additive(mixes[m].seq);
    printf("  %-22s %9.1f |", mixes[m].nm, E);
    for (int map = 0; map < 4; ++map) {
      Res r = run_mix(mainT, mainO, mixes[m].seq, map, nullptr);
      printf(" %6.1f %5.3f", r.gbs, r.gbs / E);
    }
    printf("\n");
  }

  // ===== §D SM 分配明细（4 映射）=====
  printf("\n===== §D 角色绑定 SM 分配明细（numSM=%d）=====\n", g_numSm);
  for (int m = 0; m < nmix; ++m) {
    for (int map = 1; map <= 3; ++map) {
      int nsm[NPAT];
      if (map == 1) alloc_sm(mixes[m].seq, 0, nsm);
      else if (map == 2) alloc_sm(mixes[m].seq, 1, nsm);
      else alloc_opt(mixes[m].seq, nsm);
      printf("  %-22s %-9s nsm=[%3d,%3d,%3d,%3d]\n", mixes[m].nm, MAPNAME(map),
             nsm[0], nsm[1], nsm[2], nsm[3]);
    }
  }

  // ===== §E coal+scalar：扫 scalar 的 SM 份额，标注 bind_opt 选点 =====
  printf("\n===== §E coal+scalar 角色绑定：扫 scalar 的 SM 份额（bind 固定份额）=====\n");
  printf("  %-8s %12s %14s %8s\n", "scSM", "measured", "E(实际SM份额)", "meas/E");
  for (int sc = 0; sc <= 132; sc += 12) {
    int nsm[NPAT] = {0, 0, 0, 0};
    nsm[2] = (sc > g_numSm) ? g_numSm : sc;
    nsm[0] = g_numSm - nsm[2];
    if (nsm[0] == 0 && nsm[2] == 0) continue;
    int seq[NPAT] = {1, 0, 1, 0};
    g_T = mainT;
    setup_bind(seq, nsm, mainO, g_numSm * mainO);
    Res r = run(g_numSm * mainO, 3);
    double act_u4 = 0.0;
    for (int p = 0; p < NPAT; ++p) act_u4 += (double)g_nblk[p] * g_BP[p];
    double gbs_act = act_u4 * 16.0 / (r.ms * 1e-3) / 1e9;
    double inv = 0.0;
    for (int p = 0; p < NPAT; ++p) {
      double fp = (double)g_nblk[p] * g_BP[p] / (act_u4 > 0 ? act_u4 : 1);
      if (fp > 0) inv += fp / g_bw_solo[p];
    }
    double E = inv > 0 ? 1.0 / inv : 0;
    printf("  %-8d %12.1f %14.1f %8.3f\n", sc, gbs_act, E, E > 0 ? gbs_act / E : 0.0);
  }
  {
    int nsm[NPAT]; int seq[NPAT] = {1, 0, 1, 0};
    alloc_opt(seq, nsm);
    printf("  bind_opt 选点： nsm=[coal=%d, scalar=%d]\n", nsm[0], nsm[2]);
  }

  // ===== §F 小结 =====
  printf("\n===== §F 结论 =====\n");
  printf("  - §C: bind_opt 若明显 > bind_bal => §73 的分区自伤主要是**常数速率模型**造成的，\n");
  printf("        按实测曲线配平可收回；bind_opt 的 coal+scalar 若仍 ~0.90 => 全局发射税分区补不回。\n");
  printf("  - §D: bind_opt 应在 sparse/l2res 上少给 SM（其速率低 SM 数即饱和），把 SM 让给 coal/scalar。\n");
  printf("\n########## §74 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 7) { printf("usage: run <T> <O> <map> <reps> <r0,r1,r2,r3>  (map 0=inter,1=bind,2=bind_bal,3=bind_opt)\n"); return 1; }
    g_T = atoi(argv[2]);
    int O = atoi(argv[3]);
    int map = atoi(argv[4]);
    int seq[NPAT] = {0, 0, 0, 0};
    if (sscanf(argv[6], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) < 4) { printf("bad seq\n"); return 1; }
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    for (int p = 0; p < NPAT; ++p) {
      if (!seq[p]) continue;
      Res s = run_solo(p, g_T, O);
      g_bw_solo[p] = s.gbs; g_r1[p] = s.gbs / (double)g_numSm;
    }
    build_rate_curve(g_T);
    int nsm[NPAT];
    Res r = run_mix(g_T, O, seq, map, nsm);
    printf("  T=%d O=%d map=%d(%s) seq=%d,%d,%d,%d grid=%d -> %s ms=%.4f GB/s=%.1f nsm=[%d,%d,%d,%d]\n",
           g_T, O, map, MAPNAME(map), seq[0], seq[1], seq[2], seq[3], g_numSm * O,
           r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs, nsm[0], nsm[1], nsm[2], nsm[3]);
    return 0;
  }
  printf("usage: hetero_sm_curve {bench|run <T> <O> <map> <reps> <r0,r1,r2,r3>}\n");
  return 1;
}
