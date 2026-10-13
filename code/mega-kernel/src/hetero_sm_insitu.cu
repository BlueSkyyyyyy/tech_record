// ============================================================================
// P87-1 (§75)：补掉 P86-1（§74）**§74.9 明文尾巴**（原话）：
//   「`bind_opt` 的速率曲线是**单跑**测得（本节的核心发现即它不够），**未做 in-situ/迭代不动点**
//    ——这是最直接的下一步」。
//
// 背景（接 §71/§72/§73/§74）：
//   - §73：把共驻从「按 block 交错」换成「按 SM 角色绑定」，量出共驻税/超加和是**全局（SM 间/HBM/L2）**
//     现象、不是 SM 内；且「按 `1/r1` 常数斜率分 SM」反而更差。
//   - §74：把分配器换成**实测『SM 数→速率』曲线 + 贪心 makespan**（`bind_opt`，曲线仍**单跑**测得）。
//     结论「一半成立、一半是新谬误」——**近线性混合**上收回并反超 `interleave`（`coal+scalar`
//     0.909→0.947、`scalar+l2res` 0.969→0.987），但在**含 HBM 饱和模式（`sparse`）**的混合上
//     **比常数斜率还差**（`coal+sparse` 0.936→**0.664**）：单跑曲线把「加 SM 到已饱和模式」的边际
//     价值估高了。
//   - §74.9 点名：**「分配改变争用、争用改变速率，是一个不动点」**，单跑曲线量不到它。
//
// 本节补这一刀：把分配器的速率来源从**单跑（solo）**换成**当前并发下的实测（in-situ）**，两条路线：
//   (a) `bind_insitu`（map=4）：**迭代不动点**——从 §74 的 `bind_opt`（单跑曲线）出发，用**实测整机
//       makespan** 做**局部爬山**（每次把一个 `step` 个 SM 从某模式挪给另一模式，改善>0.5% 才接受，
//       `step` 逐步减半到 1）。它**只用 in-situ（带真实并发争用）的测量**，不需要任何速率模型；
//   (b) **条件 in-situ 速率曲线**（§D）：对两模式混合扫 `n_p`，量**并发下**每个分区的实测速率
//       `rate_p(n | 其它模式占 numSM-n)`，与 §74 的**单跑**曲线 `rate_p(n)` 并排——直接展示
//       「加 SM 到已饱和模式」在并发下边际价值为**负/零**，从而解释单跑曲线为何把 `sparse` 喂到 123 SM。
//
// 四模式、逻辑字节定义、载体（寄存器读 + U=4 ILP、`__maxnreg__(32)`）、既有映射（interleave/bind/
//   bind_bal/bind_opt）**完全复用 §71–§74**（`src/hetero_sm_curve.cu`）。**唯一新变量 = 速率来源**
//   （单跑曲线 → in-situ 实测 / 迭代不动点）。
//
// 用法：./hetero_sm_insitu bench
//       ./hetero_sm_insitu run <T> <O> <map> <seq r0,r1,r2,r3>  # map:0=inter,1=bind,2=bind_bal,3=bind_opt,4=bind_insitu
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
#define NPN 14                     // §74 单跑速率曲线采样点数

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 四模式读函数（与 §71/§72/§73/§74 逐字一致）----
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

// ---- 统一 kernel（与 §73/§74 一致，另加 clock64 计时）----
__global__ void __maxnreg__(32) pat_hetero(const void* __restrict__ raw, long long aux, int bind,
                                            const int* __restrict__ assign,
                                            const int* __restrict__ mode_of_sm,
                                            const long long* __restrict__ bb_base,
                                            const long long* __restrict__ bb_bp,
                                            const long long* __restrict__ sm_base,
                                            const long long* __restrict__ sm_bp,
                                            unsigned int* __restrict__ blk_smid,
                                            unsigned long long* __restrict__ blk_start,
                                            unsigned long long* __restrict__ blk_end,
                                            unsigned long long* sink) {
  const int tid = threadIdx.x;
  const int T = blockDim.x;
  unsigned int smid = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (tid == 0) { blk_smid[blockIdx.x] = smid; blk_start[blockIdx.x] = clock64(); }
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
  __syncthreads();
  if (tid == 0) blk_end[blockIdx.x] = clock64();
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
static unsigned long long* g_blk_start = nullptr;
static unsigned long long* g_blk_end = nullptr;
static long long g_data_cap = 0;
static int g_assign_cap = 0, g_smcap = 0, g_gridcap = 0;
static int g_numSm = 0;
static int g_T = 256;
static int g_clockKHz = 0;

static long long g_W[NPAT], g_BP[NPAT], g_basep[NPAT], g_phys[NPAT];
static int g_nblk[NPAT];
static int g_mode_of_sm_host[4096];   // host 侧镜像（由 setup_bind 填，供 run_timed 归因）
static double g_insitu_gbs = 0.0;     // alloc_insitu 收敛后的实测 GB/s

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
    if (g_blk_start) CK(cudaFree(g_blk_start));
    CK(cudaMalloc(&g_blk_start, (size_t)grid * sizeof(unsigned long long)));
    if (g_blk_end) CK(cudaFree(g_blk_end));
    CK(cudaMalloc(&g_blk_end, (size_t)grid * sizeof(unsigned long long)));
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
                            g_bb_base, g_bb_bp, g_sm_base, g_sm_bp, g_blk_smid,
                            g_blk_start, g_blk_end, g_sink);
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
  for (int s = 0; s < g_numSm && s < 4096; ++s) g_mode_of_sm_host[s] = hmos[s];
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

// 每模式 in-situ 分区时间（cycles）与到达时间；由 run() 填
static double g_tp_cyc[NPAT];        // 每模式分区 span（max_end - min_start）
static double g_rate_insitu[NPAT];   // 每模式 in-situ 逻辑 GB/s

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

// 带 per-mode in-situ 计时的运行（用于 §D 条件曲线；reps 取 1 以避免多次覆盖）
static Res run_timed(int grid, int reps) {
  Res r = run(grid, reps);
  if (r.unsupported) return r;
  int nb = grid;
  unsigned int* hsm = (unsigned int*)malloc((size_t)nb * sizeof(unsigned int));
  unsigned long long* hs = (unsigned long long*)malloc((size_t)nb * sizeof(unsigned long long));
  unsigned long long* he = (unsigned long long*)malloc((size_t)nb * sizeof(unsigned long long));
  CK(cudaMemcpy(hsm, g_blk_smid, (size_t)nb * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hs, g_blk_start, (size_t)nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(he, g_blk_end, (size_t)nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  double span[NPAT]; long long cnt[NPAT];
  for (int p = 0; p < NPAT; ++p) { span[p] = 0.0; cnt[p] = 0; }
  for (int b = 0; b < nb; ++b) {
    int p = g_cur_bind ? g_mode_of_sm_host[hsm[b]] : -1;   // 交错模式不做 per-mode 计时
    if (p < 0 || p >= NPAT) continue;
    // 用每条 block 自己的 (end-start)：`clock64` 逐 SM 频/相位可能不同，跨 SM 相减会串台
    double sp = (double)he[b] - (double)hs[b];
    if (sp > span[p]) span[p] = sp;
    cnt[p]++;
  }
  for (int p = 0; p < NPAT; ++p) {
    if (cnt[p] > 0) {
      g_tp_cyc[p] = span[p];
      double sec = g_tp_cyc[p] / ((double)g_clockKHz * 1000.0);
      double bytes = (double)g_nblk[p] * (double)g_BP[p] * 16.0;
      g_rate_insitu[p] = (sec > 0) ? bytes / sec / 1e9 : 0.0;
    } else { g_tp_cyc[p] = 0; g_rate_insitu[p] = 0.0; }
  }
  free(hsm); free(hs); free(he);
  return r;
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

// 常数斜率分配（=§73）：mode 0 = naive（∝ 份额）；mode 1 = balanced（∝ 份额/每SM速率）
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

// ---- §74 单跑速率曲线 rate_p(n) + 贪心 makespan ----
static int g_npn = 0;
static int g_raten[NPN];
static double g_rate[NPAT][NPN];

static double rate_of(int p, int n) {
  if (n <= 0) return 0.0;
  if (n <= g_raten[0]) return g_rate[p][0] * (double)n / (double)g_raten[0];
  for (int i = 1; i < g_npn; ++i) {
    if (n <= g_raten[i]) {
      double x0 = g_raten[i - 1], x1 = g_raten[i];
      double y0 = g_rate[p][i - 1], y1 = g_rate[p][i];
      return y0 + (y1 - y0) * (n - x0) / (x1 - x0);
    }
  }
  return g_rate[p][g_npn - 1];
}

static double measure_rate_solo(int pat, int n, int T_O1) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[pat] = 1;
  g_T = T_O1;
  setup_interleave(seq, 1, n);
  int grid = n;
  Res r = run(grid, 3);
  if (r.unsupported) return 0.0;
  double act = (double)g_nblk[pat] * (double)g_BP[pat];
  return act * 16.0 / (r.ms * 1e-3) / 1e9;
}

static void build_rate_curve(int T_O1) {
  static const int samp[NPN] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 132};
  g_npn = NPN;
  for (int i = 0; i < NPN; ++i) g_raten[i] = (samp[i] <= g_numSm) ? samp[i] : g_numSm;
  int w = 0;
  for (int i = 0; i < NPN; ++i)
    if (i == 0 || g_raten[i] != g_raten[i - 1]) g_raten[w++] = g_raten[i];
  g_npn = w;
  if (g_raten[g_npn - 1] != g_numSm) g_raten[g_npn - 1] = g_numSm;
  for (int p = 0; p < NPAT; ++p)
    for (int i = 0; i < g_npn; ++i)
      g_rate[p][i] = measure_rate_solo(p, g_raten[i], T_O1);
}

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

// ---- §75 新增：in-situ 迭代不动点分配（爬山，仅用实测 makespan）----
static double eval_alloc(const int* seq, const int* nsm, int O, int T, int reps) {
  g_T = T;
  setup_bind(seq, nsm, O, g_numSm * O);
  Res r = run(g_numSm * O, reps);
  return r.unsupported ? 0.0 : r.gbs;
}

static void alloc_insitu(const int* seq, int* nsm_final, int O, int T,
                         int* out_evals, double* out_gbs, int verbose) {
  int nsm[NPAT];
  alloc_opt(seq, nsm);                          // 从 §74 单跑曲线的 bind_opt 出发
  double best = eval_alloc(seq, nsm, O, T, 5);  // in-situ 实测
  int evals = 1;
  int active[NPAT], na = 0;
  for (int p = 0; p < NPAT; ++p) if (seq[p] > 0) active[na++] = p;
  if (verbose)
    printf("    iter0(solo start) nsm=[%d,%d,%d,%d] gbs=%.1f\n",
           nsm[0], nsm[1], nsm[2], nsm[3], best);
  for (int step = 16; step >= 1; step >>= 1) {
    int improved = 1;
    while (improved) {
      improved = 0;
      for (int di = 0; di < na && !improved; ++di) {
        int d = active[di];
        int move = step;
        if (nsm[d] - move < 1) continue;
        for (int ri = 0; ri < na && !improved; ++ri) {
          int r = active[ri];
          if (r == d) continue;
          int cand[NPAT];
          memcpy(cand, nsm, sizeof(cand));
          cand[d] -= move; cand[r] += move;
          double g = eval_alloc(seq, cand, O, T, 5);
          evals++;
          if (g > best * 1.005) {                 // 要求 >0.5% 改善（抗噪）
            memcpy(nsm, cand, sizeof(nsm));
            best = g; improved = 1;
            if (verbose)
              printf("    step=%2d move %d:%-2d->%-2d nsm=[%d,%d,%d,%d] gbs=%.1f\n",
                     step, d, move, r, nsm[0], nsm[1], nsm[2], nsm[3], best);
          }
        }
      }
    }
  }
  memcpy(nsm_final, nsm, sizeof(nsm));
  *out_evals = evals;
  *out_gbs = best;
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

static Res run_mix(int T, int O, const int* seq, int mapping, int* nsm_out,
                   int* out_evals, int verbose) {
  int grid = g_numSm * O;
  g_T = T;
  int nsm[NPAT] = {0, 0, 0, 0};
  if (mapping == 0) {
    setup_interleave(seq, O, grid);
  } else if (mapping == 4) {
    alloc_insitu(seq, nsm, O, T, out_evals, &g_insitu_gbs, verbose);
    setup_bind(seq, nsm, O, grid);
  } else {
    if (mapping == 2) alloc_sm(seq, 1, nsm);
    else if (mapping == 3) alloc_opt(seq, nsm);
    else alloc_sm(seq, 0, nsm);
    setup_bind(seq, nsm, O, grid);
  }
  if (nsm_out) for (int p = 0; p < NPAT; ++p) nsm_out[p] = nsm[p];
  return run(grid, 5);
}

static const char* MAPNAME(int m) {
  switch (m) { case 0: return "interleave"; case 1: return "bind"; case 2: return "bind_bal";
               case 3: return "bind_opt"; case 4: return "bind_insitu"; }
  return "?";
}

// 条件 in-situ 速率曲线：两模式 (p,q) 扫 n_p，量并发下各自实测速率（§D）
static void insitu_curve_two(int p, int q, int O, int T, int step) {
  printf("\n  [条件 in-situ 曲线] %s(n) | %s(%d-n) 并发（O=%d,T=%d）；n 扫 %d\n",
         PATNAME(p), PATNAME(q), g_numSm, O, T, step);
  printf("  %-5s %11s | %13s %13s | %13s %13s\n",
         "n_p", "overall", "rate_p(in-situ)", "rate_q(in-situ)",
         "solo_p(n_p)", "solo_q(numSM-n)");
  int seq[NPAT] = {0, 0, 0, 0}; seq[p] = 1; seq[q] = 1;
  g_T = T;
  for (int np = 0; np <= g_numSm; np += step) {
    int nsm[NPAT] = {0, 0, 0, 0};
    nsm[p] = np; nsm[q] = g_numSm - np;
    setup_bind(seq, nsm, O, g_numSm * O);
    fill_mode_of_sm(nsm, g_mode_of_sm_host);
    Res r = run_timed(g_numSm * O, 1);
    // overall 用**实际完成**的逻辑字节（若某模式 0 个 SM 则不计其份额），否则会被常数 TOT 高估
    double act = 0.0;
    for (int pp = 0; pp < NPAT; ++pp) act += (double)g_nblk[pp] * (double)g_BP[pp] * 16.0;
    double overall = act / (r.ms * 1e-3) / 1e9;
    double rsolo_p = (np > 0) ? rate_of(p, np) : 0.0;
    double rsolo_q = (g_numSm - np > 0) ? rate_of(q, g_numSm - np) : 0.0;
    printf("  %-5d %11.1f | %13.1f %13.1f | %13.1f %13.1f\n",
           np, overall,
           np > 0 ? g_rate_insitu[p] : 0.0,
           (g_numSm - np) > 0 ? g_rate_insitu[q] : 0.0,
           rsolo_p, rsolo_q);
  }
}

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&g_clockKHz, cudaDevAttrClockRate, 0));
  printf("########################################################################\n");
  printf("# §75 SM 分配器：**单跑**速率曲线(bind_opt) vs **in-situ 迭代不动点**(bind_insitu)\n");
  printf("#   补 §74.9 尾巴「bind_opt 曲线是单跑测得、未做 in-situ/迭代不动点」\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；clock=%.2f MHz；逻辑总读 1 GiB（uint4 计）\n",
         g_numSm, g_clockKHz / 1000.0);
  printf("#   reg U=%d；__maxnreg__(32)；问：in-situ 能否收回 coal+sparse 的 0.664 崩盘？\n", U_ILP);
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

  // ===== §B solo baseline + 单跑速率曲线（复现 §74）=====
  printf("\n===== §B 单模式 baseline（T=1024,O=1）：逻辑 GB/s =====\n");
  for (int p = 0; p < NPAT; ++p) {
    Res r = run_solo(p, mainT, mainO);
    g_bw_solo[p] = r.gbs; g_r1[p] = r.gbs / (double)g_numSm;
    printf("  %-11s %8.1f GB/s   (每 SM %6.2f GB/s)\n", PATNAME(p), r.gbs, g_r1[p]);
  }
  printf("\n===== §B2 单跑『SM 数→速率』曲线 rate_p(n)（=§74，用于 bind_opt）=====\n");
  build_rate_curve(mainT);
  printf("  %-6s", "n");
  for (int p = 0; p < NPAT; ++p) printf(" %11s", PATNAME(p));
  printf("\n");
  for (int i = 0; i < g_npn; ++i) {
    printf("  %-6d", g_raten[i]);
    for (int p = 0; p < NPAT; ++p) printf(" %11.1f", g_rate[p][i]);
    printf("\n");
  }

  struct Mix { const char* nm; int seq[NPAT]; };
  Mix mixes[] = {
    {"equal 25x4",          {1, 1, 1, 1}},
    {"50/50 coal+scalar",   {1, 0, 1, 0}},
    {"50/50 coal+sparse",   {1, 1, 0, 0}},
    {"50/50 coal+l2res",    {1, 0, 0, 1}},
    {"50/50 scalar+l2res",  {0, 0, 1, 1}},
  };
  int nmix = (int)(sizeof(mixes) / sizeof(mixes[0]));

  // ===== §C 四映射 + in-situ 迭代不动点（headline）=====
  printf("\n===== §C 五映射对照（T=1024,O=1）：measured GB/s 与 meas/E =====\n");
  printf("  %-22s %9s | %11s %11s %11s %11s %11s\n",
         "mixture", "E(切片)", "interleave", "bind", "bind_bal", "bind_opt", "bind_insitu");
  for (int m = 0; m < nmix; ++m) {
    double E = expected_additive(mixes[m].seq);
    printf("  %-22s %9.1f |", mixes[m].nm, E);
    for (int map = 0; map < 5; ++map) {
      int ev = 0;
      Res r = run_mix(mainT, mainO, mixes[m].seq, map, nullptr, &ev, 0);
      printf(" %6.1f %5.3f", r.gbs, r.gbs / E);
    }
    printf("\n");
  }

  // ===== §D 条件 in-situ 速率曲线（解释单跑曲线为何把 sparse 喂到 123）=====
  printf("\n===== §D 条件 in-situ 速率曲线（并发下实测 vs 单跑）=====\n");
  printf("#  `coal+sparse`：单跑曲线说『sparse 再多给 SM 还有用』，in-situ 说不。\n");
  insitu_curve_two(0, 1, mainO, mainT, 16);
  insitu_curve_two(0, 2, mainO, mainT, 16);

  // ===== §E SM 分配明细 =====
  printf("\n===== §E 角色绑定 SM 分配明细（numSM=%d）=====\n", g_numSm);
  for (int m = 0; m < nmix; ++m) {
    for (int map = 1; map <= 4; ++map) {
      int nsm[NPAT];
      int ev = 0; double gb = 0;
      if (map == 1) alloc_sm(mixes[m].seq, 0, nsm);
      else if (map == 2) alloc_sm(mixes[m].seq, 1, nsm);
      else if (map == 3) alloc_opt(mixes[m].seq, nsm);
      else alloc_insitu(mixes[m].seq, nsm, mainO, mainT, &ev, &gb, 0);
      printf("  %-22s %-11s nsm=[%3d,%3d,%3d,%3d]%s\n", mixes[m].nm, MAPNAME(map),
             nsm[0], nsm[1], nsm[2], nsm[3],
             map == 4 ? "  <- in-situ 迭代" : "");
    }
  }

  // ===== §F in-situ 迭代过程（只看崩盘案 coal+sparse 与 equal4）=====
  printf("\n===== §F in-situ 迭代过程（从单跑 bind_opt 出发的爬山轨迹）=====\n");
  {
    int seq[NPAT] = {1, 1, 0, 0};
    int nsm[NPAT]; int ev = 0; double gb = 0;
    printf("  [coal+sparse] 单跑 bind_opt 起点 -> in-situ 爬山：\n");
    alloc_insitu(seq, nsm, mainO, mainT, &ev, &gb, 1);
    printf("    收敛 nsm=[%d,%d,%d,%d] gbs=%.1f  evals=%d\n", nsm[0], nsm[1], nsm[2], nsm[3], gb, ev);
  }
  {
    int seq[NPAT] = {1, 0, 1, 0};
    int nsm[NPAT]; int ev = 0; double gb = 0;
    printf("  [coal+scalar] 单跑 bind_opt 起点 -> in-situ 爬山：\n");
    alloc_insitu(seq, nsm, mainO, mainT, &ev, &gb, 1);
    printf("    收敛 nsm=[%d,%d,%d,%d] gbs=%.1f  evals=%d\n", nsm[0], nsm[1], nsm[2], nsm[3], gb, ev);
  }
  {
    int seq[NPAT] = {1, 1, 1, 1};
    int nsm[NPAT]; int ev = 0; double gb = 0;
    printf("  [equal 25x4] 单跑 bind_opt 起点 -> in-situ 爬山：\n");
    alloc_insitu(seq, nsm, mainO, mainT, &ev, &gb, 1);
    printf("    收敛 nsm=[%d,%d,%d,%d] gbs=%.1f  evals=%d\n", nsm[0], nsm[1], nsm[2], nsm[3], gb, ev);
  }

  // ===== §G 结论 =====
  printf("\n===== §G 结论 =====\n");
  printf("  - §C: bind_insitu 若把 coal+sparse 从 bind_opt 的 0.664 抬回 >1.0 => §74 的崩盘是**单跑曲线**\n");
  printf("        的错、in-situ 测量可收回；coal+scalar 若仍 ~0.95 => 全局发射税依旧不可分区解。\n");
  printf("  - §D: in-situ 曲线里 sparse 的分区速率应在小 n 即压平（单跑曲线看不出，故贪心狂喂 sparse）。\n");
  printf("  - §F: 爬山轨迹应显示 in-situ 把大量 SM 从 sparse 挪给 coal，收敛到 n_sparse~32 附近。\n");
  printf("\n########## §75 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 6) { printf("usage: run <T> <O> <map> <seq r0,r1,r2,r3>  (map 0=inter,1=bind,2=bind_bal,3=bind_opt,4=bind_insitu)\n"); return 1; }
    g_T = atoi(argv[2]);
    int O = atoi(argv[3]);
    int map = atoi(argv[4]);
    int seq[NPAT] = {0, 0, 0, 0};
    if (sscanf(argv[5], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) < 4) { printf("bad seq\n"); return 1; }
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaDeviceGetAttribute(&g_clockKHz, cudaDevAttrClockRate, 0));
    fill_all();
    for (int p = 0; p < NPAT; ++p) {
      if (!seq[p]) continue;
      Res s = run_solo(p, g_T, O);
      g_bw_solo[p] = s.gbs; g_r1[p] = s.gbs / (double)g_numSm;
    }
    build_rate_curve(g_T);
    int nsm[NPAT]; int ev = 0;
    Res r = run_mix(g_T, O, seq, map, nsm, &ev, 1);
    printf("  T=%d O=%d map=%d(%s) seq=%d,%d,%d,%d grid=%d -> %s ms=%.4f GB/s=%.1f nsm=[%d,%d,%d,%d] evals=%d\n",
           g_T, O, map, MAPNAME(map), seq[0], seq[1], seq[2], seq[3], g_numSm * O,
           r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs, nsm[0], nsm[1], nsm[2], nsm[3], ev);
    return 0;
  }
  printf("usage: hetero_sm_insitu {bench|run <T> <O> <map> <seq r0,r1,r2,r3>}\n");
  return 1;
}
