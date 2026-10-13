// ============================================================================
// P88-1 (§76)：补掉 P87-1（§75）**§75.9 头号尾巴**（原话）：
//   「`bind_insitu` 是**局部爬山**（可能停在局部最优，`equal4` 尤甚——本轮两次运行收敛到不同分配
//    `[25,32,30,45]`/`[8,84,20,20]`，因 `l2res16M` 的 L2 驻留对运行序敏感）」。
//
// 背景（接 §71–§75）：§75 把 SM 分配器的速率来源从**单跑曲线**换成**当前并发下的实测 makespan +
//   局部爬山**（`bind_insitu`），收回了 §74 在 `coal+sparse` 上的崩盘（0.664→1.024）。但 §75.9 承认
//   `bind_insitu` 只是**单起点局部爬山**——① 可能停在局部最优；② 结果**不可复现**（`equal4` 两次运行
//   收敛到不同分配，`l2res16M` 的 L2 驻留对运行序敏感）。这是一个「以 in-situ 测量为黑盒的优化器」的
//   **可靠性**问题。
//
// 本节把 §75 的**单起点爬山**升级并做压力测试，四个问题：
//   Q1（可复现性，§C）：同一单起点（§74 单跑 `bind_opt`）跑 `N` 次，收敛分配/吞吐的**离散度**多大？
//   Q2（多起点，§D）：`bind_global`（map=5）=**多起点爬山**（结构化起点：单跑 `bind_opt` / 均分 /
//       每个模式偏置 / 加 `K` 个固定种子随机起点）+ **最优保留**，能否**追平 oracle**、比单起点好多少？
//   Q3（oracle，§E）：对每个 50/50 两模式混合做**穷举 in-situ 扫描** `n_p=1..numSM-1`，得角色绑定族
//       **真最优**，量出单起点/多起点/`bind_opt`/`interleave` 各自的**最优性 gap**。
//   Q4（评价噪声，§F）：把收敛分配**重复评估** `R` 次，量出 in-situ 评价本身的**噪声地板**——它是
//       Q1 不可复现的来源之一。
//
// 四模式、逻辑字节定义、载体（寄存器读 + U=4 ILP、`__maxnreg__(32)`）、既有映射（interleave/bind/
//   bind_bal/bind_opt/bind_insitu）**完全复用 §71–§75**（`src/hetero_sm_insitu.cu`）。
//   **唯一新变量 = 起点集合 + oracle 穷举**（单起点 → 多起点 / 真最优）。
//
// 用法：./hetero_sm_global bench
//       ./hetero_sm_global run <T> <O> <map> <seq r0,r1,r2,r3>  # map:0=inter,1=bind,2=bind_bal,3=bind_opt,4=bind_insitu,5=bind_global
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

// 分配合法性：活跃模式各 >=1、总和 == numSM。总和不符会让评测「丢 work」→ 虚高（§76 开发中踩过的坑）。
static int alloc_valid(const int* seq, const int* nsm) {
  int sum = 0;
  for (int p = 0; p < NPAT; ++p) {
    if (seq[p] > 0 && nsm[p] < 1) return 0;
    sum += nsm[p];
  }
  return sum == g_numSm;
}

// 通用局部爬山：从传入起点 `nsm` 出发、用实测 makespan 逐步挪 SM，收敛就地写回 `nsm`。
// （§75 的 alloc_insitu 主体，抽出来供单起点/多起点共用。）
static void hill_from(const int* seq, int* nsm, int O, int T,
                      int* out_evals, double* out_gbs, int verbose) {
  double best = eval_alloc(seq, nsm, O, T, 5);  // in-situ 实测
  int evals = 1;
  int active[NPAT], na = 0;
  for (int p = 0; p < NPAT; ++p) if (seq[p] > 0) active[na++] = p;
  if (verbose)
    printf("      start nsm=[%d,%d,%d,%d] gbs=%.1f\n",
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
            memcpy(nsm, cand, sizeof(int) * NPAT);
            best = g; improved = 1;
            if (verbose)
              printf("      step=%2d move %d:%-2d->%-2d nsm=[%d,%d,%d,%d] gbs=%.1f\n",
                     step, d, move, r, nsm[0], nsm[1], nsm[2], nsm[3], best);
          }
        }
      }
    }
  }
  *out_evals = evals;
  *out_gbs = best;
}

// §75 的单起点 in-situ 爬山（起点 = §74 单跑曲线 bind_opt）。
static void alloc_insitu(const int* seq, int* nsm_final, int O, int T,
                         int* out_evals, double* out_gbs, int verbose) {
  int nsm[NPAT];
  alloc_opt(seq, nsm);                          // 从 §74 单跑曲线的 bind_opt 出发
  if (verbose) printf("    iter0(solo bind_opt start) ->\n");
  hill_from(seq, nsm, O, T, out_evals, out_gbs, verbose);
  memcpy(nsm_final, nsm, sizeof(nsm));
}

// ---- §76 新增：多起点爬山 + oracle ----
static uint32_t g_rng = 20240501u;
static uint32_t rng_next() { g_rng = g_rng * 1664525u + 1013904223u; return g_rng; }

// 固定种子随机起点：把 numSM 随机切成 na 份（每份 >=1）
static void random_start(const int* active, int na, int* nsm) {
  for (int p = 0; p < NPAT; ++p) nsm[p] = 0;
  for (int i = 0; i < na; ++i) nsm[active[i]] = 1;
  int rem = g_numSm - na;
  for (int r = 0; r < rem; ++r) nsm[active[rng_next() % na]]++;
}

// 多起点爬山：结构化起点（bind_opt / 均分 / 每模式偏置）+ nrand 个随机起点，全部爬山后保留最优。
static void alloc_multistart(const int* seq, int* nsm_final, int O, int T, int nrand,
                             int* out_evals, double* out_gbs, int verbose) {
  int active[NPAT], na = 0;
  for (int p = 0; p < NPAT; ++p) if (seq[p] > 0) active[na++] = p;
  g_rng = 20240501u;                    // 固定种子：多起点结果可复现
  int best[NPAT], bests[NPAT]; double bestg = -1.0; int evals = 0, nstarts = 0;

  // 起点 0：§74 单跑 bind_opt
  alloc_opt(seq, bests);
  // 起点 1：均分（余数给第一个）
  int uni[NPAT] = {0, 0, 0, 0};
  for (int i = 0; i < na; ++i) uni[active[i]] = g_numSm / na;
  uni[active[0]] += g_numSm - (g_numSm / na) * na;
  // 起点 2..2+na-1：每个模式偏置（该模式拿走几乎全部）
  int nstruct = 2 + na;
  int total = nstruct + nrand;
  for (int s = 0; s < total; ++s) {
    int cand[NPAT];
    if (s == 0) { memcpy(cand, bests, sizeof(cand)); }
    else if (s == 1) { memcpy(cand, uni, sizeof(cand)); }
    else if (s < nstruct) {
      int d = active[s - 2];
      for (int p = 0; p < NPAT; ++p) cand[p] = seq[p] > 0 ? 1 : 0;
      cand[d] = g_numSm - (na - 1);
    } else { random_start(active, na, cand); }
    double g; int ev;
    hill_from(seq, cand, O, T, &ev, &g, 0);
    evals += ev; nstarts++;
    if (verbose)
      printf("    start#%d %-9s -> nsm=[%d,%d,%d,%d] gbs=%.1f (evals=%d)\n",
             s, (s == 0 ? "bind_opt" : s == 1 ? "uniform" : s < nstruct ? "bias" : "random"),
             cand[0], cand[1], cand[2], cand[3], g, ev);
    if (g > bestg) { bestg = g; memcpy(best, cand, sizeof(best)); }
  }
  memcpy(nsm_final, best, sizeof(best));
  if (!alloc_valid(seq, nsm_final))
    printf("    WARN alloc_multistart 非法分配 nsm=[%d,%d,%d,%d]（丢 work 会虚高）\n",
           nsm_final[0], nsm_final[1], nsm_final[2], nsm_final[3]);
  *out_evals = evals;
  *out_gbs = bestg;
}

// oracle：对两模式混合穷举 n_p（角色绑定族真最优），返回最优 n_p 与实测 GB/s。
static double oracle_two(int p, int q, int O, int T, int step, int* out_bestn) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[p] = 1; seq[q] = 1;
  double bestg = -1.0; int bestn = -1;
  for (int np = 1; np < g_numSm; np += step) {
    int nsm[NPAT] = {0, 0, 0, 0}; nsm[p] = np; nsm[q] = g_numSm - np;
    double g = eval_alloc(seq, nsm, O, T, 5);
    if (g > bestg) { bestg = g; bestn = np; }
  }
  // 补上 numSM-1（step 不整除时）
  { int nsm[NPAT] = {0, 0, 0, 0}; nsm[p] = g_numSm - 1; nsm[q] = 1;
    double g = eval_alloc(seq, nsm, O, T, 5);
    if (g > bestg) { bestg = g; bestn = g_numSm - 1; } }
  if (out_bestn) *out_bestn = bestn;
  return bestg;
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
  } else if (mapping == 5) {
    alloc_multistart(seq, nsm, O, T, 6, out_evals, &g_insitu_gbs, verbose);
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
               case 3: return "bind_opt"; case 4: return "bind_insitu"; case 5: return "bind_global"; }
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
  printf("# §76 SM 分配器：**单起点局部爬山** vs **多起点爬山** vs **oracle 穷举**\n");
  printf("#   补 §75.9 头号尾巴「bind_insitu 是局部爬山、结果不可复现（equal4 两次收敛到不同分配）」\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；clock=%.2f MHz；逻辑总读 1 GiB（uint4 计）\n",
         g_numSm, g_clockKHz / 1000.0);
  printf("#   reg U=%d；__maxnreg__(32)；问：多起点能否追平 oracle？单起点差多少？\n", U_ILP);
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

  // ===== §B solo baseline + 单跑速率曲线（复现 §74，bind_opt 起点需要）=====
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

  // ===== §C（Q1 可复现性）单起点 bind_insitu 重复 N 次 =====
  printf("\n===== §C（Q1 可复现性）单起点 bind_insitu 重复 N=8 次：收敛分配/吞吐离散度 =====\n");
  {
    int NREP = 8;
    int repset[2] = {0, 2};              // equal4（§75.9 头号案）与 coal+sparse
    for (int mi = 0; mi < 2; ++mi) {
      int idx = repset[mi];
      double gs[16]; int ns[16][NPAT];
      for (int r = 0; r < NREP; ++r) {
        int ev = 0; double gb = 0; int nsm[NPAT];
        alloc_insitu(mixes[idx].seq, nsm, mainO, mainT, &ev, &gb, 0);
        gs[r] = gb; memcpy(ns[r], nsm, sizeof(nsm));
      }
      double lo = gs[0], hi = gs[0], sum = 0;
      for (int r = 0; r < NREP; ++r) { if (gs[r] < lo) lo = gs[r]; if (gs[r] > hi) hi = gs[r]; sum += gs[r]; }
      printf("  [%s] N=%d：min=%.1f med=%.1f max=%.1f  spread(max/min)=%.3f\n",
             mixes[idx].nm, NREP, lo, gs[NREP / 2], hi, hi / lo);
      for (int r = 0; r < NREP; ++r)
        printf("    run%-2d nsm=[%3d,%3d,%3d,%3d] gbs=%7.1f\n",
               r, ns[r][0], ns[r][1], ns[r][2], ns[r][3], gs[r]);
    }
  }

  // ===== §D（Q2 多起点）六映射对照 + meas/E =====
  printf("\n===== §D（Q2）六映射对照（T=1024,O=1）：measured GB/s 与 meas/E =====\n");
  printf("  %-22s %9s | %11s %10s %10s %11s %11s %11s\n",
         "mixture", "E(切片)", "interleave", "bind", "bind_bal", "bind_opt", "bind_insitu", "bind_global");
  for (int m = 0; m < nmix; ++m) {
    double E = expected_additive(mixes[m].seq);
    printf("  %-22s %9.1f |", mixes[m].nm, E);
    for (int map = 0; map <= 5; ++map) {
      int ev = 0;
      Res r = run_mix(mainT, mainO, mixes[m].seq, map, nullptr, &ev, 0);
      printf(" %6.1f %5.3f", r.gbs, r.gbs / E);
    }
    printf("\n");
  }

  // ===== §E（Q3 oracle）两模式穷举 in-situ 扫描 =====
  printf("\n===== §E（Q3 oracle）两模式穷举 in-situ 扫描：真最优 vs 各分配器 =====\n");
  printf("  %-20s %11s %11s %11s %11s | %8s %8s %8s\n",
         "mixture", "interleave", "bind_opt", "bind_insitu", "bind_global", "org_n", "oracle", "bG/org");
  for (int m = 0; m < nmix; ++m) {
    int na = 0; for (int p = 0; p < NPAT; ++p) if (mixes[m].seq[p]) na++;
    if (na != 2) continue;
    int bestn = -1;
    int a2[2], k = 0; for (int p = 0; p < NPAT; ++p) if (mixes[m].seq[p]) a2[k++] = p;
    double org = oracle_two(a2[0], a2[1], mainO, mainT, 1, &bestn);
    int evr = 0;
    double gi = run_mix(mainT, mainO, mixes[m].seq, 0, nullptr, &evr, 0).gbs;
    double go = run_mix(mainT, mainO, mixes[m].seq, 3, nullptr, &evr, 0).gbs;
    double gs = run_mix(mainT, mainO, mixes[m].seq, 4, nullptr, &evr, 0).gbs;
    double gg = run_mix(mainT, mainO, mixes[m].seq, 5, nullptr, &evr, 0).gbs;
    printf("  %-20s %11.1f %11.1f %11.1f %11.1f | %8d %8.1f %8.3f\n",
           mixes[m].nm, gi, go, gs, gg, bestn, org, gg / org);
  }

  // ===== §F（Q4 噪声地板 + 多起点轨迹）=====
  printf("\n===== §F（Q4）同一收敛分配的重复评估噪声地板（equal4，R=8）=====\n");
  {
    int seq[NPAT] = {1, 1, 1, 1};
    int nsm[NPAT]; int ev = 0; double gb = 0;
    alloc_insitu(seq, nsm, mainO, mainT, &ev, &gb, 0);
    double gs[16]; int R = 8;
    for (int r = 0; r < R; ++r) gs[r] = eval_alloc(seq, nsm, mainO, mainT, 5);
    double lo = gs[0], hi = gs[0];
    for (int r = 0; r < R; ++r) { if (gs[r] < lo) lo = gs[r]; if (gs[r] > hi) hi = gs[r]; }
    printf("  nsm=[%d,%d,%d,%d] 重复评估 R=%d：min=%.1f max=%.1f spread(max/min)=%.3f\n",
           nsm[0], nsm[1], nsm[2], nsm[3], R, lo, hi, hi / lo);
  }
  printf("\n===== §F2 多起点爬山轨迹（equal4 与 coal+sparse）=====\n");
  {
    int seq4[NPAT] = {1, 1, 1, 1};
    int nsm[NPAT]; int ev = 0; double gb = 0;
    printf("  [equal 25x4] bind_global (6 随机起点)：\n");
    alloc_multistart(seq4, nsm, mainO, mainT, 6, &ev, &gb, 1);
    printf("    收敛 nsm=[%d,%d,%d,%d] gbs=%.1f  evals=%d\n", nsm[0], nsm[1], nsm[2], nsm[3], gb, ev);
    int seq2[NPAT] = {1, 1, 0, 0};
    printf("  [coal+sparse] bind_global：\n");
    g_rng = 20240501u;
    alloc_multistart(seq2, nsm, mainO, mainT, 6, &ev, &gb, 1);
    printf("    收敛 nsm=[%d,%d,%d,%d] gbs=%.1f  evals=%d\n", nsm[0], nsm[1], nsm[2], nsm[3], gb, ev);
  }

  // ===== §G 结论 =====
  printf("\n===== §G 结论 =====\n");
  printf("  - §C: 若单起点重复跑的 spread>1.02 => bind_insitu 结果不可复现（§75.9 尾巴坐实）。\n");
  printf("  - §D/§E: bind_global 的 meas/E 与 bG/org 若 ~1.0 => 多起点追平 oracle、单起点 gap 被收回。\n");
  printf("  - §F: 噪声地板 spread 给出评价本身的可分辨下限。\n");
  printf("\n########## §76 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 6) { printf("usage: run <T> <O> <map> <seq r0,r1,r2,r3>  (map 0=inter,1=bind,2=bind_bal,3=bind_opt,4=bind_insitu,5=bind_global)\n"); return 1; }
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
  printf("usage: hetero_sm_global {bench|run <T> <O> <map> <seq r0,r1,r2,r3>}\n");
  return 1;
}
