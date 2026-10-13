// ============================================================================
// P89-1 (§77)：补掉 §73–§76 反复出现的明文尾巴（§73.8 / §74.9 / §75.9 / §76.10，原话）：
//   「`bind` 用 **O=1**（每 SM 一 block），**未测每 SM 多块的角色绑定**」。
//
// 背景（接 §71–§76）：
//   - §72：四访存模式共驻（按 block 交错）大体加和（±13%），唯一系统性裂缝 = 发射受限
//     `scalar4B` × 访存受限；`sparse` 低掺杂毒化。§72.8 缓解 (a)：「把发射受限 task 聚到
//     一起**或加 warp 摊**」。
//   - §73：把「按 block 交错」换成「按 SM 角色绑定」（每 SM 只跑一种模式），五混合
//     `meas/E` 与交错**逐行几乎相同**（差 ≤1.1%）⇒ 共驻税与 `l2res` 超加和是**全局
//     （SM 间/HBM/L2）现象、不是 SM 内发射槽共享**；`bind_bal`（SM 数 ∝1/每SM速率）反而更差。
//   - §74/§75/§76：把 SM 分配器从常数速率→实测曲线→in-situ→多起点，收回了 `bind` 的
//     分区自伤，但**始终在 O=1** 下量（每 SM 恰一个 block、`T=1024`、32 warp/SM）。
//
// 但真实 megakernel 的 SM 角色绑定**不一定每 SM 一 block**：同一角色的 SM 上常驻**多块**
//   （提高每 SM warp 数、摊发射/延迟）。§73 的 `bind` 路径**只对 O=1 正确**——每 SM 的 O 个
//   block 会读到**同一段** `sm_base/sm_bp`，O>1 时重复读、结果错。故此前无法测 O>1。
//
// 本节把 `bind` 路径**正确推广到 O>1**：每 SM 的 O 个 block 用**设备侧 per-SM 原子游标**
//   领一个**局部序号** `local∈[0,O)`，读该 SM 区的第 `local` 个 `BP` 子段（XOR 可交换，
//   序号与正确性无关）。于是「SM 角色绑定」可在**任意每 SM block 数**下评测。
//
// 关键对照（**唯一新变量 = 每 SM block 数 O**，同时保持**每 SM 总线程 `O·T` 不变 = 1024**
//   ——§70/§71 已证吞吐由 `O·T` 这个合量决定）：
//     O=1,T=1024  （= §73–§76 的既有口径）
//     O=2,T=512
//     O=4,T=256
//   三档在**同一逻辑工作、同一模式份额、同一 SM 分配**下对跑，五混合 × 三映射。
//
// 问：
//   Q1 O>1 的 bind 正确吗？（per-SM 原子游标分配子段）
//   Q2 §73 的 headline「bind ≈ interleave、税是全局」在 O>1 下**是否仍成立**？→ 若 O 一变
//      税就消失，说明它是**每 SM block 数（warp 摊）**可回收的 SM 内现象，§73 的结论需限定在 O=1；
//      若逐行不变，则「全局税」在三个 O 下都稳，§73 结论对 O 稳健。
//   Q3 §72.8 缓解 (a)「加 warp 摊」是否兑现？（O↑ = 同 SM 更多 warp）
//   Q4 `bind_bal`（∝1/每SM速率）在 O>1 下是否仍更差？
//   Q5 equal4（4 模式）角色绑定在 O>1 下的极值。
//
// 四模式 / 逻辑字节 / XOR 参考 / 误差扩散 assign[] / SM 分配（naive、balanced）**完全复用
//   §71–§76**；载体固定寄存器读 + U=4 ILP、`__maxnreg__(32)`。**唯一新变量 = O（bind 的每
//   SM block 数）+ 使 bind 在 O>1 正确的 per-SM 原子子段游标**。
//
// 用法：./hetero_sm_bind_occ bench
//       ./hetero_sm_bind_occ run <T> <O> <map> <reps> <r0,r1,r2,r3>   # map: 0=inter,1=bind,2=bind_bal
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
#define SPARSE_STRIDE 8            // m1：每 8 个 uint4 读 1 个 → 物理 8×（扇区 4×）
#define L2N4 (1 << 20)             // m3：常驻 16 MiB = 1M uint4（< 50 MB L2）
#define NPAT 4
#define MAXGRID 4096

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

// ---- 统一 kernel ----
//   interleave（bind=0）：每 block 的区 = bb_base[blockIdx], 长度 bb_bp[blockIdx]（host 预计算）。
//   bind（bind=1）：pat = mode_of_sm[smid]；该 SM 的 O 个 block 用 per-SM 原子游标取局部序号
//                   local∈[0,O)，读区 [sm_base[smid] + local*sm_bp[smid], +sm_bp)。O=1 时即 §73。
//   `blk_smid` 记录 smid；`sm_ctr` 为 per-SM 原子游标（每次 launch 前清零）。
__global__ void __maxnreg__(32) pat_hetero(const void* __restrict__ raw, long long aux, int bind,
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

// 每模式逻辑工作 W_p（uint4）、BP_p、物理区起点 base_p、物理长度 phys_p
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

// 由 W_p 计算 BP_p、物理 base_p、phys_p（稀疏占 8×；l2res 只读前 16MiB，区从 raw 起点算）
static long long compute_layout(int O, long long* out_need) {
  long long phys_acc = 0, need = L2N4 + 1024;
  for (int p = 0; p < NPAT; ++p) {
    if (!g_nblk[p] || g_W[p] <= 0) { g_BP[p] = 0; g_basep[p] = phys_acc; g_phys[p] = 0; continue; }
    g_BP[p] = g_W[p] / g_nblk[p];
    if (g_BP[p] < 1) g_BP[p] = 1;
    g_basep[p] = phys_acc;
    long long stride = (p == 1) ? SPARSE_STRIDE : 1;
    if (p == 3) { g_phys[p] = 0; }                       // l2res 不用自己的区
    else g_phys[p] = (long long)g_nblk[p] * g_BP[p] * stride;
    phys_acc += g_phys[p];
    long long endp = (p == 1) ? g_basep[p] + g_phys[p] : g_basep[p] + g_phys[p];
    if (endp > need) need = endp;
  }
  need += TOT_U4;   // 1 GiB 余量（SM 放置不均时兜底）
  *out_need = need;
  return phys_acc;
}

static void launch_hetero(int grid) {
  long long aux = L2N4 - 1;
  CK(cudaMemset(g_sm_ctr, 0, (size_t)g_numSm * sizeof(int)));
  pat_hetero<<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_O, g_assign, g_mode_of_sm,
                            g_bb_base, g_bb_bp, g_sm_base, g_sm_bp, g_sm_ctr, g_blk_smid, g_sink);
  g_last_ok = (cudaGetLastError() == cudaSuccess);
}

// 误差扩散法生成 assign[]（= §72）
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

// round-robin 填 mode_of_sm[]（要求 Σnsm == numSM）
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

// setup：interleave（用 seq 决定每 block 模式；W_p = f_p·TOT）
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
  // 每 block 的区：base_p + rank(p)*BP_p
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

// setup：bind（SM 数 nsm 由调用方给；W_p = f_p·TOT 与 nsm 无关；O = 每 SM block 数）
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
      // SM s 覆盖 O 块 = O*g_BP[p]，起点按 rank 前进 O 块
      sbase[s] = g_basep[p] + (long long)rank[p] * O * g_BP[p];
      sbp[s] = g_BP[p];           // 每 block 子段长度
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
  res.gbs = (double)TOT_U4 * 16.0 / (best * 1e-3) / 1e9;   // 按**逻辑**字节算 GB/s
  res.unsupported = unsupported;
  return res;
}

static const char* PATNAME(int p) {
  switch (p) { case 0: return "coalesced"; case 1: return "sparse8x";
               case 2: return "scalar4B"; case 3: return "l2res16M"; }
  return "?";
}

// ---- solo 基线：全部 SM 同模式（走 bind）----
static double g_bw_solo[NPAT], g_r1[NPAT];

static Res run_solo(int pat, int T, int O) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[pat] = 1;
  int nsm[NPAT] = {0, 0, 0, 0}; nsm[pat] = g_numSm;
  int grid = g_numSm * O;
  g_T = T;
  setup_bind(seq, nsm, O, grid);
  return run(grid, 3);
}

// ---- host 参考（小规模）：按 (pat, base, BP) 的区求 XOR ----
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

// SM 分配：mode 0 = naive（∝ 份额）；mode 1 = balanced（∝ 份额/每SM速率）
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

// 运行一个混合在给定映射下；返回实测，nsm_out 给 bind 的 SM 分配
static Res run_mix(int T, int O, const int* seq, int mapping, int* nsm_out) {
  int grid = g_numSm * O;
  g_T = T; g_O = O;
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

// per-SM 放置诊断：跑一次 bind，读回 blk_smid，统计每 SM block 数分布与最大/最小
static void placement_check(int grid) {
  unsigned int* hsmid = (unsigned int*)malloc((size_t)grid * sizeof(unsigned int));
  CK(cudaMemcpy(hsmid, g_blk_smid, (size_t)grid * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  int cnt[4096]; for (int i = 0; i < 4096; ++i) cnt[i] = 0;
  for (int i = 0; i < grid; ++i) if (hsmid[i] < 4096) cnt[hsmid[i]]++;
  int used = 0, cmax = 0, cmin = 1 << 30;
  for (int i = 0; i < g_numSm; ++i) { if (cnt[i] > 0) { used++; if (cnt[i] > cmax) cmax = cnt[i]; if (cnt[i] < cmin) cmin = cnt[i]; } }
  printf("    placement: 占用 SM=%d/%d  每SM块数 min=%d max=%d（期望=%d）\n",
         used, g_numSm, used ? cmin : 0, cmax, g_O);
  free(hsmid);
}

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  printf("########################################################################\n");
  printf("# §77 SM 角色绑定 × 每 SM 多块（O>1）：补 §73.8/§74.9/§75.9/§76.10 反复出现的尾巴\n");
  printf("#   『bind 用 O=1（每 SM 一 block），未测每 SM 多块的角色绑定』\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 1 GiB（uint4 计）；reg U=%d；__maxnreg__(32)\n",
         g_numSm, U_ILP);
  printf("# 唯一新变量 = 每 SM block 数 O（保持 O·T=1024：O=1/T=1024, O=2/T=512, O=4/T=256）\n");
  printf("########################################################################\n");

  fill_all();

  struct Occ { int O; int T; };
  Occ occs[3] = {{1, 1024}, {2, 512}, {4, 256}};

  // ===== §A 正确性（O=1/2/4）=====
  {
    long long BP = 1 << 12;
    int NBLK = 16;
    int seq[NPAT] = {4, 4, 4, 4};
    printf("\n===== §A 正确性（interleave 小图 NBLK=%d；bind 小图 grid=numSM·O）=====\n", NBLK);
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
    printf("  interleave : sink=%llu ref=%llu -> %s\n", s_i, ref_i, s_i == ref_i ? "OK" : "WRONG");
    free(h_ass); free(h_b); free(h_bp);

    for (int oi = 0; oi < 3; ++oi) {
      int O = occs[oi].O;
      g_T = occs[oi].T;
      int nsm[NPAT] = {0, 0, 0, 0}; alloc_sm(seq, 0, nsm);
      int grid = g_numSm * O;
      setup_bind(seq, nsm, O, grid);
      CK(cudaMemset(g_sink, 0, 8));
      launch_hetero(grid);
      CK(cudaDeviceSynchronize());
      unsigned long long s_b = 0; CK(cudaMemcpy(&s_b, g_sink, 8, cudaMemcpyDeviceToHost));
      unsigned long long ref_b = 0;
      for (int p = 0; p < NPAT; ++p)
        for (int k = 0; k < g_nblk[p]; ++k)
          ref_b ^= ref_region(p, g_basep[p] + (long long)k * g_BP[p], g_BP[p]);
      printf("  bind O=%d T=%4d: sink=%llu ref=%llu -> %s\n", O, occs[oi].T, s_b, ref_b,
             s_b == ref_b ? "OK" : "WRONG");
      if (oi == 2) placement_check(grid);
    }
  }

  // ===== §B solo baseline（每 (O,T)）=====
  printf("\n===== §B 单模式 baseline：逻辑 GB/s（每 SM 总线程 O·T=1024）=====\n");
  printf("  %-11s %12s %12s %12s\n", "pattern", "O1/T1024", "O2/T512", "O4/T256");
  double solo[3][NPAT];
  for (int p = 0; p < NPAT; ++p) {
    for (int oi = 0; oi < 3; ++oi) {
      Res r = run_solo(p, occs[oi].T, occs[oi].O);
      solo[oi][p] = r.gbs;
    }
    printf("  %-11s %8.1f GB/s %8.1f GB/s %8.1f GB/s\n", PATNAME(p),
           solo[0][p], solo[1][p], solo[2][p]);
  }
  // §C 用 O=1 基线作 E（与 §73 一致）
  for (int p = 0; p < NPAT; ++p) { g_bw_solo[p] = solo[0][p]; g_r1[p] = solo[0][p] / (double)g_numSm; }
  printf("  [对照 §72/§73：coal~3050 sp~707 sc~2600 l2~25k]\n");

  // ===== §C（headline）三映射 × O 对照 =====
  struct Mix { const char* nm; int seq[NPAT]; };
  Mix mixes[] = {
    {"equal 25x4",          {1, 1, 1, 1}},
    {"50/50 coal+scalar",   {1, 0, 1, 0}},
    {"50/50 coal+sparse",   {1, 1, 0, 0}},
    {"50/50 coal+l2res",    {1, 0, 0, 1}},
    {"50/50 scalar+l2res",  {0, 0, 1, 1}},
  };
  int nmix = (int)(sizeof(mixes) / sizeof(mixes[0]));
  printf("\n===== §C（headline）三映射 × O：measured GB/s 与 meas/E（E 用 O=1 切片期望）=====\n");
  for (int oi = 0; oi < 3; ++oi) {
    int O = occs[oi].O, T = occs[oi].T;
    printf("  -- O=%d T=%d (grid=numSM·O=%d) --\n", O, T, g_numSm * O);
    printf("    %-22s %9s | %-15s %-15s %-15s\n", "mixture", "E",
           "interleave", "bind", "bind_bal");
    for (int m = 0; m < nmix; ++m) {
      double E = expected_additive(mixes[m].seq);
      Res ri = run_mix(T, O, mixes[m].seq, 0, nullptr);
      Res rb = run_mix(T, O, mixes[m].seq, 1, nullptr);
      Res rbal = run_mix(T, O, mixes[m].seq, 2, nullptr);
      printf("    %-22s %9.1f | %6.1f %5.3f   %6.1f %5.3f   %6.1f %5.3f\n",
             mixes[m].nm, E,
             ri.gbs, ri.gbs / E, rb.gbs, rb.gbs / E, rbal.gbs, rbal.gbs / E);
    }
  }

  // ===== §D SM 分配明细（O=1 口径）=====
  printf("\n===== §D 角色绑定 SM 分配（numSM=%d；blocks = SM 数 × O）=====\n", g_numSm);
  for (int m = 0; m < nmix; ++m) {
    int nsm[NPAT];
    alloc_sm(mixes[m].seq, 0, nsm);
    printf("  %-22s %-9s nsm=[%3d,%3d,%3d,%3d]\n", mixes[m].nm, "bind",
           nsm[0], nsm[1], nsm[2], nsm[3]);
    alloc_sm(mixes[m].seq, 1, nsm);
    printf("  %-22s %-9s nsm=[%3d,%3d,%3d,%3d]\n", mixes[m].nm, "bind_bal",
           nsm[0], nsm[1], nsm[2], nsm[3]);
  }

  // ===== §E coal+scalar 角色绑定：扫 scalar 的 SM 份额（O=1 vs O=4）=====
  printf("\n===== §E coal+scalar 角色绑定：扫 scalar SM 份额 × O =====\n");
  printf("  %-10s %14s %14s\n", "scSM", "O=1 meas/E", "O=4 meas/E");
  for (int sc = 0; sc <= 132; sc += 12) {
    double cell[2]; int oi2[2] = {0, 2};
    for (int j = 0; j < 2; ++j) {
      int oi = oi2[j];
      int O = occs[oi].O, T = occs[oi].T;
      int nsm[NPAT] = {0, 0, 0, 0};
      nsm[2] = (sc > g_numSm) ? g_numSm : sc;
      nsm[0] = g_numSm - nsm[2];
      if (nsm[0] == 0 && nsm[2] == 0) { cell[j] = 0; continue; }
      int seq[NPAT] = {1, 0, 1, 0};
      g_T = T; g_O = O;
      setup_bind(seq, nsm, O, g_numSm * O);
      Res r = run(g_numSm * O, 3);
      double act_u4 = 0.0;
      for (int p = 0; p < NPAT; ++p) act_u4 += (double)g_nblk[p] * g_BP[p];
      double gbs_act = act_u4 * 16.0 / (r.ms * 1e-3) / 1e9;
      double inv = 0.0;
      for (int p = 0; p < NPAT; ++p) {
        double fp = (double)g_nblk[p] * g_BP[p] / (act_u4 > 0 ? act_u4 : 1);
        if (fp > 0) inv += fp / g_bw_solo[p];
      }
      double E = inv > 0 ? 1.0 / inv : 0;
      cell[j] = E > 0 ? gbs_act / E : 0.0;
    }
    printf("  %-10d %14.3f %14.3f\n", sc, cell[0], cell[1]);
  }

  // ===== §F 结论 =====
  printf("\n===== §F 结论 =====\n");
  printf("  - §C: 若 bind/interleave 的 meas/E 在三档 O 下**逐行不变** => §73『税是全局』对每 SM\n");
  printf("        block 数稳健；若 O↑ 让 coal+scalar 的税消失 => 它是 warp 摊可回收的 SM 内现象。\n");
  printf("  - §B: 三档 O 若近似相等 => 复现 §70/§71『吞吐由合量 O·T 定』；块级 prologue 让大 O 略慢\n");
  printf("        是次要项。\n");
  printf("  - §A: bind O>1 逐位对 host 参考 => per-SM 原子子段游标正确。\n");
  printf("\n########## §77 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "dbg")) {
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    int seq[NPAT] = {1, 0, 0, 0};
    int nsm[NPAT] = {0, 0, 0, 0}; nsm[0] = g_numSm;
    g_T = 1024;
    setup_bind(seq, nsm, 1, g_numSm);
    int* hmos = (int*)malloc((size_t)g_numSm * sizeof(int));
    long long* hb = (long long*)malloc((size_t)g_numSm * sizeof(long long));
    long long* hp = (long long*)malloc((size_t)g_numSm * sizeof(long long));
    CK(cudaMemcpy(hmos, g_mode_of_sm, (size_t)g_numSm * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb, g_sm_base, (size_t)g_numSm * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hp, g_sm_bp, (size_t)g_numSm * sizeof(long long), cudaMemcpyDeviceToHost));
    printf("dbg: nblk=[%d,%d,%d,%d] basep=%lld BP=%lld\n", g_nblk[0], g_nblk[1], g_nblk[2], g_nblk[3], g_basep[0], g_BP[0]);
    for (int s = 0; s < 4; ++s) printf("  sm%d mode=%d base=%lld bp=%lld\n", s, hmos[s], hb[s], hp[s]);
    CK(cudaMemset(g_sink, 0, 8));
    launch_hetero(g_numSm);
    CK(cudaDeviceSynchronize());
    int* hctr = (int*)malloc((size_t)g_numSm * sizeof(int));
    CK(cudaMemcpy(hctr, g_sm_ctr, (size_t)g_numSm * sizeof(int), cudaMemcpyDeviceToHost));
    long long sum = 0; for (int i = 0; i < g_numSm; ++i) sum += hctr[i];
    printf("  sm_ctr sum=%lld (expect %d)\n", sum, g_numSm);
    unsigned long long sv = 0; CK(cudaMemcpy(&sv, g_sink, 8, cudaMemcpyDeviceToHost));
    printf("  sink=%llu\n", sv);
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 7) { printf("usage: run <T> <O> <map> <reps> <r0,r1,r2,r3>  (map 0=inter,1=bind,2=bind_bal)\n"); return 1; }
    g_T = atoi(argv[2]);
    int O = atoi(argv[3]);
    int map = atoi(argv[4]);
    int reps = atoi(argv[5]);
    int seq[NPAT] = {0, 0, 0, 0};
    if (sscanf(argv[6], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) < 4) { printf("bad seq\n"); return 1; }
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    for (int p = 0; p < NPAT; ++p) {
      if (!seq[p]) continue;
      Res s = run_solo(p, g_T, O);
      g_bw_solo[p] = s.gbs; g_r1[p] = s.gbs / (double)g_numSm;
    }
    int nsm[NPAT];
    Res r = run_mix(g_T, O, seq, map, nsm);
    printf("  T=%d O=%d map=%d seq=%d,%d,%d,%d grid=%d -> %s ms=%.4f GB/s=%.1f\n",
           g_T, O, map, seq[0], seq[1], seq[2], seq[3], g_numSm * O,
           r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs);
    return 0;
  }
  printf("usage: hetero_sm_bind_occ {bench|run <T> <O> <map> <reps> <r0,r1,r2,r3>}\n");
  return 1;
}
