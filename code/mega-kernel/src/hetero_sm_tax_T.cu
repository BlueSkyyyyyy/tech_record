// ============================================================================
// P94-1 (§82)：补掉 P93-1（§81）§81.7 明留的尾巴——
//   「`U ≤ 32`、`R ≤ 128`、`T=256` 固定；`U=32,R=32` 才 spill（门槛偏窄），**`R<32` 未测**」+
//   「§C 固定 `R=128` 时 `occ` 随 `U` 变（**ILP 与占用率在 §C 内仍部分纠缠**），靠 §D 的单变量
//     `R` 扫解耦」。
//
// 背景（接 §71–§81）：
//   §80/§81 把「发射受限 × 访存受限」共驻税（`coal+scalar` 的 `meas/E`）压到**每 SM 在飞 load 量
//   `Wsm·U`**（`Wsm`=每 SM warp 数、`U`=每线程 ILP）；`§81 §C` 在**固定 `R=128`** 下扫 `U` 得到
//   「税封顶在 `U≈8`」，但那一列 `occ`（`Wsm`）随 `U` 同时变——**ILP 收益与占用率损失纠缠在一起**，
//   无法干净回答「固定 `Wsm` 下 `U` 单独值多少」。
//
// 本节把 **每块线程数 `T`** 变成第三旋钮（§81 固定 `T=256`），并用它把「`Wsm` 固定、`U` 变」与
//   「`U` 固定、块粒度 `T` 变」两种解耦都做出来；同时把**寄存器预算下限 `R<32`** 的角点补上。
//   口径关键：H100 每 SM 的占用率由四个上限取 min——**线程上限 `2048/T`**、**寄存器上限
//   `65536/(regs·T)`**、**块上限 32**、**SMEM 上限 `SMEM_perSM/block_smem`**。若 `regs ≤ 32`，
//   寄存器上限恒 `≥ 2048/T`，即**占用率被线程上限封顶**，此时把 `R` 压到 32 以下**买不到任何
//   额外占用率**（只会引入 spill）。这正是 §81「最小不 spill 的 R」规则在 Hopper 上的根。
//   （本内核每块带 8 KiB 静态归约数组，故小 `T` 时 SMEM 反成第四约束——§B/§D 会把它分离出来。）
//
//   Q1 **ILP 天花板（去占用率耦合）**：把 `Wsm` 钉在 2048（满占用率），用 `R=32` 让 `U≤16` 都
//      0 spill、`occ` 不随 `U` 变，扫 `U` —— 得到的 `meas/E` 才是**纯 ILP** 曲线；与 §81 `§C`
//      （`R=128`、`occ` 随 `U` 降）对照，量出「占用率耦合」贡献多少。
//   Q2 **块粒度 `T` 独立性**：固定 `Wsm=2048`，用 `T∈{64,128,256,512}`（`occ=32/16/8/4`）看
//      同一 `U` 下 `mixed`/`meas/E` 是否相同——检验 §79「税只认 `Wsm`，与块粒度无关」在
//      `coal+scalar` 上是否也成立（§80/§92 报告过绝对量认 `T`）。
//   Q3 **`R<32` 角点**：`R∈{24,28,32,48,...}`——`__maxnreg__` 的**下限是 24**（ptxas 会把 <24
//      调到 24，已实测），故 `R=24/28` 是可达的最低预算；看它们能否提升占用率（预期否：线程上限
//      已封顶）以及 spill 门槛。
//
// 完全复用 §72–§81 的四访存模式 / 逻辑字节 / XOR 参考 / 误差扩散 `assign[]` / SM 分配 / per-SM
//   原子子段游标。载体固定寄存器读 + `__maxnreg__(R)`（模板参数）。
//
// 用法：./hetero_sm_tax_T bench
//       ./hetero_sm_tax_T run <U> <R> <T> <map> <r0,r1,r2,r3>   # map 0=inter,1=bind
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
#define TOT_U4 (16777216LL)        // 逻辑总读 256 MiB（uint4 计）；所有混合份额同一逻辑字节数
#define SPARSE_STRIDE 8            // m1：每 8 个 uint4 读 1 个 → 物理 8×（扇区 4×）
#define L2N4 (1 << 20)             // m3：常驻 16 MiB = 1M uint4（< 50 MB L2）
#define NPAT 4
#define MAXGRID 8192
#define NU 5                       // U ∈ {1,2,4,8,16}
#define NR 7                       // R ∈ {24,28,32,48,64,96,128}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 四模式读函数（与 §71–§81 逐字一致）----
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

// ---- 统一 kernel（模板化在每线程 ILP `U` 与寄存器预算 `R` 上；`T` 是运行期 blockDim）----
template <int U, int R>
__global__ void __maxnreg__(R) pat_T(const void* __restrict__ raw, long long aux, int bind,
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
// dispatch：U ∈ {1,2,4,8,16} × R ∈ {24,28,32,48,64,96,128}
// ============================================================================
static const void* kernel_ptr(int u, int r) {
#define CASE(U, R) if (u == U && r == R) return (const void*)&pat_T<U, R>;
  CASE(1, 24) CASE(1, 28) CASE(1, 32) CASE(1, 48) CASE(1, 64) CASE(1, 96) CASE(1, 128)
  CASE(2, 24) CASE(2, 28) CASE(2, 32) CASE(2, 48) CASE(2, 64) CASE(2, 96) CASE(2, 128)
  CASE(4, 24) CASE(4, 28) CASE(4, 32) CASE(4, 48) CASE(4, 64) CASE(4, 96) CASE(4, 128)
  CASE(8, 24) CASE(8, 28) CASE(8, 32) CASE(8, 48) CASE(8, 64) CASE(8, 96) CASE(8, 128)
  CASE(16, 24) CASE(16, 28) CASE(16, 32) CASE(16, 48) CASE(16, 64) CASE(16, 96) CASE(16, 128)
#undef CASE
  return nullptr;
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
static int g_R = 32;

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

#define LAUNCH(U, R)                                                                          \
  pat_T<U, R><<<grid, g_T>>>((const void*)g_data, aux, g_cur_bind, g_O, g_assign,             \
                             g_mode_of_sm, g_bb_base, g_bb_bp, g_sm_base, g_sm_bp,            \
                             g_sm_ctr, g_blk_smid, g_sink)

static void launch_T(int grid) {
  long long aux = L2N4 - 1;
  CK(cudaMemset(g_sm_ctr, 0, (size_t)g_numSm * sizeof(int)));
  if (g_U == 1) {
    if (g_R == 24) LAUNCH(1, 24); else if (g_R == 28) LAUNCH(1, 28);
    else if (g_R == 32) LAUNCH(1, 32); else if (g_R == 48) LAUNCH(1, 48);
    else if (g_R == 64) LAUNCH(1, 64); else if (g_R == 96) LAUNCH(1, 96);
    else LAUNCH(1, 128);
  } else if (g_U == 2) {
    if (g_R == 24) LAUNCH(2, 24); else if (g_R == 28) LAUNCH(2, 28);
    else if (g_R == 32) LAUNCH(2, 32); else if (g_R == 48) LAUNCH(2, 48);
    else if (g_R == 64) LAUNCH(2, 64); else if (g_R == 96) LAUNCH(2, 96);
    else LAUNCH(2, 128);
  } else if (g_U == 4) {
    if (g_R == 24) LAUNCH(4, 24); else if (g_R == 28) LAUNCH(4, 28);
    else if (g_R == 32) LAUNCH(4, 32); else if (g_R == 48) LAUNCH(4, 48);
    else if (g_R == 64) LAUNCH(4, 64); else if (g_R == 96) LAUNCH(4, 96);
    else LAUNCH(4, 128);
  } else if (g_U == 8) {
    if (g_R == 24) LAUNCH(8, 24); else if (g_R == 28) LAUNCH(8, 28);
    else if (g_R == 32) LAUNCH(8, 32); else if (g_R == 48) LAUNCH(8, 48);
    else if (g_R == 64) LAUNCH(8, 64); else if (g_R == 96) LAUNCH(8, 96);
    else LAUNCH(8, 128);
  } else {
    if (g_R == 24) LAUNCH(16, 24); else if (g_R == 28) LAUNCH(16, 28);
    else if (g_R == 32) LAUNCH(16, 32); else if (g_R == 48) LAUNCH(16, 48);
    else if (g_R == 64) LAUNCH(16, 64); else if (g_R == 96) LAUNCH(16, 96);
    else LAUNCH(16, 128);
  }
  g_last_ok = (cudaGetLastError() == cudaSuccess);
}
#undef LAUNCH

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
    launch_T(grid);
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

static int occ_blocks(int U, int R, int T) {
  const void* k = kernel_ptr(U, R);
  if (!k) return 0;
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, T, 0) != cudaSuccess) return 0;
  return nb;
}

static int regs_of(int U, int R) {
  const void* k = kernel_ptr(U, R);
  cudaFuncAttributes a;
  if (!k) return -1;
  if (cudaFuncGetAttributes(&a, k) != cudaSuccess) return -1;
  return a.numRegs;
}

static long long local_bytes_of(int U, int R) {
  const void* k = kernel_ptr(U, R);
  cudaFuncAttributes a;
  if (!k) return -1;
  if (cudaFuncGetAttributes(&a, k) != cudaSuccess) return -1;
  return (long long)a.localSizeBytes;
}

static int smem_bytes_of(int U, int R) {
  const void* k = kernel_ptr(U, R);
  cudaFuncAttributes a;
  if (!k) return -1;
  if (cudaFuncGetAttributes(&a, k) != cudaSuccess) return -1;
  return (int)a.sharedSizeBytes;
}

static Res run_solo_occ(int pat, int T, int U, int R, int* out_O) {
  int seq[NPAT] = {0, 0, 0, 0}; seq[pat] = 1;
  int nsm[NPAT] = {0, 0, 0, 0}; nsm[pat] = g_numSm;
  int occ = occ_blocks(U, R, T);
  int O = occ > 0 ? occ : 1;
  if (O > 64) O = 64;
  int grid = g_numSm * O;
  g_T = T; g_U = U; g_R = R; g_O = O;
  if (out_O) *out_O = O;
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

static Res run_mix_occ(int T, int U, int R, const int* seq, int* out_O, int* nsm_out) {
  int occ = occ_blocks(U, R, T);
  int O = occ > 0 ? occ : 1;
  if (O > 64) O = 64;
  int grid = g_numSm * O;
  g_T = T; g_U = U; g_R = R; g_O = O;
  int nsm[NPAT] = {0, 0, 0, 0};
  alloc_sm(seq, 0, nsm);
  setup_bind(seq, nsm, O, grid);
  if (out_O) *out_O = O;
  if (nsm_out) for (int p = 0; p < NPAT; ++p) nsm_out[p] = nsm[p];
  return run(grid, 3);
}

// ============================================================================
// P96-1（§84）稳健采样：§82.6/§83.5 的 T 趋势只有 ~3–5%、只取 3 次 run，**在噪声地板附近**。
//   本模式把「每配置 S 次单采样 + round-robin 交错（消除时钟/热漂移）+ bootstrap 置信区间」
//   做出来，直接判定该 T 趋势是真效应还是噪声；并配合 ncu MemoryWorkloadAnalysis 做内存延迟分解。
// ============================================================================
static double sample_once(int grid) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaMemset(g_sink, 0, 8));
  CK(cudaEventRecord(e0));
  launch_T(grid);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return (double)ms;
}

static int cmp_d(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x < y) ? -1 : ((x > y) ? 1 : 0);
}
static double median_of(const double* v, int n) {
  double* c = (double*)malloc((size_t)n * sizeof(double));
  memcpy(c, v, (size_t)n * sizeof(double));
  qsort(c, n, sizeof(double), cmp_d);
  double m = (n & 1) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
  free(c);
  return m;
}
static double mean_of(const double* v, int n) {
  double s = 0; for (int i = 0; i < n; ++i) s += v[i]; return s / n;
}
static double std_of(const double* v, int n) {
  double m = mean_of(v, n), s = 0;
  for (int i = 0; i < n; ++i) { double d = v[i] - m; s += d * d; }
  return (n > 1) ? sqrt(s / (n - 1)) : 0.0;
}
static double min_of(const double* v, int n) {
  double m = v[0]; for (int i = 1; i < n; ++i) if (v[i] < m) m = v[i]; return m;
}

// 为稳健模式准备一个配置；solo_pat>=0 => 单模式（该模式独占整机 SM），否则按 seq 份额 alloc_sm。
static int setup_config(const int* seq, int T, int U, int R, int solo_pat, int* out_occ) {
  g_T = T; g_U = U; g_R = R;
  int occ = occ_blocks(U, R, T);
  int O = occ > 0 ? occ : 1; if (O > 64) O = 64;
  int grid = g_numSm * O;
  g_O = O;
  int nsm[NPAT] = {0, 0, 0, 0};
  if (solo_pat >= 0) nsm[solo_pat] = g_numSm;
  else alloc_sm(seq, 0, nsm);
  setup_bind(seq, nsm, O, grid);
  if (out_occ) *out_occ = O;
  return grid;
}

// bootstrap：ratio = median(b)/median(a) 的 95% CI（B 次重采样，各自有放回）。
static void boot_ratio_ci(const double* a, const double* b, int n, int B,
                          double* lo, double* hi, double* med) {
  double* ra = (double*)malloc((size_t)n * sizeof(double));
  double* rb = (double*)malloc((size_t)n * sizeof(double));
  double* rs = (double*)malloc((size_t)B * sizeof(double));
  srand(12345);
  for (int t = 0; t < B; ++t) {
    for (int i = 0; i < n; ++i) { ra[i] = a[rand() % n]; rb[i] = b[rand() % n]; }
    double ma = median_of(ra, n), mb = median_of(rb, n);
    rs[t] = (ma > 0) ? mb / ma : 0.0;
  }
  qsort(rs, B, sizeof(double), cmp_d);
  *lo = rs[(int)(0.025 * B)];
  *hi = rs[(int)(0.975 * B)];
  *med = rs[B / 2];
  free(ra); free(rb); free(rs);
}

#define ROB_NC 7
#define ROB_S 30
static void fill_all();
struct RobCfg { const char* name; int seq[NPAT]; int T; int solo; int U; int R; };

static void robust() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  printf("########################################################################\n");
  printf("# §84 块粒度 T 次级效应的统计稳健性 + 内存延迟分解（补 §83.5 尾巴，P96-1）\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 256 MiB（uint4 计）；U=8 R=32 map=bind\n", g_numSm);
  printf("# 关键：§82.6/§83.5 的 T 趋势只有 ~3–5%%、只取 3 次 run（min-of-3），在噪声地板附近。\n");
  printf("# 本模式：每配置 S=%d 次**单采样**、round-robin 交错（消除时钟/热漂移），给 min/median/mean/std\n", ROB_S);
  printf("#   + T512/T128、T256/T128 的 median 比与 bootstrap 95%% CI（B=2000）。\n");
  printf("########################################################################\n");

  fill_all();

  RobCfg cfgs[ROB_NC] = {
      {"mixed scalar+l2res", {0, 0, 1, 1}, 128, -1, 8, 32},
      {"mixed scalar+l2res", {0, 0, 1, 1}, 256, -1, 8, 32},
      {"mixed scalar+l2res", {0, 0, 1, 1}, 512, -1, 8, 32},
      {"l2res16M solo",      {0, 0, 0, 1}, 128,  3, 8, 32},
      {"l2res16M solo",      {0, 0, 0, 1}, 512,  3, 8, 32},
      {"scalar4B solo",      {0, 0, 1, 0}, 128,  2, 8, 32},
      {"scalar4B solo",      {0, 0, 1, 0}, 512,  2, 8, 32},
  };

  int grid[ROB_NC], occ[ROB_NC];
  for (int c = 0; c < ROB_NC; ++c)
    grid[c] = setup_config(cfgs[c].seq, cfgs[c].T, cfgs[c].U, cfgs[c].R, cfgs[c].solo, &occ[c]);

  printf("\n===== 资源：每配置 occ / Wsm / grid =====\n");
  printf("  %-22s %5s %5s %6s %6s\n", "config", "T", "occ", "Wsm", "grid");
  for (int c = 0; c < ROB_NC; ++c)
    printf("  %-22s %5d %5d %6d %6d\n", cfgs[c].name, cfgs[c].T, occ[c], occ[c] * cfgs[c].T, grid[c]);

  static double samp[ROB_NC][ROB_S];
  // warmup：每配置先跑 3 次（时钟爬坡 + L2 预热；不采样）
  for (int c = 0; c < ROB_NC; ++c) {
    setup_config(cfgs[c].seq, cfgs[c].T, cfgs[c].U, cfgs[c].R, cfgs[c].solo, nullptr);
    for (int w = 0; w < 3; ++w) (void)sample_once(grid[c]);
  }
  // 交错采样
  for (int s = 0; s < ROB_S; ++s)
    for (int c = 0; c < ROB_NC; ++c) {
      setup_config(cfgs[c].seq, cfgs[c].T, cfgs[c].U, cfgs[c].R, cfgs[c].solo, nullptr);
      samp[c][s] = sample_once(grid[c]);
    }

  printf("\n===== §A 稳健计时（S=%d，交错；GB/s = 256 MiB / 时间）=====\n", ROB_S);
  printf("  %-22s %5s %10s %10s %10s %9s %11s %11s\n",
         "config", "T", "min_ms", "med_ms", "mean_ms", "std_%", "GB/s(med)", "GB/s(min)");
  for (int c = 0; c < ROB_NC; ++c) {
    double mn = min_of(samp[c], ROB_S), md = median_of(samp[c], ROB_S), me = mean_of(samp[c], ROB_S);
    double sd = std_of(samp[c], ROB_S);
    double gmd = (double)TOT_U4 * 16.0 / (md * 1e-3) / 1e9;
    double gmn = (double)TOT_U4 * 16.0 / (mn * 1e-3) / 1e9;
    printf("  %-22s %5d %10.4f %10.4f %10.4f %9.2f %11.1f %11.1f\n",
           cfgs[c].name, cfgs[c].T, mn, md, me, (me > 0 ? 100.0 * sd / me : 0.0), gmd, gmn);
  }

  printf("\n===== §B T 趋势是否超过噪声（median 比 + bootstrap 95%% CI）=====\n");
  printf("  若 CI 含 1.000 => 该 T 比在 95%% 置信下不可区分于噪声。\n");
  printf("  %-22s %14s %22s %14s %22s\n", "config",
         "T512/T128", "CI95[T512/T128]", "T256/T128", "CI95[T256/T128]");
  // 混合：c=0(128),1(256),2(512)
  {
    double lo, hi, md;
    boot_ratio_ci(samp[0], samp[2], ROB_S, 2000, &lo, &hi, &md);
    double lo2, hi2, md2;
    boot_ratio_ci(samp[0], samp[1], ROB_S, 2000, &lo2, &hi2, &md2);
    printf("  %-22s %14.4f [%13.4f,%8.4f] %14.4f [%13.4f,%8.4f]\n",
           "mixed scalar+l2res", md, lo, hi, md2, lo2, hi2);
    double loa, hia, mda;
    boot_ratio_ci(samp[3], samp[4], ROB_S, 2000, &loa, &hia, &mda);
    printf("  %-22s %14.4f [%13.4f,%8.4f] %14s %22s\n",
           "l2res16M solo", mda, loa, hia, "-", "-");
    double lob, hib, mdb;
    boot_ratio_ci(samp[5], samp[6], ROB_S, 2000, &lob, &hib, &mdb);
    printf("  %-22s %14.4f [%13.4f,%8.4f] %14s %22s\n",
           "scalar4B solo", mdb, lob, hib, "-", "-");
  }

  printf("\n===== §C 结论（自动判定）=====\n");
  {
    double lo, hi, md;
    boot_ratio_ci(samp[0], samp[2], ROB_S, 2000, &lo, &hi, &md);
    printf("  mixed T512/T128 median 比 = %.4f，CI95=[%.4f,%.4f] -> %s\n", md, lo, hi,
           (lo > 1.0 || hi < 1.0) ? "显著（真效应）" : "含 1.0（与噪声不可区分）");
  }
  printf("\n########## §84 robust 完成 ##########\n");
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

static const int ULIST[NU] = {1, 2, 4, 8, 16};
static const int RLIST[NR] = {24, 28, 32, 48, 64, 96, 128};
static const int TLIST[5] = {64, 128, 256, 512, 1024};

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  printf("########################################################################\n");
  printf("# §82 每块线程数 T 作为第三旋钮：占用率「由谁封顶」、R<32 角点、固定 Wsm 的纯 ILP 曲线（补 §81.7 尾巴，P94-1）\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；逻辑总读 256 MiB（uint4 计）；寄存器读 + __maxnreg__(R)\n", g_numSm);
  printf("# 内核模板化在 <U,R>；T 是运行期 blockDim；O 由 occupancy API 实测（O=occ），grid=numSM·occ\n");
  printf("# H100 占用率 = min(线程上限 2048/T, 寄存器上限 65536/(regs·T), 块上限 32, SMEM 上限)\n");
  printf("########################################################################\n");

  fill_all();

  // ===== §A 正确性 =====
  {
    int NBLK = 16;
    int seq[NPAT] = {4, 4, 4, 4};
    printf("\n===== §A 正确性（interleave 小图 NBLK=%d；bind 全图 occupancy-aware）=====\n", NBLK);
    int tri[][3] = {{4, 24, 256}, {8, 32, 256}, {16, 32, 256}, {4, 24, 128}};
    int ntr = (int)(sizeof(tri) / sizeof(tri[0]));
    for (int k = 0; k < ntr; ++k) {
      int U = tri[k][0], R = tri[k][1], T = tri[k][2];
      g_T = T; g_U = U; g_R = R;
      setup_interleave(seq, 1, NBLK);
      CK(cudaMemset(g_sink, 0, 8));
      launch_T(NBLK);
      if (!g_last_ok) { printf("  interleave U=%2d R=%3d T=%4d: UNSUP\n", U, R, T); continue; }
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
      printf("  interleave U=%2d R=%3d T=%4d: sink=%llu ref=%llu -> %s\n", U, R, T, s_i, ref_i,
             s_i == ref_i ? "OK" : "WRONG");
      free(h_ass); free(h_b); free(h_bp);
    }
    for (int k = 0; k < ntr; ++k) {
      int U = tri[k][0], R = tri[k][1], T = tri[k][2];
      g_T = T; g_U = U; g_R = R;
      int nsm[NPAT] = {0, 0, 0, 0}; alloc_sm(seq, 0, nsm);
      int occ = occ_blocks(U, R, T); int O = occ > 0 ? occ : 1; if (O > 64) O = 64;
      int grid = g_numSm * O;
      setup_bind(seq, nsm, O, grid);
      CK(cudaMemset(g_sink, 0, 8));
      launch_T(grid);
      if (!g_last_ok) { printf("  bind U=%2d R=%3d T=%4d: UNSUP\n", U, R, T); continue; }
      CK(cudaDeviceSynchronize());
      unsigned long long s_b = 0; CK(cudaMemcpy(&s_b, g_sink, 8, cudaMemcpyDeviceToHost));
      unsigned long long ref_b = 0;
      for (int p = 0; p < NPAT; ++p)
        for (int kk = 0; kk < g_nblk[p]; ++kk)
          ref_b ^= ref_region(p, g_basep[p] + (long long)kk * g_BP[p], g_BP[p]);
      printf("  bind U=%2d R=%3d T=%4d occ=%2d (Wsm=%4d, grid=%4d): sink=%llu ref=%llu -> %s\n",
             U, R, T, occ, O * T, grid, s_b, ref_b, s_b == ref_b ? "OK" : "WRONG");
    }
  }

  // ===== §B 资源表：T × R 的 occ 与封顶项 =====
  printf("\n===== §B 资源表（U=8）：T×R -> numRegs / occ / Wsm；封顶项 min(线程2048/T, 寄存器65536/(r·T), 块32) =====\n");
  printf("  %-6s", "T\\R");
  for (int ri = 0; ri < NR; ++ri) printf(" R=%-5d", RLIST[ri]);
  printf("\n");
  for (int ti = 0; ti < 5; ++ti) {
    int T = TLIST[ti];
    printf("  T=%-4d", T);
    for (int ri = 0; ri < NR; ++ri) {
      int rg = regs_of(8, RLIST[ri]);
      int oc = occ_blocks(8, RLIST[ri], T);
      char tmp[16]; snprintf(tmp, sizeof(tmp), "%d/%d", rg, oc);
      printf(" %-8s", tmp);
    }
    printf("   (r/occ)\n");
  }
  printf("  # 计算封顶（U=8, R=32, numRegs=32）：每 T 的 线程/寄存器/**SMEM**/块 四上限：\n");
  int smemPerSm = 0;
  CK(cudaDeviceGetAttribute(&smemPerSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  int smemBlk = smem_bytes_of(8, 32);
  printf("       (每块静态 SMEM=%d B = reduce_xor<MAXT=1024> 的 8 KiB 归约数组 + s_local；SMEM/SM=%d B)\n",
         smemBlk, smemPerSm);
  for (int ti = 0; ti < 5; ++ti) {
    int T = TLIST[ti];
    int thr = 2048 / T;
    int reg = (int)(65536 / (32 * T));
    int blk = 32;
    int sm = smemBlk > 0 ? smemPerSm / smemBlk : 9999;
    int mn = thr < reg ? thr : reg; if (blk < mn) mn = blk; if (sm < mn) mn = sm;
    printf("       T=%-5d 线程=%-3d 寄存器=%-3d SMEM=%-3d 块=%-3d -> min=%d（occ API 实测=%d）\n",
           T, thr, reg, sm, blk, mn, occ_blocks(8, 32, T));
  }

  // ===== §C（headline 一）固定 Wsm=2048 的纯 ILP 曲线 =====
  printf("\n===== §C（headline 一）固定 Wsm=2048（R=32、满占用率、0 spill）扫 U：纯 ILP 曲线 =====\n");
  printf("  对照 §81 §C 的 R=128 列（occ 随 U 降 => ILP 与占用率纠缠）。\n");
  int seqCS[NPAT] = {1, 0, 1, 0};
  for (int ti = 0; ti < 3; ++ti) {  // T=256(occ8),128(occ16),64(occ32)
    int T = TLIST[2 - ti];  // 256,128,64
    printf("\n  -- T=%d（Wsm 期望 %d）--\n", T, 2048);
    printf("  %-4s %5s %6s %4s %6s %10s %10s %10s %9s\n",
           "U", "R", "regs", "occ", "Wsm", "coal", "scalar", "mixed", "meas/E");
    for (int ui = 0; ui < NU; ++ui) {
      int U = ULIST[ui], R = 32, o;
      Res rc = run_solo_occ(0, T, U, R, &o); g_bw_solo[0] = rc.gbs; g_r1[0] = rc.gbs / g_numSm;
      Res rs = run_solo_occ(2, T, U, R, &o); g_bw_solo[2] = rs.gbs; g_r1[2] = rs.gbs / g_numSm;
      double E = expected_additive(seqCS);
      Res rm = run_mix_occ(T, U, R, seqCS, &o, nullptr);
      printf("  %-4d %5d %6d %4d %6d %10.1f %10.1f %10.1f %9.3f\n",
             U, R, regs_of(U, R), o, o * T, rc.gbs, rs.gbs, rm.gbs, rm.gbs / E);
    }
  }

  // ===== §D（headline 二）固定 Wsm=2048、固定 U，扫块粒度 T =====
  printf("\n===== §D（headline 二）固定 Wsm 意图=2048（R=32）、固定 U，扫 T∈{64,128,256,512} =====\n");
  printf("  检验 §79「税只认 Wsm，与块粒度无关」在 coal+scalar 上是否成立。\n");
  printf("  注意 T=64 被每块静态 SMEM（%d B）封顶到 occ=24（Wsm=1536，非 2048；按 233472/%d≈28 估，\n",
         smem_bytes_of(8, 32), smem_bytes_of(8, 32));
  printf("  实测 24，差在 SMEM 分配粒度/carveout）——SMEM 是独立于寄存器的第四个占用率上限；\n");
  printf("  干净对比取 T∈{128,256,512}（Wsm 均 2048）。\n");
  for (int uu = 0; uu < 2; ++uu) {
    int U = (uu == 0) ? 4 : 8;
    printf("\n  -- U=%d --\n", U);
    printf("  %-6s %5s %4s %6s %10s %10s %10s %9s\n",
           "T", "occ", "Wsm", "regs", "coal", "scalar", "mixed", "meas/E");
    for (int ti = 0; ti < 4; ++ti) {
      int T = TLIST[ti], R = 32, o;
      Res rc = run_solo_occ(0, T, U, R, &o); g_bw_solo[0] = rc.gbs; g_r1[0] = rc.gbs / g_numSm;
      Res rs = run_solo_occ(2, T, U, R, &o); g_bw_solo[2] = rs.gbs; g_r1[2] = rs.gbs / g_numSm;
      double E = expected_additive(seqCS);
      Res rm = run_mix_occ(T, U, R, seqCS, &o, nullptr);
      printf("  %-6d %5d %4d %6d %10.1f %10.1f %10.1f %9.3f\n",
             T, o, o * T, regs_of(U, R), rc.gbs, rs.gbs, rm.gbs, rm.gbs / E);
    }
  }

  // ===== §E R<32 角点：固定 T=128、U=8，扫 R =====
  printf("\n===== §E R<32 角点：固定 T=128、U=8，扫 R∈{24,28,32,48,64,96,128} =====\n");
  printf("  __maxnreg__ 下限 24：R<24（如 16/20）ptxas 会调到 24 并告警，故 R=24 是可达最低预算。\n");
  printf("  %-5s %6s %4s %6s %9s %10s %10s %9s\n", "R", "regs", "occ", "Wsm", "localB", "coal", "mixed", "meas/E");
  for (int ri = 0; ri < NR; ++ri) {
    int R = RLIST[ri], o;
    int U = 8, T = 128;
    Res rc = run_solo_occ(0, T, U, R, &o); g_bw_solo[0] = rc.gbs; g_r1[0] = rc.gbs / g_numSm;
    Res rs = run_solo_occ(2, T, U, R, &o); g_bw_solo[2] = rs.gbs; g_r1[2] = rs.gbs / g_numSm;
    double E = expected_additive(seqCS);
    Res rm = run_mix_occ(T, U, R, seqCS, &o, nullptr);
    printf("  %-5d %6d %4d %6d %9lld %10.1f %10.1f %9.3f\n", R, regs_of(U, R), o, o * T,
           local_bytes_of(U, R), rc.gbs, rm.gbs, rm.gbs / E);
  }

  // ===== §F 第二混合 scalar+l2res：固定 Wsm=2048 扫 T（U=8）=====
  printf("\n===== §F（Q3）第二混合 scalar+l2res：固定 Wsm=2048（R=32,U=8）扫 T =====\n");
  int seqSL[NPAT] = {0, 0, 1, 1};
  printf("  %-6s %5s %4s %10s %10s %9s\n", "T", "occ", "Wsm", "l2res", "mixed", "meas/E");
  for (int ti = 0; ti < 4; ++ti) {
    int T = TLIST[ti], R = 32, U = 8, o;
    Res rl = run_solo_occ(3, T, U, R, &o); g_bw_solo[3] = rl.gbs; g_r1[3] = rl.gbs / g_numSm;
    Res rs = run_solo_occ(2, T, U, R, &o); g_bw_solo[2] = rs.gbs; g_r1[2] = rs.gbs / g_numSm;
    double E = expected_additive(seqSL);
    Res rm = run_mix_occ(T, U, R, seqSL, &o, nullptr);
    printf("  %-6d %5d %4d %10.1f %10.1f %9.3f\n", T, o, o * T, rl.gbs, rm.gbs, rm.gbs / E);
  }

  printf("\n===== §G 结论 =====\n");
  printf("  - §B: R≤32 时占用率被线程上限 2048/T 封顶 => R<32 买不到额外占用率；\n");
  printf("  - §C: 固定 Wsm=2048 的纯 ILP 曲线 vs §81 §C（R=128、occ 随 U 降）=> 分离 ILP 与占用率；\n");
  printf("  - §D: 固定 Wsm 扫 T => 「税只认 Wsm」在 coal+scalar 上是否成立；\n");
  printf("  - §E: R=24/28 能否提升 occ（预期否）与 spill 门槛；\n");
  printf("  - §F: scalar+l2res 是否也认 Wsm。\n");
  printf("\n########## §82 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) { bench(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "robust")) { robust(); return 0; }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 7) { printf("usage: run <U> <R> <T> <map> <r0,r1,r2,r3>\n"); return 1; }
    g_U = atoi(argv[2]);
    g_R = atoi(argv[3]);
    g_T = atoi(argv[4]);
    int map = atoi(argv[5]);
    int seq[NPAT] = {0, 0, 0, 0};
    if (sscanf(argv[6], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) < 4) return 1;
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    fill_all();
    int occ = occ_blocks(g_U, g_R, g_T); int O = occ > 0 ? occ : 1; if (O > 64) O = 64;
    int grid = g_numSm * O;
    g_O = O;
    int nsm[NPAT] = {0, 0, 0, 0};
    if (map == 0) setup_interleave(seq, O, grid);
    else { alloc_sm(seq, 0, nsm); setup_bind(seq, nsm, O, grid); }
    Res r = run(grid, 3);
    printf("  U=%d R=%d T=%d map=%d occ=%d Wsm=%d seq=%d,%d,%d,%d grid=%d -> %s ms=%.4f GB/s=%.1f\n",
           g_U, g_R, g_T, map, O, O * g_T, seq[0], seq[1], seq[2], seq[3], grid,
           r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs);
    return 0;
  }
  printf("usage: hetero_sm_tax_T {bench|robust|run <U> <R> <T> <map> <r0,r1,r2,r3>}\n");
  return 1;
}
