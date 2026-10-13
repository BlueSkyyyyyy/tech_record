// ============================================================================
// P82-1 (§70)：把 §69 的尾巴「未测 T 与 O 同时大于 1 的二维面」补掉——
//   在**固定每线程寄存器预算**（`__maxnreg__(R)`）下，同时扫**每块线程数 T**与**每 SM 块数 O**，
//   检验数据面是否只认「每 SM 总线程 `O·T`」（×每线程 MLP `U`）这一个合量。
//
// 背景（接 §60 / §61 / §68 / §69）：
//   - §68：`grid=numSM·O` 钉占用率、`__launch_bounds__` 压寄存器 → 「同占用率下多给寄存器快 1.70×」，
//     且「寄存器预算与占用率 O 是替代品」（32-reg 靠 O=1→4 补 1.83×）。
//   - §69：`__maxnreg__(R)` 精确钉寄存器；§D 固定 O=1 只扫 T → 「R=32 靠 T=128→1024 从 899 补到
//     3136，逼近 R=64 的 3191」——把替代性推广到 T 维度。
//   - §69.7 明文尾巴：**「未测 T 与 O 同时大于 1 的二维面（O·T 联合钉『每 SM 总线程』）」**。
//
// 本节问：把 T 和 O **同时**打开，数据面吞吐是否只由 `O·T`（每 SM 总线程）决定？
//   若「是」，则 (T,O) 落在同一条 `O·T=const` 对角线上的所有配置应给出**同一吞吐**，
//   且**两条天花板**同时存在：
//     (a) 线程面：`O·T ≤ 每 SM 最大线程数`（H100 = 2048）；
//     (b) 寄存器面：`O·T ≤ 65536/R`（每 SM 寄存器文件 64K，R 越大能驻留的线程越少）。
//   故同一 R=32 vs R=64：前者线程面 2048、后者寄存器面 1024——**寄存器预算把可达的 O·T 封顶**。
//
// 载体：寄存器 `__ldcs` uint4 + U=4 路 ILP（= §60/§61/§68/§69 数据面）。
// 用法：./t_occ_surface bench
//       ./t_occ_surface run <R> <T> <O> <reps>
// 目标卡 H100 / sm90a。安全：脚本带 timeout；纯读微基准，无环协议/无死锁。
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
#define TOT_U4 (134217728LL)  // 总读 2 GiB（uint4 计）

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 数据面：寄存器 `__ldcs` uint4 + U 路 ILP（= §60/§61/§68/§69 载体）----
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

template <int U, int MAXT_>
__device__ __forceinline__ void body_reg(const uint4* __restrict__ base, long long BP,
                                         int tid, int T, unsigned long long* sink) {
  __shared__ unsigned long long red[MAXT_];
  unsigned long long part = read_reg<U>(base, BP, tid, T);
  red[tid] = part;
  __syncthreads();
  for (int s = T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(sink, red[0]);
}

// ---- 精确寄存器上限：__maxnreg__(R)（CUDA 12.4+ / PTX .maxnreg）----
#define DEF_MR(R)                                                                       \
  __global__ void __maxnreg__(R) mr_##R(const uint4* __restrict__ data, long long BP,   \
                                        unsigned long long* __restrict__ sink) {         \
    const int tid = threadIdx.x;                                                         \
    const int T = blockDim.x;                                                            \
    const uint4* base = data + (long long)blockIdx.x * BP;                               \
    body_reg<U_ILP, MAXT>(base, BP, tid, T, sink);                                       \
  }
DEF_MR(32)
DEF_MR(40)
DEF_MR(64)
#undef DEF_MR

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

// ============================================================================
// host
// ============================================================================
static uint4* g_data = nullptr;
static unsigned long long* g_sink = nullptr;
static unsigned long long* g_dref = nullptr;
static long long g_data_cap = 0;
static int g_numSm = 0;
static int g_R = 32, g_T = 256;

static void ensure(long long data_u4) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_dref) CK(cudaMalloc(&g_dref, 8));
}

static int g_last_ok = 1;

static void launch_mr(int grid, int T, long long BP) {
#define TRY_MR(R)                                                                       \
  if (g_R == R) {                                                                       \
    mr_##R<<<grid, T>>>(g_data, BP, g_sink);                                            \
    g_last_ok = (cudaGetLastError() == cudaSuccess);                                    \
    return;                                                                             \
  }
  TRY_MR(32); TRY_MR(40); TRY_MR(64);
#undef TRY_MR
  printf("  !! unknown R=%d\n", g_R);
  g_last_ok = 0;
}

static int occ_mr(int R, int T) {
  int nb = 0;
  cudaError_t e = cudaErrorInvalidValue;
#define Q_MR(RR)                                                                        \
  if (R == RR) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mr_##RR, T, 0);
  Q_MR(32) Q_MR(40) Q_MR(64)
#undef Q_MR
  if (e != cudaSuccess) return -1;
  return nb;
}

struct Res {
  double ms;
  double gbs;
  int unsupported;
};

static Res run(int O, int reps) {
  int grid = g_numSm * O;
  long long BP = TOT_U4 / grid;  // 总读 2 GiB / grid（uint4 计）
  ensure((long long)grid * BP);
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    launch_mr(grid, g_T, BP);
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

struct Point {
  int R, T, O, prod;   // prod = T*O（每 SM 目标线程数）
  int lim, resident;   // API blk/SM 上限、实际驻留块数
  double gbs;
  int unsupported;
};
#define MAXPTS 256
static Point g_pts[MAXPTS];
static int g_npts = 0;

// 扫一个 R：Ts[] 给 T 列表，Os[] 给每档 T 的 O 列表（二维不规则面）
static void sweep_R(int R, const int* Ts, const int* nO, const int nT, int reps) {
  g_R = R;
  printf("\n===== R=%d（__maxnreg__）2D 面：行=T（每块线程数），列=O（每 SM 块数），格=GB/s =====\n", R);
  printf("  %-6s", "T\\O");
  int Omax = 0;
  for (int t = 0; t < nT; ++t) if (nO[t] > Omax) Omax = nO[t];
  for (int o = 1; o <= Omax; ++o) printf(" %8d", o);
  printf("   %-10s %-14s\n", "T·O=1..", "note");
  for (int t = 0; t < nT; ++t) {
    int T = Ts[t];
    printf("  %-6d", T);
    for (int o = 1; o <= Omax; ++o) {
      if (o > nO[t]) { printf(" %8s", "-"); continue; }
      g_T = T;
      Res r = run(o, reps);
      int lim = occ_mr(R, T);
      int resident = (lim > 0 && o > lim) ? lim : o;
      printf(" %8.1f", r.unsupported ? -1.0 : r.gbs);
      if (g_npts < MAXPTS) {
        Point p; p.R = R; p.T = T; p.O = o; p.prod = T * o;
        p.lim = lim; p.resident = resident; p.gbs = r.gbs; p.unsupported = r.unsupported;
        g_pts[g_npts++] = p;
      }
    }
    int lim = occ_mr(R, T);
    int thr_cap = lim > 0 ? lim * T : -1;
    printf("   blk/SM(T=%d,API)=%d → 线程面 %d", T, lim, thr_cap);
    if (lim > 0 && nO[t] > lim) printf(" (给的最大 O=%d 超过 API 上限，resident 被钳)", nO[t]);
    printf("\n");
  }
}

static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));

  printf("########################################################################\n");
  printf("# §70 T×O 二维面：固定每线程寄存器预算（__maxnreg__），数据面是否只认『每 SM 总线程 O·T』\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；总读 2 GiB = grid·BP（uint4=16B）；载体 reg U=%d\n",
         g_numSm, U_ILP);
  printf("# 每 SM 两个天花板：线程面 ≤2048；寄存器面 ≤65536/R（R=32→2048、R=40→1638、R=64→1024）\n");
  printf("########################################################################\n");

  // ===== §A 正确性 =====
  {
    long long BP = 1 << 14;
    int NBLK = 16;
    long long tot = (long long)NBLK * BP;
    unsigned long long href = 0;
    for (long long u = 0; u < tot; ++u) {
      uint32_t x = mix32((uint32_t)u, 0u), y = mix32((uint32_t)u, 1u);
      uint32_t z = mix32((uint32_t)u, 2u), w = mix32((uint32_t)u, 3u);
      href += (unsigned long long)(x ^ y ^ z ^ w);
    }
    ensure(tot);
    CK(cudaMemset(g_dref, 0, 8));
    fill_data_kernel<<<(int)((tot + 255) / 256), 256>>>(g_data, tot, g_dref);
    printf("\n===== §A 正确性（NBLK=%d，BP=%lld）：不同 (R,T,O) vs host 参考 =====\n", NBLK, BP);
    struct { int R, T, O; } cas[] = {{32,128,1},{32,256,4},{40,512,1},{64,1024,1},{64,256,2}};
    for (int i = 0; i < 5; ++i) {
      g_R = cas[i].R; g_T = cas[i].T;
      int grid = NBLK * cas[i].O;
      long long BPi = tot / grid;   // 每块读 tot/grid，总读仍 = tot（对 host 参考一致）
      CK(cudaMemset(g_sink, 0, 8));
      launch_mr(grid, cas[i].T, BPi);
      CK(cudaDeviceSynchronize());
      unsigned long long s = 0; CK(cudaMemcpy(&s, g_sink, 8, cudaMemcpyDeviceToHost));
      printf("  R=%d T=%-5d O=%d : sink=%llu ref=%llu -> %s\n", cas[i].R, cas[i].T, cas[i].O,
             s, href, s == href ? "OK" : "WRONG");
    }
  }

  // ===== §B R=32 的 2D 面：T·O 最大到线程面 2048 =====
  {
    int Ts[]  = {128, 256, 512, 1024};
    int nO[]  = {16,   8,   4,    2};   // 均恰好到 T·O=2048（线程面天花板）
    sweep_R(32, Ts, nO, 4, 3);
  }

  // ===== §C R=64 的 2D 面：T·O 最大到寄存器面 1024（65536/64）=====
  {
    int Ts[]  = {128, 256, 512, 1024};
    int nO[]  = {8,   4,   2,    1};    // 均恰好到 T·O=1024（寄存器面天花板）
    sweep_R(64, Ts, nO, 4, 3);
  }

  // ===== §D R=40 的 2D 面：T·O 最大到寄存器面 1638（实际落 1024/2048 之间）=====
  {
    int Ts[]  = {128, 256, 512, 1024};
    int nO[]  = {8,   4,   2,    1};    // T·O 最大 1024（再大被 40-reg 封顶）
    sweep_R(40, Ts, nO, 4, 3);
  }

  // ===== §E 等积（iso O·T）聚合：同一条对角线上所有配置吞吐是否一致 =====
  printf("\n===== §E 等积聚合：按『每 SM 总线程 O·T』分组，看吞吐是否只是 O·T 的函数 =====\n");
  printf("  (同一 O·T 组内把不同 (R,T,O) 的 GB/s 列在一起；组内极差 = max−min)\n");
  {
    int prods[] = {128, 256, 512, 1024, 2048};
    for (int pi = 0; pi < 5; ++pi) {
      int pr = prods[pi];
      double mn = 1e30, mx = -1e30; int cnt = 0;
      for (int i = 0; i < g_npts; ++i) {
        if (g_pts[i].prod == pr && !g_pts[i].unsupported) {
          cnt++;
          if (g_pts[i].gbs < mn) mn = g_pts[i].gbs;
          if (g_pts[i].gbs > mx) mx = g_pts[i].gbs;
        }
      }
      printf("  --- O·T=%-5d (%d 个配置) ---\n", pr, cnt);
      if (cnt == 0) continue;
      for (int i = 0; i < g_npts; ++i) {
        if (g_pts[i].prod == pr && !g_pts[i].unsupported) {
          printf("             R=%-3d T=%-5d O=%-2d (blk/SM≤%d, resident=%d) -> %7.1f\n",
                 g_pts[i].R, g_pts[i].T, g_pts[i].O, g_pts[i].lim,
                 g_pts[i].resident, g_pts[i].gbs);
        }
      }
      printf("             → 组内 min=%.1f max=%.1f 极差=%.2f×\n", mn, mx, mn > 0 ? mx / mn : 0.0);
    }
  }

  // ===== §F 函数属性与 occupancy API =====
  printf("\n===== §F 函数属性（numRegs / localB）与 occupancy API（各 T）=====\n");
  {
    cudaFuncAttributes at;
    int Ts[] = {128, 256, 512, 1024};
#define OA_MR(RR)                                                                       \
    do {                                                                                \
      CK(cudaFuncGetAttributes(&at, mr_##RR));                                          \
      printf("  mr_%-3d numRegs=%-4d localB=%-3zu :", RR, at.numRegs, at.localSizeBytes); \
      for (int ti = 0; ti < 4; ++ti) {                                                  \
        int nb = 0;                                                                     \
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mr_##RR, Ts[ti], 0));     \
        printf(" T=%d→%d", Ts[ti], nb);                                                 \
      }                                                                                 \
      printf(" blk/SM\n");                                                             \
    } while (0)
    OA_MR(32); OA_MR(40); OA_MR(64);
#undef OA_MR
    printf("  (线程面 2048 = H100 每 SM 最大线程/64 warp；寄存器面 = 65536/numRegs)\n");
  }

  printf("\n########## §70 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) {
    bench();
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 6) { printf("usage: run <R> <T> <O> <reps>\n"); return 1; }
    g_R = atoi(argv[2]);
    g_T = atoi(argv[3]);
    int O = atoi(argv[4]);
    int reps = atoi(argv[5]);
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    Res r = run(O, reps);
    int lim = occ_mr(g_R, g_T);
    printf("  R=%d T=%d O=%d grid=%d blk/SM(API)=%d -> %s ms=%.4f GB/s=%.1f\n",
           g_R, g_T, O, g_numSm * O, lim, r.unsupported ? "UNSUP" : "OK", r.ms, r.gbs);
    return 0;
  }
  printf("usage: t_occ_surface {bench|run ...}\n");
  return 1;
}
