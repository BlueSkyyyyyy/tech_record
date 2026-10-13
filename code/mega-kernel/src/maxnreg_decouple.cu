// ============================================================================
// P81-1 (§69)：用 `__maxnreg__` 做**精确**寄存器上限，检验 §68 的「32 vs ≥40 regs」
// 分界，并把「每线程寄存器预算」与「每 SM 线程数 / 占用率」再拆一层。
//
// 背景（接 §60 / §61 / §67 / §68 / §72）：
//   - §68 用 `__launch_bounds__(256,MINB)` 当寄存器预算旋钮，把占用率用 grid=numSM·O 钉死，
//     得出 headline「同占用率下多给寄存器让数据面快 1.70×」——**数据面认的是每线程在飞 load 数
//     （寄存器预算 ILP），不是占用率本身**。
//   - 但 §68.7 明留尾巴（原话）：**「`MINB` 只是 ptxas 的一个 ceiling/hint，实测 numRegs=
//     32/40/44/40（MINB=8/6/4/2）并不精确等于旋钮档位；**未用 `__maxnreg__` 做精确上限**；
//     且占用率是 **block 粒度**（grid 钉每 SM 块数），**未扫 T（每块线程数）**把
//     『每线程寄存器预算』与『每 SM 驻留 warp 数』分开。」**
//
// 本节用 CUDA 12.4+ 的 `__maxnreg__(R)`（PTX `.maxnreg`）把**每线程寄存器上限**设成**精确**值 R，
// 与占用率（grid + blockDim 独立控制）解耦，回答三问：
//   Q1（§B，固定 O=1、T=256，只扫精确 R）：§68 的「32 vs ≥40」阶跃能否在**精确**旋钮下复现？
//       还是它只是「ptxas 挑了一个低寄存器 schedule」的副作用？
//   Q2（§C，`__maxnreg__` vs `__launch_bounds__` 在**相同 numRegs** 下对跑）：是「寄存器数」本身
//       决定吞吐，还是「用哪个属性压出来」也参与？
//   Q3（§D，固定 O=1、固定 R，只扫 T）：数据面认的是**每线程寄存器预算 ILP**，还是**每 SM 总在飞
//       load 数**（≈ O·T·U）？若后者，R=32 加大 T 应能补回 R=64。
//
// 载体：寄存器 `__ldcs` uint4 + U 路 ILP（= §60/§61/§68 的数据面，U 固定 4）。
// 用法：./maxnreg_decouple bench
//       ./maxnreg_decouple run <R> <T> <O> <reps>
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

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 数据面：寄存器 `__ldcs` uint4 + U 路 ILP（= §60/§61/§68 载体）----
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
DEF_MR(36)
DEF_MR(40)
DEF_MR(44)
DEF_MR(48)
DEF_MR(64)
DEF_MR(96)
DEF_MR(128)
#undef DEF_MR

// ---- 对照：__launch_bounds__(256,MINB)（§68 的 hint 旋钮），同一 body ----
#define DEF_LB(MB)                                                                      \
  __global__ void __launch_bounds__(256, MB) lb_##MB(                                   \
      const uint4* __restrict__ data, long long BP,                                     \
      unsigned long long* __restrict__ sink) {                                          \
    const int tid = threadIdx.x;                                                         \
    const int T = blockDim.x;                                                            \
    const uint4* base = data + (long long)blockIdx.x * BP;                               \
    body_reg<U_ILP, MAXT>(base, BP, tid, T, sink);                                       \
  }
DEF_LB(8)
DEF_LB(6)
DEF_LB(4)
DEF_LB(2)
#undef DEF_LB

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
static unsigned long long* g_sink = nullptr;
static unsigned long long* g_dref = nullptr;
static long long g_data_cap = 0;
static int g_numSm = 0;

static void ensure(long long data_u4) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_dref) CK(cudaMalloc(&g_dref, 8));
}

static int g_R = 32, g_MINB = 8, g_T = 256;
static int g_last_ok = 1;

// __maxnreg__ 版本：R 编译期 dispatch
static void launch_mr(int grid, int T, long long BP) {
#define TRY_MR(R)                                                                       \
  if (g_R == R) {                                                                       \
    mr_##R<<<grid, T>>>(g_data, BP, g_sink);                                            \
    g_last_ok = (cudaGetLastError() == cudaSuccess);                                    \
    return;                                                                             \
  }
  TRY_MR(32); TRY_MR(36); TRY_MR(40); TRY_MR(44);
  TRY_MR(48); TRY_MR(64); TRY_MR(96); TRY_MR(128);
#undef TRY_MR
  printf("  !! unknown R=%d\n", g_R);
  g_last_ok = 0;
}

// __launch_bounds__ 版本：MINB 编译期 dispatch
static void launch_lb(int grid, int T, long long BP) {
#define TRY_LB(MB)                                                                      \
  if (g_MINB == MB) {                                                                   \
    lb_##MB<<<grid, T>>>(g_data, BP, g_sink);                                           \
    g_last_ok = (cudaGetLastError() == cudaSuccess);                                    \
    return;                                                                             \
  }
  TRY_LB(8); TRY_LB(6); TRY_LB(4); TRY_LB(2);
#undef TRY_LB
  printf("  !! unknown MINB=%d\n", g_MINB);
  g_last_ok = 0;
}

struct Res {
  double ms;
  int unsupported;
};

// use_lb=0 → maxnreg(R)；use_lb=1 → launch_bounds(MINB)
static Res run(int O, int reps, int use_lb) {
  int grid = g_numSm * O;
  long long BP = 134217728LL / grid;  // 总读 2 GiB / grid
  ensure((long long)grid * BP);
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    if (use_lb) launch_lb(grid, g_T, BP);
    else launch_mr(grid, g_T, BP);
    if (!g_last_ok) { unsupported = 1; break; }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) best = ms;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  Res res; res.ms = best; res.unsupported = unsupported;
  return res;
}

static int occ_mr(int R, int T) {
  int nb = 0;
  cudaError_t e = cudaErrorInvalidValue;
#define Q_MR(RR)                                                                        \
  if (R == RR) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mr_##RR, T, 0);
  Q_MR(32) Q_MR(36) Q_MR(40) Q_MR(44) Q_MR(48) Q_MR(64) Q_MR(96) Q_MR(128)
#undef Q_MR
  if (e != cudaSuccess) return -1;
  return nb;
}

static int occ_lb(int MB, int T) {
  int nb = 0;
  cudaError_t e = cudaErrorInvalidValue;
#define Q_LB(MB_)                                                                       \
  if (MB == MB_) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, lb_##MB_, T, 0);
  Q_LB(8) Q_LB(6) Q_LB(4) Q_LB(2)
#undef Q_LB
  if (e != cudaSuccess) return -1;
  return nb;
}

static void line_mr(const char* tag, int R, int T, int O, int reps) {
  g_R = R; g_T = T;
  Res r = run(O, reps, 0);
  double gbs = (134217728.0 * 16.0) / (r.ms * 1e-3) / 1e9;
  int lim = occ_mr(R, T);
  int resident = (lim > 0 && O > lim) ? lim : O;
  printf("  %-14s maxnreg=%-4d T=%-5d O=%d grid=%-5d blk/SM(API)=%-2d resident=%-2d -> %-5s "
         "ms=%.4f GB/s=%7.1f\n",
         tag, R, T, O, g_numSm * O, lim, resident, r.unsupported ? "UNSUP" : "OK", r.ms, gbs);
}

static void line_lb(const char* tag, int MB, int T, int O, int reps) {
  g_MINB = MB; g_T = T;
  Res r = run(O, reps, 1);
  double gbs = (134217728.0 * 16.0) / (r.ms * 1e-3) / 1e9;
  int lim = occ_lb(MB, T);
  int resident = (lim > 0 && O > lim) ? lim : O;
  printf("  %-14s MINB=%-4d   T=%-5d O=%d grid=%-5d blk/SM(API)=%-2d resident=%-2d -> %-5s "
         "ms=%.4f GB/s=%7.1f\n",
         tag, MB, T, O, g_numSm * O, lim, resident, r.unsupported ? "UNSUP" : "OK", r.ms, gbs);
}

// ============================================================================
// bench
// ============================================================================
static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));

  printf("########################################################################\n");
  printf("# §69 精确寄存器上限（__maxnreg__）× 每 SM 线程数（T）：再解一层 §68 的尾巴\n");
  printf("# 目标卡 H100/sm90a；numSM=%d；总读 2 GiB = grid·BP（uint4=16B）；载体 reg U=%d\n",
         g_numSm, U_ILP);
  printf("# __maxnreg__(R) 精确钉每线程寄存器上限；占用率=grid=numSM·O、每块线程数=T 独立\n");
  printf("########################################################################\n");

  // ===== §A 正确性：mr_32 / mr_64 折叠逐位对 host 参考 =====
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
    printf("\n===== §A 正确性（NBLK=%d，BP=%lld）：maxnreg 载体 vs host 参考 =====\n", NBLK, BP);
    for (int ri = 0; ri < 2; ++ri) {
      int R = ri == 0 ? 32 : 64;
      g_R = R; g_T = 256;
      CK(cudaMemset(g_sink, 0, 8));
      launch_mr(NBLK, 256, BP);
      CK(cudaDeviceSynchronize());
      unsigned long long s = 0; CK(cudaMemcpy(&s, g_sink, 8, cudaMemcpyDeviceToHost));
      printf("  mr R=%d : sink=%llu ref=%llu -> %s\n", R, s, href, s == href ? "OK" : "WRONG");
    }
  }

  // ===== §B 固定 O=1、T=256，只扫精确寄存器上限 R（headline Q1）=====
  printf("\n===== §B 固定占用率 O=1、T=256，只扫精确 R=__maxnreg__ (headline Q1) =====\n");
  printf("  (grid=numSM 不变；R 直接写进 .maxnreg；对照 §68 的 MINB hint)\n");
  {
    int Rs[] = {32, 36, 40, 44, 48, 64, 96, 128};
    for (int i = 0; i < 8; ++i) {
      char tag[24]; snprintf(tag, 24, "B-R%d", Rs[i]);
      line_mr(tag, Rs[i], 256, 1, 3);
    }
  }

  // ===== §C maxnreg vs launch_bounds，在相同 numRegs 下对跑（Q2）=====
  printf("\n===== §C __maxnreg__(R) vs __launch_bounds__(256,MINB)，匹配 numRegs 对跑（Q2）=====\n");
  printf("  (§68 实测 MINB=8/6/4 → numRegs 32/40/44；与 maxnreg 32/40/44 逐档对照，O=1、T=256)\n");
  {
    line_mr("C-mr32", 32, 256, 1, 3);
    line_lb("C-lb8", 8, 256, 1, 3);
    line_mr("C-mr40", 40, 256, 1, 3);
    line_lb("C-lb6", 6, 256, 1, 3);
    line_mr("C-mr44", 44, 256, 1, 3);
    line_lb("C-lb4", 4, 256, 1, 3);
  }

  // ===== §D 固定 O=1、固定 R，只扫每块线程数 T（Q3 headline）=====
  printf("\n===== §D 固定 O=1、固定 R，只扫 T（每 SM 驻留 warp 数）（Q3 headline）=====\n");
  printf("  (数据面认『每线程寄存器预算 ILP』还是『每 SM 总在飞 load 数 ≈ O·T·U』？)\n");
  {
    int Rs[] = {32, 64};
    int Ts[] = {128, 256, 512, 1024};
    for (int ri = 0; ri < 2; ++ri) {
      printf("  --- R=%d ---\n", Rs[ri]);
      for (int ti = 0; ti < 4; ++ti) {
        char tag[24]; snprintf(tag, 24, "D-R%d-T%d", Rs[ri], Ts[ti]);
        line_mr(tag, Rs[ri], Ts[ti], 1, 3);
      }
    }
  }

  // ===== §E 扫 O（每 SM 块数）在精确 R 下，确认 §68 的 (R,O) 替代性 =====
  printf("\n===== §E 精确 R 下扫占用率 O（Q2 补充：寄存器预算 vs 占用率是否仍替代）=====\n");
  {
    int Rs[] = {32, 64};
    int Os[] = {1, 2, 4};
    for (int ri = 0; ri < 2; ++ri) {
      printf("  --- R=%d ---\n", Rs[ri]);
      for (int oi = 0; oi < 3; ++oi) {
        char tag[24]; snprintf(tag, 24, "E-R%d-O%d", Rs[ri], Os[oi]);
        line_mr(tag, Rs[ri], 256, Os[oi], 3);
      }
    }
  }

  // ===== §F 函数属性（numRegs / localB）与 occupancy API =====
  printf("\n===== §F 函数属性（numRegs / localB）与 occupancy API（T=256 基准）=====\n");
  {
    cudaFuncAttributes at;
#define OA_MR(RR)                                                                       \
    do {                                                                                \
      CK(cudaFuncGetAttributes(&at, mr_##RR));                                          \
      int nb = 0;                                                                       \
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mr_##RR, 256, 0));          \
      printf("  %-16s numRegs=%-4d localB=%-3zu -> blk/SM(T=256)=%-2d\n", "mr_" #RR,    \
             at.numRegs, at.localSizeBytes, nb);                                        \
    } while (0)
    OA_MR(32); OA_MR(36); OA_MR(40); OA_MR(44);
    OA_MR(48); OA_MR(64); OA_MR(96); OA_MR(128);
#undef OA_MR
#define OA_LB(MB_)                                                                      \
    do {                                                                                \
      CK(cudaFuncGetAttributes(&at, lb_##MB_));                                         \
      int nb = 0;                                                                       \
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, lb_##MB_, 256, 0));         \
      printf("  %-16s numRegs=%-4d localB=%-3zu -> blk/SM(T=256)=%-2d\n", "lb_" #MB_,   \
             at.numRegs, at.localSizeBytes, nb);                                        \
    } while (0)
    OA_LB(8); OA_LB(6); OA_LB(4); OA_LB(2);
#undef OA_LB
  }

  printf("\n########## §69 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) {
    bench();
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 6) {
      printf("usage: run <R> <T> <O> <reps>\n");
      return 1;
    }
    g_R = atoi(argv[2]);
    g_T = atoi(argv[3]);
    int O = atoi(argv[4]);
    int reps = atoi(argv[5]);
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    line_mr("run", g_R, g_T, O, reps);
    return 0;
  }
  printf("usage: maxnreg_decouple {bench|run ...}\n");
  return 1;
}
