// ============================================================================
// P80-1 (§68)：把「寄存器预算」与「占用率」解耦——补 §67.4 明留的尾巴
//
// 背景（接 §60 / §61 / §67 / §72）：
//   - §67 把 `MINB`（`__launch_bounds__(TB, MINB)` 的「每 SM 最小 block 数」）当**寄存器预算旋钮**：
//     `MINB=8/6/4 → regs ≤ 32/42/64`（T=256）。它量出「`MINB` 8→4 让 plain 数据面
//     G≤132 提速 ~1.6–2.4×」，并**推断**主导量是「寄存器预算 → 数据面 ILP」，而不是那 8 B 溢出。
//   - 但 §67.4 明留尾巴（原话）：**「`MINB` 同时改**寄存器预算**与**占用率**两个物理量，本轮
//     **未把二者解耦**（如用 `__maxnreg__` 单独压寄存器、或 `T` 单独改占用率）；「ILP 主导」是
//     排除法 + 推断，非单变量隔离。」**——因为 `MINB` 一降，寄存器上限抬高的**同时**每 SM 驻留
//     block 上限也变（reg-limited occupancy = 65536/(regs·T)），两者被同一个旋钮绑死。
//
// 本节把二者拆开：
//   - **寄存器预算**：仍用 `MINB∈{8,6,4,2}`（32/42/64/128 regs 上限，编译期 template）。
//   - **占用率（resident blocks/SM）**：**用 grid 大小钉死**——只发 `numSM·O` 个 block（每个 SM
//     恰好拿到 O 个），于是「实际驻留 blocks/SM = min(O, reg-limit)」。只要 `O ≤ reg-limit(MINB)`，
//     占用率就**只由 O 决定**，与寄存器预算无关。
//   再加一个**独立**的占用率控制（SMEM padding：用动态共享内存把每 SM 可驻留块数压到 O，
//   对照 grid 钉死），确认两种占用率口径一致。
//
// 于是可以分别回答：
//   Q1（§B，固定占用率 O，只扫寄存器预算 MINB）：**同占用率下，多给寄存器能否让数据面更快？**
//      若能 ⇒「寄存器预算 → 每条线程的在飞 load 数（ILP）」是独立的一阶变量，不是占用率的代理。
//   Q2（§C，固定寄存器预算 MINB，只扫占用率 O）：**同寄存器预算下，占用率影响多大？**
//   Q3（§D，SMEM-pad 钉占用率 vs grid 钉占用率）：两种口径是否给出同一吞吐。
//
// 载体（编译期 template）：0=寄存器 `__ldcs` uint4 + U 路 ILP（§60/§61 的数据面）；
//                          1=`cp.async` 单缓冲进 SMEM → 折叠（MPK worker 载体的寄存器省档，§65/§67）。
// 用法：./reg_occ_decouple bench
//       ./reg_occ_decouple run <O> <MINB> <U> <carrier> <reps> [padKB] [DC]
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

#define MAXT 256
#define TB 256

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---- 载体 0：寄存器 `__ldcs` uint4 + U 路 ILP（= §60/§61 的数据面）----
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

// ---- 载体 1：16 B `cp.async` 单缓冲进 SMEM → 折叠（MPK worker 原生载体，§65）----
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
  uint32_t sp = (uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, 16;\n" ::"r"(sp),
               "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ unsigned long long read_ca(const uint4* __restrict__ base,
                                                      long long BP, int tid, int T,
                                                      char* smem, int DC) {
  int nchunk = (int)((BP + DC - 1) / DC);
  unsigned long long part = 0;
  for (int c = 0; c < nchunk; ++c) {
    long long off = (long long)c * DC;
    int n = (int)((BP - off < DC) ? (BP - off) : DC);
    char* dst = smem;
    const char* src = reinterpret_cast<const char*>(base + off);
    for (int b = tid; b < n; b += T) cp_async16(dst + (size_t)b * 16, src + (size_t)b * 16);
    cp_commit();
    cp_wait0();
    __syncthreads();
    const uint4* b4 = reinterpret_cast<const uint4*>(smem);
    unsigned long long acc = 0;
#pragma unroll 4
    for (int j = tid; j < n; j += T) {
      uint4 v = b4[j];
      acc += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
    }
    part += acc;
    __syncthreads();
  }
  return part;
}

template <int U, int MINB, int CARRIER>
__global__ void __launch_bounds__(TB, MINB)
    ro_kernel(int NBLK, long long BP, int DC, const uint4* __restrict__ data,
              unsigned long long* __restrict__ sink) {
  extern __shared__ __align__(16) char smem[];
  __shared__ unsigned long long red[MAXT];
  const int tid = threadIdx.x;
  const int T = blockDim.x;
  const uint4* base = data + (long long)blockIdx.x * BP;
  unsigned long long part;
  if constexpr (CARRIER == 0)
    part = read_reg<U>(base, BP, tid, T);
  else
    part = read_ca(base, BP, tid, T, smem, DC);
  red[tid] = part;
  __syncthreads();
  for (int s = T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(sink, red[0]);
  (void)NBLK;
}

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
static int g_smem_sm = 0;

static void ensure(long long data_u4) {
  if (data_u4 > g_data_cap) {
    if (g_data) CK(cudaFree(g_data));
    CK(cudaMalloc(&g_data, (size_t)data_u4 * 16));
    g_data_cap = data_u4;
  }
  if (!g_sink) CK(cudaMalloc(&g_sink, 8));
  if (!g_dref) CK(cudaMalloc(&g_dref, 8));
}

static int g_U = 4, g_MINB = 8, g_carrier = 0, g_DC = 2048;
static int g_last_ok = 1;

static void launch(int grid, size_t req, int NBLK, long long BP) {
#define L(UU, MB, CC)                                                                   \
  do {                                                                                  \
    CK(cudaFuncSetAttribute(ro_kernel<UU, MB, CC>,                                      \
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));  \
    ro_kernel<UU, MB, CC><<<grid, TB, req>>>(NBLK, BP, g_DC, g_data, g_sink);           \
    g_last_ok = (cudaGetLastError() == cudaSuccess);                                    \
  } while (0)

#define TRY(UU, MB, CC)                                                                 \
  if (g_U == UU && g_MINB == MB && g_carrier == CC) {                                   \
    L(UU, MB, CC);                                                                      \
    return;                                                                             \
  }
  TRY(1, 8, 0); TRY(2, 8, 0); TRY(4, 8, 0); TRY(8, 8, 0);
  TRY(1, 6, 0); TRY(2, 6, 0); TRY(4, 6, 0); TRY(8, 6, 0);
  TRY(1, 4, 0); TRY(2, 4, 0); TRY(4, 4, 0); TRY(8, 4, 0);
  TRY(1, 2, 0); TRY(2, 2, 0); TRY(4, 2, 0); TRY(8, 2, 0);
  TRY(1, 8, 1); TRY(2, 8, 1); TRY(4, 8, 1); TRY(8, 8, 1);
  TRY(1, 6, 1); TRY(2, 6, 1); TRY(4, 6, 1); TRY(8, 6, 1);
  TRY(1, 4, 1); TRY(2, 4, 1); TRY(4, 4, 1); TRY(8, 4, 1);
  TRY(1, 2, 1); TRY(2, 2, 1); TRY(4, 2, 1); TRY(8, 2, 1);
#undef TRY
  L(4, 8, 0);  // 兜底
#undef L
}

struct Res {
  double ms;
  unsigned long long sink;
  int unsupported;
};

static Res run(int O, int padKB, int reps) {
  int grid = g_numSm * O;
  long long BP = 134217728LL / grid;  // 总读 2 GiB / grid
  ensure((long long)grid * BP);
  size_t req = 0;
  if (g_carrier == 1) req = (size_t)g_DC * 16 + 64;
  req += (size_t)padKB * 1024;
  double best = 1e30;
  int unsupported = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaEventRecord(e0));
    launch(grid, req, grid, BP);
    if (!g_last_ok) { unsupported = 1; break; }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) best = ms;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  Res res; res.ms = best; res.sink = 0; res.unsupported = unsupported;
  if (!unsupported) CK(cudaMemcpy(&res.sink, g_sink, 8, cudaMemcpyDeviceToHost));
  return res;
}

// 用 occupancy API 报告某 (U,MINB,carrier,req) 的 blk/SM 上限
static int occ_blocks(size_t req) {
  int nb = 0;
  cudaError_t e = cudaErrorInvalidValue;
#define Q(UU, MB, CC)                                                                   \
  if (g_U == UU && g_MINB == MB && g_carrier == CC)                                     \
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ro_kernel<UU, MB, CC>, TB,   \
                                                      req);
  Q(1, 8, 0) Q(2, 8, 0) Q(4, 8, 0) Q(8, 8, 0)
  Q(1, 6, 0) Q(2, 6, 0) Q(4, 6, 0) Q(8, 6, 0)
  Q(1, 4, 0) Q(2, 4, 0) Q(4, 4, 0) Q(8, 4, 0)
  Q(1, 2, 0) Q(2, 2, 0) Q(4, 2, 0) Q(8, 2, 0)
  Q(1, 8, 1) Q(2, 8, 1) Q(4, 8, 1) Q(8, 8, 1)
  Q(1, 6, 1) Q(2, 6, 1) Q(4, 6, 1) Q(8, 6, 1)
  Q(1, 4, 1) Q(2, 4, 1) Q(4, 4, 1) Q(8, 4, 1)
  Q(1, 2, 1) Q(2, 2, 1) Q(4, 2, 1) Q(8, 2, 1)
#undef Q
  if (e != cudaSuccess) return -1;
  return nb;
}

static const char* cname() { return g_carrier == 0 ? "reg" : "ca"; }

static void line(const char* tag, int O, int padKB, int reps) {
  Res r = run(O, padKB, reps);
  double read_bytes = 134217728.0 * 16.0;  // 2 GiB
  double gbs = read_bytes / (r.ms * 1e-3) / 1e9;
  size_t req = 0;
  if (g_carrier == 1) req = (size_t)g_DC * 16 + 64;
  req += (size_t)padKB * 1024;
  int lim = occ_blocks(req);
  int grid = g_numSm * O;
  int resident = (lim > 0 && O > lim) ? lim : O;
  const char* verdict = r.unsupported ? "UNSUP" : "OK";
  printf("  %-16s O=%-2d pad=%-4dKB grid=%-5d blk/SM(API)=%-2d resident=%-2d MINB=%d "
         "U=%d %-3s -> %-5s ms=%.4f GB/s=%7.1f\n",
         tag, O, padKB, grid, lim, resident, g_MINB, g_U, cname(), verdict, r.ms, gbs);
}

// ============================================================================
// bench
// ============================================================================
static void bench() {
  CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&g_smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));

  printf("########################################################################\n");
  printf("# §68 寄存器预算 vs 占用率 解耦：网格钉占用率 × MINB 钉寄存器预算\n");
  printf("# 目标卡 H100/sm90a；numSM=%d，smem/SM=%d B；总读 2 GiB = grid·BP（uint4=16B）\n",
         g_numSm, g_smem_sm);
  printf("# 占用率用 grid=numSM·O 钉死（每 SM 恰好 O 块），寄存器预算用 MINB∈{8,6,4,2}\n");
  printf("########################################################################\n");

  // ===== §A 正确性：两载体折叠逐位对 host 参考 =====
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
    printf("\n===== §A 正确性（NBLK=%d，BP=%lld）：两载体 vs host 参考 =====\n", NBLK, BP);
    // 直接用一个 grid=NBLK、BP 的 mini 运行（绕过 bench 的 2GiB 口径）
    g_U = 4; g_MINB = 8; g_carrier = 0;
    {
      int grid = NBLK; size_t req = 0;
      CK(cudaMemset(g_sink, 0, 8));
      launch(grid, req, grid, BP);
      CK(cudaDeviceSynchronize());
      unsigned long long s = 0; CK(cudaMemcpy(&s, g_sink, 8, cudaMemcpyDeviceToHost));
      printf("  reg U=4 : sink=%llu ref=%llu -> %s\n", s, href, s == href ? "OK" : "WRONG");
    }
    g_carrier = 1; g_DC = 2048;
    {
      int grid = NBLK; size_t req = (size_t)g_DC * 16 + 64;
      CK(cudaMemset(g_sink, 0, 8));
      launch(grid, req, grid, BP);
      CK(cudaDeviceSynchronize());
      unsigned long long s = 0; CK(cudaMemcpy(&s, g_sink, 8, cudaMemcpyDeviceToHost));
      printf("  ca  DC=2048: sink=%llu ref=%llu -> %s\n", s, href, s == href ? "OK" : "WRONG");
    }
    g_carrier = 0; g_DC = 2048;
  }

  // ===== §B 固定占用率，扫寄存器预算（headline Q1）=====
  printf("\n===== §B 固定占用率 O，只扫寄存器预算 MINB（U=4，reg 载体）=====\n");
  printf("  (每个 O 一行：grid=numSM·O 不变，只改 MINB → 寄存器上限 32/42/64/128)\n");
  {
    int Os[] = {1, 2, 4};
    int mbs[] = {8, 6, 4, 2};
    for (int oi = 0; oi < 3; ++oi) {
      printf("  --- O=%d ---\n", Os[oi]);
      for (int mi = 0; mi < 4; ++mi) {
        g_MINB = mbs[mi]; g_U = 4; g_carrier = 0;
        char tag[24]; snprintf(tag, 24, "B-O%d-MINB%d", Os[oi], mbs[mi]);
        line(tag, Os[oi], 0, 3);
      }
    }
    g_MINB = 8;
  }

  // ===== §C 固定寄存器预算，扫占用率（Q2）=====
  printf("\n===== §C 固定寄存器预算 MINB，只扫占用率 O（U=4，reg 载体）=====\n");
  {
    int mbs[] = {8, 4};
    for (int mi = 0; mi < 2; ++mi) {
      int mb = mbs[mi];
      g_MINB = mb; g_U = 4; g_carrier = 0;
      printf("  --- MINB=%d（regs ≤ %d）---\n", mb, 65536 / (mb * TB));
      int Os[] = {1, 2, 4, 8};
      for (int oi = 0; oi < 4; ++oi) {
        char tag[24]; snprintf(tag, 24, "C-MINB%d-O%d", mb, Os[oi]);
        line(tag, Os[oi], 0, 3);
      }
    }
    g_MINB = 8;
  }

  // ===== §D SMEM-pad 钉占用率 vs grid 钉占用率（Q3 交叉验证）=====
  printf("\n===== §D 用 SMEM padding 钉占用率（grid=numSM·8 固定）vs §B 的 grid 钉法 =====\n");
  printf("  (MINB=8、U=4、reg 载体；pad 越大 regfile 越富余，但实际驻留块数被 SMEM 限住)\n");
  {
    g_MINB = 8; g_U = 4; g_carrier = 0;
    // smem_sm ≈ 228KB usable；pad 使每 block 占 smem 从而限住 blk/SM
    struct { int padKB; int Ohat; } pads[] = {{0, 8}, {24, 4}, {56, 3}, {96, 2}, {160, 1}};
    for (int i = 0; i < 5; ++i) {
      char tag[24]; snprintf(tag, 24, "D-pad%d", pads[i].padKB);
      line(tag, 8, pads[i].padKB, 3);
    }
  }

  // ===== §E 载体交叉验证：SMEM 载体（cp.async）对寄存器预算是否免疫 =====
  printf("\n===== §E 载体交叉验证：ca（cp.async 单缓冲）在固定 O 下扫 MINB =====\n");
  {
    int Os[] = {2, 4};
    int mbs[] = {8, 4};
    for (int oi = 0; oi < 2; ++oi) {
      printf("  --- O=%d ---\n", Os[oi]);
      for (int mi = 0; mi < 2; ++mi) {
        g_MINB = mbs[mi]; g_U = 1; g_carrier = 1; g_DC = 2048;
        char tag[24]; snprintf(tag, 24, "E-O%d-MINB%d", Os[oi], mbs[mi]);
        line(tag, Os[oi], 0, 3);
      }
    }
    g_MINB = 8; g_carrier = 0;
  }

  // ===== §F 函数属性/占用率报告 =====
  printf("\n===== §F 函数属性（numRegs / localB）与 occupancy API =====\n");
  {
    cudaFuncAttributes at;
#define OA(UU, MB, CC, LAB)                                                             \
    do {                                                                                \
      CK(cudaFuncGetAttributes(&at, ro_kernel<UU, MB, CC>));                            \
      int nb = 0;                                                                       \
      size_t req = (CC == 1) ? (size_t)2048 * 16 + 64 : 0;                              \
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ro_kernel<UU, MB, CC>, TB,  \
                                                       req));                           \
      printf("  %-22s numRegs=%-4d localB=%-3zu -> blk/SM=%-2d (req=%zu)\n", LAB,       \
             at.numRegs, at.localSizeBytes, nb, req);                                   \
    } while (0)
    OA(4, 8, 0, "reg U=4 MINB=8");
    OA(4, 6, 0, "reg U=4 MINB=6");
    OA(4, 4, 0, "reg U=4 MINB=4");
    OA(4, 2, 0, "reg U=4 MINB=2");
    OA(8, 8, 0, "reg U=8 MINB=8");
    OA(8, 4, 0, "reg U=8 MINB=4");
    OA(1, 8, 1, "ca  U=1 MINB=8");
    OA(1, 4, 1, "ca  U=1 MINB=4");
#undef OA
  }

  printf("\n########## §68 完成 ##########\n");
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (argc >= 2 && !strcmp(argv[1], "bench")) {
    bench();
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "run")) {
    if (argc < 7) {
      printf("usage: run <O> <MINB> <U> <carrier> <reps> [padKB] [DC]\n");
      return 1;
    }
    int O = atoi(argv[2]);
    g_MINB = atoi(argv[3]);
    g_U = atoi(argv[4]);
    g_carrier = atoi(argv[5]);
    int reps = atoi(argv[6]);
    int padKB = (argc >= 8) ? atoi(argv[7]) : 0;
    if (argc >= 9) g_DC = atoi(argv[8]);
    CK(cudaDeviceGetAttribute(&g_numSm, cudaDevAttrMultiProcessorCount, 0));
    line("run", O, padKB, reps);
    return 0;
  }
  printf("usage: reg_occ_decouple {bench|run ...}\n");
  return 1;
}
