// ============================================================================
// P161-1 (§86)：`__maxnreg__` 寄存器预算 × Thread Block Cluster —— 常驻 megakernel
// 里「寄存器足迹」（§8/§68–§85 的占用率主题）与「跨 SM cluster 原语」（§18–§23）何时
// 相互作用。
//
// 背景：
//   - §18（P29-1）量了 DSM（远端 SMEM 读 / ring 段位移）与 cluster.sync，但**从未扫
//     占用率/寄存器预算**：全是默认寄存器、1 block/SM。
//   - §61/§68–§85（P73/P80–P97）证明「数据面吞吐认每 SM 在飞 load 数（占用率×ILP）」，
//     且 `__maxnreg__(R)` 可把每线程寄存器上限精确钉到 R（P81-1/§69）。
//   - 两者从未叠加。常驻 megakernel 恰恰既想把整机铺满（低寄存器高占用率）又用 cluster
//     传中间量——「寄存器预算会不会成为 cluster 交换吞吐的旋钮，甚至成为 cluster 能否
//     launch 的约束」是本专题一直缺的一格。
//
// 本节四问（目标卡 H100 / sm90a；每个配置单独进程，cluster launch failure 会 sticky）：
//   Q1 §A/§B：DSM 跨 SM 通道吞吐随寄存器预算 R（⇒占用率）如何变？像普通数据面那样吃
//             占用率，还是（因 1 block/SM）与 R 无关？
//   Q2 §C：`cluster.sync()` 的**延迟**随 R/占用率如何变？随 cluster 大小 C 如何变？
//   Q3 §D：**可行性**——低占用率（大 R）下 cluster 还能否 launch？C 与占用率是否联合受限？
//   Q4 §E：可绕开性——用 `__maxnreg__` 把寄存器压低换取更高占用率，能否收回/抬升 cluster
//             交换吞吐（对上 §61 的 `__launch_bounds__` 两等价路径）。
//
// 用法：nvcc -O3 -arch=sm_90a src/cluster_maxnreg.cu -o cluster_maxnreg
//       cluster_maxnreg A
//       cluster_maxnreg R <C> <grid> <sbKB> <iters> <Rv>
//       cluster_maxnreg S <C> <grid> <rounds> <Rv>
//       cluster_maxnreg C <C> <grid> <reps> <Rv>
//       cluster_maxnreg F <C> <grid> <T> <Rv>
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t _e = (x);                                                            \
    if (_e != cudaSuccess) {                                                         \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

static const int TH = 256;

template <typename K, typename... Args>
static cudaError_t launch_cluster(int C, dim3 grid, dim3 block, size_t smem,
                                  K kern, Args... args) {
  cudaFuncSetAttribute((const void*)kern,
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (smem > 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, args...);
  if (e != cudaSuccess) return e;
  return cudaDeviceSynchronize();
}

template <typename K, typename... Args>
static double time_cluster(int C, dim3 grid, dim3 block, size_t smem, int iters,
                           K kern, Args... args) {
  cudaError_t e = launch_cluster(C, grid, block, smem, kern, args...);
  if (e != cudaSuccess) { printf("[FAIL:%s] ", cudaGetErrorString(e)); return -1; }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    e = launch_cluster(C, grid, block, smem, kern, args...);
    if (e != cudaSuccess) { printf("[iter FAIL:%s] ", cudaGetErrorString(e)); return -1; }
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}

// ---------- DSM 远端读（延迟/发射受限；C=1 退化为读自己 SMEM）----------
// 每个线程每轮读邻居 SMEM 的一个 float；寄存器预算 R 由 __maxnreg__ 钉死。
#define DEF_READ(R)                                                                     \
  __global__ void __maxnreg__(R) read_##R(float* __restrict__ out, int R_iters,         \
                                          int sb) {                                     \
    extern __shared__ float sm[];                                                       \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x)                                  \
      sm[i] = (float)(i & 1023) * 1e-3f + rank;                                         \
    cl.sync();                                                                          \
    float* src = cl.map_shared_rank(sm, (rank + 1) % C);                                \
    volatile float* vsrc = src;                                                         \
    float acc = 0.f;                                                                    \
    for (int r = 0; r < R_iters; r++) acc += vsrc[threadIdx.x];                         \
    out[blockIdx.x] = acc;                                                              \
  }
DEF_READ(32) DEF_READ(40) DEF_READ(48) DEF_READ(64) DEF_READ(96)
DEF_READ(128) DEF_READ(168) DEF_READ(232) DEF_READ(255)
#undef DEF_READ

// ---------- DSM 远端读「重」变体：U 路在飞远程读（自然寄存器高，可被 __maxnreg__ 压）----------
// 真实 megakernel 的 task 身体远比 §A 的轻读重；用它检验「寄存器预算 R 压到自然需求以下
// → 占用率上升 → DSM 聚合吞吐是否随之变」。
#ifndef READ_ILP
#define READ_ILP 8
#endif
#define DEF_HREAD(R)                                                                    \
  __global__ void __maxnreg__(R) hread_##R(float* __restrict__ out, int R_iters,        \
                                           int sb) {                                     \
    extern __shared__ float sm[];                                                       \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x)                                  \
      sm[i] = (float)(i & 1023) * 1e-3f + rank;                                         \
    cl.sync();                                                                          \
    float* src = cl.map_shared_rank(sm, (rank + 1) % C);                                \
    volatile float* vsrc = src;                                                         \
    float acc = 0.f;                                                                    \
    int tid = threadIdx.x;                                                              \
    int mask = sb - 1;                                                                  \
    for (int r = 0; r < R_iters; r += READ_ILP) {                                       \
      float v[READ_ILP];                                                                \
      int b = (r * TH + tid) & mask;                                                    \
      _Pragma("unroll") for (int u = 0; u < READ_ILP; u++)                             \
          v[u] = vsrc[(b + u * TH) & mask];                                             \
      _Pragma("unroll") for (int u = 0; u < READ_ILP; u++) acc += v[u];                 \
    }                                                                                   \
    out[blockIdx.x] = acc;                                                              \
  }
DEF_HREAD(32) DEF_HREAD(40) DEF_HREAD(48) DEF_HREAD(64) DEF_HREAD(96)
DEF_HREAD(128) DEF_HREAD(168) DEF_HREAD(232) DEF_HREAD(255)
#undef DEF_HREAD

// ---------- 对照：同样的重 DSM 读，但**不挂 `__maxnreg__`**（自然寄存器 48）----------
__global__ void hreadp(float* __restrict__ out, int R_iters, int sb) {
  extern __shared__ float sm[];
  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  for (int i = threadIdx.x; i < sb; i += blockDim.x)
    sm[i] = (float)(i & 1023) * 1e-3f + rank;
  cl.sync();
  float* src = cl.map_shared_rank(sm, (rank + 1) % C);
  volatile float* vsrc = src;
  float acc = 0.f;
  int tid = threadIdx.x;
  int mask = sb - 1;
  for (int r = 0; r < R_iters; r += READ_ILP) {
    float v[READ_ILP];
    int b = (r * TH + tid) & mask;
    _Pragma("unroll") for (int u = 0; u < READ_ILP; u++)
        v[u] = vsrc[(b + u * TH) & mask];
    _Pragma("unroll") for (int u = 0; u < READ_ILP; u++) acc += v[u];
  }
  out[blockIdx.x] = acc;
}

// ---------- DSM ring 段位移（带宽受限；K=4 固定；ping-pong 双缓冲）----------
#define DEF_SHIFT(R)                                                                    \
  __global__ void __maxnreg__(R) shift_##R(float* __restrict__ out, int R_rounds) {     \
    const int K = 4;                                                                    \
    const int seg = K * TH;                                                             \
    extern __shared__ float sm[];                                                       \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    long gid = blockIdx.x;                                                              \
    float* buf[2];                                                                      \
    float* nbuf[2];                                                                     \
    buf[0] = sm; buf[1] = sm + seg;                                                     \
    float* base = cl.map_shared_rank(sm, (rank + C - 1) % C);                           \
    nbuf[0] = base; nbuf[1] = base + seg;                                               \
    for (int k = 0; k < K; k++)                                                         \
      buf[0][threadIdx.x + k * TH] = (float)(gid * 1000 + threadIdx.x + k * TH);        \
    cl.sync();                                                                          \
    for (int r = 0; r < R_rounds; r++) {                                                \
      int cur = r & 1;                                                                  \
      for (int k = 0; k < K; k++)                                                       \
        buf[cur ^ 1][threadIdx.x + k * TH] = nbuf[cur][threadIdx.x + k * TH];           \
      cl.sync();                                                                        \
    }                                                                                   \
    if (threadIdx.x == 0) out[gid] = buf[R_rounds & 1][0];                              \
  }
DEF_SHIFT(32) DEF_SHIFT(40) DEF_SHIFT(48) DEF_SHIFT(64) DEF_SHIFT(96)
DEF_SHIFT(128) DEF_SHIFT(168) DEF_SHIFT(232) DEF_SHIFT(255)
#undef DEF_SHIFT

// ---------- cluster.sync() 成本 ----------
#define DEF_CSYNC(R)                                                                    \
  __global__ void __maxnreg__(R) csync_##R(int reps) {                                  \
    cg::cluster_group cl = cg::this_cluster();                                          \
    for (int r = 0; r < reps; r++) cl.sync();                                           \
  }
DEF_CSYNC(32) DEF_CSYNC(40) DEF_CSYNC(48) DEF_CSYNC(64) DEF_CSYNC(96)
DEF_CSYNC(128) DEF_CSYNC(168) DEF_CSYNC(232) DEF_CSYNC(255)
#undef DEF_CSYNC

// ============================ host dispatch ============================
static int g_R = 32;

static void* pick_read(int R) {
#define P(RV) if (R == RV) return (void*)read_##RV;
  P(32) P(40) P(48) P(64) P(96) P(128) P(168) P(232) P(255)
#undef P
  return nullptr;
}
static void* pick_hread(int R) {
#define P(RV) if (R == RV) return (void*)hread_##RV;
  P(32) P(40) P(48) P(64) P(96) P(128) P(168) P(232) P(255)
#undef P
  return nullptr;
}
static void* pick_shift(int R) {
#define P(RV) if (R == RV) return (void*)shift_##RV;
  P(32) P(40) P(48) P(64) P(96) P(128) P(168) P(232) P(255)
#undef P
  return nullptr;
}
static void* pick_csync(int R) {
#define P(RV) if (R == RV) return (void*)csync_##RV;
  P(32) P(40) P(48) P(64) P(96) P(128) P(168) P(232) P(255)
#undef P
  return nullptr;
}

static int res_row(const char* tag, void* fn, int T, size_t smem, double* out_occ) {
  cudaFuncAttributes a;
  CK(cudaFuncGetAttributes(&a, fn));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, T, smem));
  if (out_occ) *out_occ = occ;
  printf("  %-8s R=%-3d numRegs=%-3d localB=%-4zu occ=%d blk/SM (T=%d, SMEM=%zu B)\n",
         tag, g_R, a.numRegs, (size_t)a.localSizeBytes, occ, T, smem);
  return a.numRegs;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: cluster_maxnreg <A|R|S|C|F> ...\n"); return 1; }
  const char* cmd = argv[1];
  int dev = 0; cudaGetDevice(&dev);

  if (!strcmp(cmd, "A")) {
    printf("# §86 §A 资源表：__maxnreg__(R) → numRegs / 占用率（T=%d）\n", TH);
    printf("# DSM read 载体 SMEM=16KB/block；shift 载体 SMEM=8KB/block\n");
    int Rs[] = {32, 40, 48, 64, 96, 128, 168, 232, 255};
    for (int Ri = 0; Ri < 9; Ri++) {
      g_R = Rs[Ri];
      res_row("read", pick_read(g_R), TH, 4096 * 4, nullptr);
      res_row("hread", pick_hread(g_R), TH, 4096 * 4, nullptr);
      res_row("hreadp", (void*)hreadp, TH, 4096 * 4, nullptr);
      res_row("shift", pick_shift(g_R), TH, 2 * 4 * TH * 4, nullptr);
      res_row("csync", pick_csync(g_R), TH, 0, nullptr);
    }
    return 0;
  }

  if (!strcmp(cmd, "R")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sb = atoi(argv[4]);
    int iters = atoi(argv[5]); g_R = atoi(argv[6]);
    float* out; CK(cudaMalloc(&out, grid * sizeof(float)));
    size_t smem = (size_t)sb * 4;
    res_row("read", pick_read(g_R), TH, smem, nullptr);
    double t = -1;
#define RC(RV) if (g_R == RV) t = time_cluster(C, dim3(grid), dim3(TH), smem, 20, read_##RV, out, iters, sb);
    RC(32) RC(40) RC(48) RC(64) RC(96) RC(128) RC(168) RC(232) RC(255)
#undef RC
    printf("R C=%2d grid=%4d sb=%3dKB R=%3d | %s", C, grid, sb * 4 / 1024, g_R,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) {
      double bytes = (double)grid * TH * 4.0 * iters;   // 每线程每轮读 4B
      printf("%9.3f us/launch | %8.1f GB/s(远程读) | %7.3f ns/lap\n",
             t * 1e3, bytes / t / 1e3, t * 1e6 / iters);
    }
    cudaFree(out);
    return t < 0 ? 2 : 0;
  }

  if (!strcmp(cmd, "H")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sb = atoi(argv[4]);
    int iters = atoi(argv[5]); g_R = atoi(argv[6]);
    float* out; CK(cudaMalloc(&out, grid * sizeof(float)));
    size_t smem = (size_t)sb * 4;
    res_row("hread", pick_hread(g_R), TH, smem, nullptr);
    double t = -1;
#define HC(RV) if (g_R == RV) t = time_cluster(C, dim3(grid), dim3(TH), smem, 20, hread_##RV, out, iters, sb);
    HC(32) HC(40) HC(48) HC(64) HC(96) HC(128) HC(168) HC(232) HC(255)
#undef HC
    printf("H C=%2d grid=%4d sb=%3dKB R=%3d | %s", C, grid, sb * 4 / 1024, g_R,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) {
      double bytes = (double)grid * TH * 4.0 * iters;
      printf("%9.3f us/launch | %8.1f GB/s(远程读) | %7.3f ns/lap\n",
             t * 1e3, bytes / t / 1e3, t * 1e6 / iters);
    }
    cudaFree(out);
    return t < 0 ? 2 : 0;
  }

  if (!strcmp(cmd, "P")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sb = atoi(argv[4]);
    int iters = atoi(argv[5]);
    float* out; CK(cudaMalloc(&out, grid * sizeof(float)));
    size_t smem = (size_t)sb * 4;
    res_row("hreadp", (void*)hreadp, TH, smem, nullptr);
    double t = time_cluster(C, dim3(grid), dim3(TH), smem, 20, hreadp, out, iters, sb);
    printf("P C=%2d grid=%4d sb=%3dKB | %s", C, grid, sb * 4 / 1024,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) {
      double bytes = (double)grid * TH * 4.0 * iters;
      printf("%9.3f us/launch | %8.1f GB/s(远程读) | %7.3f ns/lap\n",
             t * 1e3, bytes / t / 1e3, t * 1e6 / iters);
    }
    cudaFree(out);
    return t < 0 ? 2 : 0;
  }

  if (!strcmp(cmd, "S")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), rounds = atoi(argv[4]);
    g_R = atoi(argv[5]);
    float* out; CK(cudaMalloc(&out, grid * sizeof(float)));
    const int K = 4, seg = K * TH;
    size_t smem = (size_t)2 * seg * 4;
    double t = -1;
#define SC(RV) if (g_R == RV) t = time_cluster(C, dim3(grid), dim3(TH), smem, 20, shift_##RV, out, rounds);
    SC(32) SC(40) SC(48) SC(64) SC(96) SC(128) SC(168) SC(232) SC(255)
#undef SC
    printf("S C=%2d grid=%4d seg=%3dKB R=%3d | %s", C, grid, seg * 4 / 1024, g_R,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) {
      double bytes = (double)grid * seg * 4.0 * 2.0 * rounds;  // 读+写整段
      printf("%9.3f us/launch | %8.1f GB/s(DSM段位移)\n", t * 1e3, bytes / t / 1e3);
    }
    cudaFree(out);
    return t < 0 ? 2 : 0;
  }

  if (!strcmp(cmd, "C")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), reps = atoi(argv[4]);
    g_R = atoi(argv[5]);
    double t = -1;
#define CC(RV) if (g_R == RV) t = time_cluster(C, dim3(grid), dim3(TH), 0, 20, csync_##RV, reps);
    CC(32) CC(40) CC(48) CC(64) CC(96) CC(128) CC(168) CC(232) CC(255)
#undef CC
    printf("C C=%2d grid=%4d R=%3d | %s", C, grid, g_R, t < 0 ? "FAILED\n" : "");
    if (t >= 0) printf("cluster.sync %7.1f ns/sync\n", t * 1e6 / reps);
    return t < 0 ? 2 : 0;
  }

  if (!strcmp(cmd, "F")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), T = atoi(argv[4]);
    g_R = atoi(argv[5]);
    int sb = 4096;
    float* out; CK(cudaMalloc(&out, grid * sizeof(float)));
    size_t smem = (size_t)sb * 4;
    cudaError_t e;
#define FC(RV) if (g_R == RV) { e = launch_cluster(C, dim3(grid), dim3(T), smem, hread_##RV, out, 32, sb); }
    FC(32) FC(40) FC(48) FC(64) FC(96) FC(128) FC(168) FC(232) FC(255)
#undef FC
    res_row("hread", pick_hread(g_R), T, smem, nullptr);
    printf("F C=%2d grid=%4d T=%4d R=%3d SMEM=%zuKB | %s\n", C, grid, T, g_R,
           smem / 1024, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    cudaFree(out);
    return 0;
  }

  printf("unknown cmd %s\n", cmd);
  return 1;
}
