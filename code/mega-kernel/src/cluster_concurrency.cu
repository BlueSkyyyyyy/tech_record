// ============================================================================
// P162-1 (§87)：cluster 跨 SM 交换的「并发度阈值 vs 远端写」——把 §86.7 的
// 「每线程多笔在飞远端读很脆弱」推进到「到底什么在脆弱、生产形态（远端写 + 本地读）
// 是否绕开它」。
//
// 背景：
//   - §18/§19（P29-1/P30-1）量了 DSM 远端读、TMA multicast；§86（P161-1）发现
//     **每线程多笔在飞（U 路）的 `ld.shared::cluster` 远端读 + cluster 在 grid≥~3×SM
//     （C=2）或任意 grid（C≥4）稳定报 `unspecified launch failure`**，而轻通道（1 笔在飞）
//     稳到 grid=1056；并给出缓解建议：「**跨 SM 交换必须低并发/批量/环形化**（MPK ring
//     barrier / DeepGEMM ring barrier）」。
//   - 但 §86 只给了「重（U=8）/轻（U=1）」两点，**没扫在飞笔数 `U` 本身**，也没验证
//     「批量/环形化」到底指哪种实现。生产 mega-kernel（DeepGEMM symmetric memory、
//     MPK 对称显存 rank 切片）实际用的是 **「远端写 + 本地读」**（生产者把 tile 写进
//     对端 SMEM，消费者读**本地** SMEM），而不是「每线程拉一堆远端读」。
//
// 本节两问（目标卡 H100 / sm90a；每配置单独进程，cluster launch failure 会 sticky）：
//   Q1 §A：脆弱性到底是不是「在飞笔数 `U`」的函数？固定 grid/C 扫 `U=1..16`，找稳定边界。
//   Q2 §B/§C：把「远端读」换成「**远端写 + 本地读**」（生产形态），稳定性与吞吐如何？
//   Q3 §D：同一 `U` 下读/写两条路的吞吐对照（在都稳的区间）。
//
// 用法：nvcc -O3 -arch=sm_90a src/cluster_concurrency.cu -o cluster_concurrency
//       cluster_concurrency A
//       cluster_concurrency R <C> <grid> <sbKB> <rounds> <U>      # 远端读 pull<U>
//       cluster_concurrency W <C> <grid> <sbKB> <rounds> <U>      # 远端写 push<U>
//       cluster_concurrency F <C> <grid> <sbKB> <U> <reps>        # 可行性探针（读）
//       cluster_concurrency G <C> <grid> <sbKB> <U> <reps>        # 可行性探针（写）
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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
  cudaError_t e = cudaLaunchKernelEx(&cfg, (cudaKernel_t)(void*)kern, args...);
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
    if (e != cudaSuccess) { printf("[iterFAIL:%s] ", cudaGetErrorString(e)); return -1; }
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}

// SMEM 布局：前 sb 个 float = snd（本 block 的源数据），后 sb 个 float = rcv（接收区）。
// 值 f(i, owner) = ((i*7 + owner*13) % 1000)，整数、float 精确可表。
__host__ __device__ __forceinline__ float fval(int i, int owner) {
  return (float)(((long long)i * 7 + (long long)owner * 13) % 1000);
}

// ---------- cluster 作用域的 release/acquire 标志访存（sm90+） ----------
// 用于「环形缓冲 + 每槽 release/done」：生产者写数据后 release-store 一个 per-thread
// 就绪标志，消费者 acquire-load 它再读数据；反向的 done 标志对称。二者都作用在
// `.shared::cluster` 状态空间、`.cluster` 作用域上，跨 SM 的 DSM 访问走同一 memory model。
// 用 volatile 访存 + cluster 作用域 fence 表达 release/acquire：写数据 → fence → 写标志；
// 读标志 → fence → 读数据。避免直接在 `.shared::cluster` 上用 st.release/ld.acquire
// （本机实测在 256 线程 × 远端共享地址下会间歇报 `unspecified launch failure`，见 §88.6）。
__device__ __forceinline__ unsigned ld_acq_cs(const unsigned* p) {
  return *(volatile const unsigned*)p;
}
__device__ __forceinline__ void st_rel_cs(unsigned* p, unsigned v) {
  asm volatile("fence.acq_rel.cluster;" ::: "memory");
  *(volatile unsigned*)p = v;
}
__device__ __forceinline__ void fence_cluster() {
  asm volatile("fence.acq_rel.cluster;" ::: "memory");
}

// ---------- 远端读：每线程 U 路在飞 `ld.shared::cluster` ----------
#define DEF_PULL(U)                                                                     \
  __global__ void pull_##U(float* __restrict__ out, int rounds, int sb) {               \
    extern __shared__ float sm[];                                                       \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x)                                  \
      snd[i] = fval(i, rank);                                                           \
    rcv[threadIdx.x] = 0.f;                                                             \
    cl.sync();                                                                          \
    float* src = cl.map_shared_rank(snd, (rank + 1) % C);                               \
    volatile float* vsrc = src;                                                         \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    long long acc = 0;                                                                  \
    int tid = threadIdx.x;                                                              \
    for (int r = 0; r < rounds; r++) {                                                  \
      int off = (r % tiles) * SEG;                                                      \
      float v[U];                                                                       \
      _Pragma("unroll") for (int u = 0; u < U; u++) v[u] = vsrc[off + u * TH + tid];    \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)v[u];             \
    }                                                                                   \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_PULL(1) DEF_PULL(2) DEF_PULL(4) DEF_PULL(8) DEF_PULL(16)
#undef DEF_PULL

// 同 pull，但外层轮次 `#pragma unroll 1` 串行化——用来验证「脆弱性来自编译器把多轮
// unroll 成多笔在飞远端读」的原假设（§86.7 的「边界随 body codegen 移动」）。
#define DEF_PULLS(U)                                                                     \
  __global__ void pulls_##U(float* __restrict__ out, int rounds, int sb) {               \
    extern __shared__ float sm[];                                                       \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x)                                  \
      snd[i] = fval(i, rank);                                                           \
    rcv[threadIdx.x] = 0.f;                                                             \
    cl.sync();                                                                          \
    float* src = cl.map_shared_rank(snd, (rank + 1) % C);                               \
    volatile float* vsrc = src;                                                         \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    long long acc = 0;                                                                  \
    int tid = threadIdx.x;                                                              \
    _Pragma("unroll 1") for (int r = 0; r < rounds; r++) {                              \
      int off = (r % tiles) * SEG;                                                      \
      float v[U];                                                                       \
      _Pragma("unroll") for (int u = 0; u < U; u++) v[u] = vsrc[off + u * TH + tid];    \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)v[u];             \
    }                                                                                   \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_PULLS(1) DEF_PULLS(2) DEF_PULLS(4) DEF_PULLS(8) DEF_PULLS(16)
#undef DEF_PULLS

// ---------- 远端写：每线程 U 路 `st.shared::cluster` 到对端 SMEM，再本地读 ----------
// 本 block 把 snd 的一段推给 (rank+1)%C 的 rcv；round 末 cluster.sync，然后本地读回累加。
// 与 pull 的远端字节数一致（每 block 每 round 推 SEG 个 float）。
#define DEF_PUSH(U)                                                                     \
  __global__ void push_##U(float* __restrict__ out, int rounds, int sb) {               \
    extern __shared__ float sm[];                                                       \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x)                                  \
      snd[i] = fval(i, rank);                                                           \
    rcv[threadIdx.x] = 0.f;                                                             \
    cl.sync();                                                                          \
    float* dst = cl.map_shared_rank(rcv, (rank + 1) % C);                               \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    long long acc = 0;                                                                  \
    int tid = threadIdx.x;                                                              \
    for (int r = 0; r < rounds; r++) {                                                  \
      int off = (r % tiles) * SEG;                                                      \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[off + u * TH + tid] = snd[off + u * TH + tid]; \
      cl.sync();                                                                        \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[off + u * TH + tid]; \
      cl.sync();                                                                        \
    }                                                                                   \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_PUSH(1) DEF_PUSH(2) DEF_PUSH(4) DEF_PUSH(8) DEF_PUSH(16)
#undef DEF_PUSH

// ---------- 环形缓冲 + 每槽 release/done：去掉每 round 的两次 `cluster.sync` ----------
// 与 push 同数据量/同远端字节数，但把「每 round 两次全 cluster 同步」换成「每线程每槽
// 一对 release/done 标志」（per-thread 通道）：
//   本 block 把 snd 的一段推给 (rank+1)%C 的 rcv 的第 j=r%S 个槽；
//   - 生产者（每个线程独立）写数据后 `st.release` 它的就绪标志 ready[j*TH+tid]=r+1；
//   - 消费者（同一线程）`ld.acquire` 本地 ready[j*TH+tid] 到 r+1 后读该槽；
//   - 读完 `st.release` 一个 done 标志 ack[j*TH+tid]=r+1 回生产者；
//   - 生产者复用槽 j 前 `ld.acquire` 本地 ack 到 r-S+1（环形深度 S 的背压）。
// 全程无 `__syncthreads`、无 `cluster.sync`（初始化后）——每线程是独立的 SPSC 通道。
// S = min(4, tiles)，S 越大流水越深、同步越摊薄。
#define DEF_RING(U)                                                                     \
  __global__ void ring_##U(float* __restrict__ out, int rounds, int sb) {               \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = tiles < 4 ? tiles : 4;                                                      \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    unsigned* ready = (unsigned*)(rcv + (size_t)S * SEG);                               \
    unsigned* ack = ready + (size_t)S * TH;                                             \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x) snd[i] = fval(i, rank);          \
    for (int i = threadIdx.x; i < S * TH; i += blockDim.x) { ready[i] = 0; ack[i] = 0; }\
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned* dstReady = (unsigned*)cl.map_shared_rank(ready, dstRank);                 \
    unsigned* ackTo = (unsigned*)cl.map_shared_rank(ack, srcRank);                      \
    unsigned* myReady = ready;   /* 本地标志：用本地指针 => 普通 ld.shared，避开 */      \
    unsigned* myAck = ack;       /* `ld.shared::cluster` 的脆弱路径（§86/§87） */       \
    int tid = threadIdx.x;                                                              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      if (r >= S) { while (ld_acq_cs(&myAck[j * TH + tid]) < (unsigned)(r - S + 1)) {} fence_cluster(); } \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      st_rel_cs(&dstReady[j * TH + tid], (unsigned)(r + 1));                            \
      while (ld_acq_cs(&myReady[j * TH + tid]) < (unsigned)(r + 1)) {} fence_cluster(); \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      st_rel_cs(&ackTo[j * TH + tid], (unsigned)(r + 1));                               \
    }                                                                                   \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_RING(1) DEF_RING(2) DEF_RING(4) DEF_RING(8) DEF_RING(16)
#undef DEF_RING

// ---------- 环形缓冲 + 单次 cluster.sync：去掉 push 每 round 的第二次同步 ----------
// push 每 round 两次 `cluster.sync`：一次「写完让邻居读到」，一次「读完防下轮覆写」。
// 但 push 的 `off=(r%tiles)*SEG` 本就把数据摊在 `tiles` 个槽上、循环复用；只要
// **槽位数 S>=2**，邻居下一 round 写的是**另一个槽**、要到 r+S 才回写本槽——而每 round
// 一次 `cluster.sync` 已把两块压在同一步，邻居不可能领先 S 轮。故**第二次同步是多余的**。
// ring1 = push - 第二次同步（SMEM/占用率/数据量/结果与 push 逐字相同，唯一变量=同步次数）。
#define DEF_RING1(U)                                                                    \
  __global__ void ring1_##U(float* __restrict__ out, int rounds, int sb) {              \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = tiles < 4 ? tiles : 4;                                                      \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    for (int i = threadIdx.x; i < sb; i += blockDim.x) snd[i] = fval(i, rank);          \
    rcv[threadIdx.x] = 0.f;                                                             \
    cl.sync();                                                                          \
    float* dst = (float*)cl.map_shared_rank(rcv, (rank + 1) % C);                       \
    int tid = threadIdx.x;                                                              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      cl.sync();                                                                        \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
    }                                                                                   \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_RING1(1) DEF_RING1(2) DEF_RING1(4) DEF_RING1(8) DEF_RING1(16)
#undef DEF_RING1

// ============================ host dispatch ============================
static void* pick_pull(int U) {
#define P(UU) if (U == UU) return (void*)pull_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_push(int U) {
#define P(UU) if (U == UU) return (void*)push_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_pulls(int U) {
#define P(UU) if (U == UU) return (void*)pulls_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_ring(int U) {
#define P(UU) if (U == UU) return (void*)ring_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_ring1(int U) {
#define P(UU) if (U == UU) return (void*)ring1_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}

// out[bx] 只存 thread0 的 acc；参考取 tid=0。
static void reference_t0(int grid, int C, int sb, int rounds, int U, int is_push,
                         long long* out_exp) {
  int SEG = TH * U, tiles = sb / SEG;
  for (int bx = 0; bx < grid; bx++) {
    int rank = bx % C;
    int owner = is_push ? ((rank - 1 + C) % C) : ((rank + 1) % C);
    long long acc = 0;
    for (int r = 0; r < rounds; r++) {
      int off = (r % tiles) * SEG;
      for (int u = 0; u < U; u++) acc += (long long)fval(off + u * TH + 0, owner);
    }
    out_exp[bx] = acc;
  }
}

static void check(int grid, int C, float* dout, int sb, int rounds, int U, int is_push) {
  float* o = (float*)malloc((size_t)grid * sizeof(float));
  CK(cudaMemcpy(o, dout, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
  long long* exp = (long long*)malloc((size_t)grid * sizeof(long long));
  reference_t0(grid, C, sb, rounds, U, is_push, exp);
  double maxerr = 0; int bad = 0;
  for (int g = 0; g < grid; g++) {
    double d = (double)o[g] - (double)exp[g];
    if (d < 0) d = -d;
    if (d > maxerr) maxerr = d;
    if (d > 1e-6) bad++;
  }
  printf("maxerr %g bad %d ", maxerr, bad);
  free(o); free(exp);
}

static int run(const char* tag, int C, int grid, int sbKB, int rounds, int U, int path) {
  int sb = sbKB * 1024 / 4;
  int SEG = TH * U;
  if (SEG > sb) { printf("%s C=%d grid=%d sb=%dKB U=%d | SKIP(SEG>sb)\n", tag, C, grid, sbKB, U); return 3; }
  int tiles = sb / SEG, S = tiles < 4 ? tiles : 4;
  size_t smem = path == 3
      ? (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * TH * 4   // snd + S*rcv + ready/ack
      : (size_t)2 * sb * 4;                                            // snd + rcv
  int is_push = (path == 1 || path == 3 || path == 4);
  void* fn = path == 1 ? pick_push(U) : (path == 2 ? pick_pulls(U)
             : (path == 3 ? pick_ring(U) : (path == 4 ? pick_ring1(U) : pick_pull(U))));
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, TH, smem));

  // 逐个 launch 计时、统计成功次数（连续 launch 会累积失败，故不丢弃部分结果）。
  const int N = 20;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 0; int ok = 0; cudaError_t last = cudaSuccess;
  // 正式计时：每次 launch 一个 event 对，取 min-of-N、统计成功次数。
#define FN(UU) (path == 1 ? (void*)push_##UU : (path == 2 ? (void*)pulls_##UU \
              : (path == 3 ? (void*)ring_##UU : (path == 4 ? (void*)ring1_##UU : (void*)pull_##UU))))
#define TIME(UU) if (U == UU) {                                                        \
    for (int i = 0; i < N; i++) {                                                      \
      cudaEventRecord(e0);                                                             \
      cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem,                    \
          FN(UU), out, rounds, sb);                                                    \
      cudaEventRecord(e1); cudaEventSynchronize(e1);                                   \
      if (e != cudaSuccess) { last = e; cudaGetLastError(); continue; }                \
      float ms = 0; cudaEventElapsedTime(&ms, e0, e1);                                 \
      if (ok == 0 || ms < best) best = ms;                                             \
      ok++;                                                                            \
    }                                                                                  \
  }
  TIME(1) TIME(2) TIME(4) TIME(8) TIME(16)
#undef TIME
#undef FN
  double t = ok ? best : -1;   // 取最快一次（min-of-N），符合本专题口径
  printf("%s C=%d grid=%4d sb=%3dKB U=%2d rounds=%3d regs=%d occ=%d | ok=%d/%d %s", tag, C,
         grid, sbKB, U, rounds, a.numRegs, occ, ok, N,
         t < 0 ? (last == cudaSuccess ? "FAILED\n" : "FAIL\n") : "");
  if (t >= 0) {
    double bytes = (double)grid * (long)rounds * SEG * 4.0;   // 远端字节
    check(grid, C, out, sb, rounds, U, is_push);
    printf("%9.3f us | %8.1f GB/s(远端) | %7.3f ns/lap\n", t * 1e3, bytes / t / 1e6,
           t * 1e6 / rounds);
  } else if (last != cudaSuccess) {
    printf("[%s]\n", cudaGetErrorString(last));
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(out);
  return ok > 0 ? 0 : 2;
}

// 可行性探针：每配置独立进程各 launch reps 次，报成功次数（规避 sticky 与并发 context）
static int probe(int C, int grid, int sbKB, int U, int reps, int path, int rounds) {
  int sb = sbKB * 1024 / 4;
  int SEG = TH * U, tiles = sb / SEG, S = tiles < 4 ? tiles : 4;
  size_t smem = path == 3
      ? (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * TH * 4
      : (size_t)2 * sb * 4;
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  int ok = 0;
  for (int i = 0; i < reps; i++) {
    cudaError_t e;
#define GO(UU) if (U == UU) e = launch_cluster(C, dim3(grid), dim3(TH), smem, \
    path == 1 ? (void*)push_##UU : (path == 2 ? (void*)pulls_##UU \
    : (path == 3 ? (void*)ring_##UU : (path == 4 ? (void*)ring1_##UU : (void*)pull_##UU))), \
    out, rounds, sb);
    GO(1) GO(2) GO(4) GO(8) GO(16)
#undef GO
    if (e == cudaSuccess) ok++;
    else cudaGetLastError();
  }
  printf("%s C=%d grid=%4d U=%2d rounds=%3d | %d/%d ok\n",
         path == 1 ? "W" : (path == 2 ? "S" : (path == 3 ? "N" : (path == 4 ? "M" : "R"))),
         C, grid, U, rounds, ok, reps);
  cudaFree(out);
  return reps - ok;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: cluster_concurrency <A|R|W|S|N|M|F|G|K|O|L> ...\n"); return 1; }
  const char* cmd = argv[1];
  // path：0=pull（远端读），1=push（远端写+本地读），2=pulls（远端读串行），
  //       3=ring（无 cluster.sync 的每槽 release/done），4=ring1（环形槽 + 单次 cluster.sync/round）
  int path = !strcmp(cmd, "W") ? 1 : (!strcmp(cmd, "S") ? 2 : (!strcmp(cmd, "G") ? 1
             : (!strcmp(cmd, "K") ? 2 : (!strcmp(cmd, "N") ? 3 : (!strcmp(cmd, "O") ? 3
             : (!strcmp(cmd, "M") ? 4 : (!strcmp(cmd, "L") ? 4 : 0)))))));
  const char* tag = path == 1 ? "W" : (path == 2 ? "S" : (path == 3 ? "N" : (path == 4 ? "M" : "R")));

  if (!strcmp(cmd, "A")) {
    printf("# §87 §A / §88 §A 资源表（T=%d，sb=32KB；pull/push/pulls SMEM=64KB，ring 见下）\n", TH);
    int sb = 8192;
    for (int Ui = 0; Ui < 5; Ui++) {
      int U = (int[]){1, 2, 4, 8, 16}[Ui];
      int SEG = TH * U, tiles = sb / SEG, S = tiles < 4 ? tiles : 4;
      size_t smem_r = (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * TH * 4;
      for (int P = 0; P < 5; P++) {
        size_t smem = P == 3 ? smem_r : (size_t)2 * sb * 4;
        void* fn = P == 0 ? pick_pull(U) : (P == 1 ? pick_push(U)
                 : (P == 2 ? pick_pulls(U) : (P == 3 ? pick_ring(U) : pick_ring1(U))));
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
        int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, TH, smem));
        printf("  %-5s U=%2d regs=%d localB=%zu occ=%d blk/SM%s\n",
               P == 0 ? "pull" : (P == 1 ? "push" : (P == 2 ? "pulls"
               : (P == 3 ? "ring" : "ring1"))), U,
               a.numRegs, (size_t)a.localSizeBytes, occ,
               P == 3 ? " (S=min(4,tiles))" : "");
      }
    }
    return 0;
  }
  if (!strcmp(cmd, "R") || !strcmp(cmd, "W") || !strcmp(cmd, "S") || !strcmp(cmd, "N")
      || !strcmp(cmd, "M")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sbKB = atoi(argv[4]);
    int rounds = atoi(argv[5]), U = atoi(argv[6]);
    return run(tag, C, grid, sbKB, rounds, U, path);
  }
  if (!strcmp(cmd, "F") || !strcmp(cmd, "G") || !strcmp(cmd, "K") || !strcmp(cmd, "O")
      || !strcmp(cmd, "L")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sbKB = atoi(argv[4]);
    int U = atoi(argv[5]), reps = atoi(argv[6]);
    int rounds = argc > 7 ? atoi(argv[7]) : 4;
    return probe(C, grid, sbKB, U, reps, path, rounds);
  }
  printf("unknown cmd %s\n", cmd);
  return 1;
}
