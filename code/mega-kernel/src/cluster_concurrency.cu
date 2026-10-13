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
// P172-1（§97）：path 10 = mtex（每槽 done 由 TMA 事务计数 expect_tx 表达）：
//       cluster_concurrency D <C> <grid> <sbKB> <rounds> <U> [S]  # mtex（计时）
//       cluster_concurrency E <C> <grid> <sbKB> <U> <reps> [r S]  # 可行性探针（mtex）
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

// P167-1（§92）：把每 block 线程数 `TH` 提升为**编译期可覆盖**的宏（`-DTH=...`），
// 以便扫「每 warp 一次 arrive」（`mbarw`）形态对每 block warp 数 `NW=TH/32` 的依赖
// （§91.8 明留的尾巴：`mbarw`/`mbarw0` 只在 T=256/NW=8 上测过）。默认仍是 256。
#ifndef TH
#define TH 256
#endif

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

// ---------- 环形槽 + 每槽 mbarrier release/done：把 §88.5 的纯 volatile 握手换成硬件原语 ----------
// §88.5 的 `ring`（纯 volatile + fence 的每槽 release/done）在 H100 DSM 上间歇失稳、边界非单调。
// §86.7/§88.6 的结论是「跨 SM 交换必须低并发/批量/环形化 + 保留硬件同步原语」。本节把这个「硬件
// 同步原语」具体化成 **cluster 作用域 mbarrier**（`mbarrier.arrive.shared::cluster` 远端到达 +
// `mbarrier.try_wait.parity` 本地等待），其余与 `ring` 逐字相同（每线程 SPSC 环形槽、无 cluster.sync）。
// 每个槽 j 在**消费者自己的 SMEM** 里放 readyBar[j]（上游生产者远端 arrive）、ackBar[j]（下游
// 消费者远端 arrive）；生产者复用槽前等自己的 ackBar，消费者读前等自己的 readyBar。
//   - count=TH：每 round 每线程 arrive 一次 → 每个槽每 S round 完成一个 phase。
//   - 第 r round 用槽 j=r%S、第 m=r/S 次复用：消费者等 readyBar[j] 的第 m 个 phase（parity=m&1），
//     生产者等 ackBar[j] 的第 m-1 个 phase（parity=(m-1)&1）。
// 指针走 CUTLASS 同款 `mapa.shared::cluster`（本地地址 + 远端 cta rank），避免 §88.6 的远端读脆弱路径。
__device__ __forceinline__ unsigned cvta_smem(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(unsigned smem_addr, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_remote(unsigned smem_addr, unsigned cta) {
  asm volatile("{\n\t.reg .b32 r;\n\tmapa.shared::cluster.u32 r, %0, %1;\n\t"
               "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [r];\n\t}"
               :: "r"(smem_addr), "r"(cta) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(unsigned smem_addr, unsigned phase) {
  unsigned r;
  // 带 suspend-time-hint（CUTLASS 同款）：让 try_wait 在未完成时 potential-block，
  // 避免 256 线程紧旋 `try_wait` 把 SMEM/LSU 单元打满、饿死远端 arrive 通路。
  unsigned ticks = 0x989680;
  asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, %3;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t}"
               : "=r"(r) : "r"(smem_addr), "r"(phase), "r"(ticks) : "memory");
  return r != 0;
}
__device__ __forceinline__ void mbar_wait(unsigned smem_addr, unsigned phase) {
  while (!mbar_try_wait(smem_addr, phase)) {}
}
// 无 suspend-hint 的 try_wait（hint=0 ⇒ 非阻塞轮询）。用于 `mbarB` 的「单线程等待」形态诊断：
// 当只有 tid0 一个线程等远端 arrive 时，带 10ms suspend-hint 的 try_wait 可能让该线程长时间
// 不被唤醒（本机实测性能双模态 15×）；hint=0 让它持续轮询。
__device__ __forceinline__ bool mbar_try_wait0(unsigned smem_addr, unsigned phase) {
  unsigned r;
  asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, %3;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t}"
               : "=r"(r) : "r"(smem_addr), "r"(phase), "r"(0u) : "memory");
  return r != 0;
}
__device__ __forceinline__ void mbar_wait0(unsigned smem_addr, unsigned phase) {
  while (!mbar_try_wait0(smem_addr, phase)) {}
}
// ---------- 远端 arrive.expect_tx + S2S bulk（§97 `mtex` 用） ----------
// 先用「远端 `mbarrier.arrive.expect_tx`」把对端 readyBar 的「到达数(1) + 期望事务字节数」一次登记，
// 再由 `cp.async.bulk.shared::cluster.shared::cta` 把本 block 的 snd 搬进对端 rcv，**完成信号由
// TMA 事务字节数（tx count）落在同一枚对端 mbarrier 上**——即「每槽 done」不再由生产者写完后发
// 一个软件 arrive 表达，而由 TMA 引擎的实际字节交付表达。登记（arrive.expect_tx）与搬运（bulk）
// 同在**生产者**线程的程序序里，故无需消费者预先 arm，避免「消费者还没 arm、TMA 已完成」的竞态。
__device__ __forceinline__ void mbar_arrive_expect_tx_remote(unsigned mbar_cluster, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
               :: "r"(mbar_cluster), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_s2s(unsigned dst_cluster, unsigned src_local,
                                         unsigned bytes, unsigned mbar_cluster) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :: "r"(dst_cluster), "r"(src_local), "r"(bytes), "r"(mbar_cluster) : "memory");
}

#define DEF_MBAR(U)                                                                     \
  __global__ void mbar_##U(float* __restrict__ out, int rounds, int sb, int Sr) {       \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = Sr < tiles ? Sr : tiles;                                                    \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    uint64_t* readyBar = (uint64_t*)(rcv + (size_t)S * SEG);                            \
    uint64_t* ackBar = readyBar + S;                                                    \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                        \
    int tid = threadIdx.x;                                                             \
    for (int i = tid; i < sb; i += blockDim.x) snd[i] = fval(i, rank);                  \
    if (tid == 0)                                                                      \
      for (int j = 0; j < S; j++) { mbar_init(cvta_smem(&readyBar[j]), TH);            \
                                    mbar_init(cvta_smem(&ackBar[j]), TH); }             \
    __syncthreads();                                                                    \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                  \
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      int m = r / S;                                                                    \
      if (m >= 1) mbar_wait(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));           \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      mbar_arrive_remote(rBar + 8 * (unsigned)j, dstRank);                              \
      mbar_wait(rBar + 8 * (unsigned)j, (unsigned)(m & 1));                             \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      mbar_arrive_remote(aBar + 8 * (unsigned)j, srcRank);                              \
    }                                                                                   \
    cl.sync();   /* teardown：确保所有远端 arrive 落定、无人提前退出释放 SMEM */           \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_MBAR(1) DEF_MBAR(2) DEF_MBAR(4) DEF_MBAR(8) DEF_MBAR(16)
#undef DEF_MBAR

// ---------- 环形槽 + 每槽 mbarrier 的「省 arrive」形态：每 block 一次 arrive（count=1） ----------
// §89.7 明留的尾巴：`mbar` 是**每线程**每槽 arrive（`count=TH`，每 round 256 次远端 arrive），
// 未测「每 block 一次 arrive（`count=1`）+ `__syncthreads` 包裹」的省 arrive 形态。本节补上：
// 语义/拓扑/数据量与 `mbar` 逐字相同，唯一变量 = **arrive 与等待的粒度**——数据由全 block 协作写、
// `__syncthreads()` 后再由 tid0 单点 arrive；消费者也由 tid0 单点等、`__syncthreads()` 后全 block 读。
// 每槽 readyBar/ackBar 的 count 从 TH 降到 1，每 round 的跨 SM arrive 次数从 `2·TH` 降到 2。
// 代价是每 round 多出 3 个 `__syncthreads()`（写后/等后/读后）——本节量这两笔的相对大小。
#define DEF_MBARB(U)                                                                    \
  __global__ void mbarb_##U(float* __restrict__ out, int rounds, int sb, int Sr) {      \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = Sr < tiles ? Sr : tiles;                                                    \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    uint64_t* readyBar = (uint64_t*)(rcv + (size_t)S * SEG);                            \
    uint64_t* ackBar = readyBar + S;                                                    \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    int tid = threadIdx.x;                                                             \
    for (int i = tid; i < sb; i += blockDim.x) snd[i] = fval(i, rank);                  \
    if (tid == 0)                                                                      \
      for (int j = 0; j < S; j++) { mbar_init(cvta_smem(&readyBar[j]), 1);             \
                                    mbar_init(cvta_smem(&ackBar[j]), 1); }              \
    __syncthreads();                                                                    \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                  \
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      int m = r / S;                                                                    \
      if (m >= 1) {                                                                     \
        if (tid == 0) mbar_wait(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));       \
        __syncthreads();                                                                \
      }                                                                                 \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      __syncthreads();                                                                  \
      if (tid == 0) {                                                                   \
        asm volatile("fence.acq_rel.cluster;" ::: "memory");                            \
        mbar_arrive_remote(rBar + 8 * (unsigned)j, dstRank);                            \
        mbar_wait(rBar + 8 * (unsigned)j, (unsigned)(m & 1));                           \
      }                                                                                 \
      __syncthreads();                                                                  \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      __syncthreads();                                                                  \
      if (tid == 0) {                                                                   \
        asm volatile("fence.acq_rel.cluster;" ::: "memory");                            \
        mbar_arrive_remote(aBar + 8 * (unsigned)j, srcRank);                            \
      }                                                                                 \
    }                                                                                   \
    cl.sync();   /* teardown：确保所有远端 arrive 落定、无人提前退出释放 SMEM */           \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_MBARB(1) DEF_MBARB(2) DEF_MBARB(4) DEF_MBARB(8) DEF_MBARB(16)
#undef DEF_MBARB

// ---------- mbarb 的「无 suspend-hint」变体：`mbarB` 的性能双模态根因对照 ----------
// 与 `mbarb` 逐字相同，唯一区别 = 单线程等待用 `mbar_wait0`（try_wait hint=0，持续轮询）而非
// `mbar_wait`（带 10ms potential-blocking hint）。本机实测：带 hint 时只有 tid0 一个线程在等远端
// arrive，10ms hint 会让该线程长期不被唤醒 → 整核双模态（多数进程 ~39750 ns/lap、偶发 ~2625 ns/lap，
// 数值恒正确）；换成 hint=0 后稳定在 ~2625 ns/lap。即 **suspend-hint 与「等待粒度」耦合**：
// CUTLASS 式 hint 是为「每线程都等」的形态调的。
#define DEF_MBARB0(U)                                                                   \
  __global__ void mbarb0_##U(float* __restrict__ out, int rounds, int sb, int Sr) {     \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = Sr < tiles ? Sr : tiles;                                                    \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    uint64_t* readyBar = (uint64_t*)(rcv + (size_t)S * SEG);                            \
    uint64_t* ackBar = readyBar + S;                                                    \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    int tid = threadIdx.x;                                                             \
    for (int i = tid; i < sb; i += blockDim.x) snd[i] = fval(i, rank);                  \
    if (tid == 0)                                                                      \
      for (int j = 0; j < S; j++) { mbar_init(cvta_smem(&readyBar[j]), 1);             \
                                    mbar_init(cvta_smem(&ackBar[j]), 1); }              \
    __syncthreads();                                                                    \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                  \
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      int m = r / S;                                                                    \
      if (m >= 1) {                                                                     \
        if (tid == 0) mbar_wait0(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));      \
        __syncthreads();                                                                \
      }                                                                                 \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      __syncthreads();                                                                  \
      if (tid == 0) {                                                                   \
        asm volatile("fence.acq_rel.cluster;" ::: "memory");                            \
        mbar_arrive_remote(rBar + 8 * (unsigned)j, dstRank);                            \
        mbar_wait0(rBar + 8 * (unsigned)j, (unsigned)(m & 1));                          \
      }                                                                                 \
      __syncthreads();                                                                  \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      __syncthreads();                                                                  \
      if (tid == 0) {                                                                   \
        asm volatile("fence.acq_rel.cluster;" ::: "memory");                            \
        mbar_arrive_remote(aBar + 8 * (unsigned)j, srcRank);                            \
      }                                                                                 \
    }                                                                                   \
    cl.sync();                                                                          \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_MBARB0(1) DEF_MBARB0(2) DEF_MBARB0(4) DEF_MBARB0(8) DEF_MBARB0(16)
#undef DEF_MBARB0

// ---------- 环形槽 + 每槽 mbarrier 的「每 warp 一次 arrive」中间形态（mbarw）----------
// §90.6 明留的尾巴：`mbarb` 把每 round 的跨 SM arrive 从 `2·TH` 压到 2，却被每 round 3–4 个
// `__syncthreads()` 全部吃掉并倒亏；未测「每 warp 一次 arrive」的中间形态。本节补上：
// 数据仍由全 block 协作写，但**以 warp 为单位**——每个 warp 写完自己那部分后 `__syncwarp()`，
// 由 lane0 用 `fence.acq_rel.cluster` + 远端 arrive 一次；consumer 端每个 warp 的 lane0 单点
// 等（hint=0，沿用 §90.4 的教训）、`__syncwarp()` 后本 warp 读。每槽 readyBar/ackBar 的 count
// 从 `TH`(mbar) / `1`(mbarb) 变成 `NW=blockDim/32`（T=256 时 8）；每 round 每 block 的跨 SM
// arrive 从 `2·TH=512`(mbar) / `2`(mbarb) 变成 `2·NW=16`，会合从 `__syncthreads` 降为
// `__syncwarp`（warp 作用域）。故它是 arrive 条数与会合代价之间的中间档：
//   mbar（512 arrive / 0 会合）  <->  mbarw（16 arrive / 3×8 warp 会合）  <->  mbarb（2 arrive / 3× block 会合）
#define DEF_MBARW(U)                                                                    \
  __global__ void mbarw_##U(float* __restrict__ out, int rounds, int sb, int Sr) {      \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = Sr < tiles ? Sr : tiles;                                                    \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    uint64_t* readyBar = (uint64_t*)(rcv + (size_t)S * SEG);                            \
    uint64_t* ackBar = readyBar + S;                                                    \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                        \
    int tid = threadIdx.x;                                                             \
    int lane = tid & 31;                                                                \
    int NW = blockDim.x >> 5;   /* 每 block 的 warp 数 = mbarrier count */                \
    for (int i = tid; i < sb; i += blockDim.x) snd[i] = fval(i, rank);                  \
    if (tid == 0)                                                                      \
      for (int j = 0; j < S; j++) { mbar_init(cvta_smem(&readyBar[j]), NW);            \
                                    mbar_init(cvta_smem(&ackBar[j]), NW); }             \
    __syncthreads();                                                                    \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                  \
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      int m = r / S;                                                                    \
      if (m >= 1) {                                                                     \
        if (lane == 0) mbar_wait0(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));     \
        __syncwarp();                                                                   \
      }                                                                                 \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      __syncwarp();                                                                     \
      if (lane == 0) {                                                                  \
        asm volatile("fence.acq_rel.cluster;" ::: "memory");                            \
        mbar_arrive_remote(rBar + 8 * (unsigned)j, dstRank);                            \
        mbar_wait0(rBar + 8 * (unsigned)j, (unsigned)(m & 1));                          \
      }                                                                                 \
      __syncwarp();                                                                     \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      __syncwarp();                                                                     \
      if (lane == 0) {                                                                  \
        asm volatile("fence.acq_rel.cluster;" ::: "memory");                            \
        mbar_arrive_remote(aBar + 8 * (unsigned)j, srcRank);                            \
      }                                                                                 \
    }                                                                                   \
    cl.sync();   /* teardown：确保所有远端 arrive 落定、无人提前退出释放 SMEM */           \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_MBARW(1) DEF_MBARW(2) DEF_MBARW(4) DEF_MBARW(8) DEF_MBARW(16)
#undef DEF_MBARW

// ---------- mbarw 的「无显式 fence」变体（mbarw0）：诊断每-warp 形态慢在哪 ----------
// 与 `mbarw` 逐字相同，唯一区别 = 去掉 lane0 在 `__syncwarp()` 之后、远端 arrive 之前的那条
// `fence.acq_rel.cluster`，改为依赖 `__syncwarp()`（warp 作用域 acquire-release）+ `mbarrier.arrive
// .release.cluster`（arrive 自带 release）这一「因果累积」发布 warp 内其余 lane 的写。加它的动机：
// §90 已证「会合」比 arrive 条数贵；mbarw 每 round 每 block 有 8 warp × 2 = 16 条 **cluster 作用域
// fence**（mbarb 只有 2 条），若 mbarw 慢在 fence 而非 `__syncwarp`，则去掉 fence 应显著变快。
// 这是个**双面探针**：既测「每-warp 发布是否必须显式 cluster fence」，又测 fence 的真实代价。
#define DEF_MBARW0(U)                                                                   \
  __global__ void mbarw0_##U(float* __restrict__ out, int rounds, int sb, int Sr) {     \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = Sr < tiles ? Sr : tiles;                                                    \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    uint64_t* readyBar = (uint64_t*)(rcv + (size_t)S * SEG);                            \
    uint64_t* ackBar = readyBar + S;                                                    \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                        \
    int tid = threadIdx.x;                                                             \
    int lane = tid & 31;                                                                \
    int NW = blockDim.x >> 5;                                                           \
    for (int i = tid; i < sb; i += blockDim.x) snd[i] = fval(i, rank);                  \
    if (tid == 0)                                                                      \
      for (int j = 0; j < S; j++) { mbar_init(cvta_smem(&readyBar[j]), NW);            \
                                    mbar_init(cvta_smem(&ackBar[j]), NW); }             \
    __syncthreads();                                                                    \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                  \
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);              \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      int m = r / S;                                                                    \
      if (m >= 1) {                                                                     \
        if (lane == 0) mbar_wait0(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));     \
        __syncwarp();                                                                   \
      }                                                                                 \
      _Pragma("unroll") for (int u = 0; u < U; u++) dst[j * SEG + u * TH + tid] = snd[off + u * TH + tid]; \
      __syncwarp();                                                                     \
      if (lane == 0) {                                                                  \
        mbar_arrive_remote(rBar + 8 * (unsigned)j, dstRank);                            \
        mbar_wait0(rBar + 8 * (unsigned)j, (unsigned)(m & 1));                          \
      }                                                                                 \
      __syncwarp();                                                                     \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      __syncwarp();                                                                     \
      if (lane == 0) {                                                                  \
        mbar_arrive_remote(aBar + 8 * (unsigned)j, srcRank);                            \
      }                                                                                 \
    }                                                                                   \
    cl.sync();                                                                          \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_MBARW0(1) DEF_MBARW0(2) DEF_MBARW0(4) DEF_MBARW0(8) DEF_MBARW0(16)
#undef DEF_MBARW0

// ---------- §97「每槽 done 由 TMA 事务计数（expect_tx）表达」（mtex）----------
// §89.7 明留的尾巴：`mbar` 系列只用 ready/ack 两枚 mbarrier 的软件 arrive 表达「数据写完」，
// **未测 `expect_tx`（TMA 事务计数）形态**。本节补上：语义/拓扑/数据量/环深与 `mbar` 逐字相同
// （每 block 把 snd 的 SEG 段推给 (rank+1)%C 的 rcv 第 j=r%S 槽，读回后 ack 给上游），唯一变量 =
// **「ready/done」如何表达**：
//   - `mbar`：生产者**全 block 每线程** `st.shared::cluster` 写数据，再每线程远端 `arrive` 对端
//     readyBar（每 round 每 block 512 次远端 arrive）。
//   - `mtex`：生产者**单线程**发一笔 `cp.async.bulk.shared::cluster.shared::cta`（TMA 引擎搬 SEG*4
//     字节），完成按 **TMA tx count** 落到对端 readyBar——「done」是硬件记账的实际字节交付，不是
//     软件 arrive。登记用**远端 `mbarrier.arrive.expect_tx`**（生产者侧、与 bulk 同程序序），故消费者
//     无需预 arm。
// ack（背压）方向与 `mbar` 逐字相同（消费者每线程远端 arrive 生产者的 ackBar，count=TH），以便
// 单变量隔离「ready/done 的表达方式」。环深 S 语义同 mbar（每槽每 S round 一个 phase）。
#define DEF_MTEX(U)                                                                     \
  __global__ void mtex_##U(float* __restrict__ out, int rounds, int sb, int Sr) {       \
    extern __shared__ float sm[];                                                       \
    int SEG = TH * U;                                                                   \
    int tiles = sb / SEG;                                                               \
    int S = Sr < tiles ? Sr : tiles;                                                    \
    float* snd = sm;                                                                    \
    float* rcv = sm + sb;                                                               \
    uint64_t* readyBar = (uint64_t*)(rcv + (size_t)S * SEG);                            \
    uint64_t* ackBar = readyBar + S;                                                    \
    cg::cluster_group cl = cg::this_cluster();                                          \
    int C = cl.num_blocks();                                                            \
    int rank = cl.block_rank();                                                         \
    int tid = threadIdx.x;                                                              \
    for (int i = tid; i < sb; i += blockDim.x) snd[i] = fval(i, rank);                  \
    if (tid == 0)                                                                       \
      for (int j = 0; j < S; j++) { mbar_init(cvta_smem(&readyBar[j]), 1);             \
                                    mbar_init(cvta_smem(&ackBar[j]), TH); }             \
    __syncthreads();                                                                    \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                  \
    cl.sync();                                                                          \
    int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;                         \
    float* dst = (float*)cl.map_shared_rank(rcv, dstRank);                              \
    unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);              \
    unsigned dstRBar0 = (unsigned)0;                                                    \
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dstRBar0)               \
                 : "r"(rBar), "r"((unsigned)dstRank));                                  \
    long long acc = 0;                                                                  \
    for (int r = 0; r < rounds; r++) {                                                  \
      int j = r % S;                                                                    \
      int off = (r % tiles) * SEG;                                                      \
      int m = r / S;                                                                    \
      if (m >= 1) mbar_wait(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));           \
      if (tid == 0) {                                                                   \
        mbar_arrive_expect_tx_remote(dstRBar0 + 8 * (unsigned)j, (unsigned)(SEG * 4));  \
        bulk_s2s(cvta_smem(dst + (size_t)j * SEG), cvta_smem(snd + off),               \
                 (unsigned)(SEG * 4), dstRBar0 + 8 * (unsigned)j);                      \
      }                                                                                 \
      mbar_wait(rBar + 8 * (unsigned)j, (unsigned)(m & 1));                             \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc += (long long)rcv[j * SEG + u * TH + tid]; \
      mbar_arrive_remote(aBar + 8 * (unsigned)j, (unsigned)srcRank);                   \
    }                                                                                   \
    cl.sync();   /* teardown：确保所有远端 arrive / bulk 落定、无人提前退出释放 SMEM */    \
    if (tid == 0) out[blockIdx.x] = (float)acc;                                         \
  }
DEF_MTEX(1) DEF_MTEX(2) DEF_MTEX(4) DEF_MTEX(8) DEF_MTEX(16)
#undef DEF_MTEX

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
static void* pick_mbar(int U) {
#define P(UU) if (U == UU) return (void*)mbar_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_mbarb(int U) {
#define P(UU) if (U == UU) return (void*)mbarb_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_mbarb0(int U) {
#define P(UU) if (U == UU) return (void*)mbarb0_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_mbarw(int U) {
#define P(UU) if (U == UU) return (void*)mbarw_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_mbarw0(int U) {
#define P(UU) if (U == UU) return (void*)mbarw0_##UU;
  P(1) P(2) P(4) P(8) P(16)
#undef P
  return nullptr;
}
static void* pick_mtex(int U) {
#define P(UU) if (U == UU) return (void*)mtex_##UU;
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

static int run(const char* tag, int C, int grid, int sbKB, int rounds, int U, int path, int Sp) {
  int sb = sbKB * 1024 / 4;
  int SEG = TH * U;
  if (SEG > sb) { printf("%s C=%d grid=%d sb=%dKB U=%d | SKIP(SEG>sb)\n", tag, C, grid, sbKB, U); return 3; }
  int tiles = sb / SEG, S = tiles < 4 ? tiles : 4;
  if (path >= 5) S = (Sp > 0 && Sp < tiles) ? Sp : S;   // mbar/mbarb/mbarb0/mbarw/mbarw0 的环深 S 可独立指定
  size_t smem = path == 3
      ? (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * TH * 4   // snd + S*rcv + ready/ack
      : (path >= 5)
      ? (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * 8        // snd + S*rcv + ready/ack mbarrier
      : (size_t)2 * sb * 4;                                            // snd + rcv
  int is_push = (path == 1 || path == 3 || path == 4 || path >= 5);
  void* fn = path == 1 ? pick_push(U) : (path == 2 ? pick_pulls(U)
             : (path == 3 ? pick_ring(U) : (path == 4 ? pick_ring1(U)
             : (path == 5 ? pick_mbar(U) : (path == 6 ? pick_mbarb(U)
             : (path == 7 ? pick_mbarb0(U) : (path == 8 ? pick_mbarw(U)
             : (path == 9 ? pick_mbarw0(U) : (path == 10 ? pick_mtex(U) : pick_pull(U))))))))));
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, TH, smem));

  // 逐个 launch 计时、统计成功次数（连续 launch 会累积失败，故不丢弃部分结果）。
  const int N = 20;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 0; int ok = 0; cudaError_t last = cudaSuccess;
  // 正式计时：每次 launch 一个 event 对，取 min-of-N、统计成功次数。
  // mbar（path 5）内核多一个 S 参数。
#define FN(UU) (path == 1 ? (void*)push_##UU : (path == 2 ? (void*)pulls_##UU \
              : (path == 3 ? (void*)ring_##UU : (path == 4 ? (void*)ring1_##UU \
              : (path == 5 ? (void*)mbar_##UU : (path == 6 ? (void*)mbarb_##UU \
              : (path == 7 ? (void*)mbarb0_##UU : (path == 8 ? (void*)mbarw_##UU \
              : (path == 9 ? (void*)mbarw0_##UU : (path == 10 ? (void*)mtex_##UU \
              : (void*)pull_##UU))))))))))
#define TIME(UU) if (U == UU) {                                                        \
    for (int i = 0; i < N; i++) {                                                      \
      cudaEventRecord(e0);                                                             \
      cudaError_t e = (path >= 5)                                                       \
          ? launch_cluster(C, dim3(grid), dim3(TH), smem, FN(UU), out, rounds, sb, S)   \
          : launch_cluster(C, dim3(grid), dim3(TH), smem, FN(UU), out, rounds, sb);     \
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
static int probe(int C, int grid, int sbKB, int U, int reps, int path, int rounds, int Sp) {
  int sb = sbKB * 1024 / 4;
  int SEG = TH * U, tiles = sb / SEG, S = tiles < 4 ? tiles : 4;
  if (path >= 5) S = (Sp > 0 && Sp < tiles) ? Sp : S;
  size_t smem = path == 3
      ? (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * TH * 4
      : (path >= 5)
      ? (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * 8
      : (size_t)2 * sb * 4;
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  void* pfn = path == 5 ? pick_mbar(U) : path == 6 ? pick_mbarb(U) : path == 7 ? pick_mbarb0(U)
            : path == 8 ? pick_mbarw(U) : path == 9 ? pick_mbarw0(U) : path == 10 ? pick_mtex(U)
            : path == 1 ? pick_push(U) : path == 2 ? pick_pulls(U)
            : path == 3 ? pick_ring(U) : path == 4 ? pick_ring1(U) : pick_pull(U);
  cudaFuncSetAttribute(pfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int ok = 0;
  for (int i = 0; i < reps; i++) {
    cudaError_t e;
#define GO(UU) if (U == UU) e = (path >= 5) \
    ? launch_cluster(C, dim3(grid), dim3(TH), smem, \
        path == 6 ? (void*)mbarb_##UU : (path == 7 ? (void*)mbarb0_##UU \
        : (path == 8 ? (void*)mbarw_##UU : (path == 9 ? (void*)mbarw0_##UU \
        : (path == 10 ? (void*)mtex_##UU : (void*)mbar_##UU)))), \
        out, rounds, sb, S) \
    : launch_cluster(C, dim3(grid), dim3(TH), smem, \
    path == 1 ? (void*)push_##UU : (path == 2 ? (void*)pulls_##UU \
    : (path == 3 ? (void*)ring_##UU : (path == 4 ? (void*)ring1_##UU : (void*)pull_##UU))), \
    out, rounds, sb);
    GO(1) GO(2) GO(4) GO(8) GO(16)
#undef GO
    if (e == cudaSuccess) ok++;
    else cudaGetLastError();
  }
  printf("%s C=%d grid=%4d U=%2d rounds=%3d | %d/%d ok\n",
         path == 1 ? "W" : path == 2 ? "S" : path == 3 ? "N" : path == 4 ? "M"
         : path == 5 ? "B" : path == 6 ? "Q" : path == 7 ? "Z" : path == 8 ? "V"
         : path == 9 ? "U" : path == 10 ? "D" : "R",
         C, grid, U, rounds, ok, reps);
  cudaFree(out);
  return reps - ok;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: cluster_concurrency <A|R|W|S|N|M|B|Q|Z|V|U|D|F|G|K|O|L|P|Y|X|T|J|E> ...\n"); return 1; }
  const char* cmd = argv[1];
  // path：0=pull（远端读），1=push（远端写+本地读），2=pulls（远端读串行），
  //       3=ring（无 cluster.sync 的每槽 release/done），4=ring1（环形槽 + 单次 cluster.sync/round），
  //       5=mbar（环形槽 + 每槽 cluster 作用域 mbarrier 的 release/done，无 cluster.sync）
  //       6=mbarb（同 mbar，但每 block 一次 arrive（count=1）+ __syncthreads 包裹的省 arrive 形态）
  //       7=mbarb0（同 mbarb，但单线程等待用 try_wait hint=0，消除 suspend-hint 造成的性能双模态）
  //       8=mbarw（同 mbar，但**每 warp 一次 arrive**（count=NW=blockDim/32）+ __syncwarp 包裹的中间形态）
  //       9=mbarw0（同 mbarw，但去掉 lane0 显式 `fence.acq_rel.cluster`，测每-warp 形态是否慢在 cluster fence）
  int path = 0;
  if (!strcmp(cmd, "W") || !strcmp(cmd, "G")) path = 1;
  else if (!strcmp(cmd, "S") || !strcmp(cmd, "K")) path = 2;
  else if (!strcmp(cmd, "N") || !strcmp(cmd, "O")) path = 3;
  else if (!strcmp(cmd, "M") || !strcmp(cmd, "L")) path = 4;
  else if (!strcmp(cmd, "B") || !strcmp(cmd, "P")) path = 5;
  else if (!strcmp(cmd, "Q") || !strcmp(cmd, "Y")) path = 6;
  else if (!strcmp(cmd, "Z") || !strcmp(cmd, "X")) path = 7;
  else if (!strcmp(cmd, "V") || !strcmp(cmd, "T")) path = 8;
  else if (!strcmp(cmd, "U") || !strcmp(cmd, "J")) path = 9;
  else if (!strcmp(cmd, "D") || !strcmp(cmd, "E")) path = 10;
  const char* tag = path == 1 ? "W" : path == 2 ? "S" : path == 3 ? "N" : path == 4 ? "M"
             : path == 5 ? "B" : path == 6 ? "Q" : path == 7 ? "Z" : path == 8 ? "V"
             : path == 9 ? "U" : path == 10 ? "D" : "R";

  if (!strcmp(cmd, "A")) {
    printf("# §87 §A / §88 §A 资源表（T=%d，sb=32KB；pull/push/pulls SMEM=64KB，ring 见下）\n", TH);
    int sb = 8192;
    for (int Ui = 0; Ui < 5; Ui++) {
      int U = (int[]){1, 2, 4, 8, 16}[Ui];
      int SEG = TH * U, tiles = sb / SEG, S = tiles < 4 ? tiles : 4;
      size_t smem_r = (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * TH * 4;
      size_t smem_m = (size_t)(sb + (size_t)S * SEG) * 4 + (size_t)2 * S * 8;
      for (int P = 0; P < 11; P++) {
        size_t smem = P == 3 ? smem_r : ((P >= 5) ? smem_m : (size_t)2 * sb * 4);
        void* fn = P == 0 ? pick_pull(U) : (P == 1 ? pick_push(U)
                 : (P == 2 ? pick_pulls(U) : (P == 3 ? pick_ring(U)
                 : (P == 4 ? pick_ring1(U) : (P == 5 ? pick_mbar(U)
                 : (P == 6 ? pick_mbarb(U) : (P == 7 ? pick_mbarb0(U)
                 : (P == 8 ? pick_mbarw(U) : (P == 9 ? pick_mbarw0(U)
                 : pick_mtex(U))))))))));
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
        int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, TH, smem));
        printf("  %-6s U=%2d regs=%d localB=%zu occ=%d blk/SM%s\n",
               P == 0 ? "pull" : (P == 1 ? "push" : (P == 2 ? "pulls"
               : (P == 3 ? "ring" : (P == 4 ? "ring1" : (P == 5 ? "mbar"
               : (P == 6 ? "mbarb" : (P == 7 ? "mbarb0" : (P == 8 ? "mbarw"
               : (P == 9 ? "mbarw0" : "mtex"))))))))), U,
               a.numRegs, (size_t)a.localSizeBytes, occ,
               (P == 3 || P >= 5) ? " (S=min(4,tiles))" : "");
      }
    }
    return 0;
  }
  if (!strcmp(cmd, "R") || !strcmp(cmd, "W") || !strcmp(cmd, "S") || !strcmp(cmd, "N")
      || !strcmp(cmd, "M") || !strcmp(cmd, "B") || !strcmp(cmd, "Q") || !strcmp(cmd, "Z")
      || !strcmp(cmd, "V") || !strcmp(cmd, "U") || !strcmp(cmd, "D")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sbKB = atoi(argv[4]);
    int rounds = atoi(argv[5]), U = atoi(argv[6]);
    int Sp = argc > 7 ? atoi(argv[7]) : 0;
    return run(tag, C, grid, sbKB, rounds, U, path, Sp);
  }
  if (!strcmp(cmd, "F") || !strcmp(cmd, "G") || !strcmp(cmd, "K") || !strcmp(cmd, "O")
      || !strcmp(cmd, "L") || !strcmp(cmd, "P") || !strcmp(cmd, "Y") || !strcmp(cmd, "X")
      || !strcmp(cmd, "T") || !strcmp(cmd, "J") || !strcmp(cmd, "E")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), sbKB = atoi(argv[4]);
    int U = atoi(argv[5]), reps = atoi(argv[6]);
    int rounds = argc > 7 ? atoi(argv[7]) : 4;
    int Sp = argc > 8 ? atoi(argv[8]) : 0;
    return probe(C, grid, sbKB, U, reps, path, rounds, Sp);
  }
  printf("unknown cmd %s\n", cmd);
  return 1;
}
