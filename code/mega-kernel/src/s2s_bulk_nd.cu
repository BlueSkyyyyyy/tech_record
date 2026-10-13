// ============================================================================
// P178-1 (§103)：S2S bulk 的「非连续（2D/3D）搬运」——把 §97 `mtex`/§98 `mtex2` 的
// `cp.async.bulk.shared::cluster.shared::cta`（SMEM→对端 SMEM 的 TMA 大块搬运）从
// **单笔连续段**推广到 **多笔线性搬运拼出的非连续 2D tile**。
//
// 背景：
//   - §97（P172-1）/§98（P173-1）用 `bulk_s2s` 把 SEG 段（连续）搬进对端 SMEM 的环形槽，
//     但每条尾巴都明写「**S2S 只搬连续段**，未测 tensor/2D 非连续」。
//   - 静态事实：PTX 里 S2S 的 bulk 只有**线性**形式——
//       `cp.async.bulk.shared::cluster.shared::cta[...]`（源/目的都是线性字节范围）；
//     带 `.tensor` 的形式只有 `.global` 源/目的（`cp.async.bulk.tensor.*.global.shared::cta`
//     / `cp.async.bulk.tensor.*.shared::cluster.global`），**不存在 shared::cluster↔shared::cta
//     的 tensor/2D 搬运**。故「把一个带行距（pitch）的 2D tile 从本 SMEM 搬到对端 SMEM」只能
//     拆成**每行（或每行再切 K 段）一笔线性 bulk**。本节点把该拆分的代价量出来。
//
// 目标卡 H100 / sm90a。每配置单独进程（一次 cluster launch failure 会 sticky 污染 CUDA context）。
//
// 语义模型：C 个 CTA 成环（rank→(rank+1)%C）。每 round，rank 把本 block 的 2D tile
// （`rows` 行 × `cols` 字，源行距 `pitch`≥`cols`）搬到 (rank+1) 的 rcv 第 j=r%S 槽
// （目的**打包**、行距=cols），读回后 ack 给上游。唯一变量 = **一笔 tile 被拆成几笔 bulk**：
//   - `pitch==cols && K==1` ⇒ 1 笔（连续，= §97 形态，吞吐天花板）；
//   - `pitch>cols && K==1`  ⇒ `rows` 笔（每行一笔，天然 2D 非连续）；
//   - `pitch>cols && K>1`   ⇒ `rows*K` 笔（每行再切 K 段，测「多笔」的纯指令代价）。
// `ready/done` 一律用 §97 的形态：生产者 tid0 先远端 `mbarrier.arrive.expect_tx`（登记到达 1 +
// 期望 **payload 总字节数**），再发 N 笔 `bulk_s2s` 落同一枚对端 mbarrier（**TMA 事务字节数累加**）；
// 故本节点顺带验证「一次 expect_tx + 多笔 bulk 落同一 mbarrier」的语义正确性。
// `nd_sw` 是软件基线（全 block 每线程 `st.shared::cluster` + `__syncthreads` + tid0 软件 arrive）。
//
// 用法：nvcc -O3 -arch=sm_90a src/s2s_bulk_nd.cu -o s2s_bulk_nd
//       s2s_bulk_nd A
//       s2s_bulk_nd T <C> <grid> <rows> <cols> <pitch> <K> <rounds>   # TMA S2S bulk（计时）
//       s2s_bulk_nd W <C> <grid> <rows> <cols> <pitch> <rounds>       # 软件基线（计时）
//       s2s_bulk_nd F <C> <grid> <rows> <cols> <pitch> <K> <reps> <rounds>  # TMA 可行性探针
//       s2s_bulk_nd G <C> <grid> <rows> <cols> <pitch> <reps> <rounds>      # 软件可行性探针
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

#ifndef TH
#define TH 256
#endif
#define SLOTS 4   // 环形槽数 S

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
  return cudaLaunchKernelEx(&cfg, (cudaKernel_t)(void*)kern, args...);
}

__host__ __device__ __forceinline__ float fval(int i, int owner) {
  return (float)(((long long)i * 7 + (long long)owner * 13) % 1000);
}

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
  unsigned ticks = 0x989680;   // CUTLASS 同款 suspend-time-hint（避免全 block 紧旋饿死远端通路）
  asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, %3;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t}"
               : "=r"(r) : "r"(smem_addr), "r"(phase), "r"(ticks) : "memory");
  return r != 0;
}
__device__ __forceinline__ void mbar_wait(unsigned smem_addr, unsigned phase) {
  while (!mbar_try_wait(smem_addr, phase)) {}
}
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

// ---------------- TMA S2S bulk：1 笔 / rows 笔 / rows*K 笔 ----------------
// SMEM：snd[rows*pitch] + rcv[SLOTS*rows*cols] + readyBar[SLOTS] + ackBar[SLOTS]。
__global__ void nd_tma(float* __restrict__ out, int rounds, int rows, int cols,
                       int pitch, int K) {
  extern __shared__ float sm[];
  size_t tile_src = (size_t)rows * pitch;
  size_t tile_dst = (size_t)rows * cols;
  float* snd = sm;
  float* rcv = sm + tile_src;
  uint64_t* readyBar = (uint64_t*)(rcv + (size_t)SLOTS * tile_dst);
  uint64_t* ackBar = readyBar + SLOTS;
  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int tid = threadIdx.x;
  for (int i = tid; i < (int)tile_src; i += blockDim.x) snd[i] = fval(i, rank);
  if (tid == 0)
    for (int j = 0; j < SLOTS; j++) { mbar_init(cvta_smem(&readyBar[j]), 1);
                                      mbar_init(cvta_smem(&ackBar[j]), TH); }
  __syncthreads();
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  cl.sync();
  int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;
  float* dst = (float*)cl.map_shared_rank(rcv, dstRank);
  unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);
  unsigned dstRBar0 = 0;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dstRBar0)
               : "r"(rBar), "r"((unsigned)dstRank));
  int cw = cols / K;                       // 每段字数（保证 *4 为 16 B 倍数）
  unsigned payload = (unsigned)(tile_dst * 4);
  long long acc = 0;
  for (int r = 0; r < rounds; r++) {
    int j = r % SLOTS, m = r / SLOTS;
    if (m >= 1) mbar_wait(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));
    if (tid == 0) {
      mbar_arrive_expect_tx_remote(dstRBar0 + 8 * (unsigned)j, payload);
      float* dslot = dst + (size_t)j * tile_dst;
      if (pitch == cols && K == 1) {
        bulk_s2s(cvta_smem(dslot), cvta_smem(snd), payload, dstRBar0 + 8 * (unsigned)j);
      } else {
        for (int rr = 0; rr < rows; rr++)
          for (int c = 0; c < K; c++)
            bulk_s2s(cvta_smem(dslot + (size_t)rr * cols + (size_t)c * cw),
                     cvta_smem(snd + (size_t)rr * pitch + (size_t)c * cw),
                     (unsigned)(cw * 4), dstRBar0 + 8 * (unsigned)j);
      }
    }
    mbar_wait(rBar + 8 * (unsigned)j, (unsigned)(m & 1));
    float* rslot = rcv + (size_t)j * tile_dst;
    for (int i = tid; i < (int)tile_dst; i += blockDim.x) acc += (long long)rslot[i];
    mbar_arrive_remote(aBar + 8 * (unsigned)j, (unsigned)srcRank);
  }
  cl.sync();   // teardown：确保远端 arrive / bulk 落定、无人提前退出释放 SMEM
  if (tid == 0) out[blockIdx.x] = (float)acc;
}

// ---------------- 软件基线：全 block 每线程 st.shared::cluster ----------------
__global__ void nd_sw(float* __restrict__ out, int rounds, int rows, int cols, int pitch) {
  extern __shared__ float sm[];
  size_t tile_src = (size_t)rows * pitch;
  size_t tile_dst = (size_t)rows * cols;
  float* snd = sm;
  float* rcv = sm + tile_src;
  uint64_t* readyBar = (uint64_t*)(rcv + (size_t)SLOTS * tile_dst);
  uint64_t* ackBar = readyBar + SLOTS;
  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int tid = threadIdx.x;
  for (int i = tid; i < (int)tile_src; i += blockDim.x) snd[i] = fval(i, rank);
  if (tid == 0)
    for (int j = 0; j < SLOTS; j++) { mbar_init(cvta_smem(&readyBar[j]), 1);
                                      mbar_init(cvta_smem(&ackBar[j]), TH); }
  __syncthreads();
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  cl.sync();
  int dstRank = (rank + 1) % C, srcRank = (rank - 1 + C) % C;
  float* dst = (float*)cl.map_shared_rank(rcv, dstRank);
  unsigned rBar = cvta_smem(&readyBar[0]), aBar = cvta_smem(&ackBar[0]);
  long long acc = 0;
  for (int r = 0; r < rounds; r++) {
    int j = r % SLOTS, m = r / SLOTS;
    if (m >= 1) mbar_wait(aBar + 8 * (unsigned)j, (unsigned)((m - 1) & 1));
    float* dslot = dst + (size_t)j * tile_dst;
    for (int i = tid; i < (int)tile_dst; i += blockDim.x)
      dslot[i] = snd[(size_t)(i / cols) * pitch + (i % cols)];
    __syncthreads();
    if (tid == 0) mbar_arrive_remote(rBar + 8 * (unsigned)j, (unsigned)dstRank);
    mbar_wait(rBar + 8 * (unsigned)j, (unsigned)(m & 1));
    float* rslot = rcv + (size_t)j * tile_dst;
    for (int i = tid; i < (int)tile_dst; i += blockDim.x) acc += (long long)rslot[i];
    mbar_arrive_remote(aBar + 8 * (unsigned)j, (unsigned)srcRank);
  }
  cl.sync();
  if (tid == 0) out[blockIdx.x] = (float)acc;
}

// out[bx] 只存 tid0 的 acc（i 为 TH 的倍数）。数据来自上游 srcRank=(rank-1+C)%C。
static void reference_t0(int grid, int C, int rows, int cols, int pitch, int rounds,
                         long long* out_exp) {
  size_t tile_dst = (size_t)rows * cols;
  for (int bx = 0; bx < grid; bx++) {
    int rank = bx % C;
    int src = (rank - 1 + C) % C;
    long long acc = 0;
    for (int r = 0; r < rounds; r++)
      for (size_t i = 0; i < tile_dst; i += TH)
        acc += (long long)fval((int)((i / cols) * pitch + (i % cols)), src);
    out_exp[bx] = acc;
  }
}

static void check(int grid, int C, float* dout, int rows, int cols, int pitch, int rounds) {
  float* o = (float*)malloc((size_t)grid * sizeof(float));
  CK(cudaMemcpy(o, dout, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
  long long* exp = (long long*)malloc((size_t)grid * sizeof(long long));
  reference_t0(grid, C, rows, cols, pitch, rounds, exp);
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

static size_t smem_bytes(int rows, int cols, int pitch) {
  return ((size_t)rows * pitch + (size_t)SLOTS * rows * cols) * 4 + (size_t)2 * SLOTS * 8;
}

static int run_tma(int C, int grid, int rows, int cols, int pitch, int K, int rounds) {
  if (K < 1 || cols % (K * 4) != 0 || pitch < cols) {
    printf("T C=%d rows=%d cols=%d pitch=%d K=%d | SKIP(bad params)\n", C, rows, cols, pitch, K);
    return 3;
  }
  size_t smem = smem_bytes(rows, cols, pitch);
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  cudaFuncSetAttribute((const void*)nd_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, nd_tma));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, nd_tma, TH, smem));
  const int N = 20;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 0; int ok = 0; cudaError_t last = cudaSuccess;
  for (int i = 0; i < N; i++) {
    cudaEventRecord(e0);
    cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, nd_tma, out, rounds, rows, cols, pitch, K);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    if (e != cudaSuccess) { last = e; cudaGetLastError(); continue; }
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    if (ok == 0 || ms < best) best = ms;
    ok++;
  }
  double t = ok ? best : -1;
  printf("T C=%d grid=%4d rows=%2d cols=%3d pitch=%3d K=%2d rounds=%3d regs=%d occ=%d | ok=%d/%d %s",
         C, grid, rows, cols, pitch, K, rounds, a.numRegs, occ, ok, N,
         t < 0 ? (last == cudaSuccess ? "FAILED\n" : "FAIL\n") : "");
  if (t >= 0) {
    double payload = (double)grid * rounds * rows * cols * 4.0;
    int xfer = (pitch == cols && K == 1) ? 1 : rows * K;
    check(grid, C, out, rows, cols, pitch, rounds);
    printf("%8.3f us | %8.1f GB/s | xfer/round=%d ns/xfer=%7.1f\n", t * 1e3, payload / t / 1e6,
           xfer, t * 1e6 / ((double)rounds * xfer));
  } else if (last != cudaSuccess) {
    printf("[%s]\n", cudaGetErrorString(last));
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(out);
  return ok > 0 ? 0 : 2;
}

static int run_sw(int C, int grid, int rows, int cols, int pitch, int rounds) {
  if (pitch < cols) { printf("W | SKIP(bad params)\n"); return 3; }
  size_t smem = smem_bytes(rows, cols, pitch);
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  cudaFuncSetAttribute((const void*)nd_sw, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, nd_sw));
  const int N = 20;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 0; int ok = 0; cudaError_t last = cudaSuccess;
  for (int i = 0; i < N; i++) {
    cudaEventRecord(e0);
    cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, nd_sw, out, rounds, rows, cols, pitch);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    if (e != cudaSuccess) { last = e; cudaGetLastError(); continue; }
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    if (ok == 0 || ms < best) best = ms;
    ok++;
  }
  double t = ok ? best : -1;
  printf("W C=%d grid=%4d rows=%2d cols=%3d pitch=%3d rounds=%3d regs=%d | ok=%d/%d %s",
         C, grid, rows, cols, pitch, rounds, a.numRegs, ok, N,
         t < 0 ? (last == cudaSuccess ? "FAILED\n" : "FAIL\n") : "");
  if (t >= 0) {
    double payload = (double)grid * rounds * rows * cols * 4.0;
    check(grid, C, out, rows, cols, pitch, rounds);
    printf("%8.3f us | %8.1f GB/s\n", t * 1e3, payload / t / 1e6);
  } else if (last != cudaSuccess) {
    printf("[%s]\n", cudaGetErrorString(last));
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(out);
  return ok > 0 ? 0 : 2;
}

static int probe_tma(int C, int grid, int rows, int cols, int pitch, int K, int reps, int rounds) {
  size_t smem = smem_bytes(rows, cols, pitch);
  float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  cudaFuncSetAttribute((const void*)nd_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int ok = 0;
  for (int i = 0; i < reps; i++) {
    cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, nd_tma, out, rounds, rows, cols, pitch, K);
    if (e == cudaSuccess) ok++; else cudaGetLastError();
  }
  printf("F C=%d grid=%4d rows=%2d cols=%3d pitch=%3d K=%2d | %d/%d ok\n",
         C, grid, rows, cols, pitch, K, ok, reps);
  cudaFree(out);
  return reps - ok;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: s2s_bulk_nd <A|T|W|F|G> ...\n");
    return 1;
  }
  const char* cmd = argv[1];
  if (!strcmp(cmd, "A")) {
    printf("# §103 资源表（T=%d，SLOTS=%d）\n", TH, SLOTS);
    int cfg[][3] = {{4,256,256},{16,256,256},{16,256,320},{32,256,320}};
    for (int i = 0; i < 4; i++) {
      int rows = cfg[i][0], cols = cfg[i][1], pitch = cfg[i][2];
      size_t smem = smem_bytes(rows, cols, pitch);
      cudaFuncSetAttribute((const void*)nd_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      cudaFuncSetAttribute((const void*)nd_sw, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      cudaFuncAttributes at, as;
      CK(cudaFuncGetAttributes(&at, nd_tma));
      CK(cudaFuncGetAttributes(&as, nd_sw));
      int occ_t = 0, occ_s = 0;
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_t, nd_tma, TH, smem));
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_s, nd_sw, TH, smem));
      printf("  rows=%2d cols=%3d pitch=%3d smem=%6zuB | tma regs=%d occ=%d | sw regs=%d occ=%d\n",
             rows, cols, pitch, smem, at.numRegs, occ_t, as.numRegs, occ_s);
    }
    return 0;
  }
  if (!strcmp(cmd, "T"))
    return run_tma(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                   atoi(argv[6]), atoi(argv[7]), atoi(argv[8]));
  if (!strcmp(cmd, "W"))
    return run_sw(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                  atoi(argv[6]), atoi(argv[7]));
  if (!strcmp(cmd, "F"))
    return probe_tma(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                     atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]));
  if (!strcmp(cmd, "G")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), rows = atoi(argv[4]), cols = atoi(argv[5]);
    int pitch = atoi(argv[6]), reps = atoi(argv[7]), rounds = atoi(argv[8]);
    size_t smem = smem_bytes(rows, cols, pitch);
    float* out; CK(cudaMalloc(&out, (size_t)grid * sizeof(float)));
    cudaFuncSetAttribute((const void*)nd_sw, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int ok = 0;
    for (int i = 0; i < reps; i++) {
      cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, nd_sw, out, rounds, rows, cols, pitch);
      if (e == cudaSuccess) ok++; else cudaGetLastError();
    }
    printf("G C=%d grid=%4d rows=%2d cols=%3d pitch=%3d | %d/%d ok\n",
           C, grid, rows, cols, pitch, ok, reps);
    cudaFree(out);
    return reps - ok;
  }
  printf("unknown cmd %s\n", cmd);
  return 1;
}
