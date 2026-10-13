// ============================================================================
// P170-1（§95）：TMA multicast 掩码的「非连续形状」与「分裂发送」——补 §94.7 明留的
//        「未测 **非连续 / 重叠子集掩码**」尾巴。
//
// 背景：
//   - §19/§20/§93/§94 的 multicast 掩码都是**连续**的：或整群 `(1<<C)-1`，或逐行
//     `((1<<cx)-1)<<(row*cx)`，或 §94 的连续分组 `((1<<g)-1)<<first`。
//   - 掩码是 **16-bit 位图、按 cluster 线性 rank 编址**（§20 已确认）。「按 rank 位图」
//     意味着掩码的**形状**（哪几位）理论上不重要，只由**位数/发起者数**决定成本；
//     但 §94/§20 从未测过**非连续**（交错）掩码，也从未测过**一个组由多个 CTA 分头发**。
//
// 本节引入一个形状/发送模式 `pattern`：
//   - pattern 0（contig，§94 基线）：C 个 CTA 按连续 g 个一组，**组首**（rank - rank%g）
//     发一次 multicast，`mask = ((1<<g)-1) << leader`（连续）。
//   - pattern 1（stride，**非连续**）：按 `S=C/g` 交错分组 `{q, q+S, q+2S, ...}`，
//     **组内最小 rank q** 发一次，`mask = Σ_k (1<<(q+k·S))`（位图**离散**）。
//   - pattern 2（contig-split，**分裂发送**）：同 pattern 0 的连续分组与连续掩码，但
//     `L` 个 SMEM 槽由**两个组内 CTA 轮流发**（偶槽由组首、奇槽由组首+1），两笔
//     multicast 的 `mask` 覆盖**同一组**、写到**不同 SMEM 槽**。
//   - pattern 3（stride-split）：同 pattern 1 的交错分组与离散掩码，槽由 q 与 q+S 轮流发。
//
// 待答问题：
//   Q1（§A，形状不变性）：固定 `(g, 发起者数)`，把**连续**掩码换成**离散**交错掩码
//       （pattern 0→1）。墙钟 / delivery / 源侧 L2 请求是否**逐位不变**？即掩码成本只由
//       「popcount + 发起者数」定、与位图形状无关？
//   Q2（§B，分裂发送）：同一组、同一掩码、同一总字节，把 L 个槽由 1 个发起者改成
//       **2 个发起者轮流**（pattern 0→2）。墙钟 / 源侧请求是否改变？multicast 的「每笔
//       服务 g 个接收者」是否要求发起者 = 组首，还是任意组内 CTA 都行？
//   Q3（§C，C×pattern 交叉）：`C=4/8/16` 下形状不变性与分裂不变性是否一致。
//   Q4（§D，ncu）：离散掩码（pattern 1）的源侧 L2 读 sector 是否仍**精确等于
//       `delivered/(32·g)`**——把 §94.6 的「精确 1/g 律」从连续掩码推广到**任意位图**。
//
// 载体：§19/§93/§94 的 `bcast_kernel` 忠实复刻（L 个 SMEM chunk 槽、每轮发 L 次 TMA、
//       一个 mbarrier `expect_tx` 累计 `L·chunk` 字节后一次等待、`cl.sync()` 收尾）。每
//       个 CTA 仍**恰是某个（唯一）组的接收者一次**，故 `expect_tx` 恒 `chunk·L`。
//
// 编译：nvcc -O3 -arch=sm_90a src/multicast_mask_shape.cu -o /tmp/multicast_mask_shape -lcuda
// 用法：见 harness/multicast_mask_shape.sh（每配置独立进程——cluster launch failure 会 sticky）
//   B <C> <grid> <chunkKB> <R> <L> <pattern> <g> <iters>
//   目标卡 H100 / sm90a。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
  const char* s = "?"; cuGetErrorString(r, &s); \
  printf("[CU FAIL] %s -> %s\n", #x, s); exit(2); } } while (0)
#define CE(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("[CUDA FAIL] %s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)

static const int TH = 128;

// ---------------- tensor map ----------------
static CUtensorMap make_tmap_2d(const float* g, uint64_t d0, uint64_t d1,
                                uint32_t b0, uint32_t b1) {
  CUtensorMap tmap;
  uint64_t globalDim[2] = {d0, d1};
  uint64_t globalStrides[1] = {d0 * 4};
  uint32_t boxDim[2] = {b0, b1};
  uint32_t elemStrides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)g,
      globalDim, globalStrides, boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { const char* s = "?"; cuGetErrorString(r, &s);
    printf("[tmap encode FAIL] %s\n", s); exit(2); }
  return tmap;
}

// ---------------- PTX helpers ----------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra WAIT_%=;\n\t}"
      ::"r"(smem_u32(mbar)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_load_mcast_2d(const CUtensorMap* tmap, uint32_t dst,
                                                  uint64_t* mbar, uint16_t mask,
                                                  int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(dst), "l"((uint64_t)tmap), "r"(smem_u32(mbar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}

// ---------------- kernel body ----------------
// pattern 0: 连续分组，单发起者（组首）        —— §94 基线
// pattern 1: 交错分组，单发起者（组内最小 rank）—— 非连续掩码
// pattern 2: 连续分组，L 个槽由组首/组首+1 轮流 —— 分裂发送
// pattern 3: 交错分组，L 个槽由 q/q+S 轮流    —— 非连续 + 分裂发送
__global__ void shape_kernel(const __grid_constant__ CUtensorMap tmap,
                             float* __restrict__ out,
                             int R, int box1, int L, int pattern, int g) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int chunk = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk * 4u;
  int S = (g > 0) ? (C / g) : C;               // 交错步长（组数）

  if (threadIdx.x == 0) { mbar_init(&mbar, 1); mbar_init_fence(); }
  cl.sync();

  // 组首与掩码（连续 vs 离散）
  int leader;
  uint16_t mask;
  if (pattern == 0 || pattern == 2) {
    leader = (rank / g) * g;
    mask = (uint16_t)(((1u << g) - 1u) << leader);        // 连续位图
  } else {
    int q = rank % S;
    leader = q;
    uint32_t m = 0;
    for (int k = 0; k < g; k++) m |= (1u << (q + k * S)); // 离散位图
    mask = (uint16_t)m;
  }
  // 第二个发起者（仅分裂模式用）
  int second = (pattern == 3) ? (leader + S) : (leader + 1);

  float acc = 0.f;
  uint32_t phase = 0;
  for (int r = 0; r < R; r++) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(&mbar, chunk_bytes * (uint32_t)L);
      for (int j = 0; j < L; j++) {
        int issuer;
        if (pattern <= 1)                              issuer = leader;
        else issuer = (((r * L + j) & 1) ? second : leader);
        if (rank == issuer) {
          uint32_t dst = smem_u32(buf + (size_t)j * chunk);
          int c1 = (r * L + j) * box1;
          tma_load_mcast_2d(&tmap, dst, &mbar, mask, 0, c1);
        }
      }
    }
    mbar_wait(&mbar, phase);
    phase ^= 1u;
    acc += buf[threadIdx.x];
    acc += buf[threadIdx.x + TH];
    cl.sync();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

static cudaError_t launch_cluster(int C, dim3 grid, dim3 block, size_t smem,
                                  void* kern, void** args) {
  cudaFuncSetAttribute((const void*)kern,
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  return cudaLaunchKernelExC(&cfg, kern, args);
}

static int num_sm() {
  int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  return n;
}

static const char* pat_name(int p) {
  switch (p) { case 0: return "contig"; case 1: return "stride";
               case 2: return "contig-split"; case 3: return "stride-split";
               default: return "?"; }
}

// 单次 run：返回 mean ms/launch，并写出 maxerr
static double run_shape(int C, int grid, int KB, int R, int L, int pattern, int g, int iters,
                        double* maxerr, int* occ_out, int* regs_out) {
  int box1 = KB * 1024 / 4 / 256;
  int chunk = 256 * box1;
  if (L < 1) L = 1;
  if (g < 1) g = 1;
  if (g > C) g = C;
  if ((C % g) != 0) { printf("[bad: C%%g != 0] "); return -1; }
  if ((pattern == 2 || pattern == 3) && g < 2) { printf("[bad: split needs g>=2] "); return -1; }
  long ntot = 256L * (long)box1 * (long)R * L;
  float *src, *out;
  CE(cudaMalloc(&src, ntot * 4));
  CE(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  {
    float* h = (float*)malloc(ntot * 4);
    for (long i = 0; i < ntot; i++) h[i] = (float)((i % 251) - 125) * 1e-3f;
    CE(cudaMemcpy(src, h, ntot * 4, cudaMemcpyHostToDevice));
    free(h);
  }
  CUtensorMap tmap = make_tmap_2d(src, 256, ntot / 256, 256, box1);

  size_t smem = (size_t)L * chunk * 4;
  void* kern = (void*)shape_kernel;
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int occ = 0, regs = 0;
  {
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*)kern); regs = fa.numRegs;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, TH, smem);
  }
  void* args[] = {(void*)&tmap, (void*)&out, (void*)&R, (void*)&box1, (void*)&L,
                  (void*)&pattern, (void*)&g};
  cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
  if (e != cudaSuccess) { printf("[FAIL %s] ", cudaGetErrorString(e)); return -1; }
  CE(cudaDeviceSynchronize());

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
    if (e != cudaSuccess) { printf("[iter FAIL %s] ", cudaGetErrorString(e)); return -1; }
  }
  CE(cudaDeviceSynchronize());
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;

  {
    float* o = (float*)malloc((size_t)grid * sizeof(float));
    CE(cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0;
    for (int gi = 0; gi < grid; gi++) {
      double exp = 0;
      for (int r = 0; r < R; r++) {
        long base = (long)r * L * chunk;
        exp += (double)((base % 251) - 125) * 1e-3;
        exp += (double)(((base + 128) % 251) - 125) * 1e-3;
      }
      double d = o[gi] - exp; if (d < 0) d = -d;
      if (d > err) err = d;
    }
    *maxerr = err;
    free(o);
  }
  *occ_out = occ; *regs_out = regs;
  cudaFree(src); cudaFree(out);
  return ms;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: multicast_mask_shape B <C> <grid> <chunkKB> <R> <L> <pattern> <g> <iters>\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];

  if (!strcmp(cmd, "B")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        L = atoi(argv[6]), pattern = atoi(argv[7]), g = atoi(argv[8]),
        iters = argc > 9 ? atoi(argv[9]) : 20;
    double err = -1; int occ = 0, regs = 0;
    double ms = run_shape(C, grid, KB, R, L, pattern, g, iters, &err, &occ, &regs);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    double deliv = (double)grid * chunk * L * R;
    double nclusters = (double)grid / C;
    double groups = (double)(C / g);              // 每组一次 multicast（与 pattern 无关）
    double src_read = nclusters * groups * chunk * L * R;
    int nsm = num_sm();
    int eff = occ > 0 ? occ : 1;
    int per_sm = grid / nsm;
    if (per_sm < eff) eff = per_sm;
    if (eff < 1) eff = 1;
    double waves = (double)grid / (double)(nsm * eff);
    printf("B C=%2d grid=%4d chunk=%2dKB L=%2d R=%4d pat=%d(%s) g=%2d it=%2d | regs=%3d occ=%2d "
           "eff=%2d waves=%4.2f | ", C, grid, KB, L, R, pattern, pat_name(pattern), g, iters,
           regs, occ, eff, waves);
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | per-blk %6.1f GB/s | src-read %7.1f GB/s | maxerr %.5f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, deliv / (ms / 1e3) / 1e9 / grid,
             src_read / (ms / 1e3) / 1e9, err);
    else printf("FAILED\n");
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
