// ============================================================================
// P169-1（§94）：TMA multicast 的「子集扇出 g」与「chunk 维度」——补 §93.8 明留的
//        「未扫 chunk、未测 multicast 子集掩码」
//
// 背景：
//   - §19（P30-1）把 multicast 当**二值**用：mode0 = 每 CTA 各发普通 TMA，
//     mode1 = 仅 rank0 发 multicast、`mask=(1<<C)-1`（整群广播）。结论：整群广播省
//     源侧 L2 请求（2×）、DRAM 字节不变、墙钟打平。
//   - §93（P168-1）把 multicast 与占用率/寄存器预算叠加，得「L2 收益随占用率放大，
//     且依赖 C/L 的请求流结构」，但**只在 C=4 与 C=8、L=2 与 L=24、chunk=8KB 几个点**
//     上测过，并明确留尾巴：**未扫 `chunk`、未测 multicast 子集掩码**。
//
// 本节引入一个**连续旋钮 `g`（fan-out / 子集掩码）**：把每个 cluster 的 C 个 CTA 按
// 连续 `g` 个一组（共 C/g 组），每组由**组首 rank** 发一次 multicast、`mask` 只覆盖本组
// 的 g 个 CTA。于是：
//   - `g == 0`：退化基线 `plain`——每个 CTA 各发普通 TMA（= §19 mode0）；
//   - `g == C`：整群广播（= §19 mode1，但由组首=rank0 发）；
//   - `1 <= g < C`：**部分广播**——介于二者之间的连续中间形态（§93 从未测过）。
//
// 待答问题：
//   Q1（§B，fan-out 扫描）：源侧 L2 请求数是否**恰按 1/g 缩放**（每组的重复请求被一组
//       一次请求替代）？墙钟/delivery 是否基本不变（收益只在 L2 余量、不在 SMEM 写墙）？
//   Q2（§C，chunk 扫描）：固定总字节，扫 chunk=2/4/8/16/32KB。「L2 合并率」随 chunk 怎么走？
//       大 chunk 是否让 L2 更省（请求更少更大）？delivery 是否吃 TMA 事务大小？
//   Q3（§D，C×g 交叉）：C=4/8/16 下同一 g 的源侧缩减是否一致（即缩减由 g 而非 C 决定）？
//   Q4（§E，ncu）：用 ncu 直接量 `lts__t_sectors_srcunit_tex_op_read`，验证 1/g 假设。
//
// 载体：burst 版 TMA 广播（= §19/§93 的 `bcast_kernel`），chunk 个 float 的 L 个 SMEM 槽，
//       每轮发 L 次 TMA、一个 mbarrier `expect_tx` 累计后等待一次。动态 SMEM = L*chunk*4。
//
// 编译：nvcc -O3 -arch=sm_90a src/multicast_fanout.cu -o /tmp/multicast_fanout -lcuda
// 用法：见 harness/multicast_fanout.sh（每配置独立进程——cluster launch failure 会 sticky）
//   B <C> <grid> <chunkKB> <R> <L> <g> <iters>
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
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, uint32_t dst,
                                            uint64_t* mbar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      ::"r"(dst), "l"((uint64_t)tmap), "r"(smem_u32(mbar)), "r"(c0), "r"(c1)
      : "memory");
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
// g == 0 : 每 CTA 各发普通 TMA（plain / §19 mode0）
// g >= 1 : 连续 g 个 CTA 一组，组首发 multicast、mask 覆盖本组（g==C 即整群广播）
__global__ void fanout_kernel(const __grid_constant__ CUtensorMap tmap,
                              float* __restrict__ out,
                              int R, int box1, int L, int g) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int chunk = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk * 4u;

  if (threadIdx.x == 0) { mbar_init(&mbar, 1); mbar_init_fence(); }
  cl.sync();

  // 组内扇出：g 个连续 rank 一组；组首 = rank - (rank % g)
  int first = (g > 0) ? (rank - (rank % g)) : 0;
  uint16_t mask = (g > 0) ? (uint16_t)(((1u << g) - 1u) << first) : 0u;
  int is_issuer = (g == 0) || (rank == first);

  float acc = 0.f;
  uint32_t phase = 0;
  for (int r = 0; r < R; r++) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(&mbar, chunk_bytes * (uint32_t)L);
      if (is_issuer) {
        for (int j = 0; j < L; j++) {
          uint32_t dst = smem_u32(buf + (size_t)j * chunk);
          int c1 = (r * L + j) * box1;
          if (g > 0) tma_load_mcast_2d(&tmap, dst, &mbar, mask, 0, c1);
          else       tma_load_2d(&tmap, dst, &mbar, 0, c1);
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

// 单次 run：返回 mean ms/launch，并写出 maxerr
static double run_bcast(int C, int grid, int KB, int R, int L, int g, int iters,
                        double* maxerr, int* occ_out, int* regs_out) {
  int box1 = KB * 1024 / 4 / 256;
  int chunk = 256 * box1;
  if (L < 1) L = 1;
  if (g < 0) g = 0;
  if (g > C) g = C;
  if (g > 0 && (C % g) != 0) { printf("[bad: C%%g != 0] "); return -1; }
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
  void* kern = (void*)fanout_kernel;
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int occ = 0, regs = 0;
  {
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*)kern); regs = fa.numRegs;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, TH, smem);
  }
  void* args[] = {(void*)&tmap, (void*)&out, (void*)&R, (void*)&box1, (void*)&L, (void*)&g};
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
  if (argc < 2) { printf("usage: multicast_fanout B <C> <grid> <chunkKB> <R> <L> <g> <iters>\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];

  if (!strcmp(cmd, "B")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        L = atoi(argv[6]), g = atoi(argv[7]), iters = argc > 8 ? atoi(argv[8]) : 20;
    double err = -1; int occ = 0, regs = 0;
    double ms = run_bcast(C, grid, KB, R, L, g, iters, &err, &occ, &regs);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    double deliv = (double)grid * chunk * L * R;
    double nclusters = (double)grid / C;
    // 源侧请求：每组一次；plain 每 CTA 一次（组数=CTA 数）
    double groups = (g == 0) ? (double)C : (double)(C / g);
    double src_read = nclusters * groups * chunk * L * R;
    int nsm = num_sm();
    int eff = occ > 0 ? occ : 1;
    int per_sm = grid / nsm;
    if (per_sm < eff) eff = per_sm;
    if (eff < 1) eff = 1;
    double waves = (double)grid / (double)(nsm * eff);
    const char* tag = (g == 0) ? "plain" : "mcast";
    printf("B C=%2d grid=%4d chunk=%2dKB L=%2d R=%4d g=%2d(%s) it=%2d | regs=%3d occ=%2d "
           "eff=%2d waves=%4.2f | ", C, grid, KB, L, R, g, tag, iters, regs, occ, eff, waves);
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | per-blk %6.1f GB/s | src-read %7.1f GB/s | maxerr %.5f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, deliv / (ms / 1e3) / 1e9 / grid,
             src_read / (ms / 1e3) / 1e9, err);
    else printf("FAILED\n");
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
