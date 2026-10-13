// ============================================================================
// P168-1 (§93)：TMA multicast × 占用率 / 寄存器预算 —— 补 §19.6 / §86 明留的组合
//
// 背景：
//   - §19（P30-1）量了 cluster 内 TMA multicast 的广播（`cp.async.bulk.tensor ...
//     .multicast::cluster`），结论是「delivery 吞吐随 C 线性放大、在 ~6.6 TB/s 饱和，
//     两模式打平；multicast 只省源侧 L2 请求（2×）、DRAM 字节不变」。§19.6 明确留：
//     「deliver 饱和在 ~6.6 TB/s：这是 SMEM 写入 / TMA issue 的上限，不是源侧带宽；
//      故墙钟打平是饱和区间的结论」。但它**只跑过 grid=128（≈1 块/SM）**，
//     **没有把「占用率（blocks/SM）」当旋钮**（§19.2 A2 全是 grid=128/C）。
//   - §86（P161-1）把 `__maxnreg__` × cluster 叠加，得出「cluster 交换（DSM lap）
//     不吃占用率、差 ~3%」。但那是 **DSM 远端读/写** 路径，不是 **TMA 载入** 路径。
//
// 本节把 §19 的 multicast 广播与 §68–§85 的「占用率 / 寄存器预算」叠加，回答：
//   Q1（§B，占用率用 grid 钉死，小 SMEM）：**多块/SM 能不能抬高 delivery 聚合吞吐？**
//      还是它本就是**每 SM 常数**（SMEM 写 / TMA issue 的 per-SM 墙）→ 占用率不是旋钮？
//   Q2（§C，占用率用 SMEM padding 钉死，固定 grid）：换一个占用率口径，Q1 的结论是否一致？
//   Q3（§D，寄存器预算 `__maxnreg__(R)`）：寄存器预算是不是这一 load-path 内核的旋钮？
//      （对照 §69：数据面认的是寄存器预算 ILP；这里 TMA 由单线程 issue，怀疑不是。）
//   Q4（§E，ncu）：multicast 省 L2 请求这一结论，在不同占用率下是否成立？
//
// 载体：burst 版 TMA 广播（= §19 的 `bcast_kernel`），chunk 个 float 的 L 个 SMEM 槽，
//       每轮发 L 次 TMA、一个 mbarrier `expect_tx` 累计后等待。mode0 = 每 CTA 各发普通 TMA，
//       mode1 = 仅 rank0 发 multicast。动态 SMEM = L*chunk*4 + pad（pad 只抬占用率、不使用）。
//
// 编译：nvcc -O3 -arch=sm_90a --ptxas-options=-v src/multicast_occupancy.cu -o /tmp/multicast_occupancy -lcuda
// 用法：见 harness/multicast_occupancy.sh（每配置独立进程 —— cluster launch failure 会 sticky 污染 context）
//   O <L> <padKB>                 打印 mcast_nat 及各 __maxnreg__ 变体在该 SMEM 下的实测占用率
//   B <C> <grid> <KB> <R> <mode> <L> <padKB> <maxr>
//   P <C> <grid> <KB> <L> <padKB> <maxr>   cluster 感知占用率 API 读数（P175-1 §100，只读）
//   T <C> <grid> <KB> <R> <mode> <L> <padKB> <maxr> [warm]
//                                         逐块时间线：记录每块 (smid, t0, t1)，聚合出
//                                         起始/结束分位与驻留块时间线（P176-1 §101，
//                                         坐实/修正 §100 的「尾波」解释）。**不改既有语义**
//   目标卡 H100 / sm90a。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
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

// ---------------- 时间线探针（P176-1 §101） ----------------
// %globaltimer = 全局 synchronized 的纳秒计数器；%smid = 物理 SM id。
__device__ __forceinline__ unsigned long long gtimer_ns() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned int sm_id() {
  unsigned int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}

// ---------------- kernel body（模板参数只是为了让 __maxnreg__ 变体实例化） ----------------
template <int DUMMY>
__device__ __forceinline__ void mcast_body(const CUtensorMap& tmap,
                                           float* __restrict__ out,
                                           int R, int box1, int L, int mode) {
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

  uint16_t mask = (uint16_t)((1u << C) - 1u);
  float acc = 0.f;
  uint32_t phase = 0;
  for (int r = 0; r < R; r++) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(&mbar, chunk_bytes * (uint32_t)L);
      if (mode == 1) {
        if (rank == 0)
          for (int j = 0; j < L; j++)
            tma_load_mcast_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, mask,
                              0, (r * L + j) * box1);
      } else {
        for (int j = 0; j < L; j++)
          tma_load_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, 0,
                      (r * L + j) * box1);
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

// 自然寄存器版本（无属性）
__global__ void mcast_nat(const __grid_constant__ CUtensorMap tmap, float* __restrict__ out,
                          int R, int box1, int L, int mode) {
  mcast_body<0>(tmap, out, R, box1, L, mode);
}
// __maxnreg__(MAXR) 版本（每个实例独立编译）
#define DEF_MR(MAXR)                                                                    \
  __global__ void __maxnreg__(MAXR) mcast_mr_##MAXR(                                    \
      const __grid_constant__ CUtensorMap tmap, float* __restrict__ out,                \
      int R, int box1, int L, int mode) {                                              \
    mcast_body<MAXR>(tmap, out, R, box1, L, mode);                                     \
  }
DEF_MR(24) DEF_MR(32) DEF_MR(40) DEF_MR(64) DEF_MR(96)

// 逐块时间线版本（P176-1 §101）：语义与 mcast_body 逐字相同，只多记 (smid,t0,t1)。
// rec 布局：rec[3*blk+0]=smid, rec[3*blk+1]=t0(ns, 调度即测), rec[3*blk+2]=t1(ns, 循环结束)。
__global__ void mcast_tl(const __grid_constant__ CUtensorMap tmap, float* __restrict__ out,
                         unsigned long long* __restrict__ rec,
                         int R, int box1, int L, int mode) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;
  unsigned long long t0 = gtimer_ns();
  unsigned int sm = sm_id();

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int chunk = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk * 4u;

  if (threadIdx.x == 0) { mbar_init(&mbar, 1); mbar_init_fence(); }
  cl.sync();

  uint16_t mask = (uint16_t)((1u << C) - 1u);
  float acc = 0.f;
  uint32_t phase = 0;
  for (int r = 0; r < R; r++) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(&mbar, chunk_bytes * (uint32_t)L);
      if (mode == 1) {
        if (rank == 0)
          for (int j = 0; j < L; j++)
            tma_load_mcast_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, mask,
                              0, (r * L + j) * box1);
      } else {
        for (int j = 0; j < L; j++)
          tma_load_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, 0,
                      (r * L + j) * box1);
      }
    }
    mbar_wait(&mbar, phase);
    phase ^= 1u;
    acc += buf[threadIdx.x];
    acc += buf[threadIdx.x + TH];
    cl.sync();
  }
  if (threadIdx.x == 0) {
    unsigned long long t1 = gtimer_ns();
    out[blockIdx.x] = acc;
    rec[(size_t)blockIdx.x * 3 + 0] = (unsigned long long)sm;
    rec[(size_t)blockIdx.x * 3 + 1] = t0;
    rec[(size_t)blockIdx.x * 3 + 2] = t1;
  }
}

static void* pick_kernel(int maxr) {
  switch (maxr) {
    case 0:  return (void*)mcast_nat;
    case 24: return (void*)mcast_mr_24;
    case 32: return (void*)mcast_mr_32;
    case 40: return (void*)mcast_mr_40;
    case 64: return (void*)mcast_mr_64;
    case 96: return (void*)mcast_mr_96;
    default: printf("[bad maxr %d]\n", maxr); exit(2);
  }
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
static double run_bcast(int C, int grid, int KB, int R, int mode, int L, int padKB,
                        int maxr, double* maxerr, int* occ_out, int* regs_out) {
  int box1 = KB * 1024 / 4 / 256;
  int chunk = 256 * box1;
  if (L < 1) L = 1;
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

  size_t smem = (size_t)L * chunk * 4 + (size_t)padKB * 1024;
  void* kern = pick_kernel(maxr);
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int occ = 0, regs = 0;
  {
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*)kern); regs = fa.numRegs;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, TH, smem);
  }
  void* args[] = {(void*)&tmap, (void*)&out, (void*)&R, (void*)&box1, (void*)&L, (void*)&mode};
  cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
  if (e != cudaSuccess) { printf("[FAIL %s] ", cudaGetErrorString(e)); return -1; }
  CE(cudaDeviceSynchronize());

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  int it = 20;
  for (int i = 0; i < it; i++) {
    e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
    if (e != cudaSuccess) { printf("[iter FAIL %s] ", cudaGetErrorString(e)); return -1; }
  }
  CE(cudaDeviceSynchronize());
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;

  {
    float* o = (float*)malloc((size_t)grid * sizeof(float));
    CE(cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0;
    for (int g = 0; g < grid; g++) {
      double exp = 0;
      for (int r = 0; r < R; r++) {
        long base = (long)r * L * chunk;
        exp += (double)((base % 251) - 125) * 1e-3;
        exp += (double)(((base + 128) % 251) - 125) * 1e-3;
      }
      double d = o[g] - exp; if (d < 0) d = -d;
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
  if (argc < 2) { printf("usage: multicast_occupancy <O|B> ...\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];

  if (!strcmp(cmd, "O")) {
    int L = atoi(argv[2]), padKB = argc > 3 ? atoi(argv[3]) : 0;
    int chunk = 256 * 8;  // 8KB chunk 假定 box1 = 8KB/4/256 = 8
    size_t smem = (size_t)L * chunk * 4 + (size_t)padKB * 1024;
    int nsm = num_sm();
    printf("O L=%d padKB=%d -> smem=%zuKB numSM=%d\n", L, padKB, smem / 1024, nsm);
    const int mrs[] = {0, 24, 32, 40, 64, 96};
    for (int i = 0; i < 6; i++) {
      void* kern = pick_kernel(mrs[i]);
      if (smem >= 48 * 1024)
        cudaFuncSetAttribute((const void*)kern,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*)kern);
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, TH, smem);
      // 寄存器上限占用率（相对 SMEM 上限）：65536/(regs*TH)
      int reg_occ = 65536 / (fa.numRegs * TH);
      printf("  maxr=%3d numRegs=%3d localB=%5zu occ=%2d block/SM (smem) | reg_occ=%2d\n",
             mrs[i], fa.numRegs, fa.localSizeBytes, occ, reg_occ);
    }
  } else if (!strcmp(cmd, "B")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        mode = atoi(argv[6]), L = atoi(argv[7]), padKB = atoi(argv[8]), maxr = atoi(argv[9]);
    double err = -1; int occ = 0, regs = 0;
    double ms = run_bcast(C, grid, KB, R, mode, L, padKB, maxr, &err, &occ, &regs);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    double deliv = (double)grid * chunk * L * R;
    double nclusters = (double)grid / C;
    double src_read = nclusters * chunk * L * R * (mode == 1 ? 1.0 : (double)C);
    int nsm = num_sm();
    int eff = occ > 0 ? occ : 1;
    int per_sm = grid / nsm;
    if (per_sm < eff) eff = per_sm;
    if (eff < 1) eff = 1;
    double waves = (double)grid / (double)(nsm * eff);
    printf("B C=%2d grid=%4d chunk=%2dKB L=%2d pad=%3dKB maxr=%3d mode=%d | regs=%3d occ=%2d "
           "eff=%2d waves=%4.2f | ", C, grid, KB, L, padKB, maxr, mode, regs, occ, eff, waves);
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | per-blk %6.1f GB/s | src-read %7.1f GB/s | maxerr %.5f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, deliv / (ms / 1e3) / 1e9 / grid,
             src_read / (ms / 1e3) / 1e9, err);
    else printf("FAILED\n");
  } else if (!strcmp(cmd, "P")) {
    // P175-1（§100）：cluster 感知占用率 API（cudaOccupancyMaxActiveClusters）的读数，
    // 与 blocks-per-SM 占用率 API（cudaOccupancyMaxActiveBlocksPerMultiprocessor，不建模
    // cluster 的 GPC 连续放置）对同一配置对账。**只读，不改任何既有语义。**
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]),
        L = atoi(argv[5]), padKB = atoi(argv[6]), maxr = atoi(argv[7]);
    int box1 = KB * 1024 / 4 / 256;
    int chunk = 256 * box1;
    size_t smem = (size_t)L * chunk * 4 + (size_t)padKB * 1024;
    void* kern = pick_kernel(maxr);
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (smem >= 48 * 1024)
      cudaFuncSetAttribute((const void*)kern,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, TH, smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(TH);
    cfg.dynamicSmemBytes = smem; cfg.stream = 0;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    int nclu = -1;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&nclu, (const void*)kern, &cfg);
    int nsm = num_sm();
    double req_clu = (double)grid / C;
    double blk_sm = (nclu > 0) ? (double)nclu * C / nsm : -1;
    printf("P C=%2d grid=%4d KB=%2d L=%2d pad=%3dKB maxr=%3d smem=%zuKB | occ=%2d blk/SM | "
           "maxActiveClusters=%d (%s) | clusters_req=%.0f | cluster_occ_blocks/SM=%.2f\n",
           C, grid, KB, L, padKB, maxr, smem / 1024, occ, nclu,
           e == cudaSuccess ? "ok" : cudaGetErrorString(e), req_clu, blk_sm);
  } else if (!strcmp(cmd, "T")) {
    // P176-1（§101）：逐块时间线。参数/语义与 B 相同，唯一新增 = 记录每块 (smid,t0,t1)。
    // 用于坐实/修正 §100 的「尾波」解释（§100.8(5)）。
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        mode = atoi(argv[6]), L = atoi(argv[7]), padKB = atoi(argv[8]), maxr = atoi(argv[9]);
    int warm = argc > 10 ? atoi(argv[10]) : 5;
    int box1 = KB * 1024 / 4 / 256;
    int chunk = 256 * box1;
    if (L < 1) L = 1;
    long ntot = 256L * (long)box1 * (long)R * L;
    float *src, *out; unsigned long long* rec;
    CE(cudaMalloc(&src, ntot * 4));
    CE(cudaMalloc(&out, (size_t)grid * sizeof(float)));
    CE(cudaMalloc(&rec, (size_t)grid * 3 * sizeof(unsigned long long)));
    {
      float* h = (float*)malloc(ntot * 4);
      for (long i = 0; i < ntot; i++) h[i] = (float)((i % 251) - 125) * 1e-3f;
      CE(cudaMemcpy(src, h, ntot * 4, cudaMemcpyHostToDevice)); free(h);
    }
    CUtensorMap tmap = make_tmap_2d(src, 256, ntot / 256, 256, box1);
    size_t smem = (size_t)L * chunk * 4 + (size_t)padKB * 1024;
    cudaFuncSetAttribute((const void*)mcast_tl,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (smem >= 48 * 1024)
      cudaFuncSetAttribute((const void*)mcast_tl,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)mcast_tl, TH, smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(TH);
    cfg.dynamicSmemBytes = smem; cfg.stream = 0;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    int nclu = -1; cudaOccupancyMaxActiveClusters(&nclu, (const void*)mcast_tl, &cfg);

    void* args[] = {(void*)&tmap, (void*)&out, (void*)&rec,
                    (void*)&R, (void*)&box1, (void*)&L, (void*)&mode};
    int fail = 0;
    for (int i = 0; i < warm + 1; i++) {
      cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, (void*)mcast_tl, args);
      if (e != cudaSuccess) { printf("T C=%d grid=%d [launch FAIL %s]\n",
                                     C, grid, cudaGetErrorString(e)); fail = 1; break; }
      CE(cudaDeviceSynchronize());
    }
    if (fail) { cudaFree(src); cudaFree(out); cudaFree(rec); return 0; }

    std::vector<unsigned long long> r((size_t)grid * 3);
    CE(cudaMemcpy(r.data(), rec, (size_t)grid * 3 * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    // 正确性哨兵（与 B 命令同式）：out[blk] = sum_r (buf[0] + buf[TH])，buf 首两行来自 src
    double err = 0;
    {
      std::vector<float> o(grid);
      CE(cudaMemcpy(o.data(), out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
      for (int g = 0; g < grid; g++) {
        double exp = 0;
        for (int rr = 0; rr < R; rr++) {
          long base = (long)rr * L * chunk;
          exp += (double)((base % 251) - 125) * 1e-3;
          exp += (double)(((base + 128) % 251) - 125) * 1e-3;
        }
        double d = o[g] - exp; if (d < 0) d = -d;
        if (d > err) err = d;
      }
    }
    int nsm = num_sm();
    unsigned long long tmin = r[1], tmax = r[2];
    unsigned maxsm = 0;
    for (int b = 0; b < grid; b++) {
      unsigned long long a = r[(size_t)b * 3 + 1], z = r[(size_t)b * 3 + 2];
      if (a < tmin) tmin = a; if (z > tmax) tmax = z;
      unsigned sm = (unsigned)r[(size_t)b * 3 + 0]; if (sm > maxsm) maxsm = sm;
    }
    unsigned long long span = tmax > tmin ? tmax - tmin : 1;
    std::vector<unsigned long long> st(grid), en(grid);
    for (int b = 0; b < grid; b++) {
      st[b] = r[(size_t)b * 3 + 1] - tmin; en[b] = r[(size_t)b * 3 + 2] - tmin;
    }
    int peak_req_clu = (nclu > 0) ? nclu : -1;
    printf("T C=%2d grid=%4d KB=%2d L=%2d pad=%3dKB maxr=%3d mode=%d warm=%d | occ=%2d "
           "maxActiveClusters=%d | numSM=%d span=%llu ns (%.2f us) maxerr=%.5f\n",
           C, grid, KB, L, padKB, maxr, mode, warm, occ, peak_req_clu, nsm, span, span / 1e3, err);
    {
      std::vector<unsigned long long> s = st; std::sort(s.begin(), s.end());
      printf("  start-decile(us):");
      for (int p = 10; p <= 100; p += 10) {
        int idx = (int)((long long)p * grid / 100) - 1; if (idx < 0) idx = 0;
        printf(" p%03d=%.2f", p, s[idx] / 1e3);
      }
      printf("\n");
      std::vector<unsigned long long> f = en; std::sort(f.begin(), f.end());
      printf("  finish-decile(us):");
      for (int p = 10; p <= 100; p += 10) {
        int idx = (int)((long long)p * grid / 100) - 1; if (idx < 0) idx = 0;
        printf(" p%03d=%.2f", p, f[idx] / 1e3);
      }
      printf("\n");
    }
    // 驻留块时间线：50 桶，统计每桶内 [t0,t1) 覆盖的块数
    const int NB = 50;
    std::vector<int> res(NB, 0);
    for (int b = 0; b < grid; b++) {
      int i0 = (int)((unsigned long long)st[b] * NB / span);
      int i1 = (int)((unsigned long long)en[b] * NB / span);
      if (i0 > NB - 1) i0 = NB - 1;
      if (i1 > NB - 1) i1 = NB - 1;
      for (int k = i0; k <= i1; k++) res[k]++;
    }
    int peak = 0; for (int k = 0; k < NB; k++) if (res[k] > peak) peak = res[k];
    printf("  resident(blocks)/50bins (peak=%d = %.2f blk/SM):", peak,
           peak ? (double)peak / nsm : 0.0);
    for (int k = 0; k < NB; k++) printf(" %d", res[k]);
    printf("\n");
    std::vector<int> smc(maxsm + 1, 0);
    for (int b = 0; b < grid; b++) smc[(unsigned)r[(size_t)b * 3 + 0]]++;
    int used = 0, max_per_sm = 0;
    for (unsigned k = 0; k <= maxsm; k++) if (smc[k]) { used++; if (smc[k] > max_per_sm) max_per_sm = smc[k]; }
    int late = 0; unsigned long long thr = (unsigned long long)(span * 0.2);
    for (int b = 0; b < grid; b++) if (st[b] > thr) late++;
    printf("  sm_used=%d/%d | max_blocks_per_SM=%d | late_start(>20%%span)=%d/%d (%.1f%%)\n",
           used, nsm, max_per_sm, late, grid, 100.0 * late / grid);
    cudaFree(src); cudaFree(out); cudaFree(rec);
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
