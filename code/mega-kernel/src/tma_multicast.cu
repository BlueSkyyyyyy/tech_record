// P30-1: cluster 内 TMA multicast —— megakernel 的「权重/掩码一次读、多 SMEM 收」广播原语
//         （目标卡 H100 / sm90）。
//
// 背景：docs/04 §18.6 明确留了「cluster 内的 mbarrier/TMA 组合未覆盖」的尾巴。P29-1 量了
// DSM（跨 SM 读邻居 SMEM，是「已到片上之后的交换」）；本节量 **load path**：把一块全局数据
// 一次读入、同时投递到 cluster 内 C 个 CTA 的 SMEM（cp.async.bulk.tensor ...multicast::cluster），
// 与「每个 CTA 各自 TMA 载入同一块」对照。生产案例 DeepGEMM Mega MoE/Gate 用 TMA multicast 在
// cluster 内广播权重/打分（docs/02 §1.2），这是它们相对「每 CTA 各读一遍」省 HBM 流量的机制。
//
// 编译：nvcc -O3 -arch=sm_90 src/tma_multicast.cu -o tma_multicast -lcuda
// 运行：见 harness/tma_multicast.sh（每配置独立进程，cluster launch failure 会 sticky 污染 context）
//
// 子命令：
//   B <C> <grid> <KB> <R> <mode>   cluster 广播：mode 0 = 每 CTA 独立 TMA，1 = rank0 multicast
//   P <C> <KB> <R>                 cluster.sync() 成本对照（复用 P29-1 机制，仅 sanity）
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

static CUtensorMap make_tmap_2d(const float* g, uint64_t d0, uint64_t d1,
                                uint32_t b0, uint32_t b1) {
  CUtensorMap tmap;
  uint64_t globalDim[2] = {d0, d1};
  uint64_t globalStrides[1] = {d0 * 4};      // bytes, for dim1
  uint32_t boxDim[2] = {b0, b1};
  uint32_t elemStrides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)g,
      globalDim, globalStrides, boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { const char* s = "?"; cuGetErrorString(r, &s);
    printf("[tmap encode FAIL] %s (d0=%llu d1=%llu b0=%u b1=%u)\n", s,
           (unsigned long long)d0, (unsigned long long)d1, b0, b1); exit(2); }
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
// 普通 TMA（每个 CTA 各读一块）
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, uint32_t dst,
                                            uint64_t* mbar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      ::"r"(dst), "l"((uint64_t)tmap), "r"(smem_u32(mbar)), "r"(c0), "r"(c1)
      : "memory");
}
// TMA multicast（一个 CTA 发、cluster 内多个 CTA 的 SMEM 各收一份）
__device__ __forceinline__ void tma_load_mcast_2d(const CUtensorMap* tmap, uint32_t dst,
                                                  uint64_t* mbar, uint16_t mask,
                                                  int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(dst), "l"((uint64_t)tmap), "r"(smem_u32(mbar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}

static const int TH = 128;

// burst 版：smem 切成 L 个 chunk 槽；每轮对 L 个槽各发一次 TMA（同一块的 L 个不同分段），
// 一个 mbarrier 累计 L*chunk 字节、一次等待 → TMA 完全重叠（深度 = L），测峰值广播吞吐。
// mode 0：每个 CTA 各自发 L 次 TMA（C× 次源读）
// mode 1：仅 rank0 发 L 次 multicast，一次投递到 C 个 CTA 的 SMEM（1× 次源读）
__global__ void bcast_kernel(const __grid_constant__ CUtensorMap tmap,
                             float* __restrict__ out, int R, int box1, int L, int mode) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int chunk = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk * 4u;

  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_init_fence();
  }
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
    // consume：每线程固定索引读（避免 P29-1 §18.6 的变界远端读编译器坑）
    acc += buf[threadIdx.x];
    acc += buf[threadIdx.x + TH];
    cl.sync();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

__global__ void cluster_sync_kernel(int R) {
  cg::cluster_group cl = cg::this_cluster();
  for (int r = 0; r < R; r++) cl.sync();
}

static cudaError_t launch_cluster(int C, dim3 grid, dim3 block, size_t smem,
                                  void* kern, void** args) {
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
  return cudaLaunchKernelExC(&cfg, kern, args);
}

// 单次 run：返回 mean ms/launch，并写出 maxerr（对照解析参考）
// L 个 chunk 槽（smem = L*chunk），每轮读 L 个不同分段；source 需要 R*L 段。
static double run_bcast(int C, int grid, int KB, int R, int mode, int* Lout, double* maxerr) {
  int box1 = KB * 1024 / 4 / 256;         // rows of 256 floats
  int chunk = 256 * box1;
  // L：让 smem ≈ 192 KB
  int L = (192 * 1024) / (chunk * 4);
  if (L < 1) L = 1;
  if (L > 2048) L = 2048;
  long ntot = 256L * (long)box1 * (long)R * L;
  float *src, *out;
  cudaMalloc(&src, ntot * 4);
  cudaMalloc(&out, (size_t)grid * sizeof(float));
  {
    float* h = (float*)malloc(ntot * 4);
    for (long i = 0; i < ntot; i++) h[i] = (float)((i % 251) - 125) * 1e-3f;
    cudaMemcpy(src, h, ntot * 4, cudaMemcpyHostToDevice);
    free(h);
  }
  CUtensorMap tmap = make_tmap_2d(src, 256, ntot / 256, 256, box1);

  size_t smem = (size_t)L * chunk * 4;
  void* args[] = {(void*)&tmap, (void*)&out, (void*)&R, (void*)&box1, (void*)&L, (void*)&mode};
  cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, (void*)bcast_kernel, args);
  if (e != cudaSuccess) { printf("[FAIL %s] ", cudaGetErrorString(e)); return -1; }
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  int it = 20;
  for (int i = 0; i < it; i++) {
    e = launch_cluster(C, dim3(grid), dim3(TH), smem, (void*)bcast_kernel, args);
    if (e != cudaSuccess) { printf("[iter FAIL %s] ", cudaGetErrorString(e)); return -1; }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;

  // 正确性：thread0 每轮 acc += buf[0] + buf[128]（第 0 个槽的两个元素）。
  // 第 r 轮槽 0 的 src 全局下标 = (r*L)*chunk + j（连续 chunk 元素）。
  {
    float* o = (float*)malloc((size_t)grid * sizeof(float));
    cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost);
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
  *Lout = L;
  cudaFree(src); cudaFree(out);
  return ms;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: tma_multicast <B|P> ...\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];

  if (!strcmp(cmd, "B")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        mode = atoi(argv[6]);
    double err = -1;
    int L = 0;
    double ms = run_bcast(C, grid, KB, R, mode, &L, &err);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    // 交付带宽：全 grid 每轮写 L 个 chunk 到 SMEM，共 R 轮（SMEM 收到的总字节/时间）
    double deliv = (double)grid * chunk * L * R;
    // 源侧「读请求」模型：multicast 每 cluster 读 L*chunk，独立则每 CTA 读 L*chunk
    double nclusters = (double)grid / C;
    double src_read = nclusters * chunk * L * R * (mode == 1 ? 1.0 : (double)C);
    printf("B C=%2d grid=%3d chunk=%4dKB L=%2d R=%3d mode=%d | %s", C, grid, KB, L, R, mode,
           ms < 0 ? "FAILED\n" : "");
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | src-read %7.1f GB/s | maxerr %.5f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, src_read / (ms / 1e3) / 1e9, err);
  } else if (!strcmp(cmd, "P")) {
    int C = atoi(argv[2]), KB = atoi(argv[3]), R = atoi(argv[4]);
    (void)KB;
    int grid = C;
    void* args[] = {(void*)&R};
    void* kern = (void*)cluster_sync_kernel;
    cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), 0, kern, args);
    if (e != cudaSuccess) { printf("P launch err %s\n", cudaGetErrorString(e)); return 2; }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; i++) launch_cluster(C, dim3(grid), dim3(TH), 0, kern, args);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("P C=%2d grid=%3d R=%d | cluster.sync %.1f ns/sync\n", C, grid, R, ms * 1e6 / R);
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
