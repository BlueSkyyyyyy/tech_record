// P33-1: 2SM 协作的 sm90 投影 —— 一对 CTA（cluster 内 2 个 CTA）如何把「一个逻辑 tile」
//         搬进两个 SM，对照 SM100 的 cta_group::2 TMA load（cp.async.bulk.tensor ... .cta_group::2）。
//         （目标卡 H100 / sm90a）
//
// 背景：docs/04 §18/§19/§20/§21 已把 cluster 的「交换（DSM）」「广播（TMA multicast）」「多维/子集掩码」
// 「进度前移原语」都量过。剩下 docs/04 §20.6 与「下一步」都点名的两条 SM100 专属原语：
//   SM100_TMA_2SM_LOAD（`cp.async.bulk.tensor ... .cta_group::2`，一个 CTA 发、tile 按 2 个 SM 分区投递）
//   与 tcgen05 2-SM MMA。SM100 指令在 H100 上不可执行，故本节点分两步：
//   (a) 资料分析：在 CUTLASS（DeepGEMM 内 third-party/cutlass，commit 057ca59）里核对 2SM 的语义；
//   (b) sm90 等价实测：把 2SM 的核心「一次发、按 SM 分区投递」用 sm90 原语投影出来 ——
//       rank0 用 `cp.async.bulk.shared::cluster.global` 把 tile A 搬进自己 SMEM、tile B 搬进对端
//       SMEM（`mapa.shared::cluster` 求对端地址与对端 mbarrier），与「两 CTA 各发各的」以及
//       「multicast 复制投递（两 CTA 拿到同一份）」公平对照。
//
// 编译：nvcc -O3 -arch=sm_90a src/two_sm_load.cu -o two_sm_load -lcuda
// 运行：见 harness/two_sm_load.sh（每配置独立进程，cluster launch failure 会 sticky 污染 context）
//
// 子命令：
//   L <C> <grid> <KB> <R> <mode>   分区/复制/独立载入对照
//     mode 0 indep  : 每个 CTA 用 cp.async.bulk 载入自己的 tile（2 次源读）
//     mode 1 mcast  : rank0 用 TMA multicast 把 tile A 复制投递到 2 个 CTA 的 SMEM（1 次源读、复制）
//     mode 2 split  : rank0 用 2 次 cp.async.bulk 把 tile A/B 分区投递到 2 个 CTA 的 SMEM（1 次源读、
//                     分区；= SM100 cta_group::2 的 sm90 投影）
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
__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
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
// 非 tensor bulk 拷贝 global -> cluster shared（dst / mbar 都可用 mapa 指到对端 CTA）
__device__ __forceinline__ void bulk_g2s(uint32_t dst_cluster, const void* g, uint32_t bytes,
                                         uint32_t mbar_cluster) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      ::"r"(dst_cluster), "l"((uint64_t)g), "r"(bytes), "r"(mbar_cluster) : "memory");
}
// TMA multicast 2D：一个 CTA 发、mask 内每个 CTA 的 SMEM 各收一份
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

// 一轮：
//   mode 0 indep：rank r 把 chunk (base + r) 用 bulk 搬进自己 SMEM
//   mode 1 mcast：rank0 把 chunk base 用 tensor multicast 复制给两个 CTA
//   mode 2 split：rank0 把 chunk base 搬进自己 SMEM、chunk base+1 搬进对端 SMEM
// 消费：每 CTA 读自己 SMEM 的 [0]、[128] 两元素求和写到 out[blockIdx.x]。
__global__ void load2sm_kernel(const __grid_constant__ CUtensorMap tmap,
                               const float* __restrict__ src, float* __restrict__ out,
                               int R, int box1, int mode) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  (void)C;
  int chunk_elems = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk_elems * 4u;

  if (threadIdx.x == 0) { mbar_init(&mbar, 1); mbar_init_fence(); }
  cl.sync();

  // cluster 内 pair id（C 恒为 2）
  int pair_id = (int)(blockIdx.x / 2);
  int npairs = (int)(gridDim.x / 2);

  float acc = 0.f;
  uint32_t phase = 0;
  for (int r = 0; r < R; r++) {
    long base = (long)r * (long)npairs * 2 + (long)pair_id * 2;  // chunk index 基（A=base, B=base+1）
    if (threadIdx.x == 0) {
      mbar_expect_tx(&mbar, chunk_bytes);
      if (mode == 0) {
        long ci = base + rank;
        bulk_g2s(smem_u32(buf), src + ci * (long)chunk_elems, chunk_bytes, smem_u32(&mbar));
      } else if (mode == 1) {
        if (rank == 0) {
          uint16_t mask = (uint16_t)((1u << 2) - 1u);
          tma_load_mcast_2d(&tmap, smem_u32(buf), &mbar, mask, 0, (int)(base * box1));
        }
      } else { // mode 2
        if (rank == 0) {
          // chunk A -> 自己 SMEM（自己 mbar）
          bulk_g2s(smem_u32(buf), src + base * (long)chunk_elems, chunk_bytes, smem_u32(&mbar));
          // chunk B -> 对端 SMEM（对端 mbar via mapa）
          uint32_t peer_buf = mapa_u32(smem_u32(buf), 1u);
          uint32_t peer_mbar = mapa_u32(smem_u32(&mbar), 1u);
          bulk_g2s(peer_buf, src + (base + 1) * (long)chunk_elems, chunk_bytes, peer_mbar);
        }
      }
    }
    mbar_wait(&mbar, phase);
    phase ^= 1u;
    acc += buf[0];
    acc += buf[TH];
    cl.sync();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
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

static double run_load2sm(int C, int grid, int KB, int R, int mode, double* maxerr) {
  int box1 = KB * 1024 / 4 / 256;
  int chunk_elems = 256 * box1;
  int npairs = grid / 2;
  long ntot = (long)chunk_elems * (long)R * (long)npairs * 2;
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

  size_t smem = (size_t)chunk_elems * 4;
  void* args[] = {(void*)&tmap, (void*)&src, (void*)&out, (void*)&R, (void*)&box1, (void*)&mode};
  cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, (void*)load2sm_kernel, args);
  if (e != cudaSuccess) { printf("[FAIL %s] ", cudaGetErrorString(e)); return -1; }
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  int it = 20;
  for (int i = 0; i < it; i++) {
    e = launch_cluster(C, dim3(grid), dim3(TH), smem, (void*)load2sm_kernel, args);
    if (e != cudaSuccess) { printf("[iter FAIL %s] ", cudaGetErrorString(e)); return -1; }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;

  {
    float* o = (float*)malloc((size_t)grid * sizeof(float));
    cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost);
    double err = 0;
    for (int g = 0; g < grid; g++) {
      int rank = g % 2;
      int pid = g / 2;
      double exp = 0;
      for (int r = 0; r < R; r++) {
        long base = (long)r * npairs * 2 + (long)pid * 2;
        long ci = (mode == 1) ? base : (base + rank);
        long p = ci * (long)chunk_elems;
        exp += (double)((p % 251) - 125) * 1e-3;
        exp += (double)(((p + TH) % 251) - 125) * 1e-3;
      }
      double d = o[g] - exp; if (d < 0) d = -d;
      if (d > err) err = d;
    }
    *maxerr = err;
    free(o);
  }
  cudaFree(src); cudaFree(out);
  return ms;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: two_sm_load L <C> <grid> <KB> <R> <mode>\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];
  if (!strcmp(cmd, "L")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        mode = atoi(argv[6]);
    double err = -1;
    double ms = run_load2sm(C, grid, KB, R, mode, &err);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    double npairs = (double)grid / 2;
    // 交付带宽：全 grid 收到的 SMEM 字节 / 时间
    double deliv;
    if (mode == 1) deliv = (double)grid * chunk * R;          // 复制：每 CTA 收整 chunk
    else           deliv = (double)grid * chunk * R;          // 分区/独立：每 CTA 收自己的 chunk
    // 源读模型：mode0 indep 每 pair 读 A、B 两块（2 次源读）；mode1 mcast 只读 A（1 次、复制）；
    // mode2 split 每 pair 读 A、B 两块（2 次源读，但只由 rank0 一个 CTA 发出）
    double src_read = npairs * chunk * R * (mode == 1 ? 1.0 : 2.0);
    printf("L C=%d grid=%3d chunk=%4dKB R=%3d mode=%d | %s", C, grid, KB, R, mode,
           ms < 0 ? "FAILED\n" : "");
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | src-read %7.1f GB/s | maxerr %.5f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, src_read / (ms / 1e3) / 1e9, err);
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
