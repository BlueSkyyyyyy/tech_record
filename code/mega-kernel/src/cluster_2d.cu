// P31-1: 2D/3D Thread Block Cluster —— megakernel 里「多维 SM 组」的 rank 布局、
//        DSM 寻址、TMA multicast 掩码与局部屏障（目标卡 H100 / sm90a）。
//
// 背景：P29-1（docs/04 §18）与 P30-1（§19）只用了 1D cluster（clusterDim.x），
// 两条局限都明确留了同一个尾巴：
//   §18.6/§19.6：未测 2D/3D cluster、cluster 内 TMA multicast 的（子集）掩码。
// 而生产案例 DeepGEMM 的 cluster 是**二维**的：Mega MoE 用 cluster_dim=(2,1,1)
// （csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp:138），Mega Gate 用
// cluster_dim = num_mma_ctas（docs/02 §1.2）。cluster 一旦不是 1D，就出现两个新问题：
//   (1) rank 到 (x,y,z) 的映射（决定 map_shared_rank / multicast mask 怎么算）；
//   (2) multicast 的 16-bit mask 是按**线性 rank** 编址的，要「只广播给某一行/某一列」
//       就得按 rank 布局拼 mask。
//
// 本程序回答四件事：(1) 2D/3D cluster 的线性 rank 布局是什么？(2) DSM 环形交换在 2D
// cluster 下是否等同 1D（按 rank）？(3) TMA multicast 的**子集掩码**（整群 vs 逐行）语义
// 对不对、源侧流量差多少？(4) cluster.sync 成本随 2D 形状怎么变？
//
// 编译：nvcc -O3 -arch=sm_90a src/cluster_2d.cu -o cluster_2d -lcuda
// 运行：见 harness/cluster_2d.sh（每配置独立进程，cluster launch failure 会 sticky）
//
// 子命令：
//   A <cx> <cy> <cz> <gx> <gy> <gz>        rank 布局（打印每 block 的 rank / 期望 rank）
//   S <cx> <cy> <K> <R>                    DSM 环形段位移（2D cluster，grid 自动配）
//   M <cx> <cy> <gridx> <gridy> <KB> <R> <mode>
//                                           TMA multicast：0=独立 / 1=整群 / 2=逐行子集
//   P <cx> <cy> <gx> <gy> <R>              cluster.sync() 成本（ns/sync）
//   C2 <cx> <cy> <cz> <gx> <gy> <gz>       cluster 容量：尝试 launch，打印 ok/错误
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

static const int TH = 128;

// ---------------- launch helper（支持 2D/3D cluster dim）----------------
template <typename K, typename... Args>
static cudaError_t launch_cluster3(dim3 cd, dim3 grid, dim3 block, size_t smem,
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
  at[0].val.clusterDim.x = cd.x;
  at[0].val.clusterDim.y = cd.y;
  at[0].val.clusterDim.z = cd.z;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, args...);
  if (e != cudaSuccess) return e;
  return cudaDeviceSynchronize();
}

template <typename K, typename... Args>
static double time_cluster3(dim3 cd, dim3 grid, dim3 block, size_t smem, int iters,
                            K kern, Args... args) {
  cudaError_t e = launch_cluster3(cd, grid, block, smem, kern, args...);
  if (e != cudaSuccess) { printf("[FAIL:%s] ", cudaGetErrorString(e)); return -1; }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    e = launch_cluster3(cd, grid, block, smem, kern, args...);
    if (e != cudaSuccess) { printf("[iter FAIL:%s] ", cudaGetErrorString(e)); return -1; }
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}

// ---------------- Section A：rank 布局 ----------------
// 每 block 记录 block_rank() 与它自己的 (blockIdx 在 cluster 内的坐标)；host 拼出
// 期望 rank = x + y*cx + z*cx*cy 并逐 block 比对。
__global__ void layout_kernel(int* rank_out, int* xyz_out, int cx, int cy, int cz) {
  (void)cz;
  cg::cluster_group cl = cg::this_cluster();
  int rank = cl.block_rank();
  int lx = blockIdx.x % cx, ly = blockIdx.y % cy, lz = blockIdx.z % cz;
  int idx = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  rank_out[idx] = rank;
  xyz_out[idx] = (lz * 10000 + ly * 100 + lx);   // 坐标打包
}

// ---------------- Section S：2D cluster 的 DSM 环形段位移 ----------------
// 与 P29-1 §18 的 Sd 相同机制（按线性 rank 环形搬），但 cluster 是 (cx,cy)。
// 每 block 拥有 seg=K*TH 个 float，初值 = gid*1000 + j；双缓冲 ping-pong，每轮一次
// cluster.sync()；R 轮后 block gid 的段 = 初始时 rank (rank-R)%C 的段。
template <int K>
__global__ void shift2d_kernel(float* __restrict__ out, int R) {
  extern __shared__ float sm[];
  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  long gid = blockIdx.y * gridDim.x + blockIdx.x;
  const int seg = K * TH;
  float* buf[2]; float* nbuf[2];
  buf[0] = sm; buf[1] = sm + seg;
  float* base = cl.map_shared_rank(sm, (rank + C - 1) % C);
  nbuf[0] = base; nbuf[1] = base + seg;
#pragma unroll
  for (int k = 0; k < K; k++) buf[0][threadIdx.x + k * TH] = (float)(gid * 1000 + threadIdx.x + k * TH);
  cl.sync();
  for (int r = 0; r < R; r++) {
    int cur = r & 1;
#pragma unroll
    for (int k = 0; k < K; k++) buf[cur ^ 1][threadIdx.x + k * TH] = nbuf[cur][threadIdx.x + k * TH];
    cl.sync();
  }
  if (threadIdx.x == 0) out[gid] = buf[R & 1][0];
}

// ---------------- Section M：2D cluster 的 TMA multicast（整群 vs 逐行子集）----------
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

// cx: cluster 在 x 方向的尺寸（= rank 布局的步长）。
// mode 0：每 CTA 各发普通 TMA（独立）
// mode 1：仅 rank0 发 multicast，mask=全群（整群广播）
// mode 2：**逐行**——每个 row 的 rank0（rank%cx==0）发 multicast，mask=该行的 cx 个 rank，
//         且各 row 读**各自不同**的源区域（c1 里带 row 偏移）→ 校验「掩码确实限制了投递范围」。
__global__ void bcast2d_kernel(const __grid_constant__ CUtensorMap tmap,
                               float* __restrict__ out, int R, int box1, int L,
                               int mode, int cx, int RL) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int row = rank / cx;                       // 2D 布局：rank = x + y*cx
  int chunk = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk * 4u;

  if (threadIdx.x == 0) { mbar_init(&mbar, 1); mbar_init_fence(); }
  cl.sync();

  float acc = 0.f;
  uint32_t phase = 0;
  for (int r = 0; r < R; r++) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(&mbar, chunk_bytes * (uint32_t)L);
      if (mode == 0) {
        for (int j = 0; j < L; j++)
          tma_load_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, 0,
                      (r * L + j) * box1);
      } else if (mode == 1) {
        if (rank == 0) {
          uint16_t mask = (uint16_t)((1u << C) - 1u);
          for (int j = 0; j < L; j++)
            tma_load_mcast_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, mask,
                              0, (r * L + j) * box1);
        }
      } else {                               // mode 2：逐行子集广播
        if ((rank % cx) == 0) {
          uint16_t mask = (uint16_t)(((1u << cx) - 1u) << (row * cx));
          int base = row * RL + r * L;       // 每行读各自区域
          for (int j = 0; j < L; j++)
            tma_load_mcast_2d(&tmap, smem_u32(buf + (size_t)j * chunk), &mbar, mask,
                              0, (base + j) * box1);
        }
      }
    }
    mbar_wait(&mbar, phase);
    phase ^= 1u;
    acc += buf[threadIdx.x];
    acc += buf[threadIdx.x + TH];
    cl.sync();
  }
  if (threadIdx.x == 0) out[blockIdx.y * gridDim.x + blockIdx.x] = acc;
}

__global__ void cluster_sync2d_kernel(int R) {
  cg::cluster_group cl = cg::this_cluster();
  for (int r = 0; r < R; r++) cl.sync();
}

// ---------------- host 侧 driver ----------------
static double run_shift2d(int cx, int cy, int K, int R, float* maxerr) {
  int C = cx * cy;
  int grid = 128;                 // 132 不一定是 C 的倍数，用 128
  int ky = 2, kx = (grid / C) / ky;   // grid=(cx*kx, cy*ky)，共 128 block
  int gx = cx * kx, gy = cy * ky;
  int seg = K * TH;
  float* out; cudaMalloc(&out, (size_t)grid * sizeof(float));
  double t = -1;
  size_t smem = (size_t)2 * seg * 4;
  switch (K) {
    case 1: t = time_cluster3(dim3(cx, cy, 1), dim3(gx, gy, 1), dim3(TH), smem, 20, shift2d_kernel<1>, out, R); break;
    case 2: t = time_cluster3(dim3(cx, cy, 1), dim3(gx, gy, 1), dim3(TH), smem, 20, shift2d_kernel<2>, out, R); break;
    case 4: t = time_cluster3(dim3(cx, cy, 1), dim3(gx, gy, 1), dim3(TH), smem, 20, shift2d_kernel<4>, out, R); break;
    case 8: t = time_cluster3(dim3(cx, cy, 1), dim3(gx, gy, 1), dim3(TH), smem, 20, shift2d_kernel<8>, out, R); break;
    default: printf("K must be 1/2/4/8\n"); cudaFree(out); return -1;
  }
  if (t >= 0) {
    float* h = (float*)malloc((size_t)grid * sizeof(float));
    cudaMemcpy(h, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost);
    double err = 0;
    for (int g = 0; g < grid; g++) {
      int bx = g % gx, by = g / gx;
      int lx = bx % cx, ly = by % cy;
      int rank = lx + ly * cx;
      int srank = (((rank - R) % C) + C) % C;
      int slx = srank % cx, sly = srank / cx;
      int cbx = bx - lx, cby = by - ly;
      long src = (long)(cby + sly) * gx + (cbx + slx);
      double expect = (double)(src * 1000);
      double d = h[g] - expect; if (d < 0) d = -d;
      if (d > err) err = d;
    }
    *maxerr = err; free(h);
  }
  cudaFree(out);
  return t;
}

static double run_bcast2d(int cx, int cy, int gx, int gy, int KB, int R, int mode,
                          int* Lout, double* maxerr) {
  int C = cx * cy;
  int grid = gx * gy;
  int box1 = KB * 1024 / 4 / 256;
  int chunk = 256 * box1;
  int L = (192 * 1024) / (chunk * 4);
  if (L < 1) L = 1;
  if (L > 2048) L = 2048;
  int RL = R * L;                                  // 每行区域大小（行数）
  long ntot = 256L * (long)box1 * (long)R * (long)L * (long)(mode == 2 ? cy : 1);
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
  cudaError_t e = launch_cluster3(dim3(cx, cy, 1), dim3(gx, gy), dim3(TH), smem,
                                  bcast2d_kernel, tmap, out, R, box1, L, mode, cx, RL);
  if (e != cudaSuccess) { printf("[FAIL %s] ", cudaGetErrorString(e)); return -1; }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  int it = 20;
  for (int i = 0; i < it; i++) {
    e = launch_cluster3(dim3(cx, cy, 1), dim3(gx, gy), dim3(TH), smem,
                        bcast2d_kernel, tmap, out, R, box1, L, mode, cx, RL);
    if (e != cudaSuccess) { printf("[iter FAIL %s] ", cudaGetErrorString(e)); return -1; }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;

  {
    float* o = (float*)malloc((size_t)grid * sizeof(float));
    cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost);
    double err = 0;
    for (int by = 0; by < gy; by++) {
      for (int bx = 0; bx < gx; bx++) {
        int gid = by * gx + bx;
        int row = (by % cy);                       // 该 block 在其 cluster 内的 row
        double exp = 0;
        for (int r = 0; r < R; r++) {
          long brow = (mode == 2 ? (long)row * RL : 0L) + (long)r * L;
          long base = brow * chunk;
          exp += (double)((base % 251) - 125) * 1e-3;
          exp += (double)(((base + 128) % 251) - 125) * 1e-3;
        }
        double d = o[gid] - exp; if (d < 0) d = -d;
        if (d > err) err = d;
      }
    }
    *maxerr = err; free(o);
  }
  *Lout = L;
  cudaFree(src); cudaFree(out);
  return ms;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: cluster_2d <A|S|M|P|C2> ...\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];

  if (!strcmp(cmd, "A")) {
    int cx = atoi(argv[2]), cy = atoi(argv[3]), cz = atoi(argv[4]);
    int gx = atoi(argv[5]), gy = atoi(argv[6]), gz = atoi(argv[7]);
    int grid = gx * gy * gz;
    int *rank, *xyz;
    cudaMalloc(&rank, grid * sizeof(int)); cudaMalloc(&xyz, grid * sizeof(int));
    cudaError_t e = launch_cluster3(dim3(cx, cy, cz), dim3(gx, gy, gz), dim3(TH), 0,
                                    layout_kernel, rank, xyz, cx, cy, cz);
    if (e != cudaSuccess) { printf("A launch err %s\n", cudaGetErrorString(e)); return 2; }
    int* hr = (int*)malloc(grid * sizeof(int));
    int* hx = (int*)malloc(grid * sizeof(int));
    cudaMemcpy(hr, rank, grid * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hx, xyz, grid * sizeof(int), cudaMemcpyDeviceToHost);
    int bad = 0;
    printf("A cluster=(%d,%d,%d) grid=(%d,%d,%d)\n", cx, cy, cz, gx, gy, gz);
    for (int z = 0; z < gz; z++)
      for (int y = 0; y < gy; y++) {
        printf("  by=%2d:", y);
        for (int x = 0; x < gx; x++) {
          int idx = (z * gy + y) * gx + x;
          int lx = x % cx, ly = y % cy, lz = z % cz;
          int expect = lx + ly * cx + lz * cx * cy;
          if (hr[idx] != expect) bad++;
          printf(" %3d", hr[idx]);
        }
        printf("\n");
      }
    printf("  rank==x+y*cx+z*cx*cy ? %s (mismatch=%d)\n", bad ? "NO" : "YES", bad);
    cudaFree(rank); cudaFree(xyz); free(hr); free(hx);
  } else if (!strcmp(cmd, "S")) {
    int cx = atoi(argv[2]), cy = atoi(argv[3]), K = atoi(argv[4]), R = atoi(argv[5]);
    float err = -1;
    double t = run_shift2d(cx, cy, K, R, &err);
    double bytes = 128.0 * K * TH * 4 * 2 * R;
    printf("S cluster=(%d,%d) C=%d seg=%3dKB grid=128 R=%d | %s", cx, cy, cx * cy,
           K * TH * 4 / 1024, R, t < 0 ? "FAILED\n" : "");
    if (t >= 0) printf("%8.4f ms | %7.1f GB/s | maxerr %.4f\n", t, bytes / t / 1e6, err);
  } else if (!strcmp(cmd, "M")) {
    int cx = atoi(argv[2]), cy = atoi(argv[3]), gx = atoi(argv[4]), gy = atoi(argv[5]);
    int KB = atoi(argv[6]), R = atoi(argv[7]), mode = atoi(argv[8]);
    double err = -1; int L = 0;
    double ms = run_bcast2d(cx, cy, gx, gy, KB, R, mode, &L, &err);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    int grid = gx * gy;
    double deliv = (double)grid * chunk * L * R;
    // 源侧「读请求」模型：mode0=每 CTA；mode1=每 cluster 1 份；mode2=每行 1 份
    double nclusters = (double)grid / (cx * cy);
    double per_cluster = (mode == 1) ? 1.0 : (mode == 2 ? (double)cy : (double)(cx * cy));
    double src_read = nclusters * chunk * L * R * per_cluster;
    printf("M cluster=(%d,%d) grid=(%d,%d) chunk=%4dKB L=%2d R=%3d mode=%d | %s",
           cx, cy, gx, gy, KB, L, R, mode, ms < 0 ? "FAILED\n" : "");
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | src-read %7.1f GB/s | maxerr %.5f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, src_read / (ms / 1e3) / 1e9, err);
  } else if (!strcmp(cmd, "P")) {
    int cx = atoi(argv[2]), cy = atoi(argv[3]);
    int gx = atoi(argv[4]), gy = atoi(argv[5]), R = atoi(argv[6]);
    double t = time_cluster3(dim3(cx, cy, 1), dim3(gx, gy, 1), dim3(TH), 0, 20,
                             cluster_sync2d_kernel, R);
    printf("P cluster=(%d,%d) C=%d grid=(%d,%d) R=%d | %s", cx, cy, cx * cy, gx, gy, R,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) printf("cluster.sync %.1f ns/sync\n", t * 1e6 / R);
  } else if (!strcmp(cmd, "C2")) {
    int cx = atoi(argv[2]), cy = atoi(argv[3]), cz = atoi(argv[4]);
    int gx = atoi(argv[5]), gy = atoi(argv[6]), gz = atoi(argv[7]);
    cudaError_t e = launch_cluster3(dim3(cx, cy, cz), dim3(gx, gy, gz), dim3(TH), 0,
                                    cluster_sync2d_kernel, 1);
    printf("C2 cluster=(%d,%d,%d) C=%d grid=(%d,%d,%d) | %s\n", cx, cy, cz, cx * cy * cz,
           gx, gy, gz, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
