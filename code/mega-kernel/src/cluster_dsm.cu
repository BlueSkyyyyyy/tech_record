// P29-1: Thread Block Clusters + Distributed Shared Memory (DSM) —— mega-kernel 的
//         跨 SM「不落 HBM」中间量交换原语（目标卡 H100 / sm90）。
//
// 背景：docs/04 §6.6 明确留了「sm90 的 cluster 未覆盖」的尾巴；DeepGEMM Mega Gate
// 用 cluster_dim = num_mma_ctas（docs/02 §1.2），而整图 mega-kernel 需要跨 SM 传中间量。
// 本程序量化 DSM 作为跨 SM 交换介质的吞吐，并与 HBM（global scratch）往返对照，并刻画
// cluster 的容量/拓扑边界。
//
// 编译：nvcc -O3 -arch=sm_90 src/cluster_dsm.cu -o cluster_dsm
// 运行：见 harness/cluster_dsm.sh（每个配置单独进程，因一次 launch failure 会污染 context）
//
// 子命令：
//   A <C> <grid> <SB> <R>          DSM 远端读 vs 本地 SMEM 读（C=1 == 本地基线）
//   Lg <grid> <MB>                 plain 全局流式读（HBM 参照）
//   Sh <C> <grid> <K> <R>          ring 段位移交换，hbm（global scratch）
//   Sd <C> <grid> <K> <R>          ring 段位移交换，dsm（邻居 SMEM）
//   C1 <C> <grid> <R>              cluster.sync() 成本（ns/sync）
//   C2 <C> <grid>                  cluster 容量：尝试 launch，打印 ok/错误
//   C3 <grid> <R>                  grid.sync() 成本（cooperative）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

static const int TH = 256;

// ---------------- Section A：DSM 远端读（C=1 退化为读自己 SMEM == 本地基线）------------
__global__ void dsm_read_kernel(float* __restrict__ out, int R, int sb) {
  extern __shared__ float sm[];
  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  for (int i = threadIdx.x; i < sb; i += blockDim.x) sm[i] = (float)(i & 1023) * 1e-3f + rank;
  cl.sync();
  float* src = cl.map_shared_rank(sm, (rank + 1) % C);
  volatile float* vsrc = src;   // volatile 防止循环提升
  float acc = 0.f;
  for (int r = 0; r < R; r++) acc += vsrc[threadIdx.x];
  out[blockIdx.x] = acc;
}

__global__ void gmem_stream_kernel(const float* __restrict__ a, float* __restrict__ out,
                                   long n, int R) {
  float acc = 0.f;
  long stride = (long)gridDim.x * blockDim.x;
  for (int r = 0; r < R; r++)
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) acc += a[i];
  if (acc == -12345.678f) out[blockIdx.x] = acc;
}

// ---------------- Section B：ring 段位移交换（确定性，可对解析参考）----------------
// C 个 block 组成 cluster，每 block 拥有 seg=K*TH 个 float 的段，初值 = gid*1000 + j。
// 双缓冲 ping-pong（槽 cur=0/1）：第 r 轮读邻居槽 r%2、写自己槽 (r+1)%2，末尾 cl.sync()。
// 每轮把整段「向左搬一格」，每轮只 1 次屏障。R 轮后 block gid 的段 = 初始时 rank
// (rank-R)%C 的段。MODE 0 = hbm（段放 global scratch）；MODE 1 = dsm（段放自己 SMEM）。
template <int K, int MODE>
__global__ void shift_kernel(float* __restrict__ scratch, float* __restrict__ out, int R) {
  extern __shared__ float sm[];
  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  long gid = blockIdx.x;
  const int seg = K * TH;
  float* buf[2];
  float* nbuf[2];
  if (MODE == 0) {
    long cstart = gid - (gid % C);
    buf[0] = scratch + gid * (long)(2 * seg); buf[1] = buf[0] + seg;
    float* base = scratch + (cstart + (rank + C - 1) % C) * (long)(2 * seg);
    nbuf[0] = base; nbuf[1] = base + seg;
  } else {
    buf[0] = sm; buf[1] = sm + seg;
    float* base = cl.map_shared_rank(sm, (rank + C - 1) % C);
    nbuf[0] = base; nbuf[1] = base + seg;
  }
#pragma unroll
  for (int k = 0; k < K; k++) buf[0][threadIdx.x + k * TH] = (float)(gid * 1000 + threadIdx.x + k * TH);
  cl.sync();
  for (int r = 0; r < R; r++) {
    int cur = r & 1;
#pragma unroll
    for (int k = 0; k < K; k++) buf[cur ^ 1][threadIdx.x + k * TH] = nbuf[cur][threadIdx.x + k * TH];
    cl.sync();
  }
  if (threadIdx.x == 0) out[gid] = buf[R & 1][0];   // 代表值（thread0）
}

// ---------------- Section C ----------------
__global__ void cluster_sync_kernel(int R) {
  cg::cluster_group cl = cg::this_cluster();
  for (int r = 0; r < R; r++) cl.sync();
}
__global__ void grid_sync_kernel(int R) {
  cg::grid_group g = cg::this_grid();
  for (int r = 0; r < R; r++) g.sync();
}

template <int K, int MODE>
static double time_shift(int C, int grid, int R, float* maxerr) {
  int seg = K * TH;
  size_t need = (size_t)grid * 2 * seg * 4;
  float *scratch = nullptr, *out;
  cudaMalloc(&out, grid * sizeof(float));
  if (MODE == 0) { cudaMalloc(&scratch, need); cudaMemset(scratch, 0, need); }
  double t = time_cluster(C, dim3(grid), dim3(TH), MODE == 1 ? (size_t)2 * seg * 4 : 0, 20,
                          shift_kernel<K, MODE>, scratch, out, R);
  if (t >= 0) {
    float* h = (float*)malloc(grid * sizeof(float));
    cudaMemcpy(h, out, grid * sizeof(float), cudaMemcpyDeviceToHost);
    double err = 0;
    for (int g = 0; g < grid; g++) {
      long cstart = g - (g % C);
      long rank = g % C;
      long src = cstart + ((rank - R) % C + C) % C;
      double expect = (double)(src * 1000);            // 初值 src*1000 + j（j=0）
      double d = h[g] - expect;
      if (d < 0) d = -d;
      if (d > err) err = d;
    }
    *maxerr = err;
    free(h);
  }
  cudaFree(out); if (scratch) cudaFree(scratch);
  return t;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: cluster_dsm <A|Lg|Sh|Sd|C1|C2|C3> ...\n"); return 1; }
  const char* cmd = argv[1];

  if (!strcmp(cmd, "A")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), SB = atoi(argv[4]), R = atoi(argv[5]);
    float* out; cudaMalloc(&out, grid * sizeof(float));
    double t = time_cluster(C, dim3(grid), dim3(TH), SB * 4, 20, dsm_read_kernel, out, R, SB);
    printf("A C=%2d grid=%3d SB=%3dKB R=%d | %s", C, grid, SB * 4 / 1024, R,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) printf("%9.3f us/launch | %6.3f ns/lap\n", t * 1e3, t * 1e6 / R);
  } else if (!strcmp(cmd, "Lg")) {
    int grid = atoi(argv[2]); long MB = atol(argv[3]);
    long n = MB * 1024 * 1024 / 4;
    float *a, *out; cudaMalloc(&a, n * 4); cudaMalloc(&out, grid * sizeof(float));
    cudaMemset(a, 1, n * 4);
    gmem_stream_kernel<<<grid, TH>>>(a, out, n, 100);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 10; i++) gmem_stream_kernel<<<grid, TH>>>(a, out, n, 100);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
    printf("Lg grid=%d MB=%ld | %8.1f GB/s\n", grid, MB, (double)n * 4 * 100 / (ms / 1e3) / 1e9);
  } else if (!strcmp(cmd, "Sh") || !strcmp(cmd, "Sd")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), K = atoi(argv[4]), R = atoi(argv[5]);
    const char* nm = cmd[1] == 'h' ? "hbm" : "dsm";
    int MODE = cmd[1] == 'h' ? 0 : 1;
    float err = -1; double t = -1;
    if (MODE == 0) {
      switch (K) {
        case 1: t = time_shift<1, 0>(C, grid, R, &err); break;
        case 2: t = time_shift<2, 0>(C, grid, R, &err); break;
        case 4: t = time_shift<4, 0>(C, grid, R, &err); break;
        case 8: t = time_shift<8, 0>(C, grid, R, &err); break;
        default: printf("K must be 1/2/4/8\n"); return 1;
      }
    } else {
      switch (K) {
        case 1: t = time_shift<1, 1>(C, grid, R, &err); break;
        case 2: t = time_shift<2, 1>(C, grid, R, &err); break;
        case 4: t = time_shift<4, 1>(C, grid, R, &err); break;
        case 8: t = time_shift<8, 1>(C, grid, R, &err); break;
        default: printf("K must be 1/2/4/8\n"); return 1;
      }
    }
    double bytes = (double)grid * K * TH * 4 * 2 * R;   // 读+写整段，R 轮
    printf("S %s C=%2d grid=%3d seg=%3dKB R=%d | %s", nm, C, grid, K * TH * 4 / 1024, R,
           t < 0 ? "FAILED\n" : "");
    if (t >= 0) printf("%8.4f ms | %7.1f GB/s | maxerr %.4f\n", t, bytes / t / 1e6, err);
  } else if (!strcmp(cmd, "C1")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), R = atoi(argv[4]);
    double t = time_cluster(C, dim3(grid), dim3(TH), 0, 20, cluster_sync_kernel, R);
    printf("C1 C=%2d grid=%3d R=%d | %s", C, grid, R, t < 0 ? "FAILED\n" : "");
    if (t >= 0) printf("cluster.sync %.1f ns/sync\n", t * 1e6 / R);
  } else if (!strcmp(cmd, "C2")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]);
    cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), 0, cluster_sync_kernel, 1);
    printf("C2 C=%2d grid=%3d | %s\n", C, grid, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
  } else if (!strcmp(cmd, "C3")) {
    int grid = atoi(argv[2]), R = atoi(argv[3]);
    void* args[] = {&R};
    cudaError_t e = cudaLaunchCooperativeKernel((void*)grid_sync_kernel, dim3(grid), dim3(TH), args);
    if (e != cudaSuccess) { printf("C3 launch err %s\n", cudaGetErrorString(e)); return 2; }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; i++)
      cudaLaunchCooperativeKernel((void*)grid_sync_kernel, dim3(grid), dim3(TH), args);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("C3 grid=%3d R=%d | grid.sync %.1f ns/sync\n", grid, R, ms * 1e6 / R);
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
