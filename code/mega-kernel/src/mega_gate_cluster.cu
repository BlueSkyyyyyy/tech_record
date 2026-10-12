// P34-1: DeepGEMM Mega Gate 的 `cluster_dim` 路径的 sm90 等价复刻。
//         （目标卡 H100 / sm90a）
//
// 背景：docs/02 §1.2 已精读 Mega Gate（一手核对 commit 057ca59）。它的「gate task」用
//   cluster_dim = dim3(config.num_mma_ctas, 1, 1)（csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp:74）
//   把 `num_logical_ctas = num_expert_groups * num_mma_ctas * num_split_k` 个 CTA 绑成**一个逻辑 CTA**
//   （heuristics/mega_gate.hpp:133；impls/sm100_bf16_mega_gate.cuh:55）。Mega Gate 本身 SM100 专属，
//   本节点把它的**协调路径**（不是 tcgen05 指令）用 sm90 原语等价复刻出来并量化：
//     (1) CTA pair（num_mma_ctas=2，cluster_dim=2）把专家维 M 对半切开、各写自己那半；
//     (2) split-K：每个 split 只算 K 的一个切片，把**部分分**写进全局 score scratch
//         `[token_global][split][expert]`（layout/mega_gate.cuh:78-107）；
//     (3) ScoreBarrier：所有逻辑 CTA 到齐后，才允许读全部 split 的部分分做归约 + top-k
//         （scheduler/mega_gate.cuh:13-39；编码 = (grid+1)*kNumMaxLogicalCtas + 到达数）。
//
// 编译：nvcc -O3 -arch=sm_90a src/mega_gate_cluster.cu -o mega_gate_cluster -lcuda
// 运行：见 harness/mega_gate_cluster.sh
//
// 子命令：
//   L <NT> <T> <E> <K> <SPLIT> <MMA> <TOPK> <mode> <iters>
//     mode 0 gate  : 忠实路径 —— split-K + 全局 score scratch + ScoreBarrier + 归约 top-k
//     mode 1 fused : 单 CTA、把 score 留在一块 SMEM 里，不算 split / 不写全局 scratch / 不设 barrier，
//                    直接在 SMEM 上 top-k（仅 SPLIT=1,MMA=1 合法；作为「不做 score round-trip」的对照）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
  const char* s = "?"; cuGetErrorString(r, &s); \
  printf("[CU FAIL] %s -> %s\n", #x, s); exit(2); } } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("[CUDA FAIL] %s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)

static const int MAX_LOGICAL = 64;   // = layout::mega_gate::kNumMaxLogicalCtas
static const int BARRIER_LINE = 128; // = layout::mega_gate::kScoreBarrierLineBytes
static const int KT = 64;            // K 方向分块
static const int KTS = KT + 1;       // smem 行 stride（+1 避开 bank conflict）

struct Params {
  int NT, T, E, K, SPLIT, MMA, TOPK, use_scratch, num_logical;
  uint64_t gen;
};

// ---------------- ScoreBarrier（= scheduler/mega_gate.cuh:13-39 的 sm90 复刻） ----------------
__device__ __forceinline__ uint64_t sb_encode(uint64_t gen, uint32_t arrived) {
  return (gen + 1) * (uint64_t)MAX_LOGICAL + arrived;
}
__device__ __forceinline__ uint64_t* sb_ptr(uint64_t* barriers, int token_block) {
  return reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(barriers) +
                                     (size_t)token_block * BARRIER_LINE);
}
__device__ __forceinline__ void sb_init(uint64_t* s, uint64_t gen) {
  *s = sb_encode(gen, 0); __threadfence();
}
__device__ __forceinline__ void sb_wait_init(uint64_t* s, uint64_t gen, int num_logical) {
  const uint64_t base = sb_encode(gen, 0); uint64_t v;
  do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(s) : "memory");
  } while (v - base >= (uint64_t)num_logical);   // 无符号距离拒绝陈旧 grid 标签
}
__device__ __forceinline__ void sb_arrive(uint64_t* s) {
  asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(s) : "memory");
}
__device__ __forceinline__ void sb_wait(uint64_t* s, uint64_t gen, int num_logical) {
  const uint64_t target = sb_encode(gen, (uint32_t)num_logical); uint64_t v;
  do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(s) : "memory");
  } while (v != target);
}

__device__ __forceinline__ float sqrt_softplus(float value) {
  const float sp = log1pf(expf(value));           // = epilogue/sm100_mega_gate.cuh:19-23
  return sqrtf(value > 20.0f ? value : sp);
}
__device__ __forceinline__ void insert_topk(int TOPK, int e, float biased, float unbiased,
                                            int* bi, float* bb, float* bu) {
  for (int i = 0; i < TOPK; i++) {
    if (biased > bb[i]) {
      for (int j = TOPK - 1; j > i; j--) { bb[j] = bb[j - 1]; bu[j] = bu[j - 1]; bi[j] = bi[j - 1]; }
      bb[i] = biased; bu[i] = unbiased; bi[i] = e; break;
    }
  }
}

// ---------------- gate kernel ----------------
// 寄存器 micro-tile：每线程算 TM×TN 个 (token, expert) 输出（静态累加器，不落 local memory）。
static const int TM = 4, TN = 4;

__global__ void __launch_bounds__(1024, 1)
gate_kernel(const float* __restrict__ X,     // [NT*T, K]
            const float* __restrict__ W,     // [E, K]
            const float* __restrict__ bias,  // [E]
            float* __restrict__ scratch,     // [NT*T, SPLIT, E]
            uint64_t* __restrict__ barriers, // [NT] lines
            int* __restrict__ topk_idx,      // [NT*T, TOPK]
            float* __restrict__ topk_w,      // [NT*T, TOPK]
            Params p) {
  extern __shared__ float sm[];
  const int rank = p.MMA > 1 ? (int)cg::this_cluster().block_rank() : 0;
  if (p.MMA > 1) cg::this_cluster().sync();

  const int mma_cluster_idx = (int)blockIdx.x / p.MMA;
  const int slice_idx       = mma_cluster_idx % p.SPLIT;     // expert_groups = 1
  const int split_k_idx     = slice_idx;
  const int logical_rank    = slice_idx * p.MMA + rank;
  const int token_block     = (int)blockIdx.x / (p.SPLIT * p.MMA);

  const int Eper = (p.MMA == 2) ? (p.E / 2) : p.E;   // CTA pair 切专家维（UMMA M）
  const int ebase = (p.MMA == 2) ? rank * (p.E / 2) : 0;
  const int kblk = p.K / p.SPLIT;
  const int kbase = split_k_idx * kblk;
  const int tb = token_block * p.T;
  const bool write_scratch = p.use_scratch;
  const int TOPK = p.TOPK;
  const int bdim = blockDim.x;

  float* xs = sm;                    // [T, KTS]
  float* ws = xs + p.T * KTS;        // [Eper, KTS]
  float* sc = ws + Eper * KTS;       // [T, Eper] 仅 fused 用（本 CTA 的全部分数）

  // ---- 1) score GEMM 部分分：smem 分块 + 寄存器 micro-tile ----
  const int tx_per = p.T / TM;              // token 方向 micro-tile 数
  const int ty_per = Eper / TN;             // expert 方向 micro-tile 数
  const int tx = threadIdx.x % tx_per;      // 本线程的 token micro-tile
  const int ty = threadIdx.x / tx_per;      // 本线程的 expert micro-tile

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

  if (threadIdx.x < tx_per * ty_per) {
    for (int k0 = 0; k0 < kblk; k0 += KT) {
      const int kt = (kblk - k0 < KT) ? (kblk - k0) : KT;
      const int kt4 = kt / 4;                       // float4 分块载入（K/KT 均 4 的倍数）
      for (int idx = threadIdx.x; idx < p.T * kt4; idx += bdim) {
        const int t = idx / kt4, kk4 = idx % kt4;
        const float4 v = *reinterpret_cast<const float4*>(X + (size_t)(tb + t) * p.K + kbase + k0 + kk4 * 4);
        xs[t * KTS + kk4 * 4 + 0] = v.x; xs[t * KTS + kk4 * 4 + 1] = v.y;
        xs[t * KTS + kk4 * 4 + 2] = v.z; xs[t * KTS + kk4 * 4 + 3] = v.w;
      }
      for (int idx = threadIdx.x; idx < Eper * kt4; idx += bdim) {
        const int ee = idx / kt4, kk4 = idx % kt4;
        const float4 v = *reinterpret_cast<const float4*>(W + (size_t)(ebase + ee) * p.K + kbase + k0 + kk4 * 4);
        ws[ee * KTS + kk4 * 4 + 0] = v.x; ws[ee * KTS + kk4 * 4 + 1] = v.y;
        ws[ee * KTS + kk4 * 4 + 2] = v.z; ws[ee * KTS + kk4 * 4 + 3] = v.w;
      }
      __syncthreads();
      const float* xr = xs + (tx * TM) * KTS;
      const float* wr = ws + (ty * TN) * KTS;
      for (int kk = 0; kk < kt; kk++) {
        float xv[TM], wv[TN];
        #pragma unroll
        for (int i = 0; i < TM; i++) xv[i] = xr[i * KTS + kk];
        #pragma unroll
        for (int j = 0; j < TN; j++) wv[j] = wr[j * KTS + kk];
        #pragma unroll
        for (int i = 0; i < TM; i++)
          #pragma unroll
          for (int j = 0; j < TN; j++) acc[i][j] += xv[i] * wv[j];
      }
      __syncthreads();
    }
  }
  // 落分数到 SMEM staging（统一路径），再按需 coalesced 写全局 scratch
  if (threadIdx.x < tx_per * ty_per) {
    #pragma unroll
    for (int i = 0; i < TM; i++)
      #pragma unroll
      for (int j = 0; j < TN; j++)
        sc[(tx * TM + i) * Eper + (ty * TN + j)] = acc[i][j];
  }
  __syncthreads();
  if (write_scratch) {
    for (int idx = threadIdx.x; idx < p.T * Eper; idx += bdim) {
      const int t = idx / Eper, ee = idx % Eper;
      scratch[((size_t)(tb + t) * p.SPLIT + split_k_idx) * p.E + (ebase + ee)] = sc[t * Eper + ee];
    }
    __syncthreads();
  }

  // ---- 2) ScoreBarrier：等本 token block 的所有逻辑 CTA 到齐 ----
  if (p.use_scratch && p.num_logical > 1) {
    uint64_t* sbar = sb_ptr(barriers, token_block);
    if (threadIdx.x == 0) {
      if (logical_rank == 0) sb_init(sbar, p.gen);
      else                   sb_wait_init(sbar, p.gen, p.num_logical);
    }
    __syncthreads();
    if (threadIdx.x == 0) sb_arrive(sbar);
    if (threadIdx.x == 0) sb_wait(sbar, p.gen, p.num_logical);
    __syncthreads();
  }

  // ---- 3) 归约 + top-k：每个逻辑 CTA 认领自己那份 token；token 分摊到线程 ----
  const int n_assigned = (p.T - logical_rank + p.num_logical - 1) / p.num_logical;
  for (int i = threadIdx.x; i < n_assigned; i += bdim) {
    const int t = logical_rank + i * p.num_logical;
    int bi[32]; float bb[32], bu[32];
    for (int q = 0; q < TOPK; q++) { bi[q] = -1; bb[q] = -1e30f; bu[q] = 0.f; }
    for (int e = 0; e < p.E; e++) {
      float sv = 0.f;
      if (write_scratch)
        for (int s = 0; s < p.SPLIT; s++)
          sv += scratch[((size_t)(tb + t) * p.SPLIT + s) * p.E + e];
      else
        sv = sc[t * Eper + e];
      const float unbiased = sqrt_softplus(sv);
      const float biased = unbiased + (bias ? bias[e] : 0.f);
      insert_topk(TOPK, e, biased, unbiased, bi, bb, bu);
    }
    float sum = 1e-20f;
    for (int q = 0; q < TOPK; q++) sum += bu[q];
    for (int q = 0; q < TOPK; q++) {
      topk_idx[(size_t)(tb + t) * TOPK + q] = bi[q];
      topk_w[(size_t)(tb + t) * TOPK + q] = bu[q] / sum;   // routed_scaling_factor = 1
    }
  }
}

// 参考：完整 GEMM + 同一 top-k 语义（一 token 一线程）
__global__ void ref_kernel(const float* __restrict__ X, const float* __restrict__ W,
                           const float* __restrict__ bias,
                           int* __restrict__ ridx, float* __restrict__ rw,
                           int NTT, int E, int K, int TOPK) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NTT) return;
  int bi[32]; float bb[32], bu[32];
  for (int i = 0; i < TOPK; i++) { bi[i] = -1; bb[i] = -1e30f; bu[i] = 0.f; }
  for (int e = 0; e < E; e++) {
    float sv = 0.f;
    const float* xr = X + (size_t)t * K;
    const float* wr = W + (size_t)e * K;
    for (int k = 0; k < K; k++) sv += xr[k] * wr[k];
    const float unbiased = sqrt_softplus(sv);
    const float biased = unbiased + (bias ? bias[e] : 0.f);
    insert_topk(TOPK, e, biased, unbiased, bi, bb, bu);
  }
  float sum = 1e-20f;
  for (int i = 0; i < TOPK; i++) sum += bu[i];
  for (int i = 0; i < TOPK; i++) {
    ridx[(size_t)t * TOPK + i] = bi[i];
    rw[(size_t)t * TOPK + i] = bu[i] / sum;
  }
}

static size_t smem_bytes(int T, int Eper) {
  return ((size_t)(T + Eper) * KTS + (size_t)T * Eper) * 4;
}

static cudaError_t launch_gate(int NT, int T, int E, int K, int SPLIT, int MMA, int TOPK,
                               int use_scratch, uint64_t gen, int iters,
                               const float* X, const float* W, const float* bias,
                               float* scratch, uint64_t* barriers,
                               int* topk_idx, float* topk_w, float* ms_out) {
  const int num_logical = SPLIT * MMA;
  const int grid = NT * num_logical;
  const int Eper = (MMA == 2) ? E / 2 : E;
  const size_t smem = smem_bytes(T, Eper);
  const int bdim = (T / TM) * (Eper / TN);
  cudaFuncSetAttribute((const void*)gate_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncSetAttribute((const void*)gate_kernel,
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(bdim); cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute at[1];
  if (MMA > 1) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = MMA; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
  }
  Params p = {NT, T, E, K, SPLIT, MMA, TOPK, use_scratch, num_logical, gen};
  void* args[] = {(void*)&X, (void*)&W, (void*)&bias, (void*)&scratch, (void*)&barriers,
                  (void*)&topk_idx, (void*)&topk_w, (void*)&p};

  CUDA(cudaMemset(barriers, 0, (size_t)NT * BARRIER_LINE));
  CUDA(cudaLaunchKernelExC(&cfg, (void*)gate_kernel, args));
  CUDA(cudaDeviceSynchronize());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  cudaEvent_t e0, e1; CUDA(cudaEventCreate(&e0)); CUDA(cudaEventCreate(&e1));
  CUDA(cudaEventRecord(e0));
  for (int it = 0; it < iters; it++) {
    p.gen = gen + it;
    CUDA(cudaLaunchKernelExC(&cfg, (void*)gate_kernel, args));
  }
  CUDA(cudaDeviceSynchronize());
  CUDA(cudaEventRecord(e1)); CUDA(cudaEventSynchronize(e1));
  float ms; CUDA(cudaEventElapsedTime(&ms, e0, e1));
  *ms_out = ms / iters;
  return cudaSuccess;
}

static void fill(float* h, long n, int seed) {
  unsigned s = seed * 2654435761u + 1;
  for (long i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; h[i] = ((s >> 8) & 0xffff) / 65536.0f - 0.5f; }
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: mega_gate_cluster L <NT> <T> <E> <K> <SPLIT> <MMA> <TOPK> <mode> <iters>\n"); return 1; }
  CK(cuInit(0));
  const char* cmd = argv[1];
  if (strcmp(cmd, "L") != 0) { printf("unknown cmd %s\n", cmd); return 1; }
  int NT = atoi(argv[2]), T = atoi(argv[3]), E = atoi(argv[4]), K = atoi(argv[5]);
  int SPLIT = atoi(argv[6]), MMA = atoi(argv[7]), TOPK = atoi(argv[8]);
  int mode = atoi(argv[9]), iters = atoi(argv[10]);
  int use_scratch = (mode == 0);
  if (mode == 1 && (SPLIT != 1 || MMA != 1)) { printf("mode 1 requires SPLIT=1 MMA=1\n"); return 1; }
  int Eper = (MMA == 2) ? E / 2 : E;
  if (T % TM != 0 || Eper % TN != 0) { printf("T must be multiple of %d, Eper of %d\n", TM, TN); return 1; }
  int bdim = (T / TM) * (Eper / TN);
  if (bdim > 1024) { printf("blockDim %d > 1024 (use smaller T/E)\n", bdim); return 1; }

  long NTT = (long)NT * T;
  float *X, *W, *bias;
  CUDA(cudaMalloc(&X, (size_t)NTT * K * 4));
  CUDA(cudaMalloc(&W, (size_t)E * K * 4));
  CUDA(cudaMalloc(&bias, (size_t)E * 4));
  {
    float* hx = (float*)malloc((size_t)NTT * K * 4);
    float* hw = (float*)malloc((size_t)E * K * 4);
    float* hb = (float*)malloc((size_t)E * 4);
    fill(hx, NTT * K, 1); fill(hw, (long)E * K, 2); fill(hb, E, 3);
    CUDA(cudaMemcpy(X, hx, (size_t)NTT * K * 4, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(W, hw, (size_t)E * K * 4, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(bias, hb, (size_t)E * 4, cudaMemcpyHostToDevice));
    free(hx); free(hw); free(hb);
  }
  float* scratch; CUDA(cudaMalloc(&scratch, (size_t)NTT * SPLIT * E * 4));
  uint64_t* barriers; CUDA(cudaMalloc(&barriers, (size_t)NT * BARRIER_LINE));
  CUDA(cudaMemset(barriers, 0, (size_t)NT * BARRIER_LINE));
  int* topk_idx; float* topk_w;
  CUDA(cudaMalloc(&topk_idx, (size_t)NTT * TOPK * 4));
  CUDA(cudaMalloc(&topk_w, (size_t)NTT * TOPK * 4));

  float ms = -1;
  cudaError_t e = launch_gate(NT, T, E, K, SPLIT, MMA, TOPK, use_scratch, 1, iters,
                              X, W, bias, scratch, barriers, topk_idx, topk_w, &ms);
  if (e != cudaSuccess) {
    printf("L NT=%d T=%d E=%d K=%d SPLIT=%d MMA=%d TOPK=%d mode=%d | FAILED %s\n",
           NT, T, E, K, SPLIT, MMA, TOPK, mode, cudaGetErrorString(e)); return 0;
  }

  int* ridx; float* rw;
  CUDA(cudaMalloc(&ridx, (size_t)NTT * TOPK * 4));
  CUDA(cudaMalloc(&rw, (size_t)NTT * TOPK * 4));
  int NTTi = (int)NTT;
  ref_kernel<<<(NTTi + 255) / 256, 256>>>(X, W, bias, ridx, rw, NTTi, E, K, TOPK);
  CUDA(cudaDeviceSynchronize());
  int* hi = (int*)malloc((size_t)NTT * TOPK * 4);
  float* hw = (float*)malloc((size_t)NTT * TOPK * 4);
  int* ri = (int*)malloc((size_t)NTT * TOPK * 4);
  float* rww = (float*)malloc((size_t)NTT * TOPK * 4);
  CUDA(cudaMemcpy(hi, topk_idx, (size_t)NTT * TOPK * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hw, topk_w, (size_t)NTT * TOPK * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(ri, ridx, (size_t)NTT * TOPK * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(rww, rw, (size_t)NTT * TOPK * 4, cudaMemcpyDeviceToHost));
  long idx_mismatch = 0; double wmax = 0;
  for (long t = 0; t < NTT; t++) {
    int a[32], b[32];
    for (int i = 0; i < TOPK; i++) { a[i] = hi[t * TOPK + i]; b[i] = ri[t * TOPK + i]; }
    for (int i = 0; i < TOPK; i++) for (int j = i + 1; j < TOPK; j++) {
      if (a[j] < a[i]) { int tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
      if (b[j] < b[i]) { int tmp = b[i]; b[i] = b[j]; b[j] = tmp; }
    }
    for (int i = 0; i < TOPK; i++) if (a[i] != b[i]) { idx_mismatch++; break; }
    for (int i = 0; i < TOPK; i++) { double d = fabs((double)hw[t * TOPK + i] - rww[t * TOPK + i]); if (d > wmax) wmax = d; }
  }
  printf("L NT=%d T=%d E=%d K=%d SPLIT=%d MMA=%d TOPK=%d mode=%d | %8.3f us | logical=%2d grid=%4d | "
         "idx_mismatch=%ld/%ld wmax=%.2e\n",
         NT, T, E, K, SPLIT, MMA, TOPK, mode, ms * 1e3, SPLIT * MMA, NT * SPLIT * MMA,
         idx_mismatch, NTT, wmax);
  free(hi); free(hw); free(ri); free(rww);
  cudaFree(X); cudaFree(W); cudaFree(bias); cudaFree(scratch); cudaFree(barriers);
  cudaFree(topk_idx); cudaFree(topk_w); cudaFree(ridx); cudaFree(rw);
  return 0;
}
