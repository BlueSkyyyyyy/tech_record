// 27 MoE（一）：router + top-k + token permutation
//
// 复现 DeepSeek-V4 / Kimi-K2.6 的 MoE 前门（gate）整条链，shape 取自
// /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, n_routed_experts=384, num_experts_per_tok=6,
//   scoring_func=sqrtsoftplus, topk_method=noaux_tc, norm_topk_prob=true,
//   routed_scaling_factor=2.5。
//
// 流程（noaux_tc / DeepseekV4 routing）：
//   1) logits = x @ Wg^T                       [M,H]x[H,E] -> [M,E]
//   2) s      = sqrt(softplus(logits))          （Kimi/V3 是 sigmoid）
//   3) choice = s + e_score_correction_bias     （bias 只用于选专家）
//   4) ids    = topk(choice, 6)                 （保序）
//   5) w      = s.gather(ids); w /= w.sum(); w *= routed_scaling_factor
//
//   6) permutation：把 M 个 token 按 expert 归拢成 permuted 布局（供 grouped
//      GEMM 消费，见 25 篇），并给出逆置换用于 unpermute。
//
// 内核：
//   gate_mma_kernel        : mma.m16n8k16 + ldmatrix + cp.async 双缓冲
//   router_topk_kernel     : 1 warp / token，sqrtsoftplus + bias + argmax×6
//   route_count_kernel     : expert 直方图
//   route_scan_kernel      : exclusive prefix sum -> expert_offsets
//   route_scatter_kernel   : 给每个 (t,j) 分配 permuted 位置（atomic 游标）
//   permute_copy_kernel    : permuted_x[pos,:] = x[t,:]（16B 向量化）
//   unpermute_kernel       : y[t] = Σ_j w[t,j]*permuted_x[pos(t,j),:]
//
// 运行：scripts/run.sh 27-moe-router/moe_router.cu [M] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

// ===========================================================================
// 模型常量（DeepSeek-V4-Pro）
// ===========================================================================
constexpr int H = 7168;           // hidden_size
constexpr int E = 384;            // n_routed_experts
constexpr int TOPK = 6;           // num_experts_per_tok
constexpr float ROUTE_SCALE = 2.5f;
constexpr bool RENORM = true;
constexpr int WT = 8;             // router top-k：warps/block

// ---------------------------------------------------------------------------
// mma / smem 工具（复用 13 篇）
// ---------------------------------------------------------------------------
constexpr int BN = 128, BK = 32;
constexpr int ASP = BK + 8;
constexpr int BNP = BN + 8;
constexpr int BM = 128;  // 默认 baseline 的 M-tile（模板参数，见 gate_mma_pipe_t）

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int BMt, int NTHRt, int LDA, int LDB>
__device__ __forceinline__ void prefetch_tiles(const bf16* __restrict__ A,
                                               const bf16* __restrict__ B, bf16 (*As)[LDA],
                                               bf16 (*Bs)[LDB], int M, int N, int K, int block_row,
                                               int block_col, int k0) {
  const int t = threadIdx.x;
  for (int i = t; i < BMt * BK / 8; i += NTHRt) {
    const int row = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    if (gr < M && gc + 7 < K) {
      __pipeline_memcpy_async(&As[row][c8], &A[(size_t)gr * K + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&As[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  for (int i = t; i < BK * BN / 8; i += NTHRt) {
    const int row = i / (BN / 8), c8 = (i % (BN / 8)) * 8;
    const int gr = k0 + row, gc = block_col + c8;
    if (gr < K && gc + 7 < N) {
      __pipeline_memcpy_async(&Bs[row][c8], &B[(size_t)gr * N + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&Bs[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  __pipeline_commit();
}

template <int WARP_Mt, int WARP_Nt, int MTMt, int MTNt>
__device__ __forceinline__ void mma_stage(const bf16 (*As)[ASP], const bf16 (*Bs)[BNP],
                                          float acc[MTMt][MTNt][4], int warp_row, int warp_col) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    uint32_t a[MTMt][4];
#pragma unroll
    for (int i = 0; i < MTMt; ++i) {
      const int row = (lane & 15);
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&As[warp_row * WARP_Mt + i * 16 + row][kk * 16 + col]);
      ldmatrix_x4(addr, a[i]);
    }
    uint32_t b[MTNt][2];
#pragma unroll
    for (int g = 0; g < MTNt / 2; ++g) {
      const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&Bs[kk * 16 + row][warp_col * WARP_Nt + g * 16 + col]);
      uint32_t d[4];
      ldmatrix_x4_trans(addr, d);
      b[g * 2][0] = d[0];
      b[g * 2][1] = d[1];
      b[g * 2 + 1][0] = d[2];
      b[g * 2 + 1][1] = d[3];
    }
#pragma unroll
    for (int i = 0; i < MTMt; ++i)
#pragma unroll
      for (int j = 0; j < MTNt; ++j) mma_m16n8k16(acc[i][j], a[i], b[j]);
  }
}

template <int MTMt, int MTNt>
__device__ __forceinline__ void zero_acc(float acc[MTMt][MTNt][4]) {
#pragma unroll
  for (int i = 0; i < MTMt; ++i)
#pragma unroll
    for (int j = 0; j < MTNt; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
}

template <int WARP_Mt, int WARP_Nt, int MTMt, int MTNt>
__device__ __forceinline__ void store_acc(float* __restrict__ C, const float acc[MTMt][MTNt][4],
                                          int N, int block_row, int block_col, int warp_row,
                                          int warp_col) {
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int i = 0; i < MTMt; ++i)
#pragma unroll
    for (int j = 0; j < MTNt; ++j) {
      const int r0 = block_row + warp_row * WARP_Mt + i * 16 + group;
      const int c0 = block_col + warp_col * WARP_Nt + j * 8 + tig * 2;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + (q >= 2 ? 8 : 0);
        const int c = c0 + (q & 1);
        C[(size_t)r * N + c] = acc[i][j][q];
      }
    }
}

// gate GEMM：logits[M,E] = X[M,H] @ WgT[H,E]
//   BMt x BN(=128) tile，BMt/64 * WNt 个 warp，每个 warp 算 (BMt/WMt) x (BN/WNt)
template <int BMt, int WMt, int WNt>
__global__ void gate_mma_pipe_t(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  constexpr int NTHRt = WMt * WNt * 32;
  constexpr int WARP_Mt = BMt / WMt;
  constexpr int WARP_Nt = BN / WNt;
  constexpr int MTMt = WARP_Mt / 16;
  constexpr int MTNt = WARP_Nt / 8;
  extern __shared__ __align__(16) unsigned char dynsm[];
  bf16(*As)[ASP] = reinterpret_cast<bf16(*)[ASP]>(dynsm);                    // [2*BMt][ASP]
  bf16(*Bs)[BNP] = reinterpret_cast<bf16(*)[BNP]>(dynsm + 2 * BMt * ASP * sizeof(bf16));
  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / WNt, warp_col = wid % WNt;
  const int block_row = blockIdx.y * BMt, block_col = blockIdx.x * BN;

  float acc[MTMt][MTNt][4];
  zero_acc<MTMt, MTNt>(acc);
  prefetch_tiles<BMt, NTHRt, ASP, BNP>(A, B, As, Bs, M, N, K, block_row, block_col, 0);

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + BK;
    if (next < K) {
      prefetch_tiles<BMt, NTHRt, ASP, BNP>(A, B, As + (stage ^ 1) * BMt, Bs + (stage ^ 1) * BK, M,
                                           N, K, block_row, block_col, next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(next < K ? 1 : 0);
    __syncthreads();
    mma_stage<WARP_Mt, WARP_Nt, MTMt, MTNt>(As + stage * BMt, Bs + stage * BK, acc, warp_row,
                                            warp_col);
    __syncthreads();
    stage ^= 1;
  }
  store_acc<WARP_Mt, WARP_Nt, MTMt, MTNt>(C, acc, N, block_row, block_col, warp_row, warp_col);
}

// ===========================================================================
// router top-k：1 warp / token
//   score = sqrt(softplus(logit))  （V4；MODE_SIGMOID 则为 sigmoid）
//   choice = score + bias；取 top-K；权重 = score[ids] 归一化 × scale
// ===========================================================================
// WT 个 warp / block，每个 warp 一个 token；COUNT=true 时顺手把 histogram 做了。
template <int MODE, int WT, bool COUNT>  // MODE: 0 = sqrtsoftplus, 1 = sigmoid
__global__ void router_topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                   int M, float* __restrict__ w_out, int* __restrict__ id_out,
                                   int* __restrict__ cnt) {
  extern __shared__ float sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x * WT + warp;
  if (t >= M) return;
  float* sc = sm + (size_t)warp * 2 * E;  // E 个未加 bias 的 score
  float* bd = sc + E;                     // E 个 choice = score + bias
  const float* lg = logits + (size_t)t * E;

  for (int e = lane; e < E; e += 32) {
    const float l = lg[e];
    float s;
    if (MODE == 0) s = sqrtf(logf(1.f + expf(l)));   // sqrt(softplus)
    else           s = 1.f / (1.f + expf(-l));        // sigmoid
    sc[e] = s;
    bd[e] = s + bias[e];
  }
  __syncwarp();

  int sel[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    float bv = -INFINITY;
    int bi = -1;
    for (int e = lane; e < E; e += 32) {
      const float v = bd[e];
      if (v > bv) { bv = v; bi = e; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, bv, off);
      const int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    bi = __shfl_sync(0xffffffffu, bi, 0);
    if (lane == 0) bd[bi] = -INFINITY;
    __syncwarp();
    sel[j] = bi;
  }
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < TOPK; ++j) sum += sc[sel[j]];
    float* w = w_out + (size_t)t * TOPK;
    int* id = id_out + (size_t)t * TOPK;
#pragma unroll
    for (int j = 0; j < TOPK; ++j) {
      float v = sc[sel[j]];
      if (RENORM) v /= sum;
      w[j] = v * ROUTE_SCALE;
      id[j] = sel[j];
      if (COUNT) atomicAdd(&cnt[sel[j]], 1);
    }
  }
}

// ===========================================================================
// 路由元数据：直方图 → 前缀和 → scatter 游标
// ===========================================================================
__global__ void route_count_kernel(const int* __restrict__ ids, int total, int* __restrict__ cnt) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x)
    atomicAdd(&cnt[ids[i]], 1);
}

// 单 block exclusive scan（E 较小）
__global__ void route_scan_kernel(const int* __restrict__ cnt, int* __restrict__ off) {
  const int e = threadIdx.x;
  // 简单的串行/共享前缀和：E=384，用 512 线程做 Hillis-Steele
  __shared__ int tmp[1024];
  tmp[e] = (e < E) ? cnt[e] : 0;
  __syncthreads();
  for (int s = 1; s < E; s <<= 1) {
    int v = (e >= s) ? tmp[e - s] : 0;
    __syncthreads();
    tmp[e] += v;
    __syncthreads();
  }
  if (e == 0) off[0] = 0;
  if (e < E) off[e + 1] = tmp[e];
}

__global__ void route_scatter_kernel(const int* __restrict__ ids, const float* __restrict__ wgt,
                                     int total, int* __restrict__ cursor, int* __restrict__ pos,
                                     int* __restrict__ ptok, float* __restrict__ pwt,
                                     int* __restrict__ inv) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const int e = ids[i];
    const int p = atomicAdd(&cursor[e], 1);
    pos[i] = p;
    ptok[p] = i / TOPK;      // token id
    pwt[p] = wgt[i];
    inv[p] = i;              // 逆置换：permuted 位置 -> 原始 (t*TOPK+j)
  }
}

// ===========================================================================
// permuted_x[pos,:] = x[t,:]，16B 向量化（一个 block 负责一个 permuted 行）
// ===========================================================================
// v1：一个 block 负责一个 permuted 行（x[t] 被重复读 K 次）
__global__ void permute_copy_pos_kernel(const bf16* __restrict__ x, const int* __restrict__ ptok,
                                        bf16* __restrict__ px, int P) {
  const int pos = blockIdx.x;
  if (pos >= P) return;
  const int t = ptok[pos];
  const int n8 = H / 8;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * H);
  uint4* dst = reinterpret_cast<uint4*>(px + (size_t)pos * H);
  for (int i = threadIdx.x; i < n8; i += blockDim.x) dst[i] = src[i];
}

// v2：一个 block 负责一个 token —— x[t] 只读一次，写 K 个 permuted 行。
//     流量从 (K+1)·M·H·2 降到 2·M·H·2（读 M·H·2 + 写 K·M·H·2）。
__global__ void permute_copy_tok_kernel(const bf16* __restrict__ x, const int* __restrict__ pos,
                                        bf16* __restrict__ px, int M) {
  const int t = blockIdx.x;
  if (t >= M) return;
  const int n8 = H / 8;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * H);
  int p[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) p[j] = pos[(size_t)t * TOPK + j];
  for (int i = threadIdx.x; i < n8; i += blockDim.x) {
    const uint4 v = src[i];
#pragma unroll
    for (int j = 0; j < TOPK; ++j)
      reinterpret_cast<uint4*>(px + (size_t)p[j] * H)[i] = v;
  }
}

// v3：同 v2，但用 streaming store（`st.global.cs`）避免写污染 L2
__global__ void permute_copy_tok_cs_kernel(const bf16* __restrict__ x, const int* __restrict__ pos,
                                           bf16* __restrict__ px, int M) {
  const int t = blockIdx.x;
  if (t >= M) return;
  const int n8 = H / 8;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * H);
  int p[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) p[j] = pos[(size_t)t * TOPK + j];
  for (int i = threadIdx.x; i < n8; i += blockDim.x) {
    const uint4 v = src[i];
#pragma unroll
    for (int j = 0; j < TOPK; ++j)
      __stcs(reinterpret_cast<uint4*>(px + (size_t)p[j] * H) + i, v);
  }
}

// y[t,j] = Σ_j w[t,j] * permuted_x[pos(t,j), :]
__global__ void unpermute_kernel(const bf16* __restrict__ px, const int* __restrict__ pos,
                                 const float* __restrict__ wgt, bf16* __restrict__ y, int M) {
  const int t = blockIdx.x;
  if (t >= M) return;
  const int n8 = H / 8;
  for (int i = threadIdx.x; i < n8; i += blockDim.x) {
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < TOPK; ++j) {
      const int p = pos[(size_t)t * TOPK + j];
      const float w = wgt[(size_t)t * TOPK + j];
      const uint4 v = reinterpret_cast<const uint4*>(px + (size_t)p * H)[i];
      const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
      for (int q = 0; q < 8; ++q) a[q] += w * __bfloat162float(b[q]);
    }
    uint4 o;
    bf16* ob = reinterpret_cast<bf16*>(&o);
#pragma unroll
    for (int q = 0; q < 8; ++q) ob[q] = __float2bfloat16(a[q]);
    reinterpret_cast<uint4*>(y + (size_t)t * H)[i] = o;
  }
}

// ===========================================================================
// host reference
// ===========================================================================
static float act_ref(float l, int mode) {
  if (mode == 0) return std::sqrt(std::log1p(std::exp(l)));
  return 1.f / (1.f + std::exp(-l));
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const char* which = (argc > 2) ? argv[2] : "all";
  const int ITER = (argc > 3) ? std::atoi(argv[3]) : 50;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nMoE router: M=%d  H=%d  E=%d  topk=%d  scale=%.1f\n", M, H, E, TOPK, ROUTE_SCALE);

  const size_t aN = (size_t)M * H, bN = (size_t)H * E, cN = (size_t)M * E;
  const int P = M * TOPK;

  bf16 *X, *WgT, *PX, *Y;
  float *logits, *bias, *topk_w;
  int *topk_ids, *cnt, *off, *cursor, *pos, *ptok_i, *inv;
  float* pwt;
  CUDA_CHECK(cudaMalloc(&X, aN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&WgT, bN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&PX, (size_t)P * H * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&Y, aN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&logits, cN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bias, E * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&topk_w, (size_t)P * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&topk_ids, (size_t)P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&cnt, E * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&off, (E + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&cursor, E * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&pos, (size_t)P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&ptok_i, (size_t)P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&pwt, (size_t)P * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&inv, (size_t)P * sizeof(int)));

  std::vector<bf16> hX(aN), hWg(E * H);
  std::vector<float> hbias(E), hlogits(cN);
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : hX) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hWg) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hbias) v = 0.01f * nd(rng);
  CUDA_CHECK(cudaMemcpy(X, hX.data(), aN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bias, hbias.data(), E * sizeof(float), cudaMemcpyHostToDevice));
  // WgT[H,E] = transpose(Wg[E,H])
  {
    std::vector<bf16> hWgT(bN);
    for (int e = 0; e < E; ++e)
      for (int h = 0; h < H; ++h) hWgT[(size_t)h * E + e] = hWg[(size_t)e * H + h];
    CUDA_CHECK(cudaMemcpy(WgT, hWgT.data(), bN * sizeof(bf16), cudaMemcpyHostToDevice));
  }

  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  // ---- 1) gate GEMM：128-tile vs 256-tile（后者减半 B 的 L2 重读） ----
  const double gemm_flops = 2.0 * M * E * H;
  constexpr int SM128 = (2 * 128 * ASP + 2 * BK * BNP) * sizeof(bf16);
  constexpr int SM256 = (2 * 256 * ASP + 2 * BK * BNP) * sizeof(bf16);
  CUDA_CHECK(cudaFuncSetAttribute(gate_mma_pipe_t<128, 4, 2>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, SM128));
  CUDA_CHECK(cudaFuncSetAttribute(gate_mma_pipe_t<256, 8, 2>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, SM256));
  // BM=128/2 CTA-SM 更快（BM=256 只剩 1 CTA/SM，实测反而 -10%）
  auto run_gate = [&] {
    dim3 g(E / BN, M / 128);
    gate_mma_pipe_t<128, 4, 2><<<g, 256, SM128>>>(X, WgT, logits, M, E, H);
  };
  auto run_gate128 = [&] {
    dim3 g(E / BN, M / 128);
    gate_mma_pipe_t<128, 4, 2><<<g, 256, SM128>>>(X, WgT, logits, M, E, H);
  };
  auto run_gate256 = [&] {
    dim3 g(E / BN, M / 256);
    gate_mma_pipe_t<256, 8, 2><<<g, 512, SM256>>>(X, WgT, logits, M, E, H);
  };
  if (want("gate") || want("all")) {
    if ((M % 128) == 0) {
      run_gate128();
      CUDA_CHECK_LAST();
      double t = bench_ms(run_gate128, 5, ITER);
      std::printf("[gate BM128 ] %8.4f ms  %8.2f TFLOPS  %5.2f GB/s(A+C)\n", t,
                  to_tflops(gemm_flops, t), to_gbps((aN + cN) * 2 + cN * 2, t));
      if (M % 256 == 0) {
        run_gate256();
        CUDA_CHECK_LAST();
        double t2 = bench_ms(run_gate256, 5, ITER);
        std::printf("[gate BM256 ] %8.4f ms  %8.2f TFLOPS  (vs BM128 %.2fx)\n", t2,
                    to_tflops(gemm_flops, t2), t / t2);
      }
    } else {
      std::printf("[gate mma  ] skip (M %% 128 != 0)\n");
    }
  }

  // ---- 2) router top-k（WT warp/block；COUNT 版顺手做直方图） ----
  const size_t topk_shm = (size_t)WT * 2 * E * sizeof(float);
  auto run_topk = [&] {
    router_topk_kernel<0, WT, false><<<div_up(M, WT), WT * 32, topk_shm>>>(logits, bias, M, topk_w,
                                                                          topk_ids, nullptr);
  };
  if (want("topk") || want("all")) {
    run_topk();
    CUDA_CHECK_LAST();
    double t = bench_ms(run_topk, 5, ITER);
    std::printf("[router topk] %8.4f ms  %8.1f GB/s(logits)  (%d tokens, %d warp/block)\n", t,
                to_gbps(cN * 4 + (size_t)P * 8, t), M, WT);
  }

  // ---- 正确性：gate + topk vs CPU（采样若干 token） ----
  if (want("check") || want("all")) {
    run_gate();
    run_topk();
    CUDA_CHECK_LAST();
    std::vector<float> hlog(cN), hw((size_t)P);
    std::vector<int> hid((size_t)P);
    CUDA_CHECK(cudaMemcpy(hlog.data(), logits, cN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw.data(), topk_w, (size_t)P * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hid.data(), topk_ids, (size_t)P * sizeof(int), cudaMemcpyDeviceToHost));
    // gate 抽样
    double gerr = 0, gref = 0;
    for (int s = 0; s < 8; ++s) {
      const int r = (s * 1009) % M, c = (s * 131) % E;
      double acc = 0;
      for (int k = 0; k < H; ++k)
        acc += (double)__bfloat162float(hX[(size_t)r * H + k]) *
               (double)__bfloat162float(hWg[(size_t)c * H + k]);
      gerr = std::max(gerr, std::fabs((double)hlog[(size_t)r * E + c] - acc));
      gref = std::max(gref, std::fabs(acc));
    }
    std::printf("[gate check] sampled max_abs_err=%.3e (ref~%.1f) %s\n", gerr, gref,
                gerr / std::max(gref, 1.0) < 2e-2 ? "OK" : "FAIL");
    // router 抽样：重算 CPU top-k 并比对（集合 + 权重）
    double werr = 0; int setok = 0;
    for (int s = 0; s < 8; ++s) {
      const int r = (s * 257) % M;
      std::vector<std::pair<float, int>> ch(E);
      for (int e = 0; e < E; ++e) {
        const float sc = act_ref(hlog[(size_t)r * E + e], 0);
        ch[e] = {sc + hbias[e], e};
      }
      std::sort(ch.begin(), ch.end(), [](auto& a, auto& b) { return a.first > b.first; });
      std::vector<int> cpu_ids(TOPK);
      float sum = 0;
      for (int j = 0; j < TOPK; ++j) {
        cpu_ids[j] = ch[j].second;
        sum += act_ref(hlog[(size_t)r * E + ch[j].second], 0);
      }
      std::sort(cpu_ids.begin(), cpu_ids.end());
      std::vector<int> gpu_ids(hid.begin() + (size_t)r * TOPK, hid.begin() + (size_t)(r + 1) * TOPK);
      std::sort(gpu_ids.begin(), gpu_ids.end());
      const bool set_eq = (cpu_ids == gpu_ids);
      setok += set_eq;
      // 权重：把 gpu 权重归位到 id 上比对
      for (int j = 0; j < TOPK; ++j) {
        int id = hid[(size_t)r * TOPK + j];
        float refw = act_ref(hlog[(size_t)r * E + id], 0) / sum * ROUTE_SCALE;
        werr = std::max(werr, std::fabs((double)hw[(size_t)r * TOPK + j] - refw));
      }
    }
    std::printf("[rtr  check] top-%d set match %d/8 ; max_abs_err(w)=%.3e %s\n", TOPK, setok, werr,
                (setok == 8 && werr < 1e-3) ? "OK" : "FAIL");
  }

  // ---- 3) 路由元数据：独立 count 版 vs 融合进 topk 版 ----
  auto run_meta = [&] {  // 独立 count kernel + scan + scatter
    CUDA_CHECK(cudaMemsetAsync(cnt, 0, E * sizeof(int)));
    route_count_kernel<<<div_up(P, 256), 256>>>(topk_ids, P, cnt);
    route_scan_kernel<<<1, 1024>>>(cnt, off);
    CUDA_CHECK(cudaMemcpyAsync(cursor, off, E * sizeof(int), cudaMemcpyDeviceToDevice));
    route_scatter_kernel<<<div_up(P, 256), 256>>>(topk_ids, topk_w, P, cursor, pos, ptok_i, pwt, inv);
  };
  auto run_route = [&] {  // topk(融合直方图) + scan + scatter
    CUDA_CHECK(cudaMemsetAsync(cnt, 0, E * sizeof(int)));
    router_topk_kernel<0, WT, true><<<div_up(M, WT), WT * 32, topk_shm>>>(logits, bias, M, topk_w,
                                                                         topk_ids, cnt);
    route_scan_kernel<<<1, 1024>>>(cnt, off);
    CUDA_CHECK(cudaMemcpyAsync(cursor, off, E * sizeof(int), cudaMemcpyDeviceToDevice));
    route_scatter_kernel<<<div_up(P, 256), 256>>>(topk_ids, topk_w, P, cursor, pos, ptok_i, pwt, inv);
  };
  if (want("meta") || want("all")) {
    run_meta();
    CUDA_CHECK_LAST();
    double t = bench_ms(run_meta, 5, ITER);
    std::printf("[route meta ] %8.4f ms  (独立 cudaMemset+count+scan+scatter, %d entries)\n", t, P);
    run_route();
    CUDA_CHECK_LAST();
    double t2 = bench_ms(run_route, 5, ITER);
    std::printf("[route+topk ] %8.4f ms  (topk 融合直方图 + scan + scatter)\n", t2);
    // 验证 counts / offsets
    std::vector<int> hcnt(E), hoff(E + 1), hid((size_t)P), hpt((size_t)P);
    CUDA_CHECK(cudaMemcpy(hcnt.data(), cnt, E * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hoff.data(), off, (E + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpt.data(), ptok_i, (size_t)P * sizeof(int), cudaMemcpyDeviceToHost));
    long long tot = 0; int bad = 0;
    for (int e = 0; e < E; ++e) {
      tot += hcnt[e];
      if (hoff[e + 1] - hoff[e] != hcnt[e]) ++bad;
    }
    // ptok 在 [off[e],off[e+1]) 内必须都等于某 token，且该 token 的 ids 含 e
    CUDA_CHECK(cudaMemcpy(hid.data(), topk_ids, (size_t)P * sizeof(int), cudaMemcpyDeviceToHost));
    for (int e = 0; e < E && bad == 0; ++e) {
      for (int p = hoff[e]; p < hoff[e + 1]; ++p) {
        int t = hpt[p]; bool ok = false;
        for (int j = 0; j < TOPK; ++j) ok |= (hid[(size_t)t * TOPK + j] == e);
        if (!ok) { ++bad; break; }
      }
    }
    std::printf("[meta check] total=%lld (==%d)  offsets=%s  ptok membership=%s\n", tot, P,
                bad == 0 ? "OK" : "FAIL", bad == 0 ? "OK" : "FAIL");
  }

  // 单独跑 perm/unperm 时也要先把路由数据准备好（否则 pos 是未初始化值）
  if (want("perm") || want("unperm")) {
    run_topk();
    run_meta();
    CUDA_CHECK_LAST();
  }

  // ---- 4) permutation copy：v1 每行 vs v2 每 token ----
  auto run_perm = [&] { permute_copy_tok_kernel<<<M, 256>>>(X, pos, PX, M); };
  if (want("perm") || want("all")) {
    permute_copy_pos_kernel<<<P, 256>>>(X, ptok_i, PX, P);
    CUDA_CHECK_LAST();
    double t1 = bench_ms([&] { permute_copy_pos_kernel<<<P, 256>>>(X, ptok_i, PX, P); }, 5, ITER);
    double b1 = 2.0 * (double)P * H * 2;  // 读 K 次 + 写 K 次
    std::printf("[perm v1/row ] %8.4f ms  %8.1f GB/s  (%5.1f%% HBM)  %.2f GB moved\n", t1,
                to_gbps(b1, t1), 100.0 * to_gbps(b1, t1) / d.mem_bw_gbps, b1 / 1e9);
    run_perm();
    CUDA_CHECK_LAST();
    double t = bench_ms(run_perm, 5, ITER);
    double bytes = (double)P * H * 2 + (double)M * H * 2;  // 读 1 次 + 写 K 次
    std::printf("[perm v2/tok ] %8.4f ms  %8.1f GB/s  (%5.1f%% HBM)  %.2f GB moved  (vs v1 %.2fx)\n",
                t, to_gbps(bytes, t), 100.0 * to_gbps(bytes, t) / d.mem_bw_gbps, bytes / 1e9,
                t1 / t);
    double t3 = bench_ms([&] { permute_copy_tok_cs_kernel<<<M, 256>>>(X, pos, PX, M); }, 5, ITER);
    std::printf("[perm v3/cs  ] %8.4f ms  %8.1f GB/s  (%5.1f%% HBM)  (streaming store)\n", t3,
                to_gbps(bytes, t3), 100.0 * to_gbps(bytes, t3) / d.mem_bw_gbps);
  }

  // ---- 5) unpermute ----
  auto run_unperm = [&] { unpermute_kernel<<<M, 256>>>(PX, pos, topk_w, Y, M); };
  if (want("unperm") || want("all")) {
    run_unperm();
    CUDA_CHECK_LAST();
    double t = bench_ms(run_unperm, 5, ITER);
    double bytes = (double)P * H * 2 + (double)M * H * 2;
    std::printf("[unpermute  ] %8.4f ms  %8.1f GB/s  (%5.1f%% HBM)  %.2f GB moved\n", t,
                to_gbps(bytes, t), 100.0 * to_gbps(bytes, t) / d.mem_bw_gbps, bytes / 1e9);
  }

  // ---- 往返检查：y[t] == x[t] * Σ_j w[t,j] ----
  if (want("rt") || want("all")) {
    run_topk(); run_meta(); run_perm(); run_unperm();
    CUDA_CHECK_LAST();
    std::vector<int> hid((size_t)P);
    std::vector<float> hw((size_t)P);
    std::vector<bf16> hy(aN), hx(aN);
    CUDA_CHECK(cudaMemcpy(hid.data(), topk_ids, (size_t)P * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw.data(), topk_w, (size_t)P * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy.data(), Y, aN * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hx.data(), X, aN * sizeof(bf16), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int s = 0; s < 8; ++s) {
      const int t = (s * 1009) % M;
      float sw = 0;
      for (int j = 0; j < TOPK; ++j) sw += hw[(size_t)t * TOPK + j];
      for (int h = 0; h < H; h += 997) {
        double e = (double)__bfloat162float(hy[(size_t)t * H + h]) -
                   (double)__bfloat162float(hx[(size_t)t * H + h]) * sw;
        err = std::max(err, std::fabs(e));
        ref = std::max(ref, std::fabs((double)__bfloat162float(hx[(size_t)t * H + h]) * sw));
      }
    }
    std::printf("[roundtrip ] max_abs_err=%.3e (ref~%.2f) %s\n", err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
  }

  // ---- 汇总（用融合 route 版） ----
  if (want("all")) {
    double tg = (M % BM == 0) ? bench_ms(run_gate, 5, ITER) : 0;
    double tr = bench_ms(run_route, 5, ITER);  // topk(融合直方图)+scan+scatter
    double tp = bench_ms(run_perm, 5, ITER);
    double tu = bench_ms(run_unperm, 5, ITER);
    std::printf("\n[total      ] gate %.3f + route(融合 topk+meta) %.3f + permute %.3f + unpermute %.3f = %.3f ms\n",
                tg, tr, tp, tu, tg + tr + tp + tu);
    std::printf("[e2e        ] router(gate+route) %.3f ms ; permutation %.3f ms ; full %.3f ms\n",
                tg + tr, tp + tu, tg + tr + tp + tu);
  }
  return 0;
}
