// 24 Muon / MuonClip：Newton–Schulz 正交化算子
//
// Muon 优化器对 2D+ 参数做的核心运算是「把动量矩阵 G 正交化」（zeropower），
// 用 5 步 Newton–Schulz 迭代（Muon 论文系数 a=3.4445, b=-4.7750, c=2.0315）：
//
//   X0 = G / (||G||_F + eps)
//   重复 5 次：
//     A = X X^T
//     B = b*A + c*A@A
//     X = a*X + B@X
//   输出 X（近似 G 的正交极因子）
//
// 每次迭代 3 个 N×N×N GEMM，5 次共 15 个 GEMM → FLOPs = 30 N^3（纯算力受限）。
//
// 本文件对比三条实现路径：
//   custom : 自研 bf16 mma.m16n8k16 + ldmatrix + cp.async GEMM，**把 f*x+g(A) 型
//            epilogue 融进 GEMM 的寄存器结果**（省掉 3 趟 elementwise 读写）；
//   cublas : cuBLAS bf16 GEMM + 独立 elementwise kernel（生产实现口径）；
//   ref    : cuBLAS fp32 走同一算法（正确性参考）。
//
// 运行：scripts/run.sh 24-muonclip/ns_muonclip.cu [N] [which] [cfg]
//   which ∈ custom | cublas | ref | all    cfg = 0..4（自研 GEMM 分块配置）
#include "../common/cuda_utils.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <vector>

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// mma / smem 基础指令
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 分块配置（模板）：BM×BN 输出块，BK 为 k 方向分块，WM×WN 个 warp。
// STAGES 级 cp.async 软流水。smem 行距 +8 消 ldmatrix bank conflict。
// ---------------------------------------------------------------------------
template <int BM_, int BN_, int BK_, int WM_, int WN_, int STAGES_>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_, STAGES = STAGES_;
  static constexpr int NTH = WM_ * WN_ * 32;
  static constexpr int WARP_M = BM / WM, WARP_N = BN / WN;
  static constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
  static constexpr int ASP = BK + 8, BNP = BN + 8;
  static constexpr size_t SMEM = (size_t)STAGES * (BM * ASP + BK * BNP) * sizeof(bf16);
};

template <class C>
__device__ __forceinline__ void prefetch_tiles(const bf16* __restrict__ A,
                                               const bf16* __restrict__ B,
                                               bf16 (*As)[C::ASP], bf16 (*Bs)[C::BNP], int M,
                                               int N, int K, int block_row, int block_col, int k0) {
  const int t = threadIdx.x;
  constexpr int BK = C::BK, BM = C::BM, BN = C::BN, NTH = C::NTH;
  for (int i = t; i < BM * BK / 8; i += NTH) {
    const int row = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    if (gr < M && gc + 7 < K) {
      __pipeline_memcpy_async(&As[row][c8], &A[(size_t)gr * K + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&As[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  for (int i = t; i < BK * BN / 8; i += NTH) {
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

template <class C>
__device__ __forceinline__ void mma_stage(const bf16 (*As)[C::ASP], const bf16 (*Bs)[C::BNP],
                                          float acc[C::MTM][C::MTN][4], int warp_row, int warp_col) {
  constexpr int BK = C::BK, MTM = C::MTM, MTN = C::MTN;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    uint32_t a[MTM][4];
#pragma unroll
    for (int i = 0; i < MTM; ++i) {
      const int row = (lane & 15);
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&As[warp_row * (C::WARP_M) + i * 16 + row][kk * 16 + col]);
      ldmatrix_x4(addr, a[i]);
    }
    uint32_t b[MTN][2];
#pragma unroll
    for (int g = 0; g < MTN / 2; ++g) {
      const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&Bs[kk * 16 + row][warp_col * (C::WARP_N) + g * 16 + col]);
      uint32_t d[4];
      ldmatrix_x4_trans(addr, d);
      b[g * 2][0] = d[0];
      b[g * 2][1] = d[1];
      b[g * 2 + 1][0] = d[2];
      b[g * 2 + 1][1] = d[3];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) mma_m16n8k16(acc[i][j], a[i], b[j]);
  }
}

// epilogue: out = sc_c * Cin[idx] + sc_d * acc
template <class C>
__device__ __forceinline__ void store_acc_epi(bf16* __restrict__ Cout, const bf16* __restrict__ Cin,
                                              const float acc[C::MTM][C::MTN][4], int N,
                                              int block_row, int block_col, int warp_row,
                                              int warp_col, float sc_c, float sc_d) {
  constexpr int MTM = C::MTM, MTN = C::MTN, WARP_M = C::WARP_M, WARP_N = C::WARP_N;
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      const int r0 = block_row + warp_row * WARP_M + i * 16 + group;
      const int c0 = block_col + warp_col * WARP_N + j * 8 + tig * 2;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + (q >= 2 ? 8 : 0);
        const int c = c0 + (q & 1);
        float v = sc_d * acc[i][j][q];
        if (Cin != nullptr) v += sc_c * __bfloat162float(Cin[(size_t)r * N + c]);
        Cout[(size_t)r * N + c] = __float2bfloat16(v);
      }
    }
}

// 通用带 epilogue 的 bf16 GEMM：C[M,N] = sc_c*Cin + sc_d*(A[M,K]@B[K,N])
template <class C>
__global__ void gemm_bf16_epi(const bf16* __restrict__ A, const bf16* __restrict__ B,
                              const bf16* __restrict__ Cin, bf16* __restrict__ Cout, int M, int N,
                              int K, float sc_c, float sc_d) {
  extern __shared__ bf16 smem[];
  bf16* As_base = smem;
  bf16* Bs_base = smem + (size_t)C::STAGES * C::BM * C::ASP;

  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / C::WN, warp_col = wid % C::WN;
  const int block_row = blockIdx.y * C::BM, block_col = blockIdx.x * C::BN;

  float acc[C::MTM][C::MTN][4];
#pragma unroll
  for (int i = 0; i < C::MTM; ++i)
#pragma unroll
    for (int j = 0; j < C::MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  int stage = 0;
#pragma unroll
  for (int s = 0; s < C::STAGES - 1; ++s) {
    if (s * C::BK < K)
      prefetch_tiles<C>(A, B, (bf16(*)[C::ASP])(As_base + (size_t)s * C::BM * C::ASP),
                        (bf16(*)[C::BNP])(Bs_base + (size_t)s * C::BK * C::BNP), M, N, K, block_row,
                        block_col, s * C::BK);
  }
  for (int k0 = 0; k0 < K; k0 += C::BK) {
    const int next = k0 + C::BK * (C::STAGES - 1);
    if (next < K) {
      const int slot = (stage + C::STAGES - 1) % C::STAGES;
      prefetch_tiles<C>(A, B, (bf16(*)[C::ASP])(As_base + (size_t)slot * C::BM * C::ASP),
                        (bf16(*)[C::BNP])(Bs_base + (size_t)slot * C::BK * C::BNP), M, N, K,
                        block_row, block_col, next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(C::STAGES - 1);
    __syncthreads();
    bf16(*As)[C::ASP] = (bf16(*)[C::ASP])(As_base + (size_t)stage * C::BM * C::ASP);
    bf16(*Bs)[C::BNP] = (bf16(*)[C::BNP])(Bs_base + (size_t)stage * C::BK * C::BNP);
    mma_stage<C>(As, Bs, acc, warp_row, warp_col);
    __syncthreads();
    stage = (stage + 1) % C::STAGES;
  }
  store_acc_epi<C>(Cout, Cin, acc, N, block_row, block_col, warp_row, warp_col, sc_c, sc_d);
}

// ---------------------------------------------------------------------------
// elementwise 辅助 kernel
// ---------------------------------------------------------------------------
__global__ void k_scale_bf16(bf16* __restrict__ x, float inv, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) x[i] = __float2bfloat16(__bfloat162float(x[i]) * inv);
}
__global__ void k_frob_partial(const bf16* __restrict__ x, float* __restrict__ part, size_t n) {
  __shared__ float sdata[256];
  float s = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float v = __bfloat162float(x[i]);
    s += v * v;
  }
  sdata[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o) sdata[threadIdx.x] += sdata[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = sdata[0];
}
__global__ void k_frob_partial_f32(const float* __restrict__ x, float* __restrict__ part, size_t n) {
  __shared__ float sdata[256];
  float s = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    s += x[i] * x[i];
  sdata[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o) sdata[threadIdx.x] += sdata[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = sdata[0];
}
__global__ void k_axpby_bf16(bf16* __restrict__ d, const bf16* __restrict__ c,
                             const bf16* __restrict__ g, float alpha, float beta, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride)
    d[i] = __float2bfloat16(alpha * __bfloat162float(c[i]) + beta * __bfloat162float(g[i]));
}
__global__ void k_axpby_f32(float* __restrict__ d, const float* __restrict__ c,
                            const float* __restrict__ g, float alpha, float beta, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) d[i] = alpha * c[i] + beta * g[i];
}
__global__ void k_scale_f32(float* __restrict__ x, float inv, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) x[i] *= inv;
}
template <typename T>
__global__ void k_transpose(const T* __restrict__ in, T* __restrict__ out, int R, int C) {
  __shared__ T tile[32][33];
  const int x = blockIdx.x * 32 + threadIdx.x;
  const int y = blockIdx.y * 32 + threadIdx.y;
  if (x < C && y < R) tile[threadIdx.y][threadIdx.x] = in[(size_t)y * C + x];
  __syncthreads();
  const int xo = blockIdx.y * 32 + threadIdx.x;
  const int yo = blockIdx.x * 32 + threadIdx.y;
  if (xo < R && yo < C) out[(size_t)yo * R + xo] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// 算法常量
// ---------------------------------------------------------------------------
constexpr float NS_A = 3.4445f, NS_B = -4.7750f, NS_C = 2.0315f;
constexpr int NS_STEPS = 5;

static float hsum(const float* p, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) s += p[i];
  return (float)s;
}

// 自研 GEMM 支持的分块配置表
using Cfg0 = Cfg<128, 128, 32, 4, 2, 2>;  // 基线
using Cfg1 = Cfg<128, 128, 64, 4, 2, 3>;
using Cfg2 = Cfg<128, 256, 32, 4, 4, 2>;
using Cfg3 = Cfg<256, 128, 32, 8, 2, 2>;
using Cfg4 = Cfg<128, 128, 32, 2, 4, 3>;
using Cfg5 = Cfg<128, 128, 128, 4, 2, 2>;
using Cfg6 = Cfg<128, 128, 32, 4, 2, 4>;

struct Bufs {
  bf16 *X, *Xt, *A, *M, *tmp;
  float* part;
};

template <class C>
static void launch_gemm(const bf16* A, const bf16* B, const bf16* Cin, bf16* Cout, int M, int N,
                        int K, float sc_c, float sc_d) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(gemm_bf16_epi<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)C::SMEM);
    inited = true;
  }
  dim3 grid(N / C::BN, M / C::BM);
  gemm_bf16_epi<C><<<grid, C::NTH, C::SMEM>>>(A, B, Cin, Cout, M, N, K, sc_c, sc_d);
}

// fuse=true：epilogue 融进 GEMM；fuse=false：GEMM 后跑独立 axpby kernel（对照）
template <class C>
static void ns_custom_cfg(const bf16* dG, bf16* dOut, int N, Bufs& b, int iterations, bool fuse) {
  const size_t nn = (size_t)N * N;
  const int gx = (int)((nn + 1023) / 1024);
  const int part_blocks = 256;
  const dim3 gtrans((N + 31) / 32, (N + 31) / 32);

  CUDA_CHECK(cudaMemcpy(b.X, dG, nn * sizeof(bf16), cudaMemcpyDeviceToDevice));
  k_frob_partial<<<part_blocks, 256>>>(b.X, b.part, nn);
  CUDA_CHECK_LAST();
  std::vector<float> hp(part_blocks);
  CUDA_CHECK(cudaMemcpy(hp.data(), b.part, part_blocks * sizeof(float), cudaMemcpyDeviceToHost));
  const float inv = 1.0f / (std::sqrt(hsum(hp.data(), part_blocks)) + 1e-7f);
  k_scale_bf16<<<gx, 256>>>(b.X, inv, nn);
  CUDA_CHECK_LAST();

  for (int it = 0; it < iterations; ++it) {
    k_transpose<bf16><<<gtrans, dim3(32, 32)>>>(b.X, b.Xt, N, N);
    CUDA_CHECK_LAST();
    launch_gemm<C>(b.X, b.Xt, nullptr, b.A, N, N, N, 0.f, 1.f);
    CUDA_CHECK_LAST();
    if (fuse) {
      launch_gemm<C>(b.A, b.A, b.A, b.M, N, N, N, NS_B, NS_C);
      CUDA_CHECK_LAST();
      launch_gemm<C>(b.M, b.X, b.X, b.tmp, N, N, N, NS_A, 1.f);
      CUDA_CHECK_LAST();
    } else {
      launch_gemm<C>(b.A, b.A, nullptr, b.M, N, N, N, 0.f, NS_C);
      CUDA_CHECK_LAST();
      k_axpby_bf16<<<gx, 256>>>(b.M, b.A, b.M, NS_B, 1.f, nn);  // M = b*A + M
      CUDA_CHECK_LAST();
      launch_gemm<C>(b.M, b.X, nullptr, b.tmp, N, N, N, 0.f, 1.f);
      CUDA_CHECK_LAST();
      k_axpby_bf16<<<gx, 256>>>(b.tmp, b.X, b.tmp, NS_A, 1.f, nn);  // tmp = a*X + tmp
      CUDA_CHECK_LAST();
    }
    std::swap(b.X, b.tmp);
  }
  CUDA_CHECK(cudaMemcpy(dOut, b.X, nn * sizeof(bf16), cudaMemcpyDeviceToDevice));
}

static void ns_custom(const bf16* dG, bf16* dOut, int N, Bufs& b, int iterations, int cfg,
                      bool fuse) {
  switch (cfg) {
    case 0: ns_custom_cfg<Cfg0>(dG, dOut, N, b, iterations, fuse); break;
    case 1: ns_custom_cfg<Cfg1>(dG, dOut, N, b, iterations, fuse); break;
    case 2: ns_custom_cfg<Cfg2>(dG, dOut, N, b, iterations, fuse); break;
    case 3: ns_custom_cfg<Cfg3>(dG, dOut, N, b, iterations, fuse); break;
    case 4: ns_custom_cfg<Cfg4>(dG, dOut, N, b, iterations, fuse); break;
    case 5: ns_custom_cfg<Cfg5>(dG, dOut, N, b, iterations, fuse); break;
    default: ns_custom_cfg<Cfg6>(dG, dOut, N, b, iterations, fuse); break;
  }
}

// ========================================================================
// 路径 2：cuBLAS bf16 GEMM + 独立 elementwise（生产口径）
// ========================================================================
static void cublas_gemm_bf16(cublasHandle_t h, const bf16* A, const bf16* B, bf16* C, int M, int N,
                             int K, float alpha, float beta) {
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K,
               &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static void ns_cublas(const bf16* dG, bf16* dOut, int N, Bufs& b, int iterations) {
  const size_t nn = (size_t)N * N;
  const int gx = (int)((nn + 1023) / 1024);
  const int part_blocks = 256;
  const dim3 gtrans((N + 31) / 32, (N + 31) / 32);

  cublasHandle_t h;
  cublasCreate(&h);

  CUDA_CHECK(cudaMemcpy(b.X, dG, nn * sizeof(bf16), cudaMemcpyDeviceToDevice));
  k_frob_partial<<<part_blocks, 256>>>(b.X, b.part, nn);
  CUDA_CHECK_LAST();
  std::vector<float> hp(part_blocks);
  CUDA_CHECK(cudaMemcpy(hp.data(), b.part, part_blocks * sizeof(float), cudaMemcpyDeviceToHost));
  const float inv = 1.0f / (std::sqrt(hsum(hp.data(), part_blocks)) + 1e-7f);
  k_scale_bf16<<<gx, 256>>>(b.X, inv, nn);
  CUDA_CHECK_LAST();

  for (int it = 0; it < iterations; ++it) {
    k_transpose<bf16><<<gtrans, dim3(32, 32)>>>(b.X, b.Xt, N, N);
    CUDA_CHECK_LAST();
    cublas_gemm_bf16(h, b.X, b.Xt, b.A, N, N, N, 1.f, 0.f);
    cublas_gemm_bf16(h, b.A, b.A, b.M, N, N, N, NS_C, 0.f);
    k_axpby_bf16<<<gx, 256>>>(b.M, b.A, b.M, NS_B, 1.f, nn);
    CUDA_CHECK_LAST();
    cublas_gemm_bf16(h, b.M, b.X, b.tmp, N, N, N, 1.f, 0.f);
    k_axpby_bf16<<<gx, 256>>>(b.tmp, b.X, b.tmp, NS_A, 1.f, nn);
    CUDA_CHECK_LAST();
    std::swap(b.X, b.tmp);
  }
  CUDA_CHECK(cudaMemcpy(dOut, b.X, nn * sizeof(bf16), cudaMemcpyDeviceToDevice));
  cublasDestroy(h);
}

// ========================================================================
// 路径 3：cuBLAS fp32（正确性参考）
// ========================================================================
static void cublas_gemm_f32(cublasHandle_t h, const float* A, const float* B, float* C, int M,
                            int N, int K, float alpha, float beta) {
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

static void ns_ref_f32(const float* dG, float* dOut, int N, float* X, float* Xt, float* A, float* M,
                       float* tmp, float* part, int iterations) {
  const size_t nn = (size_t)N * N;
  const int gx = (int)((nn + 1023) / 1024);
  const int part_blocks = 256;
  const dim3 gtrans((N + 31) / 32, (N + 31) / 32);
  cublasHandle_t h;
  cublasCreate(&h);

  CUDA_CHECK(cudaMemcpy(X, dG, nn * sizeof(float), cudaMemcpyDeviceToDevice));
  k_frob_partial_f32<<<part_blocks, 256>>>(X, part, nn);
  CUDA_CHECK_LAST();
  std::vector<float> hp(part_blocks);
  CUDA_CHECK(cudaMemcpy(hp.data(), part, part_blocks * sizeof(float), cudaMemcpyDeviceToHost));
  const float inv = 1.0f / (std::sqrt(hsum(hp.data(), part_blocks)) + 1e-7f);
  k_scale_f32<<<gx, 256>>>(X, inv, nn);
  CUDA_CHECK_LAST();

  for (int it = 0; it < iterations; ++it) {
    k_transpose<float><<<gtrans, dim3(32, 32)>>>(X, Xt, N, N);
    CUDA_CHECK_LAST();
    cublas_gemm_f32(h, X, Xt, A, N, N, N, 1.f, 0.f);
    cublas_gemm_f32(h, A, A, M, N, N, N, NS_C, 0.f);
    k_axpby_f32<<<gx, 256>>>(M, A, M, NS_B, 1.f, nn);
    CUDA_CHECK_LAST();
    cublas_gemm_f32(h, M, X, tmp, N, N, N, 1.f, 0.f);
    k_axpby_f32<<<gx, 256>>>(tmp, X, tmp, NS_A, 1.f, nn);
    CUDA_CHECK_LAST();
    std::swap(X, tmp);
  }
  CUDA_CHECK(cudaMemcpy(dOut, X, nn * sizeof(float), cudaMemcpyDeviceToDevice));
  cublasDestroy(h);
}

// ========================================================================
int main(int argc, char** argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const char* which = (argc > 2) ? argv[2] : "all";
  int cfg = (argc > 3) ? std::atoi(argv[3]) : 0;

  DeviceInfo d = device_info(0);
  print_device_info(d);

  const size_t nn = (size_t)N * N;
  const double flops = 30.0 * (double)N * N * N;
  std::printf("\nNewton-Schulz (Muon): N=%d  steps=%d  FLOPs = %.2f TFLOP\n\n", N, NS_STEPS,
              flops / 1e12);

  std::vector<bf16> hG(nn);
  srand(1234);
  for (size_t i = 0; i < nn; ++i)
    hG[i] = __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f));
  bf16* dG;
  CUDA_CHECK(cudaMalloc(&dG, nn * sizeof(bf16)));
  CUDA_CHECK(cudaMemcpy(dG, hG.data(), nn * sizeof(bf16), cudaMemcpyHostToDevice));

  Bufs b{};
  CUDA_CHECK(cudaMalloc(&b.X, nn * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&b.Xt, nn * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&b.A, nn * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&b.M, nn * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&b.tmp, nn * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&b.part, 256 * sizeof(float)));
  bf16* dOutC;
  CUDA_CHECK(cudaMalloc(&dOutC, nn * sizeof(bf16)));
  std::vector<bf16> hOut(nn), hRef(nn);

  bool do_custom = !strcmp(which, "all") || !strcmp(which, "custom");
  bool do_unfused = !strcmp(which, "all") || !strcmp(which, "unfused");
  bool do_cublas = !strcmp(which, "all") || !strcmp(which, "cublas");
  bool do_ref = !strcmp(which, "all") || !strcmp(which, "ref");
  bool do_single = !strcmp(which, "single");

  if (do_ref || do_custom || do_cublas || do_unfused) {
    float *X, *Xt, *A, *M, *tmp, *part, *dOutF, *dGf;
    CUDA_CHECK(cudaMalloc(&X, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Xt, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&M, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&part, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutF, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGf, nn * sizeof(float)));
    std::vector<float> hGf(nn);
    for (size_t i = 0; i < nn; ++i) hGf[i] = __bfloat162float(hG[i]);
    CUDA_CHECK(cudaMemcpy(dGf, hGf.data(), nn * sizeof(float), cudaMemcpyHostToDevice));
    double t = bench_ms([&] { ns_ref_f32(dGf, dOutF, N, X, Xt, A, M, tmp, part, NS_STEPS); }, 2, 5);
    std::vector<float> hF(nn);
    CUDA_CHECK(cudaMemcpy(hF.data(), dOutF, nn * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nn; ++i) hRef[i] = __float2bfloat16(hF[i]);
    std::printf("ref(fp32 cublas)   %8.3f ms   %7.2f TFLOPS\n", t, to_tflops(flops, t));
    double ierr = 0;
    for (int s = 0; s < 64; ++s) {
      int r = (s * 61) % N, c = (s * 197 + 7) % N;
      double dot = 0;
      for (int k = 0; k < N; ++k)
        dot += (double)hF[(size_t)r * N + k] * (double)hF[(size_t)c * N + k];
      double eye = (r == c) ? 1.0 : 0.0;
      ierr += (dot - eye) * (dot - eye);
    }
    std::printf("  ref orthogonality  ||X X^T - I||(sample) = %.4f\n\n", std::sqrt(ierr / 64));
    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(Xt));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(M));
    CUDA_CHECK(cudaFree(tmp));
    CUDA_CHECK(cudaFree(part));
    CUDA_CHECK(cudaFree(dOutF));
    CUDA_CHECK(cudaFree(dGf));
  }

  if (do_single) {
    const double g1 = 2.0 * (double)N * N * N;
    double t0 = bench_ms([&] { launch_gemm<Cfg1>(b.X, b.Xt, nullptr, b.A, N, N, N, 0.f, 1.f); }, 5,
                         30);
    std::printf("single GEMM custom cfg1   %8.3f ms  %7.2f TFLOPS\n", t0, to_tflops(g1, t0));
    cublasHandle_t hs;
    cublasCreate(&hs);
    double tc = bench_ms([&] { cublas_gemm_bf16(hs, b.X, b.Xt, b.A, N, N, N, 1.f, 0.f); }, 5, 30);
    cublasDestroy(hs);
    std::printf("single GEMM cuBLAS        %8.3f ms  %7.2f TFLOPS\n", tc, to_tflops(g1, tc));
    std::printf("  -> custom = %.1f%% of cuBLAS (time %.2fx)\n", 100.0 * tc / t0, t0 / tc);
  }

  auto check = [&](const char* name) {
    double err = 0, ref = 0;
    for (size_t i = 0; i < nn; ++i) {
      double a = __bfloat162float(hOut[i]), e = __bfloat162float(hRef[i]);
      err = std::max(err, std::fabs(a - e));
      ref = std::max(ref, std::fabs(e));
    }
    std::printf("    [%s] max_abs_err vs fp32 ref = %.3e (ref~%.4f) %s\n", name, err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  };

  if (do_custom) {
    double t = bench_ms([&] { ns_custom(dG, dOutC, N, b, NS_STEPS, cfg, true); }, 2, 10);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOutC, nn * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::printf("custom fused cfg=%d    %8.3f ms   %7.2f TFLOPS  (%5.1f%% peak)\n", cfg, t,
                to_tflops(flops, t), 100.0 * to_tflops(flops, t) / 989.0);
    if (do_ref) check("custom");
  }
  if (do_unfused) {
    double t = bench_ms([&] { ns_custom(dG, dOutC, N, b, NS_STEPS, cfg, false); }, 2, 10);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOutC, nn * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::printf("custom unfused cfg=%d  %8.3f ms   %7.2f TFLOPS  (%5.1f%% peak)\n", cfg, t,
                to_tflops(flops, t), 100.0 * to_tflops(flops, t) / 989.0);
    if (do_ref) check("custom_unfused");
  }
  if (do_cublas) {
    double t = bench_ms([&] { ns_cublas(dG, dOutC, N, b, NS_STEPS); }, 2, 10);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOutC, nn * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::printf("cublas+elementwise    %8.3f ms   %7.2f TFLOPS  (%5.1f%% peak)\n", t,
                to_tflops(flops, t), 100.0 * to_tflops(flops, t) / 989.0);
    if (do_ref) check("cublas");
  }

  CUDA_CHECK(cudaFree(dG));
  CUDA_CHECK(cudaFree(dOutC));
  CUDA_CHECK(cudaFree(b.X));
  CUDA_CHECK(cudaFree(b.Xt));
  CUDA_CHECK(cudaFree(b.A));
  CUDA_CHECK(cudaFree(b.M));
  CUDA_CHECK(cudaFree(b.tmp));
  CUDA_CHECK(cudaFree(b.part));
  return 0;
}
