// 14 融合与 epilogue：GEMM + bias + 激活（GELU/ReLU）的 epilogue 融合
//
//   C[M,N] = act(A[M,K] * B[K,N] + bias[N])
//
// 在 13 的 `mma_pipe` 骨架（128×128×32 bf16 Tensor Core + ldmatrix padding +
// cp.async 双缓冲）上，对比三种实现形态：
//
//   base      : 纯 GEMM，输出裸 C（对标 13 的 mma_pipe）
//   sep       : GEMM 落盘 C，再启一个 epilogue kernel 读 C→加 bias→激活→写回（两趟显存）
//   fused     : 在 GEMM 累加器仍在寄存器里时直接做 bias+激活再落盘（一趟显存）
//   fused_relu: 同上，激活换成 ReLU，验证 epilogue「可插拔」
//
// 运行：scripts/run.sh 14-fusion-epilogue/fusion_epilogue.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>

using bf16 = __nv_bfloat16;

// 与 13 篇一致的 block 形状：128×128 输出，BK=32，256 线程 = 8 warp，warp 布局 4×2。
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int NTHREADS = 256;
constexpr int WM = 4, WN = 2;
constexpr int WARP_M = BM / WM;  // 32
constexpr int WARP_N = BN / WN;  // 64
constexpr int MTM = WARP_M / 16;  // 2
constexpr int MTN = WARP_N / 8;   // 8

constexpr int ASP = BK + 8;  // A smem 行距 padding
constexpr int BNP = BN + 8;  // B smem 行距 padding

// epilogue 模式
enum EpiMode { EPI_NONE = 0, EPI_BIAS = 1, EPI_BIAS_GELU = 2, EPI_BIAS_RELU = 3 };

// ---------------------------------------------------------------------------
// BF16 / smem / mma 工具（与 13 篇相同）
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

template <int LDA, int LDB>
__device__ __forceinline__ void prefetch_tiles(const bf16* __restrict__ A,
                                               const bf16* __restrict__ B, bf16 (*As)[LDA],
                                               bf16 (*Bs)[LDB], int M, int N, int K, int block_row,
                                               int block_col, int k0) {
  const int t = threadIdx.x;
  for (int i = t; i < BM * BK / 8; i += NTHREADS) {
    const int row = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    if (gr < M && gc + 7 < K) {
      __pipeline_memcpy_async(&As[row][c8], &A[(size_t)gr * K + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&As[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  for (int i = t; i < BK * BN / 8; i += NTHREADS) {
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

__device__ __forceinline__ void mma_stage(const bf16 (*As)[ASP], const bf16 (*Bs)[BNP],
                                          float acc[MTM][MTN][4], int warp_row, int warp_col) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    uint32_t a[MTM][4];
#pragma unroll
    for (int i = 0; i < MTM; ++i) {
      const int row = (lane & 15);
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&As[warp_row * WARP_M + i * 16 + row][kk * 16 + col]);
      ldmatrix_x4(addr, a[i]);
    }
    uint32_t b[MTN][2];
#pragma unroll
    for (int g = 0; g < MTN / 2; ++g) {
      const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&Bs[kk * 16 + row][warp_col * WARP_N + g * 16 + col]);
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

__device__ __forceinline__ void zero_acc(float acc[MTM][MTN][4]) {
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
}

// GELU（tanh 近似，与 PyTorch nn.GELU(approximate='tanh') 一致）
__host__ __device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

template <int MODE>
__device__ __forceinline__ float apply_epilogue(float v, float b) {
  if (MODE == EPI_NONE) return v;
  const float x = v + b;
  if (MODE == EPI_BIAS) return x;
  if (MODE == EPI_BIAS_GELU) return gelu_tanh(x);
  return fmaxf(x, 0.0f);  // EPI_BIAS_RELU
}

// 把寄存器里的累加器经 epilogue 后写回全局（列方向 bias 的全局索引就是输出列号）
template <int MODE>
__device__ __forceinline__ void store_acc_epi(float* __restrict__ C, const float* __restrict__ bias,
                                              const float acc[MTM][MTN][4], int N, int block_row,
                                              int block_col, int warp_row, int warp_col) {
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
        const float bv = (MODE == EPI_NONE) ? 0.f : bias[c];
        C[(size_t)r * N + c] = apply_epilogue<MODE>(acc[i][j][q], bv);
      }
    }
}

// ---------------------------------------------------------------------------
// 主 GEMM：mma_pipe + 可选 epilogue（fused 与 base 共用，模板化 MODE）
// ---------------------------------------------------------------------------
template <int MODE>
__global__ void gemm_fused_epi(const bf16* __restrict__ A, const bf16* __restrict__ B,
                               const float* __restrict__ bias, float* __restrict__ C, int M, int N,
                               int K) {
  __shared__ bf16 As[2][BM][ASP];
  __shared__ bf16 Bs[2][BK][BNP];

  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / WN, warp_col = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  float acc[MTM][MTN][4];
  zero_acc(acc);

  prefetch_tiles<ASP, BNP>(A, B, As[0], Bs[0], M, N, K, block_row, block_col, 0);

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + BK;
    if (next < K) {
      prefetch_tiles<ASP, BNP>(A, B, As[stage ^ 1], Bs[stage ^ 1], M, N, K, block_row, block_col,
                               next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(next < K ? 1 : 0);
    __syncthreads();
    mma_stage(As[stage], Bs[stage], acc, warp_row, warp_col);
    __syncthreads();
    stage ^= 1;
  }
  store_acc_epi<MODE>(C, bias, acc, N, block_row, block_col, warp_row, warp_col);
}

// ---------------------------------------------------------------------------
// 独立 epilogue kernel：读 C → 加 bias → 激活 → 写回 C（两趟显存，非融合对照）
// 用 float4 向量化 + grid-stride，尽量跑满 HBM，避免把「两趟显存」做成稻草人。
// ---------------------------------------------------------------------------
template <int MODE>
__global__ void epilogue_kernel(float* __restrict__ C, const float* __restrict__ bias, int N,
                                int n4) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = t; idx < n4; idx += stride) {
    const int c = (idx * 4) % N;
    float4 v = reinterpret_cast<float4*>(C)[idx];
    const float4 b = *reinterpret_cast<const float4*>(&bias[c]);
    v.x = apply_epilogue<MODE>(v.x, b.x);
    v.y = apply_epilogue<MODE>(v.y, b.y);
    v.z = apply_epilogue<MODE>(v.z, b.z);
    v.w = apply_epilogue<MODE>(v.w, b.w);
    reinterpret_cast<float4*>(C)[idx] = v;
  }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
  int N = (argc > 2) ? std::atoi(argv[2]) : 2048;
  int K = (argc > 3) ? std::atoi(argv[3]) : 2048;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nFused epilogue GEMM: M=N=K=%d, FLOPs = %.2f GFLOP\n\n", M, flops / 1e9);
  if (M % BM || N % BN || K % BK) {
    std::printf("M/N must be multiple of %d, K multiple of %d\n", BM, BK);
    return 1;
  }

  const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  bf16 *A, *B;
  float *C, *Cref, *bias;
  CUDA_CHECK(cudaMalloc(&A, aN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&B, bN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&C, cN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&Cref, cN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bias, N * sizeof(float)));

  std::vector<bf16> hA(aN), hB(bN);
  std::vector<float> hC(cN), hBias(N);
  srand(1234);
  for (size_t i = 0; i < aN; ++i)
    hA[i] = __float2bfloat16(1.0f + 0.5f * (float)rand() / RAND_MAX);
  for (size_t i = 0; i < bN; ++i)
    hB[i] = __float2bfloat16(0.5f + 0.5f * (float)rand() / RAND_MAX);
  for (int i = 0; i < N; ++i) hBias[i] = 0.25f * ((float)rand() / RAND_MAX - 0.5f);
  CUDA_CHECK(cudaMemcpy(A, hA.data(), aN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), bN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bias, hBias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // 参考：CPU 算 GEMM + bias + 激活（与 kernel 同一 GELU 近似）
  auto cpu_ref = [&](int r, int c, int mode) {
    double s = 0;
    for (int k = 0; k < K; ++k)
      s += (double)__bfloat162float(hA[(size_t)r * K + k]) * __bfloat162float(hB[(size_t)k * N + c]);
    float x = (float)s + hBias[c];
    if (mode == EPI_BIAS) return x;
    if (mode == EPI_BIAS_GELU) return gelu_tanh(x);
    if (mode == EPI_BIAS_RELU) return fmaxf(x, 0.0f);
    return (float)s;
  };
  auto check = [&](const char* tag, int mode) {
    CUDA_CHECK(cudaMemcpy(hC.data(), C, cN * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 16; ++i) {
      int r = (i * 257) % M, c = (i * 131) % N;
      double e = cpu_ref(r, c, mode);
      err = std::max(err, std::fabs((double)hC[(size_t)r * N + c] - e));
      ref = std::max(ref, std::fabs(e));
    }
    std::printf("  [%-9s] sampled max_abs_err = %.3e (ref~%.1f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 2e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms) {
    double tflops = to_tflops(flops, ms);
    std::printf("%-10s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of bf16 TC peak)\n", tag, ms, tflops,
                100.0 * tflops / 989.0);
  };

  dim3 block(NTHREADS);
  dim3 grid(div_up(N, BN), div_up(M, BM));
  const int n4 = (int)(cN / 4);
  const int epi_blocks = 132 * 16;  // grid-stride，足够覆盖 HBM 并发
  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };

  if (want("base")) {
    gemm_fused_epi<EPI_NONE><<<grid, block>>>(A, B, bias, C, M, N, K);
    CUDA_CHECK_LAST();
    check("base", EPI_NONE);
    double t = bench_ms([&] { gemm_fused_epi<EPI_NONE><<<grid, block>>>(A, B, bias, C, M, N, K); });
    report("base", t);
  }
  if (want("sep")) {
    auto run_sep = [&] {
      gemm_fused_epi<EPI_NONE><<<grid, block>>>(A, B, bias, C, M, N, K);
      epilogue_kernel<EPI_BIAS_GELU><<<epi_blocks, 256>>>(C, bias, N, n4);
    };
    run_sep();
    CUDA_CHECK_LAST();
    check("sep", EPI_BIAS_GELU);
    double t = bench_ms(run_sep);
    report("sep", t);
  }
  if (want("fused")) {
    gemm_fused_epi<EPI_BIAS_GELU><<<grid, block>>>(A, B, bias, C, M, N, K);
    CUDA_CHECK_LAST();
    check("fused", EPI_BIAS_GELU);
    double t =
        bench_ms([&] { gemm_fused_epi<EPI_BIAS_GELU><<<grid, block>>>(A, B, bias, C, M, N, K); });
    report("fused", t);
  }
  if (want("fused_relu")) {
    gemm_fused_epi<EPI_BIAS_RELU><<<grid, block>>>(A, B, bias, C, M, N, K);
    CUDA_CHECK_LAST();
    check("fused_relu", EPI_BIAS_RELU);
    double t = bench_ms(
        [&] { gemm_fused_epi<EPI_BIAS_RELU><<<grid, block>>>(A, B, bias, C, M, N, K); });
    report("fused_relu", t);
  }
  if (want("epi")) {
    gemm_fused_epi<EPI_NONE><<<grid, block>>>(A, B, bias, C, M, N, K);
    epilogue_kernel<EPI_BIAS_GELU><<<epi_blocks, 256>>>(C, bias, N, n4);
    CUDA_CHECK_LAST();
    check("epi", EPI_BIAS_GELU);
    double t = bench_ms([&] { epilogue_kernel<EPI_BIAS_GELU><<<epi_blocks, 256>>>(C, bias, N, n4); });
    const double bytes = 2.0 * cN * sizeof(float);  // 读 + 写 C
    std::printf("%-10s %8.4f ms  %8.1f GB/s  (%.1f MB moved)\n", "epi_only", t,
                bytes / (t * 1e-3) / 1e9, bytes / 1e6);
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(Cref));
  CUDA_CHECK(cudaFree(bias));
  return 0;
}
