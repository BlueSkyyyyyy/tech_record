// 20 MLA 极限冲刺（一）：SW128 swizzle 的 GMMA 冒烟 GEMM
//
// 目的：验证「K-major + 128B swizzle」的 wgmma 描述符与 k16 步进是否正确。
// C[M,N] = A[M,K] * B[N,K]^T，A/B 均 K-major bf16，tile 用 SW128 布局存 smem。
//
// 配置：BM=128, BN=256, BK=64，2 个 warpgroup（256 线程），每个 WG 用
//       wgmma.m64n256k16 算 64×256，K 方向 4 个 k16 步。
//
// 运行：ARCH=sm_90a scripts/run.sh 20-mla-wgmma-sw128/gemm_sw128.cu [M] [N] [K]
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

constexpr int BM = 128, BN = 256, BK = 64, NWG = 2, T = NWG * 128;

// A[M][BK] -> smem SW128，row=m（沿 K-major）
__device__ __forceinline__ void load_tile_sw128(char* tile, const bf16* __restrict__ g, int rows,
                                                int gld, int m0, int k0, int tid) {
  // 每行 BK/8 个 uint4；索引 idx = tid + q*T
  const int per_row = BK / 8;
  const int total = rows * per_row;
  for (int idx = tid; idx < total; idx += T) {
    const int row = idx / per_row, cu = idx % per_row;
    uint4 v = *reinterpret_cast<const uint4*>(&g[(size_t)(m0 + row) * gld + k0 + cu * 8]);
    sw128_store16(tile, row, cu * 8, BK, v);
  }
}

__global__ void __launch_bounds__(T) gemm_sw128_kernel(const bf16* __restrict__ A,
                                                       const bf16* __restrict__ B,
                                                       float* __restrict__ C, int M, int N, int K) {
  extern __shared__ __align__(1024) char smem[];
  char* As = smem;                       // [BM][BK] SW128 = 128*64*2 = 16KB
  char* Bs = smem + BM * BK * 2;         // [BN][BK] SW128 = 256*64*2 = 32KB

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wg = tid >> 7;       // warpgroup 0/1
  const int W = tid >> 5;        // 全局 warp 0..7
  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;  // m64 内行号（0..63）

  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  float O[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) O[i] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile_sw128(As, A, BM, K, m0, k0, tid);
    load_tile_sw128(Bs, B, BN, K, n0, k0, tid);
    __syncthreads();

    const uint32_t as_a = smem_u32(As) + wg * 8 * 1024;  // WG 的 64 行起始（8 个行组）
    const uint32_t bs_a = smem_u32(Bs);
    const uint32_t sbo = (uint32_t)((BK / 64) * 1024);
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < BK / 16; ++s) {
      wgmma_m64n256k16(O, make_desc_sw128(sw128_k16_addr(as_a, s), sbo),
                       make_desc_sw128(sw128_k16_addr(bs_a, s), sbo));
    }
    wgmma_commit();
    wgmma_wait0();
    __syncthreads();
  }

  // epilogue：C[BM][BN] fp32
  const int mbase = m0 + wg * 64;
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    const int col = n0 + t * 8 + (lane & 3) * 2;
    C[(size_t)(mbase + r0) * N + col] = O[t * 4 + 0];
    C[(size_t)(mbase + r0) * N + col + 1] = O[t * 4 + 1];
    C[(size_t)(mbase + r1) * N + col] = O[t * 4 + 2];
    C[(size_t)(mbase + r1) * N + col + 1] = O[t * 4 + 3];
  }
}

// 朴素参考
__global__ void gemm_ref_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += __bfloat162float(A[(size_t)m * K + k]) * __bfloat162float(B[(size_t)n * K + k]);
  C[(size_t)m * N + n] = acc;
}

int main(int argc, char** argv) {
  const int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const int K = (argc > 3) ? std::atoi(argv[3]) : 4096;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nSW128 wgmma GEMM: M=%d N=%d K=%d  (BM=%d BN=%d BK=%d NWG=%d)\n", M, N, K, BM, BN,
              BK, NWG);

  bf16 *A, *B;
  float *C, *Cref;
  CUDA_CHECK(cudaMalloc(&A, (size_t)M * K * 2));
  CUDA_CHECK(cudaMalloc(&B, (size_t)N * K * 2));
  CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&Cref, (size_t)M * N * 4));
  std::vector<bf16> hA((size_t)M * K), hB((size_t)N * K);
  srand(7);
  auto rnd = [] { return __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (auto& x : hA) x = rnd();
  for (auto& x : hB) x = rnd();
  CUDA_CHECK(cudaMemcpy(A, hA.data(), (size_t)M * K * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), (size_t)N * K * 2, cudaMemcpyHostToDevice));

  const size_t shm = (size_t)BM * BK * 2 + (size_t)BN * BK * 2;
  auto fn = gemm_sw128_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(N / BN, M / BM);
  auto run = [&] { fn<<<grid, T, shm>>>(A, B, C, M, N, K); };
  run();
  CUDA_CHECK_LAST();

  // 抽几个点对拍（走朴素 kernel 全量算，shape 小时）
  if ((size_t)M * N * K <= (size_t)1024 * 1024 * 1024) {
    dim3 b(32, 8), g(div_up(N, 32), div_up(M, 8));
    gemm_ref_kernel<<<g, b>>>(A, B, Cref, M, N, K);
    CUDA_CHECK_LAST();
    std::vector<float> hC((size_t)M * N), hR((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), C, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR.data(), Cref, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
      err = std::max(err, std::fabs((double)hC[i] - hR[i]));
      ref = std::max(ref, std::fabs((double)hR[i]));
    }
    std::printf("  [sw128 ] max_abs_err=%.3e (ref~%.3f) %s\n", err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }

  double t = bench_ms(run, 5, 30);
  double tfl = to_tflops(flops, t);
  std::printf("gemm_sw128  %8.4f ms  %8.2f TFLOPS (%5.1f%% peak)\n", t, tfl, 100.0 * tfl / 989.0);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(Cref));
  return 0;
}
