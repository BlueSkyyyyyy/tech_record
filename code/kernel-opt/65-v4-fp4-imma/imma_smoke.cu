// 65 冒烟：验证 FP4(e2m1) -> IMMA m16n8k32 的片段映射与 host 置换布局。
//
// 目标：把一组 (M=16 token, N=16 channel, K=128) 的小 GEMM 用
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
// 算出来，和 CPU 参考对拍。A（激活）int8 行主序；B（权重）由 FP4 现场解码。
//
// 关键约定（与 48 篇一致）：
//   A 片段：a0=(r,kw*4), a1=(r+8,kw*4), a2=(r,kw*4+16), a3=(r+8,kw*4+16)，r=lane>>2, kw=lane&3
//   B 片段：b0=(n=lane>>2, k=kw*4..+3), b1=(同n, k=kw*4+16..+19)
//   C 片段：c0=(r,(lane&3)*2), c1=(r,(lane&3)*2+1), c2=(r+8,...), c3=(r+8,...)
//
// host 置换布局：把每个 lane 需要的 8 个 fp4 nibble 装进一个 uint32（nibble 顺序即
//   [k=kw*4..+3, k=kw*4+16..+19]），于是 `decode_word_prmt` 一次出一个 lane 的 (b0,b1)。
#include "../common/cuda_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <vector>

__device__ __forceinline__ int e2m1_i8(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  int v = (int)(((e ? 2u : 0u) + m) << (e ? e - 1 : 0u));
  return s ? -v : v;
}
__host__ __device__ __forceinline__ float e2m1f(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  float v = (e == 0) ? (m * 0.5f) : ((1.f + m * 0.5f) * exp2f((int)e - 1));
  return s ? -v : v;
}

// 62 篇原样：一个 uint32（8 nibble）-> 8 个有符号 int8（a=低 4，b=高 4）。
__device__ __forceinline__ void decode_word_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  constexpr uint32_t T0 = 0x03020100u;
  constexpr uint32_t T1 = 0x0C080604u;
  constexpr uint32_t T2 = 0xFDFEFF00u;
  constexpr uint32_t T3 = 0xF4F8FAFCu;
  const uint32_t hi = w >> 16;
  const uint32_t mlo = __byte_perm(T0, T1, w);
  const uint32_t mhi = __byte_perm(T0, T1, hi);
  const uint32_t slo = __byte_perm(T2, T3, w);
  const uint32_t shi = __byte_perm(T2, T3, hi);
  const uint32_t klo = __byte_perm(0u, ~0u, ((w >> 3) & 0x11111111u) * 5u);
  const uint32_t khi = __byte_perm(0u, ~0u, ((w >> 19) & 0x11111111u) * 5u);
  a = mlo ^ ((mlo ^ slo) & klo);
  b = mhi ^ ((mhi ^ shi) & khi);
}

__device__ __forceinline__ void imma_m16n8k32(int c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 置换：Wp[N][K/2] FP4 -> Wperm[nblock][kblock][chunk][lane(32)] uint32
__global__ void permute_fp4(const uint8_t* Wp, uint32_t* Wout, int N, int K) {
  const int NK = K / 128;
  const long total = (long)(N / 8) * NK * 4 * 32;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int lane = i & 31;
    long r = i >> 5;
    const int chunk = r & 3; r >>= 2;
    const int kblock = (int)(r % NK); r /= NK;
    const int nblock = (int)r;
    const int n = nblock * 8 + (lane >> 2);
    const int g = lane & 3;
    const int base = kblock * 128 + chunk * 32 + g * 4;
    uint32_t word = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k = base + t;
      const int nib = (Wp[(size_t)n * (K / 2) + k / 2] >> ((k & 1) * 4)) & 0xF;
      word |= (uint32_t)nib << (4 * t);
      const int k2 = base + 16 + t;
      const int nib2 = (Wp[(size_t)n * (K / 2) + k2 / 2] >> ((k2 & 1) * 4)) & 0xF;
      word |= (uint32_t)nib2 << (4 * (4 + t));
    }
    Wout[i] = word;
  }
}

// 单 block、单 warp 的 IMMA：M=16, N=16(MTN=2), K=128
__global__ void imma_smoke(const int8_t* __restrict__ A, const uint32_t* __restrict__ Wperm,
                           const float* __restrict__ sw, const float* __restrict__ sa,
                           float* __restrict__ C, int K) {
  constexpr int MTN = 2;
  const int ASP = K + 16;
  extern __shared__ __align__(16) char sm[];
  int8_t* As = reinterpret_cast<int8_t*>(sm);
  for (int i = threadIdx.x; i < 16 * K; i += 32) {
    const int r = i / K, c = i % K;
    As[r * ASP + c] = A[r * K + c];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int r0 = lane >> 2, aqw = lane & 3;
  const int NK = K / 128;
  const int NK32 = K / 32;
  const uint32_t* As32 = reinterpret_cast<const uint32_t*>(As);
  const int ASP4 = ASP / 4;

  float facc[MTN][4];
#pragma unroll
  for (int j = 0; j < MTN; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) facc[j][q] = 0.f;

  for (int kb = 0; kb < NK; ++kb) {
    const float s0 = sa[r0 * NK + kb];
    const float s1 = sa[(r0 + 8) * NK + kb];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint32_t a[4];
      a[0] = As32[r0 * ASP4 + c * 8 + aqw];
      a[1] = As32[(r0 + 8) * ASP4 + c * 8 + aqw];
      a[2] = As32[r0 * ASP4 + c * 8 + 4 + aqw];
      a[3] = As32[(r0 + 8) * ASP4 + c * 8 + 4 + aqw];
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        int acci[4] = {0, 0, 0, 0};
        const uint32_t w = Wperm[(((size_t)j * NK + kb) * 4 + c) * 32 + lane];
        uint32_t b[2];
        decode_word_prmt(w, b[0], b[1]);
        imma_m16n8k32(acci, a, b);
        const int n0 = (lane & 3) * 2;
        const float w0 = sw[(j * 8 + n0) * NK32 + kb * 4 + c];
        const float w1 = sw[(j * 8 + n0 + 1) * NK32 + kb * 4 + c];
        facc[j][0] += s0 * w0 * (float)acci[0];
        facc[j][1] += s0 * w1 * (float)acci[1];
        facc[j][2] += s1 * w0 * (float)acci[2];
        facc[j][3] += s1 * w1 * (float)acci[3];
      }
    }
  }
#pragma unroll
  for (int j = 0; j < MTN; ++j) {
    const int n0 = (lane & 3) * 2;
    C[r0 * 16 + j * 8 + n0] = 0.5f * facc[j][0];
    C[r0 * 16 + j * 8 + n0 + 1] = 0.5f * facc[j][1];
    C[(r0 + 8) * 16 + j * 8 + n0] = 0.5f * facc[j][2];
    C[(r0 + 8) * 16 + j * 8 + n0 + 1] = 0.5f * facc[j][3];
  }
}

int main() {
  const int M = 16, N = 16, K = 128;
  std::vector<int8_t> hA(M * K);
  std::vector<uint8_t> hWp(N * (K / 2));
  std::vector<float> hsw(N * (K / 32)), hsa(M * (K / 128)), hC(M * N), hRef(M * N, 0.f);
  unsigned st = 7u;
  auto rnd = [&]() { st = st * 1664525u + 1013904223u; return st >> 8; };
  for (auto& v : hA) v = (int8_t)((int)(rnd() % 17) - 8);
  for (auto& v : hWp) v = (uint8_t)(rnd() % 256);
  for (auto& v : hsw) v = exp2f((float)((int)(rnd() % 9) - 4));
  for (auto& v : hsa) v = ((int)(rnd() % 100) + 1) / 50.f;
  if (getenv("ONES")) {
    for (auto& v : hA) v = 1;
    for (auto& v : hWp) v = 0x22;
    for (auto& v : hsw) v = 1.f;
    for (auto& v : hsa) v = 1.f;
  }

  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) {
        const uint8_t byte = hWp[(size_t)n * (K / 2) + k / 2];
        const unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
        acc += (double)e2m1f(nib) * (double)hA[(size_t)r * K + k] *
               (double)hsw[(size_t)n * (K / 32) + k / 32] * (double)hsa[(size_t)r * (K / 128) + k / 128];
      }
      hRef[(size_t)r * N + n] = (float)acc;
    }

  int8_t *dA; uint32_t* dW; float *dsw, *dsa, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size()));
  CUDA_CHECK(cudaMalloc(&dW, hWp.size()));
  CUDA_CHECK(cudaMalloc(&dsw, hsw.size() * 4));
  CUDA_CHECK(cudaMalloc(&dsa, hsa.size() * 4));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsw, hsw.data(), hsw.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsa, hsa.data(), hsa.size() * 4, cudaMemcpyHostToDevice));
  {
    uint8_t* dWp; CUDA_CHECK(cudaMalloc(&dWp, hWp.size()));
    CUDA_CHECK(cudaMemcpy(dWp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
    permute_fp4<<<64, 256>>>(dWp, dW, N, K);
    CUDA_CHECK_LAST();
    cudaFree(dWp);
  }
  const int smem = (K + 16) * 16;
  imma_smoke<<<1, 32, smem>>>(dA, dW, dsw, dsa, dC, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));

  double mr = 0, mref = 0;
  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n) {
      const double a = hC[(size_t)r * N + n], b = hRef[(size_t)r * N + n];
      mr = std::max(mr, std::fabs(a - b));
      mref = std::max(mref, std::fabs(b));
    }
  std::printf("imma_smoke: max_abs_err=%.4e  max_ref=%.4e  %s\n", mr, mref,
              mr < 1e-2 * mref + 1e-4 ? "OK" : "FAIL");
  if (mr >= 1e-2 * mref + 1e-4) {
    for (int r = 0; r < 4; ++r) {
      for (int n = 0; n < 8; ++n)
        std::printf("  [%d,%d] got=%9.4f ref=%9.4f\n", r, n, hC[r * N + n], hRef[r * N + n]);
    }
  }
  return 0;
}
