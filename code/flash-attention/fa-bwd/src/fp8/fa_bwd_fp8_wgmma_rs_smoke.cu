// =============================================================================
// fa_bwd_fp8_wgmma_rs_smoke.cu —— F3b/GEMM3-5 前置：fp8 wgmma **RS_TN**（A 在寄存器）冒烟
// =============================================================================
// 背景（第 174 轮的 SASS 发现，见 docs/03 §102）：用 ncu `--page source --print-source sass`
// 剖析 TE 的 `cudnn_generated_..._flash_bprop_wgmma_f8_knob_26_64x64x128_1x4x1_cga1x1x1`，
// 其 SASS 里 GEMM3/4/5 用的是 **带寄存器 A 操作数的 QGMMA**：
//     QGMMA.64x128x32.F32.E4M3.E5M2 R152, R216, gdesc[UR20], R152
// 即 `wgmma.mma_async ... A-regs, B-desc`（RS 形式），而 ours 的 GEMM3/4/5 仍是
// `HMMA.16816 + LDSM`。ROADMAP「阻塞」里「GEMM3/4/5 上不了 fp8 wgmma」的论据只针对
// **SS_TN（A/B 均描述符）**——RS 允许 A 来自寄存器，于是可以把「需要转置的那个操作数」
// 经 `ldmatrix`（可转置）放进寄存器，绕开「B 必须物理转置 SW128」的死结。
//
// 本冒烟只钉死一个最不确定的点：**`ldmatrix.x4`（对行主序 [M,K] K-major tile）取回的
// 4 个 uint32，是否就是 `wgmma.m64n32k32` RS_TN 的 A 片段布局**（= CUTLASS
// `ALayout_64x32`，即 mma.m16n8k32 的 A 片段在 4 个 warp 上铺 64 行）。B 仍走
// SW128 描述符（已验证），K=128 = 4×k32。
//
//   C[m,n] = Σ_k A[m,k]·B[n,k]  （A: [64,128] 行主序；B: [32,128] SW128 K-major）
//
// 输入取 {-3..3} 整数，e4m3/e5m2 都能精确表示，CPU 参考直接用整数，避开 host fp8 坑。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//         scripts/run.sh src/fp8/fa_bwd_fp8_wgmma_rs_smoke.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---------- fp8 SW128 布局（与 fp8 kernels.cuh / wgmma2_smoke 逐字一致） ----------
__device__ __forceinline__ int sw128_off_fp8(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
__device__ __forceinline__ void sw128_store16_fp8(char* tile, int row, int k0, int K,
                                                  uint4 v) {
  *reinterpret_cast<uint4*>(tile + sw128_off_fp8(row, k0, K)) = v;
}
__device__ __forceinline__ uint32_t sw128_k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t make_desc_sw128_fp8(uint32_t addr,
                                                        uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;  // LBO = 1（K-major 恒 1）
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}

// ---------- wgmma RS_TN：A 在寄存器（4×u32），B 走描述符 ----------
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_m64n32k32_rs_e4e4(float (&d)[16],
                                                        const uint32_t a[4],
                                                        uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "{%16,%17,%18,%19}, %20, p, %22, %23;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1), "n"(1),
        "n"(1));
}
__device__ __forceinline__ void wgmma_m64n32k32_rs_e5e4(float (&d)[16],
                                                        const uint32_t a[4],
                                                        uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e5m2.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "{%16,%17,%18,%19}, %20, p, %22, %23;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1), "n"(1),
        "n"(1));
}

constexpr int BM = 64, BN = 32, HD = 128, THREADS = 128;

// A [BM][HD] 行主序（无 swizzle）；B [BN][HD] SW128 K-major。C[BM][BN]。
template <bool AE5>
__global__ void __launch_bounds__(THREADS) wgmma_rs_smoke_kernel(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ B,
    float* __restrict__ C) {
  __shared__ __align__(16) char sA[BM * HD];             // 行主序 64×128
  __shared__ __align__(1024) char sB[(BN / 8) * (HD / 128) * 1024];  // SW128 32×128
  const int tid = threadIdx.x;
  for (int i = tid; i < BM * HD; i += THREADS) sA[i] = A[i];
  const int n16 = HD / 16;
  for (int i = tid; i < BN * n16; i += THREADS) {
    const int row = i / n16, k0 = (i % n16) * 16;
    sw128_store16_fp8(sB, row, k0, HD, *reinterpret_cast<const uint4*>(B + row * HD + k0));
  }
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const uint32_t sbo = (uint32_t)((HD / 128) * 1024);
  const uint32_t ba = smem_u32(sB);

  float d[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) d[i] = 0.f;

  wgmma_fence();
#pragma unroll
  for (int s = 0; s < HD / 32; ++s) {
    const int koff = s * 32;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int acol = (lane >> 4) * 16;
    uint32_t av[4];
    ldmatrix_x4(smem_u32(&sA[(wid * 16 + arow) * HD + koff + acol]), av);
    uint64_t db = make_desc_sw128_fp8(sw128_k32_addr(ba, s), sbo);
    if (AE5) wgmma_m64n32k32_rs_e5e4(d, av, db);
    else wgmma_m64n32k32_rs_e4e4(d, av, db);
  }
  wgmma_commit();
  wgmma_wait0();

  // CLayout_64x32：row = wl*16 + g + (q>=2?8:0)，col = j*8 + c2 + (q&1)，j∈0..3
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = wid * 16 + g + (q >= 2 ? 8 : 0);
      const int col = j * 8 + c2 + (q & 1);
      C[row * BN + col] = d[j * 4 + q];
    }
}

static unsigned char enc(float x, bool e5) {
  return (unsigned char)(e5 ? __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)
                            : __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

static void run(bool ae5) {
  std::vector<float> A(BM * HD), B(BN * HD);
  srand(2024);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : A) x = rnd();
  for (auto& x : B) x = rnd();
  std::vector<float> Cref(BM * BN);
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < BN; ++n) {
      float a = 0.f;
      for (int k = 0; k < HD; ++k) a += A[m * HD + k] * B[n * HD + k];
      Cref[m * BN + n] = a;
    }
  std::vector<unsigned char> Ab(BM * HD), Bb(BN * HD);
  for (int i = 0; i < BM * HD; ++i) Ab[i] = enc(A[i], ae5);
  for (int i = 0; i < BN * HD; ++i) Bb[i] = enc(B[i], false);

  unsigned char *dA, *dB;
  float *dC;
  CUDA_CHECK(cudaMalloc(&dA, Ab.size()));
  CUDA_CHECK(cudaMalloc(&dB, Bb.size()));
  CUDA_CHECK(cudaMalloc(&dC, BM * BN * 4));
  CUDA_CHECK(cudaMemcpy(dA, Ab.data(), Ab.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, Bb.data(), Bb.size(), cudaMemcpyHostToDevice));
  if (ae5) wgmma_rs_smoke_kernel<true><<<1, THREADS>>>(dA, dB, dC);
  else wgmma_rs_smoke_kernel<false><<<1, THREADS>>>(dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> C(BM * BN);
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
  double e = 0;
  for (int i = 0; i < BM * BN; ++i) e = std::max(e, (double)std::fabs(C[i] - Cref[i]));
  printf("wgmma RS_TN m64n32k32 %s A(ldmatrix,x4) vs CPU: max_abs=%.3e  %s\n",
         ae5 ? "e5m2×e4m3" : "e4m3×e4m3", e, e < 1e-3 ? "PASS" : "FAIL");
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
  printf("=== fp8 wgmma RS_TN（A 来自 ldmatrix 寄存器）冒烟 ===\n");
  run(false);
  run(true);
  return 0;
}
