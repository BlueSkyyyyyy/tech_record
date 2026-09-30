// =============================================================================
// fa_bwd_fp8_wgmma345_smoke.cu —— F3b 主体前置第二步：把 GEMM3/4/5 的转置 B 操作数
//   **从真实的 SW128 tile 里**造出来，并验证 wgmma 能吃「小 K」的 no-swizzle B 描述符
// =============================================================================
// 背景（第 174/175 轮 O79/O80）：
//   * O79：`ldmatrix.x4`（行主序 [M,K] K-major）取回的 4×u32 即 `wgmma RS_TN` 的 A 片段。
//   * O80：把「逐字节转置」用 `ldmatrix.x4.trans + PRMT 0x5140` 钉死（源 **行主序**），
//     并把转置结果直接落成 SW128，跑通 wgmma RS GEMM3（max_abs=0）。
//   * 但真实 fp8 主 kernel 里 Q/K/dO **以 SW128 存储**（TMA 搬入），B=dOᵀ/Qᵀ/Kᵀ 要从
//     SW128 源转置；且默认几何 BM=64/BN=32 ⇒ GEMM5(dQ) 的 K=BN=32 < 128B，**用不了 SW128
//     描述符**（SW128 要求 K 维连续 128B）。TE 的 64x64x128/384 线程之所以能 QGMMA，是因为
//     其 tile 更宽。
//
// 本冒烟钉死两个「接进 fp8_mma_body 前必须确定」的点：
//   (A) **no-swizzle K-major 描述符**（`layout_type=0`）：B [N][K] 行主序、K 连续 <128B 时，
//       wgmma 是否可正确消费。扫几组 (LBO,SBO) 编码找对的那个。
//   (B) **从 SW128 源做逐字节转置**：把 [R][C] SW128 tile 转成行主序 [C][R]（K-major），
//       验证 `ldmatrix.x4.trans` 在 SW128 源上的 lane 地址公式（只改地址、不改 PRMT 数学）。
//   (C) 端到端 **GEMM5（dQ）**：A=dS2[64][32] e5m2 行主序（stride 48）经 ldmatrix；
//       B=Kᵀ[128][32] no-swizzle（由 (B) 从 SW128 K 造）；wgmma m64n32k32 ×4 与 CPU 比。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//         scripts/run.sh src/fp8/fa_bwd_fp8_wgmma345_smoke.cu
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

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
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

// SW128（K-major）物理偏移：与 fp8 kernels.cuh / O80 smoke 逐字一致。
__device__ __forceinline__ int sw128_off_fp8(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}

// no-swizzle（`layout_type=0`）K-major 描述符：B [N][K] 行主序、K 连续。
//   LBO = 相邻 core matrix（8×16B）沿 K 的字节距 = 16B（K≥16）；
//   SBO = 相邻 core matrix 沿 N 的字节距 = 8 行 × row_stride。
__device__ __forceinline__ uint64_t make_desc_noswz(uint32_t addr, uint32_t lbo_b,
                                                    uint32_t sbo_b) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_b >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_b >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 62;  // layout_type = 0（no swizzle / interleave）
  return d;
}

// CUTLASS canonical Major-K INTERLEAVE（无 swizzle）物理布局（单位 uint128=16B）：
//   ((8,n),2):((1,SBO),LBO)  ⇒ 8 行的 stride 恒为 1 个 uint128（16B 相邻），
//   n（8 行组）的 stride = SBO、K core（16 元素）的 stride = LBO。
//   元素 (r,k) 字节偏移 = 16*((r&7) + SBO*(r>>3) + LBO*(k>>4)) + (k&15)。
__device__ __forceinline__ int inter_k_off(int r, int k, int SBO_u, int LBO_u) {
  return 16 * ((r & 7) + SBO_u * (r >> 3) + LBO_u * (k >> 4)) + (k & 15);
}

// -----------------------------------------------------------------------------
// (A) no-swizzle B 描述符：C[M=64,N=64]=A[64,32]·B[64,32]ᵀ，A e5m2 RS，B e4m3。
//   B 按 canonical INTERLEAVE Major-K 存（LBO_u/SBO_u 由参数给）。扫候选编码。
// -----------------------------------------------------------------------------
__global__ void noswz_kernel(const unsigned char* __restrict__ A,
                             const unsigned char* __restrict__ B, int lbo_u,
                             int sbo_u, float* __restrict__ Cout) {
  constexpr int M = 64, N = 64, K = 32;
  __shared__ __align__(1024) char sA[M * K];
  __shared__ __align__(1024) char sB[4096];  // 覆盖扫描里的最大偏移
  const int tid = threadIdx.x;
  for (int i = tid; i < M * K; i += 128) sA[i] = A[i];
  for (int i = tid; i < N * K; i += 128)
    sB[inter_k_off(i / K, i % K, sbo_u, lbo_u)] = B[i];
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const uint32_t ba = smem_u32(sB);
  float d[2][16];
#pragma unroll
  for (int nn = 0; nn < 2; ++nn)
#pragma unroll
    for (int i = 0; i < 16; ++i) d[nn][i] = 0.f;

  wgmma_fence();
#pragma unroll
  for (int s = 0; s < K / 32; ++s) {  // 1 步
    const int koff = s * 32;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int acol = (lane >> 4) * 16;
    uint32_t av[4];
    ldmatrix_x4(smem_u32(&sA[(wid * 16 + arow) * K + koff + acol]), av);
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      // 第 nn 个 32 行 n-tile = 4 个 8 行组 ⇒ 字节偏移 = 4*SBO_u*16。
      uint32_t a = ba + (uint32_t)(nn * 4 * sbo_u * 16);
      uint64_t db = make_desc_noswz(a, (uint32_t)(lbo_u * 16), (uint32_t)(sbo_u * 16));
      wgmma_m64n32k32_rs_e5e4(d[nn], av, db);
    }
  }
  wgmma_commit();
  wgmma_wait0();

#pragma unroll
  for (int nn = 0; nn < 2; ++nn)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = wid * 16 + g + (q >= 2 ? 8 : 0);
        const int col = nn * 32 + j * 8 + c2 + (q & 1);
        Cout[row * N + col] = d[nn][j * 4 + q];
      }
}

// -----------------------------------------------------------------------------
// (B) 从 **SW128 源** 逐字节转置 [R][C]->[C][R]（行主序）。
//   与 O80 的 transpose_store 相同数学，但每 lane 的源地址走 `sw128_off`。
// -----------------------------------------------------------------------------
__device__ __forceinline__ void transpose_store_sw128_src(
    const unsigned char* sSrc, unsigned char* sDst, int R, int C, int wid,
    int lane) {
  const int nblkC = C / 64;
  const int nwork = (R / 8) * nblkC;
  for (int w = wid; w < nwork; w += 4) {
    const int rblk = w / nblkC, cblk = w % nblkC;
    const int r0 = rblk * 8, c0 = cblk * 64;
    const int mat = lane >> 3, row = lane & 7;
    uint32_t reg[4];
    // 源地址：逻辑 (r0+row, c0+mat*16) 的 SW128 物理位置。
    ldmatrix_x4_trans(smem_u32(sSrc + sw128_off_fp8(r0 + row, c0 + mat * 16, C)),
                      reg);
    const int p = lane & 3, q = lane >> 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 8 * i + q;
      const uint32_t word = __byte_perm(reg[i], reg[i] >> 16, 0x5140);
      const int d0 = c0 + 2 * k, d1 = d0 + 1;
      const int m = r0 + 2 * p;
      *reinterpret_cast<uint16_t*>(sDst + (size_t)d0 * R + m) =
          (uint16_t)(word & 0xffffu);
      *reinterpret_cast<uint16_t*>(sDst + (size_t)d1 * R + m) =
          (uint16_t)(word >> 16);
    }
  }
}

// 与上同，但目标写成 canonical INTERLEAVE Major-K（K=R 连续、8 行组 + 16 元素 core）。
//   SBO_u=16、LBO_u=8。目标 d 是「行」(N=C)、m 是 K。
__device__ __forceinline__ void transpose_store_sw128_src_inter(
    const unsigned char* sSrc, unsigned char* sDst, int R, int C, int wid,
    int lane) {
  const int nblkC = C / 64;
  const int nwork = (R / 8) * nblkC;
  for (int w = wid; w < nwork; w += 4) {
    const int rblk = w / nblkC, cblk = w % nblkC;
    const int r0 = rblk * 8, c0 = cblk * 64;
    const int mat = lane >> 3, row = lane & 7;
    uint32_t reg[4];
    ldmatrix_x4_trans(smem_u32(sSrc + sw128_off_fp8(r0 + row, c0 + mat * 16, C)),
                      reg);
    const int p = lane & 3, q = lane >> 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 8 * i + q;
      const uint32_t word = __byte_perm(reg[i], reg[i] >> 16, 0x5140);
      const int d0 = c0 + 2 * k, d1 = d0 + 1;
      const int m = r0 + 2 * p;
      *reinterpret_cast<uint16_t*>(sDst + inter_k_off(d0, m, 16, 8)) =
          (uint16_t)(word & 0xffffu);
      *reinterpret_cast<uint16_t*>(sDst + inter_k_off(d1, m, 16, 8)) =
          (uint16_t)(word >> 16);
    }
  }
}

__global__ void sw128_transpose_kernel(const unsigned char* __restrict__ X,
                                       unsigned char* __restrict__ Y, int R,
                                       int C) {
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* sX = smem;             // [R][C] SW128
  unsigned char* sY = smem + (size_t)(R / 8) * (C / 128) * 1024;  // [C][R]
  const int tid = threadIdx.x;
  for (int i = tid; i < R * C; i += 128) {
    const int r = i / C, c = i % C;
    sX[sw128_off_fp8(r, c, C)] = X[(size_t)r * C + c];
  }
  __syncthreads();
  transpose_store_sw128_src(sX, sY, R, C, tid >> 5, tid & 31);
  __syncthreads();
  for (int i = tid; i < R * C; i += 128) {
    const int c = i / R, r = i % R;
    Y[(size_t)i] = sY[(size_t)c * R + r];
  }
}

// -----------------------------------------------------------------------------
// (C) 端到端 GEMM5（dQ）：A=dS2[64][32] e5m2 行主序（stride 48，含 padding）经 ldmatrix；
//   B=Kᵀ[128][32] no-swizzle（由 (B) 从 SW128 K[32][128] 造）。m64n32k32 ×4。
//   C[m][d]=Σ_j dS2[m][j]·K[j][d]。
// -----------------------------------------------------------------------------
__global__ void gemm5_kernel(const unsigned char* __restrict__ dS2,  // [64][32]
                             const unsigned char* __restrict__ K,     // [32][128]
                             float* __restrict__ Cout) {
  constexpr int BM = 64, BN = 32, HD = 128;
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* sA = smem;                                  // [64*48] dS2 行主序
  unsigned char* sKs = sA + BM * 48;                         // [32][128] SW128 K
  unsigned char* sKt = sKs + (BN / 8) * (HD / 128) * 1024;   // [128][32] Kᵀ INTERLEAVE
  const int tid = threadIdx.x;
  for (int i = tid; i < BM * 32; i += 128) {
    const int r = i / 32, c = i % 32;
    sA[r * 48 + c] = dS2[i];  // 打 stride 48 padding
  }
  for (int i = tid; i < BN * HD; i += 128) {
    const int r = i / HD, c = i % HD;
    sKs[sw128_off_fp8(r, c, HD)] = K[i];
  }
  __syncthreads();
  transpose_store_sw128_src_inter(sKs, sKt, BN, HD, tid >> 5, tid & 31);  // [128][32]
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const uint32_t ba = smem_u32(sKt);
  float d[4][16];
#pragma unroll
  for (int nn = 0; nn < 4; ++nn)
#pragma unroll
    for (int i = 0; i < 16; ++i) d[nn][i] = 0.f;

  wgmma_fence();
  {
    const int koff = 0;  // K = BN = 32，单步
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int acol = (lane >> 4) * 16;
    uint32_t av[4];
    ldmatrix_x4(smem_u32(&sA[(wid * 16 + arow) * 48 + koff + acol]), av);
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      // Kᵀ [HD][BN] INTERLEAVE：第 nn 个 32 行 n-tile ⇒ 4 个 8 行组 ⇒ SBO=16。
      uint64_t db = make_desc_noswz(ba + (uint32_t)(nn * 4 * 16 * 16), 8 * 16, 16 * 16);
      wgmma_m64n32k32_rs_e5e4(d[nn], av, db);
    }
  }
  wgmma_commit();
  wgmma_wait0();

#pragma unroll
  for (int nn = 0; nn < 4; ++nn)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = wid * 16 + g + (q >= 2 ? 8 : 0);
        const int col = nn * 32 + j * 8 + c2 + (q & 1);
        Cout[row * HD + col] = d[nn][j * 4 + q];
      }
}

static unsigned char enc(float x, bool e5) {
  return (unsigned char)(e5 ? __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)
                            : __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

static void run_noswz() {
  constexpr int M = 64, N = 64, K = 32;
  std::vector<float> A(M * K), B(N * K);
  srand(7);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : A) x = rnd();
  for (auto& x : B) x = rnd();
  std::vector<float> Cref(M * N, 0.f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float a = 0.f;
      for (int k = 0; k < K; ++k) a += A[m * K + k] * B[n * K + k];
      Cref[m * N + n] = a;
    }
  std::vector<unsigned char> Ab(M * K), Bb(N * K);
  for (int i = 0; i < M * K; ++i) Ab[i] = enc(A[i], true);
  for (int i = 0; i < N * K; ++i) Bb[i] = enc(B[i], false);
  unsigned char *dA, *dB;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, Ab.size()));
  CUDA_CHECK(cudaMalloc(&dB, Bb.size()));
  CUDA_CHECK(cudaMalloc(&dC, M * N * 4));
  CUDA_CHECK(cudaMemcpy(dA, Ab.data(), Ab.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, Bb.data(), Bb.size(), cudaMemcpyHostToDevice));
  // 候选 (LBO,SBO)（单位 uint128=16B）：K=32 有 2 个 core ⇒ 自然编码 LBO=8、SBO=16。
  int lbos[] = {8, 16, 8, 2};
  int sbos[] = {16, 16, 32, 16};
  for (int ci = 0; ci < 4; ++ci) {
    CUDA_CHECK(cudaMemset(dC, 0, M * N * 4));
    noswz_kernel<<<1, 128>>>(dA, dB, lbos[ci], sbos[ci], dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> C(M * N);
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
    double e = 0;
    for (int i = 0; i < M * N; ++i) e = std::max(e, (double)std::fabs(C[i] - Cref[i]));
    printf("  no-swizzle (LBO_u=%d,SBO_u=%d): max_abs=%.3e %s\n", lbos[ci], sbos[ci],
           e, e < 1e-3 ? "PASS" : "fail");
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

static void run_sw128_transpose() {
  for (int R : {32, 64}) {
    const int C = 128;
    std::vector<unsigned char> X(R * C), Yref(R * C);
    for (int i = 0; i < R * C; ++i) X[i] = (unsigned char)((i * 131 + 7) & 0xff);
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) Yref[c * R + r] = X[r * C + c];
    unsigned char *dX, *dY;
    CUDA_CHECK(cudaMalloc(&dX, X.size()));
    CUDA_CHECK(cudaMalloc(&dY, X.size()));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
    const int smem = R * C + R * C;  // 源 SW128 + 目标（足够大）
    CUDA_CHECK(cudaFuncSetAttribute(sw128_transpose_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    sw128_transpose_kernel<<<1, 128, smem>>>(dX, dY, R, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> Y(X.size());
    CUDA_CHECK(cudaMemcpy(Y.data(), dY, Y.size(), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < Y.size(); ++i) bad += (Y[i] != Yref[i]);
    printf("  SW128-src 转置 [%d][%d]->[%d][%d]: mismatches=%d/%zu %s\n", R, C, C, R,
           bad, Y.size(), bad == 0 ? "PASS" : "FAIL");
    cudaFree(dX); cudaFree(dY);
  }
}

static void run_gemm5() {
  constexpr int BM = 64, BN = 32, HD = 128;
  std::vector<float> dS2(BM * BN), K(BN * HD);
  srand(11);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : dS2) x = rnd();
  for (auto& x : K) x = rnd();
  std::vector<float> Cref(BM * HD, 0.f);
  for (int m = 0; m < BM; ++m)
    for (int d = 0; d < HD; ++d) {
      float a = 0.f;
      for (int j = 0; j < BN; ++j) a += dS2[m * BN + j] * K[j * HD + d];
      Cref[m * HD + d] = a;
    }
  std::vector<unsigned char> Ab(BM * BN), Kb(BN * HD);
  for (int i = 0; i < BM * BN; ++i) Ab[i] = enc(dS2[i], true);
  for (int i = 0; i < BN * HD; ++i) Kb[i] = enc(K[i], false);
  unsigned char *dA, *dK;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, Ab.size()));
  CUDA_CHECK(cudaMalloc(&dK, Kb.size()));
  CUDA_CHECK(cudaMalloc(&dC, BM * HD * 4));
  CUDA_CHECK(cudaMemcpy(dA, Ab.data(), Ab.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, Kb.data(), Kb.size(), cudaMemcpyHostToDevice));
  const int smem = BM * 48 + (BN / 8) * 1024 + HD * BN;
  CUDA_CHECK(cudaFuncSetAttribute(gemm5_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  gemm5_kernel<<<1, 128, smem>>>(dA, dK, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> C(BM * HD);
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
  double e = 0;
  int bad = 0;
  for (int i = 0; i < BM * HD; ++i) {
    e = std::max(e, (double)std::fabs(C[i] - Cref[i]));
    if (std::fabs(C[i] - Cref[i]) > 1e-3) ++bad;
  }
  printf("  GEMM5 dQ[m][d]=Σ_j dS2[m][j]·K[j][d] (BM64 BN32 HD128, A ldmatrix, "
         "B=Kᵀ no-swizzle from SW128): max_abs=%.3e bad=%d %s\n",
         e, bad, bad == 0 ? "PASS" : "FAIL");
  cudaFree(dA); cudaFree(dK); cudaFree(dC);
}

int main() {
  printf("=== (A) no-swizzle K-major B 描述符（K=32） ===\n");
  run_noswz();
  printf("=== (B) 从 SW128 源逐字节转置 ===\n");
  run_sw128_transpose();
  printf("=== (C) 端到端 GEMM5（dQ） wgmma RS ===\n");
  run_gemm5();
  return 0;
}
