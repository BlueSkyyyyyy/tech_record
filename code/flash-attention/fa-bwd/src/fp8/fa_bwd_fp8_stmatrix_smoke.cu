// =============================================================================
// fa_bwd_fp8_stmatrix_smoke.cu —— F3b 主体第一步：用 `ldmatrix.x4.trans` + `PRMT`
//   把行主序 fp8 tile **逐字节转置**成 wgmma 可消费的 K-major 操作数
// =============================================================================
// 背景（第 174 轮 O79 + 第 175 轮 TE SASS 复核）：
//   * O79 已证 `ldmatrix.x4`（行主序 K-major [64][K]）取回的 4×u32 恰是
//     `wgmma.m64n32k32` RS_TN 的 A 片段（ALayout_64x32）⇒ A（寄存器）可解。
//   * 但 GEMM3/4/5 的 **B 操作数**（dOᵀ/Qᵀ/Kᵀ）需要逐字节转置（fp8 的
//     `ldmatrix.trans` 只交换 8×8 b16 的配对方向，不是字节转置，见 O4b）。
//   * 本轮用 `ncu --page source --print-source sass` 复核 TE 反向 kernel，
//     它真正的做法是：
//         LDSM.16.MT88.4  (ldmatrix.x4.trans)
//         PRMT ...        (逐字节修正 b16 配对方向)
//         STSM.16.M88.4   (stmatrix.x4 落盘)
//     即「转置读 → 字节重排 → 矩阵写」造出 wgmma 可消费的操作数。
//
// 本冒烟只钉死最不确定的一步：**逐字节转置的 lane 映射**。
//   源 X[R][C]（fp8 行主序，C 连续）。目标 Y[C][R]（逐字节转置，R 连续 = K-major）。
//   取一个 8 行 × 64 fp8 的块做 `ldmatrix.x4.trans`（4 个 8×8 b16 矩阵并排），
//   推导/验证每个 lane 拿到的 4×u32，再用 `__byte_perm(reg, reg>>16, 0x5410)` 修正，
//   最后以两个 16-bit 存写回 Y。与 CPU 逐字节参考比。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//         scripts/run.sh src/fp8/fa_bwd_fp8_stmatrix_smoke.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
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
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(addr));
}

// -----------------------------------------------------------------------------
// 逐字节转置一个 [R][C] fp8 tile（R、C 均为 8 的倍数）：
//   Y[c][r] = X[r][c]，Y 行主序（R 连续，即 K=R 的 K-major 布局）。
//   线程数 = 128（4 warp）。每个 warp 负责若干 8 行块（R 方向）。
//   每个 8 行块的 C 方向用 4 个 8×8 b16 矩阵的 `ldmatrix.x4.trans` 一次覆盖
//   C = 4×16 = 64 fp8 列；C > 64 时外层再沿 C 分块。
// -----------------------------------------------------------------------------
__global__ void byte_transpose_kernel(const unsigned char* __restrict__ X,
                                      unsigned char* __restrict__ Y, int R,
                                      int C) {
  extern __shared__ unsigned char smem[];
  unsigned char* sX = smem;               // [R][C]
  const int T = blockDim.x;
  const int tid = threadIdx.x;
  for (int i = tid; i < R * C; i += T) sX[i] = X[i];
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int nblkR = R / 8;                // 8 行块数
  const int nblkC = C / 64;               // 64 列块数（每个 x4.trans 覆盖 64 fp8）
  // 每个 warp 领 (rblk, cblk) 工作：块 id 线性铺开、warp-stride。
  const int nwork = nblkR * nblkC;
  for (int w = wid; w < nwork; w += 4) {
    const int rblk = w / nblkC, cblk = w % nblkC;
    const int r0 = rblk * 8, c0 = cblk * 64;
    // ldmatrix.x4.trans：matrix i=L>>3 覆盖 fp8 列 [c0+16i, c0+16i+16)；行 = r0+(L&7)。
    const int mat = lane >> 3, row = lane & 7;
    uint32_t reg[4];
    ldmatrix_x4_trans(smem_u32(sX + (r0 + row) * C + c0 + mat * 16), reg);
    // 每个 reg 的推导（见文件头注释）：
    //   reg_i 低 16 = Xb16[2p][8i+(L>>2)]，高 16 = Xb16[2p+1][8i+(L>>2)]，p=L&3。
    //   PRMT 0x5410 得到 (X[2p][2k], X[2p+1][2k]) | (X[2p][2k+1], X[2p+1][2k+1])
    //            = (Y[2k][r0+2p], Y[2k][r0+2p+1]) | (Y[2k+1][r0+2p], Y[2k+1][r0+2p+1])
    const int p = lane & 3;
    const int kbase = lane >> 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 8 * i + kbase;        // b16 列（全局 d-pair 索引）
      const uint32_t word = __byte_perm(reg[i], reg[i] >> 16, 0x5140);
      const uint16_t lo = (uint16_t)(word & 0xffffu);       // Y[2k][r0+2p..+2]
      const uint16_t hi = (uint16_t)((word >> 16) & 0xffffu);  // Y[2k+1][r0+2p..+2]
      const int d0 = c0 + 2 * k, d1 = d0 + 1;   // 加本列块基址（cblk>0 时 d 从 c0 起）
      const int r = r0 + 2 * p;
      if (d0 < C) *reinterpret_cast<uint16_t*>(Y + (size_t)d0 * R + r) = lo;
      if (d1 < C) *reinterpret_cast<uint16_t*>(Y + (size_t)d1 * R + r) = hi;
    }
  }
}

// =============================================================================
// 第二阶段：用上面的逐字节转置造出 GEMM3 的操作数，跑 **wgmma RS**：
//     C[j][d] = Σ_m P[m][j] · dO[m][d]        （dV 的形状）
//   A = Pᵀ [BN=64][BM=128] 行主序 K-major（m 连续）→ `ldmatrix.x4` 装进寄存器（RS 的 A）。
//   B = dOᵀ [HD=64][BM=128] K-major，且**逐字节转置后直接写进 SW128 tile**（wgmma 的 B）。
//   wgmma.mma_async.m64n32k32（4 个 k=32 步、2 个 n=32 块）逐元素与 CPU 比。
// =============================================================================
__device__ __forceinline__ int sw128_off_fp8(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
__device__ __forceinline__ uint64_t make_desc_sw128_fp8(uint32_t addr,
                                                        uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;  // LBO = 1（K-major 恒 1）
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
__device__ __forceinline__ uint32_t sw128_k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
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
// D[16] += A(4×u32, 寄存器) · B(描述符)，fp8 e4m3(A) × e5m2(B)，m64n32k32。
__device__ __forceinline__ void wgmma_m64n32k32_rs_e4e5(float (&d)[16],
                                                        const uint32_t a[4],
                                                        uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e5m2 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "{%16,%17,%18,%19}, %20, p, %22, %23;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1), "n"(1),
        "n"(1));
}

// 把一个 [R][C] 源块用 `ldmatrix.x4.trans`+PRMT 重新落盘：
//   * PLAIN=false：写行主序 Y[C][R]（K-major，供 ldmatrix.x4 读 A）；
//   * PLAIN=true ：写 K=R、行主序的 SW128 tile Y[C][R]（供 wgmma B 描述符）。
// 每 warp 领 (rblk,cblk)。SRC_STRIDE 是源 tile 行距（= C）。
template <bool SW128>
__device__ __forceinline__ void transpose_store(const unsigned char* sSrc,
                                                unsigned char* sDst, int R, int C,
                                                int wid, int lane) {
  const int nblkC = C / 64;
  const int nwork = (R / 8) * nblkC;
  for (int w = wid; w < nwork; w += 4) {
    const int rblk = w / nblkC, cblk = w % nblkC;
    const int r0 = rblk * 8, c0 = cblk * 64;
    const int mat = lane >> 3, row = lane & 7;
    uint32_t reg[4];
    ldmatrix_x4_trans(smem_u32(sSrc + (r0 + row) * C + c0 + mat * 16), reg);
    const int p = lane & 3, q = lane >> 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 8 * i + q;
      const uint32_t word = __byte_perm(reg[i], reg[i] >> 16, 0x5140);
      const int d0 = c0 + 2 * k, d1 = d0 + 1;
      const int m = r0 + 2 * p;
      if (SW128) {
        *reinterpret_cast<uint16_t*>(sDst + sw128_off_fp8(d0, m, R)) =
            (uint16_t)(word & 0xffffu);
        *reinterpret_cast<uint16_t*>(sDst + sw128_off_fp8(d1, m, R)) =
            (uint16_t)(word >> 16);
      } else {
        *reinterpret_cast<uint16_t*>(sDst + (size_t)d0 * R + m) =
            (uint16_t)(word & 0xffffu);
        *reinterpret_cast<uint16_t*>(sDst + (size_t)d1 * R + m) =
            (uint16_t)(word >> 16);
      }
    }
  }
}

// BM=128（K）、BN=64（M）、HD=64（N）。sP/sdO 为 [128][64] 行主序源。
__global__ void gemm3_wgmma_kernel(const unsigned char* __restrict__ P,
                                   const unsigned char* __restrict__ dO,
                                   float* __restrict__ Cout) {
  constexpr int BM = 128, BN = 64, HD = 64;
  extern __shared__ unsigned char smem[];
  unsigned char* sP = smem;                       // [128][64]
  unsigned char* sdO = smem + BM * HD;            // [128][64]
  unsigned char* sA = sdO + BM * HD;              // [64][128] Pᵀ 行主序
  unsigned char* sB = sA + BN * BM;               // [64][128] dOᵀ SW128
  float* sC = reinterpret_cast<float*>(sB + BN * BM);  // [64][64]
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  for (int i = tid; i < BM * HD; i += 128) {
    sP[i] = P[i];
    sdO[i] = dO[i];
  }
  __syncthreads();
  transpose_store<false>(sP, sA, BM, HD, wid, lane);
  transpose_store<true>(sdO, sB, BM, HD, wid, lane);
  __syncthreads();

  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const uint32_t sbo = (uint32_t)((BM / 128) * 1024);
  const uint32_t ba = smem_u32(sB);
  float acc[2][16];
#pragma unroll
  for (int nn = 0; nn < 2; ++nn)
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[nn][i] = 0.f;

  wgmma_fence();
#pragma unroll
  for (int s = 0; s < BM / 32; ++s) {           // 4 个 k=32 步
    const int koff = s * 32;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int acol = (lane >> 4) * 16;
    uint32_t av[4];
    // A = Pᵀ 已是行主序 K-major（m 连续），用**非转置** ldmatrix.x4 取 A 片段。
    ldmatrix_x4(smem_u32(sA + (wid * 16 + arow) * BM + koff + acol), av);
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      uint64_t db = make_desc_sw128_fp8(
          sw128_k32_addr(ba + (uint32_t)(nn * 32 / 8) * 1024, s), sbo);
      wgmma_m64n32k32_rs_e4e5(acc[nn], av, db);
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
        const int row = g + (q >= 2 ? 8 : 0);       // 本 warp 的 16 行（wid*16 + ...）
        const int col = nn * 32 + j * 8 + c2 + (q & 1);
        sC[(wid * 16 + row) * HD + col] = acc[nn][j * 4 + q];
      }
  __syncthreads();
  for (int i = tid; i < BN * HD; i += 128) Cout[i] = sC[i];
}

// 参考：把 [R][C] uint8 在 CPU 上逐字节转置成 [C][R]。
static void transpose_ref(const std::vector<unsigned char>& X,
                          std::vector<unsigned char>& Y, int R, int C) {
  Y.assign(size_t(R) * C, 0);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) Y[size_t(c) * R + r] = X[size_t(r) * C + c];
}

static int run_case(int R, int C) {
  std::vector<unsigned char> X(size_t(R) * C);
  for (size_t i = 0; i < X.size(); ++i) X[i] = (unsigned char)((i * 131 + 7) & 0xff);
  std::vector<unsigned char> Yref;
  transpose_ref(X, Yref, R, C);

  unsigned char *dX, *dY;
  CUDA_CHECK(cudaMalloc(&dX, X.size()));
  CUDA_CHECK(cudaMalloc(&dY, Yref.size()));
  CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dY, 0, Yref.size()));
  const int smem = R * C;
  CUDA_CHECK(cudaFuncSetAttribute(byte_transpose_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  byte_transpose_kernel<<<1, 128, smem>>>(dX, dY, R, C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned char> Y(Yref.size());
  CUDA_CHECK(cudaMemcpy(Y.data(), dY, Y.size(), cudaMemcpyDeviceToHost));

  int bad = 0, first = -1;
  for (size_t i = 0; i < Y.size(); ++i)
    if (Y[i] != Yref[i]) {
      if (first < 0) first = (int)i;
      ++bad;
    }
  printf("byte_transpose [%d][%d] -> [%d][%d]: mismatches=%d/%zu",
         R, C, C, R, bad, Y.size());
  if (bad) {
    int d = first / R, r = first % R;
    printf("  first@Y[%d][%d] got=%02x want=%02x", d, r, Y[first], Yref[first]);
  }
  printf("  %s\n", bad == 0 ? "PASS" : "FAIL");
  cudaFree(dX);
  cudaFree(dY);
  return bad == 0 ? 0 : 1;
}

static unsigned char enc(float x, bool e5) {
  return (unsigned char)(e5 ? __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)
                            : __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// 第二阶段：wgmma RS 的 GEMM3（dV 形状）。
static int run_gemm3() {
  constexpr int BM = 128, BN = 64, HD = 64;
  std::vector<float> P(BM * BN), dO(BM * HD);
  srand(2024);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : P) x = rnd();
  for (auto& x : dO) x = rnd();
  std::vector<float> Cref(BN * HD, 0.f);
  for (int j = 0; j < BN; ++j)
    for (int d = 0; d < HD; ++d) {
      float a = 0.f;
      for (int m = 0; m < BM; ++m) a += P[m * BN + j] * dO[m * HD + d];
      Cref[j * HD + d] = a;
    }
  std::vector<unsigned char> Pb(BM * BN), dOb(BM * HD);
  for (int i = 0; i < BM * BN; ++i) Pb[i] = enc(P[i], false);
  for (int i = 0; i < BM * HD; ++i) dOb[i] = enc(dO[i], true);

  unsigned char *dP, *dO8;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dP, Pb.size()));
  CUDA_CHECK(cudaMalloc(&dO8, dOb.size()));
  CUDA_CHECK(cudaMalloc(&dC, BN * HD * 4));
  CUDA_CHECK(cudaMemcpy(dP, Pb.data(), Pb.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dO8, dOb.data(), dOb.size(), cudaMemcpyHostToDevice));
  const int smem = BM * HD * 2 + BN * BM * 2 + BN * HD * 4;
  CUDA_CHECK(cudaFuncSetAttribute(gemm3_wgmma_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  gemm3_wgmma_kernel<<<1, 128, smem>>>(dP, dO8, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> C(BN * HD);
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
  double e = 0;
  int bad = 0;
  for (int i = 0; i < BN * HD; ++i) {
    e = std::max(e, (double)std::fabs(C[i] - Cref[i]));
    if (std::fabs(C[i] - Cref[i]) > 1e-3) ++bad;
  }
  printf("wgmma RS GEMM3 dV[j][d]=Σ_m P[m][j]·dO[m][d] (BM128 BN64 HD64, "
         "A=Pᵀ ldmatrix, B=dOᵀ SW128字节转置): max_abs=%.3e bad=%d  %s\n",
         e, bad, bad == 0 ? "PASS" : "FAIL");
  cudaFree(dP);
  cudaFree(dO8);
  cudaFree(dC);
  return bad == 0 ? 0 : 1;
}

int main() {
  printf("=== fp8 逐字节转置：ldmatrix.x4.trans + PRMT ===\n");
  int rc = 0;
  rc |= run_case(128, 64);   // GEMM3 的 dO[BM=128][HD=64] -> [64][128]
  rc |= run_case(64, 64);    // 方阵
  rc |= run_case(64, 128);   // C>64（多列块路径）
  rc |= run_case(32, 128);   // R=32 的最小行块
  printf("=== 第二阶段：wgmma RS GEMM3（转置操作数直喂 wgmma） ===\n");
  rc |= run_gemm3();
  printf("=== %s ===\n", rc == 0 ? "ALL PASS" : "FAIL");
  return rc;
}
