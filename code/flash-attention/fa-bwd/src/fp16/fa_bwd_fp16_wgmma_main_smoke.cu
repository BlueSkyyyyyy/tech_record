// =============================================================================
// fa_bwd_fp16_wgmma_main_smoke.cu —— O9b 前置：主 kernel 的 wgmma + SW128 桥接冒烟
// =============================================================================
// O9a 已在 LSE（单个 QKᵀ）上跑通 `wgmma.m64n64k16 + SW128`。O9b 要把 wgmma 推到主 kernel
// 的 5 个 GEMM，难点在于：GEMM1/2 可用 wgmma（A/B 都 K-major SS），但 GEMM3/4/5 的 B
// （dO/Q/K）在主 kernel 里是转置读（BTRANS），且这些张量同时要被 wgmma 当 K-major 操作数。
// 若把 Q/K/V/dO 都存成 **SW128 K-major**，则：
//   * wgmma 直接读（GEMM1/2）；
//   * GEMM3/4/5 的 B（BTRANS）用 `ldmatrix.x2.trans` + `sw128_off` 算出的地址从同一块
//     SW128 tile 里读（SW128 只在 16B 粒度做置换，ldmatrix 每个 lane 只要一个 16B 地址）。
// 本冒烟逐位验证这三件事：
//   1) wgmma QKᵀ（Q/K 存 SW128）；
//   2) GEMM3 形状：dV = Pᵀ·dO，A=P（ATRANS 读 [BM][BN]），B=dO 从 SW128 转置读；
//   3) GEMM5 形状：dQ = P·K， A=P（普通读 [BM][BN]），   B=K  从 SW128 转置读。
// 全部与 CPU 参考（fp32）比对。
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" scripts/run.sh ...
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

constexpr int M = 64, N = 64, K = 128;   // Q[64][128]·Kᵀ[128][64] → [64][64]
constexpr int THREADS = 128;

// ---------------- SW128 ----------------
__device__ __forceinline__ int sw128_off(int row, int k, int Kk) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int cc = (kk >> 3) ^ rr;
  return (rg * (Kk >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ void sw128_store16(char* tile, int row, int k0, int Kk,
                                              uint4 v) {
  *reinterpret_cast<uint4*>(tile + sw128_off(row, k0, Kk)) = v;
}
__device__ __forceinline__ uint32_t sw128_k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;
  d |= (uint64_t)1 << 62;
  return d;
}

// ---------------- wgmma ----------------
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_m64n64k16_f16(float (&d)[32], uint64_t da,
                                                    uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1));
}

// ---------------- mma / ldmatrix ----------------
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
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t addr, uint32_t d[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4],
                                        const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A=[M_TILE][K_TILE] row-major（行距 asld，half）。
//   ATRANS=true：As 存 [K_TILE][M_TILE] 的转置副本 → ldmatrix.x4.trans（bit3/4 互换）。
// B 为 SW128 K-major tile，逻辑布局 [row=token][hd]，行宽 BK（half）。BTRANS 读 [K=token][N=hd]。
template <int WARP_M, int WARP_N, int K_TILE, bool ATRANS, int BK>
__device__ __forceinline__ void mma_block_swb(const __half* As, int asld,
                                              const char* Bsw,
                                              float acc[WARP_M / 16][WARP_N / 8][4],
                                              int wm, int wn, int lane) {
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
#pragma unroll
  for (int kk = 0; kk < K_TILE / 16; ++kk) {
    const int koff = kk * 16;
    uint32_t av[MTM][4];
    if constexpr (ATRANS) {
      const int krow = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int mcol = ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        ldmatrix_x4_trans(smem_u32(As + (koff + krow) * asld +
                                   (wm * WARP_M + i * 16 + mcol)), av[i]);
    } else {
      const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int acol = (lane >> 4) * 8;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        ldmatrix_x4(smem_u32(As + (wm * WARP_M + i * 16 + arow) * asld + koff + acol),
                    av[i]);
    }
    uint32_t bv[MTN][2];
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      // BTRANS 从 SW128 读：lane 给 8 个 token 行各一个 16B（8 个相邻 hd）。
      const int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int ncol = wn * WARP_N + j * 8;
      uint32_t d[2];
      ldmatrix_x2_trans(smem_u32(Bsw + sw128_off(koff + krow, ncol, BK)), d);
      bv[j][0] = d[0];
      bv[j][1] = d[1];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) mma_f16(acc[i][j], av[i], bv[j]);
  }
}

__global__ void __launch_bounds__(THREADS) smoke_kernel(
    const __half* Q, const __half* Kt, const __half* V, const __half* dO,
    const __half* P, float* Sout, float* dVout, float* dQout) {
  // smem：Q/K/V/dO 各 SW128（(64/8)*(128/64)*1024=16KB），P row-major [64][64]（LDS=72）。
  __shared__ __align__(1024) char sQ[(M / 8) * (K / 64) * 1024];
  __shared__ __align__(1024) char sK[(N / 8) * (K / 64) * 1024];
  __shared__ __align__(1024) char sV[(N / 8) * (K / 64) * 1024];
  __shared__ __align__(1024) char sDO[(M / 8) * (K / 64) * 1024];
  constexpr int LDS = N + 8;
  __shared__ __half sP[M * LDS];
  const int tid = threadIdx.x;
  const int wn2 = K / 8;  // 每行 16B unit 数
  for (int i = tid; i < M * K / 8; i += THREADS) {
    const int row = i / wn2, k0 = (i % wn2) * 8;
    sw128_store16(sQ, row, k0, K, *reinterpret_cast<const uint4*>(Q + row * K + k0));
    sw128_store16(sDO, row, k0, K, *reinterpret_cast<const uint4*>(dO + row * K + k0));
  }
  for (int i = tid; i < N * K / 8; i += THREADS) {
    const int row = i / wn2, k0 = (i % wn2) * 8;
    sw128_store16(sK, row, k0, K, *reinterpret_cast<const uint4*>(Kt + row * K + k0));
    sw128_store16(sV, row, k0, K, *reinterpret_cast<const uint4*>(V + row * K + k0));
  }
  for (int i = tid; i < M * N; i += THREADS)
    sP[(i / N) * LDS + (i % N)] = P[i];
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int wr = wid / 2, wc = wid % 2;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // ---- (1) S = Q·Kᵀ via wgmma m64n64k16（整 CTA 一个 tile）----
  {
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.f;
    wgmma_fence();
    const uint32_t qa = smem_u32(sQ), ka = smem_u32(sK);
    const uint32_t sbo = (uint32_t)((K / 64) * 1024);
#pragma unroll
    for (int s = 0; s < K / 16; ++s) {
      wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(qa, s), sbo),
                          make_desc_sw128(sw128_k16_addr(ka, s), sbo));
    }
    wgmma_commit();
    wgmma_wait0();
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = wid * 16 + g + (q >= 2 ? 8 : 0);
        const int col = j * 8 + c2 + (q & 1);
        Sout[row * N + col] = d[j * 4 + q];
      }
  }

  // ---- (2) dV = Pᵀ·dO：A=P (ATRANS, [64][64])，B=dO 从 SW128；2×2 warp，GMV=32,GNV=64 ----
  {
    constexpr int GMV = 32, GNV = 64;
    float acc[GMV / 16][GNV / 8][4];
#pragma unroll
    for (int i = 0; i < GMV / 16; ++i)
#pragma unroll
      for (int j = 0; j < GNV / 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    mma_block_swb<GMV, GNV, M, true, K>(sP, LDS, sDO, acc, wr, wc, lane);
#pragma unroll
    for (int i = 0; i < GMV / 16; ++i)
#pragma unroll
      for (int j = 0; j < GNV / 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int row = wr * GMV + i * 16 + g + (q >= 2 ? 8 : 0);
          const int col = wc * GNV + j * 8 + c2 + (q & 1);
          dVout[row * K + col] = acc[i][j][q];
        }
  }

  // ---- (3) dQ = P·K：A=P (普通读 [64][64])，B=K 从 SW128；输出 [64][128]，GMQ=32,GNQ=64 ----
  {
    constexpr int GMQ = 32, GNQ = 64;
    float acc[GMQ / 16][GNQ / 8][4];
#pragma unroll
    for (int i = 0; i < GMQ / 16; ++i)
#pragma unroll
      for (int j = 0; j < GNQ / 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    mma_block_swb<GMQ, GNQ, N, false, K>(sP, LDS, sK, acc, wr, wc, lane);
#pragma unroll
    for (int i = 0; i < GMQ / 16; ++i)
#pragma unroll
      for (int j = 0; j < GNQ / 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int row = wr * GMQ + i * 16 + g + (q >= 2 ? 8 : 0);
          const int col = wc * GNQ + j * 8 + c2 + (q & 1);
          dQout[row * K + col] = acc[i][j][q];
        }
  }
}

int main() {
  std::vector<float> Q(M * K), Kt(N * K), V(N * K), dO(M * K), P(M * N);
  std::vector<float> Sref(M * N), dVref(M * K), dQref(M * K);
  srand(123);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : Q) x = rnd();
  for (auto& x : Kt) x = rnd();
  for (auto& x : V) x = rnd();
  for (auto& x : dO) x = rnd();
  for (auto& x : P) x = (float)((int)(rand() % 5));  // 非负，模拟 P

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float a = 0;
      for (int k = 0; k < K; ++k) a += Q[m * K + k] * Kt[n * K + k];
      Sref[m * N + n] = a;
    }
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      float b = 0;
      for (int m = 0; m < M; ++m) b += P[m * N + n] * dO[m * K + k];  // dV[n][k]
      dVref[n * K + k] = b;
    }
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      float c = 0;
      for (int n = 0; n < N; ++n) c += P[m * N + n] * Kt[n * K + k];  // dQ[m][k]
      dQref[m * K + k] = c;
    }

  std::vector<__half> hQ(M * K), hK(N * K), hV(N * K), hdO(M * K), hP(M * N);
  for (int i = 0; i < M * K; ++i) { hQ[i] = __float2half(Q[i]); hdO[i] = __float2half(dO[i]); }
  for (int i = 0; i < N * K; ++i) { hK[i] = __float2half(Kt[i]); hV[i] = __float2half(V[i]); }
  for (int i = 0; i < M * N; ++i) hP[i] = __float2half(P[i]);

  __half *dQ_, *dK_, *dV_, *dDO_, *dP_;
  float *dS, *ddV, *ddQ;
  CUDA_CHECK(cudaMalloc(&dQ_, hQ.size() * 2));
  CUDA_CHECK(cudaMalloc(&dK_, hK.size() * 2));
  CUDA_CHECK(cudaMalloc(&dV_, hV.size() * 2));
  CUDA_CHECK(cudaMalloc(&dDO_, hdO.size() * 2));
  CUDA_CHECK(cudaMalloc(&dP_, hP.size() * 2));
  CUDA_CHECK(cudaMalloc(&dS, M * N * 4));
  CUDA_CHECK(cudaMalloc(&ddV, M * K * 4));
  CUDA_CHECK(cudaMalloc(&ddQ, M * K * 4));
  CUDA_CHECK(cudaMemcpy(dQ_, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK_, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV_, hV.data(), hV.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDO_, hdO.data(), hdO.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dP_, hP.data(), hP.size() * 2, cudaMemcpyHostToDevice));
  smoke_kernel<<<1, THREADS>>>(dQ_, dK_, dV_, dDO_, dP_, dS, ddV, ddQ);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> S(M * N), dV(M * K), dQ(M * K);
  CUDA_CHECK(cudaMemcpy(S.data(), dS, M * N * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dV.data(), ddV, M * K * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dQ.data(), ddQ, M * K * 4, cudaMemcpyDeviceToHost));

  auto maxerr = [](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, (double)std::fabs(a[i] - b[i]));
    return e;
  };
  const double eS = maxerr(S, Sref), eDV = maxerr(dV, dVref), eDQ = maxerr(dQ, dQref);
  printf("wgmma QKᵀ     vs CPU: max_abs=%.3e\n", eS);
  printf("mma dV=PᵀdO   vs CPU: max_abs=%.3e  (B from SW128)\n", eDV);
  printf("mma dQ=P·K    vs CPU: max_abs=%.3e  (B from SW128)\n", eDQ);
  bool ok = eS < 1e-2 && eDV < 1e-2 && eDQ < 1e-2;
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dQ_); cudaFree(dK_); cudaFree(dV_); cudaFree(dDO_); cudaFree(dP_);
  cudaFree(dS); cudaFree(ddV); cudaFree(ddQ);
  return ok ? 0 : 1;
}
