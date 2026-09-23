// =============================================================================
// fa_bwd_fp16_wgmma_bwd_smoke.cu —— O9b-2 前置：GEMM3/4/5 的「转置读」wgmma 布局冒烟
// =============================================================================
// O9b 已把主 kernel 的 GEMM1/2（S=QKᵀ、dP=dO·Vᵀ）换成 `wgmma.m64n64k16`。O9b-2 要把
// GEMM3/4/5（dV=PᵀdO、dK=dSᵀQ、dQ=dS·K）也上 wgmma。它们的 A/B 在 1colblock 数据流里是
// **转置读**，而 Q/K/V/dO/P/dS 又同时要被 wgmma 当 K-major 操作数。FA3 的做法是
// `dKV_swapAB`：**同一份 K-major SW128 tile，用 `Major::MN` 描述符去读 = 读它的转置**。
//
// 本冒烟逐位验证这套「K-major 存储 + MN-major 描述符 + trans=1」的三件事：
//   (1) S  = Q·Kᵀ      ：A=Q K-major，B=K K-major（对照 O9b）；
//   (2) dV = Pᵀ·dO     ：A=P 转置（MN），B=dO 转置（MN），输出 [BN][HD]；
//   (3) dK = dSᵀ·Q     ：A=dS 转置（MN），B=Q 转置（MN），输出 [BN][HD]；
//   (4) dQ = dS·K      ：A=dS K-major，B=K 转置（MN），输出 [BM][HD]。
// 全部与 CPU fp32 参考比对。
//
// MN-major 描述符（对 `sw128_off` 的 K-major SW128 tile，宽 W 元素）：
//   LBO=64（相邻 64 列组的 u128 步长），SBO=(W/64)*64（相邻 8 行组的 u128 步长）；
//   trans 操作数的第 s 个 k16 slab 地址 = base + s * 2*SBO*16 字节（K=行，每次前进 16 行=2 行组）。
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

constexpr int BM = 64, BN = 64, HD = 128, THREADS = 128;

// ---------------- SW128（与主 kernel 逐字一致）----------------
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int cc = (kk >> 3) ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ void sw128_store16(char* tile, int row, int k0, int K,
                                              uint4 v) {
  *reinterpret_cast<uint4*>(tile + sw128_off(row, k0, K)) = v;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint32_t sw128_k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
// K-major 描述符（LBO=1，SBO=(W/64)*64），地址给定为**字节**值。
__device__ __forceinline__ uint64_t make_desc_k(uint32_t addr, uint32_t W) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((((W >> 6) * 64)) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // B128
  return d;
}
// MN-major（转置）描述符：LBO=64，SBO=(W/64)*64，地址给定为字节值。
__device__ __forceinline__ uint64_t make_desc_mn(uint32_t addr, uint32_t W) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)(64u & 0x3FFF) << 16;  // LBO
  d |= (uint64_t)((((W >> 6) * 64)) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // B128
  return d;
}
// 转置操作数第 s 个 k16 slab 的字节地址（K=行，前进 16 行）。
__device__ __forceinline__ uint32_t trans_k16_addr(uint32_t base, int s, uint32_t W) {
  return base + (uint32_t)(s * 2 * ((W >> 6) * 64) * 16);
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
// tnspA/tnspB: 0=K-major(0), 1=MN-major。用模板立即数。
template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n64k16_f16(float (&d)[32], uint64_t da,
                                                    uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, 1, 1, %35, %36;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1), "n"(TA), "n"(TB));
}

__global__ void __launch_bounds__(THREADS) smoke_kernel(
    const __half* Q, const __half* Kt, const __half* V, const __half* dO,
    const __half* P, const __half* dS, float* Sout, float* dVout, float* dKout,
    float* dQout) {
  __shared__ __align__(1024) char sQ[BM * HD * 2];
  __shared__ __align__(1024) char sK[BN * HD * 2];
  __shared__ __align__(1024) char sV[BN * HD * 2];
  __shared__ __align__(1024) char sDO[BM * HD * 2];
  __shared__ __align__(1024) char sP[BM * BN * 2];
  __shared__ __align__(1024) char sDS[BM * BN * 2];
  const int tid = threadIdx.x;
  const int hdU = HD / 8, bnU = BN / 8;
  for (int i = tid; i < BM * hdU; i += THREADS) {
    const int row = i / hdU, k0 = (i % hdU) * 8;
    sw128_store16(sQ, row, k0, HD, *reinterpret_cast<const uint4*>(Q + row * HD + k0));
    sw128_store16(sDO, row, k0, HD, *reinterpret_cast<const uint4*>(dO + row * HD + k0));
  }
  for (int i = tid; i < BN * hdU; i += THREADS) {
    const int row = i / hdU, k0 = (i % hdU) * 8;
    sw128_store16(sK, row, k0, HD, *reinterpret_cast<const uint4*>(Kt + row * HD + k0));
    sw128_store16(sV, row, k0, HD, *reinterpret_cast<const uint4*>(V + row * HD + k0));
  }
  for (int i = tid; i < BM * bnU; i += THREADS) {
    const int row = i / bnU, k0 = (i % bnU) * 8;
    sw128_store16(sP, row, k0, BN, *reinterpret_cast<const uint4*>(P + row * BN + k0));
    sw128_store16(sDS, row, k0, BN, *reinterpret_cast<const uint4*>(dS + row * BN + k0));
  }
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  auto store_out = [&](float* dst, int N, int nhalf, const float (&d)[32]) {
    for (int j = 0; j < 8; ++j)
      for (int q = 0; q < 4; ++q) {
        const int row = wid * 16 + g + (q >= 2 ? 8 : 0);
        const int col = nhalf * 64 + j * 8 + c2 + (q & 1);
        if (row < 64 && col < N) dst[row * N + col] = d[j * 4 + q];
      }
  };

  // (1) S = Q·Kᵀ（K-major × K-major）
  {
    float d[32]; for (int i = 0; i < 32; ++i) d[i] = 0.f;
    wgmma_fence();
    const uint32_t qa = smem_u32(sQ), ka = smem_u32(sK);
    for (int s = 0; s < HD / 16; ++s)
      wgmma_m64n64k16_f16<0, 0>(d, make_desc_k(sw128_k16_addr(qa, s), HD),
                                make_desc_k(sw128_k16_addr(ka, s), HD));
    wgmma_commit(); wgmma_wait0();
    store_out(Sout, BN, 0, d);
  }

  // (2) dV = Pᵀ·dO：A=P 转置(MN,W=BN)，B=dO 转置(MN,W=HD)，K=BM；输出 [BN][HD] 两个 n64
  {
    const uint32_t pa = smem_u32(sP), doa = smem_u32(sDO);
    for (int nh = 0; nh < 2; ++nh) {
      float d[32]; for (int i = 0; i < 32; ++i) d[i] = 0.f;
      wgmma_fence();
      const uint32_t doa_n = doa + (uint32_t)(nh * 1024);  // dO 的 N 半（kg=nh）
      for (int s = 0; s < BM / 16; ++s)
        wgmma_m64n64k16_f16<1, 1>(d, make_desc_mn(trans_k16_addr(pa, s, BN), BN),
                                  make_desc_mn(trans_k16_addr(doa_n, s, HD), HD));
      wgmma_commit(); wgmma_wait0();
      store_out(dVout, HD, nh, d);
    }
  }

  // (3) dK = dSᵀ·Q：A=dS 转置(MN,W=BN)，B=Q 转置(MN,W=HD)，K=BM；输出 [BN][HD]
  {
    const uint32_t dsa = smem_u32(sDS), qa = smem_u32(sQ);
    for (int nh = 0; nh < 2; ++nh) {
      float d[32]; for (int i = 0; i < 32; ++i) d[i] = 0.f;
      wgmma_fence();
      const uint32_t qa_n = qa + (uint32_t)(nh * 1024);
      for (int s = 0; s < BM / 16; ++s)
        wgmma_m64n64k16_f16<1, 1>(d, make_desc_mn(trans_k16_addr(dsa, s, BN), BN),
                                  make_desc_mn(trans_k16_addr(qa_n, s, HD), HD));
      wgmma_commit(); wgmma_wait0();
      store_out(dKout, HD, nh, d);
    }
  }

  // (4) dQ = dS·K：A=dS K-major(W=BN)，B=K 转置(MN,W=HD)，K=BN；输出 [BM][HD]
  {
    const uint32_t dsa = smem_u32(sDS), ka = smem_u32(sK);
    for (int nh = 0; nh < 2; ++nh) {
      float d[32]; for (int i = 0; i < 32; ++i) d[i] = 0.f;
      wgmma_fence();
      const uint32_t ka_n = ka + (uint32_t)(nh * 1024);
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n64k16_f16<0, 1>(d, make_desc_k(sw128_k16_addr(dsa, s), BN),
                                  make_desc_mn(trans_k16_addr(ka_n, s, HD), HD));
      wgmma_commit(); wgmma_wait0();
      store_out(dQout, HD, nh, d);
    }
  }
}

int main() {
  std::vector<float> Q(BM * HD), Kt(BN * HD), V(BN * HD), dO(BM * HD);
  std::vector<float> P(BM * BN), dS(BM * BN);
  srand(123);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : Q) x = rnd();
  for (auto& x : Kt) x = rnd();
  for (auto& x : V) x = rnd();
  for (auto& x : dO) x = rnd();
  for (auto& x : P) x = (float)((int)(rand() % 5));
  for (auto& x : dS) x = (float)((int)(rand() % 5) - 2);

  // refs: dV[BN][HD], dK[BN][HD], dQ[BM][HD]
  std::vector<float> Sref(BM * BN), dVref(BN * HD), dKref(BN * HD), dQref(BM * HD);
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < BN; ++n) {
      float a = 0; for (int k = 0; k < HD; ++k) a += Q[m*HD+k] * Kt[n*HD+k];
      Sref[m*BN+n] = a;
    }
  for (int n = 0; n < BN; ++n)
    for (int k = 0; k < HD; ++k) {
      float b = 0; for (int m = 0; m < BM; ++m) b += P[m*BN+n] * dO[m*HD+k];
      dVref[n*HD+k] = b;
    }
  for (int n = 0; n < BN; ++n)
    for (int k = 0; k < HD; ++k) {
      float b = 0; for (int m = 0; m < BM; ++m) b += dS[m*BN+n] * Q[m*HD+k];
      dKref[n*HD+k] = b;
    }
  for (int m = 0; m < BM; ++m)
    for (int k = 0; k < HD; ++k) {
      float c = 0; for (int n = 0; n < BN; ++n) c += dS[m*BN+n] * Kt[n*HD+k];
      dQref[m*HD+k] = c;
    }

  std::vector<__half> hQ(BM*HD), hK(BN*HD), hV(BN*HD), hDO(BM*HD), hP(BM*BN), hDS(BM*BN);
  for (int i = 0; i < BM*HD; ++i) { hQ[i] = __float2half(Q[i]); hDO[i] = __float2half(dO[i]); }
  for (int i = 0; i < BN*HD; ++i) { hK[i] = __float2half(Kt[i]); hV[i] = __float2half(V[i]); }
  for (int i = 0; i < BM*BN; ++i) { hP[i] = __float2half(P[i]); hDS[i] = __float2half(dS[i]); }

  __half *dQ_, *dK_, *dV_, *dDO_, *dP_, *dDS_;
  float *dS_, *ddV, *ddK, *ddQ;
  CUDA_CHECK(cudaMalloc(&dQ_, hQ.size()*2)); CUDA_CHECK(cudaMalloc(&dK_, hK.size()*2));
  CUDA_CHECK(cudaMalloc(&dV_, hV.size()*2)); CUDA_CHECK(cudaMalloc(&dDO_, hDO.size()*2));
  CUDA_CHECK(cudaMalloc(&dP_, hP.size()*2)); CUDA_CHECK(cudaMalloc(&dDS_, hDS.size()*2));
  CUDA_CHECK(cudaMalloc(&dS_, BM*BN*4)); CUDA_CHECK(cudaMalloc(&ddV, BN*HD*4));
  CUDA_CHECK(cudaMalloc(&ddK, BN*HD*4)); CUDA_CHECK(cudaMalloc(&ddQ, BM*HD*4));
  CUDA_CHECK(cudaMemcpy(dQ_, hQ.data(), hQ.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK_, hK.data(), hK.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV_, hV.data(), hV.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDO_, hDO.data(), hDO.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dP_, hP.data(), hP.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDS_, hDS.data(), hDS.size()*2, cudaMemcpyHostToDevice));
  smoke_kernel<<<1, THREADS>>>(dQ_, dK_, dV_, dDO_, dP_, dDS_, dS_, ddV, ddK, ddQ);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> S(BM*BN), dV(BN*HD), dK(BN*HD), dQ(BM*HD);
  CUDA_CHECK(cudaMemcpy(S.data(), dS_, S.size()*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dV.data(), ddV, dV.size()*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dK.data(), ddK, dK.size()*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dQ.data(), ddQ, dQ.size()*4, cudaMemcpyDeviceToHost));

  auto maxerr = [](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0; for (size_t i = 0; i < a.size(); ++i) e = std::max(e, (double)std::fabs(a[i]-b[i])); return e;
  };
  const double eS = maxerr(S, Sref), eDV = maxerr(dV, dVref), eDK = maxerr(dK, dKref), eDQ = maxerr(dQ, dQref);
  printf("wgmma S=QKᵀ       vs CPU: max_abs=%.3e\n", eS);
  printf("wgmma dV=PᵀdO(trans) vs CPU: max_abs=%.3e\n", eDV);
  printf("wgmma dK=dSᵀQ(trans) vs CPU: max_abs=%.3e\n", eDK);
  printf("wgmma dQ=dS·K(B trans) vs CPU: max_abs=%.3e\n", eDQ);
  bool ok = eS < 1e-2 && eDV < 1e-2 && eDK < 1e-2 && eDQ < 1e-2;
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dQ_); cudaFree(dK_); cudaFree(dV_); cudaFree(dDO_); cudaFree(dP_); cudaFree(dDS_);
  cudaFree(dS_); cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
  return ok ? 0 : 1;
}
