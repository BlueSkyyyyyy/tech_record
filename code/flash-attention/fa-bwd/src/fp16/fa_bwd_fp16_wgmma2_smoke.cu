// =============================================================================
// fa_bwd_fp16_wgmma2_smoke.cu —— O17 前置：BM=128 的「跨 BM 转置归约」wgmma 布局冒烟
// =============================================================================
// O17 把主 kernel 从「1 warpgroup / BM=64」扩到「2 warpgroups / BM=128」，目的是让
// **dK/dV 的跨 CTA atomic 字节数减半**（一个 KV 元素被 nblk/2 个 CTA 贡献而不是 nblk 个）。
// 做法：2 个 warpgroup 各算自己那 64 行 Q 的 P/dS；dK/dV 由 **wg0 一个 warpgroup** 对
// 全 BM=128 归约——**同一条 `wgmma.m64n64k16` 累加器连续吃两个 m64 半（s=0..7 共 128 行）**，
// 得到的就是两半之和，最后一次 `red.global.add`。
//
// 本冒烟逐位验证「对 [128][*] 的 K-major SW128 tile，用 MN-major 描述符按 s=0..7 读转置」：
//   (1) dV = Pᵀ·dO：A=P[128][BN] 转置(MN)，B=dO[128][HD] 转置(MN)，K=BM=128 → 输出 [BN][HD]
//   (2) dK = dSᵀ·Q：同构
//   (3) dQ = dS·K：2 个 m64 半（各 64 行）分别读 dS 的对应 64 行块（A K-major）
// 与 CPU fp32 参考比对（fp16 输入，误差应在 1e-2 内）。
//
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

constexpr int BM = 128, BN = 64, HD = 128, THREADS = 128;

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
__device__ __forceinline__ uint64_t make_desc_k(uint32_t addr, uint32_t W) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((((W >> 6) * 64)) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // B128
  return d;
}
__device__ __forceinline__ uint64_t make_desc_mn(uint32_t addr, uint32_t W) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)(64u & 0x3FFF) << 16;  // LBO
  d |= (uint64_t)((((W >> 6) * 64)) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // B128
  return d;
}
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

__global__ void __launch_bounds__(THREADS) smoke2_kernel(
    const __half* Q, const __half* Kt, const __half* V, const __half* dO,
    const __half* P, const __half* dS, float* dVout, float* dKout, float* dQout) {
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

  // (1) dV = Pᵀ·dO：A=P[128][BN] 转置(MN,W=BN)，B=dO[128][HD] 转置(MN,W=HD)，K=BM=128
  {
    const uint32_t pa = smem_u32(sP), doa = smem_u32(sDO);
    for (int nh = 0; nh < 2; ++nh) {
      float d[32]; for (int i = 0; i < 32; ++i) d[i] = 0.f;
      wgmma_fence();
      const uint32_t doa_n = doa + (uint32_t)(nh * 1024);
      for (int s = 0; s < BM / 16; ++s)
        wgmma_m64n64k16_f16<1, 1>(d, make_desc_mn(trans_k16_addr(pa, s, BN), BN),
                                  make_desc_mn(trans_k16_addr(doa_n, s, HD), HD));
      wgmma_commit(); wgmma_wait0();
      store_out(dVout, HD, nh, d);
    }
  }

  // (2) dK = dSᵀ·Q：A=dS[128][BN] 转置(MN,W=BN)，B=Q[128][HD] 转置(MN,W=HD)
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

  // (3) dQ = dS·K：两个 m64 半（行 0-63 / 64-127），A=dS 的对应 64 行块（K-major），
  //     B=K 转置(MN, W=HD)，K=BN；输出 [BM][HD]
  {
    const uint32_t dsa = smem_u32(sDS), ka = smem_u32(sK);
    for (int mh = 0; mh < 2; ++mh) {
      for (int nh = 0; nh < 2; ++nh) {
        float d[32]; for (int i = 0; i < 32; ++i) d[i] = 0.f;
        wgmma_fence();
        const uint32_t dsa_m = dsa + (uint32_t)(mh * (64 / 8) * (BN / 64) * 1024);
        const uint32_t ka_n = ka + (uint32_t)(nh * 1024);
        for (int s = 0; s < BN / 16; ++s)
          wgmma_m64n64k16_f16<0, 1>(d, make_desc_k(sw128_k16_addr(dsa_m, s), BN),
                                    make_desc_mn(trans_k16_addr(ka_n, s, HD), HD));
        wgmma_commit(); wgmma_wait0();
        for (int j = 0; j < 8; ++j)
          for (int q = 0; q < 4; ++q) {
            const int row = mh * 64 + wid * 16 + g + (q >= 2 ? 8 : 0);
            const int col = nh * 64 + j * 8 + c2 + (q & 1);
            if (row < BM && col < HD) dQout[row * HD + col] = d[j * 4 + q];
          }
      }
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

  std::vector<float> dVref(BN * HD), dKref(BN * HD), dQref(BM * HD);
  for (int n = 0; n < BN; ++n)
    for (int k = 0; k < HD; ++k) {
      float b = 0; for (int m = 0; m < BM; ++m) b += P[m*BN+n] * dO[m*HD+k];
      dVref[n*HD+k] = b;
      float c = 0; for (int m = 0; m < BM; ++m) c += dS[m*BN+n] * Q[m*HD+k];
      dKref[n*HD+k] = c;
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
  float *ddV, *ddK, *ddQ;
  CUDA_CHECK(cudaMalloc(&dQ_, hQ.size()*2)); CUDA_CHECK(cudaMalloc(&dK_, hK.size()*2));
  CUDA_CHECK(cudaMalloc(&dV_, hV.size()*2)); CUDA_CHECK(cudaMalloc(&dDO_, hDO.size()*2));
  CUDA_CHECK(cudaMalloc(&dP_, hP.size()*2)); CUDA_CHECK(cudaMalloc(&dDS_, hDS.size()*2));
  CUDA_CHECK(cudaMalloc(&ddV, BN*HD*4)); CUDA_CHECK(cudaMalloc(&ddK, BN*HD*4));
  CUDA_CHECK(cudaMalloc(&ddQ, BM*HD*4));
  CUDA_CHECK(cudaMemcpy(dQ_, hQ.data(), hQ.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK_, hK.data(), hK.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV_, hV.data(), hV.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDO_, hDO.data(), hDO.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dP_, hP.data(), hP.size()*2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDS_, hDS.data(), hDS.size()*2, cudaMemcpyHostToDevice));
  smoke2_kernel<<<1, THREADS>>>(dQ_, dK_, dV_, dDO_, dP_, dDS_, ddV, ddK, ddQ);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> dV(BN*HD), dK(BN*HD), dQ(BM*HD);
  CUDA_CHECK(cudaMemcpy(dV.data(), ddV, dV.size()*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dK.data(), ddK, dK.size()*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dQ.data(), ddQ, dQ.size()*4, cudaMemcpyDeviceToHost));

  auto maxerr = [](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0; for (size_t i = 0; i < a.size(); ++i) e = std::max(e, (double)std::fabs(a[i]-b[i])); return e;
  };
  const double eDV = maxerr(dV, dVref), eDK = maxerr(dK, dKref), eDQ = maxerr(dQ, dQref);
  printf("wgmma2 dV=PᵀdO (BM=128 trans) vs CPU: max_abs=%.3e\n", eDV);
  printf("wgmma2 dK=dSᵀQ (BM=128 trans) vs CPU: max_abs=%.3e\n", eDK);
  printf("wgmma2 dQ=dS·K (2x m64)         vs CPU: max_abs=%.3e\n", eDQ);
  bool ok = eDV < 1e-2 && eDK < 1e-2 && eDQ < 1e-2;
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dQ_); cudaFree(dK_); cudaFree(dV_); cudaFree(dDO_); cudaFree(dP_); cudaFree(dDS_);
  cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
  return ok ? 0 : 1;
}
