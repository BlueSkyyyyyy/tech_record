// =============================================================================
// fa_bwd_fp8_wgmma2_smoke.cu —— F6 前置：双 warpgroup（256 线程）fp8 wgmma 冒烟
// =============================================================================
// 背景（ROADMAP『fp8 专项冲刺』F6）：默认 fp8 main 的墙是 dK/dV 跨 CTA 的 L2
// `red`（114.5M 扇区，占 L2 流量 74%），唯一真杠杆是把 **BM 64→128** 让每个 KV 元素
// 被一半的 CTA 贡献。已有 `fa_bwd_fp8_wg2_kernel<128>`（BM=128、2 warpgroup、mma +
// cp.async）确实把 `red` 砍半，但它 **217 regs / 131KB smem → 1 CTA/SM**，L2 只到
// 20.4%（延迟/occupancy bound），反而慢 0.53×。ROADMAP 判定的解法是「双 warpgroup +
// TMA + wgmma」——用 Hopper 原语把寄存器/指令压下来，才能在 BM=128 下保住 L2 利用率。
//
// 本冒烟把 F6 最不确定的一块先钉死：**在 256 线程（2 个 warpgroup）的 CTA 里，用
// K-major SW128 描述符跑 fp8 `wgmma.m64n64k32`，每个 warpgroup 各算 BM=128 的一半
// （64 行）**，并逐元素对拍 CPU fp32 参考：
//   (1) GEMM1  S = scale·Q·Kᵀ  ：Q[128][128] e4m3，K[64][128] e4m3
//   (2) GEMM2  dP = dO·Vᵀ      ：dO[128][128] e5m2，V[64][128] e4m3
// 验证点：
//   * wgmma 是 **warpgroup 级**：2 个 WG 必须各自发起自己的 wgmma；
//   * 第 2 个 WG 的 A 描述符基址要按 `(64 行 / 8) × atom(1024B) = 8192B` 偏移；
//   * accumulator 布局 `row = wg*64 + wl*16 + g + (q>=2?8:0)`、
//     `col = j*8 + 2*(lane%4) + (q&1)` 在 2-WG 下两侧都对（GEMM3/4/5 的 epilogue 依赖）。
//
// 输入取 {-3..3} 整数：e4m3（3 位尾数）与 e5m2（2 位尾数）都能**精确**表示，
// CPU 参考可直接用整数值，避开 host 侧 fp8 反量化坑（见 kernel-opt 32 篇）。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//         scripts/run.sh src/fp8/fa_bwd_fp8_wgmma2_smoke.cu
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

// ---------- fp8 SW128 布局（与 fp8 kernels.cuh / wgmma_smoke 逐字一致） ----------
// 布局 [row/8][k/128][8 行][128 元素]，atom 1024B；16B chunk c' = (k/16 & 7) ^ (row&7)。
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
  d |= (uint64_t)0 << 49;  // base_offset
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}

// ---------- wgmma ----------
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_m64n64k32_e4e4(float (&d)[32], uint64_t da,
                                                     uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, %35, %36;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
}
__device__ __forceinline__ void wgmma_m64n64k32_e5e4(float (&d)[32], uint64_t da,
                                                     uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, %35, %36;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
}

constexpr int BM = 128, BN = 64, HD = 128, THREADS = 256;  // 256 = 2 个 warpgroup

// 2-WG 主 kernel：Q[BM][HD]、K[BN][HD]（Kᵀ 的每行是一个 key）、dO[BM][HD]、V[BN][HD]。
// 输出 S[BM][BN]（GEMM1）与 dP[BM][BN]（GEMM2），均为 fp32。
__global__ void __launch_bounds__(THREADS) wgmma2_smoke_kernel(
    const unsigned char* __restrict__ Q, const unsigned char* __restrict__ Kt,
    const unsigned char* __restrict__ dO, const unsigned char* __restrict__ V,
    float* __restrict__ Sout, float* __restrict__ dPout) {
  // sQ / sDO：[BM/8] 个 row-group × 1 个 k-group × 1024B = 16384B。
  // sK / sV ：[BN/8] × 1 × 1024B = 8192B。
  __shared__ __align__(1024) char sQ[(BM / 8) * (HD / 128) * 1024];   // 16384
  __shared__ __align__(1024) char sK[(BN / 8) * (HD / 128) * 1024];   // 8192
  __shared__ __align__(1024) char sDO[(BM / 8) * (HD / 128) * 1024];  // 16384
  __shared__ __align__(1024) char sV[(BN / 8) * (HD / 128) * 1024];   // 8192
  const int tid = threadIdx.x;
  const int n16 = HD / 16;  // 每行 16B chunk 数
  for (int i = tid; i < BM * n16; i += THREADS) {
    const int row = i / n16, k0 = (i % n16) * 16;
    sw128_store16_fp8(sQ, row, k0, HD, *reinterpret_cast<const uint4*>(Q + row * HD + k0));
    sw128_store16_fp8(sDO, row, k0, HD,
                      *reinterpret_cast<const uint4*>(dO + row * HD + k0));
  }
  for (int i = tid; i < BN * n16; i += THREADS) {
    const int row = i / n16, k0 = (i % n16) * 16;
    sw128_store16_fp8(sK, row, k0, HD,
                      *reinterpret_cast<const uint4*>(Kt + row * HD + k0));
    sw128_store16_fp8(sV, row, k0, HD, *reinterpret_cast<const uint4*>(V + row * HD + k0));
  }
  __syncthreads();

  const int wid = tid >> 5, lane = tid & 31;
  const int wg = wid >> 2;   // 0/1：两个 warpgroup
  const int wl = wid & 3;    // warpgroup 内 warp 号
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const uint32_t sbo = (uint32_t)((HD / 128) * 1024);  // K=128 → 1024B
  // 第 2 个 warpgroup 的第 0 行是全局第 64 行 ⇒ row-group 偏移 (64/8)=8 个 atom。
  const uint32_t qa = smem_u32(sQ) + (uint32_t)(wg * (64 / 8) * 1024);
  const uint32_t doa = smem_u32(sDO) + (uint32_t)(wg * (64 / 8) * 1024);
  const uint32_t ka = smem_u32(sK), va = smem_u32(sV);

  auto epilogue = [&](float* dst, const float (&d)[32]) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
        const int col = j * 8 + c2 + (q & 1);
        dst[row * BN + col] = d[j * 4 + q];
      }
  };

  // (1) S = Q·Kᵀ（e4m3×e4m3），K 归约维 = HD = 128 = 4×k32。
  {
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.f;
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < HD / 32; ++s)
      wgmma_m64n64k32_e4e4(d, make_desc_sw128_fp8(sw128_k32_addr(qa, s), sbo),
                           make_desc_sw128_fp8(sw128_k32_addr(ka, s), sbo));
    wgmma_commit();
    wgmma_wait0();
    epilogue(Sout, d);
  }
  // (2) dP = dO·Vᵀ（e5m2×e4m3），同一 warpgroup 再来一发。
  {
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.f;
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < HD / 32; ++s)
      wgmma_m64n64k32_e5e4(d, make_desc_sw128_fp8(sw128_k32_addr(doa, s), sbo),
                           make_desc_sw128_fp8(sw128_k32_addr(va, s), sbo));
    wgmma_commit();
    wgmma_wait0();
    epilogue(dPout, d);
  }
}

static unsigned char enc(float x, bool e5) {
  __nv_fp8_storage_t b = e5 ? __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)
                            : __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  return (unsigned char)b;
}

int main() {
  std::vector<float> Q(BM * HD), Kt(BN * HD), dO(BM * HD), V(BN * HD);
  srand(2024);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };  // {-3..3} 二者皆精确
  for (auto& x : Q) x = rnd();
  for (auto& x : Kt) x = rnd();
  for (auto& x : dO) x = rnd();
  for (auto& x : V) x = rnd();

  std::vector<float> Sref(BM * BN), dPref(BM * BN);
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < BN; ++n) {
      float a = 0.f, b = 0.f;
      for (int k = 0; k < HD; ++k) {
        a += Q[m * HD + k] * Kt[n * HD + k];
        b += dO[m * HD + k] * V[n * HD + k];
      }
      Sref[m * BN + n] = a;
      dPref[m * BN + n] = b;
    }

  std::vector<unsigned char> Qb(BM * HD), Kb(BN * HD), dob(BM * HD), Vb(BN * HD);
  for (int i = 0; i < BM * HD; ++i) {
    Qb[i] = enc(Q[i], false);
    dob[i] = enc(dO[i], true);  // dO 用 e5m2
  }
  for (int i = 0; i < BN * HD; ++i) {
    Kb[i] = enc(Kt[i], false);
    Vb[i] = enc(V[i], false);
  }

  unsigned char *dQ, *dK, *dDO, *dV;
  float *dS, *dP;
  CUDA_CHECK(cudaMalloc(&dQ, Qb.size()));
  CUDA_CHECK(cudaMalloc(&dK, Kb.size()));
  CUDA_CHECK(cudaMalloc(&dDO, dob.size()));
  CUDA_CHECK(cudaMalloc(&dV, Vb.size()));
  CUDA_CHECK(cudaMalloc(&dS, BM * BN * 4));
  CUDA_CHECK(cudaMalloc(&dP, BM * BN * 4));
  CUDA_CHECK(cudaMemcpy(dQ, Qb.data(), Qb.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, Kb.data(), Kb.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDO, dob.data(), dob.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, Vb.data(), Vb.size(), cudaMemcpyHostToDevice));

  wgmma2_smoke_kernel<<<1, THREADS>>>(dQ, dK, dDO, dV, dS, dP);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> S(BM * BN), dPv(BM * BN);
  CUDA_CHECK(cudaMemcpy(S.data(), dS, S.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dPv.data(), dP, dPv.size() * 4, cudaMemcpyDeviceToHost));

  double e1 = 0, e2 = 0;
  for (int i = 0; i < BM * BN; ++i) {
    e1 = std::max(e1, (double)std::fabs(S[i] - Sref[i]));
    e2 = std::max(e2, (double)std::fabs(dPv[i] - dPref[i]));
  }
  printf("wgmma2(2 WG) GEMM1 S=QKᵀ  e4m3×e4m3  BM=128 vs CPU: max_abs=%.3e\n", e1);
  printf("wgmma2(2 WG) GEMM2 dP=dO·Vᵀ e5m2×e4m3 BM=128 vs CPU: max_abs=%.3e\n", e2);
  const bool ok = e1 < 1e-3 && e2 < 1e-3;
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dQ); cudaFree(dK); cudaFree(dDO); cudaFree(dV); cudaFree(dS); cudaFree(dP);
  return ok ? 0 : 1;
}
