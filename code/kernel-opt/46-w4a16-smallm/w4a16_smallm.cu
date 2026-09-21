// 46 W4A16 的 M∈[1,64] decode：GEMV 与张量核 GEMM 的交叉点 + 运行时自适应（主题 26f）
//
// 44 篇算过 W4A16 的算术强度：AI(M) = 2MNK / (0.53·N·K) ≈ 3.76·M，
// 而 H100 的 bf16 张量核 ridge = 989/3350 ≈ 295 FLOP/byte，于是
//   M* ≈ 295 / 3.76 ≈ 78
// ——只有 M ≳ 78，张量核 GEMM 才可能跑赢「权重带宽最优」的 GEMV。44 篇 M=1
// GEMV 0.0284ms vs 43 篇张量核 0.0798ms（2.81×）印证了这点。
//
// 但 43 篇的张量核骨架用的是 `wgmma.m64n128k16`：一个 tile 固定算 64 行。
// 在 M∈[2,16] 这个「小 batch decode」区间，它 63/64 ~ 15/16 的行在空转。
// 本篇问三件事：
//   1) 小 M 到底谁赢？把 M=1..64 全部实测出来，画交叉点；
//   2) 换成 `mma.sync.m16n8k16` 的 BM=16/32 小 tile，能不能靠「不白算行」
//      把交叉点往左推（让张量核在 M=16~64 就能用）？
//   3) 写一个按 M 选 kernel 的运行时分派器，量化它相对「只用一种 kernel」
//      的净收益。
//
// 真实 shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408, group_size=128, 对称 int4。
// 默认 N=17408（MLP up/gate），K=5120（hidden）。权重 int4 44.6MB + scale 2.8MB。
//
// 运行（含 wgmma，需 sm_90a）：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//     scripts/run.sh 46-w4a16-smallm/w4a16_smallm.cu [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int GROUP = 128;  // 每 128 个 k 一个 fp32 scale
constexpr int BK = 64;      // bf16 SW128 atom：64 元素 = 128B
constexpr double BF16_PEAK = 989.0;

// ===========================================================================
//  通用小工具
// ===========================================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}
__device__ __forceinline__ uint32_t pack2b(bf16 lo, bf16 hi) {
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
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

// ---------------------------------------------------------------------------
// (A) W4A16 GEMV：每 warp 一行（RWW 行）、warp 内沿 K 并行、MT 个 token 共享
//     一次权重读（44 篇最佳骨架 + 多 token 扩展）。
//     A 是 bf16 [M][K]，权重 int4 [N][K/2] + scale [N][K/128]。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_a16_kernel(const bf16* __restrict__ A,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int m0, int M, int N,
                                                       int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);
  float* ssm = reinterpret_cast<float*>(smem + (size_t)MT * KK * 2);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 8; i += 256) {
    const int m = i / (KK / 8), c8 = i % (KK / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (m0 + m < M) v = *reinterpret_cast<const uint4*>(&A[(size_t)(m0 + m) * KK + c8 * 8]);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c8 * 8]) = v;
  }
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += 256) {
    const int n = i / NGRP, g = i % NGRP;
    ssm[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    union U4 {
      uint4 u;
      bf16 b[8];
    };
    U4 xu[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        xu[m][q].u = *reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0 + q * 8]);

    const int g = k0 / GROUP;
#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row0 + warp * RWW + r;
      const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + c * 16));
      const uint32_t wcomp[4] = {wv.x, wv.y, wv.z, wv.w};
      float dq[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const uint32_t byte = (wcomp[e >> 2] >> (8 * (e & 3))) & 0xFF;
        dq[2 * e] = (float)(int)(byte & 0xF) - 8.f;
        dq[2 * e + 1] = (float)(int)(byte >> 4) - 8.f;
      }
      const float s = ssm[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        float part = 0.f;
#pragma unroll
        for (int idx = 0; idx < 32; ++idx)
          part += dq[idx] * __bfloat162float(xu[m][idx >> 3].b[idx & 7]);
        acc[r][m] += s * part;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0 && m0 + m < M) C[(size_t)(m0 + m) * N + row0 + warp * RWW + r] = v;
    }
}

// ---------------------------------------------------------------------------
// (B) W4A16 小 tile 张量核 GEMM：`mma.sync.m16n8k16`，BM 可到 16/32/64。
//     · A  bf16 [M][K]     → Araw[STAGES][BM][BK] （cp.async + ldmatrix 非转置）
//     · W  int4 [N][K/2]   → Wraw[STAGES][BN][BK/2]（cp.async）再反量化到
//                            Bs[STAGES][BN][BKP] bf16（n-major，k 连续）
//     · B 片段用**非转置** ldmatrix.x4 从 Bs 取：Bs[n][k] 的行就是 k 连续，
//       ldmatrix 给出的 (row=l/4, col=(l%4)*2) 恰好等于 mma B 想要的
//       (k=(l%4)*2, n=l/4)（见文章推导）——不再需要 .trans。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BKi, int STAGES, int WM, int WN, int KSPLIT = 1>
__global__ void __launch_bounds__(WM* WN* 32) gemm_mma_w4a16_kernel(
    const bf16* __restrict__ A, const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
    float* __restrict__ C, int M, int N, int K) {
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M % 16 == 0 && WARP_N % 8 == 0, "warp tile 必须是 16 的倍数");
  constexpr int MTM = WARP_M / 16;
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = BKi + 8;   // A smem 行距 padding
  constexpr int BKP = BKi + 8;   // B smem 行距 padding
  constexpr int NGRP = 5120 / GROUP;
  constexpr int NW4 = BKi / 8;   // 每个 n 行的 uint32 数（8 个 nibble / word）

  extern __shared__ __align__(16) char smem[];
  bf16* Araw = reinterpret_cast<bf16*>(smem);                       // [STAGES][BM][ASP]
  uint8_t* Wraw = reinterpret_cast<uint8_t*>(Araw + (size_t)STAGES * BM * ASP);  // [STAGES][BN][BKi/2]
  bf16* Bs = reinterpret_cast<bf16*>(Wraw + (size_t)STAGES * BN * (BKi / 2));    // [STAGES][BN][BKP]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_row = wid / WN;
  const int warp_col = wid % WN;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / BKi;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;

  auto load_tile = [&](int st, int k0) {
    bf16* a = Araw + (size_t)st * BM * ASP;
    uint8_t* wp = Wraw + (size_t)st * BN * (BKi / 2);
    for (int i = tid; i < BM * (BKi / 8); i += NTHREADS) {
      const int r = i / (BKi / 8), c8 = (i % (BKi / 8)) * 8;
      const int gr = block_row + r;
      if (gr < M) {
        __pipeline_memcpy_async(&a[r * ASP + c8], &A[(size_t)gr * K + k0 + c8], 16);
      } else {
        *reinterpret_cast<uint4*>(&a[r * ASP + c8]) = make_uint4(0, 0, 0, 0);
      }
    }
    for (int i = tid; i < BN * (BKi / 2 / 16); i += NTHREADS) {
      const int r = i / (BKi / 2 / 16), c = (i % (BKi / 2 / 16)) * 16;
      __pipeline_memcpy_async(&wp[r * (BKi / 2) + c],
                              &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
    }
    __pipeline_commit();
  };
  auto dequant = [&](int st, int k0) {
    uint8_t* wp = Wraw + (size_t)st * BN * (BKi / 2);
    bf16* b = Bs + (size_t)st * BN * BKP;
    const int g = k0 / GROUP;  // BKi=64 整除 GROUP=128 → 一个 tile 落在单个 group
    for (int i = tid; i < BN * NW4; i += NTHREADS) {
      const int n = i / NW4, w = i % NW4;
      const uint32_t word = *reinterpret_cast<const uint32_t*>(&wp[n * (BKi / 2) + w * 4]);
      const float s = __ldg(&Ws[(size_t)(block_col + n) * NGRP + g]);
      bf16 out[8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int q0 = (int)((word >> (8 * q)) & 0xF) - 8;
        const int q1 = (int)((word >> (8 * q + 4)) & 0xF) - 8;
        out[2 * q] = __float2bfloat16((float)q0 * s);
        out[2 * q + 1] = __float2bfloat16((float)q1 * s);
      }
      *reinterpret_cast<uint4*>(&b[n * BKP + w * 8]) = *reinterpret_cast<const uint4*>(out);
    }
  };

  float acc[MTM][MTN][4];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  load_tile(0, kb0 * BKi);
  int stage = 0;
  const int nt = kb1 - kb0;
#pragma unroll 1
  for (int t = 0; t < nt; ++t) {
    const int k0 = (kb0 + t) * BKi;
    if (t + 1 < nt) load_tile(stage ^ 1, k0 + BKi);
    else __pipeline_commit();
    __pipeline_wait_prior(t + 1 < nt ? 1 : 0);
    __syncthreads();
    dequant(stage, k0);
    __syncthreads();
    // ---- mma ----
    const bf16* a = Araw + (size_t)stage * BM * ASP;
    const bf16* b = Bs + (size_t)stage * BN * BKP;
#pragma unroll
    for (int kk = 0; kk < BKi / 16; ++kk) {
      uint32_t af[MTM][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i) {
        const int row = (lane & 15);
        const int col = (lane >> 4) * 8;
        ldmatrix_x4(smem_u32(&a[(warp_row * WARP_M + i * 16 + row) * ASP + kk * 16 + col]), af[i]);
      }
      uint32_t bf[MTN][2];
#pragma unroll
      for (int g = 0; g < MTN / 2; ++g) {
        const int n_row = (lane & 7) + ((lane >> 4) & 1) * 8;
        const int k_off = ((lane >> 3) & 1) * 8;
        uint32_t d[4];
        ldmatrix_x4(
            smem_u32(&b[(warp_col * WARP_N + g * 16 + n_row) * BKP + kk * 16 + k_off]), d);
        bf[g * 2][0] = d[0];
        bf[g * 2][1] = d[1];
        bf[g * 2 + 1][0] = d[2];
        bf[g * 2 + 1][1] = d[3];
      }
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j) mma_m16n8k16(acc[i][j], af[i], bf[j]);
    }
    __syncthreads();
    stage ^= 1;
  }

  // ---- epilogue ----
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
        if (r < M) {
          if (KSPLIT == 1) C[(size_t)r * N + c] = acc[i][j][q];
          else atomicAdd(&C[(size_t)r * N + c], acc[i][j][q]);
        }
      }
    }
}

// ---------------------------------------------------------------------------
// (C) 43/41 篇的张量核 baseline：wgmma.m64n128k16（BM 固定 64）——小 M 时
//     63/64 行空转的「行税」代表。原样搬来做同进程 head-to-head。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;  // LBO = 1 (16B)
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
__device__ __forceinline__ uint32_t k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\nLAB_WAIT%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

template <int BM, int BN, int STAGES, int KSPLIT, int NCONS, bool DQ = true, int NPRODW = 1,
          int MINB = 1>
__global__ void __launch_bounds__((NPRODW + NCONS) * 128, MINB) w4a16_ws_kernel(
    const bf16* __restrict__ A, const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
    float* __restrict__ C, int M, int N, int K) {
  static_assert(BM == 64, "BM 固定 64");
  constexpr int NPB = BK / 2;
  constexpr int NPROD = NPRODW * 128;
  constexpr int T = (NPRODW + NCONS) * 128;
  constexpr int SBO = 1024;
  constexpr int ABYTES = BM * BK * 2;
  constexpr int WB_BYTES = BN * NPB;
  constexpr int SW_BYTES = BN * BK * 2;

  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + ((2 * STAGES * 8 + 1023) / 1024) * 1024;
  char* Wps = As + (size_t)STAGES * ABYTES;
  char* sW = Wps + (size_t)STAGES * WB_BYTES;
  float* sctab = reinterpret_cast<float*>(sW + (size_t)STAGES * SW_BYTES);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nblk_z = kb1 - kb0;
  const int ngrp = K / GROUP;
  const int NGR = (nblk_z + 1) >> 1;

  for (int s = tid; s < STAGES; s += T) {
    mbar_init(full + s, 1);
    mbar_init(empty + s, NCONS * 128);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();

  if (tid < NPROD) {
    const int g0 = kb0 >> 1;
    const int ngrp_rel = (nblk_z + 1) >> 1;
    for (int i = tid; i < BN * ngrp_rel; i += NPROD) {
      const int n = i / ngrp_rel, gg = i % ngrp_rel;
      sctab[n * NGR + gg] = __ldg(&Ws[(size_t)(block_col + n) * ngrp + g0 + gg]);
    }
    auto load_stage = [&](int st, int kb) {
      const int k0 = kb * BK;
      char* a = As + (size_t)st * ABYTES;
      char* wp = Wps + (size_t)st * WB_BYTES;
      for (int i = tid; i < BM * (BK / 8); i += NPROD) {
        const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
        __pipeline_memcpy_async(a + sw128_off(r, c8, BK),
                                &A[(size_t)(block_row + r) * K + k0 + c8], 16);
      }
      for (int i = tid; i < BN * (NPB / 16); i += NPROD) {
        const int r = i / (NPB / 16), c = (i % (NPB / 16)) * 16;
        __pipeline_memcpy_async(wp + r * NPB + c,
                                &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
      }
      __pipeline_commit();
    };
    auto dequant_stage = [&](int st, int kb) {
      if constexpr (!DQ) return;
      char* w = sW + (size_t)st * SW_BYTES;
      const int gg = (kb - kb0) >> 1;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(Wps + (size_t)st * WB_BYTES);
      constexpr int M4 = NPB / 4;
      for (int i = tid; i < BN * M4; i += NPROD) {
        const int n = i / M4, m4 = i % M4;
        const uint32_t word = wp[i];
        const float s = sctab[n * NGR + gg];
        bf16 out[8];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int q0 = (int)((word >> (8 * b)) & 0xF) - 8;
          const int q1 = (int)((word >> (8 * b + 4)) & 0xF) - 8;
          out[2 * b] = __float2bfloat16((float)q0 * s);
          out[2 * b + 1] = __float2bfloat16((float)q1 * s);
        }
        const int kk = (8 * m4) & 63;
        const int rr = n & 7, rg = n >> 3;
        const int cc = (kk >> 3) ^ rr;
        const int off = rg * 1024 + (rr * 8 + cc) * 16;
        *reinterpret_cast<uint4*>(w + off) = *reinterpret_cast<const uint4*>(out);
      }
    };
    constexpr int PF = 1;
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      if (p < nblk_z) load_stage(p, kb0 + p);
      else __pipeline_commit();
    }
    for (int t = 0; t < nblk_z; ++t) {
      const int s = t % STAGES;
      const int tn = t + PF;
      if (tn < nblk_z) {
        if (tn >= STAGES) {
          const uint32_t ph = (uint32_t)((tn / STAGES - 1) & 1);
          mbar_wait(empty + tn % STAGES, ph);
        }
        load_stage(tn % STAGES, kb0 + tn);
      } else {
        __pipeline_commit();
      }
      __pipeline_wait_prior(PF);
      named_bar_sync(1, NPROD);
      dequant_stage(s, kb0 + t);
      named_bar_sync(1, NPROD);
      fence_proxy_async();
      if (tid == 0) mbar_arrive(full + s);
    }
  } else {
    const int cg = (warp - NPRODW * 4) >> 2;
    const int nbase = cg * 128;
    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    char* myw = sW + (size_t)nbase * BK * 2;

    for (int t = 0; t < nblk_z; ++t) {
      const int s = t % STAGES;
      mbar_wait(full + s, (uint32_t)((t / STAGES) & 1));
      char* a = As + (size_t)s * ABYTES;
      char* w = myw + (size_t)s * SW_BYTES;
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), kk), SBO);
        uint64_t db = make_desc_sw128(k16_addr(smem_u32(w), kk), SBO);
        wgmma_m64n128k16(acc, da, db, 1);
      }
      wgmma_commit();
      wgmma_wait0();
      mbar_arrive(empty + s);
    }
    wgmma_wait0();

    const int row0 = (warp & 3) * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = block_col + nbase + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8;
      if (KSPLIT == 1) {
        if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + col]) = make_float2(acc[j * 4 + 0], acc[j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + col]) = make_float2(acc[j * 4 + 2], acc[j * 4 + 3]);
      } else {
        if (r0 < M) {
          atomicAdd(&C[(size_t)r0 * N + col], acc[j * 4 + 0]);
          atomicAdd(&C[(size_t)r0 * N + col + 1], acc[j * 4 + 1]);
        }
        if (r1 < M) {
          atomicAdd(&C[(size_t)r1 * N + col], acc[j * 4 + 2]);
          atomicAdd(&C[(size_t)r1 * N + col + 1], acc[j * 4 + 3]);
        }
      }
    }
  }
}

// 只读权重（int4）+ scale 的上界，同 44/45 roof 口径
__global__ void wread_roofline_kernel(const uint8_t* __restrict__ Wp,
                                      const float* __restrict__ Ws, float* __restrict__ out, int N,
                                      int K) {
  const size_t nword = (size_t)N * (K / 2) / 16;
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nword;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(Wp + i * 16);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  const size_t ns = ((size_t)N * (K / GROUP)) / 8;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ns;
       i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = *reinterpret_cast<const float4*>(Ws + i * 4);
    acc ^= __float_as_uint(v.x) ^ __float_as_uint(v.y) ^ __float_as_uint(v.z) ^ __float_as_uint(v.w);
  }
  out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
}

// ===========================================================================
//  host
// ===========================================================================
static bf16 f2b(float x) { return __float2bfloat16(x); }

struct Buf {
  bf16* A;
  uint8_t* Wp;
  float* Ws;
  float* C;
  float* Out;
};

// name + 「给定 M 的完整 launch」+ 权重读遍数除数
struct Spec {
  const char* name;
  int wpass_div;  // 权重读遍数 = ceil(M / wpass_div)
  std::function<void(int M)> launch;
};

template <int RWW, int MT>
static void add_gemv(std::vector<Spec>& specs, const char* name, const Buf& buf, int N, int K) {
  auto fn = gemv_a16_kernel<RWW, MT, 5120>;
  const int BN = 8 * RWW;
  const size_t shm = (size_t)MT * K * 2 + (size_t)BN * (K / GROUP) * 4;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  specs.push_back({name, MT, [=](int M) {
                     for (int m0 = 0; m0 < M; m0 += MT)
                       fn<<<N / BN, 256, shm>>>(buf.A, buf.Wp, buf.Ws, buf.C, m0, M, N, K);
                   }});
}

template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1>
static void add_mma(std::vector<Spec>& specs, const char* name, const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a16_kernel<BM, BN, BK, STAGES, WM, WN, KSPLIT>;
  const int NT = WM * WN * 32;
  const size_t shm = (size_t)STAGES * (BM * (BK + 8) * 2 + BN * (BK / 2) + BN * (BK + 8) * 2);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  specs.push_back({name, BM, [=](int M) {
                     if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
                     fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(
                         buf.A, buf.Wp, buf.Ws, buf.C, M, N, K);
                   }});
}

template <int STAGES, int KSPLIT>
static void add_wgmma(std::vector<Spec>& specs, const char* name, const Buf& buf, int N, int K) {
  auto fn = w4a16_ws_kernel<64, 128, STAGES, KSPLIT, 1, true, 1, 1>;
  const int nblk_ = K / BK, ngrmax = (nblk_ / KSPLIT + 1) / 2 + 1;
  const size_t shm =
      1024 + (size_t)STAGES * (64 * BK * 2 + 128 * (BK / 2) + 128 * BK * 2) + 128 * ngrmax * 4;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  specs.push_back({name, 64, [=](int M) {
                     fn<<<dim3(div_up(N, 128), div_up(M, 64), KSPLIT), 256, shm>>>(
                         buf.A, buf.Wp, buf.Ws, buf.C, M, N, K);
                   }});
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* which = (argc > 1) ? argv[1] : "all";
  const char* only = (argc > 2) ? argv[2] : "";
  const int N = 17408;
  const int K = 5120;
  const int MAXM = 64;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double w_bytes_int4 = (double)N * K / 2;
  const double s_bytes = (double)N * (K / GROUP) * 4;
  const double total_w = w_bytes_int4 + s_bytes;
  std::printf("\nW4A16 small-M sweep: N=%d K=%d group=%d ; W_int4=%.1f MB scales=%.1f MB total=%.1f MB\n",
              N, K, GROUP, w_bytes_int4 / 1e6, s_bytes / 1e6, total_w / 1e6);
  std::printf("张量核 ridge M* = 989/(2/(0.53)) ... AI(M)=3.77*M, M* = %.0f\n\n", (BF16_PEAK * 1e3) / (d.mem_bw_gbps) / 3.77);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<bf16> hA((size_t)MAXM * K);
  std::vector<float> hWf((size_t)N * K), hWdeq((size_t)N * K);
  std::vector<uint8_t> hWp((size_t)N * (K / 2));
  std::vector<float> hWs((size_t)N * (K / GROUP));
  for (auto& v : hA) v = f2b(0.5f * dist(rng));
  for (auto& v : hWf) v = 0.1f * dist(rng);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < K / GROUP; ++g) {
      float amax = 0.f;
      for (int k = 0; k < GROUP; ++k)
        amax = std::max(amax, std::fabs(hWf[(size_t)n * K + g * GROUP + k]));
      const float s = amax > 0 ? amax / 7.f : 1.f;
      hWs[(size_t)n * (K / GROUP) + g] = s;
      for (int k = 0; k < GROUP; ++k) {
        const size_t idx = (size_t)n * K + g * GROUP + k;
        int q = std::max(-8, std::min(7, (int)std::lround(hWf[idx] / s)));
        hWdeq[idx] = q * s;
        const size_t pk = (size_t)n * (K / 2) + (g * GROUP + k) / 2;
        if (((g * GROUP + k) & 1) == 0)
          hWp[pk] = (uint8_t)((hWp[pk] & 0xF0) | ((q + 8) & 0xF));
        else
          hWp[pk] = (uint8_t)((hWp[pk] & 0x0F) | (((q + 8) & 0xF) << 4));
      }
    }

  Buf buf;
  CUDA_CHECK(cudaMalloc(&buf.A, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&buf.Wp, hWp.size()));
  CUDA_CHECK(cudaMalloc(&buf.Ws, hWs.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.C, (size_t)MAXM * N * 4));
  CUDA_CHECK(cudaMalloc(&buf.Out, (size_t)4096 * 256 * 4));
  CUDA_CHECK(cudaMemcpy(buf.A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Wp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Ws, hWs.data(), hWs.size() * 4, cudaMemcpyHostToDevice));

  std::vector<float> hC((size_t)MAXM * N);
  const int CHK_M = MAXM;  // 用最大的 M 做一次完整精度校验
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), buf.C, (size_t)CHK_M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 48; ++i) {
      const int m = (i * 1009 + 3) % CHK_M, n = (i * 997 + 7) % N;
      double s = 0;
      for (int k = 0; k < K; ++k)
        s += (double)__bfloat162float(hA[(size_t)m * K + k]) * (double)hWdeq[(size_t)n * K + k];
      err = std::max(err, std::fabs((double)hC[(size_t)m * N + n] - s));
      ref = std::max(ref, std::fabs(s));
    }
    const bool ok = err / std::max(ref, 1e-6) < 3e-2;
    std::printf("  [%-14s] max_abs_err=%.3e (ref~%.1f) rel=%.2e %s\n", tag, err, ref,
                err / std::max(ref, 1e-6), ok ? "OK" : "FAIL");
    return ok;
  };
  const double flops_base = 2.0 * N * K;  // M=1
  (void)flops_base;
  auto want = [&](const char* n) { return strcmp(which, "all") == 0 || strcmp(which, n) == 0; };

  if (want("roof")) {
    const int grids[] = {528, 1056, 2112};
    for (int nb : grids) {
      char nm[48];
      std::snprintf(nm, sizeof(nm), "roof_g%d", nb);
      auto launch = [&] { wread_roofline_kernel<<<nb, 256>>>(buf.Wp, buf.Ws, buf.Out, N, K); };
      launch();
      CUDA_CHECK_LAST();
      const double ms = bench_ms(launch, 3, 30);
      std::printf("  %-18s %8.4f ms  W+s %7.1f GB/s (%4.1f%% HBM)\n", nm, ms,
                  total_w * 1000.0 / ms / 1e9, 100.0 * (total_w * 1000.0 / ms / 1e9) / d.mem_bw_gbps);
    }
  }

  // ---- launcher 列表：name + 「给定 M 的完整 launch」+ 权重读遍数 ----
  //   权重读遍数 passes 决定真实 DRAM 流量：
  //     GEMV(MT)   : ceil(M/MT)   （MT 个 token 共享一次权重读）
  //     mma(BM)    : ceil(M/BM)   （B 被每个 m-tile 重读）
  //     wgmma(m64) : ceil(M/64)
  std::vector<Spec> specs;
  add_gemv<1, 1>(specs, "gemv_r1_mt1", buf, N, K);
  add_gemv<2, 1>(specs, "gemv_r2_mt1", buf, N, K);
  add_gemv<4, 1>(specs, "gemv_r4_mt1", buf, N, K);
  add_gemv<2, 2>(specs, "gemv_r2_mt2", buf, N, K);
  add_gemv<1, 2>(specs, "gemv_r1_mt2", buf, N, K);
  add_gemv<4, 2>(specs, "gemv_r4_mt2", buf, N, K);
  add_gemv<2, 4>(specs, "gemv_r2_mt4", buf, N, K);
  add_gemv<1, 8>(specs, "gemv_r1_mt8", buf, N, K);
  add_gemv<2, 8>(specs, "gemv_r2_mt8", buf, N, K);
  add_mma<16, 128, 2, 1, 4>(specs, "mma_b16", buf, N, K);
  add_mma<16, 128, 2, 1, 4, 4>(specs, "mma_b16_k4", buf, N, K);
  add_mma<16, 128, 2, 1, 4, 8>(specs, "mma_b16_k8", buf, N, K);
  add_mma<32, 128, 2, 2, 4, 1>(specs, "mma_b32", buf, N, K);
  add_mma<32, 128, 2, 2, 4, 4>(specs, "mma_b32_k4", buf, N, K);
  add_mma<32, 128, 2, 2, 4, 8>(specs, "mma_b32_k8", buf, N, K);
  add_mma<64, 128, 2, 4, 4, 1>(specs, "mma_b64", buf, N, K);
  add_mma<64, 128, 2, 4, 4, 4>(specs, "mma_b64_k4", buf, N, K);
  add_wgmma<3, 1>(specs, "wgmma_m64_k1", buf, N, K);
  add_wgmma<3, 2>(specs, "wgmma_m64_k2", buf, N, K);
  add_wgmma<3, 3>(specs, "wgmma_m64_k3", buf, N, K);
  add_wgmma<3, 5>(specs, "wgmma_m64_k5", buf, N, K);

  // 只跑某个 spec（配合 ncu，argv[2]）
  std::vector<int> active;
  for (size_t si = 0; si < specs.size(); ++si)
    if (!*only || strcmp(specs[si].name, only) == 0) active.push_back((int)si);

  std::printf("=== 精度校验（M=%d, 抽样 48 点）===\n", CHK_M);
  for (int si : active) {
    cudaMemsetAsync(buf.C, 0, (size_t)CHK_M * N * 4);
    specs[si].launch(CHK_M);
    CUDA_CHECK_LAST();
    check(specs[si].name);
  }

  // ---- 主扫描 ----
  const int Ms[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
  const int NM = sizeof(Ms) / sizeof(int);
  std::printf("\n=== M 扫描（同进程、warmup=5 iters=50；括号=权重读遍数）===\n");
  std::printf("%-20s", "kernel\\M");
  for (int mi = 0; mi < NM; ++mi) std::printf("%11d", Ms[mi]);
  std::printf("\n");
  std::vector<std::vector<double>> tab(specs.size(), std::vector<double>(NM, 1e30));
  for (int si : active) {
    std::printf("%-20s", specs[si].name);
    for (int mi = 0; mi < NM; ++mi) {
      const int M = Ms[mi];
      auto launch = [&] {
        cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
        specs[si].launch(M);
      };
      launch();
      CUDA_CHECK_LAST();
      const double ms = bench_ms(launch, 5, 50);
      tab[si][mi] = ms;
      std::printf("%9.4f(%d)", ms, (M + specs[si].wpass_div - 1) / specs[si].wpass_div);
    }
    std::printf("\n");
  }
  if (*only) {
    CUDA_CHECK(cudaFree(buf.A));
    return 0;
  }

  // ---- 每个 M 的最优 + 真实流量带宽 ----
  std::printf("\n=== 每个 M 的最优 kernel（按 ms）===\n");
  std::printf("%-6s %-14s %9s %9s %9s %9s\n", "M", "best", "ms", "TFLOPS", "Wpass", "realGB/s");
  double sum_gemv = 0, sum_mma = 0, sum_wgmma = 0, sum_adapt = 0;
  for (int mi = 0; mi < NM; ++mi) {
    int bi = 0;
    for (size_t si = 1; si < specs.size(); ++si)
      if (tab[si][mi] < tab[bi][mi]) bi = si;
    const int M = Ms[mi];
    const int wpass = (M + specs[bi].wpass_div - 1) / specs[bi].wpass_div;
    const double real_bytes = (double)wpass * total_w + (double)M * K * 2 + (double)M * N * 4;
    const double ms = tab[bi][mi];
    std::printf("%-6d %-14s %9.4f %9.2f %9d %9.1f\n", M, specs[bi].name, ms,
                to_tflops(2.0 * M * N * K, ms), wpass, real_bytes * 1000.0 / ms / 1e9);
    double b_gemv = 1e30, b_mma = 1e30, b_wgmma = 1e30;
    for (size_t si = 0; si < specs.size(); ++si) {
      const bool is_gemv = strncmp(specs[si].name, "gemv", 4) == 0;
      const bool is_wgmma = strncmp(specs[si].name, "wgmma", 5) == 0;
      if (is_gemv) b_gemv = std::min(b_gemv, tab[si][mi]);
      else if (is_wgmma) b_wgmma = std::min(b_wgmma, tab[si][mi]);
      else b_mma = std::min(b_mma, tab[si][mi]);
    }
    sum_gemv += b_gemv; sum_mma += b_mma; sum_wgmma += b_wgmma; sum_adapt += ms;
  }
  std::printf("\n汇总（全部 %d 个 M 点各取最优之和，relative 以 always-wgmma 为 1.000）：\n", NM);
  std::printf("  always GEMV  : %.4f ms  (relative %.3f)\n", sum_gemv, sum_gemv / sum_wgmma);
  std::printf("  always mma   : %.4f ms  (relative %.3f)\n", sum_mma, sum_mma / sum_wgmma);
  std::printf("  always wgmma : %.4f ms  (relative %.3f)\n", sum_wgmma, 1.0);
  std::printf("  adaptive     : %.4f ms  (relative %.3f)  -> 相对 always-wgmma %.2fx / always-GEMV %.2fx\n",
              sum_adapt, sum_adapt / sum_wgmma, sum_wgmma / sum_adapt, sum_gemv / sum_adapt);

  CUDA_CHECK(cudaFree(buf.A));
  CUDA_CHECK(cudaFree(buf.Wp));
  CUDA_CHECK(cudaFree(buf.Ws));
  CUDA_CHECK(cudaFree(buf.C));
  CUDA_CHECK(cudaFree(buf.Out));
  return 0;
}
