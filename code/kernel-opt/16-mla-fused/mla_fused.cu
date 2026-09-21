// 16 MLA 注意力（二）：FlashMLA 式单 kernel 融合（prefill, MQA 吸收形式）
//
// 承接 15 篇：15 篇把 absorbed MLA 拆成「QK^T -> softmax -> PV」三个 kernel，
// 中间量 S(fp32)/P(bf16) 要各写读一遍显存，流量 ∝ H·Sq·Sk，Sq=Sk=1024 时约 1.6GB。
// 本篇把三者融进**一个 kernel**：
//   * QK^T 的 mma 累加器直接留在寄存器里做 online softmax（不落盘 S）；
//   * softmax 后的 P 直接打包成 mma 的 A 片段喂给 PV（不物化 bf16 P）；
//   * KV 分块常驻 smem，在 block 内多个 query 之间复用。
//
// 关键技巧：m16n8k16 的 fp32 累加器 C 的线程布局，与下一步 PV 的 A 片段布局
// **天然对齐**（C 的一对 (c0,c1) 就是 A 的一对 (a0,a1)），所以 P 的
// 「累加器 -> A 片段」转换不需要任何 shuffle，只需 f32->bf16 打包。
//
// 为了不让 DV=512 的输出累加器把寄存器打爆，把 DV 切成 DVGRP 份分给不同 warp，
// 代价是同一组 query 的 QK^T 会被 DVGRP 个 warp 重复算（duplication）。
// 用 ROWG / DVGRP 两个旋钮在「寄存器压力/occupancy」与「重复算力」之间取舍。
//
// 运行：scripts/run.sh 16-mla-fused/mla_fused.cu [Sq] [H] [Sk] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>

using bf16 = __nv_bfloat16;

// DeepSeek-V3/V4 MLA「吸收」维度（FlashMLA 的 MQA 口径：576/512）
constexpr int DC = 512;  // kv_lora_rank / NoPE
constexpr int DR = 64;   // decoupled RoPE
constexpr int DV = 512;  // 吸收后 v 维度 = kv_lora_rank
constexpr int DK = DC + DR;

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
// 加载 16x16 的 A 片段（行主序 [m][k]）
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
// 从 [k][n] 行主序取 mma 的 B(.col) 片段
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack2(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float4 make_f4(float a, float b, float c, float d) {
  return make_float4(a, b, c, d);
}

// ===========================================================================
// 融合 kernel
//   ROWG  : block 内 16-row 组的个数（决定 BM = 16*ROWG）
//   DVGRP : DV 被切成几份（每份 DVW = DV/DVGRP 个 warp 承担）
//   WARPS : block 内 warp 数（= ROWG * DVGRP）
//   KT    : 每次 KV 分块的列数（同时是 S 的 N 维）
// ===========================================================================
template <int ROWG, int DVGRP, int KT>
__global__ void __launch_bounds__(ROWG * DVGRP * 32) mla_fused_kernel(
    const bf16* __restrict__ qa, const bf16* __restrict__ qr,
    const bf16* __restrict__ ckv, const bf16* __restrict__ kr, float* __restrict__ out,
    int Sq, int Sk) {
  constexpr int WARPS = ROWG * DVGRP;
  constexpr int BM = ROWG * 16;
  constexpr int DVW = DV / DVGRP;
  constexpr int DKP = DK + 8;   // 8 个 bf16 的 padding，消 ldmatrix bank conflict
  constexpr int NT = KT / 8;    // S 的 n8 tile 数
  constexpr int NDV = DVW / 8;  // O 的 n8 tile 数

  extern __shared__ bf16 smem[];
  bf16 (*qs)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem);
  bf16 (*ks)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem + BM * DKP);

  const int h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int rowg = w % ROWG;   // 该 warp 负责的 16-row 组
  const int dvg = w / ROWG;    // 该 warp 负责的 DV 份
  const int T = WARPS * 32;

  // ---- 载入 Q = [q_abs | q_rope] -> qs[BM][DK] ----
  for (int i = tid; i < BM * DC / 8; i += T) {
    const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
    const int gq = q0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qa[((size_t)h * Sq + gq) * DC + c8]);
    *reinterpret_cast<uint4*>(&qs[row][c8]) = v;
  }
  for (int i = tid; i < BM * DR / 8; i += T) {
    const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
    const int gq = q0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qr[((size_t)h * Sq + gq) * DR + c8]);
    *reinterpret_cast<uint4*>(&qs[row][DC + c8]) = v;
  }

  float S[NT][4];
  float O[NDV][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int q = 0; q < 4; ++q) S[t][q] = 0.f;
#pragma unroll
  for (int t = 0; t < NDV; ++t)
#pragma unroll
    for (int q = 0; q < 4; ++q) O[t][q] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  __syncthreads();

  for (int k0 = 0; k0 < Sk; k0 += KT) {
    // ---- 载入 KV tile: [c_kv | k_rope] -> ks[KT][DK] ----
    for (int i = tid; i < KT * DC / 8; i += T) {
      const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
      *reinterpret_cast<uint4*>(&ks[row][c8]) =
          *reinterpret_cast<const uint4*>(&ckv[(size_t)(k0 + row) * DC + c8]);
    }
    for (int i = tid; i < KT * DR / 8; i += T) {
      const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
      *reinterpret_cast<uint4*>(&ks[row][DC + c8]) =
          *reinterpret_cast<const uint4*>(&kr[(size_t)(k0 + row) * DR + c8]);
    }
    __syncthreads();

    // ---- QK^T : S = Q @ K^T，K 以 [n=kv][k=dk] 行主序存于 ks ----
    // 每块开头清零 S（mma 是累加语义）
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q) S[t][q] = 0.f;
#pragma unroll
    for (int kk = 0; kk < DK / 16; ++kk) {
      uint32_t a[4];
      ldmatrix_x4(smem_u32(&qs[rowg * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]), a);
      // 非转置 ldmatrix 直接得到 B(.col) 片段：lane 行通道 = n，列通道 = k
      const int n_off = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int k_off = ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int nb = 0; nb < KT / 16; ++nb) {
        uint32_t d[4];
        ldmatrix_x4(smem_u32(&ks[nb * 16 + n_off][kk * 16 + k_off]), d);
        mma16816(S[nb * 2], a, d);
        mma16816(S[nb * 2 + 1], a, d + 2);
      }
    }

    // ---- block 内 softmax（行方向 = KV），再做 online 更新 ----
    float bmx0 = -INFINITY, bmx1 = -INFINITY;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      bmx0 = fmaxf(bmx0, fmaxf(S[t][0], S[t][1]));  // 行 r = lane/4
      bmx1 = fmaxf(bmx1, fmaxf(S[t][2], S[t][3]));  // 行 r+8
    }
    bmx0 = fmaxf(bmx0, __shfl_xor_sync(~0u, bmx0, 1));
    bmx0 = fmaxf(bmx0, __shfl_xor_sync(~0u, bmx0, 2));
    bmx1 = fmaxf(bmx1, __shfl_xor_sync(~0u, bmx1, 1));
    bmx1 = fmaxf(bmx1, __shfl_xor_sync(~0u, bmx1, 2));
    const float nm0 = fmaxf(m0, bmx0), nm1 = fmaxf(m1, bmx1);
    const float al0 = __expf(m0 - nm0), al1 = __expf(m1 - nm1);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      S[t][0] = __expf(S[t][0] - nm0);
      S[t][1] = __expf(S[t][1] - nm0);
      S[t][2] = __expf(S[t][2] - nm1);
      S[t][3] = __expf(S[t][3] - nm1);
      s0 += S[t][0] + S[t][1];
      s1 += S[t][2] + S[t][3];
    }
    s0 += __shfl_xor_sync(~0u, s0, 1);
    s0 += __shfl_xor_sync(~0u, s0, 2);
    s1 += __shfl_xor_sync(~0u, s1, 1);
    s1 += __shfl_xor_sync(~0u, s1, 2);
    l0 = l0 * al0 + s0;
    l1 = l1 * al1 + s1;
    m0 = nm0;
    m1 = nm1;
#pragma unroll
    for (int t = 0; t < NDV; ++t) {
      O[t][0] *= al0;
      O[t][1] *= al0;
      O[t][2] *= al1;
      O[t][3] *= al1;
    }

    // ---- PV : O += P @ V，V = c_kv 以 [k=kv][n=dv] 行主序存于 ks ----
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      // 累加器布局 == A 片段布局，无需 shuffle
      uint32_t pa[4];
      pa[0] = pack2(S[c * 2][0], S[c * 2][1]);
      pa[1] = pack2(S[c * 2][2], S[c * 2][3]);
      pa[2] = pack2(S[c * 2 + 1][0], S[c * 2 + 1][1]);
      pa[3] = pack2(S[c * 2 + 1][2], S[c * 2 + 1][3]);
      const int r = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int dn = 0; dn < DVW / 16; ++dn) {
        uint32_t d[4];
        ldmatrix_x4_trans(
            smem_u32(&ks[c * 16 + r][dvg * DVW + dn * 16 + (lane >> 4) * 8]), d);
        mma16816(O[dn * 2], pa, d);
        mma16816(O[dn * 2 + 1], pa, d + 2);
      }
    }
    __syncthreads();
  }

  // ---- 输出 O / l ----
  const float iv0 = 1.f / l0, iv1 = 1.f / l1;
  const int row0 = q0 + rowg * 16 + (lane >> 2);
  const int row1 = row0 + 8;
#pragma unroll
  for (int t = 0; t < NDV; ++t) {
    const int col = dvg * DVW + t * 8 + (lane & 3) * 2;
    if (row0 < Sq) {
      out[((size_t)h * Sq + row0) * DV + col] = O[t][0] * iv0;
      out[((size_t)h * Sq + row0) * DV + col + 1] = O[t][1] * iv0;
    }
    if (row1 < Sq) {
      out[((size_t)h * Sq + row1) * DV + col] = O[t][2] * iv1;
      out[((size_t)h * Sq + row1) * DV + col + 1] = O[t][3] * iv1;
    }
  }
}

// ===========================================================================
// v5：共享 P 消除 QK 重复（dup=1）
//
// f4 里每个 16-row 组被 DVGRP=2 个 warp 各算一遍完整 QK（重复 2×）。
// 这里改成：同一 row 组的两个 warp **各算一半 KV 的 QK**（不重复），
// 把 S 的 block max / sum 通过极小的 smem 交换，再把 P(bf16) 写进 smem 共享；
// PV 阶段两 warp 各读完整 P、只算自己的 DV 半边。总 QK 算力降为 1×。
//
//   warp (rg, dv): QK 只算 KV[dv*32 : dv*32+32]；P 写 smem[rg 的 16 行][KV]
//   行 max/sum 跨两个 warp 用 pmax/psum smem 交换
// ===========================================================================
template <int ROWG, int DVGRP, int KT>
__global__ void __launch_bounds__(ROWG * DVGRP * 32) mla_shared_kernel(
    const bf16* __restrict__ qa, const bf16* __restrict__ qr,
    const bf16* __restrict__ ckv, const bf16* __restrict__ kr, float* __restrict__ out, int Sq,
    int Sk) {
  constexpr int WARPS = ROWG * DVGRP;
  constexpr int BM = ROWG * 16;
  constexpr int DVW = DV / DVGRP;
  constexpr int DKP = DK + 8;
  constexpr int KH = KT / DVGRP;  // 每个 warp 算的 KV 半边
  constexpr int KTP = KT + 8;     // P smem 行距（+8 padding 消 ldmatrix 冲突）
  constexpr int NT = KH / 8;      // 每 warp 的 S n8 tile 数
  constexpr int NDH = DVW / 8;

  extern __shared__ bf16 smem[];
  bf16 (*qs)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem);
  bf16 (*ks)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem + BM * DKP);
  bf16 (*ps)[KTP] = reinterpret_cast<bf16(*)[KTP]>(smem + (BM + KT) * DKP);
  float* pmax = reinterpret_cast<float*>(ps + BM);
  float* psum = pmax + DVGRP * ROWG * 16;

  const int h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int rowg = w % ROWG;
  const int dvg = w / ROWG;
  const int T = WARPS * 32;

  for (int i = tid; i < BM * DC / 8; i += T) {
    const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
    const int gq = q0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qa[((size_t)h * Sq + gq) * DC + c8]);
    *reinterpret_cast<uint4*>(&qs[row][c8]) = v;
  }
  for (int i = tid; i < BM * DR / 8; i += T) {
    const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
    const int gq = q0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qr[((size_t)h * Sq + gq) * DR + c8]);
    *reinterpret_cast<uint4*>(&qs[row][DC + c8]) = v;
  }

  float S[NT][4];
  float O[NDH][4];
#pragma unroll
  for (int t = 0; t < NDH; ++t)
#pragma unroll
    for (int q = 0; q < 4; ++q) O[t][q] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r = lane >> 2;

  __syncthreads();

  for (int k0 = 0; k0 < Sk; k0 += KT) {
    for (int i = tid; i < KT * DC / 8; i += T) {
      const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
      *reinterpret_cast<uint4*>(&ks[row][c8]) =
          *reinterpret_cast<const uint4*>(&ckv[(size_t)(k0 + row) * DC + c8]);
    }
    for (int i = tid; i < KT * DR / 8; i += T) {
      const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
      *reinterpret_cast<uint4*>(&ks[row][DC + c8]) =
          *reinterpret_cast<const uint4*>(&kr[(size_t)(k0 + row) * DR + c8]);
    }
    __syncthreads();

    // ---- QK：只算 KV 的 dv 半边 ----
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q) S[t][q] = 0.f;
    const int nbase = dvg * KH;
#pragma unroll
    for (int kk = 0; kk < DK / 16; ++kk) {
      uint32_t a[4];
      ldmatrix_x4(smem_u32(&qs[rowg * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]), a);
      const int n_off = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int k_off = ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int nb = 0; nb < KH / 16; ++nb) {
        uint32_t d[4];
        ldmatrix_x4(smem_u32(&ks[nbase + nb * 16 + n_off][kk * 16 + k_off]), d);
        mma16816(S[nb * 2], a, d);
        mma16816(S[nb * 2 + 1], a, d + 2);
      }
    }

    // ---- 局部 max -> smem 交换 -> 全 max ----
    float b0 = -INFINITY, b1 = -INFINITY;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      b0 = fmaxf(b0, fmaxf(S[t][0], S[t][1]));
      b1 = fmaxf(b1, fmaxf(S[t][2], S[t][3]));
    }
    b0 = fmaxf(b0, __shfl_xor_sync(~0u, b0, 1));
    b0 = fmaxf(b0, __shfl_xor_sync(~0u, b0, 2));
    b1 = fmaxf(b1, __shfl_xor_sync(~0u, b1, 1));
    b1 = fmaxf(b1, __shfl_xor_sync(~0u, b1, 2));
    pmax[(dvg * ROWG + rowg) * 16 + r] = b0;
    pmax[(dvg * ROWG + rowg) * 16 + 8 + r] = b1;
    __syncthreads();
    float fm0 = -INFINITY, fm1 = -INFINITY;
#pragma unroll
    for (int e = 0; e < DVGRP; ++e) {
      fm0 = fmaxf(fm0, pmax[(e * ROWG + rowg) * 16 + r]);
      fm1 = fmaxf(fm1, pmax[(e * ROWG + rowg) * 16 + 8 + r]);
    }
    const float nm0 = fmaxf(m0, fm0), nm1 = fmaxf(m1, fm1);
    const float al0 = __expf(m0 - nm0), al1 = __expf(m1 - nm1);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      S[t][0] = __expf(S[t][0] - nm0);
      S[t][1] = __expf(S[t][1] - nm0);
      S[t][2] = __expf(S[t][2] - nm1);
      S[t][3] = __expf(S[t][3] - nm1);
      s0 += S[t][0] + S[t][1];
      s1 += S[t][2] + S[t][3];
    }
    s0 += __shfl_xor_sync(~0u, s0, 1);
    s0 += __shfl_xor_sync(~0u, s0, 2);
    s1 += __shfl_xor_sync(~0u, s1, 1);
    s1 += __shfl_xor_sync(~0u, s1, 2);
    psum[(dvg * ROWG + rowg) * 16 + r] = s0;
    psum[(dvg * ROWG + rowg) * 16 + 8 + r] = s1;
    // 写 P（bf16）到共享 smem：行 = 该组 16 行，列 = KV[dvg*KH ...]
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int col = dvg * KH + t * 8 + (lane & 3) * 2;
      ps[rowg * 16 + r][col] = __float2bfloat16(S[t][0]);
      ps[rowg * 16 + r][col + 1] = __float2bfloat16(S[t][1]);
      ps[rowg * 16 + 8 + r][col] = __float2bfloat16(S[t][2]);
      ps[rowg * 16 + 8 + r][col + 1] = __float2bfloat16(S[t][3]);
    }
#pragma unroll
    for (int t = 0; t < NDH; ++t) {
      O[t][0] *= al0;
      O[t][1] *= al0;
      O[t][2] *= al1;
      O[t][3] *= al1;
    }
    m0 = nm0;
    m1 = nm1;
    __syncthreads();
    float fs0 = 0.f, fs1 = 0.f;
#pragma unroll
    for (int e = 0; e < DVGRP; ++e) {
      fs0 += psum[(e * ROWG + rowg) * 16 + r];
      fs1 += psum[(e * ROWG + rowg) * 16 + 8 + r];
    }
    l0 = l0 * al0 + fs0;
    l1 = l1 * al1 + fs1;

    // ---- PV：读共享 P（ldmatrix A 片段）+ V 的 dv 半边 ----
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      uint32_t pa[4];
      ldmatrix_x4(smem_u32(&ps[rowg * 16 + (lane & 15)][c * 16 + (lane >> 4) * 8]), pa);
      const int rr = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int dn = 0; dn < DVW / 16; ++dn) {
        uint32_t d[4];
        ldmatrix_x4_trans(
            smem_u32(&ks[c * 16 + rr][dvg * DVW + dn * 16 + (lane >> 4) * 8]), d);
        mma16816(O[dn * 2], pa, d);
        mma16816(O[dn * 2 + 1], pa, d + 2);
      }
    }
    __syncthreads();
  }

  const float iv0 = 1.f / l0, iv1 = 1.f / l1;
  const int row0 = q0 + rowg * 16 + r, row1 = row0 + 8;
#pragma unroll
  for (int t = 0; t < NDH; ++t) {
    const int col = dvg * DVW + t * 8 + (lane & 3) * 2;
    if (row0 < Sq) {
      out[((size_t)h * Sq + row0) * DV + col] = O[t][0] * iv0;
      out[((size_t)h * Sq + row0) * DV + col + 1] = O[t][1] * iv0;
    }
    if (row1 < Sq) {
      out[((size_t)h * Sq + row1) * DV + col] = O[t][2] * iv1;
      out[((size_t)h * Sq + row1) * DV + col + 1] = O[t][3] * iv1;
    }
  }
}

// ===========================================================================
// host
// ===========================================================================
static void mla_row_ref(int h, int q, int Sq, int Sk, const std::vector<bf16>& qa,
                        const std::vector<bf16>& qr, const std::vector<bf16>& ckv,
                        const std::vector<bf16>& kr, std::vector<double>& out_row) {
  const size_t qoff = ((size_t)h * Sq + q);
  double mx = -1e30;
  std::vector<double> s(Sk);
  for (int k = 0; k < Sk; ++k) {
    double acc = 0;
    for (int d = 0; d < DC; ++d)
      acc += (double)__bfloat162float(qa[qoff * DC + d]) *
             (double)__bfloat162float(ckv[(size_t)k * DC + d]);
    for (int r = 0; r < DR; ++r)
      acc += (double)__bfloat162float(qr[qoff * DR + r]) *
             (double)__bfloat162float(kr[(size_t)k * DR + r]);
    s[k] = acc;
    mx = std::max(mx, acc);
  }
  double sum = 0;
  for (int k = 0; k < Sk; ++k) {
    s[k] = std::exp(s[k] - mx);
    sum += s[k];
  }
  out_row.assign(DV, 0.0);
  for (int k = 0; k < Sk; ++k) {
    const double p = s[k] / sum;
    for (int d = 0; d < DV; ++d)
      out_row[d] += p * (double)__bfloat162float(ckv[(size_t)k * DC + d]);
  }
}

template <int ROWG, int DVGRP, int KT>
static void launch_case(const char* tag, const bf16* qa, const bf16* qr, const bf16* ckv,
                        const bf16* kr, float* out, int H, int Sq, int Sk, double flops,
                        const std::vector<bf16>& hqa, const std::vector<bf16>& hqr,
                        const std::vector<bf16>& hck, const std::vector<bf16>& hkr,
                        std::vector<float>& hout, bool do_check) {
  constexpr int BM = ROWG * 16;
  constexpr int WARPS = ROWG * DVGRP;
  const size_t shm = (size_t)(BM + KT) * (DK + 8) * sizeof(bf16);
  auto fn = mla_fused_kernel<ROWG, DVGRP, KT>;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));

  dim3 grid(div_up(Sq, BM), H);
  auto run = [&] { fn<<<grid, WARPS * 32, shm>>>(qa, qr, ckv, kr, out, Sq, Sk); };
  run();
  CUDA_CHECK_LAST();

  if (do_check) {
    CUDA_CHECK(cudaMemcpy(hout.data(), out, hout.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 3; ++i) {
      const int h = (i * 61 + 7) % H, q = (i * 197 + 3) % Sq;
      std::vector<double> row;
      mla_row_ref(h, q, Sq, Sk, hqa, hqr, hck, hkr, row);
      for (int dd = 0; dd < DV; ++dd) {
        const double e = row[dd];
        err = std::max(err, std::fabs((double)hout[((size_t)h * Sq + q) * DV + dd] - e));
        ref = std::max(ref, std::fabs(e));
      }
    }
    std::printf("  [%-6s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }

  double t = bench_ms(run, 3, 10);
  const double tfl = to_tflops(flops, t);
  const int dup = DVGRP;
  std::printf("%-8s BM=%-3d DVGRP=%d KT=%-3d warps=%-2d  %8.4f ms  %8.2f TFLOPS (%5.1f%% peak, "
              "QK dup %dx -> 有效 %5.1f%%)\n",
              tag, BM, DVGRP, KT, WARPS, t, tfl, 100.0 * tfl / 989.0, dup,
              100.0 * tfl / (989.0 * dup));
}

template <int ROWG, int DVGRP, int KT>
static void launch_shared(const char* tag, const bf16* qa, const bf16* qr, const bf16* ckv,
                          const bf16* kr, float* out, int H, int Sq, int Sk, double flops,
                          const std::vector<bf16>& hqa, const std::vector<bf16>& hqr,
                          const std::vector<bf16>& hck, const std::vector<bf16>& hkr,
                          std::vector<float>& hout, bool do_check) {
  constexpr int BM = ROWG * 16;
  constexpr int WARPS = ROWG * DVGRP;
  const size_t shm = (size_t)(BM + KT) * (DK + 8) * sizeof(bf16) +
                     (size_t)BM * (KT + 8) * sizeof(bf16) +
                     2 * (size_t)DVGRP * ROWG * 16 * sizeof(float);
  auto fn = mla_shared_kernel<ROWG, DVGRP, KT>;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(div_up(Sq, BM), H);
  auto run = [&] { fn<<<grid, WARPS * 32, shm>>>(qa, qr, ckv, kr, out, Sq, Sk); };
  run();
  CUDA_CHECK_LAST();
  if (do_check) {
    CUDA_CHECK(cudaMemcpy(hout.data(), out, hout.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 3; ++i) {
      const int h = (i * 61 + 7) % H, q = (i * 197 + 3) % Sq;
      std::vector<double> row;
      mla_row_ref(h, q, Sq, Sk, hqa, hqr, hck, hkr, row);
      for (int dd = 0; dd < DV; ++dd) {
        const double e = row[dd];
        err = std::max(err, std::fabs((double)hout[((size_t)h * Sq + q) * DV + dd] - e));
        ref = std::max(ref, std::fabs(e));
      }
    }
    std::printf("  [%-6s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }
  double t = bench_ms(run, 3, 10);
  const double tfl = to_tflops(flops, t);
  std::printf("%-8s BM=%-3d DVGRP=%d KT=%-3d warps=%-2d  %8.4f ms  %8.2f TFLOPS (%5.1f%% peak, "
              "QK dup 1x)\n",
              tag, BM, DVGRP, KT, WARPS, t, tfl, 100.0 * tfl / 989.0);
}

int main(int argc, char** argv) {
  int Sq = (argc > 1) ? std::atoi(argv[1]) : 1024;
  int H = (argc > 2) ? std::atoi(argv[2]) : 128;
  int Sk = (argc > 3) ? std::atoi(argv[3]) : 1024;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);

  const size_t qaN = (size_t)H * Sq * DC, qrN = (size_t)H * Sq * DR;
  const size_t ckN = (size_t)Sk * DC, krN = (size_t)Sk * DR, outN = (size_t)H * Sq * DV;
  const double flops = 2.0 * Sq * H * Sk * (DC + DR + DV);

  std::printf("\nMLA fused attention (MQA): H=%d Sq=%d Sk=%d  DC=%d DR=%d DV=%d\n", H, Sq, Sk, DC,
              DR, DV);
  std::printf("FLOPs = %.2f GFLOP   (KV tile smem 需 Sq%%BM==0, Sk%%KT==0)\n\n", flops / 1e9);

  bf16 *qa, *qr, *ckv, *kr;
  float* out;
  CUDA_CHECK(cudaMalloc(&qa, qaN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&qr, qrN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&ckv, ckN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&kr, krN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&out, outN * sizeof(float)));

  std::vector<bf16> hqa(qaN), hqr(qrN), hck(ckN), hkr(krN);
  std::vector<float> hout(outN);
  srand(1234);
  auto rnd = [] { return __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (size_t i = 0; i < qaN; ++i) hqa[i] = rnd();
  for (size_t i = 0; i < qrN; ++i) hqr[i] = rnd();
  for (size_t i = 0; i < ckN; ++i) hck[i] = rnd();
  for (size_t i = 0; i < krN; ++i) hkr[i] = rnd();
  CUDA_CHECK(cudaMemcpy(qa, hqa.data(), qaN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(qr, hqr.data(), qrN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ckv, hck.data(), ckN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(kr, hkr.data(), krN * sizeof(bf16), cudaMemcpyHostToDevice));

  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };

  // 变体：ROWG x DVGRP x KT
  if (want("f4")) launch_case<4, 2, 64>("f4", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr,
                                        hck, hkr, hout, true);
  if (want("f2")) launch_case<2, 4, 64>("f2", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr,
                                        hck, hkr, hout, true);
  if (want("f2k32"))
    launch_case<2, 4, 32>("f2k32", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr, hck, hkr,
                          hout, true);
  if (want("f4s"))
    launch_shared<4, 2, 64>("f4s", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr, hck, hkr,
                            hout, true);
  if (want("f2s"))
    launch_shared<2, 4, 64>("f2s", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr, hck, hkr,
                            hout, true);
  if (want("f4k32"))
    launch_case<4, 2, 32>("f4k32", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr, hck, hkr,
                          hout, true);
  if (want("f1")) launch_case<1, 8, 64>("f1", qa, qr, ckv, kr, out, H, Sq, Sk, flops, hqa, hqr,
                                        hck, hkr, hout, true);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
