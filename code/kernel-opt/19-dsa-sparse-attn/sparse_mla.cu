// 19 DSA 稀疏注意力（二）：稀疏 MLA 消费端（gather top-k + online softmax）
//
// 承接 18 篇：18 篇把 DSA 的「打分」（lightning indexer）与「精确 top-k」
// （radix-select）做出来了，稀疏 attention 只是按稠密吞吐折算的估算。
// 本篇把 top-k 索引真正接进来，写一个稀疏 MLA prefill kernel：
//
//   对每个 query 位置 t（prefill 里新增的 token），它有一份自己的 top-k 索引
//   idx[t][0..k)，只对这 k 个 key 做 MLA。所有 H 个 head 共享同一份 idx[t]
//   ——这正是能省的根源：gather 一次，喂给一个 head block。
//
// 数据流（一个 CTA = 一个 query token t × 一个 BH head 的 block）：
//
//   Q[BH][576]  ──┐
//                 ├─► QK^T (mma) ─► mask(-inf) ─► online softmax ─► PV (mma) ─► O[BH][512]
//   gather(idx[t])┘
//     c_kv[idx][512] + k_rope[idx][64] -> smem tile (KT 个 key)
//
// 与 16 篇 f4s 单 kernel 融合相比，只有两处新东西：
//   * KV tile 不再来自连续区间 [k0, k0+KT)，而是按 idx[t] gather；每行 1KB，
//     warp 内每个线程读同一行的连续 16B，所以行内仍然合并。
//   * 每个 key 有一个 valid 标志（causal 不足 k 个时由 -1 补齐），QK 后把无效列
//     置 -inf —— 这就是 FlashMLA sparse prefill 里 `is_kv_valid` 掩码的作用。
//
// 运行：scripts/run.sh 19-dsa-sparse-attn/sparse_mla.cu [Sq] [Sk] [K] [which]
//   which = all | s1 | s2 | s3 | dense
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

// DeepSeek-V4-Pro MLA「吸收」维度（FlashMLA 的 MQA 口径：576/512）
constexpr int DC = 512;  // kv_lora_rank / NoPE
constexpr int DR = 64;   // decoupled RoPE
constexpr int DV = 512;  // 吸收后 v 维度 = kv_lora_rank
constexpr int DK = DC + DR;

// ---------------------------------------------------------------------------
// PTX helpers（与 16/18 篇同一套）
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
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

// ===========================================================================
// 稀疏 MLA 融合 kernel（在 16 篇 mla_shared_kernel 基础上改造）
//   ROWG  : 16-row 组个数（M = BH = 16*ROWG，这里是 **head** 维）
//   DVGRP : DV 被切成几份（每份 DVW = DV/DVGRP）；也决定 QK 列对半分工
//   KT    : 每次 gather 的 key 数
// ===========================================================================
template <int ROWG, int DVGRP, int KT>
__global__ void __launch_bounds__(ROWG * DVGRP * 32) sparse_mla_kernel(
    const bf16* __restrict__ qa,      // [H][Sq][DC]
    const bf16* __restrict__ qr,      // [H][Sq][DR]
    const bf16* __restrict__ ckv,     // [Sk][DC]
    const bf16* __restrict__ kr,      // [Sk][DR]
    const int* __restrict__ topk_idx, // [Sq][K]  (-1 = 无效)
    const int* __restrict__ topk_len, // [Sq]
    float* __restrict__ out,          // [H][Sq][DV]
    int H, int Sq, int Sk, int K) {
  constexpr int WARPS = ROWG * DVGRP;
  constexpr int BH = ROWG * 16;
  constexpr int DVW = DV / DVGRP;
  constexpr int DKP = DK + 8;
  constexpr int KH = KT / DVGRP;  // 每个 warp 算的 KV 半边
  constexpr int KTP = KT + 8;
  constexpr int NT = KH / 8;
  constexpr int NDH = DVW / 8;

  extern __shared__ bf16 smem[];
  bf16 (*qs)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem);
  bf16 (*ks)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem + BH * DKP);
  bf16 (*ps)[KTP] = reinterpret_cast<bf16(*)[KTP]>(smem + (BH + KT) * DKP);
  float* pmax = reinterpret_cast<float*>(ps + BH);
  float* psum = pmax + DVGRP * ROWG * 16;
  int* kvflag = reinterpret_cast<int*>(psum + DVGRP * ROWG * 16);  // [KT]

  const int t = blockIdx.y;              // query token
  const int head0 = blockIdx.x * BH;     // head block
  const int valid = topk_len[t];         // 该 query 的 top-k 有效个数
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int rowg = w % ROWG;
  const int dvg = w / ROWG;
  const int T = WARPS * 32;
  const int r = lane >> 2;

  // ---- 载入 Q = [q_abs | q_rope]（BH 个 head，同一个 token t）----
  for (int i = tid; i < BH * DC / 8; i += T) {
    const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
    const int ghead = head0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ghead < H) v = *reinterpret_cast<const uint4*>(&qa[((size_t)ghead * Sq + t) * DC + c8]);
    *reinterpret_cast<uint4*>(&qs[row][c8]) = v;
  }
  for (int i = tid; i < BH * DR / 8; i += T) {
    const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
    const int ghead = head0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ghead < H) v = *reinterpret_cast<const uint4*>(&qr[((size_t)ghead * Sq + t) * DR + c8]);
    *reinterpret_cast<uint4*>(&qs[row][DC + c8]) = v;
  }

  float S[NT][4];
  float O[NDH][4];
#pragma unroll
  for (int i = 0; i < NDH; ++i)
#pragma unroll
    for (int q = 0; q < 4; ++q) O[i][q] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  __syncthreads();

  for (int k0 = 0; k0 < valid; k0 += KT) {
    // ---- gather KV tile：按 idx 从 global 抓 KT 个 key 的 [c_kv | k_rope] ----
    // 行内连续线程读同一行的连续 16B -> 行内合并；行间随机但每行 1KB 用满。
    // idx 每 16B 重读 global（同一行的线程同址 -> 广播、L1 命中），
    // 顺带写 kvflag[row]（每行冗余写 ~DC/8 次，仅 smem 写，便宜）。
    const bool full = (k0 + KT <= valid);
    for (int i = tid; i < KT * DC / 8; i += T) {
      const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
      const int j = k0 + row;
      const int idx = (j < valid) ? topk_idx[(size_t)t * K + j] : -1;
      kvflag[row] = (idx >= 0 && idx < Sk) ? idx : -1;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (idx >= 0 && idx < Sk) v = *reinterpret_cast<const uint4*>(&ckv[(size_t)idx * DC + c8]);
      *reinterpret_cast<uint4*>(&ks[row][c8]) = v;
    }
    for (int i = tid; i < KT * DR / 8; i += T) {
      const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
      const int j = k0 + row;
      const int idx = (j < valid) ? topk_idx[(size_t)t * K + j] : -1;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (idx >= 0 && idx < Sk) v = *reinterpret_cast<const uint4*>(&kr[(size_t)idx * DR + c8]);
      *reinterpret_cast<uint4*>(&ks[row][DC + c8]) = v;
    }
    __syncthreads();

    // ---- QK^T：只算 KV 的 dv 半边 ----
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int q = 0; q < 4; ++q) S[i][q] = 0.f;
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

    // ---- mask：无效 key（idx<0）的列置 -inf ----
    // 非尾部 tile 全部有效，跳过（省 smem load + 分支）。
    if (!full) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int c0 = nbase + i * 8 + (lane & 3) * 2;
        if (kvflag[c0] < 0) {
          S[i][0] = -INFINITY;
          S[i][2] = -INFINITY;
        }
        if (kvflag[c0 + 1] < 0) {
          S[i][1] = -INFINITY;
          S[i][3] = -INFINITY;
        }
      }
    }

    // ---- 局部 max -> smem 交换 -> 全 max（跨 DVGRP 个 warp）----
    float b0 = -INFINITY, b1 = -INFINITY;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      b0 = fmaxf(b0, fmaxf(S[i][0], S[i][1]));
      b1 = fmaxf(b1, fmaxf(S[i][2], S[i][3]));
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
    for (int i = 0; i < NT; ++i) {
      S[i][0] = __expf(S[i][0] - nm0);
      S[i][1] = __expf(S[i][1] - nm0);
      S[i][2] = __expf(S[i][2] - nm1);
      S[i][3] = __expf(S[i][3] - nm1);
      s0 += S[i][0] + S[i][1];
      s1 += S[i][2] + S[i][3];
    }
    s0 += __shfl_xor_sync(~0u, s0, 1);
    s0 += __shfl_xor_sync(~0u, s0, 2);
    s1 += __shfl_xor_sync(~0u, s1, 1);
    s1 += __shfl_xor_sync(~0u, s1, 2);
    psum[(dvg * ROWG + rowg) * 16 + r] = s0;
    psum[(dvg * ROWG + rowg) * 16 + 8 + r] = s1;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int col = nbase + i * 8 + (lane & 3) * 2;
      ps[rowg * 16 + r][col] = __float2bfloat16(S[i][0]);
      ps[rowg * 16 + r][col + 1] = __float2bfloat16(S[i][1]);
      ps[rowg * 16 + 8 + r][col] = __float2bfloat16(S[i][2]);
      ps[rowg * 16 + 8 + r][col + 1] = __float2bfloat16(S[i][3]);
    }
#pragma unroll
    for (int i = 0; i < NDH; ++i) {
      O[i][0] *= al0;
      O[i][1] *= al0;
      O[i][2] *= al1;
      O[i][3] *= al1;
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

    // ---- PV：读共享 P + V 的 dv 半边（V = c_kv）----
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      uint32_t pa[4];
      ldmatrix_x4(smem_u32(&ps[rowg * 16 + (lane & 15)][c * 16 + (lane >> 4) * 8]), pa);
      const int rr = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int dn = 0; dn < DVW / 16; ++dn) {
        uint32_t d[4];
        ldmatrix_x4_trans(smem_u32(&ks[c * 16 + rr][dvg * DVW + dn * 16 + (lane >> 4) * 8]), d);
        mma16816(O[dn * 2], pa, d);
        mma16816(O[dn * 2 + 1], pa, d + 2);
      }
    }
    __syncthreads();
  }

  // ---- 输出 O / l ----
  const float iv0 = 1.f / l0, iv1 = 1.f / l1;
  const int row0 = head0 + rowg * 16 + r, row1 = row0 + 8;
#pragma unroll
  for (int i = 0; i < NDH; ++i) {
    const int col = dvg * DVW + i * 8 + (lane & 3) * 2;
    if (row0 < H) {
      out[((size_t)row0 * Sq + t) * DV + col] = O[i][0] * iv0;
      out[((size_t)row0 * Sq + t) * DV + col + 1] = O[i][1] * iv0;
    }
    if (row1 < H) {
      out[((size_t)row1 * Sq + t) * DV + col] = O[i][2] * iv1;
      out[((size_t)row1 * Sq + t) * DV + col + 1] = O[i][3] * iv1;
    }
  }
}

// ===========================================================================
// 双缓冲 cp.async 流水版：把 gather 的 global 延迟藏起来
//
// 上面那版每个 tile 都是「同步加载 -> __syncthreads -> 算」，8/16 个 warp 时
// global 延迟直接暴露（ncu: long_scoreboard 4.15）。这里用 cp.async 预取下一个
// KV tile 到 ping-pong 缓冲，当前 tile 计算与下一 tile 的 gather 重叠。
//
// 掩码不再需要 kvflag：只存 valid 个索引，列 j 无效当且仅当 k0+j >= valid。
// ===========================================================================
template <int ROWG, int DVGRP, int KT>
__global__ void __launch_bounds__(ROWG * DVGRP * 32) sparse_mla_pipe_kernel(
    const bf16* __restrict__ qa, const bf16* __restrict__ qr, const bf16* __restrict__ ckv,
    const bf16* __restrict__ kr, const int* __restrict__ topk_idx, const int* __restrict__ topk_len,
    float* __restrict__ out, int H, int Sq, int Sk, int K) {
  constexpr int WARPS = ROWG * DVGRP;
  constexpr int BH = ROWG * 16;
  constexpr int DVW = DV / DVGRP;
  constexpr int DKP = DK + 8;
  constexpr int KH = KT / DVGRP;
  constexpr int KTP = KT + 8;
  constexpr int NT = KH / 8;
  constexpr int NDH = DVW / 8;

  extern __shared__ bf16 smem[];
  bf16 (*qs)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem);
  bf16 (*ks)[DKP] = reinterpret_cast<bf16(*)[DKP]>(smem + BH * DKP);  // 2*KT 行
  bf16 (*ps)[KTP] = reinterpret_cast<bf16(*)[KTP]>(smem + (BH + 2 * KT) * DKP);
  float* pmax = reinterpret_cast<float*>(ps + BH);
  float* psum = pmax + DVGRP * ROWG * 16;

  const int t = blockIdx.y;
  const int head0 = blockIdx.x * BH;
  const int valid = topk_len[t];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int rowg = w % ROWG;
  const int dvg = w / ROWG;
  const int T = WARPS * 32;
  const int r = lane >> 2;
  const int nbase = dvg * KH;

  for (int i = tid; i < BH * DC / 8; i += T) {
    const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
    const int ghead = head0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ghead < H) v = *reinterpret_cast<const uint4*>(&qa[((size_t)ghead * Sq + t) * DC + c8]);
    *reinterpret_cast<uint4*>(&qs[row][c8]) = v;
  }
  for (int i = tid; i < BH * DR / 8; i += T) {
    const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
    const int ghead = head0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ghead < H) v = *reinterpret_cast<const uint4*>(&qr[((size_t)ghead * Sq + t) * DR + c8]);
    *reinterpret_cast<uint4*>(&qs[row][DC + c8]) = v;
  }

  // 发一个 tile 的 gather（KT 个 key 的 c_kv + k_rope）到指定 stage
  auto issue = [&](int k0, int stage) {
    bf16 (*buf)[DKP] = ks + stage * KT;
    for (int i = tid; i < KT * DC / 8; i += T) {
      const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
      const int j = k0 + row;
      if (j < valid) {
        const int idx = topk_idx[(size_t)t * K + j];
        __pipeline_memcpy_async(&buf[row][c8], &ckv[(size_t)idx * DC + c8], 16);
      } else {
        *reinterpret_cast<uint4*>(&buf[row][c8]) = make_uint4(0, 0, 0, 0);
      }
    }
    for (int i = tid; i < KT * DR / 8; i += T) {
      const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
      const int j = k0 + row;
      if (j < valid) {
        const int idx = topk_idx[(size_t)t * K + j];
        __pipeline_memcpy_async(&buf[row][DC + c8], &kr[(size_t)idx * DR + c8], 16);
      } else {
        *reinterpret_cast<uint4*>(&buf[row][DC + c8]) = make_uint4(0, 0, 0, 0);
      }
    }
    __pipeline_commit();
  };

  float S[NT][4];
  float O[NDH][4];
#pragma unroll
  for (int i = 0; i < NDH; ++i)
#pragma unroll
    for (int q = 0; q < 4; ++q) O[i][q] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  __syncthreads();

  if (valid > 0) issue(0, 0);

  int stage = 0;
  for (int k0 = 0; k0 < valid; k0 += KT, stage ^= 1) {
    const bool has_next = (k0 + KT < valid);
    if (has_next) issue(k0 + KT, stage ^ 1);
    __pipeline_wait_prior(has_next ? 1 : 0);
    __syncthreads();
    bf16 (*kbuf)[DKP] = ks + stage * KT;

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int q = 0; q < 4; ++q) S[i][q] = 0.f;
#pragma unroll
    for (int kk = 0; kk < DK / 16; ++kk) {
      uint32_t a[4];
      ldmatrix_x4(smem_u32(&qs[rowg * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]), a);
      const int n_off = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int k_off = ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int nb = 0; nb < KH / 16; ++nb) {
        uint32_t d[4];
        ldmatrix_x4(smem_u32(&kbuf[nbase + nb * 16 + n_off][kk * 16 + k_off]), d);
        mma16816(S[nb * 2], a, d);
        mma16816(S[nb * 2 + 1], a, d + 2);
      }
    }

    if (k0 + KT > valid) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int c0 = k0 + nbase + i * 8 + (lane & 3) * 2;
        if (c0 >= valid) {
          S[i][0] = -INFINITY;
          S[i][2] = -INFINITY;
        }
        if (c0 + 1 >= valid) {
          S[i][1] = -INFINITY;
          S[i][3] = -INFINITY;
        }
      }
    }

    float b0 = -INFINITY, b1 = -INFINITY;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      b0 = fmaxf(b0, fmaxf(S[i][0], S[i][1]));
      b1 = fmaxf(b1, fmaxf(S[i][2], S[i][3]));
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
    for (int i = 0; i < NT; ++i) {
      S[i][0] = __expf(S[i][0] - nm0);
      S[i][1] = __expf(S[i][1] - nm0);
      S[i][2] = __expf(S[i][2] - nm1);
      S[i][3] = __expf(S[i][3] - nm1);
      s0 += S[i][0] + S[i][1];
      s1 += S[i][2] + S[i][3];
    }
    s0 += __shfl_xor_sync(~0u, s0, 1);
    s0 += __shfl_xor_sync(~0u, s0, 2);
    s1 += __shfl_xor_sync(~0u, s1, 1);
    s1 += __shfl_xor_sync(~0u, s1, 2);
    psum[(dvg * ROWG + rowg) * 16 + r] = s0;
    psum[(dvg * ROWG + rowg) * 16 + 8 + r] = s1;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int col = nbase + i * 8 + (lane & 3) * 2;
      ps[rowg * 16 + r][col] = __float2bfloat16(S[i][0]);
      ps[rowg * 16 + r][col + 1] = __float2bfloat16(S[i][1]);
      ps[rowg * 16 + 8 + r][col] = __float2bfloat16(S[i][2]);
      ps[rowg * 16 + 8 + r][col + 1] = __float2bfloat16(S[i][3]);
    }
#pragma unroll
    for (int i = 0; i < NDH; ++i) {
      O[i][0] *= al0;
      O[i][1] *= al0;
      O[i][2] *= al1;
      O[i][3] *= al1;
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

#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      uint32_t pa[4];
      ldmatrix_x4(smem_u32(&ps[rowg * 16 + (lane & 15)][c * 16 + (lane >> 4) * 8]), pa);
      const int rr = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int dn = 0; dn < DVW / 16; ++dn) {
        uint32_t d[4];
        ldmatrix_x4_trans(smem_u32(&kbuf[c * 16 + rr][dvg * DVW + dn * 16 + (lane >> 4) * 8]), d);
        mma16816(O[dn * 2], pa, d);
        mma16816(O[dn * 2 + 1], pa, d + 2);
      }
    }
    __syncthreads();
  }

  const float iv0 = (l0 > 0.f) ? 1.f / l0 : 0.f, iv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  const int row0 = head0 + rowg * 16 + r, row1 = row0 + 8;
#pragma unroll
  for (int i = 0; i < NDH; ++i) {
    const int col = dvg * DVW + i * 8 + (lane & 3) * 2;
    if (row0 < H) {
      out[((size_t)row0 * Sq + t) * DV + col] = O[i][0] * iv0;
      out[((size_t)row0 * Sq + t) * DV + col + 1] = O[i][1] * iv0;
    }
    if (row1 < H) {
      out[((size_t)row1 * Sq + t) * DV + col] = O[i][2] * iv1;
      out[((size_t)row1 * Sq + t) * DV + col + 1] = O[i][3] * iv1;
    }
  }
}

// ===========================================================================
// host 参考：对单个 (head, token) 精确算稀疏 attention
// ===========================================================================
static void sparse_row_ref(int h, int t, int Sq, int Sk, const std::vector<bf16>& qa,
                           const std::vector<bf16>& qr, const std::vector<bf16>& ckv,
                           const std::vector<bf16>& kr, const int* idx, int valid,
                           std::vector<double>& out_row) {
  const size_t qoff = (size_t)h * Sq + t;
  double mx = -1e30;
  std::vector<double> s(valid);
  for (int j = 0; j < valid; ++j) {
    const int kx = idx[j];
    double acc = 0;
    for (int d = 0; d < DC; ++d)
      acc += (double)__bfloat162float(qa[qoff * DC + d]) *
             (double)__bfloat162float(ckv[(size_t)kx * DC + d]);
    for (int d = 0; d < DR; ++d)
      acc += (double)__bfloat162float(qr[qoff * DR + d]) *
             (double)__bfloat162float(kr[(size_t)kx * DR + d]);
    s[j] = acc;
    mx = std::max(mx, acc);
  }
  double sum = 0;
  for (int j = 0; j < valid; ++j) {
    s[j] = std::exp(s[j] - mx);
    sum += s[j];
  }
  out_row.assign(DV, 0.0);
  for (int j = 0; j < valid; ++j) {
    const double p = s[j] / sum;
    const int kx = idx[j];
    for (int d = 0; d < DV; ++d) out_row[d] += p * (double)__bfloat162float(ckv[(size_t)kx * DC + d]);
  }
}

// ===========================================================================
// 生成 top-k 索引：query token t 的全局位置 p = Sk-Sq+t，causal 可见 [0, p]。
// valid = min(K, p+1)；从 [0, p] 里无重复抽样 valid 个，其余填 -1。
// ===========================================================================
static void gen_topk(std::vector<int>& idx, std::vector<int>& len, int Sq, int Sk, int K,
                     unsigned seed) {
  idx.assign((size_t)Sq * K, -1);
  len.assign(Sq, 0);
  std::mt19937 rng(seed);
  for (int t = 0; t < Sq; ++t) {
    const long p = (long)Sk - Sq + t;
    const int v = (int)std::min<long>(K, p + 1);
    len[t] = v;
    std::uniform_int_distribution<int> dist(0, (int)p);
    // 稀疏抽样：随机落点，重复重抽（v << p 时够快）
    std::vector<char> seen((size_t)p + 1, 0);
    int got = 0;
    while (got < v) {
      int cand = dist(rng);
      if (!seen[cand]) {
        seen[cand] = 1;
        idx[(size_t)t * K + got] = cand;
        ++got;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// device fill
// ---------------------------------------------------------------------------
__global__ void fill_bf16_kernel(bf16* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = (unsigned)(i * 2654435761u + seed);
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  p[i] = __float2bfloat16((float)(h & 0xffff) / 65536.f - 0.5f);
}

template <int ROWG, int DVGRP, int KT>
static void launch_sparse(const char* tag, const bf16* qa, const bf16* qr, const bf16* ckv,
                          const bf16* kr, const int* idx, const int* len, float* out, int H,
                          int Sq, int Sk, int K, double flops, const std::vector<bf16>& hqa,
                          const std::vector<bf16>& hqr, const std::vector<bf16>& hck,
                          const std::vector<bf16>& hkr, const std::vector<int>& hidx,
                          const std::vector<int>& hlen, std::vector<float>& hout, bool do_check) {
  constexpr int BH = ROWG * 16;
  constexpr int WARPS = ROWG * DVGRP;
  const size_t shm = (size_t)(BH + KT) * (DK + 8) * sizeof(bf16) +
                     (size_t)BH * (KT + 8) * sizeof(bf16) +
                     2 * (size_t)DVGRP * ROWG * 16 * sizeof(float) + (size_t)KT * sizeof(int);
  auto fn = sparse_mla_kernel<ROWG, DVGRP, KT>;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(div_up(H, BH), Sq);
  auto run = [&] { fn<<<grid, WARPS * 32, shm>>>(qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K); };
  run();
  CUDA_CHECK_LAST();

  if (do_check) {
    CUDA_CHECK(cudaMemcpy(hout.data(), out, hout.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 3; ++i) {
      const int h = (i * 61 + 7) % H;
      const int t = (i * 197 + 3) % Sq;
      std::vector<double> row;
      sparse_row_ref(h, t, Sq, Sk, hqa, hqr, hck, hkr, &hidx[(size_t)t * K], hlen[t], row);
      for (int dd = 0; dd < DV; ++dd) {
        const double e = row[dd];
        err = std::max(err, std::fabs((double)hout[((size_t)h * Sq + t) * DV + dd] - e));
        ref = std::max(ref, std::fabs(e));
      }
    }
    std::printf("  [%-6s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }

  double time = bench_ms(run, 3, 20);
  const double tfl = to_tflops(flops, time);
  std::printf("%-8s BH=%-3d DVGRP=%d KT=%-3d warps=%-2d  %9.4f ms  %8.2f TFLOPS (%5.1f%% peak)\n",
              tag, BH, DVGRP, KT, WARPS, time, tfl, 100.0 * tfl / 989.0);
}

template <int ROWG, int DVGRP, int KT>
static void launch_pipe(const char* tag, const bf16* qa, const bf16* qr, const bf16* ckv,
                        const bf16* kr, const int* idx, const int* len, float* out, int H, int Sq,
                        int Sk, int K, double flops, const std::vector<bf16>& hqa,
                        const std::vector<bf16>& hqr, const std::vector<bf16>& hck,
                        const std::vector<bf16>& hkr, const std::vector<int>& hidx,
                        const std::vector<int>& hlen, std::vector<float>& hout, bool do_check) {
  constexpr int BH = ROWG * 16;
  constexpr int WARPS = ROWG * DVGRP;
  const size_t shm = (size_t)(BH + 2 * KT) * (DK + 8) * sizeof(bf16) +
                     (size_t)BH * (KT + 8) * sizeof(bf16) +
                     2 * (size_t)DVGRP * ROWG * 16 * sizeof(float);
  auto fn = sparse_mla_pipe_kernel<ROWG, DVGRP, KT>;
  if (shm > (size_t)227 * 1024) {
    std::printf("%-8s skipped (smem %zu B > 227KB)\n", tag, shm);
    return;
  }
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(div_up(H, BH), Sq);
  auto run = [&] { fn<<<grid, WARPS * 32, shm>>>(qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K); };
  run();
  CUDA_CHECK_LAST();

  if (do_check) {
    CUDA_CHECK(cudaMemcpy(hout.data(), out, hout.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 3; ++i) {
      const int h = (i * 61 + 7) % H;
      const int t = (i * 197 + 3) % Sq;
      std::vector<double> row;
      sparse_row_ref(h, t, Sq, Sk, hqa, hqr, hck, hkr, &hidx[(size_t)t * K], hlen[t], row);
      for (int dd = 0; dd < DV; ++dd) {
        const double e = row[dd];
        err = std::max(err, std::fabs((double)hout[((size_t)h * Sq + t) * DV + dd] - e));
        ref = std::max(ref, std::fabs(e));
      }
    }
    std::printf("  [%-6s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }

  double time = bench_ms(run, 3, 20);
  const double tfl = to_tflops(flops, time);
  std::printf("%-8s BH=%-3d DVGRP=%d KT=%-3d warps=%-2d  %9.4f ms  %8.2f TFLOPS (%5.1f%% peak)\n",
              tag, BH, DVGRP, KT, WARPS, time, tfl, 100.0 * tfl / 989.0);
}

int main(int argc, char** argv) {
  int Sq = (argc > 1) ? std::atoi(argv[1]) : 1024;
  int Sk = (argc > 2) ? std::atoi(argv[2]) : 32768;
  int K = (argc > 3) ? std::atoi(argv[3]) : 1024;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);

  constexpr int H = 128;
  const size_t qaN = (size_t)H * Sq * DC, qrN = (size_t)H * Sq * DR;
  const size_t ckN = (size_t)Sk * DC, krN = (size_t)Sk * DR, outN = (size_t)H * Sq * DV;

  std::vector<int> hidx, hlen;
  gen_topk(hidx, hlen, Sq, Sk, K, 20260921);
  long sum_valid = 0;
  for (int t = 0; t < Sq; ++t) sum_valid += hlen[t];
  const double flops = 2.0 * H * (DK + DV) * (double)sum_valid;
  const double dense_flops = 2.0 * H * (DK + DV) *
                             ((double)Sq * Sk - (double)Sq * (Sq - 1) / 2.0);

  std::printf("\nDSA sparse MLA consumer: H=%d Sq=%d Sk=%d topk=%d  (DC=%d DR=%d DV=%d)\n", H, Sq,
              Sk, K, DC, DR, DV);
  std::printf("sparse FLOPs = %.3f TFLOP (sum valid=%ld, avg %.0f) ; dense causal = %.3f TFLOP "
              "(ratio %.1fx)\n\n",
              flops / 1e12, sum_valid, (double)sum_valid / Sq, dense_flops / 1e12,
              dense_flops / flops);

  bf16 *qa, *qr, *ckv, *kr;
  float* out;
  int *idx, *len;
  CUDA_CHECK(cudaMalloc(&qa, qaN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&qr, qrN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&ckv, ckN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&kr, krN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&out, outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&idx, (size_t)Sq * K * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&len, Sq * sizeof(int)));

  std::vector<bf16> hqa(qaN), hqr(qrN), hck(ckN), hkr(krN);
  std::vector<float> hout(outN);
  {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : hqa) v = __float2bfloat16(0.2f * dist(rng));
    for (auto& v : hqr) v = __float2bfloat16(0.2f * dist(rng));
    for (auto& v : hck) v = __float2bfloat16(0.2f * dist(rng));
    for (auto& v : hkr) v = __float2bfloat16(0.2f * dist(rng));
  }
  fill_bf16_kernel<<<div_up((int)((qaN + 255) / 256), 1), 256>>>(qa, qaN, 5);
  fill_bf16_kernel<<<div_up((int)((qrN + 255) / 256), 1), 256>>>(qr, qrN, 6);
  fill_bf16_kernel<<<div_up((int)((ckN + 255) / 256), 1), 256>>>(ckv, ckN, 7);
  fill_bf16_kernel<<<div_up((int)((krN + 255) / 256), 1), 256>>>(kr, krN, 8);
  CUDA_CHECK(cudaMemcpy(idx, hidx.data(), hidx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(len, hlen.data(), hlen.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_LAST();

  // 参考用 host 数据（与 device fill 不同源，故从 device 拷回，保证对拍一致）
  CUDA_CHECK(cudaMemcpy(hqa.data(), qa, qaN * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hqr.data(), qr, qrN * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hck.data(), ckv, ckN * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hkr.data(), kr, krN * sizeof(bf16), cudaMemcpyDeviceToHost));

  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };

  if (want("s1"))
    launch_sparse<4, 2, 64>("s1", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                            hck, hkr, hidx, hlen, hout, true);
  if (want("s2"))
    launch_sparse<4, 2, 32>("s2", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                            hck, hkr, hidx, hlen, hout, true);
  if (want("s3"))
    launch_sparse<2, 4, 64>("s3", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                            hck, hkr, hidx, hlen, hout, true);
  if (want("s4"))
    launch_sparse<4, 4, 64>("s4", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                            hck, hkr, hidx, hlen, hout, true);
  if (want("p1"))
    launch_pipe<4, 2, 64>("p1", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                          hck, hkr, hidx, hlen, hout, true);
  if (want("p2"))
    launch_pipe<4, 2, 32>("p2", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                          hck, hkr, hidx, hlen, hout, true);
  if (want("p4"))
    launch_pipe<4, 4, 64>("p4", qa, qr, ckv, kr, idx, len, out, H, Sq, Sk, K, flops, hqa, hqr,
                          hck, hkr, hidx, hlen, hout, true);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(idx));
  CUDA_CHECK(cudaFree(len));
  return 0;
}
