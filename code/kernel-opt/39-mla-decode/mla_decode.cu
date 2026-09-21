// 39 MLA 吸收式 decode + FP8 KV-cache（主题 25b）
//
// 推理 decode 阶段：q_len = 1，KV 是 MLA「吸收」后的单头潜向量：
//   c_kv : [B][Lmax][DC]   (kv_lora_rank / NoPE, DC=512)
//   k_pe : [B][Lmax][DR]   (decoupled RoPE, DR=64)
// 所有 H=128 个 query head 共享同一份潜 KV（MQA），Q 也被吸收成
//   q_nope : [H][B][DC]   q_pe : [H][B][DR]
//   score(h,j) = q_nope_h · c_kv_j + q_pe_h · k_pe_j        （DK = DC+DR = 576）
//   O_h        = sum_j softmax(score) * c_kv_j              （V 维度 = DC = 512）
// 真实 shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json（num_attention_heads=128,
// head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1）。
//
// decode 是「算力/带宽平衡」算子：FLOPs = 2*H*(DK+DV)*B*L，字节 = B*L*(DK)*2。
// AI = H*(DK+DV)/DK ≈ 241，略低于 bf16 tensor core ridge（~295）——偏带宽；
// 把 KV 换成 FP8（e4m3）后字节减半，若能维持算力就近似 2×。
//
// 变体：
//   v1  bf16 mma            ：基础张量核版（一 CTA = 一个 (batch, head block)）
//   v2  bf16 mma + cp.async ：双缓冲把 KV 的 global 延迟藏起来
//   v3  FP8 KV + cp.async   ：KV 存 e4m3 + per-channel scale，装载时反量化；
//                             scale 折进 Q（score）和输出（O），反量化只剩转换
//
// 运行：scripts/run.sh 39-mla-decode/mla_decode.cu [B] [L] [which]
//   which = all | v1 | v2 | v3 | scan
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

// DeepSeek-V4-Pro MLA 吸收维度（FlashMLA 的 MQA 口径）
constexpr int DC = 512;  // kv_lora_rank / NoPE
constexpr int DR = 64;   // decoupled RoPE
constexpr int DV = 512;  // 吸收后 v 维度 = kv_lora_rank（K/V 同一张张量）
constexpr int DK = DC + DR;
constexpr int H = 128;  // num_attention_heads

// ---------------------------------------------------------------------------
// PTX helpers（与 16/19 篇同一套）
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

// 8 个 e4m3 -> 8 个 bf16（无 scale；scale 已折进 Q / 输出）
__device__ __forceinline__ uint4 dequant8_e4m3(uint2 packed) {
  uint4 r;
  bf16* p = reinterpret_cast<bf16*>(&r);
  const unsigned char* b = reinterpret_cast<const unsigned char*>(&packed);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b[i], __NV_E4M3);
    p[i] = __float2bfloat16(__half2float(*reinterpret_cast<__half*>(&hr)));
  }
  return r;
}

// ===========================================================================
// MLA decode 融合 kernel
//   ROWG  : head 方向 16-row 组个数（BH = ROWG*16 个 head / CTA）
//   DVGRP : KV 列对半分工数（每 warp 算 KH = KT/DVGRP 个 key；PV 里算 DVW=DV/DVGRP 个 V 维）
//   KT    : 每轮 KV tile 的 key 数
//   FP8   : KV 是否 e4m3（装载时反量化；scale 已折进 Q 与输出）
//   PIPE  : 是否用 cp.async 双缓冲
//   PAGED : KV 是否分页（page size PS，block_table[b][blk]）
// ===========================================================================
template <int ROWG, int DVGRP, int KT, bool FP8, bool PIPE, bool PAGED>
__global__ void __launch_bounds__(ROWG* DVGRP * 32) mla_decode_kernel(
    const bf16* __restrict__ qa,        // [H][B][DC]
    const bf16* __restrict__ qr,        // [H][B][DR]
    const void* __restrict__ ckv,       // [B][Lmax][DC]  (bf16 或 uint8)
    const void* __restrict__ kr,        // [B][Lmax][DR]
    const int* __restrict__ block_table,// PAGED 时 [B][maxb]
    const int* __restrict__ seqlen,     // [B]
    const float* __restrict__ vscale,   // FP8: [DC] 输出反缩放
    float* __restrict__ out,            // [H][B][DV]
    int B, int Lmax, int maxb) {
  constexpr int WARPS = ROWG * DVGRP;
  constexpr int BH = ROWG * 16;
  constexpr int DVW = DV / DVGRP;
  constexpr int DKP = DK + 8;
  constexpr int KH = KT / DVGRP;
  constexpr int KTP = KT + 8;
  constexpr int NT = KH / 8;
  constexpr int NDH = DVW / 8;
  constexpr int PS = 64;  // page size
  static_assert(KH % 16 == 0, "KH 必须是 16 的倍数（QK 内层按 16 key/nb 展开）");
  static_assert(KT % 16 == 0 && DV / DVGRP % 16 == 0, "PV tile 必须是 16 的倍数");

  extern __shared__ bf16 smem[];
  bf16* base = smem;
  bf16 (*qs)[DKP] = reinterpret_cast<bf16(*)[DKP]>(base);
  base += BH * DKP;
  constexpr int NKBUF = (PIPE && !FP8) ? 2 : 1;  // bf16 直载需双缓冲；fp8 由 raw staging 双缓冲
  constexpr int NSTG = (PIPE && FP8) ? 2 : 0;    // raw fp8 staging
  bf16 (*ks)[DKP] = reinterpret_cast<bf16(*)[DKP]>(base);
  base += NKBUF * KT * DKP;
  bf16 (*ps)[KTP] = reinterpret_cast<bf16(*)[KTP]>(base);
  base += BH * KTP;
  float* pmax = reinterpret_cast<float*>(base);
  float* psum = pmax + DVGRP * ROWG * 16;
  base += (2 * DVGRP * ROWG * 16 + 1) / 2;  // 2 个 float 数组
  // raw fp8 staging：[NSTG][KT][DC] + [NSTG][KT][DR]（字节）
  unsigned char* stage8 = reinterpret_cast<unsigned char*>(base);

  const int b = blockIdx.y;
  const int head0 = blockIdx.x * BH;
  const int L = seqlen[b];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int rowg = w % ROWG;
  const int dvg = w / ROWG;
  const int T = WARPS * 32;
  const int r = lane >> 2;

  // ---- 载入 Q（BH 个 head，同一 batch b）----
  for (int i = tid; i < BH * DC / 8; i += T) {
    const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
    const int ghead = head0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ghead < H) v = *reinterpret_cast<const uint4*>(&qa[((size_t)ghead * B + b) * DC + c8]);
    *reinterpret_cast<uint4*>(&qs[row][c8]) = v;
  }
  for (int i = tid; i < BH * DR / 8; i += T) {
    const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
    const int ghead = head0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ghead < H) v = *reinterpret_cast<const uint4*>(&qr[((size_t)ghead * B + b) * DR + c8]);
    *reinterpret_cast<uint4*>(&qs[row][DC + c8]) = v;
  }

  // key j 的 [c_kv | k_pe] 源地址（返回 c_kv 基址，k_pe 用同样公式 + kr 基址）
  constexpr int ELEM = FP8 ? 1 : 2;  // 每元素字节数（KV cache 的 stride 用）
  auto ck_ptr = [&](int j) -> const void* {
    if (PAGED) {
      const int page = block_table[(size_t)b * maxb + j / PS];
      return reinterpret_cast<const char*>(ckv) + ((size_t)page * PS + (j % PS)) * DC * ELEM;
    }
    return reinterpret_cast<const char*>(ckv) + ((size_t)b * Lmax + j) * DC * ELEM;
  };
  auto kr_ptr = [&](int j) -> const void* {
    if (PAGED) {
      const int page = block_table[(size_t)b * maxb + j / PS];
      return reinterpret_cast<const char*>(kr) + ((size_t)page * PS + (j % PS)) * DR * ELEM;
    }
    return reinterpret_cast<const char*>(kr) + ((size_t)b * Lmax + j) * DR * ELEM;
  };

  float S[NT][4];
  float O[NDH][4];
#pragma unroll
  for (int i = 0; i < NDH; ++i)
#pragma unroll
    for (int q = 0; q < 4; ++q) O[i][q] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  // ---- v1：同步向量化装载（bf16 直接 uint4；fp8 读 8B -> 反量化 16B）----
  auto load_sync = [&](int k0, bf16 (*buf)[DKP]) {
    if (!FP8) {
      for (int i = tid; i < KT * DC / 8; i += T) {
        const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
        const int j = k0 + row;
        *reinterpret_cast<uint4*>(&buf[row][c8]) =
            *reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(ck_ptr(j)) + c8);
      }
      for (int i = tid; i < KT * DR / 8; i += T) {
        const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
        const int j = k0 + row;
        *reinterpret_cast<uint4*>(&buf[row][DC + c8]) =
            *reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(kr_ptr(j)) + c8);
      }
    } else {
      for (int i = tid; i < KT * DC / 8; i += T) {
        const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
        const int j = k0 + row;
        uint4 v = dequant8_e4m3(*reinterpret_cast<const uint2*>(
            reinterpret_cast<const unsigned char*>(ck_ptr(j)) + c8));
        *reinterpret_cast<uint4*>(&buf[row][c8]) = v;
      }
      for (int i = tid; i < KT * DR / 8; i += T) {
        const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
        const int j = k0 + row;
        uint4 v = dequant8_e4m3(*reinterpret_cast<const uint2*>(
            reinterpret_cast<const unsigned char*>(kr_ptr(j)) + c8));
        *reinterpret_cast<uint4*>(&buf[row][DC + c8]) = v;
      }
    }
  };

  // ---- PIPE：cp.async 预取。bf16 直接进 buf；fp8 先进紧凑 staging ----
  unsigned char* s8ck[2];
  unsigned char* s8kr[2];
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    s8ck[s] = stage8 + (size_t)s * KT * DC;
    s8kr[s] = stage8 + (size_t)NSTG * KT * DC + (size_t)s * KT * DR;
  }
  auto issue_async = [&](int k0, int slot) {
    if (!FP8) {
      bf16 (*buf)[DKP] = ks + slot * KT;
      for (int i = tid; i < KT * DC / 8; i += T) {
        const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
        const int j = k0 + row;
        if (j < L)
          __pipeline_memcpy_async(&buf[row][c8],
                                  reinterpret_cast<const bf16*>(ck_ptr(j)) + c8, 16);
        else
          *reinterpret_cast<uint4*>(&buf[row][c8]) = make_uint4(0, 0, 0, 0);
      }
      for (int i = tid; i < KT * DR / 8; i += T) {
        const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
        const int j = k0 + row;
        if (j < L)
          __pipeline_memcpy_async(&buf[row][DC + c8],
                                  reinterpret_cast<const bf16*>(kr_ptr(j)) + c8, 16);
        else
          *reinterpret_cast<uint4*>(&buf[row][DC + c8]) = make_uint4(0, 0, 0, 0);
      }
    } else {
      for (int i = tid; i < KT * DC / 8; i += T) {
        const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
        const int j = k0 + row;
        if (j < L)
          __pipeline_memcpy_async(s8ck[slot] + (size_t)row * DC + c8,
                                  reinterpret_cast<const unsigned char*>(ck_ptr(j)) + c8, 8);
      }
      for (int i = tid; i < KT * DR / 8; i += T) {
        const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
        const int j = k0 + row;
        if (j < L)
          __pipeline_memcpy_async(s8kr[slot] + (size_t)row * DR + c8,
                                  reinterpret_cast<const unsigned char*>(kr_ptr(j)) + c8, 8);
      }
    }
    __pipeline_commit();
  };
  // fp8 staging -> bf16 smem tile
  auto dequant_tile = [&](int slot, bf16 (*buf)[DKP]) {
    for (int i = tid; i < KT * DC / 8; i += T) {
      const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
      uint4 v = dequant8_e4m3(
          *reinterpret_cast<const uint2*>(s8ck[slot] + (size_t)row * DC + c8));
      *reinterpret_cast<uint4*>(&buf[row][c8]) = v;
    }
    for (int i = tid; i < KT * DR / 8; i += T) {
      const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
      uint4 v = dequant8_e4m3(
          *reinterpret_cast<const uint2*>(s8kr[slot] + (size_t)row * DR + c8));
      *reinterpret_cast<uint4*>(&buf[row][DC + c8]) = v;
    }
  };

  // 计算一个已就绪 tile
  auto compute_tile = [&](int k0, bf16 (*kbuf)[DKP]) {
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
        ldmatrix_x4(smem_u32(&kbuf[nbase + nb * 16 + n_off][kk * 16 + k_off]), d);
        mma16816(S[nb * 2], a, d);
        mma16816(S[nb * 2 + 1], a, d + 2);
      }
    }
    // 尾 tile 掩码
    if (k0 + KT > L) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int c0 = k0 + nbase + i * 8 + (lane & 3) * 2;
        if (c0 >= L) {
          S[i][0] = -INFINITY;
          S[i][2] = -INFINITY;
        }
        if (c0 + 1 >= L) {
          S[i][1] = -INFINITY;
          S[i][3] = -INFINITY;
        }
      }
    }
    // 局部 max -> smem 交换 -> 全 max
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

    // PV：读共享 P + V 的 dv 半边（V = c_kv）
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      uint32_t pa[4];
      ldmatrix_x4(smem_u32(&ps[rowg * 16 + (lane & 15)][c * 16 + (lane >> 4) * 8]), pa);
      const int rr = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int dn = 0; dn < DVW / 16; ++dn) {
        uint32_t d[4];
        ldmatrix_x4_trans(smem_u32(&kbuf[c * 16 + rr][dvg * DVW + dn * 16 + (lane >> 4) * 8]),
                          d);
        mma16816(O[dn * 2], pa, d);
        mma16816(O[dn * 2 + 1], pa, d + 2);
      }
    }
  };

  __syncthreads();

  if (!PIPE) {
    for (int k0 = 0; k0 < L; k0 += KT) {
      load_sync(k0, ks);
      __syncthreads();
      compute_tile(k0, ks);
      __syncthreads();
    }
  } else {
    if (L > 0) issue_async(0, 0);
    int slot = 0;
    for (int k0 = 0; k0 < L; k0 += KT, slot ^= 1) {
      const bool has_next = (k0 + KT < L);
      if (has_next) issue_async(k0 + KT, slot ^ 1);
      __pipeline_wait_prior(has_next ? 1 : 0);
      __syncthreads();
      bf16 (*kbuf)[DKP] = FP8 ? ks : ks + slot * KT;
      if (FP8) dequant_tile(slot, kbuf);
      __syncthreads();
      compute_tile(k0, kbuf);
      __syncthreads();
    }
  }

  // ---- 写输出 O / l，FP8 时乘回 vscale ----
  const float iv0 = (l0 > 0.f) ? 1.f / l0 : 0.f, iv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  const int row0 = head0 + rowg * 16 + r, row1 = row0 + 8;
#pragma unroll
  for (int i = 0; i < NDH; ++i) {
    const int col = dvg * DVW + i * 8 + (lane & 3) * 2;
    float sv0 = 1.f, sv1 = 1.f;
    if (FP8) {
      sv0 = vscale[col];
      sv1 = vscale[col + 1];
    }
    if (row0 < H) {
      out[((size_t)row0 * B + b) * DV + col] = O[i][0] * iv0 * sv0;
      out[((size_t)row0 * B + b) * DV + col + 1] = O[i][1] * iv0 * sv1;
    }
    if (row1 < H) {
      out[((size_t)row1 * B + b) * DV + col] = O[i][2] * iv1 * sv0;
      out[((size_t)row1 * B + b) * DV + col + 1] = O[i][3] * iv1 * sv1;
    }
  }
}

// ===========================================================================
// host 参考：对单个 (head, batch) 精确算 MLA decode attention（用反量化后的 bf16 KV）
// ===========================================================================
static void decode_row_ref(int h, int b, int B, int L, const std::vector<bf16>& qa,
                           const std::vector<bf16>& qr, const std::vector<bf16>& ckv,
                           const std::vector<bf16>& kr, const std::vector<float>* vscale,
                           std::vector<double>& out_row) {
  const size_t qoff = (size_t)h * B + b;
  double mx = -1e30;
  std::vector<double> s(L);
  for (int j = 0; j < L; ++j) {
    double acc = 0;
    for (int d = 0; d < DC; ++d)
      acc += (double)__bfloat162float(qa[qoff * DC + d]) *
             (double)__bfloat162float(ckv[((size_t)b * L + j) * DC + d]);
    for (int d = 0; d < DR; ++d)
      acc += (double)__bfloat162float(qr[qoff * DR + d]) *
             (double)__bfloat162float(kr[((size_t)b * L + j) * DR + d]);
    s[j] = acc;
    mx = std::max(mx, acc);
  }
  double sum = 0;
  for (int j = 0; j < L; ++j) {
    s[j] = std::exp(s[j] - mx);
    sum += s[j];
  }
  out_row.assign(DV, 0.0);
  for (int j = 0; j < L; ++j) {
    const double p = s[j] / sum;
    for (int d = 0; d < DV; ++d)
      out_row[d] += p * (double)__bfloat162float(ckv[((size_t)b * L + j) * DC + d]);
  }
  if (vscale) {
    for (int d = 0; d < DV; ++d) out_row[d] *= (double)(*vscale)[d];
  }
}

// ===========================================================================
// 数据构建 / 量化
// ===========================================================================
struct Data {
  int B, L, Lmax, maxb;
  std::vector<bf16> hqa, hqr, hck, hkr;          // 原始 bf16（已按 ckv scale 预缩放 Q）
  std::vector<bf16> hqa_raw;                     // 未预缩放 Q（用于生成 bf16 参考）
  std::vector<unsigned char> hck8, hkr8;         // e4m3
  std::vector<float> sc_ck, sc_kr;               // per-channel scale
  double q_err_ck = 0, q_err_kr = 0;             // 量化最大相对误差
  std::vector<int> hblock, hseq;
  std::vector<int> hck8_paged, hkr8_paged;       // 分页重排后的 fp8
  std::vector<int> page_of_b;                    // (调试用)
};

// 生成随机数据 + per-channel FP8 量化，并返回反量化后的 bf16（作为 kernel 对拍输入）
static Data make_data(int B, int L, unsigned seed) {
  Data d;
  d.B = B;
  d.L = L;
  d.Lmax = L;
  d.maxb = (L + 63) / 64;
  const size_t qaN = (size_t)H * B * DC, qrN = (size_t)H * B * DR;
  const size_t ckN = (size_t)B * L * DC, krN = (size_t)B * L * DR;
  d.hqa_raw.resize(qaN);
  d.hqr.resize(qrN);
  d.hck.resize(ckN);
  d.hkr.resize(krN);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto& v : d.hqa_raw) v = __float2bfloat16(0.2f * dist(rng));
  for (auto& v : d.hqr) v = __float2bfloat16(0.2f * dist(rng));
  // c_kv 用带 outlier 的分布，逼近真实 KV
  for (size_t i = 0; i < ckN; ++i) {
    float x = 0.2f * dist(rng);
    if ((i * 2654435761u) % 1000 == 0) x *= 20.f;
    d.hck[i] = __float2bfloat16(x);
  }
  for (auto& v : d.hkr) v = __float2bfloat16(0.2f * dist(rng));

  // per-channel scale（每维一个，取全 batch/token 的 amax）：单趟扫描
  std::vector<float> amax_ck(DC, 0.f), amax_kr(DR, 0.f);
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < L; ++j) {
      const bf16* pc = &d.hck[((size_t)b * L + j) * DC];
      for (int dc = 0; dc < DC; ++dc) {
        float x = std::fabs(__bfloat162float(pc[dc]));
        if (x > amax_ck[dc]) amax_ck[dc] = x;
      }
      const bf16* pr = &d.hkr[((size_t)b * L + j) * DR];
      for (int dr = 0; dr < DR; ++dr) {
        float x = std::fabs(__bfloat162float(pr[dr]));
        if (x > amax_kr[dr]) amax_kr[dr] = x;
      }
    }
  d.sc_ck.resize(DC);
  d.sc_kr.resize(DR);
  for (int dc = 0; dc < DC; ++dc) d.sc_ck[dc] = amax_ck[dc] > 0 ? amax_ck[dc] / 448.f : 1.f;
  for (int dr = 0; dr < DR; ++dr) d.sc_kr[dr] = amax_kr[dr] > 0 ? amax_kr[dr] / 448.f : 1.f;

  // 量化 -> fp8，再反量化回 bf16 作为对拍输入
  d.hck8.resize(ckN);
  d.hkr8.resize(krN);
  auto q8 = [](float x, float s) -> unsigned char {
    float y = x / s;
    y = std::fmax(-448.f, std::fmin(448.f, y));
    __nv_fp8_e4m3 f = __nv_fp8_e4m3(y);
    return *reinterpret_cast<unsigned char*>(&f);
  };
  auto dq8 = [](unsigned char c) -> float {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&hr));
  };
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < L; ++j)
      for (int dc = 0; dc < DC; ++dc) {
        const size_t i = ((size_t)b * L + j) * DC + dc;
        float orig = __bfloat162float(d.hck[i]);
        d.hck8[i] = q8(orig, d.sc_ck[dc]);
        float deq = dq8(d.hck8[i]) * d.sc_ck[dc];
        if (std::fabs(orig) > 1e-3f)
          d.q_err_ck = std::max(d.q_err_ck, (double)std::fabs(deq - orig) / std::fabs(orig));
        d.hck[i] = __float2bfloat16(dq8(d.hck8[i]));  // 反量化后的无 scale 值（对拍输入）
      }
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < L; ++j)
      for (int dr = 0; dr < DR; ++dr) {
        const size_t i = ((size_t)b * L + j) * DR + dr;
        float orig = __bfloat162float(d.hkr[i]);
        d.hkr8[i] = q8(orig, d.sc_kr[dr]);
        float deq = dq8(d.hkr8[i]) * d.sc_kr[dr];
        if (std::fabs(orig) > 1e-3f)
          d.q_err_kr = std::max(d.q_err_kr, (double)std::fabs(deq - orig) / std::fabs(orig));
        d.hkr[i] = __float2bfloat16(dq8(d.hkr8[i]));
      }

  // 预缩放 Q：q_nope * sc_ck，q_pe * sc_kr（scale 折进 Q，反量化只剩转换）
  d.hqa = d.hqa_raw;
  for (int h = 0; h < H; ++h)
    for (int b = 0; b < B; ++b)
      for (int dc = 0; dc < DC; ++dc)
        d.hqa[((size_t)h * B + b) * DC + dc] = __float2bfloat16(
            __bfloat162float(d.hqa_raw[((size_t)h * B + b) * DC + dc]) * d.sc_ck[dc]);
  for (int h = 0; h < H; ++h)
    for (int b = 0; b < B; ++b)
      for (int dr = 0; dr < DR; ++dr)
        d.hqr[((size_t)h * B + b) * DR + dr] = __float2bfloat16(
            __bfloat162float(d.hqr[((size_t)h * B + b) * DR + dr]) * d.sc_kr[dr]);

  // 分页重排：把 [B][L][DC] -> pages
  d.hblock.resize((size_t)B * d.maxb, -1);
  int npage = 0;
  std::vector<int> page_start(B, 0);
  for (int b = 0; b < B; ++b) {
    page_start[b] = npage;
    int nb = (L + 63) / 64;
    for (int blk = 0; blk < nb; ++blk) d.hblock[(size_t)b * d.maxb + blk] = npage++;
  }
  d.hck8_paged.assign((size_t)npage * 64 * DC, 0);
  d.hkr8_paged.assign((size_t)npage * 64 * DR, 0);
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < L; ++j) {
      int pg = d.hblock[(size_t)b * d.maxb + j / 64];
      size_t dst = ((size_t)pg * 64 + (j % 64));
      memcpy(&d.hck8_paged[dst * DC], &d.hck8[((size_t)b * L + j) * DC], DC);
      memcpy(&d.hkr8_paged[dst * DR], &d.hkr8[((size_t)b * L + j) * DR], DR);
    }
  return d;
}

// ---------------------------------------------------------------------------
// 启动器
// ---------------------------------------------------------------------------
// 抽样参考：只对 nref 个 (head, batch) 行算精确参考（全量在 L=32k 时太慢）
struct RefSet {
  std::vector<int> hs, bs;   // nref
  std::vector<double> v;     // nref * DV
};

template <int ROWG, int DVGRP, int KT, bool FP8, bool PIPE, bool PAGED>
static void launch(const char* tag, Data& d, const bf16* qa, const bf16* qr, const void* ck,
                   const void* kr, const int* bt, const int* sl, const float* vsc, float* out,
                   double bytes, double flops, const RefSet* refs) {
  constexpr int BH = ROWG * 16;
  constexpr int WARPS = ROWG * DVGRP;
  constexpr int NKBUF = (PIPE && !FP8) ? 2 : 1;
  constexpr int NSTG = (PIPE && FP8) ? 2 : 0;
  size_t shm = (size_t)BH * (DK + 8) * sizeof(bf16) +
               (size_t)NKBUF * KT * (DK + 8) * sizeof(bf16) +
               (size_t)BH * (KT + 8) * sizeof(bf16) +
               2 * (size_t)DVGRP * ROWG * 16 * sizeof(float);
  if (NSTG) shm += (size_t)NSTG * KT * (DC + DR);
  if (shm > (size_t)227 * 1024) {
    std::printf("%-4s ROWG=%d DVGRP=%d KT=%-3d warps=%-2d  skipped (smem %zuKB)\n", tag, ROWG,
                DVGRP, KT, WARPS, shm / 1024);
    return;
  }
  auto fn = mla_decode_kernel<ROWG, DVGRP, KT, FP8, PIPE, PAGED>;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(div_up(H, BH), d.B);
  const bf16* qap = qa;
  const bf16* qrp = qr;
  auto run = [&] {
    fn<<<grid, WARPS * 32, shm>>>(qap, qrp, ck, kr, bt, sl, vsc, out, d.B, d.Lmax, d.maxb);
  };
  run();
  CUDA_CHECK_LAST();

  if (refs) {
    std::vector<float> hout((size_t)H * d.B * DV);
    CUDA_CHECK(cudaMemcpy(hout.data(), out, hout.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, base = 0;
    for (size_t i = 0; i < refs->hs.size(); ++i) {
      const int h = refs->hs[i], b = refs->bs[i];
      const double* rr = refs->v.data() + i * DV;
      for (int dd = 0; dd < DV; ++dd) {
        const double e = rr[dd];
        err = std::max(err, std::fabs((double)hout[((size_t)h * d.B + b) * DV + dd] - e));
        base = std::max(base, std::fabs(e));
      }
    }
    std::printf("     [%-4s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, base,
                err / std::max(base, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }

  double time = bench_ms(run, 3, 20);
  const double tfl = to_tflops(flops, time);
  const double gbps = bytes * 1000.0 / time / 1e9;
  std::printf("%-4s ROWG=%d DVGRP=%d KT=%-3d warps=%-2d %s%s %9.4f ms  %7.1f GB/s (%5.1f%% HBM)  "
              "%7.1f TFLOPS (%5.1f%%)\n",
              tag, ROWG, DVGRP, KT, WARPS, FP8 ? "fp8" : "bf16", PIPE ? "+pipe" : "     ", time,
              gbps, 100.0 * gbps / 3350.0, tfl, 100.0 * tfl / 989.0);
}

int main(int argc, char** argv) {
  int B = (argc > 1) ? std::atoi(argv[1]) : 64;
  int L = (argc > 2) ? std::atoi(argv[2]) : 32768;
  const char* which = (argc > 3) ? argv[3] : "all";

  DeviceInfo di = device_info(0);
  print_device_info(di);

  Data d = make_data(B, L, 20260922);
  const long valid = (long)B * L;
  const double flops = 2.0 * H * (DK + DV) * (double)valid;
  const double bytes_bf16 = (double)valid * DK * 2;
  const double bytes_fp8 = (double)valid * DK * 1;

  std::printf("\nMLA absorbed decode: H=%d B=%d L=%d  (DC=%d DR=%d DV=%d DK=%d)\n", H, B, L, DC,
              DR, DV, DK);
  std::printf("FLOPs = %.3f TFLOP ; bf16 KV bytes = %.3f GB (AI=%.0f) ; fp8 bytes = %.3f GB\n",
              flops / 1e12, bytes_bf16 / 1e9, flops / bytes_bf16, bytes_fp8 / 1e9);
  std::printf("per-channel e4m3 量化：c_kv 最大相对误差 %.3f%% / k_pe %.3f%%\n\n",
              d.q_err_ck * 100, d.q_err_kr * 100);

  bf16 *qa, *qr;
  unsigned char *ck8, *kr8;
  int *bt8, *sl;
  float *vsc, *out;
  CUDA_CHECK(cudaMalloc(&qa, (size_t)H * B * DC * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&qr, (size_t)H * B * DR * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&ck8, d.hck8.size()));
  CUDA_CHECK(cudaMalloc(&kr8, d.hkr8.size()));
  CUDA_CHECK(cudaMalloc(&bt8, d.hblock.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&sl, B * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&vsc, DC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, (size_t)H * B * DV * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(qa, d.hqa.data(), (size_t)H * B * DC * sizeof(bf16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(qr, d.hqr.data(), (size_t)H * B * DR * sizeof(bf16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ck8, d.hck8.data(), d.hck8.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(kr8, d.hkr8.data(), d.hkr8.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bt8, d.hblock.data(), d.hblock.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  std::vector<int> slv(B, L);
  CUDA_CHECK(cudaMemcpy(sl, slv.data(), B * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(vsc, d.sc_ck.data(), DC * sizeof(float), cudaMemcpyHostToDevice));

  // 分页 KV（fp8）
  unsigned char *ck8p, *kr8p;
  CUDA_CHECK(cudaMalloc(&ck8p, d.hck8_paged.size()));
  CUDA_CHECK(cudaMalloc(&kr8p, d.hkr8_paged.size()));
  CUDA_CHECK(cudaMemcpy(ck8p, d.hck8_paged.data(), d.hck8_paged.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(kr8p, d.hkr8_paged.data(), d.hkr8_paged.size(),
                        cudaMemcpyHostToDevice));

  // bf16 KV（用于 v1/v2 对拍；把 fp8 反量化值当输入，保证 v1/v2/v3 数学一致）
  bf16 *ckb, *krb;
  CUDA_CHECK(cudaMalloc(&ckb, (size_t)valid * DC * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&krb, (size_t)valid * DR * sizeof(bf16)));
  CUDA_CHECK(cudaMemcpy(ckb, d.hck.data(), d.hck.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(krb, d.hkr.data(), d.hkr.size() * sizeof(bf16), cudaMemcpyHostToDevice));

  // host 参考（只对 3 个抽样行；bf16 无 vscale，fp8 再乘 vscale）
  RefSet ref_bf, ref_fp8;
  const int nhb = 3;
  for (int i = 0; i < nhb; ++i) {
    ref_bf.hs.push_back((i * 61 + 7) % H);
    ref_bf.bs.push_back((i * 13 + 1) % B);
  }
  ref_bf.v.assign((size_t)nhb * DV, 0.0);
  ref_fp8.hs = ref_bf.hs;
  ref_fp8.bs = ref_bf.bs;
  ref_fp8.v.assign((size_t)nhb * DV, 0.0);
  for (int i = 0; i < nhb; ++i) {
    std::vector<double> row;
    decode_row_ref(ref_bf.hs[i], ref_bf.bs[i], B, L, d.hqa, d.hqr, d.hck, d.hkr, nullptr, row);
    std::copy(row.begin(), row.end(), ref_bf.v.begin() + (size_t)i * DV);
    for (int dd = 0; dd < DV; ++dd) row[dd] *= d.sc_ck[dd];
    std::copy(row.begin(), row.end(), ref_fp8.v.begin() + (size_t)i * DV);
  }

  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  if (want("v1")) {
    launch<4, 4, 64, false, false, false>("v1", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                          bytes_bf16, flops, &ref_bf);
  }
  if (want("v2")) {
    launch<4, 2, 32, false, true, false>("v2", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, &ref_bf);
    launch<2, 2, 64, false, true, false>("v2", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, &ref_bf);
    launch<2, 4, 64, false, true, false>("v2", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, &ref_bf);
    launch<4, 4, 64, false, false, false>("v2", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                          bytes_bf16, flops, &ref_bf);
  }
  if (want("v3")) {
    launch<4, 4, 64, true, false, false>("v3", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                         bytes_fp8, flops, &ref_fp8);
    launch<4, 2, 32, true, true, false>("v3", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                        bytes_fp8, flops, &ref_fp8);
    launch<2, 2, 64, true, true, false>("v3", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                        bytes_fp8, flops, &ref_fp8);
    launch<2, 4, 64, true, true, false>("v3", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                        bytes_fp8, flops, &ref_fp8);
    launch<4, 2, 32, true, true, true>("v3p", d, qa, qr, ck8p, kr8p, bt8, sl, vsc, out, bytes_fp8,
                                       flops, &ref_fp8);
  }

  if (want("prof")) {
    launch<4, 2, 32, false, true, false>("p", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, nullptr);
  }
  if (want("proffp8")) {
    launch<4, 4, 64, true, false, false>("pf", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                         bytes_fp8, flops, &ref_fp8);
  }

  if (want("scan")) {
    // 扫：RO WG × DVGRP × KT × {非pipe,pipe} × {bf16,fp8}（只列合法几何：KH≥16 且 16 整除）
    launch<4, 2, 32, false, true, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, nullptr);
    launch<4, 2, 64, false, false, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                          bytes_bf16, flops, nullptr);
    launch<4, 4, 64, false, false, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                          bytes_bf16, flops, nullptr);
    launch<2, 4, 64, false, true, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, nullptr);
    launch<2, 4, 128, false, false, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                           bytes_bf16, flops, nullptr);
    launch<2, 8, 128, false, false, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                           bytes_bf16, flops, nullptr);
    launch<2, 2, 64, false, true, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, nullptr);
    launch<2, 2, 32, false, true, false>("s", d, qa, qr, ckb, krb, nullptr, sl, nullptr, out,
                                         bytes_bf16, flops, nullptr);
    // fp8（KV 减半）
    launch<4, 2, 32, true, true, false>("s", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                        bytes_fp8, flops, nullptr);
    launch<2, 4, 64, true, true, false>("s", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                        bytes_fp8, flops, nullptr);
    launch<2, 4, 128, true, false, false>("s", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                          bytes_fp8, flops, nullptr);
    launch<2, 8, 128, true, false, false>("s", d, qa, qr, ck8, kr8, nullptr, sl, vsc, out,
                                          bytes_fp8, flops, nullptr);
  }

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ck8));
  CUDA_CHECK(cudaFree(kr8));
  CUDA_CHECK(cudaFree(bt8));
  CUDA_CHECK(cudaFree(sl));
  CUDA_CHECK(cudaFree(vsc));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(ck8p));
  CUDA_CHECK(cudaFree(kr8p));
  CUDA_CHECK(cudaFree(ckb));
  CUDA_CHECK(cudaFree(krb));
  return 0;
}
