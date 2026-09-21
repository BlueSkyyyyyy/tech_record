// 38 Paged KV-cache / flash-decoding 推理注意力（主题 25）
//
// 推理 decode 阶段的注意力：q_len = 1，但 KV 是**分页**的
//   k_cache / v_cache : [num_blocks, block_size, n_kv_heads, head_dim]
//   block_table       : [batch, max_blocks]  （request -> 物理页号）
//   seqlen            : [batch]              （变长）
//
// GQA：n_q_heads = n_kv_heads * G，G 个 query head 共享同一份 KV。
// decode 是**纯访存**算子：FLOPs = 4*B*Hq*L*D，字节 = 2*B*Hkv*L*D*2(KV)，
//   arithmetic intensity = Hq/(2*Hkv) = G/2 ≈ 2.5 → 远低于 H800 的 ridge。
//
// 三条实现：
//   v0 naive  : 一 warp 一个 (request, q_head)，直接 global 读 KV（KV 被读 G 遍）
//   v1 shared : 一 CTA 一个 (request, kv_head)，G 个 warp 各管一个 q_head，
//               KV tile 协作搬进 smem（KV 只读 1 遍）+ 16B 向量化
//   v2 flash  : 在 v1 上把 KV 沿序列 split 成多块（flash-decoding），
//               各块写 partial (O,m,l)，再用 combine kernel 合并
//   另有 dense  : 同 v1 但 KV 用连续布局（无 block_table），量 page-table 开销
//
// 参考实现：naive kernel（结构独立） + 详见 paged_ref.py 的 torch 交叉验证。
//
// 运行：scripts/run.sh 38-paged-kv/paged_decode.cu [B] [Lmax] [nsplit]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;

// Qwen3-8B（/ssd/models/qwen3-8B/config.json）
constexpr int Hq = 40;          // num_attention_heads
constexpr int Hkv = 8;          // num_key_value_heads
constexpr int G = Hq / Hkv;     // GQA group = 5
constexpr int D = 128;          // head_dim
constexpr int P = 16;           // page / block size

#define DEV_INLINE __device__ __forceinline__

DEV_INLINE float dot4(float q[4], const bf16* k) {
  __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(k);
  __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(k + 2);
  float2 f01 = __bfloat1622float2(k01);
  float2 f23 = __bfloat1622float2(k23);
  return q[0] * f01.x + q[1] * f01.y + q[2] * f23.x + q[3] * f23.y;
}

DEV_INLINE float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---------------------------------------------------------------------------
// mma.m16n8k16 bf16 helpers（与 16 篇同一套）
// ---------------------------------------------------------------------------
DEV_INLINE uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
DEV_INLINE void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
DEV_INLINE void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
DEV_INLINE void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV_INLINE uint32_t pack2(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

// ---------------------------------------------------------------------------
// v0：naive —— 一 warp 一个 (request, q_head)，KV 直接 global 读，不 split。
//      G 个 head 各读一遍 KV => KV 流量 ×G。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32) paged_decode_naive(
    const bf16* __restrict__ Q, const bf16* __restrict__ Kc,
    const bf16* __restrict__ Vc, const int* __restrict__ block_table,
    const int* __restrict__ seqlen, float* __restrict__ out, int B, int maxb) {
  int b = blockIdx.x / Hq, hq = blockIdx.x % Hq;
  int kv = hq / G, lane = threadIdx.x;
  int L = seqlen[b];
  float q[4], o[4] = {0, 0, 0, 0};
  const bf16* qp = Q + ((size_t)(b * Hq + hq)) * D;
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = __bfloat162float(qp[lane * 4 + i]);
  float m = -INFINITY, l = 0.f;
  for (int j = 0; j < L; ++j) {
    int page = block_table[b * maxb + j / P];
    int p = j % P;
    const bf16* k = Kc + (((size_t)page * P + p) * Hkv + kv) * D + lane * 4;
    float s = warp_sum(dot4(q, k));
    float mn = fmaxf(m, s);
    float a = __expf(m - mn), pv = __expf(s - mn);
    l = l * a + pv;
    const bf16* v = Vc + (((size_t)page * P + p) * Hkv + kv) * D + lane * 4;
    __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(v);
    __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(v + 2);
    float2 f01 = __bfloat1622float2(v01), f23 = __bfloat1622float2(v23);
    o[0] = o[0] * a + pv * f01.x;
    o[1] = o[1] * a + pv * f01.y;
    o[2] = o[2] * a + pv * f23.x;
    o[3] = o[3] * a + pv * f23.y;
    m = mn;
  }
  float* op = out + ((size_t)(b * Hq + hq)) * D;
  float inv = 1.f / l;
#pragma unroll
  for (int i = 0; i < 4; ++i) op[lane * 4 + i] = o[i] * inv;
}

// ---------------------------------------------------------------------------
// v1/v2：一 CTA 一个 (request, kv_head) × 一个 KV split。
//   G 个 warp = G 个 q_head，KV tile 协作搬进 smem（只读一遍 + 16B 向量化），
//   每 key 做一次 warp 归约得 score，online softmax。
//   PAGED=false 时 KV 是连续布局 [B,Hkv,Lmax,D]，用来量 page-table 开销。
// ---------------------------------------------------------------------------
template <bool PAGED, int NPAGES>
__global__ void __launch_bounds__(G * 32) paged_decode_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ Kc,
    const bf16* __restrict__ Vc, const int* __restrict__ block_table,
    const int* __restrict__ seqlen, float* __restrict__ po,
    float* __restrict__ pm, float* __restrict__ pl, int B, int maxb,
    int nsplit, int Lmax) {
  int by = blockIdx.y;                 // b*Hkv + kv
  int b = by / Hkv, kv = by % Hkv;
  int L = seqlen[b];
  int nblk = (L + P - 1) / P;
  int s = blockIdx.x;
  int bps = (nblk + nsplit - 1) / nsplit;
  int blk0 = s * bps, blk1 = min(nblk, blk0 + bps);

  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int qh = kv * G + warp;
  const bf16* qp = Q + ((size_t)(b * Hq + qh)) * D;
  float q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = __bfloat162float(qp[lane * 4 + i]);

  __shared__ bf16 ks[P][D];
  __shared__ bf16 vs[P][D];

  float m = -INFINITY, l = 0.f, o[4] = {0, 0, 0, 0};

  constexpr int VEC = 8;  // 每个线程一次搬 8 个 bf16 = 16B
  const int nvec = P * (D / VEC);
  for (int blk = blk0; blk < blk1; ++blk) {
    int page = PAGED ? block_table[b * maxb + blk] : b;  // dense: b 当页基
    long long base;
    if (PAGED)
      base = ((long long)page * P * Hkv + kv) * D;
    else
      base = (((long long)b * Hkv + kv) * Lmax + blk * P) * D;
    // 协作搬 K/V tile：连续 v -> 连续 d8（行内连续 256B）
    for (int v = threadIdx.x; v < nvec; v += G * 32) {
      int p = v / (D / VEC), d8 = (v % (D / VEC)) * VEC;
      long long off = PAGED ? base + (long long)p * Hkv * D + d8
                            : base + (long long)p * D + d8;
      *reinterpret_cast<float4*>(&ks[p][d8]) =
          *reinterpret_cast<const float4*>(Kc + off);
      *reinterpret_cast<float4*>(&vs[p][d8]) =
          *reinterpret_cast<const float4*>(Vc + off);
    }
    __syncthreads();

    int valid = min(P, L - blk * P);
    for (int p = 0; p < valid; ++p) {
      float sc = warp_sum(dot4(q, &ks[p][lane * 4]));
      float mn = fmaxf(m, sc);
      float a = __expf(m - mn), pv = __expf(sc - mn);
      l = l * a + pv;
      __nv_bfloat162 v01 =
          *reinterpret_cast<const __nv_bfloat162*>(&vs[p][lane * 4]);
      __nv_bfloat162 v23 =
          *reinterpret_cast<const __nv_bfloat162*>(&vs[p][lane * 4 + 2]);
      float2 f01 = __bfloat1622float2(v01), f23 = __bfloat1622float2(v23);
      o[0] = o[0] * a + pv * f01.x;
      o[1] = o[1] * a + pv * f01.y;
      o[2] = o[2] * a + pv * f23.x;
      o[3] = o[3] * a + pv * f23.y;
      m = mn;
    }
    __syncthreads();
  }

  // 写 partial：po[nsplit][B][Hkv][G][D], pm/pl[nsplit][B][Hkv][G]
  size_t pbase = (((size_t)s * B + b) * Hkv + kv) * G;
  size_t obase = (pbase + warp) * D + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) po[obase + i] = o[i];
  pm[pbase + warp] = m;
  pl[pbase + warp] = l;
}

// ---------------------------------------------------------------------------
// v3：tensor-core 版。一 CTA = 一个 (request, kv_head) × 一个 split；
//   CTA 内 NW 个 warp **各管一组页**（无跨 warp 同步），每个 warp 用
//   mma.m16n8k16 一次算完全部 G 个 q_head（M=16，G=5 补齐）：
//     S[G,P]  = Q[G,D] · K[P,D]^T   （K 以 [n=p][k=d] 行主序 -> 非转置 ldmatrix）
//     O[G,D] += P[G,P] · V[P,D]     （V 以 [k=p][n=d] 行主序 -> 转置 ldmatrix）
//   softmax 的行 = q_head（lane/4），列 = key。
//   每个 warp 写自己的 partial（子 split），combine 再跨 nsplit*NW 合并。
// ---------------------------------------------------------------------------
template <int NW>
__global__ void __launch_bounds__(NW * 32) paged_decode_mma(
    const bf16* __restrict__ Q, const bf16* __restrict__ Kc,
    const bf16* __restrict__ Vc, const int* __restrict__ block_table,
    const int* __restrict__ seqlen, float* __restrict__ po,
    float* __restrict__ pm, float* __restrict__ pl, int B, int maxb,
    int nsplit, int Lmax) {
  constexpr int DKP = D + 8;    // 8 个 bf16 padding，消 ldmatrix bank conflict
  constexpr int NT = P / 8;     // S 的 n8 tile 数 = 2
  constexpr int NDV = D / 8;    // O 的 n8 tile 数 = 16
  int by = blockIdx.y;
  int b = by / Hkv, kv = by % Hkv;
  int L = seqlen[b];
  int nblk = (L + P - 1) / P;
  int s = blockIdx.x;
  int bps = (nblk + nsplit - 1) / nsplit;
  int blk0 = s * bps, blk1 = min(nblk, blk0 + bps);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  __shared__ bf16 qs[16][DKP];
  extern __shared__ bf16 smem_dyn[];
  bf16(*ksv)[2][P][DKP] = reinterpret_cast<bf16(*)[2][P][DKP]>(smem_dyn);

  // Q -> smem（只 G 行有效，其余置 0；每 CTA 共享，只搬一次）
  for (int i = threadIdx.x; i < 16 * D; i += NW * 32) {
    int r = i / D, c = i % D;
    qs[r][c] = (r < G) ? Q[((size_t)(b * Hq + kv * G + r)) * D + c]
                       : __float2bfloat16(0.f);
  }
  __syncthreads();

  uint32_t qa[D / 16][4];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk)
    ldmatrix_x4(smem_u32(&qs[(lane & 15)][kk * 16 + (lane >> 4) * 8]), qa[kk]);

  bf16(*ks)[DKP] = ksv[warp][0];
  bf16(*vs)[DKP] = ksv[warp][1];

  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float O[NDV][4];
#pragma unroll
  for (int t = 0; t < NDV; ++t)
#pragma unroll
    for (int q = 0; q < 4; ++q) O[t][q] = 0.f;

  const int n_off = (lane & 7) + ((lane >> 4) & 1) * 8;
  const int k_off = ((lane >> 3) & 1) * 8;
  const int r_off = (lane & 7) + ((lane >> 3) & 1) * 8;

  for (int blk = blk0 + warp; blk < blk1; blk += NW) {
    int page = block_table[b * maxb + blk];
    const bf16* kbase = Kc + ((size_t)page * P * Hkv + kv) * D;
    const bf16* vbase = Vc + ((size_t)page * P * Hkv + kv) * D;
#pragma unroll
    for (int i = lane; i < P * (D / 8); i += 32) {
      int p = i / (D / 8), d8 = (i % (D / 8)) * 8;
      *reinterpret_cast<float4*>(&ks[p][d8]) =
          *reinterpret_cast<const float4*>(kbase + (size_t)p * Hkv * D + d8);
      *reinterpret_cast<float4*>(&vs[p][d8]) =
          *reinterpret_cast<const float4*>(vbase + (size_t)p * Hkv * D + d8);
    }
    __syncwarp();

    // ---- QK^T ----
    float S[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q) S[t][q] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint32_t d[4];
      ldmatrix_x4(smem_u32(&ks[n_off][kk * 16 + k_off]), d);
      mma16816(S[0], qa[kk], d);
      mma16816(S[1], qa[kk], d + 2);
    }

    // 尾页掩码：key >= valid 置 -inf
    int valid = min(P, L - blk * P);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      int c0 = t * 8 + (lane & 3) * 2;
      if (c0 >= valid) {
        S[t][0] = -INFINITY;
        S[t][1] = -INFINITY;
      } else if (c0 + 1 >= valid) {
        S[t][1] = -INFINITY;
      }
    }

    // ---- softmax（行 = head，列 = key）----
    float bmx0 = -INFINITY, bmx1 = -INFINITY;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      bmx0 = fmaxf(bmx0, fmaxf(S[t][0], S[t][1]));
      bmx1 = fmaxf(bmx1, fmaxf(S[t][2], S[t][3]));
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

    // ---- PV（累加器布局 == A 片段布局，零 shuffle）----
    uint32_t pa[4];
    pa[0] = pack2(S[0][0], S[0][1]);
    pa[1] = pack2(S[0][2], S[0][3]);
    pa[2] = pack2(S[1][0], S[1][1]);
    pa[3] = pack2(S[1][2], S[1][3]);
#pragma unroll
    for (int dn = 0; dn < D / 16; ++dn) {
      uint32_t d[4];
      ldmatrix_x4_trans(smem_u32(&vs[r_off][dn * 16 + (lane >> 4) * 8]), d);
      mma16816(O[dn * 2], pa, d);
      mma16816(O[dn * 2 + 1], pa, d + 2);
    }
    __syncwarp();
  }

  // ---- 写 partial：一个 warp 一个子 split ----
  const int sp = s * NW + warp;
  size_t pbase = (((size_t)sp * B + b) * Hkv + kv) * G;
#pragma unroll
  for (int t = 0; t < NDV; ++t) {
    int col = t * 8 + (lane & 3) * 2;
    int row0 = lane >> 2, row1 = row0 + 8;
    if (row0 < G) {
      po[(pbase + row0) * D + col] = O[t][0];
      po[(pbase + row0) * D + col + 1] = O[t][1];
    }
    if (row1 < G) {
      po[(pbase + row1) * D + col] = O[t][2];
      po[(pbase + row1) * D + col + 1] = O[t][3];
    }
  }
  if ((lane & 3) == 0) {
    int row0 = lane >> 2, row1 = row0 + 8;
    if (row0 < G) {
      pm[pbase + row0] = m0;
      pl[pbase + row0] = l0;
    }
    if (row1 < G) {
      pm[pbase + row1] = m1;
      pl[pbase + row1] = l1;
    }
  }
}

// ---------------------------------------------------------------------------
// combine：把 nsplit 个 partial 按 online-softmax 公式合并成最终 fp32 输出。
//   out[b,hq,d] = sum_s exp(m_s-m) * O_s[d] / sum_s exp(m_s-m) * l_s
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(D) combine_kernel(
    const float* __restrict__ po, const float* __restrict__ pm,
    const float* __restrict__ pl, float* __restrict__ out, int B, int nsplit) {
  int t = blockIdx.x;                 // b*Hq + hq
  int b = t / Hq, hq = t % Hq;
  int kv = hq / G, g = hq % G;
  int d = threadIdx.x;
  float m = -INFINITY;
  for (int s = 0; s < nsplit; ++s) {
    size_t idx = ((size_t)s * B + b) * Hkv * G + kv * G + g;
    m = fmaxf(m, pm[idx]);
  }
  float o = 0.f, l = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    size_t base = ((size_t)s * B + b) * Hkv * G;
    float w = __expf(pm[base + kv * G + g] - m);
    l += pl[base + kv * G + g] * w;
    o += po[(base + kv * G + g) * D + d] * w;
  }
  out[((size_t)t) * D + d] = o / l;
}

// ---------------------------------------------------------------------------
// 参考：把 paged KV gather 成 dense，再做两趟（非 online）softmax。
// ---------------------------------------------------------------------------
__global__ void gather_kv_kernel(const bf16* __restrict__ Kc,
                                 const bf16* __restrict__ Vc,
                                 const int* __restrict__ block_table,
                                 const int* __restrict__ seqlen, bf16* __restrict__ Kd,
                                 bf16* __restrict__ Vd, int B, int maxb, int Lmax) {
  int b = blockIdx.y;
  int L = seqlen[b];
  for (int j = blockIdx.x; j < L; j += gridDim.x) {
    int page = block_table[b * maxb + j / P], p = j % P;
    for (int kv = 0; kv < Hkv; ++kv) {
      size_t src = (((size_t)page * P + p) * Hkv + kv) * D;
      size_t dst = (((size_t)b * Hkv + kv) * Lmax + j) * D;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        Kd[dst + d] = Kc[src + d];
        Vd[dst + d] = Vc[src + d];
      }
    }
  }
}

__global__ void ref_scores_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ Kd,
                                  const int* __restrict__ seqlen, float* __restrict__ S,
                                  int B, int Lmax) {
  int b = blockIdx.x / Hq, hq = blockIdx.x % Hq;
  int kv = hq / G;
  int L = seqlen[b];
  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    float s = 0.f;
    for (int d = 0; d < D; ++d)
      s += __bfloat162float(Q[((size_t)(b * Hq + hq)) * D + d]) *
           __bfloat162float(Kd[(((size_t)b * Hkv + kv) * Lmax + j) * D + d]);
    S[((size_t)(b * Hq + hq)) * Lmax + j] = s;
  }
}

__global__ void ref_out_kernel(const float* __restrict__ S, const bf16* __restrict__ Vd,
                               const int* __restrict__ seqlen, float* __restrict__ out,
                               int B, int Lmax) {
  int t = blockIdx.x;  // b*Hq+hq
  int b = t / Hq, hq = t % Hq;
  int kv = hq / G;
  int L = seqlen[b];
  const float* srow = S + (size_t)t * Lmax;
  // 两趟：max 再 exp-sum
  float lm = -INFINITY;
  for (int j = 0; j < L; ++j) lm = fmaxf(lm, srow[j]);
  float lsum = 0.f;
  for (int j = 0; j < L; ++j) lsum += __expf(srow[j] - lm);
  // 每线程负责一个 d
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float o = 0.f;
    for (int j = 0; j < L; ++j) {
      float p = __expf(srow[j] - lm);
      o += p * __bfloat162float(Vd[(((size_t)b * Hkv + kv) * Lmax + j) * D + d]);
    }
    out[(size_t)t * D + d] = o / lsum;
  }
}

// ---------------------------------------------------------------------------
__global__ void fill_bf16_kernel(bf16* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned h = (unsigned)(i * 2654435761u ^ seed * 40503u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    float x = (float)(h & 0xffff) / 32768.f - 1.f;
    p[i] = __float2bfloat16(x);
  }
}
static void fill_bf16(bf16* p, size_t n, unsigned seed) {
  fill_bf16_kernel<<<512, 256>>>(p, n, seed);
  CUDA_CHECK_LAST();
}

int main(int argc, char** argv) {
  int B = argc > 1 ? atoi(argv[1]) : 64;
  int Lmax = argc > 2 ? atoi(argv[2]) : 32768;
  int nsplit = argc > 3 ? atoi(argv[3]) : 4;
  if (nsplit < 1) nsplit = 1;  // dense / v3 单点都需要 >=1
  const char* which = argc > 4 ? argv[4] : "all";

  DeviceInfo dev = device_info(0);
  print_device_info(dev);

  bool uniform = std::getenv("UNIFORM") != nullptr;  // UNIFORM=1 让所有序列等长（与 dense 对标）
  std::vector<int> h_len(B);
  int nb_total = 0;
  for (int b = 0; b < B; ++b) {
    // 变长：90%~100% Lmax（真实 decode 的 tail 场景也接近满长）
    h_len[b] = uniform ? Lmax : Lmax - (int)((double)(b % 10) * 0.01 * Lmax);
    nb_total += (h_len[b] + P - 1) / P;
  }
  int maxb = (Lmax + P - 1) / P;
  long long sum_len = 0;
  for (int b = 0; b < B; ++b) sum_len += h_len[b];
  long long kv_bytes = (long long)2 * Hkv * sum_len * D * 2;  // K+V 全部读取（按真实长度）
  std::printf("B=%d Lmax=%d nsplit=%d  n_blocks=%d  KV bytes=%.2f GB\n", B, Lmax,
              nsplit, nb_total, kv_bytes / 1e9);

  bf16 *dQ, *dK, *dV, *dKd, *dVd;
  int *dlen, *dbt;
  constexpr int MAXSPLIT = 64;
  float *dpo, *dpm, *dpl, *dout, *dout_ref, *dS;
  CUDA_CHECK(cudaMalloc(&dQ, (size_t)B * Hq * D * 2));
  CUDA_CHECK(cudaMalloc(&dK, (size_t)nb_total * P * Hkv * D * 2));
  CUDA_CHECK(cudaMalloc(&dV, (size_t)nb_total * P * Hkv * D * 2));
  CUDA_CHECK(cudaMalloc(&dKd, (size_t)B * Hkv * maxb * P * D * 2));
  CUDA_CHECK(cudaMalloc(&dVd, (size_t)B * Hkv * maxb * P * D * 2));
  CUDA_CHECK(cudaMalloc(&dlen, B * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dbt, (size_t)B * maxb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dS, (size_t)B * Hq * Lmax * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, (size_t)B * Hq * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout_ref, (size_t)B * Hq * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dpo, (size_t)MAXSPLIT * B * Hkv * G * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dpm, (size_t)MAXSPLIT * B * Hkv * G * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dpl, (size_t)MAXSPLIT * B * Hkv * G * sizeof(float)));

  // 物理页号：默认顺序分配；RANDOM=1 时全局打乱（模拟真实 KV-cache 的碎片化）
  std::vector<int> phys(nb_total);
  for (int i = 0; i < nb_total; ++i) phys[i] = i;
  if (std::getenv("RANDOM")) {
    std::mt19937 rng(12345);
    std::shuffle(phys.begin(), phys.end(), rng);
  }
  std::vector<int> h_bt((size_t)B * maxb);
  {
    int page = 0;
    for (int b = 0; b < B; ++b) {
      int nblk = (h_len[b] + P - 1) / P;
      for (int j = 0; j < nblk; ++j) h_bt[(size_t)b * maxb + j] = phys[page++];
      for (int j = nblk; j < maxb; ++j) h_bt[(size_t)b * maxb + j] = -1;
    }
  }
  fill_bf16(dQ, (size_t)B * Hq * D, 1);
  fill_bf16(dK, (size_t)nb_total * P * Hkv * D, 2);
  fill_bf16(dV, (size_t)nb_total * P * Hkv * D, 3);
  CUDA_CHECK(cudaMemcpy(dlen, h_len.data(), B * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbt, h_bt.data(), (size_t)B * maxb * sizeof(int),
                        cudaMemcpyHostToDevice));

  // ---- 参考（gather + 两趟 softmax）----
  {
    dim3 g(std::min(maxb, 256), B);
    gather_kv_kernel<<<g, D>>>(dK, dV, dbt, dlen, dKd, dVd, B, maxb, maxb * P);
    ref_scores_kernel<<<B * Hq, 256>>>(dQ, dKd, dlen, dS, B, maxb * P);
    ref_out_kernel<<<B * Hq, 128>>>(dS, dVd, dlen, dout_ref, B, maxb * P);
    CUDA_CHECK_LAST();
  }

  std::vector<float> h_out((size_t)B * Hq * D), h_ref((size_t)B * Hq * D);
  CUDA_CHECK(cudaMemcpy(h_ref.data(), dout_ref, h_out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  auto report_err = [&](const char* name) {
    CUDA_CHECK(cudaMemcpy(h_out.data(), dout, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double mx = 0, sum = 0;
    for (size_t i = 0; i < h_out.size(); ++i) {
      double e = std::fabs(h_out[i] - h_ref[i]);
      mx = std::max(mx, e);
      sum += e;
    }
    std::printf("  [check] %-22s max_abs_err=%.3e mean=%.3e\n", name, mx,
                sum / h_out.size());
  };

  // ---- v0 naive（nsplit=1）----
  if (std::string(which) == "all" || std::string(which) == "v0") {
    paged_decode_naive<<<B * Hq, 32>>>(dQ, dK, dV, dbt, dlen, dout, B, maxb);
    CUDA_CHECK_LAST();
    report_err("v0 naive");
    double ms = bench_ms([&] {
      paged_decode_naive<<<B * Hq, 32>>>(dQ, dK, dV, dbt, dlen, dout, B, maxb);
    }, 20, 50);
    report_mem("v0 naive", ms, kv_bytes, dev);
    std::printf("        v0 naive  %8.4f ms  %.2f TFLOPS\n", ms,
                to_tflops(4.0 * B * Hq * Lmax * D, ms));
  }

  // ---- v1 shared（nsplit=1）----
  if (std::string(which) == "all" || std::string(which) == "v1") {
    auto run_v1 = [&] {
      paged_decode_kernel<true, 0>
          <<<dim3(1, B * Hkv), G * 32>>>(dQ, dK, dV, dbt, dlen, dpo, dpm, dpl, B,
                                         maxb, 1, maxb * P);
      combine_kernel<<<B * Hq, D>>>(dpo, dpm, dpl, dout, B, 1);
    };
    run_v1();
    CUDA_CHECK_LAST();
    report_err("v1 shared");
    double ms = bench_ms(run_v1, 20, 50);
    report_mem("v1 shared", ms, kv_bytes, dev);
    std::printf("        v1 shared %8.4f ms  %.2f TFLOPS\n", ms,
                to_tflops(4.0 * B * Hq * Lmax * D, ms));
  }

  // ---- v2 flash-decoding：扫 split ----
  if (std::string(which) == "all" || std::string(which) == "v2") {
    int splits[] = {1, 2, 4, 8, 16, 32};
    for (int ns : splits) {
      if (std::string(which) == "v2" && ns != nsplit) continue;
      float *po = dpo, *pm = dpm, *pl = dpl;
      auto run = [&] {
        paged_decode_kernel<true, 0>
            <<<dim3(ns, B * Hkv), G * 32>>>(dQ, dK, dV, dbt, dlen, po, pm, pl, B,
                                            maxb, ns, maxb * P);
        combine_kernel<<<B * Hq, D>>>(po, pm, pl, dout, B, ns);
      };
      run();
      CUDA_CHECK_LAST();
      char nm[64];
      std::snprintf(nm, sizeof(nm), "v2 split=%d", ns);
      report_err(nm);
      double ms = bench_ms(run, 20, 50);
      report_mem(nm, ms, kv_bytes, dev);
      std::printf("        %-12s %8.4f ms  %.2f TFLOPS\n", nm, ms,
                  to_tflops(4.0 * B * Hq * Lmax * D, ms));
    }
  }

  // ---- v3 tensor-core（NW warp/CTA，每个 warp 一组页）----
  if (std::string(which) == "all" || std::string(which) == "v3" ||
      std::string(which) == "v3all" || std::string(which) == "v3b") {
    auto launch_nw = [&](int nw, int ns) {
      if (nw != 4 && nw != 8 && nw != 16) return;  // 只实例化了这三档
      if (ns * nw > MAXSPLIT) {
        std::printf("  skip nw=%d ns=%d (exceeds MAXSPLIT)\n", nw, ns);
        return;
      }
      float *po = dpo, *pm = dpm, *pl = dpl;
      int ntot = ns * nw;
      size_t smem_sz = (size_t)nw * 2 * P * (D + 8) * sizeof(bf16);
      auto run = [&] {
        if (nw == 4) {
          static bool c4 = false;
          if (!c4) { CUDA_CHECK(cudaFuncSetAttribute(paged_decode_mma<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz)); c4 = true; }
          paged_decode_mma<4><<<dim3(ns, B * Hkv), 4 * 32, smem_sz>>>(dQ, dK, dV, dbt, dlen, po, pm, pl, B, maxb, ns, maxb * P);
        } else if (nw == 8) {
          static bool c8 = false;
          if (!c8) { CUDA_CHECK(cudaFuncSetAttribute(paged_decode_mma<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz)); c8 = true; }
          paged_decode_mma<8><<<dim3(ns, B * Hkv), 8 * 32, smem_sz>>>(dQ, dK, dV, dbt, dlen, po, pm, pl, B, maxb, ns, maxb * P);
        } else {
          static bool c16 = false;
          if (!c16) { CUDA_CHECK(cudaFuncSetAttribute(paged_decode_mma<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz)); c16 = true; }
          paged_decode_mma<16><<<dim3(ns, B * Hkv), 16 * 32, smem_sz>>>(dQ, dK, dV, dbt, dlen, po, pm, pl, B, maxb, ns, maxb * P);
        }
        combine_kernel<<<B * Hq, D>>>(po, pm, pl, dout, B, ntot);
      };
      run();
      CUDA_CHECK_LAST();
      char nm[64];
      std::snprintf(nm, sizeof(nm), "v3 split=%dx%d", ns, nw);
      report_err(nm);
      double ms = bench_ms(run, 20, 50);
      report_mem(nm, ms, kv_bytes, dev);
      std::printf("        %-12s %8.4f ms  %.2f TFLOPS\n", nm, ms,
                  to_tflops(4.0 * B * Hq * Lmax * D, ms));
    };
    if (std::string(which) == "v3") {
      launch_nw(8, nsplit);
    } else if (std::string(which) == "v3b") {
      launch_nw(4, nsplit);
    } else {
      launch_nw(4, 1);
      launch_nw(4, 2);
      launch_nw(4, 4);
      launch_nw(4, 8);
      launch_nw(4, 16);
      launch_nw(8, 2);
      launch_nw(8, 4);
      launch_nw(16, 2);
      launch_nw(16, 4);
    }
  }

  // ---- dense（连续布局，无 page-table；只跑一次 nsplit=nsplit）----
  if (std::string(which) == "all" || std::string(which) == "dense") {
    float *po = dpo, *pm = dpm, *pl = dpl;
    auto run = [&] {
      paged_decode_kernel<false, 0>
          <<<dim3(nsplit, B * Hkv), G * 32>>>(dQ, dKd, dVd, dbt, dlen, po, pm, pl,
                                              B, maxb, nsplit, maxb * P);
      combine_kernel<<<B * Hq, D>>>(po, pm, pl, dout, B, nsplit);
    };
    run();
    CUDA_CHECK_LAST();
    report_err("dense");
    double ms = bench_ms(run, 20, 50);
    report_mem("dense", ms, kv_bytes, dev);
    std::printf("        dense      %8.4f ms  %.2f TFLOPS\n", ms,
                to_tflops(4.0 * B * Hq * Lmax * D, ms));
  }

  return 0;
}
