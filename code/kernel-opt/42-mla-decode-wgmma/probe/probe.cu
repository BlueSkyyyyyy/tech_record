// 42 MLA 吸收式 decode：wgmma + 拆分 QK（消 dup）
//
// 39 篇诊断出 decode 的两堵墙：① `mma.sync + ldmatrix` 的 `short_scoreboard`
// （Q 片段每 tile 从 smem 重读、寄存器压力大）② 换了 FP8 KV 存储却没换算力，
// 导致「字节减半但更慢」。本篇把 QK/PV 都换成 **wgmma SS**（操作数直接从 smem
// 描述符读，绕开 ldmatrix 与寄存器墙），并用 **拆键（split-KV）** 消掉 prefill
// 融合 kernel 里 QK 被多个 warpgroup 各算一份的 dup。
//
// 结构（一 CTA = 一个 batch × 一组 BM=64 个 head，NWG 个 warpgroup）：
//   NWG 个 WG 各算 key[wg*KT/NWG, (wg+1)*KT/NWG) 的 S；行内 max/sum 经 smem
//   跨 WG 合并；各写自己的 P 段到 smem；再各做 DV/NWG 列的 PV。
//   → 每个 key 的 QK 只算一次，PV 的列 split 不重复 QK。
//
//   Q[BM,DK] SW128 · K[KT,DK] SW128 · P[BM,KT] SW128 · V^T[DV,KT] SW128
//
// 真实 shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   num_attention_heads=128, head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1
//   → H=128, DC=512, DR=64, DV=512, DK=576
//
// 运行：ARCH="" scripts/run.sh 42-mla-decode-wgmma/mla_decode_wgmma.cu [B] [L] [which]
//   which = all | bf16 | n2 | n4 | scan
#include "../../common/cuda_utils.cuh"
#include "../../20-mla-wgmma-sw128/wgmma_sw128.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int DC = 512, DR = 64, DV = 512, DK = DC + DR, H = 128;

constexpr int BM = 64, KT = 64;
constexpr int SBO_K = (DK / 64) * 1024;  // Q/K 的 8 行组步长（DK=576 → 9*1024）
constexpr int SBO_P = 1024;              // P/V 的 KT=64 → 1*1024

__device__ __forceinline__ int row0_of(int w_, int lane) { return 16 * (w_ & 3) + (lane >> 2); }

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

// ---- wgmma m64n16k16 / m64n32k16 / m64n128k16（m64n64 / m64n256 在 header）----
__device__ __forceinline__ void wgmma_m64n16k16(float (&d)[8], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7},\n"
      "%8, %9, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1));
}

__device__ __forceinline__ void wgmma_m64n32k16(float (&d)[16], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "%16, %17, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(1));
}


// 只对需要的 tile 尺寸分派
template <int N>
__device__ __forceinline__ void wgmma_qk(float (&S)[N / 8 * 4], uint64_t da, uint64_t db) {
  if constexpr (N == 16) wgmma_m64n16k16(S, da, db);
  else if constexpr (N == 32) wgmma_m64n32k16(S, da, db);
  else if constexpr (N == 64) wgmma_m64n64k16(S, da, db);
  else static_assert(N == 16 || N == 32 || N == 64, "QK n 只支持 16/32/64");
}
template <int N>
__device__ __forceinline__ void wgmma_pv(float (&O)[N / 8 * 4], uint64_t da, uint64_t db) {
  if constexpr (N == 128) wgmma_m64n128k16(O, da, db);
  else if constexpr (N == 256) wgmma_m64n256k16(O, da, db);
  else static_assert(N == 128 || N == 256, "PV n 只支持 128/256");
}

// ===========================================================================
// SPLIT=true：拆键（每 WG 算 KT/NWGT 个 key，跨 WG 合并 row max/sum）
// SPLIT=false：dup（NWG=2，每 WG 都算全部 KT 个 key，只有 WG0 写 P）
// ===========================================================================
template <int NWGT, bool SPLIT>
__global__ void __launch_bounds__(NWGT* 128) mla_dec_wg_kernel(const bf16* __restrict__ qa,
                                                               const bf16* __restrict__ qr,
                                                               const bf16* __restrict__ ckv,
                                                               const bf16* __restrict__ kr,
                                                               float* __restrict__ out, int B,
                                                               int L) {
  constexpr int T = NWGT * 128;
  constexpr int KTH = KT / NWGT;
  constexpr int DVW = DV / NWGT;
  constexpr int NS = SPLIT ? KTH / 8 : KT / 8;  // S 的 n8 tile 数
  constexpr int NWGQ = SPLIT ? KTH : KT;        // QK 的 n（每 WG 算多少 key）

  extern __shared__ __align__(1024) char smem[];
  char* qs = smem;                             // [BM][DK]  SW128
  char* ks = qs + (size_t)BM * DK * 2;         // [KT][DK]  SW128
  char* ps = ks + (size_t)KT * DK * 2;         // [BM][KT]  SW128
  char* vs = ps + (size_t)BM * KT * 2;         // [DV][KT]  SW128（转置）
  float* red = reinterpret_cast<float*>(vs + (size_t)DV * KT * 2);  // [2*NWGT][BM]

  const int head0 = blockIdx.x * BM;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wg = tid >> 7;
  const int W = tid >> 5;
  const int r0 = row0_of(W, lane), r1 = r0 + 8;

  const char* ck_b = reinterpret_cast<const char*>(ckv) + (size_t)b * L * DC * 2;
  const char* kr_b = reinterpret_cast<const char*>(kr) + (size_t)b * L * DR * 2;

  // ---- 载入 Q [BM][DK]（head0+m 行，batch b）----
  for (int idx = tid; idx < BM * (DC / 8); idx += T) {
    const int m = idx / (DC / 8), cu = idx % (DC / 8);
    const int gh = head0 + m;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gh < H) v = *reinterpret_cast<const uint4*>(&qa[((size_t)gh * B + b) * DC + cu * 8]);
    sw128_store16(qs, m, cu * 8, DK, v);
  }
  for (int idx = tid; idx < BM * (DR / 8); idx += T) {
    const int m = idx / (DR / 8), cu = idx % (DR / 8);
    const int gh = head0 + m;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gh < H) v = *reinterpret_cast<const uint4*>(&qr[((size_t)gh * B + b) * DR + cu * 8]);
    sw128_store16(qs, m, DC + cu * 8, DK, v);
  }

  float S[NS * 4];
  float O[DVW / 8 * 4];
#pragma unroll
  for (int i = 0; i < NS * 4; ++i) S[i] = 0.f;
#pragma unroll
  for (int i = 0; i < DVW / 8 * 4; ++i) O[i] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const uint32_t qs_a = smem_u32(qs), ks_a = smem_u32(ks), ps_a = smem_u32(ps), vs_a = smem_u32(vs);
  const uint32_t ks_w = ks_a + (uint32_t)((SPLIT ? wg * (KTH / 8) : 0) * SBO_K);
  const int nbase = SPLIT ? wg * KTH : 0;

  // ---- prologue：载入第 0 块 K/V ----
  for (int idx = tid; idx < KT * (DC / 8); idx += T) {
    const int kv = idx / (DC / 8), cu = idx % (DC / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (kv < L) v = *reinterpret_cast<const uint4*>(ck_b + (size_t)kv * DC * 2 + cu * 16);
    sw128_store16(ks, kv, cu * 8, DK, v);
  }
  for (int idx = tid; idx < KT * (DR / 8); idx += T) {
    const int kv = idx / (DR / 8), cu = idx % (DR / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (kv < L) v = *reinterpret_cast<const uint4*>(kr_b + (size_t)kv * DR * 2 + cu * 16);
    sw128_store16(ks, kv, DC + cu * 8, DK, v);
  }
  for (int i = tid; i < (KT / 8) * DV; i += T) {
    const int kb = i / DV, dv = i % DV;
    uint4 v;
    unsigned short* sp = reinterpret_cast<unsigned short*>(&v);
    const int kk0 = kb * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      sp[j] = (kk0 + j < L)
                  ? *reinterpret_cast<const unsigned short*>(ck_b + (size_t)(kk0 + j) * DC * 2 + dv * 2)
                  : (unsigned short)0;
    sw128_store16(vs, dv, kb * 8, KT, v);
  }
  __syncthreads();

  for (int k0 = 0; k0 < L; k0 += KT) {
    const bool has_next = (k0 + KT < L);
    // ---- QK^T ----
#pragma unroll
    for (int i = 0; i < NS * 4; ++i) S[i] = 0.f;
    wgmma_fence();
    if constexpr (SPLIT) {
#pragma unroll
      for (int s = 0; s < DK / 16; ++s)
        wgmma_qk<NWGQ>(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
                       make_desc_sw128(sw128_k16_addr(ks_w, s), SBO_K));
    } else {
#pragma unroll
      for (int s = 0; s < DK / 16; ++s)
        wgmma_qk<NWGQ>(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
                       make_desc_sw128(sw128_k16_addr(ks_a, s), SBO_K));
    }
    wgmma_commit();
    wgmma_wait0();

    // ---- 尾 tile 掩码 ----
    if (k0 + KT > L) {
#pragma unroll
      for (int t = 0; t < NS; ++t) {
        const int c0 = k0 + nbase + t * 8 + (lane & 3) * 2;
        if (c0 >= L) { S[t * 4 + 0] = -INFINITY; S[t * 4 + 2] = -INFINITY; }
        if (c0 + 1 >= L) { S[t * 4 + 1] = -INFINITY; S[t * 4 + 3] = -INFINITY; }
      }
    }

    // ---- 行 max ----
    float b0 = -INFINITY, b1 = -INFINITY;
#pragma unroll
    for (int t = 0; t < NS; ++t) {
      b0 = fmaxf(b0, fmaxf(S[t * 4 + 0], S[t * 4 + 1]));
      b1 = fmaxf(b1, fmaxf(S[t * 4 + 2], S[t * 4 + 3]));
    }
    b0 = fmaxf(b0, __shfl_xor_sync(~0u, b0, 1));
    b0 = fmaxf(b0, __shfl_xor_sync(~0u, b0, 2));
    b1 = fmaxf(b1, __shfl_xor_sync(~0u, b1, 1));
    b1 = fmaxf(b1, __shfl_xor_sync(~0u, b1, 2));
    float fm0 = b0, fm1 = b1;
    if constexpr (SPLIT) {
      red[wg * BM + 16 * W + r0] = b0;
      red[wg * BM + 16 * W + r1] = b1;
      __syncthreads();
      fm0 = -INFINITY; fm1 = -INFINITY;
#pragma unroll
      for (int e = 0; e < NWGT; ++e) {
        fm0 = fmaxf(fm0, red[e * BM + 16 * W + r0]);
        fm1 = fmaxf(fm1, red[e * BM + 16 * W + r1]);
      }
    }
    const float nm0 = fmaxf(m0, fm0), nm1 = fmaxf(m1, fm1);
    const float al0 = __expf(m0 - nm0), al1 = __expf(m1 - nm1);

    // ---- exp + 行 sum ----
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int t = 0; t < NS; ++t) {
      S[t * 4 + 0] = __expf(S[t * 4 + 0] - nm0);
      S[t * 4 + 1] = __expf(S[t * 4 + 1] - nm0);
      S[t * 4 + 2] = __expf(S[t * 4 + 2] - nm1);
      S[t * 4 + 3] = __expf(S[t * 4 + 3] - nm1);
      s0 += S[t * 4 + 0] + S[t * 4 + 1];
      s1 += S[t * 4 + 2] + S[t * 4 + 3];
    }
    s0 += __shfl_xor_sync(~0u, s0, 1);
    s0 += __shfl_xor_sync(~0u, s0, 2);
    s1 += __shfl_xor_sync(~0u, s1, 1);
    s1 += __shfl_xor_sync(~0u, s1, 2);
    if constexpr (SPLIT) {
      red[(NWGT + wg) * BM + 16 * W + r0] = s0;
      red[(NWGT + wg) * BM + 16 * W + r1] = s1;
    }
    // ---- 写 P ----
    if (SPLIT || wg == 0) {
#pragma unroll
      for (int t = 0; t < NS; ++t) {
        const int col = nbase + t * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(ps + sw128_off(r0, col, KT)) = pack2(S[t * 4 + 0], S[t * 4 + 1]);
        *reinterpret_cast<uint32_t*>(ps + sw128_off(r1, col, KT)) = pack2(S[t * 4 + 2], S[t * 4 + 3]);
      }
    }
    // ---- O rescale ----
#pragma unroll
    for (int t = 0; t < DVW / 8; ++t) {
      O[t * 4 + 0] *= al0;
      O[t * 4 + 1] *= al0;
      O[t * 4 + 2] *= al1;
      O[t * 4 + 3] *= al1;
    }
    // ---- 预取下一块 K（ks 已读完）----
    if (has_next) {
      for (int idx = tid; idx < KT * (DC / 8); idx += T) {
        const int kv = idx / (DC / 8), cu = idx % (DC / 8);
        cp_async16(ks_a + sw128_off(kv, cu * 8, DK),
                   ck_b + (size_t)(k0 + KT + kv) * DC * 2 + cu * 16);
      }
      for (int idx = tid; idx < KT * (DR / 8); idx += T) {
        const int kv = idx / (DR / 8), cu = idx % (DR / 8);
        cp_async16(ks_a + sw128_off(kv, DC + cu * 8, DK),
                   kr_b + (size_t)(k0 + KT + kv) * DR * 2 + cu * 16);
      }
      cp_commit();
    }
    __syncthreads();
    if constexpr (SPLIT) {
      float fs0 = 0.f, fs1 = 0.f;
#pragma unroll
      for (int e = 0; e < NWGT; ++e) {
        fs0 += red[(NWGT + e) * BM + 16 * W + r0];
        fs1 += red[(NWGT + e) * BM + 16 * W + r1];
      }
      l0 = l0 * al0 + fs0;
      l1 = l1 * al1 + fs1;
    } else {
      l0 = l0 * al0 + s0;
      l1 = l1 * al1 + s1;
    }
    m0 = nm0;
    m1 = nm1;

    // ---- PV: O += P @ V（本 WG 的 DVW 列）----
    const uint32_t vs_w = vs_a + (uint32_t)(wg * (DVW / 8) * 1024);
    wgmma_fence();
#pragma unroll
    for (int c = 0; c < KT / 16; ++c)
      wgmma_pv<DVW>(O, make_desc_sw128(sw128_k16_addr(ps_a, c), SBO_P),
                    make_desc_sw128(sw128_k16_addr(vs_w, c), SBO_P));
    wgmma_commit();
    if (has_next) cp_wait0();
    wgmma_wait0();
    // ---- 载入下一块 V ----
    if (false && has_next) {
      for (int i = tid; i < (KT / 8) * DV; i += T) {
        const int kb = i / DV, dv = i % DV;
        uint4 v;
        unsigned short* sp = reinterpret_cast<unsigned short*>(&v);
        const int kk0 = k0 + KT + kb * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          sp[j] = (kk0 + j < L)
                      ? *reinterpret_cast<const unsigned short*>(ck_b + (size_t)(kk0 + j) * DC * 2 + dv * 2)
                      : (unsigned short)0;
        sw128_store16(vs, dv, kb * 8, KT, v);
      }
    }
    __syncthreads();
  }

  // ---- 输出 ----
  const float iv0 = (l0 > 0.f) ? 1.f / l0 : 0.f, iv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  const int qr0 = head0 + r0, qr1 = head0 + r1;
#pragma unroll
  for (int t = 0; t < DVW / 8; ++t) {
    const int col = wg * DVW + t * 8 + (lane & 3) * 2;
    if (qr0 < H) {
      out[((size_t)qr0 * B + b) * DV + col] = O[t * 4 + 0] * iv0;
      out[((size_t)qr0 * B + b) * DV + col + 1] = O[t * 4 + 1] * iv0;
    }
    if (qr1 < H) {
      out[((size_t)qr1 * B + b) * DV + col] = O[t * 4 + 2] * iv1;
      out[((size_t)qr1 * B + b) * DV + col + 1] = O[t * 4 + 3] * iv1;
    }
  }
}

// ===========================================================================
// host 参考
// ===========================================================================
static void decode_row_ref(int h, int b, int B, int L, const std::vector<bf16>& qa,
                           const std::vector<bf16>& qr, const std::vector<bf16>& ckv,
                           const std::vector<bf16>& kr, std::vector<double>& out_row) {
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
}

int main(int argc, char** argv) {
  int B = (argc > 1) ? std::atoi(argv[1]) : 64;
  int L = (argc > 2) ? std::atoi(argv[2]) : 32768;
  const char* which = (argc > 3) ? argv[3] : "all";

  DeviceInfo di = device_info(0);
  print_device_info(di);

  const size_t qaN = (size_t)H * B * DC, qrN = (size_t)H * B * DR;
  const size_t ckN = (size_t)B * L * DC, krN = (size_t)B * L * DR, outN = (size_t)H * B * DV;
  const double flops = 2.0 * H * (DK + DV) * (double)B * L;
  const double bytes_bf16 = (double)B * L * DK * 2;

  std::printf("\nMLA absorbed decode (wgmma): H=%d B=%d L=%d  (DC=%d DR=%d DV=%d DK=%d)\n", H, B, L,
              DC, DR, DV, DK);
  std::printf("FLOPs = %.3f TFLOP ; bf16 KV bytes = %.3f GB (AI=%.0f)\n", flops / 1e12,
              bytes_bf16 / 1e9, flops / bytes_bf16);

  std::vector<bf16> hqa(qaN), hqr(qrN), hck(ckN), hkr(krN);
  std::mt19937 rng(20260922);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto& v : hqa) v = __float2bfloat16(0.2f * dist(rng));
  for (auto& v : hqr) v = __float2bfloat16(0.2f * dist(rng));
  for (size_t i = 0; i < ckN; ++i) {
    float x = 0.2f * dist(rng);
    if ((i * 2654435761u) % 1000 == 0) x *= 20.f;
    hck[i] = __float2bfloat16(x);
  }
  for (auto& v : hkr) v = __float2bfloat16(0.2f * dist(rng));

  bf16 *qa, *qr, *ckv, *kr;
  float* out;
  CUDA_CHECK(cudaMalloc(&qa, qaN * 2));
  CUDA_CHECK(cudaMalloc(&qr, qrN * 2));
  CUDA_CHECK(cudaMalloc(&ckv, ckN * 2));
  CUDA_CHECK(cudaMalloc(&kr, krN * 2));
  CUDA_CHECK(cudaMalloc(&out, outN * 4));
  CUDA_CHECK(cudaMemcpy(qa, hqa.data(), qaN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(qr, hqr.data(), qrN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ckv, hck.data(), ckN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(kr, hkr.data(), krN * 2, cudaMemcpyHostToDevice));

  const int nhb = 3;
  std::vector<int> rhs(nhb), rbs(nhb);
  std::vector<float> refv((size_t)nhb * DV);
  for (int i = 0; i < nhb; ++i) {
    rhs[i] = (i * 61 + 7) % H;
    rbs[i] = (i * 13 + 1) % B;
    std::vector<double> row;
    decode_row_ref(rhs[i], rbs[i], B, L, hqa, hqr, hck, hkr, row);
    for (int dd = 0; dd < DV; ++dd) refv[(size_t)i * DV + dd] = (float)row[dd];
  }

  auto check = [&](const char* tag) {
    std::vector<float> hout(outN);
    CUDA_CHECK(cudaMemcpy(hout.data(), out, outN * 4, cudaMemcpyDeviceToHost));
    double err = 0, base = 0;
    for (int i = 0; i < nhb; ++i)
      for (int dd = 0; dd < DV; ++dd) {
        const double e = refv[(size_t)i * DV + dd];
        err = std::max(err, std::fabs((double)hout[((size_t)rhs[i] * B + rbs[i]) * DV + dd] - e));
        base = std::max(base, std::fabs(e));
      }
    std::printf("     [%-5s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, base,
                err / std::max(base, 1e-6) < 5e-2 ? "OK" : "FAIL");
  };

  auto run = [&](const char* tag, int nwg, bool refs) {
    const size_t shm_base = (size_t)BM * DK * 2 + (size_t)KT * DK * 2 + (size_t)BM * KT * 2 +
                            (size_t)DV * KT * 2 + (size_t)2 * 4 * BM * 4 + 1024;
    dim3 grid(div_up(H, BM), B);
    double fl = flops, by = bytes_bf16;
    int th;
    void* fn = nullptr;
    if (nwg == 4) { fn = (void*)mla_dec_wg_kernel<4, true>; th = 512; }
    else { fn = (void*)mla_dec_wg_kernel<2, true>; th = 256; }
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_base));
    auto launch_fn = [&](void* f, int t) {
      if (t == 512) ((void (*)(const bf16*, const bf16*, const bf16*, const bf16*, float*, int, int))f)
                        <<<grid, 512, shm_base>>>(qa, qr, ckv, kr, out, B, L);
      else ((void (*)(const bf16*, const bf16*, const bf16*, const bf16*, float*, int, int))f)
               <<<grid, 256, shm_base>>>(qa, qr, ckv, kr, out, B, L);
    };
    launch_fn(fn, th);
    CUDA_CHECK_LAST();
    if (refs) check(tag);
    auto runc = [&] { launch_fn(fn, th); };
    double t = bench_ms(runc, 3, 20);
    const double tfl = to_tflops(fl, t);
    const double gbps = by * 1000.0 / t / 1e9;
    std::printf("%-5s split%sg  %9.4f ms  %7.1f GB/s (%5.1f%% HBM)  %7.1f TFLOPS (%5.1f%%)\n", tag,
                (nwg == 4 ? "4" : "2"), t, gbps, 100.0 * gbps / 3350.0, tfl, 100.0 * tfl / 989.0);
  };

  auto run_dup = [&](const char* tag, bool refs) {
    const size_t shm_base = (size_t)BM * DK * 2 + (size_t)KT * DK * 2 + (size_t)BM * KT * 2 +
                            (size_t)DV * KT * 2 + (size_t)2 * 4 * BM * 4 + 1024;
    auto fn = mla_dec_wg_kernel<2, false>;
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_base));
    dim3 grid(div_up(H, BM), B);
    auto runc = [&] { fn<<<grid, 256, shm_base>>>(qa, qr, ckv, kr, out, B, L); };
    runc();
    CUDA_CHECK_LAST();
    if (refs) check(tag);
    double t = bench_ms(runc, 3, 20);
    const double tfl = to_tflops(flops, t);
    const double gbps = bytes_bf16 * 1000.0 / t / 1e9;
    std::printf("%-5s dup2   %9.4f ms  %7.1f GB/s (%5.1f%% HBM)  %7.1f TFLOPS (%5.1f%%)\n", tag, t,
                gbps, 100.0 * gbps / 3350.0, tfl, 100.0 * tfl / 989.0);
  };

  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };
  if (want("bf16") || want("all")) {
    run("v1", 4, true);
    run("v1", 2, true);
    run_dup("v1", true);
  }
  if (want("n2")) run("n2", 2, true);
  if (want("n4")) run("n4", 4, true);
  if (want("prof")) run("p", 4, false);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
