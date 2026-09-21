// 15 MLA 注意力（一）：把 DeepSeek/Kimi 的 MLA 拆成可复用 kernel
//
// MLA（Multi-head Latent Attention）的「吸收（absorb）」推理形式，等价于一个
//   MQA（所有 query head 共享同一份 latent KV），维度：
//     q_abs   : [H, Sq, DC]   （q_nope 经 W_uk_nope 吸收后的 512 维）
//     q_rope  : [H, Sq, DR]   （decoupled RoPE 的那 64 维）
//     c_kv    : [Sk, DC]      （compressed KV latent，所有 head 共享）
//     k_rope  : [Sk, DR]      （共享的 RoPE key）
//     out     : [H, Sq, DV]   （latent 输出；乘 W_uv 后才是 v_head_dim）
//   score[i,j] = q_abs[i]·c_kv[j] + q_rope[i]·k_rope[j]
//   out[i]     = softmax_j(score) · c_kv            （DV = DC = kv_lora_rank）
//
// 三个版本（本系列的「优化阶梯」）：
//   naive     : 一个 warp 负责一个 (query, head)，直接从 global 读 KV。
//               —— 所有 query / head 各自重读一遍 c_kv，读放大 = H。
//   head_reuse: 一个 warp 负责一个 query 的 HG 个 head，KV 只用载入一次寄存器，
//               在 HG 个 head 间复用。读放大 = H/HG。
//   smem      : 一个 block 负责 BQ 个 query × HG 个 head，KV 分块搬进 smem，
//               在所有 query 的 warp 间复用。读放大 = H/HG/BQ。
//
// 运行：scripts/run.sh 15-mla-attn/mla_attn.cu [Sq] [H] [Sk] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>

using bf16 = __nv_bfloat16;

// ---- DeepSeek-V3/V4 的 MLA「吸收」维度（FlashMLA 的 MQA 口径：576/512）----
constexpr int DC = 512;  // kv_lora_rank / NoPE
constexpr int DR = 64;   // decoupled RoPE 维度
constexpr int DV = 512;  // 吸收后 v 的维度 = kv_lora_rank

constexpr int LANE_Q = DC / 32;  // 每个 lane 持有 16 个 q_abs / c_kv 元素
constexpr int LANE_R = DR / 32;  // 每个 lane 持有 2 个 rope 元素

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---- Tensor Core 用的 PTX helpers（同第 13 篇）----
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
__device__ __forceinline__ void gl_mma(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 从 uint4（8 个 bf16）解包成 float
__device__ __forceinline__ void unpack8(uint4 v, float f[8]) {
  const unsigned short* s = reinterpret_cast<const unsigned short*>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) f[i] = __bfloat162float(__ushort_as_bfloat16(s[i]));
}

__device__ __forceinline__ void load16(const bf16* __restrict__ p, float f[16]) {
  unpack8(*reinterpret_cast<const uint4*>(p), f);
  unpack8(*reinterpret_cast<const uint4*>(p + 8), f + 8);
}

__device__ __forceinline__ void load_rope(const bf16* __restrict__ p, float r[2]) {
  r[0] = __bfloat162float(p[0]);
  r[1] = __bfloat162float(p[1]);
}

// online softmax 的一步：更新 m, l, acc[16]
__device__ __forceinline__ void online_update(float& m, float& l, float acc[16],
                                              const float cf[16], float s) {
  const float m_new = fmaxf(m, s);
  const float alpha = __expf(m - m_new);
  const float p = __expf(s - m_new);
  l = l * alpha + p;
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = acc[i] * alpha + p * cf[i];
  m = m_new;
}

// ===========================================================================
// v0：naive —— 一个 warp 一个 (query, head)
// ===========================================================================
__global__ void mla_naive(const bf16* __restrict__ qa, const bf16* __restrict__ qr,
                          const bf16* __restrict__ ckv, const bf16* __restrict__ kr,
                          float* __restrict__ out, int Sq, int Sk) {
  const int q = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x;

  float qaf[16], qrf[2];
  load16(qa + ((size_t)h * Sq + q) * DC + lane * LANE_Q, qaf);
  load_rope(qr + ((size_t)h * Sq + q) * DR + lane * LANE_R, qrf);

  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int k = 0; k < Sk; ++k) {
    float cf[16], rf[2];
    load16(ckv + (size_t)k * DC + lane * LANE_Q, cf);
    load_rope(kr + (size_t)k * DR + lane * LANE_R, rf);

    float part = qrf[0] * rf[0] + qrf[1] * rf[1];
#pragma unroll
    for (int i = 0; i < 16; ++i) part += qaf[i] * cf[i];
    const float s = warp_reduce_sum(part);
    online_update(m, l, acc, cf, s);
  }

  const float inv = 1.f / l;
  float* ob = out + ((size_t)h * Sq + q) * DV + lane * LANE_Q;
#pragma unroll
  for (int i = 0; i < 16; ++i) ob[i] = acc[i] * inv;
}

// ===========================================================================
// v1：head_reuse —— 一个 warp 一个 query，HG 个 head；KV 载入一次复用 HG 次
// ===========================================================================
template <int HG>
__global__ void mla_head_reuse(const bf16* __restrict__ qa, const bf16* __restrict__ qr,
                               const bf16* __restrict__ ckv, const bf16* __restrict__ kr,
                               float* __restrict__ out, int Sq, int Sk) {
  const int q = blockIdx.x;
  const int h0 = blockIdx.y * HG;
  const int lane = threadIdx.x;

  float qaf[HG][16], qrf[HG][2];
#pragma unroll
  for (int hh = 0; hh < HG; ++hh) {
    const size_t qoff = ((size_t)(h0 + hh) * Sq + q);
    load16(qa + qoff * DC + lane * LANE_Q, qaf[hh]);
    load_rope(qr + qoff * DR + lane * LANE_R, qrf[hh]);
  }

  float acc[HG][16];
  float m[HG], l[HG];
#pragma unroll
  for (int hh = 0; hh < HG; ++hh) {
    m[hh] = -INFINITY;
    l[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[hh][i] = 0.f;
  }

  for (int k = 0; k < Sk; ++k) {
    float cf[16], rf[2];
    load16(ckv + (size_t)k * DC + lane * LANE_Q, cf);
    load_rope(kr + (size_t)k * DR + lane * LANE_R, rf);

    const float rope = rf[0], rope2 = rf[1];
    // 先算 QK^T 需要的 partial（q_abs 部分每 head 不同）
    float part[HG];
#pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
      float s = qrf[hh][0] * rope + qrf[hh][1] * rope2;
#pragma unroll
      for (int i = 0; i < 16; ++i) s += qaf[hh][i] * cf[i];
      part[hh] = s;
    }
    // 同一份 cf 复用给 HG 个 head 的 value 累加
#pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
      const float s = warp_reduce_sum(part[hh]);
      online_update(m[hh], l[hh], acc[hh], cf, s);
    }
  }

#pragma unroll
  for (int hh = 0; hh < HG; ++hh) {
    const float inv = 1.f / l[hh];
    float* ob = out + ((size_t)(h0 + hh) * Sq + q) * DV + lane * LANE_Q;
#pragma unroll
    for (int i = 0; i < 16; ++i) ob[i] = acc[hh][i] * inv;
  }
}

// ===========================================================================
// v2：smem —— 一个 block 负责 BQ 个 query（每 warp 一个）× HG 个 head，
//      KV 分块搬进 smem，在 BQ 个 warp 间复用
// ===========================================================================
template <int HG, int BQ, int KT>
__global__ void mla_smem(const bf16* __restrict__ qa, const bf16* __restrict__ qr,
                         const bf16* __restrict__ ckv, const bf16* __restrict__ kr,
                         float* __restrict__ out, int Sq, int Sk) {
  constexpr int DCP = DC + 8;  // padding 缓解 smem bank conflict
  __shared__ bf16 cks[KT][DCP];
  __shared__ bf16 krs[KT][DR];

  const int q0 = blockIdx.x * BQ;
  const int h0 = blockIdx.y * HG;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q = q0 + w;
  const bool q_valid = q < Sq;

  float qaf[HG][16], qrf[HG][2];
#pragma unroll
  for (int hh = 0; hh < HG; ++hh) {
    if (q_valid) {
      const size_t qoff = ((size_t)(h0 + hh) * Sq + q);
      load16(qa + qoff * DC + lane * LANE_Q, qaf[hh]);
      load_rope(qr + qoff * DR + lane * LANE_R, qrf[hh]);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) qaf[hh][i] = 0.f;
      qrf[hh][0] = qrf[hh][1] = 0.f;
    }
  }

  float acc[HG][16];
  float m[HG], l[HG];
#pragma unroll
  for (int hh = 0; hh < HG; ++hh) {
    m[hh] = -INFINITY;
    l[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[hh][i] = 0.f;
  }

  for (int k0 = 0; k0 < Sk; k0 += KT) {
    // 协作把 KV tile 搬进 smem（8 个 bf16 = 16B 一次）
    for (int i = threadIdx.x; i < KT * DC / 8; i += blockDim.x) {
      const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
      *reinterpret_cast<uint4*>(&cks[row][c8]) =
          *reinterpret_cast<const uint4*>(&ckv[(size_t)(k0 + row) * DC + c8]);
    }
    for (int i = threadIdx.x; i < KT * DR / 8; i += blockDim.x) {
      const int row = i / (DR / 8), c8 = (i % (DR / 8)) * 8;
      *reinterpret_cast<uint4*>(&krs[row][c8]) =
          *reinterpret_cast<const uint4*>(&kr[(size_t)(k0 + row) * DR + c8]);
    }
    __syncthreads();

    const int kt = min(KT, Sk - k0);
    for (int k = 0; k < kt; ++k) {
      float cf[16], rf[2];
      load16(&cks[k][lane * LANE_Q], cf);
      load_rope(&krs[k][lane * LANE_R], rf);
      const float rope = rf[0], rope2 = rf[1];

      if (q_valid) {
#pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
          float s = qrf[hh][0] * rope + qrf[hh][1] * rope2;
#pragma unroll
          for (int i = 0; i < 16; ++i) s += qaf[hh][i] * cf[i];
          s = warp_reduce_sum(s);
          online_update(m[hh], l[hh], acc[hh], cf, s);
        }
      }
    }
    __syncthreads();
  }

  if (!q_valid) return;
#pragma unroll
  for (int hh = 0; hh < HG; ++hh) {
    const float inv = 1.f / l[hh];
    float* ob = out + ((size_t)(h0 + hh) * Sq + q) * DV + lane * LANE_Q;
#pragma unroll
    for (int i = 0; i < 16; ++i) ob[i] = acc[hh][i] * inv;
  }
}

// ===========================================================================
// v3：Tensor Core —— 把 MLA 拆成 QK^T / softmax / PV 三个 GEMM+kernel
//
//   QK^T : S[h] = Qcat[h] @ K^T   （Qcat=[q_abs|q_rope] [Sq,576], K^T [576,Sk]）
//   softmax : 逐行 softmax，输出 bf16 的 P
//   PV   : O[h] = P[h] @ c_kv      （c_kv [Sk,512]）
//
// 结构照搬第 13/14 篇的 mma.m16n8k16 + ldmatrix + cp.async 双缓冲 GEMM。
// 代价：要把 S/P 物化到显存（Sq·Sk·H·(4+2) 字节），所以只适合中等 Sq。
// ===========================================================================
constexpr int GBM = 128, GBN = 128, GBK = 32, GT = 256;
constexpr int GWM = 4, GWN = 2;
constexpr int GWARP_M = GBM / GWM, GWARP_N = GBN / GWN;
constexpr int GMTM = GWARP_M / 16, GMTN = GWARP_N / 8;
constexpr int GASP = GBK + 8, GBNP = GBN + 8;

__device__ __forceinline__ void gload_tiles(const bf16* __restrict__ A,
                                            const bf16* __restrict__ B, bf16 (*As)[GASP],
                                            bf16 (*Bs)[GBNP], int M, int N, int K, int block_row,
                                            int block_col, int k0) {
  const int t = threadIdx.x;
  for (int i = t; i < GBM * GBK / 8; i += GT) {
    const int row = i / (GBK / 8), c8 = (i % (GBK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gr < M && gc + 7 < K) v = *reinterpret_cast<const uint4*>(&A[(size_t)gr * K + gc]);
    *reinterpret_cast<uint4*>(&As[row][c8]) = v;
  }
  for (int i = t; i < GBK * GBN / 8; i += GT) {
    const int row = i / (GBN / 8), c8 = (i % (GBN / 8)) * 8;
    const int gr = k0 + row, gc = block_col + c8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gr < K && gc + 7 < N) v = *reinterpret_cast<const uint4*>(&B[(size_t)gr * N + gc]);
    *reinterpret_cast<uint4*>(&Bs[row][c8]) = v;
  }
}

__device__ __forceinline__ void gprefetch(const bf16* __restrict__ A,
                                          const bf16* __restrict__ B, bf16 (*As)[GASP],
                                          bf16 (*Bs)[GBNP], int M, int N, int K, int block_row,
                                          int block_col, int k0) {
  const int t = threadIdx.x;
  for (int i = t; i < GBM * GBK / 8; i += GT) {
    const int row = i / (GBK / 8), c8 = (i % (GBK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    if (gr < M && gc + 7 < K) {
      __pipeline_memcpy_async(&As[row][c8], &A[(size_t)gr * K + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&As[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  for (int i = t; i < GBK * GBN / 8; i += GT) {
    const int row = i / (GBN / 8), c8 = (i % (GBN / 8)) * 8;
    const int gr = k0 + row, gc = block_col + c8;
    if (gr < K && gc + 7 < N) {
      __pipeline_memcpy_async(&Bs[row][c8], &B[(size_t)gr * N + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&Bs[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  __pipeline_commit();
}

__device__ __forceinline__ void gmma_stage(const bf16 (*As)[GASP], const bf16 (*Bs)[GBNP],
                                           float acc[GMTM][GMTN][4], int warp_row, int warp_col) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kk = 0; kk < GBK / 16; ++kk) {
    uint32_t a[GMTM][4];
#pragma unroll
    for (int i = 0; i < GMTM; ++i) {
      const int row = (lane & 15), col = (lane >> 4) * 8;
      ldmatrix_x4(smem_u32(&As[warp_row * GWARP_M + i * 16 + row][kk * 16 + col]), a[i]);
    }
    uint32_t b[GMTN][2];
#pragma unroll
    for (int g = 0; g < GMTN / 2; ++g) {
      const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int col = (lane >> 4) * 8;
      uint32_t d[4];
      ldmatrix_x4_trans(smem_u32(&Bs[kk * 16 + row][warp_col * GWARP_N + g * 16 + col]), d);
      b[g * 2][0] = d[0];
      b[g * 2][1] = d[1];
      b[g * 2 + 1][0] = d[2];
      b[g * 2 + 1][1] = d[3];
    }
#pragma unroll
    for (int i = 0; i < GMTM; ++i)
#pragma unroll
      for (int j = 0; j < GMTN; ++j) gl_mma(acc[i][j], a[i], b[j]);
  }
}

__device__ __forceinline__ void gzero(float acc[GMTM][GMTN][4]) {
#pragma unroll
  for (int i = 0; i < GMTM; ++i)
#pragma unroll
    for (int j = 0; j < GMTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
}

__device__ __forceinline__ void gstore(float* __restrict__ C, const float acc[GMTM][GMTN][4],
                                       int N, int block_row, int block_col, int warp_row,
                                       int warp_col) {
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int i = 0; i < GMTM; ++i)
#pragma unroll
    for (int j = 0; j < GMTN; ++j) {
      const int r0 = block_row + warp_row * GWARP_M + i * 16 + group;
      const int c0 = block_col + warp_col * GWARP_N + j * 8 + tig * 2;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + (q >= 2 ? 8 : 0);
        const int c = c0 + (q & 1);
        if (r < 0) continue;
        C[(size_t)r * N + c] = acc[i][j][q];
      }
    }
}

// batched over blockIdx.z：A/C 按 head 偏移，B 所有 head 共享（MQA 的 KV 只读一份）
__global__ void gemm_mma_pipe(const bf16* __restrict__ A, const bf16* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
  const int hz = blockIdx.z;
  A += (size_t)hz * M * K;
  C += (size_t)hz * M * N;
  __shared__ bf16 As[2][GBM][GASP];
  __shared__ bf16 Bs[2][GBK][GBNP];
  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / GWN, warp_col = wid % GWN;
  const int block_row = blockIdx.y * GBM, block_col = blockIdx.x * GBN;
  float acc[GMTM][GMTN][4];
  gzero(acc);
  gprefetch(A, B, As[0], Bs[0], M, N, K, block_row, block_col, 0);
  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += GBK) {
    const int next = k0 + GBK;
    if (next < K) {
      gprefetch(A, B, As[stage ^ 1], Bs[stage ^ 1], M, N, K, block_row, block_col, next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(next < K ? 1 : 0);
    __syncthreads();
    gmma_stage(As[stage], Bs[stage], acc, warp_row, warp_col);
    __syncthreads();
    stage ^= 1;
  }
  gstore(C, acc, N, block_row, block_col, warp_row, warp_col);
}

// 逐行 softmax -> bf16
__global__ void softmax_rows_bf16(const float* __restrict__ in, bf16* __restrict__ out, int rows,
                                  int cols) {
  const int r = blockIdx.x;
  const float* x = in + (size_t)r * cols;
  bf16* y = out + (size_t)r * cols;
  __shared__ float red[8];
  const int t = threadIdx.x;
  float mx = -INFINITY;
  for (int j = t; j < cols; j += blockDim.x) mx = fmaxf(mx, x[j]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((t & 31) == 0) red[t >> 5] = mx;
  __syncthreads();
  if (t < 8) {
    float v = red[t];
    for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xff, v, o));
    red[t] = v;
  }
  __syncthreads();
  mx = red[0];
  float sum = 0.f;
  for (int j = t; j < cols; j += blockDim.x) sum += __expf(x[j] - mx);
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((t & 31) == 0) red[t >> 5] = sum;
  __syncthreads();
  if (t < 8) {
    float v = red[t];
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xff, v, o);
    red[t] = v;
  }
  __syncthreads();
  const float inv = 1.f / red[0];
  for (int j = t; j < cols; j += blockDim.x) y[j] = __float2bfloat16(__expf(x[j] - mx) * inv);
}

// ===========================================================================
// host
// ===========================================================================
static double mla_row_ref(int h, int q, int Sq, int Sk, const std::vector<bf16>& qa,
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
  double sum = 0, var = 0;
  for (int k = 0; k < Sk; ++k) {
    s[k] = std::exp(s[k] - mx);
    sum += s[k];
    var += s[k] * s[k];
  }
  out_row.assign(DV, 0.0);
  for (int k = 0; k < Sk; ++k) {
    const double p = s[k] / sum;
    for (int d = 0; d < DV; ++d)
      out_row[d] += p * (double)__bfloat162float(ckv[(size_t)k * DC + d]);
  }
  (void)var;
  return sum;
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

  std::printf("\nMLA absorbed attention (MQA): H=%d Sq=%d Sk=%d  DC=%d DR=%d DV=%d\n", H, Sq,
              Sk, DC, DR, DV);
  std::printf("FLOPs = %.2f GFLOP\n\n", flops / 1e9);

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

  auto check = [&](const char* tag, const float* dev = nullptr) {
    CUDA_CHECK(cudaMemcpy(hout.data(), dev ? dev : out, outN * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    const int n = 3;
    for (int i = 0; i < n; ++i) {
      const int h = (i * 61 + 7) % H, q = (i * 197 + 3) % Sq;
      std::vector<double> row;
      mla_row_ref(h, q, Sq, Sk, hqa, hqr, hck, hkr, row);
      for (int dd = 0; dd < DV; ++dd) {
        const double e = row[dd];
        err = std::max(err, std::fabs((double)hout[((size_t)h * Sq + q) * DV + dd] - e));
        ref = std::max(ref, std::fabs(e));
      }
    }
    std::printf("  [%-10s] sampled max_abs_err = %.3e (ref~%.3f) %s\n", tag, err, ref,
                err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  };

  auto report = [&](const char* tag, double ms, double bytes) {
    const double tflops = to_tflops(flops, ms);
    const double gbps = to_gbps(bytes, ms);
    std::printf("%-11s %9.4f ms  %8.2f TFLOPS (%5.1f%% peak)  %8.1f GB/s  (read %.2f GB, amp %.1fx)\n",
                tag, ms, tflops, 100.0 * tflops / 989.0, gbps, bytes / 1e9,
                bytes / ((double)Sk * (DC + DR) * 2.0));
  };

  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };

  // 理论最少读取量：c_kv + k_rope 各读一遍
  const double min_bytes = (double)Sk * (DC + DR) * 2.0;

  if (want("naive")) {
    // naive 读放大 = H，大 shape 会慢到不可用，提示用户用小 shape
    const double bytes = (double)Sq * H * Sk * (DC + DR) * 2.0;
    dim3 grid(Sq, H);
    mla_naive<<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk);
    CUDA_CHECK_LAST();
    check("naive");
    double t = bench_ms([&] { mla_naive<<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk); }, 3, 10);
    report("naive", t, bytes);
  }

  if (want("head4")) {
    constexpr int HG = 4;
    const double bytes = (double)Sq * (H / HG) * Sk * (DC + DR) * 2.0;
    dim3 grid(Sq, H / HG);
    mla_head_reuse<HG><<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk);
    CUDA_CHECK_LAST();
    check("head4");
    double t = bench_ms([&] { mla_head_reuse<HG><<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk); }, 3, 10);
    report("head4", t, bytes);
  }

  if (want("head2") || want("head8")) {
    const bool h2 = want("head2");
    if (h2) {
      constexpr int HG = 2;
      const double bytes = (double)Sq * (H / HG) * Sk * (DC + DR) * 2.0;
      dim3 grid(Sq, H / HG);
      mla_head_reuse<HG><<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk);
      CUDA_CHECK_LAST();
      check("head2");
      double t = bench_ms([&] { mla_head_reuse<HG><<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk); }, 3, 10);
      report("head2", t, bytes);
    } else {
      constexpr int HG = 8;
      const double bytes = (double)Sq * (H / HG) * Sk * (DC + DR) * 2.0;
      dim3 grid(Sq, H / HG);
      mla_head_reuse<HG><<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk);
      CUDA_CHECK_LAST();
      check("head8");
      double t = bench_ms([&] { mla_head_reuse<HG><<<grid, 32>>>(qa, qr, ckv, kr, out, Sq, Sk); }, 3, 10);
      report("head8", t, bytes);
    }
  }

  if (want("smem")) {
    constexpr int HG = 4, BQ = 4, KT = 32;
    const double bytes = (double)div_up(Sq, BQ) * (H / HG) * Sk * (DC + DR) * 2.0;
    dim3 grid(div_up(Sq, BQ), H / HG);
    mla_smem<HG, BQ, KT><<<grid, 32 * BQ>>>(qa, qr, ckv, kr, out, Sq, Sk);
    CUDA_CHECK_LAST();
    check("smem");
    double t = bench_ms([&] { mla_smem<HG, BQ, KT><<<grid, 32 * BQ>>>(qa, qr, ckv, kr, out, Sq, Sk); }, 3, 10);
    report("smem", t, bytes);
  }

  if (want("smem8")) {
    constexpr int HG = 4, BQ = 8, KT = 32;
    const double bytes = (double)div_up(Sq, BQ) * (H / HG) * Sk * (DC + DR) * 2.0;
    dim3 grid(div_up(Sq, BQ), H / HG);
    mla_smem<HG, BQ, KT><<<grid, 32 * BQ>>>(qa, qr, ckv, kr, out, Sq, Sk);
    CUDA_CHECK_LAST();
    check("smem8");
    double t = bench_ms([&] { mla_smem<HG, BQ, KT><<<grid, 32 * BQ>>>(qa, qr, ckv, kr, out, Sq, Sk); }, 3, 10);
    report("smem8", t, bytes);
  }

  std::printf("\n(理论最少读取量 = %.3f GB，即 c_kv+k_rope 只读一遍)\n", min_bytes / 1e9);

  if (want("tc")) {
    // Tensor Core 三 kernel：QK^T / softmax / PV
    const int DK = DC + DR;
    if (Sq % GBM || Sk % GBN || DK % GBK || Sk % GBK || DC % GBN) {
      std::printf("tc: shape 需满足 Sq%%%d=0, Sk%%%d=0, DC%%%d=0\n", GBM, GBN, GBN);
    } else {
      // 拼出 Qcat[H,Sq,576] = [q_abs | q_rope] 与 K^T[576,Sk] = [c_kv^T ; k_rope^T]
      std::vector<bf16> hqcat((size_t)H * Sq * DK), hkt((size_t)DK * Sk);
      for (int h = 0; h < H; ++h)
        for (int q = 0; q < Sq; ++q) {
          const size_t src = ((size_t)h * Sq + q);
          bf16* dst = &hqcat[src * DK];
          for (int d = 0; d < DC; ++d) dst[d] = hqa[src * DC + d];
          for (int r = 0; r < DR; ++r) dst[DC + r] = hqr[src * DR + r];
        }
      for (int k = 0; k < Sk; ++k)
        for (int d = 0; d < DC; ++d) hkt[(size_t)d * Sk + k] = hck[(size_t)k * DC + d];
      for (int k = 0; k < Sk; ++k)
        for (int r = 0; r < DR; ++r) hkt[(size_t)(DC + r) * Sk + k] = hkr[(size_t)k * DR + r];

      bf16* qcat;
      bf16* kt;
      float* sbuf;
      bf16* pbuf;
      float* obuf;
      CUDA_CHECK(cudaMalloc(&qcat, hqcat.size() * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc(&kt, hkt.size() * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc(&sbuf, (size_t)H * Sq * Sk * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&pbuf, (size_t)H * Sq * Sk * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc(&obuf, outN * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(qcat, hqcat.data(), hqcat.size() * sizeof(bf16), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(kt, hkt.data(), hkt.size() * sizeof(bf16), cudaMemcpyHostToDevice));

      dim3 gblock(GT);

      auto run_tc = [&] {
        gemm_mma_pipe<<<dim3(Sk / GBN, Sq / GBM, H), gblock>>>(qcat, kt, sbuf, Sq, Sk, DK);
        softmax_rows_bf16<<<H * Sq, 256>>>(sbuf, pbuf, H * Sq, Sk);
        gemm_mma_pipe<<<dim3(DC / GBN, Sq / GBM, H), gblock>>>(pbuf, ckv, obuf, Sq, DC, Sk);
      };
      run_tc();
      CUDA_CHECK_LAST();
      check("tc", obuf);

      const double qk_ms = bench_ms([&] {
        gemm_mma_pipe<<<dim3(Sk / GBN, Sq / GBM, H), gblock>>>(qcat, kt, sbuf, Sq, Sk, DK);
      }, 3, 10);
      const double sm_ms = bench_ms([&] {
        softmax_rows_bf16<<<H * Sq, 256>>>(sbuf, pbuf, H * Sq, Sk);
      }, 3, 10);
      const double pv_ms = bench_ms([&] {
        gemm_mma_pipe<<<dim3(DC / GBN, Sq / GBM, H), gblock>>>(pbuf, ckv, obuf, Sq, DC, Sk);
      }, 3, 10);
      const double all_ms = bench_ms([&] { run_tc(); }, 3, 10);
      const double qk_flops = 2.0 * Sq * Sk * DK * H;
      const double pv_flops = 2.0 * Sq * DC * Sk * H;
      std::printf("tc(qk)      %9.4f ms  %8.2f TFLOPS\n", qk_ms, to_tflops(qk_flops, qk_ms));
      std::printf("tc(softmax) %9.4f ms\n", sm_ms);
      std::printf("tc(pv)      %9.4f ms  %8.2f TFLOPS\n", pv_ms, to_tflops(pv_flops, pv_ms));
      std::printf("tc(total)   %9.4f ms  %8.2f TFLOPS (%5.1f%% peak)\n", all_ms,
                  to_tflops(flops, all_ms), 100.0 * to_tflops(flops, all_ms) / 989.0);
      const double sp_bytes =
          (double)H * Sq * Sk * (4.0 + 2.0) * 2.0;  // S 写+读、P 写+读
      std::printf("   >> S/P 物化流量 = %.2f GB，按 %.0f GB/s 约 %.2f ms\n", sp_bytes / 1e9,
                  d.mem_bw_gbps, sp_bytes / (d.mem_bw_gbps * 1e6));

      CUDA_CHECK(cudaFree(qcat));
      CUDA_CHECK(cudaFree(kt));
      CUDA_CHECK(cudaFree(sbuf));
      CUDA_CHECK(cudaFree(pbuf));
      CUDA_CHECK(cudaFree(obuf));
    }
  }

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
