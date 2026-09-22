// =============================================================================
// fa_bwd_bf16_kernels.cuh —— FlashAttention 反向（bf16）**两文件版的 device 部分**
// =============================================================================
// 由单文件版 `fa_bwd_bf16_onefile.cu` 拆分而来（P2-2），device 代码与单文件**逐字一致**：
//   * preprocess_kernel：逐行算 LSE 与 delta=rowsum(dO∘O)
//   * fa_bwd_bf16_kernel：1colblock 反向主 kernel（recompute P，fp32 累加）
//   * convert_kernel：fp32 累加缓冲 → bf16 输出
// host 侧（npy 读取 / launcher / 自测）见 `fa_bwd_bf16_main.cu`。
//
// 数学（见 docs/00 与 ref_impl.py）：
//   S  = scale·QKᵀ ; P = softmax(S) ; D = rowsum(dO∘O)
//   dP = dO Vᵀ ; dS = P∘(dP−D) ; dV = Pᵀ dO ; dQ = scale·dS K ; dK = scale·dSᵀ Q
//
// bf16 专属优化（P2-1）：K/V 的 smem 行距 +2 元素（256→260B），消 9.5-way bank
// conflict（89% 多余 wavefront）。详见 ROADMAP P2-1 与 docs/01b-bf16-bwd-impl.md。
// =============================================================================

#ifndef FA_BWD_BF16_KERNELS_CUH_
#define FA_BWD_BF16_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int kHeadDim = 128;   // 本版本固定 head_dim=128
static constexpr int BM       = 64;    // Q 块行数
static constexpr int BN       = 32;    // K/V 块行数
static constexpr int THREADS  = 128;   // 4 warps
static constexpr int WM_ROWS  = BM / 4;  // 每个 warp 负责的 Q 行数 = 16
static constexpr int WN_ROWS  = BN / 4;  // 每个 warp 负责的 K/V 行数 = 8

// K/V 的 smem 行距 padding：kHeadDim=128 个 bf16 = 256B，是 128B bank 周期的整数倍，
// 于是 QKᵀ/dP 循环里「lane↔K/V 行」的读（跨 lane 步长 256B）会全部落到同一 bank
// （ncu：9.5-way、89% 多余 wavefront）。行距 +2 个元素（+4B）后跨 lane 步长 260B，
// 65 words mod 32 = 1 ⇒ 32 个 lane 恰好铺满 32 个 bank，冲突归零。
// 注：fp16 版因为 ptxas 用 LDS.64/HADD2.F32 成对读，冲突只有 4.6-way，未 padding；
// bf16 标量生成（LDS.U16 + SHF）把它放大到 ~2.9× 慢，故这里显式 padding。
static constexpr int kKVStride = kHeadDim + 2;  // K/V 行距（元素数）

using bf16 = __nv_bfloat16;

// 动态 smem 布局（字节）：
//   Qs[BM*HD] + Ks[BN*KVSTRIDE] + Vs[BN*KVSTRIDE] + dOs[BM*HD]   （bf16）
//   Ss[BM*BN] + Ps[BM*BN]                                        （float）
//   dQs[BM*HD]                                                   （float）
static constexpr int SMEM_BYTES =
    (BM * kHeadDim + BN * kKVStride + BN * kKVStride + BM * kHeadDim) * (int)sizeof(bf16) +
    (BM * BN + BM * BN) * (int)sizeof(float) +
    (BM * kHeadDim) * (int)sizeof(float);

// =============================================================================
// 1) preprocess：逐行算 LSE 与 delta=rowsum(dO∘O)
//   grid = (S, H, B)，block = THREADS
// =============================================================================
__global__ void preprocess_kernel(const bf16* __restrict__ q,
                                  const bf16* __restrict__ k,
                                  const bf16* __restrict__ o,
                                  const bf16* __restrict__ do_,
                                  float* __restrict__ delta,   // [B*S*H]
                                  float* __restrict__ lse,     // [B*S*H]
                                  int S, int H, float scale, int causal) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const bf16* qr = q + row * kHeadDim;

  // --- online softmax：单趟求 (m, l)，无需物化整行 S ---
  float m = -INFINITY;
  float l = 0.f;
  const int jmax = causal ? (s + 1) : S;   // causal：只算 j <= s
  for (int j = tid; j < jmax; j += blockDim.x) {
    const bf16* kr = k + (((size_t)(b * S + j)) * H + h) * kHeadDim;
    float dot = 0.f;
#pragma unroll 8
    for (int d = 0; d < kHeadDim; ++d)
      dot += __bfloat162float(qr[d]) * __bfloat162float(kr[d]);
    dot *= scale;
    float mn = fmaxf(m, dot);
    l = l * expf(m - mn) + expf(dot - mn);
    m = mn;
  }

  // --- block 归约 (m, l)：按 online-softmax 的合并规则 ---
  __shared__ float sh_m[THREADS];
  __shared__ float sh_l[THREADS];
  sh_m[tid] = m;
  sh_l[tid] = l;
  __syncthreads();
  for (int off = THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) {
      float m1 = sh_m[tid], l1 = sh_l[tid];
      float m2 = sh_m[tid + off], l2 = sh_l[tid + off];
      float mn = fmaxf(m1, m2);
      float c1 = (m1 == -INFINITY) ? 0.f : l1 * expf(m1 - mn);
      float c2 = (m2 == -INFINITY) ? 0.f : l2 * expf(m2 - mn);
      sh_m[tid] = mn;
      sh_l[tid] = c1 + c2;
    }
    __syncthreads();
  }

  // --- delta = sum_d O∘dO ---
  float dp = 0.f;
  const bf16* orow = o + row * kHeadDim;
  const bf16* dorow = do_ + row * kHeadDim;
  for (int d = tid; d < kHeadDim; d += blockDim.x)
    dp += __bfloat162float(orow[d]) * __bfloat162float(dorow[d]);
  __shared__ float sh_delta[THREADS];
  sh_delta[tid] = dp;
  __syncthreads();
  for (int off = THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) sh_delta[tid] += sh_delta[tid + off];
    __syncthreads();
  }

  if (tid == 0) {
    delta[row] = sh_delta[0];
    lse[row] = sh_m[0] + logf(sh_l[0]);
  }
}

// =============================================================================
// 2) main kernel：1colblock 反向
//   grid = (ceil(S/BM), H, B)，block = THREADS
//   dq_acc 直接写（每个 Q 块独占行）；dk_acc/dv_acc 用 atomicAdd。
// =============================================================================
__global__ void __launch_bounds__(THREADS)
fa_bwd_bf16_kernel(const bf16* __restrict__ q,
                   const bf16* __restrict__ k,
                   const bf16* __restrict__ v,
                   const bf16* __restrict__ do_,
                   const float* __restrict__ delta,
                   const float* __restrict__ lse,
                   float* __restrict__ dq_acc,
                   float* __restrict__ dk_acc,
                   float* __restrict__ dv_acc,
                   int S, int H, float scale, int causal) {
  extern __shared__ char smem[];
  bf16* Qs  = reinterpret_cast<bf16*>(smem);
  bf16* Ks  = Qs + BM * kHeadDim;
  bf16* Vs  = Ks + BN * kKVStride;
  bf16* dOs = Vs + BN * kKVStride;
  float* Ss = reinterpret_cast<float*>(dOs + BM * kHeadDim);  // 先存 S，后覆盖为 dS
  float* Ps = Ss + BM * BN;
  float* dQs = Ps + BM * BN;                                   // dQ 的 smem 累加器

  const int mblk = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int m0 = mblk * BM;

  // ---- 载入本 Q 块的 Q 与 dO（越界补 0）----
  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim, d = i % kHeadDim;
    int qi = m0 + r;
    bf16 qv = __float2bfloat16(0.f), ov = __float2bfloat16(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * kHeadDim + d;
      qv = q[idx];
      ov = do_[idx];
    }
    Qs[i] = qv;
    dOs[i] = ov;
  }
  for (int i = tid; i < BM * kHeadDim; i += THREADS) dQs[i] = 0.f;
  __syncthreads();

  // causal：只需处理到本 Q 块最后一行的列；否则到 S。
  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;

    // ---- 载入 K/V 块（行距 kKVStride，padding 消 bank conflict）----
    for (int i = tid; i < BN * kHeadDim; i += THREADS) {
      int j = i / kHeadDim, d = i % kHeadDim;
      int jg = j0 + j;
      bf16 kv = __float2bfloat16(0.f), vv = __float2bfloat16(0.f);
      if (jg < S) {
        size_t idx = (((size_t)(b * S + jg)) * H + h) * kHeadDim + d;
        kv = k[idx];
        vv = v[idx];
      }
      Ks[j * kKVStride + d] = kv;
      Vs[j * kKVStride + d] = vv;
    }
    __syncthreads();

    // ---- S = scale·QKᵀ，P = exp(S − LSE)；同一个 warp 负责 WM_ROWS 个 Q 行，
    //      lane 对应 BN 个列中的一列（BN=32=warpSize）----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      float dot = 0.f;
      if (qi < S) {
        const bf16* qrow = Qs + r * kHeadDim;
        const bf16* krow = Ks + lane * kKVStride;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d)
          dot += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
        dot *= scale;
      }
      int jg = j0 + lane;
      float p = 0.f;
      if (qi < S && jg < S && !(causal && jg > qi))
        p = expf(dot - lse[((size_t)(b * S + qi)) * H + h]);
      Ps[r * BN + lane] = p;
    }
    __syncthreads();

    // ---- dP = dO Vᵀ；dS = P∘(dP − D)（覆盖写入 Ss）----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      float dp = 0.f;
      if (qi < S) {
        const bf16* drow = dOs + r * kHeadDim;
        const bf16* vrow = Vs + lane * kKVStride;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d)
          dp += __bfloat162float(drow[d]) * __bfloat162float(vrow[d]);
      }
      float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;
      Ss[r * BN + lane] = Ps[r * BN + lane] * (dp - del);
    }
    __syncthreads();

    // ---- dV = Pᵀ dO：warp 负责 WN_ROWS 个 K 行，lane 覆盖 4 个 head_dim 槽 ----
#pragma unroll
    for (int jj = 0; jj < WN_ROWS; ++jj) {
      int j = warp * WN_ROWS + jj;
      int jg = j0 + j;
      if (jg >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float p = Ps[i * BN + j];          // 同一列，warp 内广播
        if (p == 0.f) continue;
        const bf16* drow = dOs + i * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += p * __bfloat162float(drow[d]);
        }
      }
      float* base = dv_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) atomicAdd(base + lane + 32 * kk, acc[kk]);
    }

    // ---- dK = scale·dSᵀ Q（dS 在 Ss）----
#pragma unroll
    for (int jj = 0; jj < WN_ROWS; ++jj) {
      int j = warp * WN_ROWS + jj;
      int jg = j0 + j;
      if (jg >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float ds = Ss[i * BN + j];
        if (ds == 0.f) continue;
        const bf16* qrow = Qs + i * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * __bfloat162float(qrow[d]);
        }
      }
      float* base = dk_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        atomicAdd(base + lane + 32 * kk, acc[kk] * scale);
    }

    // ---- dQ += scale·dS K：累积到 smem dQs（本 CTA 独占 Q 行，无需 atomic）----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      if (qi >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int j = 0; j < BN; ++j) {
        float ds = Ss[r * BN + j];
        if (ds == 0.f) continue;
        const bf16* krow = Ks + j * kKVStride;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * __bfloat162float(krow[d]);
        }
      }
      float* dqr = dQs + r * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        dqr[lane + 32 * kk] += acc[kk] * scale;
    }
    __syncthreads();
  }

  // ---- 写回 dQ（fp32 缓冲，稍后 convert）----
  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim;
    int qi = m0 + r;
    if (qi < S)
      dq_acc[(((size_t)(b * S + qi)) * H + h) * kHeadDim + (i % kHeadDim)] = dQs[i];
  }
}

// =============================================================================
// 3) convert：fp32 累加缓冲 → bf16 输出
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               bf16* __restrict__ dq,
                               bf16* __restrict__ dk,
                               bf16* __restrict__ dv,
                               size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    dq[i] = __float2bfloat16(dq_acc[i]);
    dk[i] = __float2bfloat16(dk_acc[i]);
    dv[i] = __float2bfloat16(dv_acc[i]);
  }
}

#endif  // FA_BWD_BF16_KERNELS_CUH_
