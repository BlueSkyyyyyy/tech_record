// =============================================================================
// fa_bwd_fp16_kernels.cuh —— FlashAttention 反向（fp16）**两文件版的 device 部分**
// =============================================================================
// 由单文件版 `fa_bwd_fp16_onefile.cu` 拆分而来（P1-4），device 代码与单文件**逐字一致**：
//   * preprocess_kernel：逐行算 LSE 与 delta=rowsum(dO∘O)
//   * fa_bwd_fp16_kernel：1colblock 反向主 kernel（recompute P，fp32 累加）
//   * convert_kernel：fp32 累加缓冲 → fp16 输出
// host 侧（npy 读取 / launcher / 自测）见 `fa_bwd_fp16_main.cu`。
//
// 数学（见 docs/00 与 ref_impl.py）：
//   S  = scale·QKᵀ ; P = softmax(S) ; D = rowsum(dO∘O)
//   dP = dO Vᵀ ; dS = P∘(dP−D) ; dV = Pᵀ dO ; dQ = scale·dS K ; dK = scale·dSᵀ Q
// =============================================================================

#ifndef FA_BWD_FP16_KERNELS_CUH_
#define FA_BWD_FP16_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int BN       = 32;    // K/V 块行数（= warpSize，lane 即列号）
static constexpr int THREADS  = 128;   // 4 warps

// P5-2：head_dim 作为模板参数 HD（运行时可选 128 / 512）。
//   * HD=128 → BM=64：与 P1/P5-1 的 MHA 路径逐位一致（回归不变）。
//   * HD=512 → BM=16：MLA 主注意力（smem 装不下 BM=64 的 dQs[BM*HD] fp32）。
// 动态 smem 布局（字节）：
//   Qs[BM*HD] + Ks[BN*HD] + Vs[BN*HD] + dOs[BM*HD]   （half）
//   Ss[BM*BN] + Ps[BM*BN]                             （float）
//   dQs[BM*HD]                                        （float）
template <int HD, int BM>
struct BwdTraits {
  static constexpr int WM_ROWS = BM / 4;   // 每个 warp 负责的 Q 行数
  static constexpr int WN_ROWS = BN / 4;   // 每个 warp 负责的 K/V 行数
  static constexpr int NCH     = HD / 32;  // lane 覆盖 head_dim 的分段数
  static constexpr int smem_bytes =
      (BM * HD + BN * HD + BN * HD + BM * HD) * (int)sizeof(__half) +
      (BM * BN + BM * BN) * (int)sizeof(float) +
      (BM * HD) * (int)sizeof(float);
};

// =============================================================================
// 1) preprocess：逐行算 LSE 与 delta=rowsum(dO∘O)
//   grid = (S, H, B)，block = THREADS
//   GQA/MQA（P5-1）：K 的第 h 个 Q 头映射到 KV 头 h/(H/Hkv)（repeat_interleave）。
// =============================================================================
__global__ void preprocess_kernel(const __half* __restrict__ q,
                                  const __half* __restrict__ k,
                                  const __half* __restrict__ o,
                                  const __half* __restrict__ do_,
                                  float* __restrict__ delta,   // [B*S*H]
                                  float* __restrict__ lse,     // [B*S*H]
                                  int S, int H, int Hkv, float scale, int causal, int HD) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int hkv = h / (H / Hkv);   // Q 头 -> KV 头
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const __half* qr = q + row * HD;

  // --- online softmax：单趟求 (m, l)，无需物化整行 S ---
  float m = -INFINITY;
  float l = 0.f;
  const int jmax = causal ? (s + 1) : S;   // causal：只算 j <= s
  for (int j = tid; j < jmax; j += blockDim.x) {
    const __half* kr = k + (((size_t)(b * S + j)) * Hkv + hkv) * HD;
    float dot = 0.f;
#pragma unroll 8
    for (int d = 0; d < HD; ++d)
      dot += __half2float(qr[d]) * __half2float(kr[d]);
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
  const __half* orow = o + row * HD;
  const __half* dorow = do_ + row * HD;
  for (int d = tid; d < HD; d += blockDim.x)
    dp += __half2float(orow[d]) * __half2float(dorow[d]);
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
template <int HD, int BM>
__global__ void __launch_bounds__(THREADS)
fa_bwd_fp16_kernel(const __half* __restrict__ q,
                   const __half* __restrict__ k,
                   const __half* __restrict__ v,
                   const __half* __restrict__ do_,
                   const float* __restrict__ delta,
                   const float* __restrict__ lse,
                   float* __restrict__ dq_acc,
                   float* __restrict__ dk_acc,
                   float* __restrict__ dv_acc,
                   int S, int H, int Hkv, float scale, int causal) {
  using T = BwdTraits<HD, BM>;
  constexpr int WM_ROWS = T::WM_ROWS;
  constexpr int WN_ROWS = T::WN_ROWS;
  constexpr int NCH     = T::NCH;
  extern __shared__ char smem[];
  __half* Qs  = reinterpret_cast<__half*>(smem);
  __half* Ks  = Qs + BM * HD;
  __half* Vs  = Ks + BN * HD;
  __half* dOs = Vs + BN * HD;
  float* Ss = reinterpret_cast<float*>(dOs + BM * HD);  // 先存 S，后覆盖为 dS
  float* Ps = Ss + BM * BN;
  float* dQs = Ps + BM * BN;                            // dQ 的 smem 累加器

  const int mblk = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int m0 = mblk * BM;
  const int hkv = h / (H / Hkv);   // Q 头 -> KV 头（GQA/MQA）

  // ---- 载入本 Q 块的 Q 与 dO（越界补 0）----
  for (int i = tid; i < BM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    __half qv = __float2half(0.f), ov = __float2half(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * HD + d;
      qv = q[idx];
      ov = do_[idx];
    }
    Qs[i] = qv;
    dOs[i] = ov;
  }
  for (int i = tid; i < BM * HD; i += THREADS) dQs[i] = 0.f;
  __syncthreads();

  // causal：只需处理到本 Q 块最后一行的列；否则到 S。
  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;

    // ---- 载入 K/V 块 ----
    for (int i = tid; i < BN * HD; i += THREADS) {
      int j = i / HD, d = i % HD;
      int jg = j0 + j;
      __half kv = __float2half(0.f), vv = __float2half(0.f);
      if (jg < S) {
        size_t idx = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d;
        kv = k[idx];
        vv = v[idx];
      }
      Ks[i] = kv;
      Vs[i] = vv;
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
        const __half* qrow = Qs + r * HD;
        const __half* krow = Ks + lane * HD;
#pragma unroll 8
        for (int d = 0; d < HD; ++d)
          dot += __half2float(qrow[d]) * __half2float(krow[d]);
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
        const __half* drow = dOs + r * HD;
        const __half* vrow = Vs + lane * HD;
#pragma unroll 8
        for (int d = 0; d < HD; ++d)
          dp += __half2float(drow[d]) * __half2float(vrow[d]);
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
      float acc[NCH] = {};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float p = Ps[i * BN + j];          // 同一列，warp 内广播
        if (p == 0.f) continue;
        const __half* drow = dOs + i * HD;
#pragma unroll
        for (int kk = 0; kk < NCH; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += p * __half2float(drow[d]);
        }
      }
      float* base = dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD;
#pragma unroll
      for (int kk = 0; kk < NCH; ++kk) atomicAdd(base + lane + 32 * kk, acc[kk]);
    }

    // ---- dK = scale·dSᵀ Q（dS 在 Ss）----
#pragma unroll
    for (int jj = 0; jj < WN_ROWS; ++jj) {
      int j = warp * WN_ROWS + jj;
      int jg = j0 + j;
      if (jg >= S) continue;
      float acc[NCH] = {};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float ds = Ss[i * BN + j];
        if (ds == 0.f) continue;
        const __half* qrow = Qs + i * HD;
#pragma unroll
        for (int kk = 0; kk < NCH; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * __half2float(qrow[d]);
        }
      }
      float* base = dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD;
#pragma unroll
      for (int kk = 0; kk < NCH; ++kk)
        atomicAdd(base + lane + 32 * kk, acc[kk] * scale);
    }

    // ---- dQ += scale·dS K：累积到 smem dQs（本 CTA 独占 Q 行，无需 atomic）----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      if (qi >= S) continue;
      float acc[NCH] = {};
#pragma unroll 4
      for (int j = 0; j < BN; ++j) {
        float ds = Ss[r * BN + j];
        if (ds == 0.f) continue;
        const __half* krow = Ks + j * HD;
#pragma unroll
        for (int kk = 0; kk < NCH; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * __half2float(krow[d]);
        }
      }
      float* dqr = dQs + r * HD;
#pragma unroll
      for (int kk = 0; kk < NCH; ++kk)
        dqr[lane + 32 * kk] += acc[kk] * scale;
    }
    __syncthreads();
  }

  // ---- 写回 dQ（fp32 缓冲，稍后 convert）----
  for (int i = tid; i < BM * HD; i += THREADS) {
    int r = i / HD;
    int qi = m0 + r;
    if (qi < S)
      dq_acc[(((size_t)(b * S + qi)) * H + h) * HD + (i % HD)] = dQs[i];
  }
}

// =============================================================================
// 3) convert：fp32 累加缓冲 → fp16 输出
//   dq 为 [B*S*H*D]，dk/dv 为 [B*S*Hkv*D]（GQA/MQA 时两者不等长）。
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               __half* __restrict__ dq,
                               __half* __restrict__ dk,
                               __half* __restrict__ dv,
                               size_t n_q, size_t n_kv) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_q;
       i += (size_t)gridDim.x * blockDim.x)
    dq[i] = __float2half(dq_acc[i]);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_kv;
       i += (size_t)gridDim.x * blockDim.x) {
    dk[i] = __float2half(dk_acc[i]);
    dv[i] = __float2half(dv_acc[i]);
  }
}

#endif  // FA_BWD_FP16_KERNELS_CUH_
