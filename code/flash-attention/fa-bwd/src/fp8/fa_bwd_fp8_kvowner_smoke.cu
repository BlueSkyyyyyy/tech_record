// =============================================================================
// fa_bwd_fp8_kvowner_smoke.cu —— F7 第一步：dK/dV 的「KV-owner 单一 owner」划分
// =============================================================================
// 背景（ROADMAP「fp8 专项冲刺」F7 / 阻塞 第一百四十七轮）：
//   默认 fp8 main（`..._mma_kvtma_...`，S4096 causal）ncu 的 L2 `red` 扇区 **114.5M**
//   （占 L2 流量 74%），而 TE 的 `..._flash_bprop_wgmma_f8_..._64x64x128` 在**同 BM=64**
//   下 `red` 仅 25.96M（**1/4.4×**）、L2 总量 1/4.2×、时间 1/6×。O67 已证「归约加宽」无效
//   （`F32x4` 与 `F32x2` 的 red 计数一字不变），⇒ 真差距是「每个 KV 元素被多少个 CTA 贡献」
//   = **工作划分 / tile 调度**。默认 ours 是「Q-owner」：每 CTA 拥有一个 query 行块、遍历
//   KV 块，dK/dV 用跨 CTA `atomicAdd`（每个 KV 元素被 ~mblk 个 CTA 各加一次）。
//
// 本文件验证 F7 的核心机制（**不换 dtype、不改数学，只换工作划分**）：
//   **KV-owner**：每 CTA 拥有一个 **KV 行块** `[j0, j0+BN)`，遍历所有 query 块，
//   把 dK/dV 在 smem 里**本地累加**，循环结束后**一次 plain store** 写回 global。
//   ⇒ 每个 KV 元素只被「唯一 owner」写一次，**全程零 global atomic / 零 red**。
//
// 对照两个 kernel（同一批 Q/K/V/dO/LSE/delta）：
//   (1) `mowner_kernel`（现有 ours 划分）：grid=(S/BM,H)，每 CTA 一块 query，遍历 KV，
//       dK/dV 逐 (m,kv) tile 做 `atomicAdd` —— red 扇区大。
//   (2) `kvowner_kernel`（F7）：grid=(S/BN,H)，每 CTA 一块 KV，遍历 query，dK/dV 本地
//       累加后一次写 —— **red 扇区 = 0**。
//   两者数值应一致（fp32 舍入内，只差加和次序）。
//
// 说明：本 smoke 用 **fp32 标量**（隔离「划分」这一变量；真实 kernel 用 fp8 mma，但
//   划分机制与 dtype 无关）。目的 = 证明划分正确 + ncu 证实 red=0，作为 F7 主体
//   （把同一划分落进 mma/TMA 主 kernel）的前置 de-risk。
//
// -----------------------------------------------------------------------------
// F7 第二步（第一百四十九届，本文件新增）：**persistent KV-owner + smem/cp.async 暂存**
// -----------------------------------------------------------------------------
// 第一步的判据：KV-owner 把跨 CTA `red` 打成 0，但 `lts__t_sectors_op_read` 从 Q-owner 的
// 1.51M 暴涨到 **44.03M（29×）**——因为「每 CTA 一块 KV、遍历所有 query」时，Q/dO 的 operand
// 没有 staging，每个 query 元素被反复从 L2 重读；单元素 reduc 收益被读放大吃回（标量 0.64×）。
//
// 本步新增 `kvowner_stage_kernel`（真实 F7 主体思路：**persistent CTA 拥有 KV 块，K/V 常驻
// smem 只读一次；Q/dO 用 `cp.async` 双缓冲流水暂存**）：
//   - 每 CTA（grid=(S/BN,H)）**只从 global 读一次自己拥有的 K/V 行块** → 常驻 smem；
//   - 遍历 query 块时，Q/dO 由 `cp.async.cg` 16B 双缓冲搬进 smem，phase1/phase2 全从 smem 读；
//   - dK/dV 在 smem 本地累加、循环外一次 plain store（与第一步一致，red=0）。
// 预期：`red` 仍为 0，且 `op_read` 回落到与 Q-owner 同量级（读放大的机制被 staging 消除）。
//
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_kvowner_smoke.cu
// 剖析：scripts/ncu.sh src/fp8/fa_bwd_fp8_kvowner_smoke.cu --metrics \
//         lts__t_sectors_op_red.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum \
//         --kernel-name regex:kvowner
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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

static constexpr int BM = 32;              // query 行块
static constexpr int BN = 32;              // KV 行块
static constexpr int HD = 128;             // head_dim
static constexpr int THREADS = 128;

// ---- cp.async 16B（Q/dO staging） ------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Q/K/V/dO/LSE/delta 在 [S,H,HD]（B=1）上的行主序；LSE/delta 在 [S,H]。

// -----------------------------------------------------------------------------
// (1) Q-owner（现有 ours 划分）：每 CTA 一块 query 行，遍历 KV，dK/dV 逐 tile atomicAdd
// -----------------------------------------------------------------------------
template <int BM_, int BN_, int HD_>
__global__ void mowner_kernel(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              const float* __restrict__ do_,
                              const float* __restrict__ lse,
                              const float* __restrict__ delta, float* __restrict__ dk,
                              float* __restrict__ dv, int S, int H, float scale) {
  const int m0 = blockIdx.x * BM_;
  const int h = blockIdx.y;
  const int S_ = S;
  __shared__ float Ps[BM_][BN_];
  __shared__ float dSs[BM_][BN_];
  const int tid = threadIdx.x;

  auto qat = [&](int i, int d) { return q[((size_t)(i * H + h)) * HD_ + d]; };
  auto kat = [&](int j, int d) { return k[((size_t)(j * H + h)) * HD_ + d]; };
  auto vat = [&](int j, int d) { return v[((size_t)(j * H + h)) * HD_ + d]; };
  auto oat = [&](int i, int d) { return do_[((size_t)(i * H + h)) * HD_ + d]; };

  // causal: 只有 kv j <= query i。m0 块内的 query 最大 index = m0+BM-1。
  const int jmax = min(m0 + BM_ - 1, S_ - 1);
  for (int j0 = 0; j0 <= jmax; j0 += BN_) {
    __syncthreads();
    for (int idx = tid; idx < BM_ * BN_; idx += THREADS) {
      const int ii = idx / BN_, jj = idx % BN_;
      const int i = m0 + ii, j = j0 + jj;
      float p = 0.f, ds = 0.f;
      if (i < S_ && j < S_ && j <= i) {
        float s = 0.f, dp = 0.f;
#pragma unroll 4
        for (int d = 0; d < HD_; ++d) {
          s += qat(i, d) * kat(j, d);
          dp += oat(i, d) * vat(j, d);
        }
        p = __expf(s * scale - lse[i * H + h]);
        ds = p * (dp - delta[i * H + h]);
      }
      Ps[ii][jj] = p;
      dSs[ii][jj] = ds;
    }
    __syncthreads();
    // dK[j][d] += sum_i dS[i][j] * Q[i][d]；dV[j][d] += sum_i P[i][j] * dO[i][d]
    for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
      const int jj = idx / HD_, d = idx % HD_;
      const int j = j0 + jj;
      if (j >= S_) continue;
      float ak = 0.f, av = 0.f;
#pragma unroll 4
      for (int ii = 0; ii < BM_; ++ii) {
        const int i = m0 + ii;
        if (i >= S_) break;
        ak += dSs[ii][jj] * qat(i, d);
        av += Ps[ii][jj] * oat(i, d);
      }
      atomicAdd(&dk[((size_t)(j * H + h)) * HD_ + d], ak);
      atomicAdd(&dv[((size_t)(j * H + h)) * HD_ + d], av);
    }
  }
  __syncthreads();
}

// -----------------------------------------------------------------------------
// (2) KV-owner（F7）：每 CTA 一块 KV 行，遍历 query，dK/dV 本地累加后一次 plain store
// -----------------------------------------------------------------------------
template <int BM_, int BN_, int HD_>
__global__ void kvowner_kernel(const float* __restrict__ q,
                               const float* __restrict__ k,
                               const float* __restrict__ v,
                               const float* __restrict__ do_,
                               const float* __restrict__ lse,
                               const float* __restrict__ delta, float* __restrict__ dk,
                               float* __restrict__ dv, int S, int H, float scale) {
  const int j0 = blockIdx.x * BN_;
  const int h = blockIdx.y;
  const int S_ = S;
  __shared__ float Ps[BM_][BN_];
  __shared__ float dSs[BM_][BN_];
  __shared__ float dKa[BN_][HD_];
  __shared__ float dVa[BN_][HD_];
  const int tid = threadIdx.x;

  auto qat = [&](int i, int d) { return q[((size_t)(i * H + h)) * HD_ + d]; };
  auto kat = [&](int j, int d) { return k[((size_t)(j * H + h)) * HD_ + d]; };
  auto vat = [&](int j, int d) { return v[((size_t)(j * H + h)) * HD_ + d]; };
  auto oat = [&](int i, int d) { return do_[((size_t)(i * H + h)) * HD_ + d]; };

  for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
    reinterpret_cast<float*>(dKa)[idx] = 0.f;
    reinterpret_cast<float*>(dVa)[idx] = 0.f;
  }
  __syncthreads();

  // KV 行块 [j0,j0+BN) 只被 query i >= j0 消费 ⇒ 从含 j0 的 query 块开始（causal 裁剪），
  // 与 Q-owner 的「kv j0<=jmax」对称，使工作量可比。
  const int mstart = (j0 / BM_) * BM_;
  for (int m0 = mstart; m0 < S_; m0 += BM_) {
    for (int idx = tid; idx < BM_ * BN_; idx += THREADS) {
      const int ii = idx / BN_, jj = idx % BN_;
      const int i = m0 + ii, j = j0 + jj;
      float p = 0.f, ds = 0.f;
      if (i < S_ && j < S_ && j <= i) {
        float s = 0.f, dp = 0.f;
#pragma unroll 4
        for (int d = 0; d < HD_; ++d) {
          s += qat(i, d) * kat(j, d);
          dp += oat(i, d) * vat(j, d);
        }
        p = __expf(s * scale - lse[i * H + h]);
        ds = p * (dp - delta[i * H + h]);
      }
      Ps[ii][jj] = p;
      dSs[ii][jj] = ds;
    }
    __syncthreads();
    for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
      const int jj = idx / HD_, d = idx % HD_;
      float ak = 0.f, av = 0.f;
#pragma unroll 4
      for (int ii = 0; ii < BM_; ++ii) {
        const int i = m0 + ii;
        if (i >= S_) break;
        ak += dSs[ii][jj] * qat(i, d);
        av += Ps[ii][jj] * oat(i, d);
      }
      dKa[jj][d] += ak;
      dVa[jj][d] += av;
    }
    __syncthreads();
  }
  // 单一 owner：一次 plain store（无 atomic）
  for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
    const int jj = idx / HD_, d = idx % HD_;
    const int j = j0 + jj;
    if (j >= S_) continue;
    dk[((size_t)(j * H + h)) * HD_ + d] = dKa[jj][d];
    dv[((size_t)(j * H + h)) * HD_ + d] = dVa[jj][d];
  }
}

// -----------------------------------------------------------------------------
// (3) KV-owner persistent + smem/cp.async staging（F7 第二步）：K/V 常驻 smem 只读一次，
//     Q/dO 由 cp.async 双缓冲流水搬入，dK/dV 本地累加后一次 plain store（red=0 且无读放大）。
// -----------------------------------------------------------------------------
template <int BM_, int BN_, int HD_>
__global__ void kvowner_stage_kernel(const float* __restrict__ q,
                                     const float* __restrict__ k,
                                     const float* __restrict__ v,
                                     const float* __restrict__ do_,
                                     const float* __restrict__ lse,
                                     const float* __restrict__ delta,
                                     float* __restrict__ dk, float* __restrict__ dv, int S,
                                     int H, float scale) {
  const int j0 = blockIdx.x * BN_;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int S_ = S;

  // 动态 smem（smem 总量 ~136KB > 48KB 静态上限）：单一 extern 缓冲手工切分（按 float 偏移）。
  extern __shared__ float smem[];
  const int off_Vs = BN_ * HD_;
  const int off_dKa = 2 * off_Vs;
  const int off_dVa = 3 * off_Vs;
  const int off_Ps = 4 * off_Vs;
  const int off_dSs = off_Ps + BM_ * BN_;
  const int off_Qs = off_dSs + BM_ * BN_;
  const int off_dOs = off_Qs + 2 * BM_ * HD_;
  float(*Ks)[HD_] = (float(*)[HD_])smem;                     // 拥有的 K 行块：只读一次
  float(*Vs)[HD_] = (float(*)[HD_])(smem + off_Vs);          // 拥有的 V 行块：只读一次
  float(*dKa)[HD_] = (float(*)[HD_])(smem + off_dKa);        // dK 本地累加器（无 atomic）
  float(*dVa)[HD_] = (float(*)[HD_])(smem + off_dVa);        // dV 本地累加器
  float(*Ps)[BN_] = (float(*)[BN_])(smem + off_Ps);
  float(*dSs)[BN_] = (float(*)[BN_])(smem + off_dSs);
  float(*Qs)[BM_][HD_] = (float(*)[BM_][HD_])(smem + off_Qs);    // Q staging（双缓冲）
  float(*dOs)[BM_][HD_] = (float(*)[BM_][HD_])(smem + off_dOs);  // dO staging（双缓冲）

  auto rowp = [&](const float* p, int i) {
    return p + ((size_t)(i * H + h)) * HD_;
  };
  constexpr int U = HD_ / 4;  // 每行 float4 单元数

  for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
    reinterpret_cast<float*>(dKa)[idx] = 0.f;
    reinterpret_cast<float*>(dVa)[idx] = 0.f;
  }
  // 拥有的 K/V 行块：一次 vectorized 载入 smem（j 越界补 0）
  for (int idx = tid; idx < BN_ * U; idx += THREADS) {
    const int r = idx / U, c4 = idx % U;
    const int j = j0 + r;
    const bool ok = (j < S_);
    const float4 kk = ok ? ((const float4*)rowp(k, j))[c4] : make_float4(0, 0, 0, 0);
    const float4 vv = ok ? ((const float4*)rowp(v, j))[c4] : make_float4(0, 0, 0, 0);
    ((float4*)Ks[r])[c4] = kk;
    ((float4*)Vs[r])[c4] = vv;
  }
  __syncthreads();

  // KV 行块 [j0,j0+BN) 只被 query i >= j0 消费 ⇒ 从含 j0 的 query 块开始（causal 裁剪）
  const int mstart = (j0 / BM_) * BM_;

  // Q/dO staging 的双缓冲：issue(stage, m0) 发一整块 BM×HD，越界补 0
  auto issue = [&](int stage, int m0) {
    for (int idx = tid; idx < BM_ * U; idx += THREADS) {
      const int r = idx / U, c4 = idx % U;
      const int i = m0 + r;
      if (i < S_) {
        cp_async16(&((float4*)Qs[stage][r])[c4], &((const float4*)rowp(q, i))[c4]);
        cp_async16(&((float4*)dOs[stage][r])[c4], &((const float4*)rowp(do_, i))[c4]);
      } else {
        ((float4*)Qs[stage][r])[c4] = make_float4(0, 0, 0, 0);
        ((float4*)dOs[stage][r])[c4] = make_float4(0, 0, 0, 0);
      }
    }
    cp_async_commit();
  };

  issue(0, mstart);
  int st = 0;
  for (int m0 = mstart; m0 < S_; m0 += BM_, st ^= 1) {
    const bool has_next = (m0 + BM_ < S_);
    if (has_next) issue(st ^ 1, m0 + BM_);
    if (has_next)
      cp_async_wait<1>();  // 只等当前 stage（下一 stage 仍在飞）
    else
      cp_async_wait<0>();
    __syncthreads();

    // phase1：S=QKᵀ、P=softmax、dP=dO·Vᵀ、dS=P∘(dP−D)，全从 smem 读
    for (int idx = tid; idx < BM_ * BN_; idx += THREADS) {
      const int ii = idx / BN_, jj = idx % BN_;
      const int i = m0 + ii, j = j0 + jj;
      float p = 0.f, ds = 0.f;
      if (i < S_ && j < S_ && j <= i) {
        float s = 0.f, dp = 0.f;
#pragma unroll 4
        for (int d = 0; d < HD_; ++d) {
          s += Qs[st][ii][d] * Ks[jj][d];
          dp += dOs[st][ii][d] * Vs[jj][d];
        }
        p = __expf(s * scale - lse[i * H + h]);
        ds = p * (dp - delta[i * H + h]);
      }
      Ps[ii][jj] = p;
      dSs[ii][jj] = ds;
    }
    __syncthreads();
    // phase2：dK += dSᵀQ、dV += PᵀdO，本地累加（无 atomic）
    for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
      const int jj = idx / HD_, d = idx % HD_;
      float ak = 0.f, av = 0.f;
#pragma unroll 4
      for (int ii = 0; ii < BM_; ++ii) {
        const int i = m0 + ii;
        if (i >= S_) break;
        ak += dSs[ii][jj] * Qs[st][ii][d];
        av += Ps[ii][jj] * dOs[st][ii][d];
      }
      dKa[jj][d] += ak;
      dVa[jj][d] += av;
    }
    __syncthreads();
  }
  // 单一 owner：一次 plain store（无 atomic）
  for (int idx = tid; idx < BN_ * HD_; idx += THREADS) {
    const int jj = idx / HD_, d = idx % HD_;
    const int j = j0 + jj;
    if (j >= S_) continue;
    dk[((size_t)(j * H + h)) * HD_ + d] = dKa[jj][d];
    dv[((size_t)(j * H + h)) * HD_ + d] = dVa[jj][d];
  }
}

// -----------------------------------------------------------------------------
// host 参考（double）：LSE / delta / dK / dV
// -----------------------------------------------------------------------------
static void ref_dk_dv(const std::vector<float>& q, const std::vector<float>& k,
                      const std::vector<float>& v, const std::vector<float>& do_,
                      int S, int H, float scale, std::vector<double>& dk,
                      std::vector<double>& dv) {
  dk.assign((size_t)S * H * HD, 0.0);
  dv.assign((size_t)S * H * HD, 0.0);
  std::vector<double> o((size_t)HD), dp((size_t)HD), ds((size_t)HD);
  for (int h = 0; h < H; ++h) {
    // lse
    std::vector<double> lse(S, 0.0), delta(S, 0.0);
    for (int i = 0; i < S; ++i) {
      double mx = -1e30;
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int d = 0; d < HD; ++d)
          s += (double)q[((size_t)(i * H + h)) * HD + d] * k[((size_t)(j * H + h)) * HD + d];
        s *= scale;
        if (s > mx) mx = s;
      }
      double sum = 0.0;
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int d = 0; d < HD; ++d)
          s += (double)q[((size_t)(i * H + h)) * HD + d] * k[((size_t)(j * H + h)) * HD + d];
        sum += exp(s * scale - mx);
      }
      lse[i] = mx + log(sum);
      for (int d = 0; d < HD; ++d) o[d] = 0.0;
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int d = 0; d < HD; ++d)
          s += (double)q[((size_t)(i * H + h)) * HD + d] * k[((size_t)(j * H + h)) * HD + d];
        double p = exp(s * scale - lse[i]);
        for (int d = 0; d < HD; ++d) o[d] += p * v[((size_t)(j * H + h)) * HD + d];
      }
      double de = 0.0;
      for (int d = 0; d < HD; ++d) de += (double)do_[((size_t)(i * H + h)) * HD + d] * o[d];
      delta[i] = de;
    }
    // dK/dV
    for (int i = 0; i < S; ++i) {
      for (int d = 0; d < HD; ++d) dp[d] = 0.0;
      for (int j = 0; j <= i; ++j) {
        for (int d = 0; d < HD; ++d)
          dp[d] += (double)do_[((size_t)(i * H + h)) * HD + d] *
                   v[((size_t)(j * H + h)) * HD + d];
      }
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int d = 0; d < HD; ++d)
          s += (double)q[((size_t)(i * H + h)) * HD + d] * k[((size_t)(j * H + h)) * HD + d];
        double p = exp(s * scale - lse[i]);
        // ds[j][d] 实际是标量（p, dp 都与 j 有关）：逐 j 算
        double dpj = 0.0;
        for (int d = 0; d < HD; ++d)
          dpj += (double)do_[((size_t)(i * H + h)) * HD + d] *
                 v[((size_t)(j * H + h)) * HD + d];
        double dsv = p * (dpj - delta[i]);
        for (int d = 0; d < HD; ++d) {
          dk[((size_t)(j * H + h)) * HD + d] +=
              dsv * (double)q[((size_t)(i * H + h)) * HD + d];
          dv[((size_t)(j * H + h)) * HD + d] +=
              p * (double)do_[((size_t)(i * H + h)) * HD + d];
        }
      }
    }
  }
  (void)dp;
  (void)ds;
}

int main() {
  const int S = 512, H = 8, B = 1;
  const float scale = 1.0f / sqrtf((float)HD);
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> q((size_t)S * H * HD), k((size_t)S * H * HD), v((size_t)S * H * HD),
      do_((size_t)S * H * HD);
  for (auto& x : q) x = nd(rng);
  for (auto& x : k) x = nd(rng);
  for (auto& x : v) x = nd(rng);
  for (auto& x : do_) x = nd(rng);

  // host 参考里顺便得到 LSE/delta（用 double 版另算，简单起见重跑一遍上面的逻辑）
  std::vector<double> rdk, rdv;
  ref_dk_dv(q, k, v, do_, S, H, scale, rdk, rdv);

  // 用 double 版同款公式生成 LSE/delta 供 device 用（fp32）
  std::vector<float> lse((size_t)S * H), delta((size_t)S * H);
  {
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < S; ++i) {
        double mx = -1e30;
        for (int j = 0; j <= i; ++j) {
          double s = 0.0;
          for (int d = 0; d < HD; ++d)
            s += (double)q[((size_t)(i * H + h)) * HD + d] *
                 k[((size_t)(j * H + h)) * HD + d];
          s *= scale;
          if (s > mx) mx = s;
        }
        double sum = 0.0;
        for (int j = 0; j <= i; ++j) {
          double s = 0.0;
          for (int d = 0; d < HD; ++d)
            s += (double)q[((size_t)(i * H + h)) * HD + d] *
                 k[((size_t)(j * H + h)) * HD + d];
          sum += exp(s * scale - mx);
        }
        double l = mx + log(sum);
        double de = 0.0;
        for (int j = 0; j <= i; ++j) {
          double s = 0.0;
          for (int d = 0; d < HD; ++d)
            s += (double)q[((size_t)(i * H + h)) * HD + d] *
                 k[((size_t)(j * H + h)) * HD + d];
          double p = exp(s * scale - l);
          for (int d = 0; d < HD; ++d)
            de += p * (double)v[((size_t)(j * H + h)) * HD + d] *
                  (double)do_[((size_t)(i * H + h)) * HD + d];
        }
        lse[i * H + h] = (float)l;
        delta[i * H + h] = (float)de;
      }
    }
  }

  float *dqq, *dkk, *dvv, *ddo, *dlse, *ddelta;
  const size_t n = (size_t)S * H * HD;
  CUDA_CHECK(cudaMalloc(&dqq, n * 4));
  CUDA_CHECK(cudaMalloc(&dkk, n * 4));
  CUDA_CHECK(cudaMalloc(&dvv, n * 4));
  CUDA_CHECK(cudaMalloc(&ddo, n * 4));
  CUDA_CHECK(cudaMalloc(&dlse, (size_t)S * H * 4));
  CUDA_CHECK(cudaMalloc(&ddelta, (size_t)S * H * 4));
  CUDA_CHECK(cudaMemcpy(dqq, q.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dkk, k.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvv, v.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ddo, do_.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dlse, lse.data(), (size_t)S * H * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ddelta, delta.data(), (size_t)S * H * 4, cudaMemcpyHostToDevice));

  dim3 gm((S + BM - 1) / BM, H), gk((S + BN - 1) / BN, H);
  CUDA_CHECK(cudaFuncSetAttribute(kvowner_stage_kernel<BM, BN, HD>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, 139264));
  // Q-owner：分两次跑做 A/B 计时；red 大
  auto run_mo = [&](float* dk, float* dv) {
    CUDA_CHECK(cudaMemset(dk, 0, n * 4));
    CUDA_CHECK(cudaMemset(dv, 0, n * 4));
    mowner_kernel<BM, BN, HD><<<gm, THREADS>>>(dqq, dkk, dvv, ddo, dlse, ddelta, dk, dv, S, H,
                                               scale);
  };
  auto run_kvo = [&](float* dk, float* dv) {
    kvowner_kernel<BM, BN, HD><<<gk, THREADS>>>(dqq, dkk, dvv, ddo, dlse, ddelta, dk, dv, S, H,
                                                scale);
  };
  auto run_kvs = [&](float* dk, float* dv) {
    kvowner_stage_kernel<BM, BN, HD><<<gk, THREADS, 139264>>>(dqq, dkk, dvv, ddo, dlse, ddelta,
                                                              dk, dv, S, H, scale);
  };

  float *mo_dk, *mo_dv, *kvo_dk, *kvo_dv, *kvs_dk, *kvs_dv;
  CUDA_CHECK(cudaMalloc(&mo_dk, n * 4));
  CUDA_CHECK(cudaMalloc(&mo_dv, n * 4));
  CUDA_CHECK(cudaMalloc(&kvo_dk, n * 4));
  CUDA_CHECK(cudaMalloc(&kvo_dv, n * 4));
  CUDA_CHECK(cudaMalloc(&kvs_dk, n * 4));
  CUDA_CHECK(cudaMalloc(&kvs_dv, n * 4));
  run_mo(mo_dk, mo_dv);
  run_kvo(kvo_dk, kvo_dv);
  run_kvs(kvs_dk, kvs_dv);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_mo_dk(n), h_mo_dv(n), h_kvo_dk(n), h_kvo_dv(n), h_kvs_dk(n),
      h_kvs_dv(n);
  CUDA_CHECK(cudaMemcpy(h_mo_dk.data(), mo_dk, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_mo_dv.data(), mo_dv, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_kvo_dk.data(), kvo_dk, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_kvo_dv.data(), kvo_dv, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_kvs_dk.data(), kvs_dk, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_kvs_dv.data(), kvs_dv, n * 4, cudaMemcpyDeviceToHost));

  auto maxabs = [&](const std::vector<float>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i]));
    return m;
  };
  auto rel = [&](const std::vector<float>& a, const std::vector<double>& b) {
    double m = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      m = std::max(m, std::fabs((double)a[i] - b[i]));
      den = std::max(den, std::fabs(b[i]));
    }
    return den > 0 ? m / den : m;
  };
  auto vs_mo = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i]));
    return m;
  };

  printf("=== F7 KV-owner smoke (S=%d H=%d HD=%d causal, fp32 scalar) ===\n", S, H, HD);
  printf("[vs host double ref]  Q-owner  dk max_abs=%.3e rel=%.3e | dv max_abs=%.3e rel=%.3e\n",
         maxabs(h_mo_dk, rdk), rel(h_mo_dk, rdk), maxabs(h_mo_dv, rdv), rel(h_mo_dv, rdv));
  printf("[vs host double ref]  KV-owner dk max_abs=%.3e rel=%.3e | dv max_abs=%.3e rel=%.3e\n",
         maxabs(h_kvo_dk, rdk), rel(h_kvo_dk, rdk), maxabs(h_kvo_dv, rdv), rel(h_kvo_dv, rdv));
  printf("[vs host double ref]  KV-owner+stage dk max_abs=%.3e rel=%.3e | dv max_abs=%.3e rel=%.3e\n",
         maxabs(h_kvs_dk, rdk), rel(h_kvs_dk, rdk), maxabs(h_kvs_dv, rdv), rel(h_kvs_dv, rdv));
  printf("[KV-owner vs Q-owner] dk max_abs=%.3e  dv max_abs=%.3e\n",
         vs_mo(h_kvo_dk, h_mo_dk), vs_mo(h_kvo_dv, h_mo_dv));
  printf("[KV-owner+stage vs Q-owner] dk max_abs=%.3e  dv max_abs=%.3e\n",
         vs_mo(h_kvs_dk, h_mo_dk), vs_mo(h_kvs_dv, h_mo_dv));

  // 计时
  auto bench = [&](auto fn, int iters) {
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b);
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    return ms / iters;
  };
  float t_mo = bench([&] { run_mo(mo_dk, mo_dv); }, 20);
  float t_kvo = bench([&] { run_kvo(kvo_dk, kvo_dv); }, 20);
  float t_kvs = bench([&] { run_kvs(kvs_dk, kvs_dv); }, 20);
  printf("[time] Q-owner atomic = %.4f ms | KV-owner single-store = %.4f ms (save %.2fx)\n",
         t_mo, t_kvo, t_mo / t_kvo);
  printf("[time] KV-owner+stage = %.4f ms | vs Q-owner %.2fx | vs KV-owner %.2fx\n", t_kvs,
         t_mo / t_kvs, t_kvo / t_kvs);

  CUDA_CHECK(cudaFree(dqq));
  CUDA_CHECK(cudaFree(dkk));
  CUDA_CHECK(cudaFree(dvv));
  CUDA_CHECK(cudaFree(ddo));
  CUDA_CHECK(cudaFree(dlse));
  CUDA_CHECK(cudaFree(ddelta));
  CUDA_CHECK(cudaFree(mo_dk));
  CUDA_CHECK(cudaFree(mo_dv));
  CUDA_CHECK(cudaFree(kvo_dk));
  CUDA_CHECK(cudaFree(kvo_dv));
  CUDA_CHECK(cudaFree(kvs_dk));
  CUDA_CHECK(cudaFree(kvs_dv));
  return 0;
}
