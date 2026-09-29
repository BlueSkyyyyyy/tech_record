// =============================================================================
// fa_bwd_fp8_kvowner_mma.cu —— F7 第三步：把 KV-owner 划分落进**真实 fp8 张量核**
//                              （dK/dV 的 mma.m16n8k32 原型）
// =============================================================================
// 背景（ROADMAP「fp8 专项冲刺」F7 / docs/03 §80–§82）：
//   默认 fp8 main 是 **Q-owner**：每个 CTA 拥有一块 query，遍历 KV，dK/dV 逐 tile 对
//   global 做跨 CTA `atomicAdd`（red）。ncu 钉死：S=4096 causal 的 L2 `red` 扇区
//   114.5M（占 L2 流量 74%），而 TE 的 `..._flash_bprop_wgmma_f8_..._64x64x128` 在**同
//   BM=64** 下 `red` 只有 25.96M（1/4.4×）、时间 1/6×。O67 证明「归约加宽」无效，
//   真差距 = **工作划分**。
//
//   §81（F7 第一步）在 fp32 标量 smoke 上验证：**KV-owner**（每 CTA 拥有一块 KV、
//   遍历所有 query、dK/dV 本地累加后一次 plain store）把跨 CTA `red` 打成 **0**，
//   但没做 operand staging ⇒ 读放大 29×、反而慢 0.64×。
//   §82（F7 第二步）在标量 smoke 上加 persistent + smem/cp.async staging ⇒ 读放大消除
//   （op_read 44.07M→1.38M）、red 仍 0、标量 3.95×。
//
//   本文件是 F7 主体的**第一步**：把同一划分落进**真实 fp8 数据通路**（E4M3/E5M2 +
//   rowwise scale + `mma.m16n8k32`），先只做 **dK/dV**（不碰 dQ，dQ 仍可由既有 Q-owner
//   路径承担）。目的 = 在真实的 mma/量化记账下证明「KV-owner + staging + 本地累加 +
//   一次 plain store」的 dK/dV 与 ref/TE 同量级、`red`=0，并给出绝对性能。
//
//   设计（HD=128, BM=64, BN=32, 128 线程, MHA, causal, B=1）：
//     * grid = (ceil(S/BN), H)：每 CTA 拥有 KV 行块 [j0, j0+BN)。**K/V 常驻 smem**，只从
//       global 读一次（Kp 不需要，因为 GEMM3 用 dOp、GEMM5 用 Qp）。
//     * 沿 query 块 m0 从 floor(j0/BM)*BM 遍历到 S（causal 裁剪）。Q/dO **staging** 到
//       smem（Qs/dOs + 配对布局 Qp/dOp）。
//     * 5 个 GEMM 里做 4 个（无 GEMM4 dQ）：
//         GEMM1 S  = scale·QKᵀ        (E4M3×E4M3)
//         GEMM2 dP = dO·Vᵀ           (E5M2×E4M3)
//         fold  P→Ap=P·dos[m] (E4M3, per-j)；dS→dS3=dS·qs[m] (E5M2, per-j)
//         GEMM3 dV += Ap·dOᵀ          (E4M3×E5M2)   → 寄存器本地累加，乘 sA[j]
//         GEMM5 dK += scale·dS3·Qᵀ    (E5M2×E4M3)   → 寄存器本地累加，乘 sds3[j]·scale
//     * 循环结束后 **一次 plain store** 写出 dK/dV ⇒ 每元素仅被 owner 写一次，**red=0**。
//
//   数值口径与 `fa_bwd_fp8_kernels.cuh` 的 Q-owner 主 kernel 完全一致（同一量化、
//   同一 rowwise scale 折算、同一 mma 布局），故 dk/dv 应与既有 `ours_dk/dv` 同量级、
//   与 fp32 ref 差在 fp8 噪声内（只差跨 CTA 加法次序）。
//
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_kvowner_mma.cu [--dir=<dump case>]
// =============================================================================

#include "fa_bwd_fp8_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
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

// =============================================================================
// KV-owner fp8 mma 主 kernel：只算 dK/dV（dK/dV-over-KV 单一 owner）
// =============================================================================
template <int HD, int BM, int BN>
__global__ void __launch_bounds__(THREADS, 3)
fp8_kvowner_dkv_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                       const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                       const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                       const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                       const float* __restrict__ lse, const float* __restrict__ delta,
                       float* __restrict__ dk, float* __restrict__ dv, int S, int H, float scale,
                       int causal) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int PSLD = Cfg::PSLD;
  constexpr int QTS = Cfg::QTS;
  constexpr int PSS = Cfg::PSS;
  constexpr int NWM = 2, NWAR = 2;            // 2×2 warp 网格（与 Q-owner 主 kernel 相同）
  constexpr int NTW = NWAR * 64;              // GEMM3/5 输出 N 维一次铺满的列数
  constexpr int GM1 = BM / NWM, GN1 = BN / NWAR;
  constexpr int MTM = GM1 / 16, MTN = GN1 / 8;
  constexpr int GM34 = BN / NWM, GN34 = NTW / NWAR;
  constexpr int MTM34 = GM34 / 16, NTM34 = GN34 / 8;
  constexpr int NTFOLD = BN / 32;
  static_assert(HD == 128 && BM == 64 && BN == 32, "本原型固定 HD=128/BM=64/BN=32");

  // ---- 动态 smem：K/V 常驻 + 每 query 块的 Q/dO staging + P/dS + 折叠操作数 ----
  constexpr int sz_Ks = BN * ASLD;
  constexpr int sz_Vs = BN * ASLD;
  constexpr int sz_Qs = BM * ASLD;
  constexpr int sz_dOs = BM * ASLD;
  constexpr int sz_Qp = (BM / 2) * PSLD * (int)sizeof(uint16_t);   // 配对布局 uint16
  constexpr int sz_dOp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_Ap = BN * QTS;             // Ap / dS3 各一块（E4M3 / E5M2）
  constexpr int sz_dS3 = BN * QTS;
  constexpr int sz_Ps = BM * PSS * (int)sizeof(float);
  constexpr int sz_Ss = BM * PSS * (int)sizeof(float);
  constexpr int sz_scales = (2 * BM + 4 * BN) * (int)sizeof(float);  // qs,dos / ks,vs,sA,sds3

  extern __shared__ __align__(16) char smem[];
  int off = 0;
  unsigned char* Ks = reinterpret_cast<unsigned char*>(smem + off); off += sz_Ks;
  unsigned char* Vs = reinterpret_cast<unsigned char*>(smem + off); off += sz_Vs;
  unsigned char* Qs = reinterpret_cast<unsigned char*>(smem + off); off += sz_Qs;
  unsigned char* dOs = reinterpret_cast<unsigned char*>(smem + off); off += sz_dOs;
  uint16_t* Qp = reinterpret_cast<uint16_t*>(smem + off); off += sz_Qp;
  uint16_t* dOp = reinterpret_cast<uint16_t*>(smem + off); off += sz_dOp;
  unsigned char* Ap = reinterpret_cast<unsigned char*>(smem + off); off += sz_Ap;
  unsigned char* dS3 = reinterpret_cast<unsigned char*>(smem + off); off += sz_dS3;
  float* Ps = reinterpret_cast<float*>(smem + off); off += sz_Ps;
  float* Ss = reinterpret_cast<float*>(smem + off); off += sz_Ss;
  float* scales = reinterpret_cast<float*>(smem + off);
  float* qs_s = scales;
  float* dos_s = qs_s + BM;
  float* ks_s = dos_s + BM;
  float* vs_s = ks_s + BN;
  float* sA = vs_s + BN;
  float* sds3 = sA + BN;

  const int j0 = blockIdx.x * BN;
  const int h = blockIdx.y;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / NWAR, wc = wid % NWAR;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // ---- 拥有的 K/V 行块：**只从 global 读一次**常驻 smem（越界补 0=E4M3(0)）----
  const int nd4 = HD / 4;
  for (int u = tid; u < (BN / 2) * nd4; u += THREADS) {
    const int rp = u / nd4, dq = (u % nd4) * 4;
    const int jr = j0 + rp * 2;
    uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    if (jr < S) {
      size_t i0 = (((size_t)jr) * H + h) * HD + dq;
      k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
      v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
    }
    if (jr + 1 < S) {
      size_t i1 = (((size_t)(jr + 1)) * H + h) * HD + dq;
      k1 = *reinterpret_cast<const uint32_t*>(k8 + i1);
      v1 = *reinterpret_cast<const uint32_t*>(v8 + i1);
    }
    *reinterpret_cast<uint32_t*>(Ks + (rp * 2) * ASLD + dq) = k0;
    *reinterpret_cast<uint32_t*>(Ks + (rp * 2 + 1) * ASLD + dq) = k1;
    *reinterpret_cast<uint32_t*>(Vs + (rp * 2) * ASLD + dq) = v0;
    *reinterpret_cast<uint32_t*>(Vs + (rp * 2 + 1) * ASLD + dq) = v1;
  }
  if (tid < BN) {
    const int jg = j0 + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)jg) * H + h] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)jg) * H + h] : 1.f;
  }
  __syncthreads();

  // ---- dK/dV 的**寄存器本地累加器**（单一 owner，循环外一次 plain store）----
  float dVacc[MTM34][NTM34][4];
  float dKacc[MTM34][NTM34][4];
#pragma unroll
  for (int i = 0; i < MTM34; ++i)
#pragma unroll
    for (int j = 0; j < NTM34; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) { dVacc[i][j][q] = 0.f; dKacc[i][j][q] = 0.f; }

  const int mstart = (j0 / BM) * BM;  // causal：KV 块 [j0,j0+BN) 只被 query i>=j0 消费
  for (int m0 = mstart; m0 < S; m0 += BM) {
    // ---- staging：Q/dO 行 [m0,m0+BM) 搬进 smem（Qs/dOs）+ 配对布局（Qp/dOp）----
    for (int u = tid; u < (BM / 2) * nd4; u += THREADS) {
      const int rp = u / nd4, dq = (u % nd4) * 4;
      const int qa = m0 + rp * 2, qb = m0 + rp * 2 + 1;
      uint32_t q0 = 0, q1 = 0, o0 = 0, o1 = 0;
      if (qa < S) {
        size_t idx = (((size_t)qa) * H + h) * HD + dq;
        q0 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o0 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      if (qb < S) {
        size_t idx = (((size_t)qb) * H + h) * HD + dq;
        q1 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o1 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      *reinterpret_cast<uint32_t*>(Qs + (rp * 2) * ASLD + dq) = q0;
      *reinterpret_cast<uint32_t*>(Qs + (rp * 2 + 1) * ASLD + dq) = q1;
      *reinterpret_cast<uint32_t*>(dOs + (rp * 2) * ASLD + dq) = o0;
      *reinterpret_cast<uint32_t*>(dOs + (rp * 2 + 1) * ASLD + dq) = o1;
      uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + dq);
      qpw[0] = __byte_perm(q0, q1, 0x5140);
      qpw[1] = __byte_perm(q0, q1, 0x7362);
      uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + dq);
      opw[0] = __byte_perm(o0, o1, 0x5140);
      opw[1] = __byte_perm(o0, o1, 0x7362);
    }
    if (tid < BM) {
      const int qi = m0 + tid;
      qs_s[tid] = (qi < S) ? qs[((size_t)qi) * H + h] : 1.f;
      dos_s[tid] = (qi < S) ? dos[((size_t)qi) * H + h] : 1.f;
    }
    __syncthreads();

    // ---- LSE/D 预装寄存器（每线程 4 个行槽：i∈{0,1} × (q>=2)）----
    float lse_r[4], del_r[4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int r = wr * 32 + i * 16 + g + (s ? 8 : 0);
        const int qi = m0 + r;
        const size_t idx = ((size_t)qi) * H + h;
        const bool ok = qi < S;
        lse_r[i * 2 + s] = ok ? lse[idx] : 0.f;
        del_r[i * 2 + s] = ok ? delta[idx] : 0.f;
      }

    // ---- GEMM1 S = scale·QKᵀ (E4M3×E4M3) + epilogue P=exp(S-LSE) ----
    float preg[MTM][MTN][4];
    {
      float acc[MTM][MTN][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<GM1, GN1, HD, E4E4>(Qs, ASLD, Ks, ASLD, acc, wr, wc, lane);
      const int r0 = wr * GM1, c0 = wc * GN1;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            const int c = c0 + j * 8 + c2 + (q & 1);
            const int qi = m0 + r, jg = j0 + c;
            float p = 0.f;
            if (qi < S && jg < S && !(causal && jg > qi)) {
              const float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
              p = fexp(sval - lse_r[i * 2 + (q >= 2 ? 1 : 0)]);
            }
            Ps[r * PSS + c] = p;
            preg[i][j][q] = p;
          }
    }

    // ---- GEMM2 dP = dO·Vᵀ (E5M2×E4M3) + epilogue dS = P∘(dP-D) ----
    //      O4a：GEMM1/GEMM2 之间无需 barrier（GEMM2 只读 dOs/Vs；其 epilogue 读回的
    //      Ps[r*PSS+c] 正是本线程刚写的同一地址）。----
    {
      float acc[MTM][MTN][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<GM1, GN1, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc, wr, wc, lane);
      const int r0 = wr * GM1, c0 = wc * GN1;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            const int c = c0 + j * 8 + c2 + (q & 1);
            const float dpv = acc[i][j][q] * dos_s[r] * vs_s[c];
            const float del = del_r[i * 2 + (q >= 2 ? 1 : 0)];
            Ss[r * PSS + c] = preg[i][j][q] * (dpv - del);
          }
    }
    __syncthreads();

    // ---- fold：Ap[j][m]=P[m][j]·dos[m] (E4M3, per-j) 与 dS3[j][m]=dS[m][j]·qs[m] (E5M2, per-j)
    //      O4a：全 128 线程均衡分工，warp 内 4-lane shfl 归约 amax（与 Q-owner 主 kernel 同款）。----
    if (wid < 4) {
      const int jl = lane >> 2, sub4 = lane & 3;
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh) {
        const int j = wid * 8 + jl + jh * 32;
        float amaxA = 0.f, amax3 = 0.f;
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            const int m = sub4 * 8 + t + half * 32;
            amaxA = fmaxf(amaxA, fabsf(Ps[m * PSS + j] * dos_s[m]));
            amax3 = fmaxf(amax3, fabsf(Ss[m * PSS + j] * qs_s[m]));
          }
        amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 1));
        amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 2));
        amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 1));
        amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 2));
        float scA = (amaxA > 0.f) ? amaxA / kE4M3Max : 1.f;
        float sc3 = (amax3 > 0.f) ? amax3 / kE5M2Max : 1.f;
        if (sub4 == 0) {
          sA[j] = scA;
          sds3[j] = sc3;
        }
        scA = __shfl_sync(0xffffffffu, scA, jl * 4);
        sc3 = __shfl_sync(0xffffffffu, sc3, jl * 4);
        const float invA = __frcp_rn(scA);
        const float inv3 = __frcp_rn(sc3);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          uint32_t pa2[2], d32[2];
#pragma unroll
          for (int t4 = 0; t4 < 2; ++t4) {
            const int m = sub4 * 8 + t4 * 4 + half * 32;
            pa2[t4] = foldpack4<true, false>(
                Ps[(m + 0) * PSS + j] * dos_s[m + 0], Ps[(m + 1) * PSS + j] * dos_s[m + 1],
                Ps[(m + 2) * PSS + j] * dos_s[m + 2], Ps[(m + 3) * PSS + j] * dos_s[m + 3], scA,
                invA);
            d32[t4] = foldpack4<true, true>(
                Ss[(m + 0) * PSS + j] * qs_s[m + 0], Ss[(m + 1) * PSS + j] * qs_s[m + 1],
                Ss[(m + 2) * PSS + j] * qs_s[m + 2], Ss[(m + 3) * PSS + j] * qs_s[m + 3], sc3,
                inv3);
          }
          *reinterpret_cast<uint2*>(Ap + j * QTS + sub4 * 8 + half * 32) =
              make_uint2(pa2[0], pa2[1]);
          *reinterpret_cast<uint2*>(dS3 + j * QTS + sub4 * 8 + half * 32) =
              make_uint2(d32[0], d32[1]);
        }
      }
    }
    __syncthreads();

    // ---- GEMM3 dV += sA[j]·(Ap·dOᵀ) (E4M3×E5M2) —— 本地寄存器累加，无 atomic ----
    {
      float acc[MTM34][NTM34][4];
#pragma unroll
      for (int i = 0; i < MTM34; ++i)
#pragma unroll
        for (int j = 0; j < NTM34; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bt<GM34, GN34, BM, E4E5>(Ap, QTS, dOp, PSLD, acc, wr, wc, lane, 0);
      const int r0 = wr * GM34;
#pragma unroll
      for (int i = 0; i < MTM34; ++i)
#pragma unroll
        for (int j = 0; j < NTM34; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            dVacc[i][j][q] += acc[i][j][q] * sA[r];
          }
    }

    // ---- GEMM5 dK += scale·sds3[j]·(dS3·Qᵀ) (E5M2×E4M3) —— 本地寄存器累加 ----
    {
      float acc[MTM34][NTM34][4];
#pragma unroll
      for (int i = 0; i < MTM34; ++i)
#pragma unroll
        for (int j = 0; j < NTM34; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bt<GM34, GN34, BM, E5E4>(dS3, QTS, Qp, PSLD, acc, wr, wc, lane, 0);
      const int r0 = wr * GM34;
#pragma unroll
      for (int i = 0; i < MTM34; ++i)
#pragma unroll
        for (int j = 0; j < NTM34; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            dKacc[i][j][q] += acc[i][j][q] * sds3[r] * scale;
          }
    }
    __syncthreads();  // 下一轮会覆写 Qs/dOs/Qp/dOp/Ps/Ss/Ap/dS3
  }

  // ---- 单一 owner：循环外一次 plain store（无 atomic / 无 red）----
#pragma unroll
  for (int i = 0; i < MTM34; ++i)
#pragma unroll
    for (int j = 0; j < NTM34; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = wr * GM34 + i * 16 + g + (q >= 2 ? 8 : 0);
        const int d = wc * GN34 + j * 8 + c2 + (q & 1);
        const int jg = j0 + r;
        if (jg < S) {
          dv[(((size_t)jg) * H + h) * HD + d] = dVacc[i][j][q];
          dk[(((size_t)jg) * H + h) * HD + d] = dKacc[i][j][q];
        }
      }
}

// =============================================================================
// host：npy 读取 / launcher / 对拍 / 计时
// =============================================================================
struct NpyF32 {
  std::vector<float> data;
  std::vector<long> shape;
};

static NpyF32 load_npy_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "无法打开 %s\n", path.c_str());
    std::exit(1);
  }
  char magic[6];
  f.read(magic, 6);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string header(hlen, '\0');
  f.read(&header[0], hlen);
  if (header.find("f4") == std::string::npos) {
    fprintf(stderr, "%s 不是 float32 npy\n", path.c_str());
    std::exit(1);
  }
  NpyF32 out;
  size_t sp = header.find("'shape'");
  size_t lp = header.find('(', sp);
  size_t rp = header.find(')', lp);
  std::string tup = header.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < tup.size();) {
    while (i < tup.size() && (tup[i] == ' ' || tup[i] == ',')) ++i;
    long v = 0;
    bool any = false;
    while (i < tup.size() && tup[i] >= '0' && tup[i] <= '9') {
      v = v * 10 + (tup[i] - '0');
      ++i;
      any = true;
    }
    if (any) out.shape.push_back(v);
  }
  std::streampos pos = f.tellg();
  f.seekg(0, std::ios::end);
  size_t total = (size_t)f.tellg();
  f.seekg(pos);
  size_t nbytes = total - (size_t)pos;
  out.data.resize(nbytes / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data.data()), nbytes);
  return out;
}

static bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

struct DiffStat {
  double max_abs, max_rel;
};

static DiffStat diff_stat(const std::vector<float>& a, const std::vector<float>& b) {
  DiffStat st{0.0, 0.0};
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > st.max_abs) st.max_abs = d;
    double r = d / (std::fabs((double)b[i]) + 1e-3);
    if (r > st.max_rel) st.max_rel = r;
  }
  return st;
}

int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8";
  bool causal = true;
  int iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  auto q_np = load_npy_f32(dir + "/q.npy");
  auto k_np = load_npy_f32(dir + "/k.npy");
  auto v_np = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  std::string o_name = file_exists(dir + "/ref_o.npy") ? "ref_o" : "o";
  auto o_np = load_npy_f32(dir + "/" + o_name + ".npy");
  auto rdk = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv = load_npy_f32(dir + "/ref_dv.npy");

  if (q_np.shape.size() != 4) { fprintf(stderr, "q 需 4D [B,S,H,D]\n"); return 1; }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  const int Hkv = (int)k_np.shape[2];
  if (B != 1 || D != 128) {
    fprintf(stderr, "本原型只支持 B=1、head_dim=128（当前 B=%d D=%d）\n", B, D);
    return 1;
  }
  if (Hkv != H) { fprintf(stderr, "本原型暂只支持 MHA（Hkv==H）；当前 H=%d Hkv=%d\n", H, Hkv); return 1; }

  const size_t nq = (size_t)B * S * H * D;
  const size_t nkv = (size_t)B * S * Hkv * D;
  const size_t rows_q = (size_t)B * S * H;
  const size_t rows_kv = (size_t)B * S * Hkv;
  const float scale = 1.0f / sqrtf((float)D);

  constexpr int BM = 64, BN = 32, HD = 128;
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kv_smem = [] {
    return BN * Cfg::ASLD + BN * Cfg::ASLD + BM * Cfg::ASLD + BM * Cfg::ASLD +
           (BM / 2) * Cfg::PSLD * 2 + (BM / 2) * Cfg::PSLD * 2 + BN * Cfg::QTS + BN * Cfg::QTS +
           BM * Cfg::PSS * 4 + BM * Cfg::PSS * 4 + (2 * BM + 4 * BN) * 4;
  }();

  const int lse_smem = Cfg::lse_smem_bytes;

  printf("=== F7 第三步：fp8 mma KV-owner dK/dV 原型 ===\n");
  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d Hkv=%d D=%d causal=%d scale=%.6f\n", B, S, H, Hkv, D, (int)causal,
         scale);
  printf("grid = (%d, %d); kvowner smem = %d B (%.1f KB); lse smem = %d B\n",
         (S + BN - 1) / BN, H, kv_smem, kv_smem / 1024.0, lse_smem);

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos, *d_delta, *d_lse, *d_dk, *d_dv;
  CUDA_CHECK(cudaMalloc(&d_q_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_o_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, nq));
  CUDA_CHECK(cudaMalloc(&d_k8, nkv));
  CUDA_CHECK(cudaMalloc(&d_v8, nkv));
  CUDA_CHECK(cudaMalloc(&d_do8, nq));
  CUDA_CHECK(cudaMalloc(&d_qs, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, nkv * 4));

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), nq * 4, cudaMemcpyHostToDevice));

  const int d_rows = (int)rows_q;
  const int d_blocks = (d_rows + (THREADS / 32) - 1) / (THREADS / 32);
  dim3 lg((S + LBM - 1) / LBM, H, B);
  dim3 kg((S + BN - 1) / BN, H);

  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<HD>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, lse_smem));
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_kernel<HD, BM, BN>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kv_smem));

  auto run = [&]() {
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_q_f, d_q8, d_qs, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_k_f, d_k8, d_ks, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_v_f, d_v8, d_vs, D, 0);
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_do_f, d_do8, d_dos, D, 1);
    lse_mma_kernel<HD><<<lg, THREADS, lse_smem>>>(d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                                  (int)causal);
    delta_warp_kernel<HD><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    fp8_kvowner_dkv_kernel<HD, BM, BN><<<kg, THREADS, kv_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, S, H, scale,
        (int)causal);
  };
  run();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dk(nkv), h_dv(nkv);
  CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));

  DiffStat sk = diff_stat(h_dk, rdk.data);
  DiffStat sv = diff_stat(h_dv, rdv.data);
  printf("\n[对拍] ours(KV-owner mma) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", sk.max_abs, sk.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", sv.max_abs, sv.max_rel);

  // 与既有 ours（Q-owner + atomic）dk/dv 比：应只差跨 CTA 加法次序 / 同一 fp8 口径。
  if (file_exists(dir + "/ours_dk.npy")) {
    auto odk = load_npy_f32(dir + "/ours_dk.npy");
    auto odv = load_npy_f32(dir + "/ours_dv.npy");
    DiffStat tk = diff_stat(h_dk, odk.data);
    DiffStat tv = diff_stat(h_dv, odv.data);
    printf("[对拍] vs ours(Q-owner atomic) dk max_abs=%.4e  dv max_abs=%.4e\n", tk.max_abs,
           tv.max_abs);
  }
  if (file_exists(dir + "/te_dk.npy")) {
    auto tdk = load_npy_f32(dir + "/te_dk.npy");
    auto tdv = load_npy_f32(dir + "/te_dv.npy");
    DiffStat tk = diff_stat(h_dk, tdk.data);
    DiffStat tv = diff_stat(h_dv, tdv.data);
    printf("[对拍] ref-vs-TE dk max_abs=%.4e  dv max_abs=%.4e | ours-vs-TE dk=%.4e dv=%.4e\n",
           diff_stat(rdk.data, tdk.data).max_abs, diff_stat(rdv.data, tdv.data).max_abs, tk.max_abs,
           tv.max_abs);
  }

  // 计时：preprocess（quant+lse+delta）+ main 分开
  auto bench = [&](auto fn, int it) {
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < it; ++i) fn();
    cudaEventRecord(b);
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / it;
  };
  float t_main = bench([&] {
    fp8_kvowner_dkv_kernel<HD, BM, BN><<<kg, THREADS, kv_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, S, H, scale,
        (int)causal);
  }, iters);
  float t_pre = bench([&] {
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_q_f, d_q8, d_qs, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_k_f, d_k8, d_ks, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_v_f, d_v8, d_vs, D, 0);
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_do_f, d_do8, d_dos, D, 1);
    lse_mma_kernel<HD><<<lg, THREADS, lse_smem>>>(d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                                  (int)causal);
    delta_warp_kernel<HD><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
  }, iters);
  double flops = 2.0 * (double)S * S * H * (double)HD * (causal ? 0.5 : 1.0) * 3.0;  // dK,dV 两个 GEMM 的前身（S 与 dP 另计，此处给 dK/dV 口径）
  printf("\n[计时] main(dK/dV) = %.4f ms (%.2f TF @dK/dV) | preprocess = %.4f ms | total = %.4f ms\n",
         t_main, flops / t_main / 1e9, t_pre, t_main + t_pre);
  printf("[计时] 说明：main 仅需 dK/dV，与 FP8 峰值 1978.8 TF 的占比 = %.3f%%\n",
         100.0 * flops / t_main / 1e9 / 1978.8);

  CUDA_CHECK(cudaFree(d_q_f)); CUDA_CHECK(cudaFree(d_k_f)); CUDA_CHECK(cudaFree(d_v_f));
  CUDA_CHECK(cudaFree(d_do_f)); CUDA_CHECK(cudaFree(d_o_f));
  CUDA_CHECK(cudaFree(d_q8)); CUDA_CHECK(cudaFree(d_k8)); CUDA_CHECK(cudaFree(d_v8));
  CUDA_CHECK(cudaFree(d_do8));
  CUDA_CHECK(cudaFree(d_qs)); CUDA_CHECK(cudaFree(d_ks)); CUDA_CHECK(cudaFree(d_vs));
  CUDA_CHECK(cudaFree(d_dos)); CUDA_CHECK(cudaFree(d_delta)); CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_dk)); CUDA_CHECK(cudaFree(d_dv));
  return 0;
}
