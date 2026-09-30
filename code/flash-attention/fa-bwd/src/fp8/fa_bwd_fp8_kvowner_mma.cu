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

#if defined(FA_WGMMA) && defined(FA_TMA)
// F7 第八步：Q/dO 的 4D-TMA 描述符（与主 kernel O37 的 `make_lse_map_fp8` 逐字同构）。
//   UINT8 dims={D,S,H,B}，box={128,BM,1,1}，SWIZZLE_128B（fp8 一行 128B = SW128 atom 整行）。
static CUtensorMap make_kvowner_qd_map(const void* ptr, long long H, long long S, long long D,
                                       long long B, uint32_t boxR) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D), (uint64_t)D, (uint64_t)(S * H * D)};
  uint32_t box[4] = {128, boxR, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled(kvowner) failed: %s\n", s);
    std::exit(1);
  }
  return map;
}

// F7 第十二步：dQ 的 **fp32 4D-TMA 归约描述符**（`cp.reduce.async.bulk.tensor.4d`，对标 TE 的
//   `UTMAREDG.4D.ADD`）。张量布局 = dq[B][S][H][D]（dims={D,S,H,B}，f32 元素 4B）；box =
//   {boxD, boxR, 1, 1}（一张 [boxR][boxD] 的 fp32 tile，行主序）。SWIZZLE_NONE ⇒ smem 源就是
//   行主序 tile，与 KV-owner 的 dQ staging 布局一致。
static CUtensorMap make_kvowner_dq_map_f32(const void* ptr, long long H, long long S, long long D,
                                           long long B, uint32_t boxD, uint32_t boxR) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D * 4), (uint64_t)(D * 4), (uint64_t)(S * H * D * 4)};
  uint32_t box[4] = {boxD, boxR, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled(kvowner dq f32) failed: %s\n", s);
    std::exit(1);
  }
  return map;
}
#endif

// F7 第十二步：把 KV-owner 的 dQ 跨 CTA 归约从逐 lane `red_add2` 换成 **TMA 4D tensor store-reduce**
//   （`cp.reduce.async.bulk.tensor.4d`），对标 TE 的 `UTMAREDG.4D.ADD`（p147 ncu：TE 全局 red 指令
//   仅 3168 条、ours 9.54M）。=1 时每个 (m0) 迭代把本 CTA 的 dQ tile 先写 smem 行主序，再由
//   `tid==0` 发**一条** 4D 归约回 global；=0 时退回 p157 的 `red_add2` 原子（逐位基线）。
#ifndef FA_KV_DQ_TMAR
#define FA_KV_DQ_TMAR 0
#endif

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
// F7 第七步：KV-owner dK/dV 原型的 GEMM1/2 换 Hopper `wgmma.m64n32k32`
// =============================================================================
// 动机（ROADMAP 第一百五十三轮 ncu）：KV-owner mma 原型（base/dyn）的墙已是
//   `wait 1.44 + short_scoreboard 0.88`（mma / `ldmatrix` 依赖），**不是**全局访存延迟
//   （`long_scoreboard` 仅 0.58）。F6 第二步（Q-owner `fa_bwd_fp8_wgmma2_kernel`）已证明
//   「GEMM1/2 换 wgmma」能把这两条独立 mma 的等待重叠掉；本步把同款改造搬到 KV-owner。
//
// 改动（device 数据通路）：
//   * Q/dO/K/V 的 smem 全部改存 **SW128 K-major**（fp8 一行 128B = 一个反交织 atom 的整行，
//     `sw128_off_fp8`），不再存行主序 ASLD。
//   * GEMM1 `S=scale·QKᵀ`(e4m3×e4m3) 与 GEMM2 `dP=dO·Vᵀ`(e5m2×e4m3) 用
//     `wgmma.m64n32k32`（1 warpgroup=128 线程；BM=64、BN=32）**直读 smem 描述符**，
//     两条异步 mma 一起发、统一 `wait0` 重叠。
//   * GEMM3(dV)/GEMM5(dK) 仍 `mma.m16n8k32 + ldmatrix`（fp8 wgmma 无转置操作数、MN-major
//     描述符无效，O9c-2/「阻塞」已三证判死），B 仍用 O4b 的 K 配对布局 `dOp/Qp`。
//   * rowwise scale 折算、fold、本地累加、一次 plain store 与 base **完全一致** ⇒ 数值同口径。
//
// 收益账：SW128 tile 比 ASLD 行主序更紧凑（Ks/Vs 4096 vs 4608、Qs/dOs 8192 vs 9216），
//   总 smem 67072B < base 70144B ⇒ **仍 3 CTA/SM**——即在不牺牲 occupancy 的前提下拿到
//   异步（脱离 SM80 `mma.sync` 兼容路径的 `wait`）。
#ifdef FA_WGMMA
// F7 第九步：wgmma 壳的 CTA/SM 目标（`__launch_bounds__` 的 minBlocks）。默认 3（历史最优点）；
//   置 2 让 ptxas 多用寄存器（消除 16B spill）、置 4 强制 128 regs 冲 4 CTA/SM，用于同 binary A/B。
#ifndef FA_KV_CTA
#define FA_KV_CTA 3
#endif
// F7 第九步：把 TMA 路径的 Qp/dOp 重建（`Qs/dOs` SW128 → 配对布局）**挪到 wgmma GEMM1/2 之后**，
//   让这段纯 smem 搬运/交织工作与**异步 wgmma 的延迟重叠**（原顺序：重建 → sync → wgmma → wait0，
//   重建串在 wgmma 之前）。置 0 退回原顺序做 A/B。
#ifndef FA_KV_OVL
#define FA_KV_OVL 0
#endif
// F7 第十步：KV-owner 原型补 **dQ（GEMM4）同循环**。=1 时每个 query 块计算 dQ 偏和并
//   用 `red_add2`（跨 CTA 原子）归约进 global `dq`；=0 时退回纯 dK/dV（p155 行为）。
//   注：dQ 的跨 CTA 贡献数 ≈ `S/BN`（=BM/BN × 原 dK/dV 的 `S/BM`）⇒ 本步先做**机制与
//   正确性**，是否转正取决于 atomic 归约的代价（见 docs/03 §90）。
#ifndef FA_KV_DQ
#define FA_KV_DQ 1
#endif
// F7 第十三步（BN≥BM）：GEMM1/2 的 N 维 = BN。BN=32 走 `m64n32k32`（16 累加器），BN=64 走
//   `m64n64k32`（32 累加器）。两者累加器映射同构（`d[j*4+q]` ↔ row=16w+g+(q>=2?8:0)、
//   col=j*8+c2+(q&1)，j<BN/8）⇒ epilogue 只需把 j 上界从 4 改成 BN/8。
template <int BN, int KIND>
__device__ __forceinline__ void wgmma_mn_issue(const char* Asw, const char* Bsw, int HD,
                                               float (&d)[BN / 2]) {
  if constexpr (BN == 32) {
    wgmma_mn32_issue<KIND>(Asw, Bsw, HD, d);
  } else {
#pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.f;
    wgmma_fence_fp8();
    const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
    const uint32_t sbo = (uint32_t)((HD / 128) * 1024);
#pragma unroll
    for (int s = 0; s < HD / 32; ++s) {
      uint64_t da = make_desc_sw128_fp8(sw128_k32_addr(aa, s), sbo);
      uint64_t db = make_desc_sw128_fp8(sw128_k32_addr(ba, s), sbo);
      if (KIND == 0)
        wgmma_m64n64k32_e4e4(d, da, db);
      else
        wgmma_m64n64k32_e5e4(d, da, db);
    }
    wgmma_commit_fp8();
  }
}
// F7 第十三步：BN=64 的壳用 `__launch_bounds__(THREADS, 2)`（smem ~111KB ⇒ 2 CTA/SM，
//   regs 上限 256/thread）；BN=32 沿用 `FA_KV_CTA`。
#ifndef FA_KV_CTA64
#define FA_KV_CTA64 2
#endif
// F7 第八步：把「Q/dO 的逐 ROW-pair 标量 global gather + __byte_perm 写 SW128」staging，
//   换成 **4D-TMA 一次性搬入 SW128 tile**（对标 TE 的 TMA 数据通路），再从 SW128 重建
//   Qp/dOp 的 K 配对布局（与主 kernel O37 逐字同构）。本 kernel = `TMA` 模板参数化后的
//   device body，两个 `__global__` 壳（cp.async/标量版 vs TMA 版）共用同一份逻辑。
template <int HD, int BM, int BN, bool TMA, bool TMAR = false>
__device__ __forceinline__ void fp8_kvowner_dkv_wgmma_body(
    const unsigned char* __restrict__ q8, const float* __restrict__ qs,
    const unsigned char* __restrict__ k8, const float* __restrict__ ks,
    const unsigned char* __restrict__ v8, const float* __restrict__ vs,
    const unsigned char* __restrict__ do8, const float* __restrict__ dos,
    const float* __restrict__ lse, const float* __restrict__ delta,
    float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ dq, int S, int H,
    float scale, int causal,
    const CUtensorMap* qmap, const CUtensorMap* dmap, const CUtensorMap* dqmap) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int PSLD = Cfg::PSLD;
  constexpr int QTS = Cfg::QTS;
  constexpr int PSS = Cfg::PSS;
  constexpr int NWM = 2, NWAR = 2;            // GEMM3/5 的 2×2 warp 网格（与 base 相同）
  constexpr int NTW = NWAR * 64;
  constexpr int GM34 = BN / NWM, GN34 = NTW / NWAR;
  constexpr int MTM34 = GM34 / 16, NTM34 = GN34 / 8;
  // F7 第十步：dQ（GEMM4）的输出是 [BM][HD]，4 warp 各 32×64。
  constexpr int GM5 = BM / NWM, GN5 = NTW / NWAR;
  constexpr int MTM5 = GM5 / 16, NTM5 = GN5 / 8;
  constexpr int NTFOLD = BN / 32;
  static_assert(HD == 128 && BM == 64 && (BN == 32 || BN == 64),
                "本原型固定 HD=128/BM=64，BN∈{32,64}");

  // SW128 tile 尺寸（fp8 一行 HD 字节 = 128B）。
  constexpr int sz_Ks = BN * HD;
  constexpr int sz_Vs = BN * HD;
  constexpr int sz_Qs = BM * HD;
  constexpr int sz_dOs = BM * HD;
  constexpr int sz_Qp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_dOp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_Ap = BN * QTS;
  constexpr int sz_dS3 = BN * QTS;
  constexpr int sz_Ps = BM * PSS * (int)sizeof(float);
  constexpr int sz_Ss = BM * PSS * (int)sizeof(float);
  // F7 第十步：dQ（GEMM4）的 B=K 配对布局 `Kp` 与 A=`dS2`（e5m2, per-m scale）。
  constexpr int sz_Kp = (BN / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_dS2 = BM * Cfg::DSS2;
  constexpr int sz_scales = (3 * BM + 4 * BN) * (int)sizeof(float);  // qs,dos,sds2 / ks,vs,sA,sds3
  // F7 第十二步：dQ 的 4D-TMA 归约 staging（[BM][HD] fp32 行主序；仅 FA_KV_DQ_TMAR 时分配）。
  constexpr int sz_dQst = TMAR ? (BM * HD * (int)sizeof(float)) : 0;

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
  uint16_t* Kp = reinterpret_cast<uint16_t*>(smem + off); off += sz_Kp;
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(smem + off); off += sz_dS2;
  float* Ps = reinterpret_cast<float*>(smem + off); off += sz_Ps;
  float* Ss = reinterpret_cast<float*>(smem + off); off += sz_Ss;
  float* scales = reinterpret_cast<float*>(smem + off);
  float* qs_s = scales;
  float* dos_s = qs_s + BM;
  float* ks_s = dos_s + BM;
  float* vs_s = ks_s + BN;
  float* sA = vs_s + BN;
  float* sds3 = sA + BN;
  float* sds2 = sds3 + BN;
  // F7 第十二步：dQ 的 4D-TMA 归约 staging（16B 对齐；恰好紧跟 scales）。
  constexpr int dqst_off = (sz_Ks + sz_Vs + sz_Qs + sz_dOs + sz_Qp + sz_dOp + sz_Ap + sz_dS3 +
                            sz_Kp + sz_dS2 + sz_Ps + sz_Ss + sz_scales + 15) &
                           ~15;
  float* dQst = reinterpret_cast<float*>(smem + dqst_off);

  // F7 第八步：Q/dO 4D-TMA 的两个 mbarrier（qbar/dbar），落在 scales 区之后、按 16B 对齐。
  constexpr int WGBASE =
      sz_Ks + sz_Vs + sz_Qs + sz_dOs + sz_Qp + sz_dOp + sz_Ap + sz_dS3 + sz_Kp + sz_dS2 +
      sz_Ps + sz_Ss + sz_scales + sz_dQst;
  uint64_t* qbars = reinterpret_cast<uint64_t*>(smem + ((WGBASE + 15) & ~15));
  (void)qmap;
  (void)dmap;
  (void)dqmap;

  const int j0 = blockIdx.x * BN;
  const int h = blockIdx.y;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / NWAR, wc = wid % NWAR;  // GEMM3/5 用
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // ---- 拥有的 K/V 行块：只从 global 读一次常驻 smem（**SW128**）----
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
    *reinterpret_cast<uint32_t*>(Ks + sw128_off_fp8(rp * 2, dq, HD)) = k0;
    *reinterpret_cast<uint32_t*>(Ks + sw128_off_fp8(rp * 2 + 1, dq, HD)) = k1;
    *reinterpret_cast<uint32_t*>(Vs + sw128_off_fp8(rp * 2, dq, HD)) = v0;
    *reinterpret_cast<uint32_t*>(Vs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = v1;
  }
  if (tid < BN) {
    const int jg = j0 + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)jg) * H + h] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)jg) * H + h] : 1.f;
  }
  __syncthreads();

  // F7 第十步：K 的配对布局 `Kp[rp][d]`（uint16 = {K[2rp][d], K[2rp+1][d]}），供 dQ 的
  //   GEMM4（`mma_block_bt`，B 用 `ldmatrix.x2.trans` 读出 [N=d][K=j] 片段）。K 常驻，
  //   故**整个 m-loop 只建一次**。从 SW128 `Ks` 读（4 个连续 k 在同一 16B chunk 内连续）。
#if FA_KV_DQ
  for (int u = tid; u < (BN / 2) * nd4; u += THREADS) {
    const int rp = u / nd4, dq = (u % nd4) * 4;
    const uint32_t k0 = *reinterpret_cast<const uint32_t*>(Ks + sw128_off_fp8(rp * 2, dq, HD));
    const uint32_t k1 = *reinterpret_cast<const uint32_t*>(Ks + sw128_off_fp8(rp * 2 + 1, dq, HD));
    uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * PSLD + dq);
    kpw[0] = __byte_perm(k0, k1, 0x5140);
    kpw[1] = __byte_perm(k0, k1, 0x7362);
  }
  __syncthreads();
#endif

  float dVacc[MTM34][NTM34][4];
  float dKacc[MTM34][NTM34][4];
#pragma unroll
  for (int i = 0; i < MTM34; ++i)
#pragma unroll
    for (int j = 0; j < NTM34; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) { dVacc[i][j][q] = 0.f; dKacc[i][j][q] = 0.f; }

  const int mstart = (j0 / BM) * BM;
  if constexpr (TMA) {
    if (tid == 0) { mbar_init(qbars + 0, 1); mbar_init(qbars + 1, 1); }
    __syncthreads();
  }
  uint32_t qph = 0;  // F7 第八步：Q/dO TMA mbarrier 的相位（每迭代翻转）
  for (int m0 = mstart; m0 < S; m0 += BM) {
    // F7 第九步：Qp/dOp 重建体（SW128 Qs/dOs → K 配对布局，与主 kernel O37 逐字同构）。
    //   `FA_KV_OVL=1` 时在 wgmma 之后调用，与异步 wgmma 重叠；=0 时在 wgmma 之前调用（原顺序）。
    auto build_paired_tma = [&]() {
      for (int u = tid; u < (BM / 2) * nd4; u += THREADS) {
        const int rp = u / nd4, dq = (u % nd4) * 4;
        const uint32_t q0 = *reinterpret_cast<const uint32_t*>(Qs + sw128_off_fp8(rp * 2, dq, HD));
        const uint32_t q1 = *reinterpret_cast<const uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, dq, HD));
        const uint32_t o0 = *reinterpret_cast<const uint32_t*>(dOs + sw128_off_fp8(rp * 2, dq, HD));
        const uint32_t o1 = *reinterpret_cast<const uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, dq, HD));
        uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + dq);
        qpw[0] = __byte_perm(q0, q1, 0x5140);
        qpw[1] = __byte_perm(q0, q1, 0x7362);
        uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + dq);
        opw[0] = __byte_perm(o0, o1, 0x5140);
        opw[1] = __byte_perm(o0, o1, 0x7362);
      }
    };
    // ---- staging：Q/dO 行 [m0,m0+BM) → SW128（Qs/dOs）+ 配对布局（Qp/dOp）----
    if constexpr (TMA) {
      // F7 第八步：4D-TMA 一次性把 Q/dO 搬进 SW128 tile（fp8 一行 128B = SW128 atom 整行）。
      if (tid == 0) {
        mbar_arrive_expect(qbars + 0, (uint32_t)(BM * HD));
        tma_load_4d(Qs, qmap, 0, m0, h, 0, qbars + 0);
        mbar_arrive_expect(qbars + 1, (uint32_t)(BM * HD));
        tma_load_4d(dOs, dmap, 0, m0, h, 0, qbars + 1);
      }
      mbar_wait(qbars + 0, qph);
      mbar_wait(qbars + 1, qph);
      qph ^= 1;
      __syncthreads();
      if (!FA_KV_OVL) build_paired_tma();
    } else {
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
        *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2, dq, HD)) = q0;
        *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = q1;
        *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2, dq, HD)) = o0;
        *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = o1;
        uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + dq);
        qpw[0] = __byte_perm(q0, q1, 0x5140);
        qpw[1] = __byte_perm(q0, q1, 0x7362);
        uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + dq);
        opw[0] = __byte_perm(o0, o1, 0x5140);
        opw[1] = __byte_perm(o0, o1, 0x7362);
      }
    }
    if (tid < BM) {
      const int qi = m0 + tid;
      qs_s[tid] = (qi < S) ? qs[((size_t)qi) * H + h] : 1.f;
      dos_s[tid] = (qi < S) ? dos[((size_t)qi) * H + h] : 1.f;
    }
    // F7 第八步修正：非 TMA 路径的 Qs/dOs 是 generic 写，wgmma 读前需 async-proxy fence
    //   （TMA 路径 Qs/dOs 本身就是 async 写，此 fence 对其冗余但无害）。
    bulk_reduce_fence();
    __syncthreads();

    // ---- LSE/D 预装寄存器（wgmma 布局：本线程持 2 个行槽 r0=wid*16+g、r0+8）----
    const int r0w = wid * 16 + g;
    const int qa_r = m0 + r0w, qb_r = m0 + r0w + 8;
    const float lse_a = (qa_r < S) ? lse[((size_t)qa_r) * H + h] : 0.f;
    const float lse_b = (qb_r < S) ? lse[((size_t)qb_r) * H + h] : 0.f;
    const float del_a = (qa_r < S) ? delta[((size_t)qa_r) * H + h] : 0.f;
    const float del_b = (qb_r < S) ? delta[((size_t)qb_r) * H + h] : 0.f;

    // ---- GEMM1 S=scale·QKᵀ (e4m3×e4m3) + GEMM2 dP=dO·Vᵀ (e5m2×e4m3)：wgmma 直读 SW128 ----
    {
      float sacc[BN / 2], dpacc[BN / 2];
      wgmma_mn_issue<BN, 0>(reinterpret_cast<const char*>(Qs), reinterpret_cast<const char*>(Ks), HD,
                            sacc);
      wgmma_mn_issue<BN, 1>(reinterpret_cast<const char*>(dOs), reinterpret_cast<const char*>(Vs), HD,
                            dpacc);
      // F7 第九步：OVL=1 时在此重建 Qp/dOp（与上面两条异步 wgmma 重叠），再等 wgmma。
      if (FA_KV_OVL && TMA) build_paired_tma();
      wgmma_wait0_fp8();
#pragma unroll
      for (int j = 0; j < BN / 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          const int c = j * 8 + c2 + (q & 1);
          const int qi = m0 + r, jg = j0 + c;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi)) {
            const float lv = (q >= 2) ? lse_b : lse_a;
            const float sval = sacc[j * 4 + q] * scale * qs_s[r] * ks_s[c];
            p = fexp(sval - lv);
          }
          Ps[r * PSS + c] = p;
        }
#pragma unroll
      for (int j = 0; j < BN / 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          const int c = j * 8 + c2 + (q & 1);
          const float dpv = dpacc[j * 4 + q] * dos_s[r] * vs_s[c];
          const float del = (q >= 2) ? del_b : del_a;
          Ss[r * PSS + c] = Ps[r * PSS + c] * (dpv - del);
        }
    }
    __syncthreads();

    // ---- fold：Ap/dS3（与 base 逐字相同）----
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
        if (sub4 == 0) { sA[j] = scA; sds3[j] = sc3; }
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
#if FA_KV_DQ
    // ---- fold：dS2[m][j]=dS[m][j]·ks[j] (e5m2, per-m scale sds2[m])，供 GEMM4 dQ ----
    //   与 Q-owner 主 kernel 的 dS2 fold 逐字同款（4 warp × 16 行 × 2 lane，16B 向量写）。
    if (wid < 4) {
      const int ml = lane >> 1, sub2 = lane & 1;
      const int m = wid * 16 + ml;
      float amax2 = 0.f;
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh)
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          const int j = sub2 * 16 + t + jh * 32;
          amax2 = fmaxf(amax2, fabsf(Ss[m * PSS + j] * ks_s[j]));
        }
      amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, 1));
      float sc2 = (amax2 > 0.f) ? amax2 / kE5M2Max : 1.f;
      if (sub2 == 0) sds2[m] = sc2;
      sc2 = __shfl_sync(0xffffffffu, sc2, ml * 2);
      const float inv2 = __frcp_rn(sc2);
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh) {
        uint32_t d2_4[4];
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
          const int j = sub2 * 16 + t4 * 4 + jh * 32;
          d2_4[t4] = foldpack4<true, true>(
              Ss[m * PSS + j + 0] * ks_s[j + 0], Ss[m * PSS + j + 1] * ks_s[j + 1],
              Ss[m * PSS + j + 2] * ks_s[j + 2], Ss[m * PSS + j + 3] * ks_s[j + 3], sc2, inv2);
        }
        *reinterpret_cast<uint4*>(dS2 + m * Cfg::DSS2 + sub2 * 16 + jh * 32) =
            make_uint4(d2_4[0], d2_4[1], d2_4[2], d2_4[3]);
      }
    }
#endif
    __syncthreads();

    // ---- GEMM3 dV += sA[j]·(Ap·dOᵀ) (e4m3×e5m2) —— 本地寄存器累加（同 base）----
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

    // ---- GEMM5 dK += scale·sds3[j]·(dS3·Qᵀ) (e5m2×e4m3) —— 本地寄存器累加 ----
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
#if FA_KV_DQ
    // ---- GEMM4 dQ += scale·(dS2·K) —— **同循环**，但 dQ 跨 KV 块（跨 CTA）不归本 CTA 独占：
    //   dQ 元素 (m,d) 收到所有 j<=m 的 KV 块的贡献。
    //   A=dS2[m][j] (e5m2), B=Kp[j/2][d] (e4m3, ldmatrix.trans)，输出 [BM][HD]，乘 sds2[m]·scale。
    //   F7 第十二步（FA_KV_DQ_TMAR）：归约从逐 lane `red_add2` 换成 **TMA 4D tensor store-reduce**
    //   （对标 TE 的 `UTMAREDG.4D.ADD`）——先把整块 [BM][HD] 写 smem 行主序，再由 tid0 发**一条**
    //   `cp.reduce.async.bulk.tensor.4d` 归约回 global。
    {
      float acc[MTM5][NTM5][4];
#pragma unroll
      for (int i = 0; i < MTM5; ++i)
#pragma unroll
        for (int j = 0; j < NTM5; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bt<GM5, GN5, BN, E5E4>(dS2, Cfg::DSS2, Kp, PSLD, acc, wr, wc, lane, 0);
      const int r0 = wr * GM5, c0 = wc * GN5;
      if constexpr (TMAR && TMA) {
        // F7 第十二步：先等上一迭代的 dQ 4D-TMA 归约读完 dQst 再覆写——放在这里（GEMM1/2/
        //   fold/GEMM3/5 之后）让归约与整轮计算充分重叠（放 loop-top 会立即串行化，慢 1.35×）。
        if (tid == 0) bulk_reduce_wait0();
        __syncthreads();
        // TMA 4D tensor reduce 的 box 内维 ≤ 256B（fp32 即 64 列）⇒ 沿 D 维拆 2 个 64 列 chunk，
        //   每个 chunk 在 smem 里紧凑存成 [BM][64]（行距 64），各发一条 reduce。
        constexpr int CH = 64;  // chunk 列数
#pragma unroll
        for (int i = 0; i < MTM5; ++i)
#pragma unroll
          for (int j = 0; j < NTM5; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              const int c = c0 + j * 8 + c2 + (q & 1);
              const int ch = c / CH, cc = c % CH;
              dQst[ch * (BM * CH) + r * CH + cc] =
                  (m0 + r < S) ? acc[i][j][q] * sds2[r] * scale : 0.f;
            }
        __syncthreads();
        if (tid == 0) {
          bulk_reduce_fence();  // generic smem 写 → async proxy 可见
#pragma unroll
          for (int ch = 0; ch < HD / CH; ++ch)
            tma_reduce_add_4d_f32(dqmap, dQst + ch * (BM * CH), /*c0=*/ch * CH, /*c1=*/m0,
                                  /*c2=*/h, /*c3=*/0);
          bulk_reduce_commit();
        }
      } else {
#pragma unroll
        for (int i = 0; i < MTM5; ++i)
#pragma unroll
          for (int j = 0; j < NTM5; ++j)
#pragma unroll
            for (int q = 0; q < 4; q += 2) {
              const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              const int c = c0 + j * 8 + c2;
              const int qi = m0 + r;
              if (qi < S)
                red_add2(dq + (((size_t)qi) * H + h) * HD + c, acc[i][j][q] * sds2[r] * scale,
                         acc[i][j][q + 1] * sds2[r] * scale);
            }
      }
    }
#endif
    __syncthreads();
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
// F7 第十四步：KV-owner **column-owner**（每 CTA 拥有 RCOL 个连续 KV 块）
// =============================================================================
// 动机（承接 §93.6「唯一未试 = 跨 warpgroup 偏和 + 二次归约」）：p157 的 KV-owner 把 dK/dV
//   的跨 CTA `red` 打成 **0**，但 dQ 仍逐 lane `red_add2`，且 dQ 元素被**所有** KV 块各贡献
//   一次（贡献数 = S/BN = 64）⇒ 总 `red` 又回到 ~102M ≈ 默认 Q-owner 的 114M（p157 判决）。
//   要真正降 `red`，必须让同一 CTA 拥有**多个连续 KV 块**，并在一次 m-迭代里把它们对 dQ 的
//   偏和**先在本地累加、再只发一次原子** ⇒ dQ 贡献数 ÷ RCOL。
//   * p160 的 BN≥BM 也把 red 减半，但把 **GEMM1/2 累加器**（BN 32→64）一起翻倍 ⇒ 顶穿 255
//     寄存器、溢出盖过收益。
//   * 本方案 **BN 仍 32**，GEMM1/2 累加器不变；只翻倍 **dK/dV 累加器**（RCOL 份）+ 一个
//     dQ 偏和累加器，寄存器更省（p160 的教训的正面解）。
// 结构：grid.x = ceil(S/(RCOL·BN))，每 CTA：
//   * K/V/Kp（RCOL 份）常驻 smem，只从 global 读一次；dK/dV 寄存器累加（RCOL 份），循环末
//     每块一次 plain store ⇒ dK/dV `red`=0；
//   * Q/dO staging（4D-TMA）与 Qp/dOp 配对布局**每 m 一次、RCOL 列共享**；
//   * 内层 `for cc in 0..RCOL`：GEMM1/2(wgmma) → P/S → fold(Ap/dS3/dS2) → GEMM3/5（累加进
//     `dVacc[cc]/dKacc[cc]`）→ GEMM4（dQ 偏和累加进 `dqacc`，乘 `sds2[m]·scale`）；
//   * 内层退出后 `dqacc` 用 `red_add2` **一次**写出 ⇒ dQ 贡献数 ÷ RCOL。
// 数值：与 KV-owner 单块版同口径（同一量化/折算/mma 布局），只差跨 CTA 加法次序 ⇒ dk/dv 对
//   ref 与既有原型同量级，dq 与单块版差 fp8 噪声。
#ifdef FA_WGMMA
#ifndef FA_KV_COL_CTA
#define FA_KV_COL_CTA 2
#endif
template <int HD, int BM, int BN, bool TMA, int RCOL>
__device__ __forceinline__ void fp8_kvowner_col_body(
    const unsigned char* __restrict__ q8, const float* __restrict__ qs,
    const unsigned char* __restrict__ k8, const float* __restrict__ ks,
    const unsigned char* __restrict__ v8, const float* __restrict__ vs,
    const unsigned char* __restrict__ do8, const float* __restrict__ dos,
    const float* __restrict__ lse, const float* __restrict__ delta,
    float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ dq, int S, int H,
    float scale, int causal, const CUtensorMap* qmap, const CUtensorMap* dmap) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int PSLD = Cfg::PSLD;
  constexpr int QTS = Cfg::QTS;
  constexpr int PSS = Cfg::PSS;
  constexpr int NWM = 2, NWAR = 2;
  constexpr int NTW = NWAR * 64;
  constexpr int GM34 = BN / NWM, GN34 = NTW / NWAR;
  constexpr int MTM34 = GM34 / 16, NTM34 = GN34 / 8;
  constexpr int GM5 = BM / NWM, GN5 = NTW / NWAR;
  constexpr int MTM5 = GM5 / 16, NTM5 = GN5 / 8;
  constexpr int NTFOLD = BN / 32;
  static_assert(HD == 128 && BM == 64 && BN == 32, "col-owner 固定 HD=128/BM=64/BN=32");

  constexpr int sz_Ks = BN * HD;
  constexpr int sz_Vs = BN * HD;
  constexpr int sz_Qs = BM * HD;
  constexpr int sz_dOs = BM * HD;
  constexpr int sz_Qp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_dOp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_Ap = BN * QTS;
  constexpr int sz_dS3 = BN * QTS;
  constexpr int sz_Kp = (BN / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_dS2 = BM * Cfg::DSS2;
  constexpr int sz_Ps = BM * PSS * (int)sizeof(float);
  constexpr int sz_Ss = BM * PSS * (int)sizeof(float);
  constexpr int sz_scales = (3 * BM + 2 * BN + 2 * RCOL * BN) * (int)sizeof(float);

  extern __shared__ __align__(16) char smem[];
  int off = 0;
  unsigned char* Ks = reinterpret_cast<unsigned char*>(smem + off); off += RCOL * sz_Ks;
  unsigned char* Vs = reinterpret_cast<unsigned char*>(smem + off); off += RCOL * sz_Vs;
  unsigned char* Qs = reinterpret_cast<unsigned char*>(smem + off); off += sz_Qs;
  unsigned char* dOs = reinterpret_cast<unsigned char*>(smem + off); off += sz_dOs;
  uint16_t* Qp = reinterpret_cast<uint16_t*>(smem + off); off += sz_Qp;
  uint16_t* dOp = reinterpret_cast<uint16_t*>(smem + off); off += sz_dOp;
  unsigned char* Ap = reinterpret_cast<unsigned char*>(smem + off); off += sz_Ap;
  unsigned char* dS3 = reinterpret_cast<unsigned char*>(smem + off); off += sz_dS3;
  uint16_t* Kp = reinterpret_cast<uint16_t*>(smem + off); off += RCOL * sz_Kp;
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(smem + off); off += sz_dS2;
  float* Ps = reinterpret_cast<float*>(smem + off); off += sz_Ps;
  float* Ss = reinterpret_cast<float*>(smem + off); off += sz_Ss;
  float* scales = reinterpret_cast<float*>(smem + off);
  float* qs_s = scales;
  float* dos_s = qs_s + BM;
  float* sA = dos_s + BM;
  float* sds3 = sA + BN;
  float* sds2 = sds3 + BN;
  float* ks_s = sds2 + BM;             // [RCOL*BN]
  float* vs_s = ks_s + RCOL * BN;      // [RCOL*BN]
  constexpr int WGBASE = RCOL * sz_Ks + RCOL * sz_Vs + sz_Qs + sz_dOs + sz_Qp + sz_dOp + sz_Ap +
                         sz_dS3 + RCOL * sz_Kp + sz_dS2 + sz_Ps + sz_Ss + sz_scales;
  uint64_t* qbars = reinterpret_cast<uint64_t*>(smem + ((WGBASE + 15) & ~15));
  (void)qmap;
  (void)dmap;

  const int j0b = blockIdx.x * (RCOL * BN);
  const int h = blockIdx.y;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / NWAR, wc = wid % NWAR;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int nd4 = HD / 4;

  // ---- RCOL 个 KV 行块常驻 smem（SW128）：只从 global 读一次 ----
  for (int cc = 0; cc < RCOL; ++cc) {
    const int j0 = j0b + cc * BN;
    unsigned char* Ksc = Ks + cc * sz_Ks;
    unsigned char* Vsc = Vs + cc * sz_Vs;
    for (int u = tid; u < (BN / 2) * nd4; u += THREADS) {
      const int rp = u / nd4, d4 = (u % nd4) * 4;
      const int jr = j0 + rp * 2;
      uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
      if (jr < S) {
        size_t i0 = (((size_t)jr) * H + h) * HD + d4;
        k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
        v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
      }
      if (jr + 1 < S) {
        size_t i1 = (((size_t)(jr + 1)) * H + h) * HD + d4;
        k1 = *reinterpret_cast<const uint32_t*>(k8 + i1);
        v1 = *reinterpret_cast<const uint32_t*>(v8 + i1);
      }
      *reinterpret_cast<uint32_t*>(Ksc + sw128_off_fp8(rp * 2, d4, HD)) = k0;
      *reinterpret_cast<uint32_t*>(Ksc + sw128_off_fp8(rp * 2 + 1, d4, HD)) = k1;
      *reinterpret_cast<uint32_t*>(Vsc + sw128_off_fp8(rp * 2, d4, HD)) = v0;
      *reinterpret_cast<uint32_t*>(Vsc + sw128_off_fp8(rp * 2 + 1, d4, HD)) = v1;
    }
    if (tid < BN) {
      const int jg = j0 + tid;
      ks_s[cc * BN + tid] = (jg < S) ? ks[((size_t)jg) * H + h] : 1.f;
      vs_s[cc * BN + tid] = (jg < S) ? vs[((size_t)jg) * H + h] : 1.f;
    }
  }
  __syncthreads();

  // ---- K 的配对布局 Kp（每列一份），供 GEMM4 的 B（ldmatrix.x2.trans）----
  for (int cc = 0; cc < RCOL; ++cc) {
    const unsigned char* Ksc = Ks + cc * sz_Ks;
    uint16_t* Kpc = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Kp) + cc * sz_Kp);
    for (int u = tid; u < (BN / 2) * nd4; u += THREADS) {
      const int rp = u / nd4, d4 = (u % nd4) * 4;
      const uint32_t k0 = *reinterpret_cast<const uint32_t*>(Ksc + sw128_off_fp8(rp * 2, d4, HD));
      const uint32_t k1 = *reinterpret_cast<const uint32_t*>(Ksc + sw128_off_fp8(rp * 2 + 1, d4, HD));
      uint32_t* kpw = reinterpret_cast<uint32_t*>(Kpc + rp * PSLD + d4);
      kpw[0] = __byte_perm(k0, k1, 0x5140);
      kpw[1] = __byte_perm(k0, k1, 0x7362);
    }
  }
  __syncthreads();

  // ---- dK/dV 寄存器累加器（RCOL 份）+ dQ 偏和（1 份，跨 cc）----
  float dVacc[RCOL][MTM34][NTM34][4];
  float dKacc[RCOL][MTM34][NTM34][4];
#pragma unroll
  for (int cc = 0; cc < RCOL; ++cc)
#pragma unroll
    for (int i = 0; i < MTM34; ++i)
#pragma unroll
      for (int j = 0; j < NTM34; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) { dVacc[cc][i][j][q] = 0.f; dKacc[cc][i][j][q] = 0.f; }

  const int mstart = (j0b / BM) * BM;
  if constexpr (TMA) {
    if (tid == 0) { mbar_init(qbars + 0, 1); mbar_init(qbars + 1, 1); }
    __syncthreads();
  }
  uint32_t qph = 0;
  for (int m0 = mstart; m0 < S; m0 += BM) {
    // ---- Q/dO staging（每 m 一次，RCOL 列共享）----
    if constexpr (TMA) {
      if (tid == 0) {
        mbar_arrive_expect(qbars + 0, (uint32_t)(BM * HD));
        tma_load_4d(Qs, qmap, 0, m0, h, 0, qbars + 0);
        mbar_arrive_expect(qbars + 1, (uint32_t)(BM * HD));
        tma_load_4d(dOs, dmap, 0, m0, h, 0, qbars + 1);
      }
      mbar_wait(qbars + 0, qph);
      mbar_wait(qbars + 1, qph);
      qph ^= 1;
      __syncthreads();
    } else {
      for (int u = tid; u < (BM / 2) * nd4; u += THREADS) {
        const int rp = u / nd4, d4 = (u % nd4) * 4;
        const int qa = m0 + rp * 2, qb = m0 + rp * 2 + 1;
        uint32_t q0 = 0, q1 = 0, o0 = 0, o1 = 0;
        if (qa < S) {
          size_t idx = (((size_t)qa) * H + h) * HD + d4;
          q0 = *reinterpret_cast<const uint32_t*>(q8 + idx);
          o0 = *reinterpret_cast<const uint32_t*>(do8 + idx);
        }
        if (qb < S) {
          size_t idx = (((size_t)qb) * H + h) * HD + d4;
          q1 = *reinterpret_cast<const uint32_t*>(q8 + idx);
          o1 = *reinterpret_cast<const uint32_t*>(do8 + idx);
        }
        *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2, d4, HD)) = q0;
        *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, d4, HD)) = q1;
        *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2, d4, HD)) = o0;
        *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, d4, HD)) = o1;
      }
    }
    // Q/dO 配对布局（SW128 → K 配对）
    for (int u = tid; u < (BM / 2) * nd4; u += THREADS) {
      const int rp = u / nd4, d4 = (u % nd4) * 4;
      const uint32_t q0 = *reinterpret_cast<const uint32_t*>(Qs + sw128_off_fp8(rp * 2, d4, HD));
      const uint32_t q1 = *reinterpret_cast<const uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, d4, HD));
      const uint32_t o0 = *reinterpret_cast<const uint32_t*>(dOs + sw128_off_fp8(rp * 2, d4, HD));
      const uint32_t o1 =
          *reinterpret_cast<const uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, d4, HD));
      uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + d4);
      qpw[0] = __byte_perm(q0, q1, 0x5140);
      qpw[1] = __byte_perm(q0, q1, 0x7362);
      uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + d4);
      opw[0] = __byte_perm(o0, o1, 0x5140);
      opw[1] = __byte_perm(o0, o1, 0x7362);
    }
    if (tid < BM) {
      const int qi = m0 + tid;
      qs_s[tid] = (qi < S) ? qs[((size_t)qi) * H + h] : 1.f;
      dos_s[tid] = (qi < S) ? dos[((size_t)qi) * H + h] : 1.f;
    }
    bulk_reduce_fence();
    __syncthreads();

    // ---- LSE/D 预装寄存器 ----
    const int r0w = wid * 16 + g;
    const int qa_r = m0 + r0w, qb_r = m0 + r0w + 8;
    const float lse_a = (qa_r < S) ? lse[((size_t)qa_r) * H + h] : 0.f;
    const float lse_b = (qb_r < S) ? lse[((size_t)qb_r) * H + h] : 0.f;
    const float del_a = (qa_r < S) ? delta[((size_t)qa_r) * H + h] : 0.f;
    const float del_b = (qb_r < S) ? delta[((size_t)qb_r) * H + h] : 0.f;

    float dqacc[MTM5][NTM5][4];
#pragma unroll
    for (int i = 0; i < MTM5; ++i)
#pragma unroll
      for (int j = 0; j < NTM5; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;

    for (int cc = 0; cc < RCOL; ++cc) {
      const int j0 = j0b + cc * BN;
      const unsigned char* Ksc = Ks + cc * sz_Ks;
      const unsigned char* Vsc = Vs + cc * sz_Vs;
      const float* ksc = ks_s + cc * BN;
      const float* vsc = vs_s + cc * BN;

      // GEMM1 S=scale·QKᵀ (e4e4) + GEMM2 dP=dO·Vᵀ (e5e4) + epilogue P/S
      {
        float sacc[BN / 2], dpacc[BN / 2];
        wgmma_mn_issue<BN, 0>(reinterpret_cast<const char*>(Qs), reinterpret_cast<const char*>(Ksc),
                              HD, sacc);
        wgmma_mn_issue<BN, 1>(reinterpret_cast<const char*>(dOs), reinterpret_cast<const char*>(Vsc),
                              HD, dpacc);
        wgmma_wait0_fp8();
#pragma unroll
        for (int j = 0; j < BN / 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int r = wid * 16 + g + (q >= 2 ? 8 : 0);
            const int c = j * 8 + c2 + (q & 1);
            const int qi = m0 + r, jg = j0 + c;
            float p = 0.f;
            if (qi < S && jg < S && !(causal && jg > qi)) {
              const float lv = (q >= 2) ? lse_b : lse_a;
              const float sval = sacc[j * 4 + q] * scale * qs_s[r] * ksc[c];
              p = fexp(sval - lv);
            }
            Ps[r * PSS + c] = p;
          }
#pragma unroll
        for (int j = 0; j < BN / 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int r = wid * 16 + g + (q >= 2 ? 8 : 0);
            const int c = j * 8 + c2 + (q & 1);
            const float dpv = dpacc[j * 4 + q] * dos_s[r] * vsc[c];
            const float del = (q >= 2) ? del_b : del_a;
            Ss[r * PSS + c] = Ps[r * PSS + c] * (dpv - del);
          }
      }
      __syncthreads();

      // fold：Ap=P·dos[m] (e4m3, per-j) / dS3=dS·qs[m] (e5m2, per-j) / dS2=dS·ks[j] (e5m2, per-m)
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
        // dS2 fold（per-m, use ksc）
        const int ml = lane >> 1, sub2 = lane & 1;
        const int m = wid * 16 + ml;
        float amax2 = 0.f;
#pragma unroll
        for (int jh = 0; jh < NTFOLD; ++jh)
#pragma unroll
          for (int t = 0; t < 16; ++t) {
            const int j = sub2 * 16 + t + jh * 32;
            amax2 = fmaxf(amax2, fabsf(Ss[m * PSS + j] * ksc[j]));
          }
        amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, 1));
        float sc2 = (amax2 > 0.f) ? amax2 / kE5M2Max : 1.f;
        if (sub2 == 0) sds2[m] = sc2;
        sc2 = __shfl_sync(0xffffffffu, sc2, ml * 2);
        const float inv2 = __frcp_rn(sc2);
#pragma unroll
        for (int jh = 0; jh < NTFOLD; ++jh) {
          uint32_t d2_4[4];
#pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            const int j = sub2 * 16 + t4 * 4 + jh * 32;
            d2_4[t4] = foldpack4<true, true>(
                Ss[m * PSS + j + 0] * ksc[j + 0], Ss[m * PSS + j + 1] * ksc[j + 1],
                Ss[m * PSS + j + 2] * ksc[j + 2], Ss[m * PSS + j + 3] * ksc[j + 3], sc2, inv2);
          }
          *reinterpret_cast<uint4*>(dS2 + m * Cfg::DSS2 + sub2 * 16 + jh * 32) =
              make_uint4(d2_4[0], d2_4[1], d2_4[2], d2_4[3]);
        }
      }
      __syncthreads();

      // GEMM3 dV += sA·(Ap·dOᵀ)  —— 累加进 dVacc[cc]
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
              dVacc[cc][i][j][q] += acc[i][j][q] * sA[r];
            }
      }
      // GEMM5 dK += scale·sds3·(dS3·Qᵀ)  —— 累加进 dKacc[cc]
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
              dKacc[cc][i][j][q] += acc[i][j][q] * sds3[r] * scale;
            }
      }
      // GEMM4 dQ 偏和（本 cc）—— 累加进 dqacc
      {
        float acc[MTM5][NTM5][4];
#pragma unroll
        for (int i = 0; i < MTM5; ++i)
#pragma unroll
          for (int j = 0; j < NTM5; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        const uint16_t* Kpc =
            reinterpret_cast<const uint16_t*>(reinterpret_cast<const unsigned char*>(Kp) + cc * sz_Kp);
        mma_block_bt<GM5, GN5, BN, E5E4>(dS2, Cfg::DSS2, Kpc, PSLD, acc, wr, wc, lane, 0);
        const int r0 = wr * GM5, c0 = wc * GN5;
#pragma unroll
        for (int i = 0; i < MTM5; ++i)
#pragma unroll
          for (int j = 0; j < NTM5; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              const int c = c0 + j * 8 + c2 + (q & 1);
              const int qi = m0 + r;
              if (qi < S) dqacc[i][j][q] += acc[i][j][q] * sds2[r] * scale;
            }
      }
      __syncthreads();  // 下一 cc 覆写 Ps/Ss/Ap/dS3/dS2
    }

    // dQ：**每 m 一次** `red_add2`（跨 RCOL 列先本地累加）⇒ 贡献数 ÷ RCOL
    {
      const int r0 = wr * GM5, c0 = wc * GN5;
#pragma unroll
      for (int i = 0; i < MTM5; ++i)
#pragma unroll
        for (int j = 0; j < NTM5; ++j)
#pragma unroll
          for (int q = 0; q < 4; q += 2) {
            const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            const int c = c0 + j * 8 + c2;
            const int qi = m0 + r;
            if (qi < S) red_add2(dq + (((size_t)qi) * H + h) * HD + c, dqacc[i][j][q],
                                 dqacc[i][j][q + 1]);
          }
    }
    __syncthreads();
  }

  // ---- 每列一次 plain store（无 atomic / 无 red）----
#pragma unroll
  for (int cc = 0; cc < RCOL; ++cc) {
    const int j0 = j0b + cc * BN;
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
            dv[(((size_t)jg) * H + h) * HD + d] = dVacc[cc][i][j][q];
            dk[(((size_t)jg) * H + h) * HD + d] = dKacc[cc][i][j][q];
          }
        }
  }
}

#if defined(FA_TMA)
template <int HD, int BM, int BN, int RCOL>
__global__ void __launch_bounds__(THREADS, FA_KV_COL_CTA)
fp8_kvowner_dkv_col_tma_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                               const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                               const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                               const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                               const float* __restrict__ lse, const float* __restrict__ delta,
                               float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ dq,
                               int S, int H, float scale, int causal,
                               const __grid_constant__ CUtensorMap qmap,
                               const __grid_constant__ CUtensorMap dmap) {
  fp8_kvowner_col_body<HD, BM, BN, true, RCOL>(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, dk, dv,
                                               dq, S, H, scale, causal, &qmap, &dmap);
}
#endif  // FA_TMA
#endif  // FA_WGMMA

// F7 第七步壳：非 TMA（标量 global gather staging）——行为与 p154 逐字一致。
template <int HD, int BM, int BN>
__global__ void __launch_bounds__(THREADS, FA_KV_CTA)
fp8_kvowner_dkv_wgmma_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                             const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                             const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                             const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                             const float* __restrict__ lse, const float* __restrict__ delta,
                             float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ dq,
                             int S, int H,
                             float scale, int causal) {
  fp8_kvowner_dkv_wgmma_body<HD, BM, BN, false, false>(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, dk,
                                                dv, dq, S, H, scale, causal, nullptr, nullptr, nullptr);
}
#ifdef FA_TMA
// F7 第八步壳：Q/dO 走 4D-TMA（需 `-DFA_TMA -lcuda` + sm90a gencode）。
// F7 第十三步：BN=64（smem ~111KB）自动切 2 CTA/SM；BN=32 沿用 FA_KV_CTA。
template <int HD, int BM, int BN>
__global__ void __launch_bounds__(THREADS, (BN == 64 ? FA_KV_CTA64 : FA_KV_CTA))
fp8_kvowner_dkv_wgmma_tma_kernel(const unsigned char* __restrict__ q8,
                                 const float* __restrict__ qs,
                                 const unsigned char* __restrict__ k8,
                                 const float* __restrict__ ks,
                                 const unsigned char* __restrict__ v8,
                                 const float* __restrict__ vs,
                                 const unsigned char* __restrict__ do8,
                                 const float* __restrict__ dos,
                                 const float* __restrict__ lse,
                                 const float* __restrict__ delta, float* __restrict__ dk,
                                 float* __restrict__ dv, float* __restrict__ dq, int S, int H,
                                 float scale, int causal,
                                 const __grid_constant__ CUtensorMap qmap,
                                 const __grid_constant__ CUtensorMap dmap) {
  fp8_kvowner_dkv_wgmma_body<HD, BM, BN, true, false>(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, dk, dv,
                                               dq, S, H, scale, causal, &qmap, &dmap, nullptr);
}
// F7 第十二步壳：在 TMA 版基础上多传一个 **dQ 的 fp32 4D-TMA 归约描述符**（`FA_KV_DQ_TMAR`）。
template <int HD, int BM, int BN>
__global__ void __launch_bounds__(THREADS, FA_KV_CTA)
fp8_kvowner_dkv_wgmma_tma_r_kernel(const unsigned char* __restrict__ q8,
                                   const float* __restrict__ qs,
                                   const unsigned char* __restrict__ k8,
                                   const float* __restrict__ ks,
                                   const unsigned char* __restrict__ v8,
                                   const float* __restrict__ vs,
                                   const unsigned char* __restrict__ do8,
                                   const float* __restrict__ dos,
                                   const float* __restrict__ lse,
                                   const float* __restrict__ delta, float* __restrict__ dk,
                                   float* __restrict__ dv, float* __restrict__ dq, int S, int H,
                                   float scale, int causal,
                                   const __grid_constant__ CUtensorMap qmap,
                                   const __grid_constant__ CUtensorMap dmap,
                                   const __grid_constant__ CUtensorMap dqmap) {
  fp8_kvowner_dkv_wgmma_body<HD, BM, BN, true, true>(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, dk, dv,
                                               dq, S, H, scale, causal, &qmap, &dmap, &dqmap);
}
#endif
#endif  // FA_WGMMA

// =============================================================================
// F7 第四步：KV-owner dK/dV 原型 + **Q/dO staging 的 cp.async 双缓冲重叠**
// =============================================================================
// 动机（ROADMAP 第一百五十轮「下一步候选 ①/②」）：第 150 轮的 KV-owner mma 原型虽然
// `red=0`，但每个 query 块的 Q/dO staging 是**同步全局读 + `__syncthreads`**，全局延迟
// 串在 GEMM1 之前（ncu `long_scoreboard`）；且每次迭代的 staging 无法与上一次迭代的
// GEMM 重叠。本变体把 Q/dO 的 staging 换成 **`cp.async.cg` 16B 双缓冲流水**：prologue 发
// stage0，循环里先 issue 下一 query 块的 stage、再 `wait_group 1` 等当前 stage，然后
// 在循环体里算 5 个 GEMM——把全局读延迟藏到上一块的 compute 后面。
//
// 代价：staging 缓冲（Qs/dOs/Qp/dOp）翻倍 ⇒ smem 70144→105984B ⇒ 1 CTA/SM 的 3 档降到
// **2 CTA/SM**。本变体就是用来做「重叠收益 vs occupancy 损失」的同 binary A/B。
//
// 数值口径与第 150 轮原型**完全一致**（同一量化/折算/mma 布局），只改搬运时序。
template <int HD, int BM, int BN>
__global__ void __launch_bounds__(THREADS, 2)
fp8_kvowner_dkv_pipe_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
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
  constexpr int NWM = 2, NWAR = 2;
  constexpr int NTW = NWAR * 64;
  constexpr int GM1 = BM / NWM, GN1 = BN / NWAR;
  constexpr int MTM = GM1 / 16, MTN = GN1 / 8;
  constexpr int GM34 = BN / NWM, GN34 = NTW / NWAR;
  constexpr int MTM34 = GM34 / 16, NTM34 = GN34 / 8;
  constexpr int NTFOLD = BN / 32;
  constexpr int STAGES = 2;
  static_assert(HD == 128 && BM == 64 && BN == 32, "本原型固定 HD=128/BM=64/BN=32");

  constexpr int sz_Ks = BN * ASLD;
  constexpr int sz_Vs = BN * ASLD;
  constexpr int sz_Q = BM * ASLD;
  constexpr int sz_Qp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_Ap = BN * QTS;
  constexpr int sz_Ps = BM * PSS * (int)sizeof(float);

  extern __shared__ __align__(16) char smem[];
  int off = 0;
  unsigned char* Ks = reinterpret_cast<unsigned char*>(smem + off); off += sz_Ks;
  unsigned char* Vs = reinterpret_cast<unsigned char*>(smem + off); off += sz_Vs;
  // staging：[stage][Qs|dOs|Qp|dOp]。用「基址 + 标量偏移」而非指针数组（运行期下标的
  // 指针数组会被推到 local memory，第 151 轮实测把 regs 顶到 243）。
  constexpr int STG = sz_Q + sz_Q + sz_Qp + sz_Qp;
  unsigned char* STG_base = reinterpret_cast<unsigned char*>(smem + off);
  off += STAGES * STG;
  unsigned char* Ap = reinterpret_cast<unsigned char*>(smem + off); off += sz_Ap;
  unsigned char* dS3 = reinterpret_cast<unsigned char*>(smem + off); off += sz_Ap;
  float* Ps = reinterpret_cast<float*>(smem + off); off += sz_Ps;
  float* Ss = reinterpret_cast<float*>(smem + off); off += sz_Ps;
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
  const int nd4 = HD / 4;

  // ---- 拥有的 K/V 行块：只从 global 读一次常驻 smem（同第 150 轮原型）----
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

  float dVacc[MTM34][NTM34][4];
  float dKacc[MTM34][NTM34][4];
#pragma unroll
  for (int i = 0; i < MTM34; ++i)
#pragma unroll
    for (int j = 0; j < NTM34; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) { dVacc[i][j][q] = 0.f; dKacc[i][j][q] = 0.f; }

  // ---- Q/dO 的异步 staging：16B cp.async 搬 Qs/dOs（行越界补 0）----
  const int nd16 = HD / 16;
  auto issue_qdo = [&](int m0, int s) {
    unsigned char* st = STG_base + s * STG;
    unsigned char* Qsd = st;
    unsigned char* Ood = st + sz_Q;
    for (int u = tid; u < (BM / 2) * nd16; u += THREADS) {
      const int rp = u / nd16, dc = (u % nd16) * 16;
      const int qa = m0 + rp * 2, qb = m0 + rp * 2 + 1;
      unsigned char* qd = Qsd + (rp * 2) * ASLD + dc;
      unsigned char* od = Ood + (rp * 2) * ASLD + dc;
      if (qa < S) {
        size_t idx = (((size_t)qa) * H + h) * HD + dc;
        cp_async16(qd, q8 + idx);
        cp_async16(od, do8 + idx);
      } else {
        cp_async16_z(qd, q8, 0);
        cp_async16_z(od, do8, 0);
      }
      if (qb < S) {
        size_t idx = (((size_t)qb) * H + h) * HD + dc;
        cp_async16(qd + ASLD, q8 + idx);
        cp_async16(od + ASLD, do8 + idx);
      } else {
        cp_async16_z(qd + ASLD, q8, 0);
        cp_async16_z(od + ASLD, do8, 0);
      }
    }
    cp_async_commit();
  };
  // 从 Qs/dOs 在 smem 上重建配对布局 Qp/dOp（同第 150 轮原型的逐字节交织写）。
  auto build_paired = [&](int s) {
    unsigned char* st = STG_base + s * STG;
    unsigned char* Qsd = st;
    unsigned char* Ood = st + sz_Q;
    uint16_t* Qpd = reinterpret_cast<uint16_t*>(st + sz_Q + sz_Q);
    uint16_t* Opd = reinterpret_cast<uint16_t*>(st + sz_Q + sz_Q + sz_Qp);
    for (int u = tid; u < (BM / 2) * nd4; u += THREADS) {
      const int rp = u / nd4, dq = (u % nd4) * 4;
      uint32_t q0 = *reinterpret_cast<uint32_t*>(Qsd + (rp * 2) * ASLD + dq);
      uint32_t q1 = *reinterpret_cast<uint32_t*>(Qsd + (rp * 2 + 1) * ASLD + dq);
      uint32_t o0 = *reinterpret_cast<uint32_t*>(Ood + (rp * 2) * ASLD + dq);
      uint32_t o1 = *reinterpret_cast<uint32_t*>(Ood + (rp * 2 + 1) * ASLD + dq);
      uint32_t* qpw = reinterpret_cast<uint32_t*>(Qpd + rp * PSLD + dq);
      qpw[0] = __byte_perm(q0, q1, 0x5140);
      qpw[1] = __byte_perm(q0, q1, 0x7362);
      uint32_t* opw = reinterpret_cast<uint32_t*>(Opd + rp * PSLD + dq);
      opw[0] = __byte_perm(o0, o1, 0x5140);
      opw[1] = __byte_perm(o0, o1, 0x7362);
    }
  };

  const int mstart = (j0 / BM) * BM;
  issue_qdo(mstart, 0);
  int cur = 0;
  for (int m0 = mstart; m0 < S; m0 += BM) {
    const int nxt = m0 + BM;
    const bool has_next = nxt < S;
    if (has_next) issue_qdo(nxt, cur ^ 1);
    if (has_next) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    build_paired(cur);
    if (tid < BM) {
      const int qi = m0 + tid;
      qs_s[tid] = (qi < S) ? qs[((size_t)qi) * H + h] : 1.f;
      dos_s[tid] = (qi < S) ? dos[((size_t)qi) * H + h] : 1.f;
    }
    __syncthreads();

    unsigned char* STGc = STG_base + cur * STG;
    unsigned char* Qc = STGc;
    unsigned char* Oc = STGc + sz_Q;
    uint16_t* Qpc = reinterpret_cast<uint16_t*>(STGc + sz_Q + sz_Q);
    uint16_t* Opc = reinterpret_cast<uint16_t*>(STGc + sz_Q + sz_Q + sz_Qp);

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

    float preg[MTM][MTN][4];
    {
      float acc[MTM][MTN][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<GM1, GN1, HD, E4E4>(Qc, ASLD, Ks, ASLD, acc, wr, wc, lane);
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
    {
      float acc[MTM][MTN][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<GM1, GN1, HD, E5E4>(Oc, ASLD, Vs, ASLD, acc, wr, wc, lane);
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
        if (sub4 == 0) { sA[j] = scA; sds3[j] = sc3; }
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

    {
      float acc[MTM34][NTM34][4];
#pragma unroll
      for (int i = 0; i < MTM34; ++i)
#pragma unroll
        for (int j = 0; j < NTM34; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bt<GM34, GN34, BM, E4E5>(Ap, QTS, Opc, PSLD, acc, wr, wc, lane, 0);
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
    {
      float acc[MTM34][NTM34][4];
#pragma unroll
      for (int i = 0; i < MTM34; ++i)
#pragma unroll
        for (int j = 0; j < NTM34; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bt<GM34, GN34, BM, E5E4>(dS3, QTS, Qpc, PSLD, acc, wr, wc, lane, 0);
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
    __syncthreads();
    cur ^= 1;
  }

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
// F7 主体第一步：KV-owner dK/dV 原型 + **persistent 调度**（对标 TE grid=132）
// =============================================================================
// 动机（ROADMAP 第一百五十一轮「下一步候选 ①」）：第 150/151 轮的 KV-owner 原型
// 每个 (KV 块, head) 一个 CTA ⇒ grid=(S/BN)·H，S4096 时 **2048 个 CTA**；而 TE 的
// `..._flash_bprop_wgmma_f8_..._64x64x128` 是 **132 个 persistent CTA**（1 CTA/SM）。
// 本变体把栅格改成 **1D persistent**：grid = min(总 tile 数, SM数×CTAS_PER_SM)，每个
// CTA 用 `tile += gridDim.x` 的 strided 循环处理多个 (h, j0) tile，**无栅格尾波量化**
// （原 2048/396=5.17 波、尾波 68/396）且 tile 顺序可控（按 head 连续 ⇒ L2 友好）。
//
// 数值口径与第 150/151 轮原型**完全一致**（每 tile 独立、每输出元素仅被其 owner 写
// 一次），故 `persist vs base` 应**逐位 = 0**（只换栅格映射）。
// DYN=false：第 152 轮的 1D static strided persistent（判定负：静态划分失去硬件
//            「块完成即回填」）；DYN=true（F7 第六步）：dynamic work-queue——每 CTA 用
//            全局 `atomicAdd(wq,1)` 领取下一个 tile。tile 编号 `h*nblk+jblk` ⇒ 小 j0
//            （重块，query 块数最多）先被领取 = LPT 调度，恢复动态回填。
template <int HD, int BM, int BN, bool DYN = false>
__global__ void __launch_bounds__(THREADS, 3)
fp8_kvowner_dkv_persist_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                               const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                               const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                               const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                               const float* __restrict__ lse, const float* __restrict__ delta,
                               float* __restrict__ dk, float* __restrict__ dv,
                               int* __restrict__ wq, int S, int H, int nblk,
                               float scale, int causal) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int PSLD = Cfg::PSLD;
  constexpr int QTS = Cfg::QTS;
  constexpr int PSS = Cfg::PSS;
  constexpr int NWM = 2, NWAR = 2;
  constexpr int NTW = NWAR * 64;
  constexpr int GM1 = BM / NWM, GN1 = BN / NWAR;
  constexpr int MTM = GM1 / 16, MTN = GN1 / 8;
  constexpr int GM34 = BN / NWM, GN34 = NTW / NWAR;
  constexpr int MTM34 = GM34 / 16, NTM34 = GN34 / 8;
  constexpr int NTFOLD = BN / 32;
  static_assert(HD == 128 && BM == 64 && BN == 32, "本原型固定 HD=128/BM=64/BN=32");

  constexpr int sz_Ks = BN * ASLD;
  constexpr int sz_Vs = BN * ASLD;
  constexpr int sz_Qs = BM * ASLD;
  constexpr int sz_dOs = BM * ASLD;
  constexpr int sz_Qp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_dOp = (BM / 2) * PSLD * (int)sizeof(uint16_t);
  constexpr int sz_Ap = BN * QTS;
  constexpr int sz_dS3 = BN * QTS;
  constexpr int sz_Ps = BM * PSS * (int)sizeof(float);
  constexpr int sz_Ss = BM * PSS * (int)sizeof(float);
  constexpr int sz_scales = (2 * BM + 4 * BN) * (int)sizeof(float);

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

  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / NWAR, wc = wid % NWAR;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int nd4 = HD / 4;
  const int total = nblk * H;

  // ---- 调度：tile = h*nblk + jblk（按 head 连续，L2 友好）----
  //   DYN=false：static 1D strided，tile = blockIdx.x + k*gridDim.x；
  //   DYN=true ：dynamic work-queue，tile = atomicAdd(wq,1)（重块先领 = LPT）。
  __shared__ int s_tile;
  int tile = DYN ? 0 : blockIdx.x;
  for (;;) {
    if (DYN) {
      if (tid == 0) s_tile = atomicAdd(wq, 1);
      __syncthreads();
      tile = s_tile;
      if (tile >= total) break;
    } else if (tile >= total) {
      break;
    }
    const int h = tile / nblk;
    const int j0 = (tile % nblk) * BN;

    // ---- 拥有的 K/V 行块：**只从 global 读一次**常驻 smem（越界补 0=E4M3(0)）----
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
  // F7 第八步修正：wgmma 经 **async proxy** 读 smem，generic 写的 Ks/Vs 必须先
  //   `fence.proxy.async` 才对 wgmma 可见（p154 遗漏 ⇒ 偶发 nondeterminism）。
  bulk_reduce_fence();
  __syncthreads();

    float dVacc[MTM34][NTM34][4];
    float dKacc[MTM34][NTM34][4];
#pragma unroll
    for (int i = 0; i < MTM34; ++i)
#pragma unroll
      for (int j = 0; j < NTM34; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) { dVacc[i][j][q] = 0.f; dKacc[i][j][q] = 0.f; }

    const int mstart = (j0 / BM) * BM;
    for (int m0 = mstart; m0 < S; m0 += BM) {
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
          if (sub4 == 0) { sA[j] = scA; sds3[j] = sc3; }
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
      __syncthreads();
    }

    // ---- 单一 owner：每 tile 循环外一次 plain store（无 atomic / 无 red）----
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
    if (!DYN) tile += gridDim.x;
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
  int pgrid = 0;  // persistent grid；0 = 自动（SM 数 × 3）
  std::string only;  // ncu 用：只跑一个 launch（dqonly / main3 / wgtma），其余跳过
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--pgrid=", 0) == 0) pgrid = atoi(a.c_str() + 8);
    else if (a.rfind("--only=", 0) == 0) only = a.substr(7);
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
  NpyF32 rdq;
  if (file_exists(dir + "/ref_dq.npy")) rdq = load_npy_f32(dir + "/ref_dq.npy");

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
  // F7 第四步：Q/dO staging 双缓冲（Qs/dOs/Qp/dOp ×2）。
  constexpr int pipe_smem = [] {
    return BN * Cfg::ASLD + BN * Cfg::ASLD +
           2 * (BM * Cfg::ASLD + BM * Cfg::ASLD + (BM / 2) * Cfg::PSLD * 2 +
                (BM / 2) * Cfg::PSLD * 2) +
           BN * Cfg::QTS + BN * Cfg::QTS + BM * Cfg::PSS * 4 + BM * Cfg::PSS * 4 +
           (2 * BM + 4 * BN) * 4;
  }();
  // F7 第七步：SW128 版（GEMM1/2 wgmma）。Ks/Vs/Qs/dOs 存 SW128（BN/BM × HD 字节）。
  // F7 第十步：+ Kp（dQ 的 B）+ dS2（dQ 的 A）+ sds2（每行 scale）。
  constexpr int wg_smem = [] {
    return BN * HD + BN * HD + BM * HD + BM * HD +
           (BM / 2) * Cfg::PSLD * 2 + (BM / 2) * Cfg::PSLD * 2 + BN * Cfg::QTS + BN * Cfg::QTS +
           (BN / 2) * Cfg::PSLD * 2 + BM * Cfg::DSS2 +
           BM * Cfg::PSS * 4 + BM * Cfg::PSS * 4 + (3 * BM + 4 * BN) * 4;
  }();
#if defined(FA_WGMMA) && defined(FA_TMA)
  // F7 第八步：Q/dO TMA 版在 wg_smem 之后加 64B 放 qbar/dbar（qbars 起点 16B 对齐）。
  constexpr int wg_tma_smem = ((wg_smem + 15) & ~15) + 64;
  // F7 第十二步：TMAR 版再加 [BM][HD] fp32 的 dQ staging（与 body 的 dQst 布局一致）。
  constexpr int wg_tmar_smem = wg_tma_smem + FA_KV_DQ_TMAR * (BM * HD * 4);
  // F7 第十三步（BN≥BM）：BN=64 的 smem 账——K/V/Ap/dS3/Kp/dS2/Ps/Ss 全随 BN 线性增长。
  //   按 Fp8Cfg<HD,64,64> 精确重算：111,360B ⇒ 2 CTA/SM（上限 116,224B）；3 CTA/SM（77,482B）放不下。
  using Cfg64 = Fp8Cfg<HD, 64, 64>;
  constexpr int BM64 = 64, BN64 = 64;
  constexpr int wg_smem64 = [] {
    return BN64 * HD + BN64 * HD + BM64 * HD + BM64 * HD +
           (BM64 / 2) * Cfg64::PSLD * 2 + (BM64 / 2) * Cfg64::PSLD * 2 + BN64 * Cfg64::QTS +
           BN64 * Cfg64::QTS + (BN64 / 2) * Cfg64::PSLD * 2 + BM64 * Cfg64::DSS2 +
           BM64 * Cfg64::PSS * 4 + BM64 * Cfg64::PSS * 4 + (3 * BM64 + 4 * BN64) * 4;
  }();
  constexpr int wg_tma_smem64 = ((wg_smem64 + 15) & ~15) + 64;
  // F7 第十四步：col-owner（RCOL=2）的 smem —— 只多 RCOL 份 K/V/Kp + 加宽的 scales。
  constexpr int RCOL = 2;
  constexpr int wg_col_smem = [] {
    return RCOL * (BN * HD + BN * HD + (BN / 2) * Cfg::PSLD * 2) + BM * HD + BM * HD +
           (BM / 2) * Cfg::PSLD * 2 + (BM / 2) * Cfg::PSLD * 2 + BN * Cfg::QTS + BN * Cfg::QTS +
           BM * Cfg::DSS2 + BM * Cfg::PSS * 4 + BM * Cfg::PSS * 4 + (3 * BM + 2 * BN + 2 * RCOL * BN) * 4;
  }();
  constexpr int wg_col_smem_tma = ((wg_col_smem + 15) & ~15) + 64;
#endif

  const int lse_smem = Cfg::lse_smem_bytes;

  printf("=== F7：fp8 mma KV-owner dK/dV 原型（base/pipe/persistent/**dynamic-wq**/**wgmma** A/B） ===\n");
  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d Hkv=%d D=%d causal=%d scale=%.6f\n", B, S, H, Hkv, D, (int)causal,
         scale);
  printf("grid = (%d, %d); kvowner smem = %d B (%.1f KB); pipe smem = %d B (%.1f KB); wgmma smem = %d B (%.1f KB); lse smem = %d B\n",
         (S + BN - 1) / BN, H, kv_smem, kv_smem / 1024.0, pipe_smem, pipe_smem / 1024.0,
         wg_smem, wg_smem / 1024.0, lse_smem);
  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos, *d_delta, *d_lse, *d_dk, *d_dv, *d_dq;
  int* d_wq = nullptr;  // F7 第六步：dynamic work-queue 计数器（global）
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
  CUDA_CHECK(cudaMalloc(&d_dq, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_wq, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), nq * 4, cudaMemcpyHostToDevice));

  const int d_rows = (int)rows_q;
  const int d_blocks = (d_rows + (THREADS / 32) - 1) / (THREADS / 32);
  dim3 lg((S + LBM - 1) / LBM, H, B);
  dim3 kg((S + BN - 1) / BN, H);

  // F7 主体：persistent 栅格。grid = min(总 tile 数, SM 数 × 3 CTA/SM)；--pgrid 可覆盖。
  int nsm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int nblk = (S + BN - 1) / BN;
  const int total_tiles = nblk * H;
  if (pgrid <= 0) pgrid = nsm * 3;
  const int pgrid_use = std::min(pgrid, total_tiles);
  printf("[F7 persist] SM=%d  total_tiles=%d  pgrid=%d（base 栅格 = %d CTA）\n", nsm, total_tiles,
         pgrid_use, total_tiles);

  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<HD>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, lse_smem));
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_kernel<HD, BM, BN>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kv_smem));
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_pipe_kernel<HD, BM, BN>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, pipe_smem));
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_persist_kernel<HD, BM, BN, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kv_smem));
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_persist_kernel<HD, BM, BN, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kv_smem));
#ifdef FA_WGMMA
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_wgmma_kernel<HD, BM, BN>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, wg_smem));
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_wgmma_tma_kernel<HD, BM, BN>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, wg_tma_smem));
  // F7 第十三步：BN=64 实例（BN≥BM）。
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_wgmma_tma_kernel<HD, BM64, BN64>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, wg_tma_smem64));
  // F7 第十四步：col-owner 实例（RCOL=2）。
  CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_col_tma_kernel<HD, BM, BN, RCOL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, wg_col_smem_tma));
  if (FA_KV_DQ_TMAR) {
    CUDA_CHECK(cudaFuncSetAttribute(fp8_kvowner_dkv_wgmma_tma_r_kernel<HD, BM, BN>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, wg_tmar_smem));
  }
#endif

  auto preprocess = [&]() {
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_q_f, d_q8, d_qs, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_k_f, d_k8, d_ks, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_v_f, d_v8, d_vs, D, 0);
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_do_f, d_do8, d_dos, D, 1);
    lse_mma_kernel<HD><<<lg, THREADS, lse_smem>>>(d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                                  (int)causal);
    delta_warp_kernel<HD><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
  };
  auto launch_base = [&]() {
    fp8_kvowner_dkv_kernel<HD, BM, BN><<<kg, THREADS, kv_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, S, H, scale,
        (int)causal);
  };
  auto launch_pipe = [&]() {
    fp8_kvowner_dkv_pipe_kernel<HD, BM, BN><<<kg, THREADS, pipe_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, S, H, scale,
        (int)causal);
  };
  auto launch_persist = [&]() {
    fp8_kvowner_dkv_persist_kernel<HD, BM, BN, false><<<pgrid_use, THREADS, kv_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, nullptr, S, H,
        nblk, scale, (int)causal);
  };
  // F7 第六步：dynamic work-queue 调度（每次 launch 前把计数器清零）。
  auto launch_dyn = [&]() {
    CUDA_CHECK(cudaMemsetAsync(d_wq, 0, sizeof(int)));
    fp8_kvowner_dkv_persist_kernel<HD, BM, BN, true><<<pgrid_use, THREADS, kv_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, d_wq, S, H,
        nblk, scale, (int)causal);
  };
#ifdef FA_WGMMA
  // F7 第七步：SW128 + GEMM1/2 wgmma（非 persistent，base 栅格，便于与 base A/B）。
  auto launch_wg = [&]() {
    fp8_kvowner_dkv_wgmma_kernel<HD, BM, BN><<<kg, THREADS, wg_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, d_dq, S, H,
        scale, (int)causal);
  };
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
  // F7 第八步：Q/dO 的 4D-TMA staging（对标 TE 数据通路）。描述符一次建好供所有 (h,b) CTA 用。
  CUtensorMap qmap_kv = make_kvowner_qd_map(d_q8, H, S, D, B, BM);
  CUtensorMap dmap_kv = make_kvowner_qd_map(d_do8, H, S, D, B, BM);
  auto launch_wgtma = [&]() {
    fp8_kvowner_dkv_wgmma_tma_kernel<HD, BM, BN><<<kg, THREADS, wg_tma_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, d_dq, S, H,
        scale, (int)causal, qmap_kv, dmap_kv);
  };
  // F7 第十三步（BN≥BM）：BN=64 —— 每 CTA 拥有 64 行 KV（=BM），dQ 的跨 CTA 贡献数从 S/32 减半到
  //   S/64、Q/dO 的跨 CTA 读放大也减半。grid.x 减半（S/64）。代价：smem 111KB ⇒ 2 CTA/SM。
  const dim3 kg64((S + BN64 - 1) / BN64, H);
  auto launch_wgtma64 = [&]() {
    fp8_kvowner_dkv_wgmma_tma_kernel<HD, BM64, BN64><<<kg64, THREADS, wg_tma_smem64>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, d_dq, S, H,
        scale, (int)causal, qmap_kv, dmap_kv);
  };
  // F7 第十四步：col-owner（RCOL=2）——每 CTA 拥有 2 个连续 KV 块，dK/dV 本地累加（red=0），
  //   dQ 跨列先本地累加再每 m 一次原子（贡献数 ÷ 2）。grid.x = ceil(S/(RCOL·BN))。
  const dim3 kg_col((S + RCOL * BN - 1) / (RCOL * BN), H);
  auto launch_col = [&]() {
    fp8_kvowner_dkv_col_tma_kernel<HD, BM, BN, RCOL><<<kg_col, THREADS, wg_col_smem_tma>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, d_dq, S, H,
        scale, (int)causal, qmap_kv, dmap_kv);
  };
  // F7 第十二步：dQ 归约换 **4D-TMA tensor store-reduce**（对标 TE `UTMAREDG.4D.ADD`）。
  // box 内维 = 64 列 fp32 = 256B（TMA box 上限）；body 沿 D 拆 2 个 chunk。
  CUtensorMap dqmap_kv = make_kvowner_dq_map_f32(d_dq, H, S, D, B, 64, BM);
  auto launch_wgtmar = [&]() {
    fp8_kvowner_dkv_wgmma_tma_r_kernel<HD, BM, BN><<<kg, THREADS, wg_tmar_smem>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_lse, d_delta, d_dk, d_dv, d_dq, S, H,
        scale, (int)causal, qmap_kv, dmap_kv, dqmap_kv);
  };
  // ---- F7 第十一步：option (a)「两 kernel」——KV-owner 出 dK/dV（red=0）＋ Q-owner dQ-only pass。
  //   复用生产 `fa_bwd_fp8_mma_kvtma_kernel`（Q/K/V/dO 全 4D-TMA）的 `DQONLY=true` 实例：
  //   保留 GEMM1/2/5 + dS2 fold，跳过 Ap/dS3 fold + GEMM3/4(dV/dK) + epilogue；dQ 在 `kRegDq`
  //   下寄存器 owned、DQONLY 用 **plain store** 写出 ⇒ 无跨 CTA `red`。ksplit=1、Q-owner 栅格。
  CUtensorMap kmap_dq = make_kvowner_qd_map(d_k8, Hkv, S, D, B, BN);
  CUtensorMap vmap_dq = make_kvowner_qd_map(d_v8, Hkv, S, D, B, BN);
  constexpr int kDqSmem = Fp8Cfg<HD, BM, BN>::smem_bytes_wgmma_kvtma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, true, true, true, true, false, false, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kDqSmem));
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, true, true, true, true, false, false, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kDqSmem));
  const dim3 dq_grid((S + BM - 1) / BM, H, B);
  auto launch_dqonly = [&]() {
    fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, true, true, true, true, false, false, true>
        <<<dq_grid, THREADS, kDqSmem>>>(
            qmap_kv, dmap_kv, kmap_dq, vmap_dq, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq, d_dk, d_dv, S, H, Hkv, scale, (int)causal, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr);
  };
  // 同 binary 的三梯度基线（生产默认 kvtma，DQONLY=false，含 dK/dV 的跨 CTA red）。
  auto launch_main3 = [&]() {
    fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, true, true, true, true, false, false, false>
        <<<dq_grid, THREADS, kDqSmem>>>(
            qmap_kv, dmap_kv, kmap_dq, vmap_dq, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq, d_dk, d_dv, S, H, Hkv, scale, (int)causal, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr);
  };
#endif
  auto run = [&]() { preprocess(); launch_base(); };
  run();
  CUDA_CHECK(cudaDeviceSynchronize());

  // ncu 专用：只跑一个 launch（预热 1 + iters 次），便于按 grid/kernel 定位。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (!only.empty()) {
    if (only == "dqonly") CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
    else { CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4)); CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4)); CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4)); }
    auto fn = [&]() {
      if (only == "dqonly") launch_dqonly();
      else if (only == "main3") launch_main3();
      else if (only == "wgtma") launch_wgtma();
      else if (only == "wgtma64") launch_wgtma64();
      else if (only == "wgtmar") launch_wgtmar();
      else if (only == "col2") launch_col();
    };
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[only=%s] done (iters=%d)\n", only.c_str(), iters);
    return 0;
  }
#endif

  std::vector<float> h_dk(nkv), h_dv(nkv), p_dk(nkv), p_dv(nkv);
  CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));

  // F7 第四步：跑 pipe 变体并落盘（沿用同一 preprocess 输出）。
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  launch_pipe();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(p_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(p_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));

  DiffStat sk = diff_stat(h_dk, rdk.data);
  DiffStat sv = diff_stat(h_dv, rdv.data);
  printf("\n[对拍] ours(KV-owner mma base) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", sk.max_abs, sk.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", sv.max_abs, sv.max_rel);
  DiffStat pk = diff_stat(p_dk, rdk.data);
  DiffStat pv = diff_stat(p_dv, rdv.data);
  printf("[对拍] ours(KV-owner mma **pipe**) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", pk.max_abs, pk.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", pv.max_abs, pv.max_rel);
  DiffStat bpk = diff_stat(p_dk, h_dk), bpv = diff_stat(p_dv, h_dv);
  printf("[对拍] pipe vs base: dk max_abs=%.4e  dv max_abs=%.4e（应=0，仅搬运时序）\n",
         bpk.max_abs, bpv.max_abs);

  // F7 主体：persistent 变体（沿用同一 preprocess 输出）。
  std::vector<float> s_dk(nkv), s_dv(nkv);
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  launch_persist();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(s_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(s_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  DiffStat rk = diff_stat(s_dk, rdk.data);
  DiffStat rv = diff_stat(s_dv, rdv.data);
  printf("[对拍] ours(KV-owner mma **persistent**) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", rk.max_abs, rk.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", rv.max_abs, rv.max_rel);
  DiffStat cpk = diff_stat(s_dk, h_dk), cpv = diff_stat(s_dv, h_dv);
  printf("[对拍] persistent vs base: dk max_abs=%.4e  dv max_abs=%.4e（应=0，仅换栅格映射）\n",
         cpk.max_abs, cpv.max_abs);

  // F7 第六步：dynamic work-queue（沿用同一 preprocess 输出）。
  std::vector<float> y_dk(nkv), y_dv(nkv);
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  launch_dyn();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(y_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  DiffStat dk_r = diff_stat(y_dk, rdk.data);
  DiffStat dv_r = diff_stat(y_dv, rdv.data);
  printf("[对拍] ours(KV-owner mma **dynamic wq**) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", dk_r.max_abs, dk_r.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", dv_r.max_abs, dv_r.max_rel);
  DiffStat dk_b = diff_stat(y_dk, h_dk), dv_b = diff_stat(y_dv, h_dv);
  printf("[对拍] dynamic vs base: dk max_abs=%.4e  dv max_abs=%.4e（应=0，仅换栅格映射）\n",
         dk_b.max_abs, dv_b.max_abs);

#ifdef FA_WGMMA
  // F7 第七步：wgmma 版（SW128 + GEMM1/2 wgmma，同一量化口径）。
  // F7 第十步：同时算 dQ（FA_KV_DQ）。
  std::vector<float> w_dk(nkv), w_dv(nkv), w_dq(nq);
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
  launch_wg();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(w_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(w_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(w_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  DiffStat wk_r = diff_stat(w_dk, rdk.data);
  DiffStat wv_r = diff_stat(w_dv, rdv.data);
  printf("[对拍] ours(KV-owner mma **wgmma**) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", wk_r.max_abs, wk_r.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", wv_r.max_abs, wv_r.max_rel);
#if FA_KV_DQ
  if (file_exists(dir + "/ref_dq.npy")) {
    DiffStat wq_r = diff_stat(w_dq, rdq.data);
    printf("  dq  max_abs=%.4e  max_rel=%.4e\n", wq_r.max_abs, wq_r.max_rel);
  }
#endif
  DiffStat wk_b = diff_stat(w_dk, h_dk), wv_b = diff_stat(w_dv, h_dv);
  printf("[对拍] wgmma vs base: dk max_abs=%.4e  dv max_abs=%.4e（同口径，应≈0）\n",
         wk_b.max_abs, wv_b.max_abs);
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
  // F7 第八步：Q/dO 4D-TMA 版（其余数据通路与 wgmma 版逐字一致）。
  std::vector<float> q_dk(nkv), q_dv(nkv), q_dq(nq);
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
  launch_wgtma();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(q_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(q_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(q_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  DiffStat qk_r = diff_stat(q_dk, rdk.data);
  DiffStat qv_r = diff_stat(q_dv, rdv.data);
  printf("[对拍] ours(KV-owner mma **wgmma+Q/dO-TMA**) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", qk_r.max_abs, qk_r.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", qv_r.max_abs, qv_r.max_rel);
#if FA_KV_DQ
  if (file_exists(dir + "/ref_dq.npy")) {
    DiffStat qq_r = diff_stat(q_dq, rdq.data);
    printf("  dq  max_abs=%.4e  max_rel=%.4e\n", qq_r.max_abs, qq_r.max_rel);
  }
#endif
  DiffStat qk_b = diff_stat(q_dk, w_dk), qv_b = diff_stat(q_dv, w_dv);
  printf("[对拍] wgmma+TMA vs wgmma: dk max_abs=%.4e  dv max_abs=%.4e（应=0，仅搬运通路）\n",
         qk_b.max_abs, qv_b.max_abs);
  // F7 第十三步（BN≥BM）：BN=64 三梯度 vs ref / vs BN=32 —— 验证「工作划分减半」不改数学口径。
  std::vector<float> b64_dk(nkv), b64_dv(nkv), b64_dq(nq);
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
  launch_wgtma64();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(b64_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b64_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b64_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  DiffStat xk_r = diff_stat(b64_dk, rdk.data), xv_r = diff_stat(b64_dv, rdv.data);
  printf("[对拍] ours(KV-owner **BN=64** 三梯度) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", xk_r.max_abs, xk_r.max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", xv_r.max_abs, xv_r.max_rel);
  if (file_exists(dir + "/ref_dq.npy")) {
    DiffStat xq_r = diff_stat(b64_dq, rdq.data);
    printf("  dq  max_abs=%.4e  max_rel=%.4e\n", xq_r.max_abs, xq_r.max_rel);
  }
  printf("[对拍] BN=64 vs BN=32(wgmma+TMA): dk=%.4e dv=%.4e dq=%.4e（fp8 噪声，仅跨 CTA 加序/分块变）\n",
         diff_stat(b64_dk, q_dk).max_abs, diff_stat(b64_dv, q_dv).max_abs,
         diff_stat(b64_dq, q_dq).max_abs);
  // F7 第十四步：col-owner（RCOL=2）三梯度 vs ref / vs wgmma+TMA（BN=32 单块）。
  std::vector<float> c2_dk(nkv), c2_dv(nkv), c2_dq(nq);
  CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
  launch_col();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c2_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(c2_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(c2_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  printf("[对拍] ours(KV-owner **col-owner RCOL=2** 三梯度) vs fp32 ref:\n");
  printf("  dk  max_abs=%.4e  max_rel=%.4e\n", diff_stat(c2_dk, rdk.data).max_abs,
         diff_stat(c2_dk, rdk.data).max_rel);
  printf("  dv  max_abs=%.4e  max_rel=%.4e\n", diff_stat(c2_dv, rdv.data).max_abs,
         diff_stat(c2_dv, rdv.data).max_rel);
  if (file_exists(dir + "/ref_dq.npy")) {
    DiffStat cq = diff_stat(c2_dq, rdq.data);
    printf("  dq  max_abs=%.4e  max_rel=%.4e\n", cq.max_abs, cq.max_rel);
  }
  printf("[对拍] col-owner RCOL=2 vs wgmma+TMA(BN=32): dk=%.4e dv=%.4e dq=%.4e（fp8 噪声，仅工作划分变）\n",
         diff_stat(c2_dk, q_dk).max_abs, diff_stat(c2_dv, q_dv).max_abs,
         diff_stat(c2_dq, q_dq).max_abs);
#if FA_KV_DQ
  // F7 第十二步：dQ 走 4D-TMA tensor store-reduce（对标 TE `UTMAREDG.4D.ADD`）。
  {
    std::vector<float> r_dk(nkv), r_dv(nkv), r_dq(nq);
    CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
    launch_wgtmar();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    printf("[对拍] ours(KV-owner mma **wgmma+Q/dO-TMA + dQ 4D-TMA-reduce**) vs fp32 ref:\n");
    printf("  dk  max_abs=%.4e  max_rel=%.4e\n", diff_stat(r_dk, rdk.data).max_abs,
           diff_stat(r_dk, rdk.data).max_rel);
    printf("  dv  max_abs=%.4e  max_rel=%.4e\n", diff_stat(r_dv, rdv.data).max_abs,
           diff_stat(r_dv, rdv.data).max_rel);
    if (file_exists(dir + "/ref_dq.npy")) {
      DiffStat sq = diff_stat(r_dq, rdq.data);
      printf("  dq  max_abs=%.4e  max_rel=%.4e\n", sq.max_abs, sq.max_rel);
    }
    printf("[对拍] dQ 4D-TMA-reduce vs atomic(red_add2): dq max_abs=%.4e（应≈0，仅归约加序）\n",
           diff_stat(r_dq, q_dq).max_abs);
    printf("[对拍] 4D-TMA-reduce vs wgmma+TMA: dk max_abs=%.4e  dv max_abs=%.4e（应=0）\n",
           diff_stat(r_dk, q_dk).max_abs, diff_stat(r_dv, q_dv).max_abs);
  }
#endif
#endif

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

#if defined(FA_WGMMA) && defined(FA_TMA)
  // F7 第十一步：Q-owner dQ-only pass 与同 binary 三梯度基线（kvtma）。
  {
    std::vector<float> dq_dq(nq), m3_dq(nq), m3_dk(nkv), m3_dv(nkv);
    CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    launch_dqonly();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dq_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    if (file_exists(dir + "/ref_dq.npy")) {
      DiffStat s = diff_stat(dq_dq, rdq.data);
      printf("[对拍] ours(Q-owner **dQ-only** pass) vs fp32 ref: dq max_abs=%.4e max_rel=%.4e\n",
             s.max_abs, s.max_rel);
    }
    CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    launch_main3();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(m3_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m3_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m3_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[对拍] 同 binary 三梯度基线(kvtma, DQONLY=0):");
    if (file_exists(dir + "/ref_dq.npy")) {
      DiffStat s = diff_stat(m3_dq, rdq.data);
      printf(" dq=%.4e", s.max_abs);
    }
    printf(" dk=%.4e dv=%.4e\n", diff_stat(m3_dk, rdk.data).max_abs,
           diff_stat(m3_dv, rdv.data).max_abs);
    printf("[对拍] dQ-only vs 三梯度基线 dq: max_abs=%.4e（应≈0，dQ 路径逐字一致）\n",
           diff_stat(dq_dq, m3_dq).max_abs);
  }
#endif

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
  float t_base = bench(launch_base, iters);
  float t_pipe = bench(launch_pipe, iters);
  float t_persist = bench(launch_persist, iters);
  float t_dyn = bench(launch_dyn, iters);
#ifdef FA_WGMMA
  float t_wg = bench(launch_wg, iters);
#else
  float t_wg = 0.f;
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
  float t_wgtma = bench(launch_wgtma, iters);
  float t_wgtma64 = bench(launch_wgtma64, iters);
  float t_col = bench(launch_col, iters);
#endif
#if defined(FA_WGMMA) && defined(FA_TMA) && FA_KV_DQ && FA_KV_DQ_TMAR
  float t_wgtmar = bench(launch_wgtmar, iters);
#endif
#if !(defined(FA_WGMMA) && defined(FA_TMA))
  float t_wgtma = 0.f;
#endif
#if !(defined(FA_WGMMA) && defined(FA_TMA))
  float t_wgtma64 = 0.f;
#endif
#if !(defined(FA_WGMMA) && defined(FA_TMA) && FA_KV_DQ && FA_KV_DQ_TMAR)
  float t_wgtmar = 0.f;
#endif
  float t_pre = bench(preprocess, iters);
  double flops = 2.0 * (double)S * S * H * (double)HD * (causal ? 0.5 : 1.0) * 3.0;
  printf("\n[计时] main(dK/dV) base = %.4f ms (%.2f TF) | **pipe** = %.4f ms (%.2f TF) | A/B = %.3f×\n",
         t_base, flops / t_base / 1e9, t_pipe, flops / t_pipe / 1e9, t_base / t_pipe);
  printf("[计时] main(dK/dV) **persistent**(pgrid=%d) = %.4f ms (%.2f TF) | persist/base = %.3f×\n",
         pgrid_use, t_persist, flops / t_persist / 1e9, t_base / t_persist);
  printf("[计时] main(dK/dV) **dynamic wq**(pgrid=%d) = %.4f ms (%.2f TF) | dyn/base = %.3f× | dyn/persist = %.3f×\n",
         pgrid_use, t_dyn, flops / t_dyn / 1e9, t_base / t_dyn, t_persist / t_dyn);
  printf("[计时] preprocess = %.4f ms | total base = %.4f | total pipe = %.4f | total persist = %.4f | total dyn = %.4f\n",
         t_pre, t_base + t_pre, t_pipe + t_pre, t_persist + t_pre, t_dyn + t_pre);
  printf("[计时] 说明：base/pipe/persist/dyn **仅 dK/dV**（2 GEMM）而 wgmma 档为**三梯度**（dK/dV/dQ，3 GEMM）；\n");
  printf("[计时]   `flops` 全按 3 GEMM 口径 ⇒ **base 档 TF 被高估 1.5×**，同档内 A/B 比值仍有效。\n");
  printf("[计时]   与 FP8 峰值 1978.8 TF 的占比 = %.3f%% (base) / %.3f%% (pipe) / %.3f%% (persist) / %.3f%% (dyn)\n",
         100.0 * flops / t_base / 1e9 / 1978.8, 100.0 * flops / t_pipe / 1e9 / 1978.8,
         100.0 * flops / t_persist / 1e9 / 1978.8, 100.0 * flops / t_dyn / 1e9 / 1978.8);
#ifdef FA_WGMMA
  printf("[计时] main(dK/dV) **wgmma**(SW128+GEMM1/2 async) = %.4f ms (%.2f TF) | wg/base = %.3f× | wg/pipe = %.3f× | wg/dyn = %.3f×\n",
         t_wg, flops / t_wg / 1e9, t_base / t_wg, t_pipe / t_wg, t_dyn / t_wg);
  printf("[计时] total wgmma = %.4f ms | 峰值占比 = %.3f%%\n", t_wg + t_pre,
          100.0 * flops / t_wg / 1e9 / 1978.8);
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
  printf("[计时] main(dK/dV/dQ 三梯度) **wgmma+Q/dO-TMA** = %.4f ms (%.2f TF) | tma/wg = %.3f× | tma/base = %.3f×\n",
          t_wgtma, flops / t_wgtma / 1e9, t_wg / t_wgtma, t_base / t_wgtma);
  printf("[计时] main(三梯度) **BN=64 (BN≥BM)** = %.4f ms (%.2f TF) | BN64/BN32 = %.3f× | smem=%d B (%.1f KB)\n",
          t_wgtma64, flops / t_wgtma64 / 1e9, t_wgtma / t_wgtma64, wg_tma_smem64,
          wg_tma_smem64 / 1024.0);
  printf("[计时] total BN=64 = %.4f ms | 峰值占比 = %.3f%%\n", t_wgtma64 + t_pre,
          100.0 * flops / t_wgtma64 / 1e9 / 1978.8);
  printf("[计时] main(三梯度) **col-owner RCOL=2** = %.4f ms (%.2f TF) | col/BN32 = %.3f× | smem=%d B (%.1f KB)\n",
         t_col, flops / t_col / 1e9, t_wgtma / t_col, wg_col_smem_tma,
         wg_col_smem_tma / 1024.0);
  printf("[计时] total col-owner RCOL=2 = %.4f ms | 峰值占比 = %.3f%%\n", t_col + t_pre,
          100.0 * flops / t_col / 1e9 / 1978.8);
  printf("[计时] total wgmma+TMA = %.4f ms | 峰值占比 = %.3f%%\n", t_wgtma + t_pre,
          100.0 * flops / t_wgtma / 1e9 / 1978.8);
#if FA_KV_DQ_TMAR
  printf("[计时] main(三梯度) **dQ 4D-TMA-reduce** = %.4f ms (%.2f TF) | tmar/wgtma = %.3f×\n",
         t_wgtmar, flops / t_wgtmar / 1e9, t_wgtma / t_wgtmar);
#endif
  // F7 第十一步：两 kernel 判决（KV-owner dK/dV[red=0] + Q-owner dQ-only[red=0] vs 单 kernel 三梯度）。
  float t_dqonly = bench(launch_dqonly, iters);
  float t_main3 = bench(launch_main3, iters);
  printf("[计时] Q-owner **dQ-only** pass = %.4f ms | 同 binary 三梯度基线(kvtma) = %.4f ms\n",
         t_dqonly, t_main3);
  printf("[计时] F7 option(a) 两 kernel = KV-owner dK/dV %.4f + dQ-only %.4f = %.4f ms | vs 单 kernel 三梯度 %.3f×\n",
         t_wgtma, t_dqonly, t_wgtma + t_dqonly, t_main3 / (t_wgtma + t_dqonly));
  printf("[计时]   注：KV-owner 档须以 -DFA_KV_DQ=0 构建方为**纯 dK/dV**（当前 FA_KV_DQ=%d）。\n",
         (int)FA_KV_DQ);
#endif

  CUDA_CHECK(cudaFree(d_q_f)); CUDA_CHECK(cudaFree(d_k_f)); CUDA_CHECK(cudaFree(d_v_f));
  CUDA_CHECK(cudaFree(d_do_f)); CUDA_CHECK(cudaFree(d_o_f));
  CUDA_CHECK(cudaFree(d_q8)); CUDA_CHECK(cudaFree(d_k8)); CUDA_CHECK(cudaFree(d_v8));
  CUDA_CHECK(cudaFree(d_do8));
  CUDA_CHECK(cudaFree(d_qs)); CUDA_CHECK(cudaFree(d_ks)); CUDA_CHECK(cudaFree(d_vs));
  CUDA_CHECK(cudaFree(d_dos)); CUDA_CHECK(cudaFree(d_delta)); CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_dk)); CUDA_CHECK(cudaFree(d_dv)); CUDA_CHECK(cudaFree(d_dq));
  CUDA_CHECK(cudaFree(d_wq));
  return 0;
}
