// 42 MLA 吸收式 decode：wgmma QK + mma/ldmatrix.trans PV（消 V 转置税）+ K 双缓冲
//
// 42 主文件把 QK/PV 都做成 wgmma SS，但 wgmma 的 B 只认 K-major，PV 必须把
// c_kv 转置成 V^T[DV,KT]——每 tile 4096 次「8 条跨 DC=512 的标量 global 读 +
// 一次 SW128 16B 写」，实测把 64×512×64 的 PV 拖成瓶颈（去掉这段后 183→282
// TFLOPS）。本文件改成 **混合**：
//   · QK：wgmma.m64n{k}k16 SS（B=K tile，天然 K-major，无需转置）
//   · PV：mma.m16n8k16 + `ldmatrix.x4.trans` **直接从同一块 SW128 K tile 读 V**
//         （SW128 只是地址置换，每个 16B chunk 完好，ldmatrix 给地址即可）
// 于是：每个 KV tile 只从 global 读一次，且 V 无需转置。再叠 **K 双缓冲**
// （smem 恰好放得下 2 块 K：3×73.7KB + P 9KB + red 2KB = 227KB）把 K 预取
// 与 softmax/PV 重叠。
//
// 运行：ARCH="" scripts/run.sh 42-mla-decode-wgmma/mla_decode_hybrid.cu [B] [L] [which]
#include "../common/cuda_utils.cuh"
#include "../20-mla-wgmma-sw128/wgmma_sw128.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int DC = 512, DR = 64, DV = 512, DK = DC + DR, H = 128;
constexpr int BM = 64, KT = 64;
constexpr int SBO_K = (DK / 64) * 1024;

__device__ __forceinline__ int row0_of(int w_, int lane) { return 16 * (w_ & 3) + (lane >> 2); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

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

__device__ __forceinline__ void wgmma_m64n16k16(float (&d)[8], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7},\n"
      "%8, %9, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7])
      : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ void wgmma_m64n32k16(float (&d)[16], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "%16, %17, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(1));
}

// ===========================================================================
// NWGT 个 warpgroup：各算 KT/NWGT 个 key 的 S（wgmma），跨 WG 合并 max/sum，
// 各写自己那段的 P（row-major），再各用 mma+ldmatrix.trans 算 DV/NWGT 列的 PV。
//   Q[BM,DK] SW128 · K[2][KT,DK] SW128 双缓冲 · P[BM][KT+8] row-major
// ===========================================================================
template <int NWGT>
__global__ void __launch_bounds__(NWGT* 128) mla_dec_hy_kernel(const bf16* __restrict__ qa,
                                                               const bf16* __restrict__ qr,
                                                               const bf16* __restrict__ ckv,
                                                               const bf16* __restrict__ kr,
                                                               float* __restrict__ out, int B,
                                                               int L) {
  constexpr int T = NWGT * 128;
  constexpr int KTH = KT / NWGT;
  constexpr int DVW = DV / NWGT;
  constexpr int KTP = KT + 8;

  extern __shared__ __align__(1024) char smem[];
  char* qs = smem;                                      // [BM][DK]  SW128
  char* ks0 = qs + (size_t)BM * DK * 2;                 // [KT][DK]  SW128
  char* ks1 = ks0 + (size_t)KT * DK * 2;                // 第二块 K（双缓冲）
  bf16 (*ps)[KTP] = reinterpret_cast<bf16(*)[KTP]>(ks1 + (size_t)KT * DK * 2);
  float* red = reinterpret_cast<float*>(reinterpret_cast<char*>(ps) + (size_t)BM * KTP * 2);

  const int head0 = blockIdx.x * BM;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wg = tid >> 7;
  const int W = tid >> 5;
  const int r0 = row0_of(W, lane), r1 = r0 + 8;

  const char* ck_b = reinterpret_cast<const char*>(ckv) + (size_t)b * L * DC * 2;
  const char* kr_b = reinterpret_cast<const char*>(kr) + (size_t)b * L * DR * 2;

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

  constexpr int NS = KTH / 8;
  float S[NS * 4];
  float O[DVW / 8 * 4];
#pragma unroll
  for (int i = 0; i < NS * 4; ++i) S[i] = 0.f;
#pragma unroll
  for (int i = 0; i < DVW / 8 * 4; ++i) O[i] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const uint32_t qs_a = smem_u32(qs);
  const uint32_t ks_a[2] = {smem_u32(ks0), smem_u32(ks1)};
  const uint32_t ks_w[2] = {ks_a[0] + (uint32_t)(wg * (KTH / 8) * SBO_K),
                            ks_a[1] + (uint32_t)(wg * (KTH / 8) * SBO_K)};
  const int nbase = wg * KTH;

  // ---- 载入第 0 块 K（同步）----
  {
    char* ks = ks0;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int n = pass ? KT * DR : KT * DC;
      const int Wd = pass ? DR : DC;
      const char* src = pass ? kr_b : ck_b;
      const int kcol = pass ? DC : 0;
      for (int idx = tid; idx < n / 8; idx += T) {
        const int kv = idx / (Wd / 8), cu = idx % (Wd / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (kv < L) v = *reinterpret_cast<const uint4*>(src + (size_t)kv * Wd * 2 + cu * 16);
        sw128_store16(ks, kv, kcol + cu * 8, DK, v);
      }
    }
  }
  __syncthreads();

  int slot = 0;
  for (int k0 = 0; k0 < L; k0 += KT, slot ^= 1) {
    const bool has_next = (k0 + KT < L);
    const uint32_t cur = ks_a[slot];
    const uint32_t curw = ks_w[slot];
    // ---- QK（wgmma，拆键）----
#pragma unroll
    for (int i = 0; i < NS * 4; ++i) S[i] = 0.f;
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < DK / 16; ++s) {
      if constexpr (KTH == 16)
        wgmma_m64n16k16(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
                        make_desc_sw128(sw128_k16_addr(curw, s), SBO_K));
      else
        wgmma_m64n32k16(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
                        make_desc_sw128(sw128_k16_addr(curw, s), SBO_K));
    }
    wgmma_commit();
    wgmma_wait0();

    // ---- 预取下一块 K 到另一 buffer（QK 已读完 cur；与 softmax/PV 重叠）----
    if (has_next) {
      const uint32_t nxt = ks_a[slot ^ 1];
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        const int n = pass ? KT * DR : KT * DC;
        const int Wd = pass ? DR : DC;
        const char* src = pass ? kr_b : ck_b;
        const int kcol = pass ? DC : 0;
        for (int idx = tid; idx < n / 8; idx += T) {
          const int kv = idx / (Wd / 8), cu = idx % (Wd / 8);
          cp_async16(nxt + sw128_off(kv, kcol + cu * 8, DK),
                     src + (size_t)(k0 + KT + kv) * Wd * 2 + cu * 16);
        }
      }
      cp_commit();
    }

    if (k0 + KT > L) {
#pragma unroll
      for (int t = 0; t < NS; ++t) {
        const int c0 = k0 + nbase + t * 8 + (lane & 3) * 2;
        if (c0 >= L) { S[t * 4 + 0] = -INFINITY; S[t * 4 + 2] = -INFINITY; }
        if (c0 + 1 >= L) { S[t * 4 + 1] = -INFINITY; S[t * 4 + 3] = -INFINITY; }
      }
    }

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
    red[wg * BM + r0] = b0;
    red[wg * BM + r1] = b1;
    __syncthreads();
    float fm0 = -INFINITY, fm1 = -INFINITY;
#pragma unroll
    for (int e = 0; e < NWGT; ++e) {
      fm0 = fmaxf(fm0, red[e * BM + r0]);
      fm1 = fmaxf(fm1, red[e * BM + r1]);
    }
    const float nm0 = fmaxf(m0, fm0), nm1 = fmaxf(m1, fm1);
    const float al0 = __expf(m0 - nm0), al1 = __expf(m1 - nm1);

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
    red[(NWGT + wg) * BM + r0] = s0;
    red[(NWGT + wg) * BM + r1] = s1;
#pragma unroll
    for (int t = 0; t < NS; ++t) {
      const int col = nbase + t * 8 + (lane & 3) * 2;
      *reinterpret_cast<__nv_bfloat162*>(&ps[r0][col]) =
          __floats2bfloat162_rn(S[t * 4 + 0], S[t * 4 + 1]);
      *reinterpret_cast<__nv_bfloat162*>(&ps[r1][col]) =
          __floats2bfloat162_rn(S[t * 4 + 2], S[t * 4 + 3]);
    }
#pragma unroll
    for (int t = 0; t < DVW / 8; ++t) {
      O[t * 4 + 0] *= al0;
      O[t * 4 + 1] *= al0;
      O[t * 4 + 2] *= al1;
      O[t * 4 + 3] *= al1;
    }
    __syncthreads();
    {
      float fs0 = 0.f, fs1 = 0.f;
#pragma unroll
      for (int e = 0; e < NWGT; ++e) {
        fs0 += red[(NWGT + e) * BM + r0];
        fs1 += red[(NWGT + e) * BM + r1];
      }
      l0 = l0 * al0 + fs0;
      l1 = l1 * al1 + fs1;
    }
    m0 = nm0;
    m1 = nm1;

    // ---- PV：mma.m16n8k16 + ldmatrix（A=P row-major；B=V 从当前 SW128 K tile）----
    const int rrow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ccol = (lane >> 4) * 8;
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      uint32_t pa[4];
      ldmatrix_x4(smem_u32(&ps[16 * (W & 3) + (lane & 15)][c * 16 + (lane >> 4) * 8]), pa);
#pragma unroll
      for (int dn = 0; dn < DVW / 16; ++dn) {
        uint32_t d[4];
        const int keyrow = c * 16 + rrow;
        const int dv = wg * DVW + dn * 16 + ccol;
        ldmatrix_x4_trans(cur + (uint32_t)sw128_off(keyrow, dv, DK), d);
        mma16816(O + (dn * 2) * 4, pa, d);
        mma16816(O + (dn * 2 + 1) * 4, pa, d + 2);
      }
    }
    if (has_next) cp_wait0();
    __syncthreads();
  }

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

// host 参考
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

  std::printf("\nMLA decode HYBRID (wgmma QK + mma/ldmatrix.trans PV + K double-buffer): H=%d B=%d L=%d\n",
              H, B, L);

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
    std::printf("     [%-4s] max_abs_err=%.3e (ref~%.3f) %s\n", tag, err, base,
                err / std::max(base, 1e-6) < 5e-2 ? "OK" : "FAIL");
  };

  auto run = [&](const char* tag, int nwg, bool refs) {
    const size_t shm = (size_t)BM * DK * 2 + 2 * (size_t)KT * DK * 2 + (size_t)BM * (KT + 8) * 2 +
                       (size_t)2 * nwg * BM * 4;
    dim3 grid(div_up(H, BM), B);
    using Fn = void (*)(const bf16*, const bf16*, const bf16*, const bf16*, float*, int, int);
    Fn fn = (nwg == 4) ? (Fn)mla_dec_hy_kernel<4> : (Fn)mla_dec_hy_kernel<2>;
    int th = (nwg == 4) ? 512 : 256;
    CUDA_CHECK(cudaFuncSetAttribute((void*)fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    auto runc = [&] { fn<<<grid, th, shm>>>(qa, qr, ckv, kr, out, B, L); };
    runc();
    CUDA_CHECK_LAST();
    if (refs) check(tag);
    double t = bench_ms(runc, 3, 20);
    const double tfl = to_tflops(flops, t);
    const double gbps = bytes_bf16 * 1000.0 / t / 1e9;
    std::printf("%-5s hyb%d  %9.4f ms  %7.1f GB/s (%5.1f%% HBM)  %7.1f TFLOPS (%5.1f%%)\n", tag, nwg,
                t, gbps, 100.0 * gbps / 3350.0, tfl, 100.0 * tfl / 989.0);
  };

  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };
  if (want("n2")) run("n2", 2, true);
  if (want("n4")) run("n4", 4, true);
  if (want("all") || want("bf16")) { run("v2", 2, true); run("v4", 4, true); }
  if (want("prof")) run("p2", 2, false);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
