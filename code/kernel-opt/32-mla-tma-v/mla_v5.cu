// 20 MLA 极限冲刺（二）：SW128 swizzle + wgmma 的融合 MLA
//
// 与 16 篇 mla_wgmma.cu 的融合思路相同（online softmax 融进 QK epilogue、KV 常驻 smem、
// P 走 smem 供 PV 复用），但把 operands 的 smem 布局从无 swizzle 的 K-major INTERLEAVE
// 换成 **K-major + 128B swizzle（SW128）**：消掉 wgmma 读取操作数时的 store bank conflict
// 与低效寻址（16 篇实测 INTERLEAVE 版只有 84.6 TFLOPS）。
//
// 运行：ARCH=sm_90a scripts/run.sh 20-mla-wgmma-sw128/mla_wgmma_sw.cu [Sq] [H] [Sk]
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"

#include <cuda_bf16.h>
#include <cmath>

using bf16 = __nv_bfloat16;
constexpr int DC = 512, DR = 64, DV = 512, DK = DC + DR;

// K-major SW128：相邻 8 行组步长 = (K/64)*1024 字节
constexpr int SBO_K = (DK / 64) * 1024;  // 9216
constexpr int SBO_P = 1024;

constexpr int BM = 64, KT = 64, NWG = 2, DVW = DV / NWG;
constexpr int T = NWG * 128;

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

__global__ void __launch_bounds__(T) mla_pipe3_kernel(const bf16* __restrict__ qa,
                                                         const bf16* __restrict__ qr,
                                                         const bf16* __restrict__ ckv,
                                                         const bf16* __restrict__ kr,
                                                         float* __restrict__ out, int Sq, int Sk) {
  extern __shared__ __align__(1024) char smem[];
  char* qs = smem;                              // [BM][DK]  SW128
  char* ks = qs + (size_t)BM * DK * 2;          // [KT][DK]  SW128
  char* ps = ks + (size_t)KT * DK * 2;          // [BM][KT]  SW128
  char* vs = ps + (size_t)BM * KT * 2;          // [DV][KT]  SW128 (转置)

  const int h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wg = tid >> 7;
  const int W = tid >> 5;
  const int r0 = row0_of(W, lane), r1 = r0 + 8;

  // ---- 载入 Q = [c_kv | k_rope] ----
  for (int idx = tid; idx < BM * (DC / 8); idx += T) {
    const int m = idx / (DC / 8), cu = idx % (DC / 8);
    const int gq = q0 + m;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qa[((size_t)h * Sq + gq) * DC + cu * 8]);
    sw128_store16(qs, m, cu * 8, DK, v);
  }
  for (int idx = tid; idx < BM * (DR / 8); idx += T) {
    const int m = idx / (DR / 8), cu = idx % (DR / 8);
    const int gq = q0 + m;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qr[((size_t)h * Sq + gq) * DR + cu * 8]);
    sw128_store16(qs, m, DC + cu * 8, DK, v);
  }

  float S[32];
  float O[128];
#pragma unroll
  for (int i = 0; i < 32; ++i) S[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 128; ++i) O[i] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  __syncthreads();
  const uint32_t qs_a = smem_u32(qs), ks_a = smem_u32(ks), ps_a = smem_u32(ps), vs_a = smem_u32(vs);

  // ---- prologue：载入第 0 块 K/V ----
  for (int idx = tid; idx < KT * (DC / 8); idx += T) {
    const int kv = idx / (DC / 8), cu = idx % (DC / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&ckv[(size_t)kv * DC + cu * 8]);
    sw128_store16(ks, kv, cu * 8, DK, v);
  }
  for (int idx = tid; idx < KT * (DR / 8); idx += T) {
    const int kv = idx / (DR / 8), cu = idx % (DR / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&kr[(size_t)kv * DR + cu * 8]);
    sw128_store16(ks, kv, DC + cu * 8, DK, v);
  }
  for (int i = tid; i < (KT / 8) * DV; i += T) {
    const int kb = i / DV, dv = i % DV;
    uint4 v;
    unsigned short* sp = reinterpret_cast<unsigned short*>(&v);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      sp[j] = __ldg(reinterpret_cast<const unsigned short*>(&ckv[(size_t)(kb * 8 + j) * DC + dv]));
    sw128_store16(vs, dv, kb * 8, KT, v);
  }
  __syncthreads();


  for (int k0 = 0; k0 < Sk; k0 += KT) {
    const bool has_next = (k0 + KT < Sk);
    // ---- QK^T: S = Q @ K^T（两 WG 各算一份，dup 2x）----
#pragma unroll
    for (int i = 0; i < 32; ++i) S[i] = 0.f;
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < DK / 16; ++s) {
      wgmma_m64n64k16(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
                      make_desc_sw128(sw128_k16_addr(ks_a, s), SBO_K));
    }
    wgmma_commit();
    wgmma_wait0();

    // ---- softmax（每 warp 负责自己的 16 行）----
    float bmx0 = -INFINITY, bmx1 = -INFINITY;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      bmx0 = fmaxf(bmx0, fmaxf(S[t * 4 + 0], S[t * 4 + 1]));
      bmx1 = fmaxf(bmx1, fmaxf(S[t * 4 + 2], S[t * 4 + 3]));
    }
    bmx0 = fmaxf(bmx0, __shfl_xor_sync(~0u, bmx0, 1));
    bmx0 = fmaxf(bmx0, __shfl_xor_sync(~0u, bmx0, 2));
    bmx1 = fmaxf(bmx1, __shfl_xor_sync(~0u, bmx1, 1));
    bmx1 = fmaxf(bmx1, __shfl_xor_sync(~0u, bmx1, 2));
    const float nm0 = fmaxf(m0, bmx0), nm1 = fmaxf(m1, bmx1);
    const float al0 = __expf(m0 - nm0), al1 = __expf(m1 - nm1);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
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
    l0 = l0 * al0 + s0;
    l1 = l1 * al1 + s1;
    m0 = nm0;
    m1 = nm1;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      O[t * 4 + 0] *= al0;
      O[t * 4 + 1] *= al0;
      O[t * 4 + 2] *= al1;
      O[t * 4 + 3] *= al1;
    }
    // ---- 写 P ----
    if (wg == 0) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int col = t * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(ps + sw128_off(r0, col, KT)) =
            pack2(S[t * 4 + 0], S[t * 4 + 1]);
        *reinterpret_cast<uint32_t*>(ps + sw128_off(r1, col, KT)) =
            pack2(S[t * 4 + 2], S[t * 4 + 3]);
      }
    }
    // ---- 预取下一块 K 到 ks（QK 已读完 ks；cp.async 与 softmax/P/PV 重叠）----
    if (has_next) {
      for (int idx = tid; idx < KT * (DC / 8); idx += T) {
        const int kv = idx / (DC / 8), cu = idx % (DC / 8);
        cp_async16(ks_a + sw128_off(kv, cu * 8, DK),
                   &ckv[(size_t)(k0 + KT + kv) * DC + cu * 8]);
      }
      for (int idx = tid; idx < KT * (DR / 8); idx += T) {
        const int kv = idx / (DR / 8), cu = idx % (DR / 8);
        cp_async16(ks_a + sw128_off(kv, DC + cu * 8, DK),
                   &kr[(size_t)(k0 + KT + kv) * DR + cu * 8]);
      }
      cp_commit();
    }
    __syncthreads();

    // ---- PV: O += P @ V ----
    const uint32_t vs_w = vs_a + (uint32_t)(wg * (DVW / 8) * 1024);
    wgmma_fence();
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      wgmma_m64n256k16(O, make_desc_sw128(sw128_k16_addr(ps_a, c), SBO_P),
                       make_desc_sw128(sw128_k16_addr(vs_w, c), SBO_P));
    }
    wgmma_commit();
    if (has_next) cp_wait0();
    wgmma_wait0();
    // ---- 载入下一块 V（PV 已读完 vs）----
    if (has_next) {
      for (int i = tid; i < (KT / 8) * DV; i += T) {
        const int kb = i / DV, dv = i % DV;
        uint4 v;
        unsigned short* s = reinterpret_cast<unsigned short*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          s[j] = __ldg(reinterpret_cast<const unsigned short*>(
              &ckv[(size_t)(k0 + KT + kb * 8 + j) * DC + dv]));
        sw128_store16(vs, dv, kb * 8, KT, v);
      }
    }
    __syncthreads();
  }

  // ---- 输出 ----
  const float iv0 = 1.f / l0, iv1 = 1.f / l1;
  const int qr0 = q0 + r0, qr1 = q0 + r1;
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    const int col = wg * DVW + t * 8 + (lane & 3) * 2;
    if (qr0 < Sq) {
      out[((size_t)h * Sq + qr0) * DV + col] = O[t * 4 + 0] * iv0;
      out[((size_t)h * Sq + qr0) * DV + col + 1] = O[t * 4 + 1] * iv0;
    }
    if (qr1 < Sq) {
      out[((size_t)h * Sq + qr1) * DV + col] = O[t * 4 + 2] * iv1;
      out[((size_t)h * Sq + qr1) * DV + col + 1] = O[t * 4 + 3] * iv1;
    }
  }
}

// ===========================================================================
static void mla_row_ref(int h, int q, int Sq, int Sk, const std::vector<bf16>& qa,
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
  double sum = 0;
  for (int k = 0; k < Sk; ++k) {
    s[k] = std::exp(s[k] - mx);
    sum += s[k];
  }
  out_row.assign(DV, 0.0);
  for (int k = 0; k < Sk; ++k) {
    const double p = s[k] / sum;
    for (int d = 0; d < DV; ++d)
      out_row[d] += p * (double)__bfloat162float(ckv[(size_t)k * DC + d]);
  }
}

int main(int argc, char** argv) {
  int Sq = (argc > 1) ? std::atoi(argv[1]) : 1024;
  int H = (argc > 2) ? std::atoi(argv[2]) : 128;
  int Sk = (argc > 3) ? std::atoi(argv[3]) : 1024;

  DeviceInfo d = device_info(0);
  print_device_info(d);

  const size_t qaN = (size_t)H * Sq * DC, qrN = (size_t)H * Sq * DR;
  const size_t ckN = (size_t)Sk * DC, krN = (size_t)Sk * DR, outN = (size_t)H * Sq * DV;
  const double flops = 2.0 * Sq * H * Sk * (DC + DR + DV);

  std::printf("\nMLA wgmma pipe3 (coalesced V + cp.async K prefetch): H=%d Sq=%d Sk=%d  (BM=%d KT=%d NWG=%d)\n", H, Sq, Sk, BM,
              KT, NWG);
  std::printf("FLOPs = %.2f GFLOP\n\n", flops / 1e9);

  bf16 *qa, *qr, *ckv, *kr;
  float* out;
  CUDA_CHECK(cudaMalloc(&qa, qaN * 2));
  CUDA_CHECK(cudaMalloc(&qr, qrN * 2));
  CUDA_CHECK(cudaMalloc(&ckv, ckN * 2));
  CUDA_CHECK(cudaMalloc(&kr, krN * 2));
  CUDA_CHECK(cudaMalloc(&out, outN * 4));

  std::vector<bf16> hqa(qaN), hqr(qrN), hck(ckN), hkr(krN);
  std::vector<float> hout(outN);
  srand(1234);
  auto rnd = [] { return __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (auto& x : hqa) x = rnd();
  for (auto& x : hqr) x = rnd();
  for (auto& x : hck) x = rnd();
  for (auto& x : hkr) x = rnd();
  CUDA_CHECK(cudaMemcpy(qa, hqa.data(), qaN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(qr, hqr.data(), qrN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ckv, hck.data(), ckN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(kr, hkr.data(), krN * 2, cudaMemcpyHostToDevice));

  const size_t shm = (size_t)BM * DK * 2 + (size_t)KT * DK * 2 + (size_t)BM * KT * 2 +
                     (size_t)DV * KT * 2;
  auto fn = mla_pipe3_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(div_up(Sq, BM), H);
  auto run = [&] { fn<<<grid, T, shm>>>(qa, qr, ckv, kr, out, Sq, Sk); };
  run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hout.data(), out, outN * 4, cudaMemcpyDeviceToHost));
  double err = 0, ref = 0;
  for (int i = 0; i < 3; ++i) {
    const int h = (i * 61 + 7) % H, q = (i * 197 + 3) % Sq;
    std::vector<double> row;
    mla_row_ref(h, q, Sq, Sk, hqa, hqr, hck, hkr, row);
    for (int dd = 0; dd < DV; ++dd) {
      const double e = row[dd];
      err = std::max(err, std::fabs((double)hout[((size_t)h * Sq + q) * DV + dd] - e));
      ref = std::max(ref, std::fabs(e));
    }
  }
  std::printf("  [sw128 ] max_abs_err=%.3e (ref~%.3f) %s\n", err, ref,
              err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  double t = bench_ms(run, 3, 10);
  const double tfl = to_tflops(flops, t);
  std::printf("wgmma_pipe3 %8.4f ms  %8.2f TFLOPS (%5.1f%% peak, QK dup 2x)\n", t, tfl,
              100.0 * tfl / 989.0);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
