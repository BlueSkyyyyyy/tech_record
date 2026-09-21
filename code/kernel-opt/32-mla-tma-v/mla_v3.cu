// 32 MLA 极限冲刺（三）：QK(i+1) 与 PV(i)/V 装载软件流水
//
// 相对 21 篇 mla_pipe3.cu 的唯一改动：把「下一块的 QK」从下一轮循环提前到本轮
// PV 之前发起，让 wgmma.mma_async 的 QK(i+1) 与同步的 V(i+1) gather、PV(i) 重叠；
// S 累积器在一轮内先被 softmax 消费完（写 P、rescale O）后再被 QK(i+1) 复用，零额外寄存器。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" scripts/run.sh 32-mla-tma-v/mla_v3.cu [Sq] [H] [Sk]
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"

#include <cuda_bf16.h>
#include <cmath>

using bf16 = __nv_bfloat16;
constexpr int DC = 512, DR = 64, DV = 512, DK = DC + DR;

constexpr int SBO_K = (DK / 64) * 1024;
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
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// QK 累加器：pred=0 时 D=A*B（覆盖），pred=1 时 D+=A*B。用 pred=0 发起第一段，
// 省掉「普通寄存器写 0」——否则 ptxas 会在流水阶段间判定 accumulator 被非 wgmma 指令定义而串行化。
__device__ __forceinline__ void wgmma_qk(float (&d)[32], uint64_t da, uint64_t db, int pred) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(pred));
}

__global__ void __launch_bounds__(T) mla_v3_kernel(const bf16* __restrict__ qa,
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
      sp[j] = *reinterpret_cast<const unsigned short*>(&ckv[(size_t)(kb * 8 + j) * DC + dv]);
    sw128_store16(vs, dv, kb * 8, KT, v);
  }
  __syncthreads();

  const uint32_t qs_a = smem_u32(qs), ks_a = smem_u32(ks), ps_a = smem_u32(ps), vs_a = smem_u32(vs);

  // prologue：发起第 0 块的 QK（s=0 用覆盖，省去清 0）
  wgmma_fence();
#pragma unroll
  for (int s = 0; s < DK / 16; ++s)
    wgmma_qk(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
             make_desc_sw128(sw128_k16_addr(ks_a, s), SBO_K), s == 0 ? 0 : 1);
  wgmma_commit();

#pragma unroll 1
  for (int k0 = 0; k0 < Sk; k0 += KT) {
    const bool has_next = (k0 + KT < Sk);
    // QK(k0) 完成
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
      const float e0 = __expf(S[t * 4 + 0] - nm0);
      const float e1 = __expf(S[t * 4 + 1] - nm0);
      const float e2 = __expf(S[t * 4 + 2] - nm1);
      const float e3 = __expf(S[t * 4 + 3] - nm1);
      s0 += e0 + e1;
      s1 += e2 + e3;
      if (wg == 0) {
        const int col = t * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(ps + sw128_off(r0, col, KT)) = pack2(e0, e1);
        *reinterpret_cast<uint32_t*>(ps + sw128_off(r1, col, KT)) = pack2(e2, e3);
      }
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
    // ---- cp.async 预取下一块 K（QK(k0) 已读完 ks）----
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
    __syncthreads();  // ps 对 PV 可见；K 预取已由各线程发起

    if (has_next) {
      cp_wait0();     // 本线程的 K 预取已落地
      __syncthreads();
      // ---- 复用 S：发起下一块 QK（与 PV、V 装载重叠；s=0 覆盖，不清 0）----
      wgmma_fence();
#pragma unroll
      for (int s = 0; s < DK / 16; ++s)
        wgmma_qk(S, make_desc_sw128(sw128_k16_addr(qs_a, s), SBO_K),
                 make_desc_sw128(sw128_k16_addr(ks_a, s), SBO_K), s == 0 ? 0 : 1);
      wgmma_commit();
    }

    // ---- PV: O += P @ V ----
    const uint32_t vs_w = vs_a + (uint32_t)(wg * (DVW / 8) * 1024);
    wgmma_fence();
#pragma unroll
    for (int c = 0; c < KT / 16; ++c)
      wgmma_m64n256k16(O, make_desc_sw128(sw128_k16_addr(ps_a, c), SBO_P),
                       make_desc_sw128(sw128_k16_addr(vs_w, c), SBO_P));
    wgmma_commit();
    if (has_next)
      wgmma_wait_group<1>();  // 等较老的 PV 组完成；QK(next) 仍可在飞
    else
      wgmma_wait_group<0>();

    // ---- 载入下一块 V（PV 已读完 vs；与 QK(next) 重叠）----
    if (has_next) {
      for (int i = tid; i < (KT / 8) * DV; i += T) {
        const int kb = i / DV, dv = i % DV;
        uint4 v;
        unsigned short* s = reinterpret_cast<unsigned short*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          s[j] = *reinterpret_cast<const unsigned short*>(
              &ckv[(size_t)(k0 + KT + kb * 8 + j) * DC + dv]);
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

  std::printf("\nMLA v3 (QK(i+1) || PV(i)+V load pipeline): H=%d Sq=%d Sk=%d  (BM=%d KT=%d NWG=%d)\n",
              H, Sq, Sk, BM, KT, NWG);
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
  auto fn = mla_v3_kernel;
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
  std::printf("  [v3    ] max_abs_err=%.3e (ref~%.3f) %s\n", err, ref,
              err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  double t = bench_ms(run, 3, 10);
  const double tfl = to_tflops(flops, t);
  std::printf("mla_v3 %8.4f ms  %8.2f TFLOPS (%5.1f%% peak, QK dup 2x)\n", t, tfl,
              100.0 * tfl / 989.0);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
