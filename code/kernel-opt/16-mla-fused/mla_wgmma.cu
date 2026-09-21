// 16 MLA 注意力（二）· wgmma 版：用 Hopper warpgroup MMA 重写融合 kernel
//
// 与 mla_fused.cu（mma.m16n8k16 + ldmatrix）相同的融合思路：
//   QK^T 的累加器直接 online softmax，P 不落 DRAM，KV 常驻 smem。
// 区别在计算引擎：用 `wgmma.mma_async.m64nNk16`（A/B 都从 smem 经描述符读取），
// 彻底消掉 ldmatrix 的 L1 共享内存流量（那是 mma 版的瓶颈）。
//
// 布局：GMMA K-major INTERLEAVE（无 swizzle）的 8x8 core-matrix 分块：
//   off(mn,k) = (mn/8)*SBO + (k/8)*128 + (mn%8)*16 + (k%8)*2   (字节)
//   描述符 LBO=128（k 方向相邻 core 的步长），SBO=(K/8)*128。
//   一个 k16 步进 = 2 个 k-block，描述符起始地址 +256。
// 配置：block=2 个 warpgroup(256 线程)，BM=64，DV 切 2 份（每 WG 256）。
//
// 运行：ARCH=sm_90a scripts/run.sh 16-mla-fused/mla_wgmma.cu [Sq] [H] [Sk]
#include "../common/cuda_utils.cuh"
#include "./wgmma_helpers.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <cstring>

using bf16 = __nv_bfloat16;
constexpr int DC = 512, DR = 64, DV = 512, DK = DC + DR;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  return d;
}
// K-major INTERLEAVE blocked 偏移
__device__ __forceinline__ int koff(int mn, int k, int sbo) {
  return (mn / 8) * sbo + (k / 8) * 128 + (mn % 8) * 16 + (k % 8) * 2;
}
// 16x16 累加器 tile 的行/列 -> 线程映射
__device__ __forceinline__ int row0_of(int w_, int lane) { return 16 * (w_ & 3) + (lane >> 2); }

constexpr int BM = 64, KT = 64, NWG = 2, DVW = DV / NWG;
constexpr int SBO_K = (DK / 8) * 128;  // Q/K 的 mn-group 步长 = 9216
constexpr int SBO_P = (KT / 8) * 128;  // P/V 的 mn-group 步长 = 1024

__global__ void __launch_bounds__(NWG * 128) mla_wgmma_kernel(
    const bf16* __restrict__ qa, const bf16* __restrict__ qr,
    const bf16* __restrict__ ckv, const bf16* __restrict__ kr, float* __restrict__ out, int Sq,
    int Sk) {
  extern __shared__ char smem[];
  bf16* qs = reinterpret_cast<bf16*>(smem);                 // [BM][DK] blocked
  bf16* ks = qs + (size_t)(BM / 8) * (DK / 8) * 64;         // [KT][DK] blocked
  bf16* ps = ks + (size_t)(KT / 8) * (DK / 8) * 64;         // [BM][KT] blocked
  bf16* vs = ps + (size_t)(BM / 8) * (KT / 8) * 64;         // [DV][KT] blocked (转置)

  const int h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wg = tid >> 7;   // warpgroup 0/1
  const int W = tid >> 5;    // 全局 warp
  const int T = NWG * 128;
  const int r0 = row0_of(W, lane), r1 = r0 + 8;  // 该 warp 的两行（块内）

  // ---- 载入 Q（uint4 向量化）----
  // 线程映射：warp 内 lane>>2 = 行(mn%8)，lane&3 = k 块，保证 store 最少 bank 冲突
  for (int j = tid >> 5; j < (BM / 8) * (DC / 8 / 4); j += T >> 5) {
    const int lane = tid & 31;
    const int rg = j % (BM / 8), bg = j / (BM / 8);
    const int m = rg * 8 + (lane >> 2), c = bg * 4 + (lane & 3);
    const int gq = q0 + m;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qa[((size_t)h * Sq + gq) * DC + c * 8]);
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(qs) + koff(m, c * 8, SBO_K)) = v;
  }
  for (int j = tid >> 5; j < (BM / 8) * (DR / 8 / 4); j += T >> 5) {
    const int lane = tid & 31;
    const int rg = j % (BM / 8), bg = j / (BM / 8);
    const int m = rg * 8 + (lane >> 2), c = bg * 4 + (lane & 3);
    const int gq = q0 + m;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gq < Sq) v = *reinterpret_cast<const uint4*>(&qr[((size_t)h * Sq + gq) * DR + c * 8]);
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(qs) + koff(m, DC + c * 8, SBO_K)) = v;
  }

  float S[32];
  float O[128];
#pragma unroll
  for (int i = 0; i < 32; ++i) S[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 128; ++i) O[i] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  __syncthreads();

  for (int k0 = 0; k0 < Sk; k0 += KT) {
    // ---- 载入 K = [c_kv | k_rope] -> ks（MN=kv），uint4 + 冲突最小映射 ----
    for (int j = tid >> 5; j < (KT / 8) * (DC / 8 / 4); j += T >> 5) {
      const int lane = tid & 31;
      const int rg = j % (KT / 8), bg = j / (KT / 8);
      const int kv = rg * 8 + (lane >> 2), c = bg * 4 + (lane & 3);
      uint4 v = *reinterpret_cast<const uint4*>(&ckv[(size_t)(k0 + kv) * DC + c * 8]);
      *reinterpret_cast<uint4*>(reinterpret_cast<char*>(ks) + koff(kv, c * 8, SBO_K)) = v;
    }
    for (int j = tid >> 5; j < (KT / 8) * (DR / 8 / 4); j += T >> 5) {
      const int lane = tid & 31;
      const int rg = j % (KT / 8), bg = j / (KT / 8);
      const int kv = rg * 8 + (lane >> 2), c = bg * 4 + (lane & 3);
      uint4 v = *reinterpret_cast<const uint4*>(&kr[(size_t)(k0 + kv) * DR + c * 8]);
      *reinterpret_cast<uint4*>(reinterpret_cast<char*>(ks) + koff(kv, DC + c * 8, SBO_K)) = v;
    }
    // ---- 载入 V = c_kv 转置 -> vs（MN=dv, K=kv）----
    // 每线程取一个 dv、沿 kv 收集 8 个，凑成 16B 连续写回
    for (int i = tid; i < DV * (KT / 8); i += T) {
      const int dv = i / (KT / 8), kb = i % (KT / 8);
      uint4 v;
      unsigned short* s = reinterpret_cast<unsigned short*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        s[j] = *reinterpret_cast<const unsigned short*>(&ckv[(size_t)(k0 + kb * 8 + j) * DC + dv]);
      *reinterpret_cast<uint4*>(reinterpret_cast<char*>(vs) + koff(dv, kb * 8, SBO_P)) = v;
    }
    __syncthreads();

    // ---- QK^T: S = Q @ K^T，两 WG 都算（重复 2x）----
    // wgmma 是累加语义，每块开头清零
#pragma unroll
    for (int i = 0; i < 32; ++i) S[i] = 0.f;
    const uint32_t qs_a = smem_u32(qs), ks_a = smem_u32(ks);
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < DK / 16; ++s) {
      wgmma_m64n64k16(S, make_desc(qs_a + s * 256, 128, SBO_K),
                      make_desc(ks_a + s * 256, 128, SBO_K));
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
    // ---- 写 P（只有 wg0 写，写全 64 行）----
    if (wg == 0) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int col = t * 8 + (lane & 3) * 2;
        *reinterpret_cast<bf16*>(reinterpret_cast<char*>(ps) + koff(r0, col, SBO_P)) =
            __float2bfloat16(S[t * 4 + 0]);
        *reinterpret_cast<bf16*>(reinterpret_cast<char*>(ps) + koff(r0, col + 1, SBO_P)) =
            __float2bfloat16(S[t * 4 + 1]);
        *reinterpret_cast<bf16*>(reinterpret_cast<char*>(ps) + koff(r1, col, SBO_P)) =
            __float2bfloat16(S[t * 4 + 2]);
        *reinterpret_cast<bf16*>(reinterpret_cast<char*>(ps) + koff(r1, col + 1, SBO_P)) =
            __float2bfloat16(S[t * 4 + 3]);
      }
    }
    __syncthreads();

    // ---- PV: O += P @ V（每 WG 一个 256 列切片）----
    const uint32_t ps_a = smem_u32(ps);
    const uint32_t vs_a = smem_u32(vs) + (uint32_t)(wg * (DVW / 8) * SBO_P);
    wgmma_fence();
#pragma unroll
    for (int c = 0; c < KT / 16; ++c) {
      wgmma_m64n256k16(O, make_desc(ps_a + c * 256, 128, SBO_P),
                       make_desc(vs_a + c * 256, 128, SBO_P));
    }
    wgmma_commit();
    wgmma_wait0();
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

  std::printf("\nMLA wgmma fused: H=%d Sq=%d Sk=%d  (BM=%d KT=%d NWG=%d)\n", H, Sq, Sk, BM, KT,
              NWG);
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

  const size_t shm = (size_t)((BM / 8) * (DK / 8) + (KT / 8) * (DK / 8) + (BM / 8) * (KT / 8) +
                              (DV / 8) * (KT / 8)) *
                     64 * sizeof(bf16);
  auto fn = mla_wgmma_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(div_up(Sq, BM), H);
  auto run = [&] { fn<<<grid, NWG * 128, shm>>>(qa, qr, ckv, kr, out, Sq, Sk); };
  run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hout.data(), out, outN * sizeof(float), cudaMemcpyDeviceToHost));
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
  std::printf("  [wgmma ] max_abs_err=%.3e (ref~%.3f) %s\n", err, ref,
              err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  double t = bench_ms(run, 3, 10);
  const double tfl = to_tflops(flops, t);
  std::printf("wgmma    %8.4f ms  %8.2f TFLOPS (%5.1f%% peak, QK dup 2x)\n", t, tfl,
              100.0 * tfl / 989.0);

  CUDA_CHECK(cudaFree(qa));
  CUDA_CHECK(cudaFree(qr));
  CUDA_CHECK(cudaFree(ckv));
  CUDA_CHECK(cudaFree(kr));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
