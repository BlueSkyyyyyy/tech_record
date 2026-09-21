// 40 W4A16 dequant-GEMM（主题 26，AWQ/GPTQ 风格）
//
// 部署侧量化推理：权重按 int4（group_size=128，per-group 对称 scale）打包，
// 激活仍是 bf16。GEMM  Y[M,N] = X[M,K] @ dequant(W[N,K])^T。
//
// 关键动机（decode 场景）：M 很小（1~64），整个算子卡在「必须把 N*K 个权重读一遍」。
//   bf16 权重  = 2*N*K 字节
//   int4 权重  = 0.5*N*K 字节  → 权重流量直接 ÷4
// 若把 dequant 融进 GEMM 主循环（而不是先物化 bf16 权重再跑 cuBLAS），
// 就能把这 4× 的字节节省兑现成时间。
//
// 真实 shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408（MLP up_proj/gate_proj）。
//   本文件默认测 M=64（小 batch decode）× N=17408 × K=5120，group_size=128。
//
// 变体：
//   v0  dequant（物化 bf16 W）+ cuBLAS bf16 GEMM —— 朴素部署路径
//   v1  fused：cp.async 流水 + 主循环内 dequant int4→bf16 SW128 + wgmma SS（可选 split-K）
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcublas" \
//         scripts/run.sh 40-w4a16-gemm/w4a16_gemm.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int GROUP = 128;        // group_size
constexpr int BK = 64;            // 主循环沿 K 的分块（bf16 SW128 atom：64 元素 = 128B）
constexpr double BF16_PEAK = 989.0;

// ---------------------------------------------------------------------------
// wgmma helpers（bf16，SW128 K-major；同 20/22 篇）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// bf16 m64n128k16：64 个 fp32 累加器
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// bf16 SW128 K-major：[row/8][k/64][8][64]，atom 1024B，16B 列 c'=c^r
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;  // LBO = 1 (16B)
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
__device__ __forceinline__ uint32_t k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

// ---------------------------------------------------------------------------
// fused W4A16 kernel
//   BM 必须 64（1 个 warpgroup=128 线程）；BN = 128*NSPLIT；BK=64。
//   cp.async 把 A（bf16）与 Wp（int4 打包）搬进 smem；
//   主循环内把 Wp 反量化成 SW128 布局的 bf16 sW，再喂 wgmma。
//   KSPLIT>1 时沿 K 切分，累加用 atomicAdd（C 需预先清零）。
// ---------------------------------------------------------------------------
template <int BM, int BN, int STAGES, int KSPLIT, bool DQ = true>
__global__ void __launch_bounds__(128) w4a16_kernel(
    const bf16* __restrict__ A,      // [M, K]
    const uint8_t* __restrict__ Wp,  // [N, K/2]  packed int4（低 nibble=k 偶，高=奇；存 q+8）
    const float* __restrict__ Ws,    // [N, K/GROUP] per-group scale
    float* __restrict__ C, int M, int N, int K) {
  static_assert(BM == 64, "BM 固定 64（1 warpgroup）");
  constexpr int NPB = BK / 2;             // 每 k-tile 的 packed 字节数
  constexpr int NSPLIT = BN / 128;
  constexpr int NT = 128;
  constexpr int SBO = 1024;               // (BK/64)*1024

  extern __shared__ __align__(1024) char smem[];
  char* As = smem;                                  // [STAGES][BM][BK] bf16 SW128
  char* Wps = As + (size_t)STAGES * BM * BK * 2;    // [STAGES][BN][NPB] bytes
  char* sW = Wps + (size_t)STAGES * BN * NPB;       // [STAGES][BN][BK] bf16 SW128

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nblk_z = kb1 - kb0;

  auto load_stage = [&](int st, int kb) {
    const int k0 = kb * BK;
    char* a = As + (size_t)st * BM * BK * 2;
    char* wp = Wps + (size_t)st * BN * NPB;
    for (int i = tid; i < BM * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      __pipeline_memcpy_async(a + sw128_off(r, c8, BK),
                              &A[(size_t)(block_row + r) * K + k0 + c8], 16);
    }
    for (int i = tid; i < BN * (NPB / 16); i += NT) {
      const int r = i / (NPB / 16), c = (i % (NPB / 16)) * 16;
      __pipeline_memcpy_async(wp + r * NPB + c,
                              &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
    }
    __pipeline_commit();
  };

  // 把 Wp[st] 反量化到 sW[st]（SW128 bf16）。每线程处理 1 个 uint32（4 packed 字节 = 8 个 k）
  auto dequant_stage = [&](int st, int kb) {
    if constexpr (!DQ) return;  // 诊断：跳过反量化算术，隔离 wgmma+搬运的地板
    char* w = sW + (size_t)st * BN * BK * 2;
    const int ngrp = K / GROUP;
    const int g = kb >> 1;                        // k0/GROUP = kb*64/128
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(Wps + (size_t)st * BN * NPB);
    constexpr int M4 = NPB / 4;                   // 每行 8 个 uint32
    for (int i = tid; i < BN * M4; i += NT) {
      const int n = i / M4, m4 = i % M4;
      const uint32_t word = wp[n * M4 + m4];
      const float s = __ldg(&Ws[(size_t)(block_col + n) * ngrp + g]);
      bf16 out[8];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int q0 = (int)((word >> (8 * b)) & 0xF) - 8;
        const int q1 = (int)((word >> (8 * b + 4)) & 0xF) - 8;
        out[2 * b] = __float2bfloat16((float)q0 * s);
        out[2 * b + 1] = __float2bfloat16((float)q1 * s);
      }
      const int kk = (8 * m4) & 63;               // 本 uint32 覆盖 k=8*m4..8*m4+7
      const int rr = n & 7, rg = n >> 3;
      const int cc = (kk >> 3) ^ rr;
      const int off = rg * 1024 + (rr * 8 + cc) * 16;
      *reinterpret_cast<uint4*>(w + off) = *reinterpret_cast<const uint4*>(out);
    }
  };

  float acc[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s)
    if (s < nblk_z) load_stage(s, kb0 + s);

  for (int t = 0; t < nblk_z; ++t) {
    const int st = t % STAGES;
    const int kb = kb0 + t;
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    dequant_stage(st, kb);
    __syncthreads();  // 让 sW 对所有线程可见
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    char* a = As + (size_t)st * BM * BK * 2;
    char* w = sW + (size_t)st * BN * BK * 2;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* wj = w + (size_t)jn * 128 * BK * 2;
#pragma unroll
      for (int s = 0; s < BK / 16; ++s) {
        uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), s), SBO);
        uint64_t db = make_desc_sw128(k16_addr(smem_u32(wj), s), SBO);
        wgmma_m64n128k16(acc[jn], da, db);
      }
    }
    wgmma_commit();
    if (t + STAGES - 1 < nblk_z) wgmma_wait_group<STAGES - 2>();
    __syncthreads();
    const int next = t + STAGES - 1;
    if (next < nblk_z) load_stage(next % STAGES, kb0 + next);
  }
  wgmma_wait0();

  const int row0 = (warp & 3) * 16 + (lane >> 2);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = block_col + jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8;
      if (KSPLIT == 1) {
        if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + col]) = make_float2(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + col]) = make_float2(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      } else {
        if (r0 < M) { atomicAdd(&C[(size_t)r0 * N + col], acc[jn][j * 4 + 0]); atomicAdd(&C[(size_t)r0 * N + col + 1], acc[jn][j * 4 + 1]); }
        if (r1 < M) { atomicAdd(&C[(size_t)r1 * N + col], acc[jn][j * 4 + 2]); atomicAdd(&C[(size_t)r1 * N + col + 1], acc[jn][j * 4 + 3]); }
      }
    }
}

// ---------------------------------------------------------------------------
// 朴素 dequant：packed int4 -> bf16 W[N,K]（物化），v0 基线的第一步（向量化）
// ---------------------------------------------------------------------------
__global__ void dequant_w_kernel(const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
                                 bf16* __restrict__ W, int N, int K) {
  const int ngrp = K / GROUP;
  const int K8 = K / 8;
  const size_t total = (size_t)N * K8;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    const int n = i / K8, c8 = (i % K8) * 8;
    const uint32_t word = *reinterpret_cast<const uint32_t*>(Wp + (size_t)n * (K / 2) + c8 / 2);
    const float s = Ws[(size_t)n * ngrp + (c8 / GROUP)];
    bf16 out[8];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      out[2 * b] = __float2bfloat16((float)((int)((word >> (8 * b)) & 0xF) - 8) * s);
      out[2 * b + 1] = __float2bfloat16((float)((int)((word >> (8 * b + 4)) & 0xF) - 8) * s);
    }
    *reinterpret_cast<uint4*>(&W[(size_t)n * K + c8]) = *reinterpret_cast<const uint4*>(out);
  }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
static bf16 f2b(float x) { return __float2bfloat16(x); }

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 64;
  int N = (argc > 2) ? std::atoi(argv[2]) : 17408;
  int K = (argc > 3) ? std::atoi(argv[3]) : 5120;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  const double w_bytes_int4 = (double)N * K / 2;
  const double w_bytes_bf16 = (double)N * K * 2;
  std::printf("\nW4A16 GEMM (AWQ/GPTQ, group=%d): M=%d N=%d K=%d\n", GROUP, M, N, K);
  std::printf("FLOPs = %.2f GFLOP ; W_int4 = %.1f MB ; W_bf16 = %.1f MB (4x)\n\n",
              flops / 1e9, w_bytes_int4 / 1e6, w_bytes_bf16 / 1e6);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<bf16> hA((size_t)M * K);
  std::vector<float> hWf((size_t)N * K);          // 原始 float 权重
  std::vector<float> hWdeq((size_t)N * K);        // 反量化（= q*scale）
  std::vector<uint8_t> hWp((size_t)N * (K / 2));
  std::vector<float> hWs((size_t)N * (K / GROUP));
  std::vector<float> hC((size_t)M * N);
  for (auto& v : hA) v = f2b(0.5f * dist(rng));
  for (auto& v : hWf) v = 0.1f * dist(rng);

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < K / GROUP; ++g) {
      float amax = 0.f;
      for (int k = 0; k < GROUP; ++k) amax = std::max(amax, std::fabs(hWf[(size_t)n * K + g * GROUP + k]));
      const float s = amax > 0 ? amax / 7.f : 1.f;
      hWs[(size_t)n * (K / GROUP) + g] = s;
      for (int k = 0; k < GROUP; ++k) {
        const size_t idx = (size_t)n * K + g * GROUP + k;
        int q = (int)std::lround(hWf[idx] / s);
        q = std::max(-8, std::min(7, q));
        hWdeq[idx] = q * s;
        const size_t pk = (size_t)n * (K / 2) + (g * GROUP + k) / 2;
        if (((g * GROUP + k) & 1) == 0)
          hWp[pk] = (uint8_t)((hWp[pk] & 0xF0) | ((q + 8) & 0xF));
        else
          hWp[pk] = (uint8_t)((hWp[pk] & 0x0F) | (((q + 8) & 0xF) << 4));
      }
    }
  }

  bf16 *dA; uint8_t *dWp; float *dWs, *dC; bf16* dWb;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&dWp, hWp.size()));
  CUDA_CHECK(cudaMalloc(&dWs, hWs.size() * 4));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&dWb, (size_t)N * K * 2));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWs, hWs.data(), hWs.size() * 4, cudaMemcpyHostToDevice));

  auto cpu_ref = [&](int m, int n) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += (double)__bfloat162float(hA[(size_t)m * K + k]) * (double)hWdeq[(size_t)n * K + k];
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 32; ++i) {
      const int m = (i * 1009 + 13) % M, n = (i * 997 + 7) % N;
      const double e = cpu_ref(m, n);
      err = std::max(err, std::fabs((double)hC[(size_t)m * N + n] - e));
      ref = std::max(ref, std::fabs(e));
    }
    const bool ok = err / std::max(ref, 1e-6) < 3e-2;
    std::printf("  [%-18s] max_abs_err=%.3e (ref~%.2f) %s\n", tag, err, ref, ok ? "OK" : "FAIL");
    return ok;
  };
  auto report = [&](const char* tag, double ms, double wbytes) {
    const double tf = to_tflops(flops, ms);
    const double wbw = wbytes * 1000.0 / ms / 1e9;
    std::printf("%-20s %9.4f ms  %8.2f TFLOPS (%5.1f%% bf16)  W-bw %7.1f GB/s\n", tag, ms, tf,
                100.0 * tf / BF16_PEAK, wbw);
  };
  auto want = [&](const char* name) { return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0; };

  // ---- v0：物化 dequant + cuBLAS bf16 ----
  if (want("v0")) {
    auto dq = [&] { dequant_w_kernel<<<1024, 256>>>(dWp, dWs, dWb, N, K); };
    dq();
    CUDA_CHECK_LAST();
    const double t_dq = bench_ms(dq, 5, 30);

    cublasHandle_t h; cublasCreate(&h);
    const float one = 1.f, zero = 0.f;
    auto gemm = [&] {
      cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one,
                   dWb, CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &zero, dC, CUDA_R_32F, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    gemm();
    CUDA_CHECK(cudaDeviceSynchronize());
    check("v0-cublas");
    const double t_gemm = bench_ms(gemm, 5, 30);
    cublasDestroy(h);
    std::printf("v0 dequant      %9.4f ms  (read %.1f MB int4, write %.1f MB bf16)\n", t_dq,
                w_bytes_int4 / 1e6, w_bytes_bf16 / 1e6);
    report("v0 cuBLAS-bf16", t_gemm, w_bytes_bf16);
    report("v0 TOTAL", t_dq + t_gemm, w_bytes_bf16);
  }

  // ---- v1：fused ----（which 直接写某个 config 名时只跑那个，便于 ncu）
  if (want("v1") || which[0] == 'f' || std::strcmp(which, "nodq") == 0) {
#define RUN_FUSED(NAME, BM, BN, ST, KS)                                                                 \
    do {                                                                                               \
      auto fn = w4a16_kernel<BM, BN, ST, KS>;                                                          \
      const size_t shm = (size_t)ST * ((size_t)BM * BK * 2 + (size_t)BN * (BK / 2) + (size_t)BN * BK * 2); \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));     \
      dim3 grid(div_up(N, BN), div_up(M, BM), KS);                                                     \
      auto launch = [&] {                                                                              \
        if (KS > 1) cudaMemsetAsync(dC, 0, (size_t)M * N * 4);                                         \
        fn<<<grid, 128, shm>>>(dA, dWp, dWs, dC, M, N, K);                                             \
      };                                                                                               \
      launch();                                                                                        \
      CUDA_CHECK_LAST();                                                                               \
      check(NAME);                                                                                     \
      const double t = bench_ms(launch, 5, 50);                                                        \
      report(NAME, t, w_bytes_int4);                                                                   \
    } while (0)

    RUN_FUSED("fused64x128s3k1", 64, 128, 3, 1);
    RUN_FUSED("fused64x128s2k1", 64, 128, 2, 1);
    RUN_FUSED("fused64x128s3k2", 64, 128, 3, 2);
    RUN_FUSED("fused64x128s3k4", 64, 128, 3, 4);
    RUN_FUSED("fused64x128s3k8", 64, 128, 3, 8);
    RUN_FUSED("fused64x128s2k4", 64, 128, 2, 4);
    RUN_FUSED("fused64x128s4k4", 64, 128, 4, 4);
    RUN_FUSED("fused64x256s3k2", 64, 256, 3, 2);
    RUN_FUSED("fused64x256s3k4", 64, 256, 3, 4);

    if (want("nodq")) {  // 诊断：关掉 dequant 算术（结果无意义，只看时间地板）
      auto fn = w4a16_kernel<64, 128, 3, 4, false>;
      const size_t shm = 3 * (64 * BK * 2 + 128 * (BK / 2) + 128 * BK * 2);
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      dim3 grid(div_up(N, 128), div_up(M, 64), 4);
      auto launch = [&] { cudaMemsetAsync(dC, 0, (size_t)M * N * 4); fn<<<grid, 128, shm>>>(dA, dWp, dWs, dC, M, N, K); };
      launch();
      CUDA_CHECK_LAST();
      const double t = bench_ms(launch, 5, 50);
      report("nodq64x128s3k4", t, w_bytes_int4);
    }
  }

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dWp)); CUDA_CHECK(cudaFree(dWs));
  CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dWb));
  return 0;
}
