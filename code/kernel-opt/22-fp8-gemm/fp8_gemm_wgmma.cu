// 22b FP8 GEMM（wgnma 版）：e4m3 + per-tensor / per-block scaling，wgmma.m64n128k32
//
// mma.sync.m16n8k32 版（fp8_gemm.cu）受限于 HMMA 发射端口，ncu 显示 tensor pipe 41%、
// math_pipe_throttle 1.12，实测 265 TFLOPS（13.4%）；cuBLAS（wgmma）1378（69.7%）。
// 本版把主循环换成 Hopper 的 wgmma（SS，A/B 直接吃 smem 描述符）：
//   - FP8 的 K-major SW128 atom 与 bf16 完全同构（8 行 × 128B，Swizzle<3,4,3>），
//     只是 atom 内一行从 64 个 bf16 变成 128 个 e4m3；k32 步进同为 32B。
//   - A[M,K]/B[N,K] 都是 K-major，天然符合 wgmma-SS 的 B=(K,N) Major::K 要求，无需转置。
//   - BM=128 = 2 warpgroup，每个 wgmma.m64n128k32；BK=128（4 个 k32）。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//         scripts/run.sh 22-fp8-gemm/fp8_gemm_wgmma.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;

// ---------------------------------------------------------------------------
// wgmma helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void wgmma_m64n128k32(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, %67, %68;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// K-major SW128：8 行 × 128B atom；fp8 一行 128 个元素。K = tile 沿 K 的元素数（128 的倍数）。
__device__ __forceinline__ int sw_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int c = kk >> 4, cc = c ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;  // LBO = 1 (16B)，B128 下忽略
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
// k32 块 s 的描述符地址增量
__device__ __forceinline__ uint32_t k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

// ---------------------------------------------------------------------------
// wgmma GEMM：BM=128(2 WG) × BN=128，BK=128，多级 cp.async 流水
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool PERBLOCK>
__global__ void __launch_bounds__((BM / 64) * 128)
fp8_wgmma_kernel(const fp8* __restrict__ A, const fp8* __restrict__ B, float* __restrict__ C,
                 int M, int N, int K, const float* __restrict__ sa, const float* __restrict__ sb,
                 int KBLK) {
  constexpr int NWG = BM / 64;    // warpgroup 数（每个管 64 行）
  constexpr int NSPLIT = BN / 128;  // 每个 warpgroup 沿 N 切几块 m64n128
  constexpr int NT = NWG * 128;
  constexpr int SBO = (BK / 128) * 1024;  // 相邻 8 行组字节距离

  extern __shared__ __align__(1024) char smem[];
  char* As = smem;                                    // [STAGES][BM][BK] SW128
  char* Bs = As + (size_t)STAGES * BM * BK;           // [STAGES][BN][BK] SW128

  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  auto stage_a = [&](int st) { return As + (size_t)st * BM * BK; };
  auto stage_b = [&](int st) { return Bs + (size_t)st * BN * BK; };

  auto load_stage = [&](int st, int k0) {
    char* a = stage_a(st);
    char* b = stage_b(st);
    for (int i = tid; i < BM * (BK / 16); i += NT) {
      const int r = i / (BK / 16), c = i % (BK / 16);
      __pipeline_memcpy_async(a + sw_off(r, c * 16, BK), &A[(size_t)(block_row + r) * K + k0 + c * 16], 16);
    }
    for (int i = tid; i < BN * (BK / 16); i += NT) {
      const int r = i / (BK / 16), c = i % (BK / 16);
      __pipeline_memcpy_async(b + sw_off(r, c * 16, BK), &B[(size_t)(block_col + r) * K + k0 + c * 16], 16);
    }
    __pipeline_commit();
  };

  float acc[NSPLIT][64];
  float fin[PERBLOCK ? NSPLIT : 1][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc[j][i] = 0.f; if (PERBLOCK) fin[j][i] = 0.f; }

  const int nblk = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s)
    if (s < nblk) load_stage(s, s * BK);

  for (int kb = 0; kb < nblk; ++kb) {
    const int st = kb % STAGES;
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    char* a = stage_a(st) + (size_t)wg * 64 * BK;
    char* b = stage_b(st);
    wgmma_fence();
    if constexpr (PERBLOCK) {
#pragma unroll
      for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
        for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;
    }
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * BK;
#pragma unroll
      for (int s = 0; s < BK / 32; ++s) {
        uint64_t da = make_desc_sw128(k32_addr(smem_u32(a), s), SBO);
        uint64_t db = make_desc_sw128(k32_addr(smem_u32(bj), s), SBO);
        wgmma_m64n128k32(acc[jn], da, db);
      }
    }
    wgmma_commit();
    if constexpr (!PERBLOCK) {
      // 只保留 STAGES-2 个 wgmma group 在飞：per-tensor 的 acc 到最后才读，
      // 不必每块 wait0；靠 wait_group 保证「读某 stage 的 wgmma」在覆盖前完成。
      if (kb + STAGES - 1 < nblk) wgmma_wait_group<STAGES - 2>();
      __syncthreads();
      const int next = kb + STAGES - 1;
      if (next < nblk) load_stage(next % STAGES, next * BK);
      continue;
    }
    wgmma_wait0();
    if constexpr (PERBLOCK) {
      const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
      const float sa0 = sa[(block_row + row0) * KBLK + kb];
      const float sa1 = sa[(block_row + row0 + 8) * KBLK + kb];
#pragma unroll
      for (int jn = 0; jn < NSPLIT; ++jn) {
        // DeepSeek weight_block=128x128：sb 在 128 个输出通道内是常数 → 每个 n-tile 一个标量
        const float sbv = sb[((block_col + jn * 128) >> 7) * KBLK + kb];
        const float s0 = sa0 * sbv, s1 = sa1 * sbv;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          fin[jn][j * 4 + 0] += s0 * acc[jn][j * 4 + 0];
          fin[jn][j * 4 + 1] += s0 * acc[jn][j * 4 + 1];
          fin[jn][j * 4 + 2] += s1 * acc[jn][j * 4 + 2];
          fin[jn][j * 4 + 3] += s1 * acc[jn][j * 4 + 3];
        }
      }
    }
    __syncthreads();
    const int next = kb + STAGES - 1;
    if (next < nblk) load_stage(next % STAGES, next * BK);
  }

  if constexpr (!PERBLOCK) wgmma_wait0();  // 收尾：等最后几个 group
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float* o = PERBLOCK ? fin[jn] : acc[jn];
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if (r0 < M) { C[(size_t)r0 * N + cc] = o[j * 4 + 0]; C[(size_t)r0 * N + cc + 1] = o[j * 4 + 1]; }
      if (r1 < M) { C[(size_t)r1 * N + cc] = o[j * 4 + 2]; C[(size_t)r1 * N + cc + 1] = o[j * 4 + 3]; }
    }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int N = (argc > 2) ? std::atoi(argv[2]) : 3072;
  int K = (argc > 3) ? std::atoi(argv[3]) : 7168;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nFP8 e4m3 wgmma GEMM: M=%d N=%d K=%d  FLOPs=%.2f GFLOP\n\n", M, N, K, flops / 1e9);

  const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  fp8 *A, *B;
  float *C, *sa, *sb;
  CUDA_CHECK(cudaMalloc(&A, aN));
  CUDA_CHECK(cudaMalloc(&B, bN));
  CUDA_CHECK(cudaMalloc(&C, cN * 4));
  CUDA_CHECK(cudaMalloc(&sa, (size_t)M * (K / 128) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sb, (size_t)(N / 128) * (K / 128) * sizeof(float)));

  std::vector<float> hAf(aN), hBf(bN);
  std::vector<fp8> hA(aN), hB(bN);
  std::vector<float> hC(cN), hsa(M), hsb(N / 128);
  srand(1234);
  auto q = [](float x) { return fp8(x); };
  for (size_t i = 0; i < aN; ++i) { float x = 2.f * ((float)rand() / RAND_MAX - .5f); hAf[i] = (float)q(x); hA[i] = q(x); }
  for (size_t i = 0; i < bN; ++i) { float x = 2.f * ((float)rand() / RAND_MAX - .5f); hBf[i] = (float)q(x); hB[i] = q(x); }
  for (int i = 0; i < M; ++i) hsa[i] = 1.0f;
  for (int i = 0; i < N / 128; ++i) hsb[i] = 1.0f;
  CUDA_CHECK(cudaMemcpy(A, hA.data(), aN, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), bN, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sa, hsa.data(), M * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sb, hsb.data(), (N / 128) * 4, cudaMemcpyHostToDevice));

  const int KBLK = K / 128;
  bool use_sc = false;
  std::vector<float> hsca((size_t)M * KBLK), hscb((size_t)(N / 128) * KBLK);
  auto cpu_entry = [&](int r, int c) {
    double s = 0;
    for (int kb = 0; kb < KBLK; ++kb) {
      double p = 0;
      for (int k = kb * 128; k < kb * 128 + 128; ++k)
        p += (double)hAf[(size_t)r * K + k] * (double)hBf[(size_t)c * K + k];
      const double sc = use_sc ? (double)hsca[(size_t)r * KBLK + kb] * hscb[(size_t)(c / 128) * KBLK + kb] : 1.0;
      s += sc * p;
    }
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), C, cN * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 32; ++i) {
      int r = (i * 1009 + 13) % M, c = (i * 997 + 7) % N;
      double e = cpu_entry(r, c);
      err = std::max(err, std::fabs((double)hC[(size_t)r * N + c] - e));
      ref = std::max(ref, std::fabs(e));
    }
    std::printf("  [%-14s] max_abs_err=%.3e (ref~%.1f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms, const DeviceInfo& d) {
    double tf = to_tflops(flops, ms);
    std::printf("%-14s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp8 peak)\n", tag, ms, tf,
                100.0 * tf / FP8_PEAK);
    (void)d;
  };
  auto want = [&](const char* name) { return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0; };

#define RUN_WG(NAME, BM, BN, BK, ST, PB)                                                        \
  do {                                                                                          \
    if (want(NAME)) {                                                                           \
      auto fn = fp8_wgmma_kernel<BM, BN, BK, ST, PB>;                                            \
      const int nt = ((BM) / 64) * 128;                                                          \
      const size_t shm = (size_t)(ST) * ((BM) + (BN)) * (BK);                                    \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
      dim3 grid(div_up(N, BN), div_up(M, BM));                                                   \
      fn<<<grid, nt, shm>>>(A, B, C, M, N, K, sa, sb, K / 128);                                  \
      CUDA_CHECK_LAST();                                                                        \
      check(NAME);                                                                              \
      double t = bench_ms([&] { fn<<<grid, nt, shm>>>(A, B, C, M, N, K, sa, sb, K / 128); }, 5, 30); \
      report(NAME, t, d);                                                                        \
    }                                                                                           \
  } while (0)

  // per-block 版：随机 per-(row,128k) / per-(col,128k) scale（模拟 DeepSeek 1x128 激活 + 128x128 权重）
  std::printf("\n-- per-block scaling (sa per 1x128, sb per 1x128 here) --\n");
  for (auto& x : hsca) x = 0.5f + (float)rand() / RAND_MAX;
  for (auto& x : hscb) x = 0.5f + (float)rand() / RAND_MAX;
  CUDA_CHECK(cudaMemcpy(sa, hsca.data(), hsca.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sb, hscb.data(), hscb.size() * 4, cudaMemcpyHostToDevice));
  use_sc = true;
  RUN_WG("pb128x128x128s3", 128, 128, 128, 3, true);
  RUN_WG("pb256x128x128s3", 256, 128, 128, 3, true);
  use_sc = false;
  CUDA_CHECK(cudaMemset(sa, 0, (size_t)M * (K / 128) * 4));
  CUDA_CHECK(cudaMemset(sb, 0, (size_t)(N / 128) * (K / 128) * 4));
  std::vector<float> ones_m(M, 1.f), ones_n(N / 128, 1.f);
  CUDA_CHECK(cudaMemcpy(sa, ones_m.data(), M * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sb, ones_n.data(), (N / 128) * 4, cudaMemcpyHostToDevice));

  std::printf("\n-- per-tensor --\n");
  RUN_WG("wg128x128x128s2", 128, 128, 128, 2, false);
  RUN_WG("wg128x128x128s3", 128, 128, 128, 3, false);
  RUN_WG("wg128x128x128s4", 128, 128, 128, 4, false);
  RUN_WG("wg128x256x128s3", 128, 256, 128, 3, false);
  RUN_WG("wg256x128x128s3", 256, 128, 128, 3, false);
  RUN_WG("wg256x256x128s3", 256, 256, 128, 3, false);
  RUN_WG("wg256x128x256s2", 256, 128, 256, 2, false);
  RUN_WG("wg128x128x256s2", 128, 128, 256, 2, false);
  RUN_WG("wg128x256x256s2", 128, 256, 256, 2, false);
  RUN_WG("wg256x128x256s3", 256, 128, 256, 3, false);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(sa));
  CUDA_CHECK(cudaFree(sb));
  return 0;
}
