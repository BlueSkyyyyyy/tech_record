// 23 FP8 GEMM（二）：TMA + mbarrier 多级流水 + warp specialization
//
// 承接 22（fp8_gemm_wgmma.cu）：同 shape 4096x3072x7168 的 per-tensor wgmma 版
// 已达 768 TFLOPS（38.8%），但 ncu 显示瓶颈是「1 CTA/SM、25% occupancy、No Eligible 59.9%」，
// 根因是 cp.async 的「算地址 + LDGSTS + 寄存器」占用发射槽和寄存器。
//
// 本版把 A/B 的搬运从「256 线程各自算 SW128 地址 + cp.async 16B」换成
//   - TMA（cp.async.bulk.tensor.2d）+ 128B swizzle：一条指令搬整个 [BM,BK] tile，
//     smem 布局与 wgmma 的 SW128 描述符逐字节一致（TMA 硬件做 swizzle）；
//   - mbarrier 生产者/消费者协议：1 个 producer warp 专职发起 TMA，消费者 warpgroup 只做 wgmma；
//   - 腾出的寄存器和发射槽让张量管线真正跑起来，并尝试 2 CTA/SM。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 23-fp8-gemm-tma/fp8_gemm_tma.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;

// ---------------------------------------------------------------------------
// wgmma helpers（同 22）
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

// K-major SW128：8 行 × 128B atom；fp8 一行 128 个元素。
__device__ __forceinline__ int sw_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int c = kk >> 4, cc = c ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;  // LBO = 1 (16B)
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
__device__ __forceinline__ uint32_t k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

// ---------------------------------------------------------------------------
// mbarrier + TMA helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// 2D TMA load（tile 模式）：[BK(元素), ROW] 的 box，内维 128B + SW128 swizzle
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}

// ---------------------------------------------------------------------------
// TMA + warp-specialization FP8 wgmma GEMM
//   BM 必须是 64 的倍数；BN 是 128 的倍数；BK=128（TMA SW128 内维上限）。
//   消费者 = (BM/64) 个 warpgroup；生产者 = 1 个 warp（32 线程）。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool PERBLOCK = false>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
fp8_tma_ws_kernel(const __grid_constant__ CUtensorMap tmA,
                  const __grid_constant__ CUtensorMap tmB,
                  float* __restrict__ C, int M, int N, int K,
                  const float* __restrict__ sa = nullptr, const float* __restrict__ sb = nullptr,
                  int KBLK = 0) {
  static_assert(BK == 128, "TMA SW128 内维固定 128 字节");
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NCONS = NWG * 128;
  constexpr int NT = NCONS + 32;
  constexpr int SBO = 1024;

  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;                       // [STAGES][BM][BK] SW128
  char* Bs = As + (size_t)STAGES * BM * BK;     // [STAGES][BN][BK] SW128

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  // ---- producer warp：专职发起 TMA ----
  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int kb = 0; kb < nblk; ++kb) {
        const int st = kb % STAGES;
        // 阶段 st 第 n 次被「消费完」的相位 = n&1；生产者在下一次装填前等上一次完成
        if (kb >= STAGES) { mbar_wait(empty + st, (uint32_t)((kb / STAGES - 1) & 1)); }
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * BK + BN * BK);
        tma_load_2d(&tmA, As + (size_t)st * BM * BK, kb * BK, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * BK, kb * BK, block_col, full + st);
      }
    }
    return;
  }

  // ---- consumer warpgroups：只做 wgmma ----
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc[NSPLIT][64];
  float fin[PERBLOCK ? NSPLIT : 1][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc[j][i] = 0.f; if (PERBLOCK) fin[j][i] = 0.f; }

  for (int kb = 0; kb < nblk; ++kb) {
    const int st = kb % STAGES;
    // full[st] 第 (kb/STAGES) 次完成 → 等待的相位就是 (kb/STAGES)&1（避免动态下标进 local mem）
    mbar_wait(full + st, (uint32_t)((kb / STAGES) & 1));
    fence_proxy_async();
    char* a = As + (size_t)st * BM * BK + (size_t)wg * 64 * BK;
    char* b = Bs + (size_t)st * BN * BK;
    if constexpr (PERBLOCK) {
#pragma unroll
      for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
        for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;
    }
    wgmma_fence();
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
    if constexpr (PERBLOCK) {
      wgmma_wait0();
      const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
      const float sa0 = sa[(block_row + row0) * KBLK + kb];
      const float sa1 = sa[(block_row + row0 + 8) * KBLK + kb];
#pragma unroll
      for (int jn = 0; jn < NSPLIT; ++jn) {
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
      mbar_arrive(empty + st);
    } else {
      if (kb >= STAGES - 2) {
        wgmma_wait_group<STAGES - 2>();
        const int rs = (kb - (STAGES - 2)) % STAGES;
        mbar_arrive(empty + rs);
      }
    }
  }
  if constexpr (!PERBLOCK) wgmma_wait0();

  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float* o = PERBLOCK ? fin[jn] : acc[jn];
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + cc]) = make_float2(o[j * 4 + 0], o[j * 4 + 1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + cc]) = make_float2(o[j * 4 + 2], o[j * 4 + 3]);
    }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
static CUtensorMap make_tmap(void* ptr, uint64_t K, uint64_t R, uint32_t boxK, uint32_t boxR) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {K, R};
  cuuint64_t strides[1] = {K};  // dim1（行）之间的字节步长
  cuuint32_t box[2] = {boxK, boxR};
  cuuint32_t es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr, dims, strides, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s ? s : "?");
    std::exit(1);
  }
  return tm;
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int N = (argc > 2) ? std::atoi(argv[2]) : 3072;
  int K = (argc > 3) ? std::atoi(argv[3]) : 7168;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nFP8 e4m3 TMA+wgmma GEMM: M=%d N=%d K=%d  FLOPs=%.2f GFLOP\n\n", M, N, K, flops / 1e9);

  const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  const int KBLK = K / 128;
  fp8 *A, *B;
  float *C, *sa, *sb;
  CUDA_CHECK(cudaMalloc(&A, aN));
  CUDA_CHECK(cudaMalloc(&B, bN));
  CUDA_CHECK(cudaMalloc(&C, cN * 4));
  CUDA_CHECK(cudaMalloc(&sa, (size_t)M * KBLK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sb, (size_t)(N / 128) * KBLK * sizeof(float)));

  std::vector<float> hAf(aN), hBf(bN);
  std::vector<fp8> hA(aN), hB(bN);
  std::vector<float> hC(cN);
  std::vector<float> hsca((size_t)M * KBLK), hscb((size_t)(N / 128) * KBLK);
  srand(1234);
  auto q = [](float x) { return fp8(x); };
  for (size_t i = 0; i < aN; ++i) { float x = 2.f * ((float)rand() / RAND_MAX - .5f); hAf[i] = (float)q(x); hA[i] = q(x); }
  for (size_t i = 0; i < bN; ++i) { float x = 2.f * ((float)rand() / RAND_MAX - .5f); hBf[i] = (float)q(x); hB[i] = q(x); }
  CUDA_CHECK(cudaMemcpy(A, hA.data(), aN, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), bN, cudaMemcpyHostToDevice));

  bool use_sc = false;
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
    std::printf("  [%-17s] max_abs_err=%.3e (ref~%.1f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
    return err / std::max(ref, 1.0) < 3e-2;
  };
  auto report = [&](const char* tag, double ms) {
    double tf = to_tflops(flops, ms);
    std::printf("%-17s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp8 peak)\n", tag, ms, tf,
                100.0 * tf / FP8_PEAK);
  };
  auto want = [&](const char* name) { return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0; };

#define RUN_TMA(NAME, BM, BN, ST, PB)                                                           \
  do {                                                                                          \
    if (want(NAME)) {                                                                           \
      CUtensorMap tmA = make_tmap(A, K, M, 128, BM);                                            \
      CUtensorMap tmB = make_tmap(B, K, N, 128, BN);                                            \
      auto fn = fp8_tma_ws_kernel<BM, BN, 128, ST, PB>;                                         \
      const int nt = ((BM) / 64) * 128 + 32;                                                    \
      const int BARR = ((int)(2 * (ST) * sizeof(uint64_t)) + 1023) / 1024 * 1024;               \
      const size_t shm = (size_t)BARR + (size_t)(ST) * ((BM) + (BN)) * 128;                     \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
      dim3 grid(div_up(N, BN), div_up(M, BM));                                                  \
      fn<<<grid, nt, shm>>>(tmA, tmB, C, M, N, K, sa, sb, KBLK);                                \
      CUDA_CHECK_LAST();                                                                        \
      check(NAME);                                                                              \
      double t = bench_ms([&] { fn<<<grid, nt, shm>>>(tmA, tmB, C, M, N, K, sa, sb, KBLK); }, 5, 30); \
      report(NAME, t);                                                                          \
    }                                                                                           \
  } while (0)

  std::printf("-- TMA + warp specialization, per-tensor --\n");
  RUN_TMA("tma128x128x128s3", 128, 128, 3, false);
  RUN_TMA("tma128x128x128s4", 128, 128, 4, false);
  RUN_TMA("tma128x128x128s5", 128, 128, 5, false);
  RUN_TMA("tma128x128x128s6", 128, 128, 6, false);
  RUN_TMA("tma128x256x128s3", 128, 256, 3, false);
  RUN_TMA("tma128x256x128s4", 128, 256, 4, false);
  RUN_TMA("tma256x128x128s2", 256, 128, 2, false);
  RUN_TMA("tma256x128x128s3", 256, 128, 3, false);
  RUN_TMA("tma256x128x128s4", 256, 128, 4, false);

  std::printf("\n-- TMA + warp specialization, per-block (sa 1x128, sb 128x128) --\n");
  for (auto& x : hsca) x = 0.5f + (float)rand() / RAND_MAX;
  for (auto& x : hscb) x = 0.5f + (float)rand() / RAND_MAX;
  CUDA_CHECK(cudaMemcpy(sa, hsca.data(), hsca.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sb, hscb.data(), hscb.size() * 4, cudaMemcpyHostToDevice));
  use_sc = true;
  RUN_TMA("pb128x128x128s3", 128, 128, 3, true);
  RUN_TMA("pb256x128x128s3", 256, 128, 3, true);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(sa));
  CUDA_CHECK(cudaFree(sb));
  return 0;
}
