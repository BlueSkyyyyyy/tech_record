// 37 FP8 per-block GEMM：把「折算」藏进 wgmma 的异步缝隙里
//
// 承接 24（fp8_gemm_pb.cu）：per-block 最佳 936.5 TFLOPS（47.3%），
// 而同二进制的 per-tensor 是 1209.6（61.2%）。根因有两层：
//   ① ptxas 把「用通用 FMA 写 wgmma 累加器」判为 C7514 → 主动串行化 wgmma；
//   ② 每 128-k 块要把 acc 折算进 fin（64 FMA/线程），wait0 后折算期间 tensor core 空闲，
//      折算 ≈ wgmma 时间的 23%（FP32 CUDA core 66.9 vs FP8 TC 1978 TFLOPS）。
//
// 本版做三件事（逐条实测）：
//   A. 用 wgmma 指令自身的 scale_d=0（ScaleOut::Zero）清零累加器，替代通用 FMA 清零
//      —— 消掉 C7514 的触发源（DeepGEMM 的 `WGMMA::wgmma(..., k)` 把 k 当 scale_d 就是这个意思）。
//   B. ping-pong 两个累加器：issue(kb) 后只 `wgmma.wait_group 1`，让 fold(kb-1)
//      与 kb 的 wgmma 在 tensor / CUDA core 上并行。
//   C. `setmaxnreg`（warpgroup_reg_alloc/dealloc）：2 个累加器 + fin = 192 regs/线程，
//      288 线程静态上限只有 227 且 ptxas 会 spill；显式给 math warpgroup 232、
//      producer 40，总 256*232+32*40=60672 < 65536，不 spill。
//
// 运行：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//     scripts/run.sh 37-fp8-pb-overlap/fp8_gemm_ws.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <type_traits>

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
// 空 asm + "+f"：告诉 ptxas 该寄存器会被非 wgmma 指令有意读写（= CUTLASS warpgroup_fence_operand）
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r) :: "memory"); }

// setmaxnreg：warp 级寄存器再分配（RegCount 必须是 8 的倍数且 ≥ 24）
template <int N>
__device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N>
__device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }

// scale_d = false 时 wgmma 用 ScaleOut::Zero 覆盖累加器（省掉通用 FMA 清零 → 不触发 C7514）
__device__ __forceinline__ void wgmma_m64n128k32(float (&d)[64], uint64_t da, uint64_t db, bool scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, %67, %68;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"((int)scale_d), "n"(1), "n"(1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
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
// mbarrier + TMA
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
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}

// ---------------------------------------------------------------------------
// per-block FP8 GEMM + warpgroup 寄存器再分配
//   MODE: 0 = 单累加器 + 每块 wait0 后折算
//         1 = ping-pong（wait_group<1>，fold 与 wgmma 重叠）
//   MATHREG / TMAREG：setmaxnreg 的初始值
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int MODE, int MATHREG, int TMAREG>
__global__ void __launch_bounds__((BM / 64) * 128 + 32, 1)
fp8_tma_pb_ws_kernel(const __grid_constant__ CUtensorMap tmA,
                     const __grid_constant__ CUtensorMap tmB,
                     float* __restrict__ C, int M, int N, int K,
                     const float* __restrict__ sa, const float* __restrict__ sb, int KBLK) {
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
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * BK;
  float* sfa = reinterpret_cast<float*>(Bs + (size_t)STAGES * BN * BK);  // [BM][KBLK]
  float* sfb = sfa + (size_t)BM * KBLK;                                  // [NSPLIT][KBLK]

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  // ---- 寄存器再分配：producer 让出、consumer 吃下 ----
  if constexpr (MATHREG > 0) {
    if (tid >= NCONS) reg_dealloc<TMAREG>();
    else reg_alloc<MATHREG>();
  }

  // ---- scale 预取 smem ----
  for (int idx = tid; idx < BM * KBLK; idx += NT) {
    const int r = idx / KBLK, kk = idx % KBLK;
    sfa[idx] = sa[(size_t)(block_row + r) * KBLK + kk];
  }
  for (int idx = tid; idx < NSPLIT * KBLK; idx += NT) {
    const int jn = idx / KBLK, kk = idx % KBLK;
    sfb[idx] = sb[(size_t)(((block_col + jn * 128) >> 7)) * KBLK + kk];
  }
  __syncthreads();

  // ---- producer warp ----
  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int kb = 0; kb < nblk; ++kb) {
        const int st = kb % STAGES;
        if (kb >= STAGES) { mbar_wait(empty + st, (uint32_t)((kb / STAGES - 1) & 1)); }
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * BK + BN * BK);
        tma_load_2d(&tmA, As + (size_t)st * BM * BK, kb * BK, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * BK, kb * BK, block_col, full + st);
      }
    }
    return;
  }

  // ---- consumer warpgroups ----
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  float acc[2][NSPLIT][64];
  float fin[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) fin[j][i] = 0.f;

  auto fold = [&](auto set_ic, int kb_fold) {
    constexpr int SET = set_ic.value;
    const float a0 = sfa[(size_t)row0 * KBLK + kb_fold];
    const float a1 = sfa[(size_t)(row0 + 8) * KBLK + kb_fold];
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      const float bv = sfb[(size_t)jn * KBLK + kb_fold];
      const float s0 = a0 * bv, s1 = a1 * bv;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        fin[jn][j * 4 + 0] += s0 * acc[SET][jn][j * 4 + 0];
        fin[jn][j * 4 + 1] += s0 * acc[SET][jn][j * 4 + 1];
        fin[jn][j * 4 + 2] += s1 * acc[SET][jn][j * 4 + 2];
        fin[jn][j * 4 + 3] += s1 * acc[SET][jn][j * 4 + 3];
      }
    }
  };

  auto issue = [&](auto set_ic, int kb, bool zero_first) {
    constexpr int SET = set_ic.value;
    const int st = kb % STAGES;
    mbar_wait(full + st, (uint32_t)((kb / STAGES) & 1));
    fence_proxy_async();
    char* a = As + (size_t)st * BM * BK + (size_t)wg * 64 * BK;
    char* b = Bs + (size_t)st * BN * BK;
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int i = 0; i < 64; ++i) fence_operand(acc[SET][jn][i]);
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * BK;
#pragma unroll
      for (int s = 0; s < BK / 32; ++s) {
        uint64_t da = make_desc_sw128(k32_addr(smem_u32(a), s), SBO);
        uint64_t db = make_desc_sw128(k32_addr(smem_u32(bj), s), SBO);
        wgmma_m64n128k32(acc[SET][jn], da, db, !(zero_first && s == 0));
      }
    }
    wgmma_commit();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int i = 0; i < 64; ++i) fence_operand(acc[SET][jn][i]);
  };
  constexpr auto S0 = std::integral_constant<int, 0>{};
  constexpr auto S1 = std::integral_constant<int, 1>{};

  if constexpr (MODE == 1) {
    int kb = 0;
    for (; kb + 1 < nblk; kb += 2) {
      issue(S0, kb, true);
      if (kb > 0) { wgmma_wait_group<1>(); mbar_arrive(empty + (kb - 1) % STAGES); fold(S1, kb - 1); }
      issue(S1, kb + 1, true);
      wgmma_wait_group<1>();
      mbar_arrive(empty + kb % STAGES);
      fold(S0, kb);
    }
    if (kb < nblk) {
      issue(S0, kb, true);
      if (kb > 0) { wgmma_wait_group<1>(); mbar_arrive(empty + (kb - 1) % STAGES); fold(S1, kb - 1); }
      wgmma_wait0();
      fold(S0, kb);
    } else {
      wgmma_wait0();
      fold(S1, nblk - 1);
    }
  } else {
    for (int kb = 0; kb < nblk; ++kb) {
      issue(S0, kb, true);
      wgmma_wait0();
      mbar_arrive(empty + kb % STAGES);
      fold(S0, kb);
    }
  }

#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + cc]) = make_float2(fin[jn][j * 4 + 0], fin[jn][j * 4 + 1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + cc]) = make_float2(fin[jn][j * 4 + 2], fin[jn][j * 4 + 3]);
    }
}

// ---------------------------------------------------------------------------
// host（与 24 同：真实 shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json）
// ---------------------------------------------------------------------------
static CUtensorMap make_tmap(void* ptr, uint64_t K, uint64_t R, uint32_t boxK, uint32_t boxR) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {K, R};
  cuuint64_t strides[1] = {K};
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
  std::printf("\nFP8 e4m3 per-block WS-reg GEMM: M=%d N=%d K=%d  FLOPs=%.2f GFLOP\n\n",
              M, N, K, flops / 1e9);

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
  for (auto& x : hsca) x = 0.5f + (float)rand() / RAND_MAX;
  for (auto& x : hscb) x = 0.5f + (float)rand() / RAND_MAX;
  CUDA_CHECK(cudaMemcpy(sa, hsca.data(), hsca.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sb, hscb.data(), hscb.size() * 4, cudaMemcpyHostToDevice));

  auto cpu_entry = [&](int r, int c) {
    double s = 0;
    for (int kb = 0; kb < KBLK; ++kb) {
      double p = 0;
      for (int k = kb * 128; k < kb * 128 + 128; ++k)
        p += (double)hAf[(size_t)r * K + k] * (double)hBf[(size_t)c * K + k];
      s += (double)hsca[(size_t)r * KBLK + kb] * hscb[(size_t)(c / 128) * KBLK + kb] * p;
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
    std::printf("  [%-22s] max_abs_err=%.3e (ref~%.1f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
    return err / std::max(ref, 1.0) < 3e-2;
  };
  auto report = [&](const char* tag, double ms) {
    double tf = to_tflops(flops, ms);
    std::printf("%-22s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp8 peak)\n", tag, ms, tf, 100.0 * tf / FP8_PEAK);
  };
  auto want = [&](const char* name) { return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0; };

#define RUN_WS(NAME, BM, BN, ST, MODE, MR, TR)                                                     \
  do {                                                                                             \
    if (want(NAME)) {                                                                              \
      CUtensorMap tmA = make_tmap(A, K, M, 128, BM);                                               \
      CUtensorMap tmB = make_tmap(B, K, N, 128, BN);                                               \
      auto fn = fp8_tma_pb_ws_kernel<BM, BN, 128, ST, MODE, MR, TR>;                               \
      const int nt = ((BM) / 64) * 128 + 32;                                                       \
      const int BARR = ((int)(2 * (ST) * sizeof(uint64_t)) + 1023) / 1024 * 1024;                  \
      size_t shm = (size_t)BARR + (size_t)(ST) * ((BM) + (BN)) * 128 +                             \
                   ((size_t)(BM) * KBLK + (size_t)((BN) / 128) * KBLK) * sizeof(float);            \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
      dim3 grid(div_up(N, BN), div_up(M, BM));                                                     \
      fn<<<grid, nt, shm>>>(tmA, tmB, C, M, N, K, sa, sb, KBLK);                                   \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      double t = bench_ms([&] { fn<<<grid, nt, shm>>>(tmA, tmB, C, M, N, K, sa, sb, KBLK); }, 10, 50); \
      report(NAME, t);                                                                             \
    }                                                                                              \
  } while (0)

  std::printf("-- A. 单累加器 + wait0（stages）--\n");
  RUN_WS("s0_128x128s3_off", 128, 128, 3, 0, 0, 0);
  RUN_WS("s0_128x128s4_off", 128, 128, 4, 0, 0, 0);
  RUN_WS("s0_128x128s4_reg", 128, 128, 4, 0, 232, 40);

  std::printf("\n-- B. ping-pong（fold 与 wgmma 重叠）+ setmaxnreg --\n");
  RUN_WS("pp_128x128s3_reg", 128, 128, 3, 1, 232, 40);
  RUN_WS("pp_128x128s4_reg", 128, 128, 4, 1, 232, 40);
  RUN_WS("pp_128x128s5_reg", 128, 128, 5, 1, 232, 40);
  RUN_WS("pp_128x128s6_reg", 128, 128, 6, 1, 232, 40);
  RUN_WS("pp_128x128s4_off", 128, 128, 4, 1, 0, 0);

  std::printf("\n-- C. BN=256 --\n");
  RUN_WS("s0_128x256s3_reg", 128, 256, 3, 0, 232, 40);
  RUN_WS("pp_128x256s3_reg", 128, 256, 3, 1, 232, 40);
  RUN_WS("pp_128x256s4_reg", 128, 256, 4, 1, 232, 40);

  std::printf("\n-- D. BM=64（单 warpgroup）--\n");
  RUN_WS("s0_64x128s4_reg", 64, 128, 4, 0, 240, 40);
  RUN_WS("pp_64x128s4_reg", 64, 128, 4, 1, 240, 40);
  RUN_WS("pp_64x128s5_reg", 64, 128, 5, 1, 240, 40);
  RUN_WS("pp_64x128s6_reg", 64, 128, 6, 1, 240, 40);
  RUN_WS("pp_64x256s4_reg", 64, 256, 4, 1, 240, 40);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(sa));
  CUDA_CHECK(cudaFree(sb));
  return 0;
}
