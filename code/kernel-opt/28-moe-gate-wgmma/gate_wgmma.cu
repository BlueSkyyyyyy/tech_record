// 28 MoE（一·续）：gate GEMM 的 wgmma + TMA + warp specialization 版
//
// 承接 27（moe_router.cu）：DeepSeek-V4-Pro 的 router 前门 gate GEMM
//   logits[M, E] = X[M, H] @ Wg[E, H]^T
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：hidden=7168, n_routed=384。
// 27 篇用 mma.sync + ldmatrix + cp.async 双缓冲只做到 204~257 TFLOPS（cuBLAS 的 ~28%），
// ncu 显示瓶颈是 No Eligible 57.6% / L2 60.7%。本版两步走：
//   (a) gate_async_kernel : wgmma SS + SW128 swizzle + cp.async 多级流水（复用 20/22）
//   (b) gate_ws_kernel    : 再加 TMA(cp.async.bulk.tensor.2d) + mbarrier + warp specialization
//       （复用 23），把装载与计算解耦。
//
// 关键点：wgmma 的 B 必须 K-major，权重天然是 Wg[E, H]（H 连续），直接可用，无需转置。
// bf16 的 SW128 atom 是 8 行 × 64 元素（128B），所以 BK 固定 64（TMA box 内维 128B）。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 28-moe-gate-wgmma/gate_wgmma.cu [M] [which]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;
constexpr double BF16_PEAK = 989.0;

// DeepSeek-V4-Pro
constexpr int H = 7168;    // K = hidden
constexpr int E = 384;     // N = n_routed_experts

// ---------------------------------------------------------------------------
// wgmma helpers（同 20/23）—— bf16 m64n128k16
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

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

// bf16 K-major SW128：8 行 × 64 元素（128B）atom，16B 列做 c' = c ^ r。
// tile 宽（沿 K）固定为 64，故 SBO（相邻 8 行组字节距离）= 1024。
__device__ __forceinline__ int sw128_off(int row, int k) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * 1 + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;  // LBO = 1 (16B)，B128 下硬件忽略
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
__device__ __forceinline__ uint32_t k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

// ---------------------------------------------------------------------------
// mbarrier + TMA helpers（同 23）
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
// 2D TMA load（byte 语义，内维 128B + SW128 swizzle）
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}

// ---------------------------------------------------------------------------
// (a) wgmma SS + cp.async 多级流水（无 warp specialization）
//     所有 BM/64 个 warpgroup 既搬又算。BK 固定 64。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES>
__global__ void __launch_bounds__((BM / 64) * 128)
gate_async_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                  float* __restrict__ C, int M, int N, int K) {
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NT = NWG * 128;
  constexpr int SBO = (BK / 64) * 1024;
  extern __shared__ __align__(1024) char smem[];
  char* As = smem;                          // [STAGES][BM][BK] SW128
  char* Bs = As + (size_t)STAGES * BM * 128;  // [STAGES][BN][BK]

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

  auto do_prefetch = [&](int kb, int stage) {
    char* a = As + (size_t)stage * BM * 128;
    char* b = Bs + (size_t)stage * BN * 128;
    for (int i = tid; i < BM * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      const int gr = block_row + r, gc = kb * BK + c8;
      char* dst = a + sw128_off(r, c8);
      if (gr < M) __pipeline_memcpy_async(dst, &A[(size_t)gr * K + gc], 16);
      else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < BN * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      const int ge = block_col + r, gk = kb * BK + c8;  // B = Wg[E, H]，行=E，列=K
      char* dst = b + sw128_off(r, c8);
      if (ge < N && gk < K) __pipeline_memcpy_async(dst, &B[(size_t)ge * K + gk], 16);
      else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
    __pipeline_commit();
  };

  // prologue
  for (int s = 0; s < STAGES - 1 && s < nblk; ++s) do_prefetch(s, s);
  if (nblk < STAGES - 1) __pipeline_commit();

  for (int kb = 0; kb < nblk; ++kb) {
    const int stage = kb % STAGES;
    const int nxt = kb + STAGES - 1;
    if (nxt < nblk) do_prefetch(nxt, nxt % STAGES);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();
    char* a = As + (size_t)stage * BM * 128 + (size_t)wg * 64 * 128;
    char* b = Bs + (size_t)stage * BN * 128;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * 128;
#pragma unroll
      for (int s = 0; s < BK / 16; ++s) {
        uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), s), SBO);
        uint64_t db = make_desc_sw128(k16_addr(smem_u32(bj), s), SBO);
        wgmma_m64n128k16(acc[jn], da, db);
      }
    }
    wgmma_commit();
    wgmma_wait0();
    __syncthreads();
  }

  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + cc]) = make_float2(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + cc]) = make_float2(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
    }
}

// ---------------------------------------------------------------------------
// (b) wgmma SS + TMA + mbarrier + warp specialization
//     1 个 producer warp 专职发 TMA；(BM/64) 个 consumer warpgroup 只做 wgmma。
//     BK 固定 64（TMA SW128 内维 128 字节 = 64 bf16）。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int KSPLIT = 1>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
gate_ws_kernel(const __grid_constant__ CUtensorMap tmA,
               const __grid_constant__ CUtensorMap tmB,
               float* __restrict__ C, int M, int N, int K) {
  static_assert(BK == 64, "bf16 SW128 内维固定 64 元素");
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NCONS = NWG * 128;
  constexpr int SBO = 1024;

  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;                          // [STAGES][BM][64] SW128
  char* Bs = As + (size_t)STAGES * BM * 128;       // [STAGES][BN][64] SW128

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk_all = K / BK;
  const int ks = blockIdx.z;                       // split-K 序号
  const int kb0 = ks * (nblk_all / KSPLIT);
  const int nk = nblk_all / KSPLIT;                // 本 CTA 的 k 块数

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int q = 0; q < nk; ++q) {
        const int st = q % STAGES;
        if (q >= STAGES) mbar_wait(empty + st, (uint32_t)((q / STAGES - 1) & 1));
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * 128 + BN * 128);
        tma_load_2d(&tmA, As + (size_t)st * BM * 128, (kb0 + q) * 128, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * 128, (kb0 + q) * 128, block_col, full + st);
      }
    }
    return;
  }

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

  for (int q = 0; q < nk; ++q) {
    const int st = q % STAGES;
    mbar_wait(full + st, (uint32_t)((q / STAGES) & 1));
    fence_proxy_async();
    char* a = As + (size_t)st * BM * 128 + (size_t)wg * 64 * 128;
    char* b = Bs + (size_t)st * BN * 128;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * 128;
#pragma unroll
      for (int s = 0; s < BK / 16; ++s) {
        uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), s), SBO);
        uint64_t db = make_desc_sw128(k16_addr(smem_u32(bj), s), SBO);
        wgmma_m64n128k16(acc[jn], da, db);
      }
    }
    wgmma_commit();
    if (q >= STAGES - 2) {
      wgmma_wait_group<STAGES - 2>();
      const int rs = (q - (STAGES - 2)) % STAGES;
      mbar_arrive(empty + rs);
    }
  }
  wgmma_wait0();

  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if (KSPLIT == 1) {
        if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + cc]) = make_float2(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + cc]) = make_float2(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      } else {
        if (r0 < M) { atomicAdd(&C[(size_t)r0 * N + cc], acc[jn][j * 4 + 0]); atomicAdd(&C[(size_t)r0 * N + cc + 1], acc[jn][j * 4 + 1]); }
        if (r1 < M) { atomicAdd(&C[(size_t)r1 * N + cc], acc[jn][j * 4 + 2]); atomicAdd(&C[(size_t)r1 * N + cc + 1], acc[jn][j * 4 + 3]); }
      }
    }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
static CUtensorMap make_tmap(const void* ptr, uint64_t rowbytes, uint64_t rows, uint32_t boxR) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {rowbytes, rows};
  cuuint64_t strides[1] = {rowbytes};
  cuuint32_t box[2] = {128, boxR};
  cuuint32_t es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
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
  int M = (argc > 1) ? std::atoi(argv[1]) : 16384;
  const char* which = (argc > 2) ? argv[2] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * E * H;
  std::printf("\nMoE gate GEMM (wgmma): M=%d N=%d K=%d  FLOPs=%.2f GFLOP\n\n", M, E, H, flops / 1e9);

  const size_t aN = (size_t)M * H, bN = (size_t)E * H, cN = (size_t)M * E;
  bf16 *X, *Wg;
  float* C;
  CUDA_CHECK(cudaMalloc(&X, aN * 2));
  CUDA_CHECK(cudaMalloc(&Wg, bN * 2));
  CUDA_CHECK(cudaMalloc(&C, cN * 4));

  std::vector<bf16> hX(aN), hWg(bN);
  std::vector<float> hC(cN);
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : hX) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hWg) v = __float2bfloat16(0.05f * nd(rng));
  CUDA_CHECK(cudaMemcpy(X, hX.data(), aN * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Wg, hWg.data(), bN * 2, cudaMemcpyHostToDevice));

  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), C, cN * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int s = 0; s < 24; ++s) {
      const int r = (s * 1009 + 13) % M, c = (s * 131 + 7) % E;
      double acc = 0;
      for (int k = 0; k < H; ++k)
        acc += (double)__bfloat162float(hX[(size_t)r * H + k]) *
               (double)__bfloat162float(hWg[(size_t)c * H + k]);
      err = std::max(err, std::fabs((double)hC[(size_t)r * E + c] - acc));
      ref = std::max(ref, std::fabs(acc));
    }
    std::printf("  [%-18s] max_abs_err=%.3e (ref~%.2f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 2e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms) {
    double tf = to_tflops(flops, ms);
    std::printf("%-18s %8.4f ms  %8.2f TFLOPS  (%5.2f%% of bf16 peak)\n", tag, ms, tf,
                100.0 * tf / BF16_PEAK);
  };
  auto want = [&](const char* name) { return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0; };

#define RUN_ASYNC(NAME, BM, BN, ST)                                                              \
  do {                                                                                           \
    if (want(NAME)) {                                                                            \
      auto fn = gate_async_kernel<BM, BN, 64, ST>;                                                \
      const int nt = ((BM) / 64) * 128;                                                          \
      const size_t shm = (size_t)(ST) * ((BM) + (BN)) * 128;                                     \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));\
      dim3 grid((E) / (BN), div_up(M, BM));                                                      \
      fn<<<grid, nt, shm>>>(X, Wg, C, M, E, H);                                                  \
      CUDA_CHECK_LAST();                                                                          \
      check(NAME);                                                                                \
      double t = bench_ms([&] { fn<<<grid, nt, shm>>>(X, Wg, C, M, E, H); }, 5, 50);             \
      report(NAME, t);                                                                            \
    }                                                                                             \
  } while (0)

#define RUN_WS(NAME, BM, BN, ST, KS)                                                             \
  do {                                                                                           \
    if (want(NAME)) {                                                                            \
      CUtensorMap tmA = make_tmap(X, (uint64_t)H * 2, M > BM ? M : BM, BM);                     \
      CUtensorMap tmB = make_tmap(Wg, (uint64_t)H * 2, E, BN);                                   \
      auto fn = gate_ws_kernel<BM, BN, 64, ST, KS>;                                              \
      const int nt = ((BM) / 64) * 128 + 32;                                                     \
      const int BARR = ((int)(2 * (ST) * sizeof(uint64_t)) + 1023) / 1024 * 1024;                \
      const size_t shm = (size_t)BARR + (size_t)(ST) * ((BM) + (BN)) * 128;                      \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));\
      dim3 grid((E) / (BN), div_up(M, BM), KS);                                                  \
      auto run = [&] {                                                                           \
        if (KS > 1) CUDA_CHECK(cudaMemsetAsync(C, 0, cN * 4));                                    \
        fn<<<grid, nt, shm>>>(tmA, tmB, C, M, E, H);                                             \
      };                                                                                          \
      run(); CUDA_CHECK_LAST();                                                                   \
      check(NAME);                                                                                \
      double t = bench_ms(run, 5, 50);                                                           \
      report(NAME, t);                                                                            \
    }                                                                                             \
  } while (0)

  std::printf("-- (a) wgmma SS + cp.async, SW128 --\n");
  RUN_ASYNC("async128x128s3", 128, 128, 3);
  RUN_ASYNC("async128x128s4", 128, 128, 4);
  RUN_ASYNC("async64x128s4", 64, 128, 4);
  RUN_ASYNC("async256x128s3", 256, 128, 3);

  std::printf("\n-- (b) wgmma SS + TMA + warp specialization, SW128 --\n");
  RUN_WS("ws64x128s2", 64, 128, 2, 1);
  RUN_WS("ws64x128s3", 64, 128, 3, 1);
  RUN_WS("ws64x128s4", 64, 128, 4, 1);
  RUN_WS("ws128x128s2", 128, 128, 2, 1);
  RUN_WS("ws128x128s3", 128, 128, 3, 1);
  RUN_WS("ws128x128s4", 128, 128, 4, 1);
  RUN_WS("ws128x128s5", 128, 128, 5, 1);
  RUN_WS("ws256x128s3", 256, 128, 3, 1);
  RUN_WS("ws256x128s4", 256, 128, 4, 1);

  std::printf("\n-- (c) split-K（攻 1.45 wave 的尾效应）--\n");
  RUN_WS("ws128x128s3k2", 128, 128, 3, 2);
  RUN_WS("ws128x128s3k4", 128, 128, 3, 4);
  RUN_WS("ws128x128s2k2", 128, 128, 2, 2);
  RUN_WS("ws64x128s3k2", 64, 128, 3, 2);

  CUDA_CHECK(cudaFree(X));
  CUDA_CHECK(cudaFree(Wg));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
