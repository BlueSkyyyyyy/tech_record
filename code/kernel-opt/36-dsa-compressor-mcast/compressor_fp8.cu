// 36b DSA compressor 投影 FP8 化（e4m3 + per-block 缩放）
//
// 背景：35 篇的合并投影（X[M,7168] @ Wm[2C,7168]^T）用 bf16 跑到 ~650 TFLOPS，ncu 显示
// 最佳配置是 tensor-pipe 受限（76%），而不是 35 诊断的 L2 受限。既然 bf16 已近本设计天花板，
// 就把投影按 DeepSeek-V4-Pro 真实的量化方案 **FP8 e4m3 + weight_block 128×128** 做一遍：
//   /ssd/models/DeepSeek-V4-Pro/config.json:
//     quantization_config = {quant_method: fp8, fmt: e4m3, activation_scheme: dynamic,
//                            weight_block_size: [128,128], scale_fmt: ue8m0}
//
// 复用 24 篇的 `fp8_tma_pb_kernel`（TMA + warp specialization + wgmma.m64n128k32 + 每 128-k
// 块折算 `fin += sa*sb*acc`）。shape 同 35：ratio=128 → N=1024；ratio=4 → N=2048，K=D=7168。
// 因为 N 小、M 大（tall-skinny），本文件同时跑 bf16 与 FP8 两条路径，同进程对照。
//
// 运行：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//     scripts/run.sh 36-dsa-compressor-mcast/compressor_fp8.cu [ratio] [M] [which]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;
constexpr double BF16_PEAK = 989.0;
constexpr double FP8_PEAK = 1978.0;

constexpr int D  = 7168;  // hidden_size (K)
constexpr int HD = 512;   // head_dim
constexpr int RD = 64;    // qk_rope_head_dim

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

#define WGMMA_M64N128K16_BODY(KM, AOP, BOP, TAIL)                                           \
  "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"                                                    \
  "wgmma.mma_async.sync.aligned.m64n128" KM ".f32." AOP "." BOP " "                      \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n" \
  "%64, %65, p, " TAIL ";\n}\n"

__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(WGMMA_M64N128K16_BODY("k16", "bf16", "bf16", "1, 1, 0, 0")
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ void wgmma_m64n128k32_e4m3(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(WGMMA_M64N128K16_BODY("k32", "e4m3", "e4m3", "%67, %68")
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
}

// K-major SW128（bf16 一行 64 元素 / fp8 一行 128 元素，逐字节同构）
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ uint32_t k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint32_t k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

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
      "{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
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
// (1a) bf16 合并投影（同 35/36 的 proj_ws_kernel，去掉 multicast）
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool BF16OUT = false>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
proj_bf16_kernel(const __grid_constant__ CUtensorMap tmA,
                 const __grid_constant__ CUtensorMap tmB,
                 void* __restrict__ C, int M, int N, int K) {
  constexpr int NWG = BM / 64, NSPLIT = BN / 128, NCONS = NWG * 128, SBO = 1024;
  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * 128;

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int q = 0; q < nblk; ++q) {
        const int st = q % STAGES;
        if (q >= STAGES) mbar_wait(empty + st, (uint32_t)((q / STAGES - 1) & 1));
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * 128 + BN * 128);
        tma_load_2d(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st);
      }
    }
    return;
  }
  const int wg = tid >> 7, lane = tid & 31, warp = tid >> 5;
  float acc[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;
  for (int q = 0; q < nblk; ++q) {
    const int st = q % STAGES;
    mbar_wait(full + st, (uint32_t)((q / STAGES) & 1));
    char* a = As + (size_t)st * BM * 128 + (size_t)wg * 64 * 128;
    char* b = Bs + (size_t)st * BN * 128;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * 128;
#pragma unroll
      for (int s = 0; s < BK / 16; ++s)
        wgmma_m64n128k16(acc[jn], make_desc_sw128(k16_addr(smem_u32(a), s), SBO),
                         make_desc_sw128(k16_addr(smem_u32(bj), s), SBO));
    }
    wgmma_commit();
    if (q >= STAGES - 2) {
      wgmma_wait_group<STAGES - 2>();
      mbar_arrive(empty + (q - (STAGES - 2)) % STAGES);
    }
  }
  wgmma_wait0();
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  bf16* Cb = reinterpret_cast<bf16*>(C);
  float* Cf = reinterpret_cast<float*>(C);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if constexpr (BF16OUT) {
        if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r0 * N + cc]) =
            __floats2bfloat162_rn(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r1 * N + cc]) =
            __floats2bfloat162_rn(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      } else {
        if (r0 < M) *reinterpret_cast<float2*>(&Cf[(size_t)r0 * N + cc]) = make_float2(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&Cf[(size_t)r1 * N + cc]) = make_float2(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      }
    }
}

// ---------------------------------------------------------------------------
// (1b) FP8 e4m3 合并投影（per-block 128×128 权重 + 1×128 激活），同 24 的 fp8_tma_pb_kernel
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
proj_fp8_kernel(const __grid_constant__ CUtensorMap tmA,
                const __grid_constant__ CUtensorMap tmB,
                float* __restrict__ C, int M, int N, int K,
                const float* __restrict__ sa, const float* __restrict__ sb, int KBLK) {
  static_assert(BK == 128, "TMA SW128 内维固定 128 字节(fd8)");
  constexpr int NWG = BM / 64, NSPLIT = BN / 128, NCONS = NWG * 128, SBO = 1024;
  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * BK;

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int kb = 0; kb < nblk; ++kb) {
        const int st = kb % STAGES;
        if (kb >= STAGES) mbar_wait(empty + st, (uint32_t)((kb / STAGES - 1) & 1));
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * BK + BN * BK);
        tma_load_2d(&tmA, As + (size_t)st * BM * BK, kb * BK, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * BK, kb * BK, block_col, full + st);
      }
    }
    return;
  }
  const int wg = tid >> 7, lane = tid & 31, warp = tid >> 5;
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  float acc[NSPLIT][64];
  float fin[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc[j][i] = 0.f; fin[j][i] = 0.f; }

  for (int kb = 0; kb < nblk; ++kb) {
    const int st = kb % STAGES;
    mbar_wait(full + st, (uint32_t)((kb / STAGES) & 1));
    char* a = As + (size_t)st * BM * BK + (size_t)wg * 64 * BK;
    char* b = Bs + (size_t)st * BN * BK;
    fence_proxy_async();
#pragma unroll
    for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
      for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * BK;
#pragma unroll
      for (int s = 0; s < BK / 32; ++s)
        wgmma_m64n128k32_e4m3(acc[jn], make_desc_sw128(k32_addr(smem_u32(a), s), SBO),
                              make_desc_sw128(k32_addr(smem_u32(bj), s), SBO));
    }
    wgmma_commit();
    wgmma_wait0();
    mbar_arrive(empty + st);
    // 折算：a0/a1 是相邻两行（row0, row0+8）的 1×128 激活 scale；sb 每个 128×128 权重块一个
    const float a0 = sa[(size_t)(block_row + row0) * KBLK + kb];
    const float a1 = sa[(size_t)(block_row + row0 + 8) * KBLK + kb];
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      const float bv = sb[(size_t)((block_col + jn * 128) >> 7) * KBLK + kb];
      const float s0 = a0 * bv, s1 = a1 * bv;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        fin[jn][j * 4 + 0] += s0 * acc[jn][j * 4 + 0];
        fin[jn][j * 4 + 1] += s0 * acc[jn][j * 4 + 1];
        fin[jn][j * 4 + 2] += s1 * acc[jn][j * 4 + 2];
        fin[jn][j * 4 + 3] += s1 * acc[jn][j * 4 + 3];
      }
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
// (2) 池化 + RMSNorm + RoPE（同 35）
// ---------------------------------------------------------------------------
template <int RATIO>
__global__ void __launch_bounds__(HD)
pool_kernel(const float* __restrict__ Y, const float* __restrict__ ape, const float* __restrict__ nw,
            const float* __restrict__ cos_t, const float* __restrict__ sin_t,
            bf16* __restrict__ out, int M, int N, int C) {
  const int t = blockIdx.x, j = threadIdx.x, tok0 = t * RATIO;
  float m = -1e30f, l = 0.f, acc = 0.f;
  auto add_slot = [&](float kv_v, float sc_v) {
    const float mn = fmaxf(m, sc_v), a = __expf(m - mn), p = __expf(sc_v - mn);
    l = l * a + p; acc = acc * a + p * kv_v; m = mn;
  };
  if constexpr (RATIO == 128) {
#pragma unroll 4
    for (int i = 0; i < RATIO; ++i) {
      const size_t base = (size_t)(tok0 + i) * N;
      add_slot(Y[base + j], Y[base + C + j] + ape[i * C + j]);
    }
  } else {
    if (t > 0)
#pragma unroll
      for (int i = 0; i < RATIO; ++i) {
        const size_t base = (size_t)(tok0 - RATIO + i) * N;
        add_slot(Y[base + j], Y[base + C + j] + ape[i * C + j]);
      }
#pragma unroll
    for (int i = 0; i < RATIO; ++i) {
      const size_t base = (size_t)(tok0 + i) * N;
      add_slot(Y[base + HD + j], Y[base + C + HD + j] + ape[i * C + HD + j]);
    }
  }
  const float v0 = acc / l;
  __shared__ float sred[HD / 32];
  float ss = v0 * v0;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) sred[warp] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float s = (threadIdx.x < HD / 32) ? sred[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) sred[0] = s;
  }
  __syncthreads();
  float v = v0 * rsqrtf(sred[0] / HD + 1e-6f) * nw[j];
  __shared__ float srow[HD];
  srow[j] = v;
  __syncthreads();
  if (j >= HD - RD) {
    const int p = (j - (HD - RD)) >> 1, par = (j - (HD - RD)) & 1;
    const float a0 = srow[HD - RD + 2 * p], a1 = srow[HD - RD + 2 * p + 1];
    const int ft = t * (RD / 2) + p;
    const float c = cos_t[ft], s = sin_t[ft];
    v = par == 0 ? (a0 * c - a1 * s) : (a0 * s + a1 * c);
  }
  out[(size_t)t * HD + j] = __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// host：tensor map / yarn / 参考
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
  if (r != CUDA_SUCCESS) { std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n"); std::exit(1); }
  return tm;
}

static double yarn_freq(int m, int rope_dim, double base, double factor,
                        double original_seq_len, int beta_fast, int beta_slow) {
  auto fdim = [&](double num_rot) {
    return rope_dim * std::log(original_seq_len / (num_rot * 2 * M_PI)) / (2 * std::log(base));
  };
  double lo = std::max(0.0, std::floor(fdim(beta_fast))), hi = std::min((double)rope_dim - 1, std::ceil(fdim(beta_slow)));
  double low = lo, high = hi;
  if (low == high) high += 0.001;
  double ramp = std::min(1.0, std::max(0.0, (m - low) / (high - low)));
  double smooth = 1.0 - ramp;
  double f = 1.0 / std::pow(base, (double)(2 * m) / rope_dim);
  return f / factor * (1.0 - smooth) + f * smooth;
}

// 完整 CPU 参考：x/Wm 已是「反量化后」（bf16 或 fp8 反量化）的值
static void cpu_ref(const std::vector<float>& x, const std::vector<float>& Wm,
                    const std::vector<float>& ape, const std::vector<float>& nw,
                    int M, int N, int C, int RATIO, std::vector<float>& out) {
  const int W = M / RATIO;
  out.assign((size_t)W * HD, 0.f);
  std::vector<float> srow(HD);
  for (int t = 0; t < W; ++t) {
    for (int j = 0; j < HD; ++j) {
      float m = -1e30f, l = 0.f, acc = 0.f;
      auto add = [&](double kv, double sc) {
        double mn = std::max((double)m, sc), a = std::exp((double)m - mn), p = std::exp(sc - mn);
        l = (float)(l * a + p); acc = (float)(acc * a + p * kv); m = (float)mn;
      };
      auto dot = [&](int tok, int row) {
        double s = 0;
        for (int k = 0; k < D; ++k) s += (double)x[(size_t)tok * D + k] * Wm[(size_t)row * D + k];
        return s;
      };
      if (RATIO == 128) {
        for (int i = 0; i < RATIO; ++i) {
          int tok = t * RATIO + i;
          add(dot(tok, j), dot(tok, C + j) + ape[(size_t)i * C + j]);
        }
      } else {
        if (t > 0) for (int i = 0; i < RATIO; ++i) {
          int tok = (t - 1) * RATIO + i;
          add(dot(tok, j), dot(tok, C + j) + ape[(size_t)i * C + j]);
        }
        for (int i = 0; i < RATIO; ++i) {
          int tok = t * RATIO + i;
          add(dot(tok, HD + j), dot(tok, C + HD + j) + ape[(size_t)i * C + HD + j]);
        }
      }
      srow[j] = acc / l;
    }
    double ss = 0;
    for (int j = 0; j < HD; ++j) ss += (double)srow[j] * srow[j];
    float rstd = (float)(1.0 / std::sqrt(ss / HD + 1e-6));
    for (int j = 0; j < HD; ++j) srow[j] = srow[j] * rstd * nw[j];
    for (int p = 0; p < RD / 2; ++p) {
      float a0 = srow[HD - RD + 2 * p], a1 = srow[HD - RD + 2 * p + 1];
      double ang = (double)t * yarn_freq(p, RD, 160000.0, 16.0, 65536.0, 32, 1);
      float c = (float)std::cos(ang), s = (float)std::sin(ang);
      srow[HD - RD + 2 * p] = a0 * c - a1 * s;
      srow[HD - RD + 2 * p + 1] = a0 * s + a1 * c;
    }
    for (int j = 0; j < HD; ++j) out[(size_t)t * HD + j] = srow[j];
  }
}

// per-1×128（激活）/ 128×128（权重）动态量化：scale=amax/448（e4m3 上限），返回反量化值
static void quant_perblock(const std::vector<bf16>& src, int R, int C_,
                           std::vector<fp8>& dst, std::vector<float>& sc, int rowblk) {
  dst.resize((size_t)R * C_);
  const int CB = C_ / 128;
  sc.assign((size_t)(R / rowblk) * CB, 1.f);
  for (int rb = 0; rb < R / rowblk; ++rb)
    for (int cb = 0; cb < CB; ++cb) {
      float amax = 0.f;
      for (int r = rb * rowblk; r < (rb + 1) * rowblk; ++r)
        for (int k = cb * 128; k < cb * 128 + 128; ++k)
          amax = std::max(amax, std::fabs(__bfloat162float(src[(size_t)r * C_ + k])));
      float s = amax > 0.f ? amax / 448.f : 1.f;
      sc[(size_t)rb * CB + cb] = s;
      for (int r = rb * rowblk; r < (rb + 1) * rowblk; ++r)
        for (int k = cb * 128; k < cb * 128 + 128; ++k)
          dst[(size_t)r * C_ + k] = fp8(__bfloat162float(src[(size_t)r * C_ + k]) / s);
    }
}

int main(int argc, char** argv) {
  int RATIO = (argc > 1) ? std::atoi(argv[1]) : 128;
  int M     = (argc > 2) ? std::atoi(argv[2]) : 32768;
  const char* which = (argc > 3) ? argv[3] : "all";
  if (RATIO != 128 && RATIO != 4) { std::fprintf(stderr, "ratio must be 128 or 4\n"); return 1; }
  const int COFF = (RATIO == 4) ? 2 : 1;
  const int C = COFF * HD, N = 2 * C;
  const int KBLK = D / 128;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * D;
  std::printf("\nDSA Compressor proj FP8 (ratio=%d): M=%d  Y[M,%d]=X[M,%d]@Wm[%d,%d]^T\n",
              RATIO, M, N, D, N, D);
  std::printf("proj FLOPs=%.2f GFLOP  Wm(bf16)=%.1f MB  out=[%d, %d]\n\n", flops / 1e9,
              (double)N * D * 2 / 1e6, M / RATIO, HD);

  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> hX((size_t)M * D), hWm((size_t)N * D);
  std::vector<float> hApe((size_t)RATIO * C), hNw(HD);
  for (auto& v : hX) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hWm) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hApe) v = 0.25f * nd(rng);
  for (auto& v : hNw) v = 1.f + 0.1f * nd(rng);

  bf16 *dX, *dWm, *dOutb, *dOutf;
  float *dYb, *dYf, *dApe, *dNw, *dsa, *dsb;
  fp8 *dXq, *dWq;
  CUDA_CHECK(cudaMalloc(&dX, (size_t)M * D * 2));
  CUDA_CHECK(cudaMalloc(&dWm, (size_t)N * D * 2));
  CUDA_CHECK(cudaMalloc(&dYb, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&dYf, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&dXq, (size_t)M * D));
  CUDA_CHECK(cudaMalloc(&dWq, (size_t)N * D));
  CUDA_CHECK(cudaMalloc(&dsa, (size_t)M * KBLK * 4));
  CUDA_CHECK(cudaMalloc(&dsb, (size_t)(N / 128) * KBLK * 4));
  CUDA_CHECK(cudaMalloc(&dApe, (size_t)RATIO * C * 4));
  CUDA_CHECK(cudaMalloc(&dNw, HD * 4));
  CUDA_CHECK(cudaMalloc(&dOutb, (size_t)(M / RATIO) * HD * 2));
  CUDA_CHECK(cudaMalloc(&dOutf, (size_t)(M / RATIO) * HD * 2));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), (size_t)M * D * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWm, hWm.data(), (size_t)N * D * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dApe, hApe.data(), (size_t)RATIO * C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dNw, hNw.data(), HD * 4, cudaMemcpyHostToDevice));

  // FP8 量化（1×128 激活 / 128×128 权重）
  std::vector<fp8> hXq, hWq;
  std::vector<float> hsa, hsb;
  quant_perblock(hX, M, D, hXq, hsa, 1);
  quant_perblock(hWm, N, D, hWq, hsb, 128);
  CUDA_CHECK(cudaMemcpy(dXq, hXq.data(), hXq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), hWq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsa, hsa.data(), hsa.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsb, hsb.data(), hsb.size() * 4, cudaMemcpyHostToDevice));

  const int NWIN = M / RATIO;
  std::vector<float> hcos((size_t)NWIN * (RD / 2)), hsin((size_t)NWIN * (RD / 2));
  for (int t = 0; t < NWIN; ++t)
    for (int p = 0; p < RD / 2; ++p) {
      double ang = (double)t * yarn_freq(p, RD, 160000.0, 16.0, 65536.0, 32, 1);
      hcos[(size_t)t * (RD / 2) + p] = (float)std::cos(ang);
      hsin[(size_t)t * (RD / 2) + p] = (float)std::sin(ang);
    }
  float *dcos, *dsin;
  CUDA_CHECK(cudaMalloc(&dcos, hcos.size() * 4));
  CUDA_CHECK(cudaMalloc(&dsin, hsin.size() * 4));
  CUDA_CHECK(cudaMemcpy(dcos, hcos.data(), hcos.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsin, hsin.data(), hsin.size() * 4, cudaMemcpyHostToDevice));

  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };

  // ---- bf16 路径 ----
  CUtensorMap tmA_b = make_tmap(dX, (uint64_t)D * 2, M, 256);
  CUtensorMap tmB_b = make_tmap(dWm, (uint64_t)D * 2, N, 128);
  auto fnb = proj_bf16_kernel<256, 128, 64, 4>;
  const int ntb = 256 / 64 * 128 + 32;
  const size_t shmb = ((size_t)(2 * 4 * 8) + 1023) / 1024 * 1024 + (size_t)4 * (256 + 128) * 128;
  CUDA_CHECK(cudaFuncSetAttribute(fnb, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmb));
  dim3 gb(div_up(N, 128), div_up(M, 256));
  auto run_bf16 = [&] { fnb<<<gb, ntb, shmb>>>(tmA_b, tmB_b, dYb, M, N, D); };

  // ---- fp8 路径（tensor map 的 boxR 必须随 BM/BN 建，否则 expect_tx 补不齐→死锁）----
  auto run_fp8 = [&](auto fn, int BM, int BN, int nt, size_t shm) {
    CUtensorMap tmA_f = make_tmap(dXq, (uint64_t)D, M, (uint32_t)BM);
    CUtensorMap tmB_f = make_tmap(dWq, (uint64_t)D, N, (uint32_t)BN);
    dim3 g(div_up(N, BN), div_up(M, BM));
    return [=] { fn<<<g, nt, shm>>>(tmA_f, tmB_f, dYf, M, N, D, dsa, dsb, KBLK); };
  };

  auto report = [&](const char* tag, double ms, double peak) {
    double tf = to_tflops(flops, ms);
    std::printf("  %-18s %8.4f ms  %8.2f TFLOPS  (%5.1f%%)\n", tag, ms, tf, 100.0 * tf / peak);
  };

  // run fp8 with selected cfg
#define RUN_FP8_T(NAME, BM, BN, ST)                                                                \
  do {                                                                                             \
    if (want(NAME) || want("all")) {                                                               \
      auto fn = proj_fp8_kernel<BM, BN, 128, ST>;                                                  \
      const int nt = (BM / 64) * 128 + 32;                                                         \
      const size_t shm = ((size_t)(2 * (ST) * 8) + 1023) / 1024 * 1024 + (size_t)(ST) * ((BM) + (BN)) * 128; \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));  \
      auto run = run_fp8(fn, BM, BN, nt, shm);                                                     \
      run(); CUDA_CHECK_LAST();                                                                    \
      report(NAME, bench_ms(run, 5, 30), FP8_PEAK);                                             \
    }                                                                                              \
  } while (0)

  if (want("all") || want("bf16")) {
    run_bf16(); CUDA_CHECK_LAST();
    report("bf16 256x128s4", bench_ms(run_bf16, 5, 30), BF16_PEAK);
  }
  RUN_FP8_T("fp8_128x128s4", 128, 128, 4);
  RUN_FP8_T("fp8_128x128s3", 128, 128, 3);
  RUN_FP8_T("fp8_128x128s5", 128, 128, 5);
  RUN_FP8_T("fp8_128x256s3", 128, 256, 3);
  RUN_FP8_T("fp8_64x128s4", 64, 128, 4);
  RUN_FP8_T("fp8_192x128s4", 192, 128, 4);

  // ---- 正确性 + 端到端精度：bf16 vs fp8 的池化输出 ----
  if (want("all") || want("acc")) {
    const int MC = 2 * RATIO;
    // CPU：bf16 输入 vs fp8 反量化输入
    std::vector<float> xb((size_t)MC * D), wb((size_t)N * D), xq((size_t)MC * D), wq((size_t)N * D);
    for (size_t i = 0; i < xb.size(); ++i) xb[i] = __bfloat162float(hX[i]);
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = __bfloat162float(hWm[i]);
    for (size_t i = 0; i < xq.size(); ++i) xq[i] = (float)hXq[i] * hsa[i / D * KBLK + (i % D) / 128];
    for (size_t i = 0; i < wq.size(); ++i)
      wq[i] = (float)hWq[i] * hsb[(i / D / 128) * KBLK + (i % D) / 128];
    std::vector<float> refb, reff;
    cpu_ref(xb, wb, hApe, hNw, MC, N, C, RATIO, refb);
    cpu_ref(xq, wq, hApe, hNw, MC, N, C, RATIO, reff);
    double qerr = 0, qref = 0;
    for (size_t i = 0; i < refb.size(); ++i) {
      qerr = std::max(qerr, (double)std::fabs(refb[i] - reff[i]));
      qref = std::max(qref, (double)std::fabs(refb[i]));
    }
    std::printf("  FP8 quantization: pooled-output max_abs_diff=%.3e (ref~%.3f, %.2f%%)\n\n",
                qerr, qref, 100.0 * qerr / qref);
  }

  CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dWm)); CUDA_CHECK(cudaFree(dYb)); CUDA_CHECK(cudaFree(dYf));
  CUDA_CHECK(cudaFree(dXq)); CUDA_CHECK(cudaFree(dWq)); CUDA_CHECK(cudaFree(dsa)); CUDA_CHECK(cudaFree(dsb));
  CUDA_CHECK(cudaFree(dApe)); CUDA_CHECK(cudaFree(dNw)); CUDA_CHECK(cudaFree(dOutb)); CUDA_CHECK(cudaFree(dOutf));
  CUDA_CHECK(cudaFree(dcos)); CUDA_CHECK(cudaFree(dsin));
  return 0;
}
