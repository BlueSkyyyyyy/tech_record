// 34 MoE（四）：fused expert FFN in FP8 —— 把 31 的 per-block grouped GEMM 接进 30 的
// 融合骨架，并把 unpermute 融进 down 的 epilogue。
//
// 承接：
//   30（bf16 fused MoE）：五段流水融成 3 kernel，K1 实测 DRAM 84.5% —— prefill FFN 是
//        **权重带宽受限**（50.7 GB 专家权重 vs 1.8 GB activation），所以「省 activation」只能
//        换来 1.02–1.08×。要再快，唯一的杠杆是把权重字节砍下来。
//   31（grouped GEMM + per-block FP8）：e4m3 + 128×128 block scale，权重字节直接减半。
//
// 本篇把 31 的 per-block grouped 骨架接进 MoE FFN 的端到端：
//   不融合（4 kernel + cast）：
//     grouped up/gate (fp8 per-block) -> GU [P,2I] fp32
//     swiglu+quant(GU)                -> A2 fp8 [P,I] + sa2 [P,I/128]
//     grouped down (fp8 per-block)    -> D [P,H] fp32
//     unpermute(w,D)                  -> Yf [M,H] fp32
//   融合（3 kernel + cast）：
//     grouped up/gate (fp8 per-block) -> GU
//     swiglu+quant                    -> A2 fp8 + sa2
//     down + unpermute（red.global.add.f32 直接归约回 token）-> Yf
//
// 附：为什么「per-block FP8 + 双累加器 SwiGLU 融合」在 Hopper 上做不了 ——
//   gate/up 双累加器 2×64=128 fp32，per-block 的 fin 再 2×64=128，合计 256 > 255
//   的每线程寄存器上限（这一段作为负结果写进文章）。
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, n_routed_experts=384,
//   num_experts_per_tok=6, e4m3 + ue8m0 + weight_block 128x128。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 34-fused-moe-fp8/moe_fp8.cu [M] [bal|rand]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;

// DeepSeek-V4-Pro
constexpr int H = 7168;   // hidden
constexpr int I = 3072;   // moe_intermediate_size
constexpr int E = 384;    // n_routed_experts
constexpr int TOPK = 6;   // num_experts_per_tok

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（同 20/22/31）
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
// mbarrier + TMA helpers（同 23/31）
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
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void red_add_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__host__ __device__ __forceinline__ float silu_f(float x) {
#if defined(__CUDA_ARCH__)
  return x * (1.f / (1.f + __expf(-x)));
#else
  return x / (1.f + std::exp(-x));
#endif
}
__device__ __forceinline__ unsigned char to_fp8(float x) {
  return (unsigned char)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// ---------------------------------------------------------------------------
// grouped per-block FP8 GEMM（contiguous，复用 31 的 TMA + mbarrier + warp specialization）
//   A=(P,K) fp8，sa=(P,K/128)；B=(E*Ntile,K) fp8，sb=(E*Ntile/128,K/128)
//   UNPERM=false: 写 D[p,:] = scale * Σ A·B·sa·sb（fp32）
//   UNPERM=true : 写 Yf[row_tok[p],:] += row_w[p] * Σ A·B·sa·sb（fp32）
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool UNPERM, int MINB = 1>
__global__ void __launch_bounds__((BM / 64) * 128 + 32, MINB)
moe_pb_kernel(const __grid_constant__ CUtensorMap tmA,
              const __grid_constant__ CUtensorMap tmB,
              float* __restrict__ D, const int* __restrict__ gl,
              const int* __restrict__ row_tok, const float* __restrict__ row_w,
              int N, int K, int mlim, float scale,
              const float* __restrict__ sa, const float* __restrict__ sb, int KBLK, int P) {
  static_assert(BK == 128, "TMA SW128 内维固定 128 字节");
  constexpr int NWG = BM / 64;
  constexpr int NCONS = NWG * 128;
  constexpr int NSPLIT = BN / 128;
  constexpr int SBO = 1024;

  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * BK;

  const int tid = threadIdx.x;
  const int block_col = blockIdx.x * BN;
  const int block_row = blockIdx.y * BM;
  const int nblk = K / BK;
  const int group = gl[block_row];

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  if (tid >= NCONS) {  // ---- producer warp ----
    if (tid == NCONS) {
      for (int kb = 0; kb < nblk; ++kb) {
        const int st = kb % STAGES;
        if (kb >= STAGES) mbar_wait(empty + st, (uint32_t)((kb / STAGES - 1) & 1));
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * BK + BN * BK);
        tma_load_2d(&tmA, As + (size_t)st * BM * BK, kb * BK, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * BK, kb * BK, group * N + block_col, full + st);
      }
    }
    return;
  }

  // ---- consumer warpgroups ----
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  const int r0g = block_row + row0;
  const int r1g = r0g + 8;

  float acc[NSPLIT][64];
  float fin[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc[j][i] = 0.f; fin[j][i] = 0.f; }

  for (int kb = 0; kb < nblk; ++kb) {
    const int st = kb % STAGES;
    mbar_wait(full + st, (uint32_t)((kb / STAGES) & 1));
    fence_proxy_async();
    char* a = As + (size_t)st * BM * BK + (size_t)wg * 64 * BK;
    char* b = Bs + (size_t)st * BN * BK;
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

    // 每 128-k 块折算（24/31 的核心代价：tensor core 在此排空）
    wgmma_wait0();
    const float sa0 = (r0g < mlim) ? sa[(size_t)r0g * KBLK + kb] : 0.f;
    const float sa1 = (r1g < mlim) ? sa[(size_t)r1g * KBLK + kb] : 0.f;
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      const float sbv = sb[((size_t)group * (N / 128) + (block_col >> 7) + jn) * KBLK + kb];
      const float w0 = sa0 * sbv, w1 = sa1 * sbv;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        fin[jn][j * 4 + 0] += w0 * acc[jn][j * 4 + 0];
        fin[jn][j * 4 + 1] += w0 * acc[jn][j * 4 + 1];
        fin[jn][j * 4 + 2] += w1 * acc[jn][j * 4 + 2];
        fin[jn][j * 4 + 3] += w1 * acc[jn][j * 4 + 3];
      }
      float* ac = acc[jn];
#pragma unroll
      for (int i = 0; i < 64; ++i) ac[i] = 0.f;
    }
    mbar_arrive(empty + st);
  }

#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int cc = block_col + col;
      const float a0 = fin[jn][j * 4 + 0], a1 = fin[jn][j * 4 + 1];
      const float a2 = fin[jn][j * 4 + 2], a3 = fin[jn][j * 4 + 3];
      if (UNPERM) {
        const int t0 = row_tok[r0g], t1 = row_tok[r1g];
        const float w0 = row_w[r0g], w1 = row_w[r1g];
        if (w0 != 0.f) { red_add_f32(&D[(size_t)t0 * N + cc], w0 * a0); red_add_f32(&D[(size_t)t0 * N + cc + 1], w0 * a1); }
        if (w1 != 0.f) { red_add_f32(&D[(size_t)t1 * N + cc], w1 * a2); red_add_f32(&D[(size_t)t1 * N + cc + 1], w1 * a3); }
      } else {
        if (r0g < mlim)
          *reinterpret_cast<float2*>(&D[(size_t)r0g * N + cc]) = make_float2(a0 * scale, a1 * scale);
        if (r1g < mlim)
          *reinterpret_cast<float2*>(&D[(size_t)r1g * N + cc]) = make_float2(a2 * scale, a3 * scale);
      }
    }
  (void)P;
}

// ---------------------------------------------------------------------------
// swiglu + per-128 动态量化：GU[P,2I] -> A2 fp8 [P,I] + sa2 [P,I/128]
//   一个 block 处理 (p, iblk) 的 128 列；128 线程各负责 1 列。
// ---------------------------------------------------------------------------
__global__ void swiglu_quant_kernel(const float* __restrict__ GU, fp8* __restrict__ A2,
                                    float* __restrict__ sa2, int P, int Id) {
  const int nblk = Id / 128;
  const int idx = blockIdx.x;
  const int p = idx / nblk, ib = idx % nblk;
  const int t = threadIdx.x;
  const int col = ib * 128 + t;
  const float g = GU[(size_t)p * 2 * Id + col];
  const float u = GU[(size_t)p * 2 * Id + Id + col];
  const float a2 = silu_f(g) * u;
  __shared__ float sm[128];
  sm[t] = fabsf(a2);
  __syncthreads();
#pragma unroll
  for (int s = 64; s > 0; s >>= 1) {
    if (t < s) sm[t] = fmaxf(sm[t], sm[t + s]);
    __syncthreads();
  }
  float scale = sm[0] / 448.0f;
  if (scale < 1e-12f) scale = 1e-12f;
  if (t == 0) sa2[(size_t)p * nblk + ib] = scale;
  reinterpret_cast<unsigned char*>(A2)[(size_t)p * Id + col] = to_fp8(a2 / scale);
}

// v2：一个 block 一行，warp 分批处理 128 列块，float4 读 + warp 内 amax（打满带宽）
__global__ void swiglu_quant_v2_kernel(const float* __restrict__ GU, fp8* __restrict__ A2,
                                       float* __restrict__ sa2, int P, int Id) {
  const int p = blockIdx.x;
  const int nb = Id / 128;
  const int nw = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float* gbase = GU + (size_t)p * 2 * Id;
  const float* ubase = gbase + Id;
  for (int ib = warp; ib < nb; ib += nw) {
    const int col = ib * 128 + lane * 4;
    const float4 g = *reinterpret_cast<const float4*>(gbase + col);
    const float4 u = *reinterpret_cast<const float4*>(ubase + col);
    float a[4];
    a[0] = silu_f(g.x) * u.x; a[1] = silu_f(g.y) * u.y;
    a[2] = silu_f(g.z) * u.z; a[3] = silu_f(g.w) * u.w;
    float m = fmaxf(fmaxf(fabsf(a[0]), fabsf(a[1])), fmaxf(fabsf(a[2]), fabsf(a[3])));
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    float scale = m / 448.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    if (lane == 0) sa2[(size_t)p * nb + ib] = scale;
    const unsigned b0 = to_fp8(a[0] / scale), b1 = to_fp8(a[1] / scale);
    const unsigned b2 = to_fp8(a[2] / scale), b3 = to_fp8(a[3] / scale);
    const unsigned packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(A2) + (size_t)p * Id)[ib * 32 + lane] =
        packed;
  }
}

// ---------------------------------------------------------------------------
// unpermute：Yf[t,:] = Σ_{rows p of t} w[p] * D[p,:]
// ---------------------------------------------------------------------------
__global__ void unpermute_kernel(const float* __restrict__ D, const int* __restrict__ tok_ptr,
                                 const int* __restrict__ tok_row, const float* __restrict__ row_w,
                                 float* __restrict__ Yf, int M) {
  const int t = blockIdx.x;
  if (t >= M) return;
  const int n4 = H / 4;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    float a[4] = {0, 0, 0, 0};
    for (int q = tok_ptr[t]; q < tok_ptr[t + 1]; ++q) {
      const int p = tok_row[q];
      const float w = row_w[p];
      const float4 v = reinterpret_cast<const float4*>(D + (size_t)p * H)[i];
      a[0] += w * v.x; a[1] += w * v.y; a[2] += w * v.z; a[3] += w * v.w;
    }
    reinterpret_cast<float4*>(Yf + (size_t)t * H)[i] = make_float4(a[0], a[1], a[2], a[3]);
  }
}

__global__ void cast_kernel(const float* __restrict__ Yf, __nv_bfloat16* __restrict__ Y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) Y[i] = __float2bfloat16(Yf[i]);
}

// ---------------------------------------------------------------------------
// 数据填充
// ---------------------------------------------------------------------------
__global__ void fill_fp8_kernel(fp8* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)(i ^ (i >> 32)) ^ (seed * 0x9E3779B9u);
    x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    float f = 2.f * ((float)((x >> 8) & 0xFFFF) / 65535.f - 0.5f);
    p[i] = __nv_fp8_e4m3(f);
  }
}
__global__ void fill_scale_kernel(float* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)(i ^ (i >> 32)) ^ (seed * 0x9E3779B9u);
    x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    p[i] = 0.5f + ((x >> 8) & 0xFFFF) / 65535.f;
  }
}

// ---------------------------------------------------------------------------
// 宿主：路由分布（token→experts），专家段对齐到 BM（同 30）
// ---------------------------------------------------------------------------
struct Dist {
  int M, Pp, nrows_actual = 0;
  std::vector<int> gl, row_tok;
  std::vector<float> row_w;
  std::vector<int> tok_ptr, tok_row;
  double padding_waste() const { return 100.0 * (Pp - nrows_actual) / Pp; }
};

static Dist build_dist(int M, int BM, unsigned seed, bool balanced) {
  Dist d;
  d.M = M;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ur(0.f, 1.f);
  std::uniform_int_distribution<int> ue(0, E - 1);
  std::vector<std::vector<int>> tok_experts(M);
  std::vector<int> count(E, 0);
  for (int t = 0; t < M; ++t) {
    if (balanced) {
      for (int j = 0; j < TOPK; ++j) {
        int e = (int)(((long long)t * TOPK + j) % E);
        bool dup = false;
        for (int y : tok_experts[t]) dup |= (y == e);
        if (!dup) { tok_experts[t].push_back(e); count[e]++; }
        else {
          while ((int)tok_experts[t].size() < j + 1) {
            int e2 = ue(rng);
            bool d2 = false;
            for (int y : tok_experts[t]) d2 |= (y == e2);
            if (!d2) { tok_experts[t].push_back(e2); count[e2]++; }
          }
        }
      }
      continue;
    }
    while ((int)tok_experts[t].size() < TOPK) {
      int e = ue(rng);
      bool dup = false;
      for (int y : tok_experts[t]) dup |= (y == e);
      if (!dup) { tok_experts[t].push_back(e); count[e]++; }
    }
  }
  std::vector<int> off(E);
  int acc = 0;
  for (int g = 0; g < E; ++g) { off[g] = acc; acc += (count[g] + BM - 1) / BM * BM; }
  d.Pp = acc;
  d.gl.assign(d.Pp, -1);
  d.row_tok.assign(d.Pp, 0);
  d.row_w.assign(d.Pp, 0.f);
  std::vector<std::vector<std::pair<int, int>>> slots(E);
  for (int t = 0; t < M; ++t)
    for (size_t j = 0; j < tok_experts[t].size(); ++j) slots[tok_experts[t][j]].push_back({t, (int)j});
  std::vector<int> fill_pos(E, 0);
  for (int g = 0; g < E; ++g)
    for (auto& s : slots[g]) {
      const int p = off[g] + fill_pos[g]++;
      d.gl[p] = g;
      d.row_tok[p] = s.first;
      d.row_w[p] = ur(rng);
      d.nrows_actual++;
    }
  d.tok_ptr.assign(M + 1, 0);
  for (int p = 0; p < d.Pp; ++p)
    if (d.row_w[p] != 0.f) d.tok_ptr[d.row_tok[p] + 1]++;
  for (int t = 0; t < M; ++t) d.tok_ptr[t + 1] += d.tok_ptr[t];
  d.tok_row.assign(d.nrows_actual, 0);
  std::vector<int> cur(d.tok_ptr.begin(), d.tok_ptr.end() - 1);
  for (int p = 0; p < d.Pp; ++p)
    if (d.row_w[p] != 0.f) d.tok_row[cur[d.row_tok[p]]++] = p;
  return d;
}

static void die_tmap(CUresult r) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s ? s : "?");
    std::exit(1);
  }
}
static CUtensorMap make_tmap_2d(void* ptr, uint64_t K, uint64_t R, uint32_t boxK, uint32_t boxR) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {K, R};
  cuuint64_t strides[1] = {K};
  cuuint32_t box[2] = {boxK, boxR};
  cuuint32_t es[2] = {1, 1};
  die_tmap(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr, dims, strides, box, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tm;
}

template <int BM, int BN, int BK, int STAGES, bool UNPERM, int MINB = 1>
static void launch_gemm(const CUtensorMap& tmA, const CUtensorMap& tmB, float* D, const int* gl,
                        const int* rtok, const float* rw, int N, int K, int mlim, float scale,
                        const float* sa, const float* sb, int KBLK, int grid_x, int grid_y) {
  auto fn = moe_pb_kernel<BM, BN, BK, STAGES, UNPERM, MINB>;
  const int nt = (BM / 64) * 128 + 32;
  const int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  const size_t shm = (size_t)BARR + (size_t)STAGES * (BM + BN) * BK;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
  dim3 grid(grid_x, grid_y, 1);
  fn<<<grid, nt, shm>>>(tmA, tmB, D, gl, rtok, rw, N, K, mlim, scale, sa, sb, KBLK, mlim);
  CUDA_CHECK_LAST();
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int M = (argc > 1) ? std::atoi(argv[1]) : 8192;
  const bool balanced = (argc > 2) && (std::strcmp(argv[2], "bal") == 0);
  constexpr int BM = 128, BK = 128;
  const int KBLK_H = H / 128;  // 56
  const int KBLK_I = I / 128;  // 24

  DeviceInfo d0 = device_info(0);
  print_device_info(d0);
  std::printf("\nFused MoE expert FFN (fp8 e4m3, per-block 128x128): M=%d E=%d topk=%d H=%d I=%d\n",
              M, E, TOPK, H, I);

  Dist dist = build_dist(M, BM, 1234, balanced);
  const int Pp = dist.Pp;
  std::printf("routing=%s  permuted rows: Pp=%d (actual=%d)  padding waste=%.1f%%\n",
              balanced ? "balanced" : "random", Pp, dist.nrows_actual, dist.padding_waste());
  if (Pp % BM) { std::printf("Pp %% BM != 0, abort\n"); return 1; }
  // 每个 expert 的 m-tile 数 = ceil(m_g/BM)；B 被每个 m-tile 重读一次
  int tiles = 0;
  {
    std::vector<int> cnt(E, 0);
    for (int p = 0; p < Pp; ++p)
      if (dist.gl[p] >= 0) cnt[dist.gl[p]]++;
    for (int g = 0; g < E; ++g) tiles += (cnt[g] + BM - 1) / BM;
  }
  const double bread_b1 = (double)tiles * (2.0 * I) * H;   // K1 的 B 实际流量
  const double bread_b2 = (double)tiles * H * (double)I;   // K2 的 B 实际流量
  std::printf("m-tiles/expert = %d 共 %d 个 -> K1 B 流量 %.2f GB, K2 B 流量 %.2f GB\n",
              tiles / E, tiles, bread_b1 / 1e9, bread_b2 / 1e9);

  // ---- 分配 ----
  fp8 *A1, *A2, *W12, *Wd;
  float *sa1, *sa2, *sw12, *swd, *GU, *D, *Yf;
  __nv_bfloat16* Y;
  int *rtok, *gl, *tokp, *tokr;
  float* rw;
  const size_t a1N = (size_t)Pp * H, a2N = (size_t)Pp * I;
  const size_t w12N = (size_t)E * 2 * I * H, wdN = (size_t)E * H * I;
  CUDA_CHECK(cudaMalloc(&A1, a1N));
  CUDA_CHECK(cudaMalloc(&A2, a2N));
  CUDA_CHECK(cudaMalloc(&W12, w12N));
  CUDA_CHECK(cudaMalloc(&Wd, wdN));
  CUDA_CHECK(cudaMalloc(&sa1, (size_t)Pp * KBLK_H * 4));
  CUDA_CHECK(cudaMalloc(&sa2, (size_t)Pp * KBLK_I * 4));
  CUDA_CHECK(cudaMalloc(&sw12, (size_t)E * 2 * I / 128 * KBLK_H * 4));
  CUDA_CHECK(cudaMalloc(&swd, (size_t)E * H / 128 * KBLK_I * 4));
  CUDA_CHECK(cudaMalloc(&GU, (size_t)Pp * 2 * I * 4));
  CUDA_CHECK(cudaMalloc(&D, (size_t)Pp * H * 4));
  CUDA_CHECK(cudaMalloc(&Yf, (size_t)M * H * 4));
  CUDA_CHECK(cudaMalloc(&Y, (size_t)M * H * 2));
  CUDA_CHECK(cudaMalloc(&rtok, (size_t)Pp * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&gl, (size_t)Pp * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&rw, (size_t)Pp * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&tokp, (size_t)(M + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tokr, (size_t)Pp * sizeof(int)));

  fill_fp8_kernel<<<1024, 256>>>(A1, a1N, 11u);
  fill_fp8_kernel<<<8192, 256>>>(W12, w12N, 22u);
  fill_fp8_kernel<<<8192, 256>>>(Wd, wdN, 33u);
  fill_scale_kernel<<<1024, 256>>>(sa1, (size_t)Pp * KBLK_H, 44u);
  fill_scale_kernel<<<4096, 256>>>(sw12, (size_t)E * 2 * I / 128 * KBLK_H, 55u);
  fill_scale_kernel<<<4096, 256>>>(swd, (size_t)E * H / 128 * KBLK_I, 66u);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(rtok, dist.row_tok.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gl, dist.gl.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rw, dist.row_w.data(), (size_t)Pp * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tokp, dist.tok_ptr.data(), (size_t)(M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tokr, dist.tok_row.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));

  // ---- tensor maps ----
  const int BN1 = 128, BN2 = 128, ST1 = 3, ST2 = 4;
  CUtensorMap tmA1 = make_tmap_2d(A1, H, Pp, 128, BM);
  CUtensorMap tmW12 = make_tmap_2d(W12, H, (uint64_t)E * 2 * I, 128, BN1);
  CUtensorMap tmA2 = make_tmap_2d(A2, I, Pp, 128, BM);
  CUtensorMap tmWd = make_tmap_2d(Wd, I, (uint64_t)E * H, 128, BN2);
  CUtensorMap tmW12_256 = make_tmap_2d(W12, H, (uint64_t)E * 2 * I, 128, 256);
  CUtensorMap tmWd_256 = make_tmap_2d(Wd, I, (uint64_t)E * H, 128, 256);

  // ---- kernel 启动器 ----
  auto k1 = [&] {  // grouped up/gate -> GU (fp32, N=2I)
    launch_gemm<BM, BN1, BK, ST1, false>(tmA1, tmW12, GU, gl, nullptr, nullptr, 2 * I, H, Pp, 1.0f,
                                         sa1, sw12, KBLK_H, 2 * I / BN1, Pp / BM);
  };
  auto kswi = [&] {  // swiglu + quant -> A2, sa2
    const int nb = Pp * (I / 128);
    swiglu_quant_kernel<<<nb, 128>>>(GU, A2, sa2, Pp, I);
  };
  auto kswi2 = [&] { swiglu_quant_v2_kernel<<<Pp, 256>>>(GU, A2, sa2, Pp, I); };
  auto k2 = [&] {  // grouped down -> D (fp32)
    launch_gemm<BM, BN2, BK, ST2, false>(tmA2, tmWd, D, gl, nullptr, nullptr, H, I, Pp, 1.0f,
                                         sa2, swd, KBLK_I, H / BN2, Pp / BM);
  };
  auto k2f = [&] {  // grouped down + unpermute -> Yf
    launch_gemm<BM, BN2, BK, ST2, true>(tmA2, tmWd, Yf, gl, rtok, rw, H, I, Pp, 1.0f,
                                        sa2, swd, KBLK_I, H / BN2, Pp / BM);
  };
  auto kunt = [&] { unpermute_kernel<<<M, 256>>>(D, tokp, tokr, rw, Yf, M); };
  auto kcast = [&] { cast_kernel<<<2048, 256>>>(Yf, Y, (size_t)M * H); };

  // ---- 正确性 ----
  // K1: GU[p, col] = Σ_k A1[p,k]*sa1[p,k/128] * W12[g*2I+col,k]*sw12[...]
  auto check_k1 = [&] {
    cudaMemset(GU, 0, (size_t)Pp * 2 * I * 4);
    k1();
    CUDA_CHECK_LAST();
    std::vector<fp8> ha(H), hb(H);
    std::vector<float> hsa(KBLK_H), hsb(KBLK_H);
    double max_rel = 0, max_ref = 0;
    for (int s = 0; s < 6; ++s) {
      const int p = (s * 7919 + 3) % Pp;
      const int g = dist.gl[p];
      if (g < 0) continue;  // 跳过 padding 行（random 路由）
      const int cc = (s * 2137 + 7) % (2 * I);
      CUDA_CHECK(cudaMemcpy(ha.data(), A1 + (size_t)p * H, H, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hsa.data(), sa1 + (size_t)p * KBLK_H, KBLK_H * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hb.data(), W12 + ((size_t)(g * 2 * I + cc) * H), H, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hsb.data(), sw12 + ((size_t)(g * 2 * I / 128 + cc / 128) * KBLK_H),
                            KBLK_H * 4, cudaMemcpyDeviceToHost));
      double e = 0;
      for (int kb = 0; kb < KBLK_H; ++kb) {
        double blk = 0;
        for (int k = kb * 128; k < (kb + 1) * 128; ++k)
          blk += (double)(float)ha[k] * (double)(float)hb[k];
        e += blk * (double)hsa[kb] * (double)hsb[kb];
      }
      float got;
      CUDA_CHECK(cudaMemcpy(&got, GU + (size_t)p * 2 * I + cc, 4, cudaMemcpyDeviceToHost));
      double rel = std::fabs((double)got - e) / std::max(std::fabs(e), 1.0);
      if (std::fabs(e) > 3.0) { max_rel = std::max(max_rel, rel); max_ref = std::max(max_ref, std::fabs(e)); }
    }
    std::printf("  [K1 up/gate fp8 pb ] max_rel_err=%.3e (ref~%.1f) %s\n", max_rel, max_ref,
                max_rel < 3e-2 ? "OK" : "FAIL");
  };

  // swiglu+quant: dequant(A2) vs silu(GU[p,col])*GU[p,I+col]
  auto check_swi = [&] {
    k1(); kswi();
    CUDA_CHECK_LAST();
    double max_rel = 0, max_ref = 0;
    for (int s = 0; s < 6; ++s) {
      const int p = (s * 1009 + 5) % Pp;
      const int cc = (s * 2137 + 11) % I;
      float g, u, sc;
      fp8 q;
      CUDA_CHECK(cudaMemcpy(&g, GU + (size_t)p * 2 * I + cc, 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&u, GU + (size_t)p * 2 * I + I + cc, 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&q, A2 + (size_t)p * I + cc, 1, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&sc, sa2 + (size_t)p * (I / 128) + cc / 128, 4, cudaMemcpyDeviceToHost));
      const double ref = (double)silu_f(g) * u;
      const double got = (double)(float)q * sc;
      const double rel = std::fabs(got - ref) / std::max(std::fabs(ref), 1e-3);
      if (std::fabs(ref) > 0.1) { max_rel = std::max(max_rel, rel); max_ref = std::max(max_ref, std::fabs(ref)); }
    }
    std::printf("  [swiglu+quant      ] max_rel_err=%.3e (ref~%.2f) %s\n", max_rel, max_ref,
                max_rel < 8e-2 ? "OK" : "FAIL");
  };

  // K2 fused: dequant 后与 CPU 逐块折算对拍
  auto check_k2 = [&] {
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    k1(); kswi(); k2f();
    CUDA_CHECK_LAST();
    std::vector<fp8> ha(I), hb(I);
    std::vector<float> hsa(KBLK_I), hsb(KBLK_I);
    double max_rel = 0, max_ref = 0;
    for (int s = 0; s < 6; ++s) {
      const int t = (s * 1009 + 5) % M;
      std::vector<int> rows;
      for (int q = dist.tok_ptr[t]; q < dist.tok_ptr[t + 1]; ++q) rows.push_back(dist.tok_row[q]);
      for (int c = 0; c < 3 && !rows.empty(); ++c) {
        const int cc = (c * 1009 + 11) % H;
        double acc = 0;
        for (int p : rows) {
          const int g = dist.gl[p];
          CUDA_CHECK(cudaMemcpy(ha.data(), A2 + (size_t)p * I, I, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(hsa.data(), sa2 + (size_t)p * KBLK_I, KBLK_I * 4, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(hb.data(), Wd + ((size_t)g * H + cc) * I, I, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(hsb.data(), swd + ((size_t)(g * H / 128 + cc / 128) * KBLK_I),
                                KBLK_I * 4, cudaMemcpyDeviceToHost));
          double dot = 0;
          for (int kb = 0; kb < KBLK_I; ++kb) {
            double blk = 0;
            for (int k = kb * 128; k < (kb + 1) * 128; ++k)
              blk += (double)(float)ha[k] * (double)(float)hb[k];
            dot += blk * (double)hsa[kb] * (double)hsb[kb];
          }
          acc += (double)dist.row_w[p] * dot;
        }
        float got;
        CUDA_CHECK(cudaMemcpy(&got, Yf + (size_t)t * H + cc, 4, cudaMemcpyDeviceToHost));
        double rel = std::fabs((double)got - acc) / std::max(std::fabs(acc), 1.0);
        if (std::fabs(acc) > 3.0) { max_rel = std::max(max_rel, rel); max_ref = std::max(max_ref, std::fabs(acc)); }
      }
    }
    std::printf("  [K2 down+unperm    ] max_rel_err=%.3e (ref~%.1f) %s\n", max_rel, max_ref,
                max_rel < 3e-2 ? "OK" : "FAIL");
  };

  const char* only = (argc > 3) ? argv[3] : "all";
  if (std::strcmp(only, "all") != 0) {  // ncu 单 kernel 剖析模式
    k1(); kswi(); CUDA_CHECK_LAST();
    if (!std::strcmp(only, "ncu1")) { for (int i = 0; i < 3; ++i) k1(); }
    else if (!std::strcmp(only, "ncu2")) { for (int i = 0; i < 3; ++i) k2(); }
    else if (!std::strcmp(only, "ncu2f")) { for (int i = 0; i < 3; ++i) k2f(); }
    else if (!std::strcmp(only, "ncuswi")) { for (int i = 0; i < 3; ++i) kswi(); }
    CUDA_CHECK_LAST();
    cudaDeviceSynchronize();
    return 0;
  }

  std::printf("\n-- 正确性 --\n");
  check_k1();
  check_swi();
  check_k2();

  // ---- 计时 ----
  const double flops = 2.0 * (double)Pp * (2.0 * I) * H + 2.0 * (double)Pp * H * I;
  const double flops_useful = flops * (double)dist.nrows_actual / Pp;
  auto report = [&](const char* tag, double ms, double bytes = 0) {
    double tf = to_tflops(flops_useful, ms);
    std::printf("%-20s %9.4f ms  %8.2f TFLOPS(useful) (%5.1f%% fp8 peak)", tag, ms, tf,
                100.0 * tf / FP8_PEAK);
    if (bytes > 0) std::printf("  B-read %6.1f GB/s", to_gbps(bytes, ms));
    std::printf("\n");
  };

  k1(); kswi(); k2f(); kcast();
  CUDA_CHECK_LAST();

  {
    double t = bench_ms([&] { k1(); }, 3, 20);
    report("K1 up/gate", t, bread_b1);
    double t0 = bench_ms([&] { kswi(); }, 3, 50);
    report("swiglu+quant v1", t0);
    double t0b = bench_ms([&] { kswi2(); }, 3, 50);
    report("swiglu+quant v2", t0b);
    double t1 = bench_ms([&] { k2(); }, 3, 20);
    report("K2 down", t1, bread_b2);
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    double t2 = bench_ms([&] { k2f(); }, 3, 20);
    report("K2 down+unperm", t2, bread_b2);
    double t3 = bench_ms([&] { kunt(); }, 3, 50);
    report("unpermute", t3);
    double t4 = bench_ms([&] { kcast(); }, 3, 50);
    report("cast", t4);
  }

  // ---- 扫参：K1/K2 的 STAGES 与 BN（找靠近 HBM roofline 的工作点） ----
  std::printf("\n-- 扫参（B-read 用专家权重字节 / 耗时）--\n");
  auto sw_k1 = [&](int st) {
    size_t shm = 0;
    if (st == 2) { auto fn = moe_pb_kernel<BM, 128, BK, 2, false, 1>; shm = 1024 + (size_t)2 * (BM + 128) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(2 * I / 128, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA1, tmW12, GU, gl, nullptr, nullptr, 2 * I, H, Pp, 1.0f, sa1, sw12, KBLK_H, Pp); }
    else if (st == 3) { auto fn = moe_pb_kernel<BM, 128, BK, 3, false, 1>; shm = 1024 + (size_t)3 * (BM + 128) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(2 * I / 128, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA1, tmW12, GU, gl, nullptr, nullptr, 2 * I, H, Pp, 1.0f, sa1, sw12, KBLK_H, Pp); }
    else { auto fn = moe_pb_kernel<BM, 256, BK, 3, false, 1>; shm = 1024 + (size_t)3 * (BM + 256) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(2 * I / 256, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA1, tmW12_256, GU, gl, nullptr, nullptr, 2 * I, H, Pp, 1.0f, sa1, sw12, KBLK_H, Pp); }
    CUDA_CHECK_LAST();
  };
  auto sw_k2 = [&](int st, int bn) {
    size_t shm = 0;
    if (bn == 128) {
      if (st == 2) { auto fn = moe_pb_kernel<BM, 128, BK, 2, false, 1>; shm = 1024 + (size_t)2 * (BM + 128) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(H / 128, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA2, tmWd, D, gl, nullptr, nullptr, H, I, Pp, 1.0f, sa2, swd, KBLK_I, Pp); }
      else { auto fn = moe_pb_kernel<BM, 128, BK, 3, false, 1>; shm = 1024 + (size_t)3 * (BM + 128) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(H / 128, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA2, tmWd, D, gl, nullptr, nullptr, H, I, Pp, 1.0f, sa2, swd, KBLK_I, Pp); }
    } else {
      if (st == 2) { auto fn = moe_pb_kernel<BM, 256, BK, 2, false, 1>; shm = 1024 + (size_t)2 * (BM + 256) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(H / 256, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA2, tmWd_256, D, gl, nullptr, nullptr, H, I, Pp, 1.0f, sa2, swd, KBLK_I, Pp); }
      else { auto fn = moe_pb_kernel<BM, 256, BK, 3, false, 1>; shm = 1024 + (size_t)3 * (BM + 256) * BK; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); fn<<<dim3(H / 256, Pp / BM), (BM / 64) * 128 + 32, shm>>>(tmA2, tmWd_256, D, gl, nullptr, nullptr, H, I, Pp, 1.0f, sa2, swd, KBLK_I, Pp); }
    }
    CUDA_CHECK_LAST();
  };
  {
    k1(); kswi(); CUDA_CHECK_LAST();
    double a = bench_ms([&] { sw_k1(2); }, 3, 20); report("k1 128 s2", a, bread_b1);
    double b = bench_ms([&] { sw_k1(3); }, 3, 20); report("k1 128 s3", b, bread_b1);
    double c = bench_ms([&] { sw_k1(4); }, 3, 20); report("k1 256 s3", c, bread_b1);
    double d = bench_ms([&] { sw_k2(2, 128); }, 3, 20); report("k2 128 s2", d, bread_b2);
    double e = bench_ms([&] { sw_k2(3, 128); }, 3, 20); report("k2 128 s3", e, bread_b2);
    double f = bench_ms([&] { sw_k2(2, 256); }, 3, 20); report("k2 256 s2", f, bread_b2);
    double g = bench_ms([&] { sw_k2(3, 256); }, 3, 20); report("k2 256 s3", g, bread_b2);
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;
  }

  // ---- 端到端 ----
  auto run_unfused = [&] { k1(); kswi2(); k2(); kunt(); kcast(); };
  auto run_fused = [&] { CUDA_CHECK(cudaMemsetAsync(Yf, 0, (size_t)M * H * 4)); k1(); kswi2(); k2f(); kcast(); };
  run_fused();
  CUDA_CHECK_LAST();
  double tf_ = bench_ms(run_fused, 3, 20);
  run_unfused();
  CUDA_CHECK_LAST();
  double tu_ = bench_ms(run_unfused, 3, 20);
  std::printf("\n[unfused 5-kernel ] %9.4f ms\n", tu_);
  std::printf("[fused   4-kernel ] %9.4f ms   speedup %.3fx\n", tf_, tu_ / tf_);

  const double B_unf = (double)Pp * H * 2 * 2                     // up/gate: 读 A1 + 读 W12
                     + (double)Pp * 2 * I * 4                      // 写 GU
                     + (double)Pp * 2 * I * 4                      // swiglu 读 GU
                     + (double)Pp * I * 1                          // 写 A2 fp8
                     + (double)Pp * I + (double)Pp * H * 4         // down: 读 A2 + 写 D
                     + (double)Pp * H * 4 + (double)M * H * 4      // unpermute 读 D + 写 Yf
                     + (double)M * H * 4;                          // cast 读 Yf
  const double B_fus = (double)Pp * H * 2 * 2                     // up/gate
                     + (double)Pp * 2 * I * 4                      // 写 GU
                     + (double)Pp * 2 * I * 4                      // swiglu 读 GU
                     + (double)Pp * I
                     + (double)Pp * I + (double)M * H * 4          // down+unperm: 读 A2 + 写 Yf
                     + (double)M * H * 4;
  const double W_bytes = (double)w12N + (double)wdN;              // 专家权重（fp8）
  std::printf("act bytes: unfused=%.2f GB  fused=%.2f GB  (-%.0f%%)\n", B_unf / 1e9, B_fus / 1e9,
              100.0 * (B_unf - B_fus) / B_unf);
  std::printf("expert weights (fp8) = %.2f GB (bf16 需 %.2f GB)\n", W_bytes / 1e9, 2 * W_bytes / 1e9);
  k1(); kswi(); CUDA_CHECK_LAST();
  double tk1 = bench_ms([&] { k1(); }, 3, 20);
  double tk2 = bench_ms([&] { k2(); }, 3, 20);
  std::printf("K1+K2 权重流量 %.2f GB -> HBM 下限 @3.35TB/s = %.2f ms (实测 K1=%.2f + K2=%.2f = %.2f ms)\n",
              (bread_b1 + bread_b2) / 1e9, (bread_b1 + bread_b2) / 3.35e12 * 1e3, tk1, tk2, tk1 + tk2);

  CUDA_CHECK(cudaFree(A1)); CUDA_CHECK(cudaFree(A2)); CUDA_CHECK(cudaFree(W12)); CUDA_CHECK(cudaFree(Wd));
  CUDA_CHECK(cudaFree(sa1)); CUDA_CHECK(cudaFree(sa2)); CUDA_CHECK(cudaFree(sw12)); CUDA_CHECK(cudaFree(swd));
  CUDA_CHECK(cudaFree(GU)); CUDA_CHECK(cudaFree(D)); CUDA_CHECK(cudaFree(Yf)); CUDA_CHECK(cudaFree(Y));
  CUDA_CHECK(cudaFree(rtok)); CUDA_CHECK(cudaFree(gl)); CUDA_CHECK(cudaFree(rw));
  CUDA_CHECK(cudaFree(tokp)); CUDA_CHECK(cudaFree(tokr));
  return 0;
}
