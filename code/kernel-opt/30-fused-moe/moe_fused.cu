// 30 MoE（三）：fused expert FFN —— gather 输入 + SwiGLU 融合 + unpermute 融合
//
// 承接 27（router + permute/unpermute）与 25（grouped GEMM）。27 实测 MoE 前门里
// 「搬运」占 69%：permute 写出 permuted_x，unpermute 再把 expert 输出加权写回 token。
// 本文把这两次「物化 + 回读」从 expert FFN 里拿掉：
//
//   不融合（baseline，5 kernel）：
//     permute(X)                      -> PX   [P,H]
//     grouped up/gate GEMM            -> G,U  [P,I] fp32
//     swiglu(G,U)                     -> A2   [P,I] bf16
//     grouped down GEMM               -> D    [P,H] fp32
//     unpermute(w, D)                 -> Yf   [M,H] fp32
//
//   融合（2 kernel + cast）：
//     up_gate_kernel(FUSE=1): A 直接按 row_tok gather X（不写 PX），
//                             epilogue 同时算 gate/up 两个累加器并做 SwiGLU -> A2
//     down_kernel(FUSE=1)   : epilogue 直接把 w*acc atomicAdd 到 Yf[row_tok[p]]（免 D/unpermute）
//     cast(Yf)              -> Y bf16
//
// 两个版本共用同一份 GEMM 代码（模板参数 FUSE），所以差异**只有**被省掉的那几趟显存搬运。
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, n_routed_experts=384,
//   num_experts_per_tok=6, scoring_func=sqrtsoftplus, topk_method=noaux_tc。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//         scripts/run.sh 30-fused-moe/moe_fused.cu [M] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;
constexpr double BF16_PEAK = 989.0;

// DeepSeek-V4-Pro
constexpr int H = 7168;   // hidden
constexpr int I = 3072;   // moe_intermediate_size
constexpr int E = 384;    // n_routed_experts
constexpr int TOPK = 6;   // num_experts_per_tok

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（bf16，同 28 篇）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

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

// bf16 K-major SW128：8 行 × 64 元素（128B）atom，16B 列 c' = c ^ r。
__device__ __forceinline__ int sw128_off(int row, int k) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
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
// fire-and-forget 归约：不回读旧值，省掉 atomicAdd 的往返依赖（unpermute 累加用）
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

// ---------------------------------------------------------------------------
// K1：grouped up+gate GEMM + SwiGLU（N 维每 CTA 256 = gate128 + up128）
//     A 按 row_tok gather；B = W12[E,2I,H]（K-major，专家 e 的 gate 在前、up 在后）
//     FUSE=1: 写 A2=bf16(silu(g)*u) [P,I]；FUSE=0: 写 G,U fp32 [P,I]
// ---------------------------------------------------------------------------
template <int BM, int BK, int STAGES, bool FUSE>
__global__ void __launch_bounds__((BM / 64) * 128)
up_gate_kernel(const bf16* __restrict__ X, const bf16* __restrict__ W12,
               const int* __restrict__ row_tok, const int* __restrict__ gl,
               bf16* __restrict__ A2, float* __restrict__ G, float* __restrict__ U,
               int P, int Hd, int Id, int Ed) {
  constexpr int NWG = BM / 64;
  constexpr int NT = NWG * 128;
  constexpr int SBO = 1024;
  extern __shared__ __align__(1024) char smem[];
  char* As = smem;                                     // [STAGES][BM][64]
  char* Bs = As + (size_t)STAGES * BM * BK * 2;        // [STAGES][256][64]

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * 128;              // over I
  const int group = gl[block_row];
  const int nblk = Hd / BK;

  auto load = [&](int kb, int stage) {
    char* a = As + (size_t)stage * BM * BK * 2;
    char* b = Bs + (size_t)stage * 256 * BK * 2;
    for (int i = tid; i < BM * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      const int tok = row_tok[block_row + r];
      __pipeline_memcpy_async(a + sw128_off(r, c8), &X[(size_t)tok * Hd + kb * BK + c8], 16);
    }
    for (int i = tid; i < 256 * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      const int jn = r >> 7, rr = r & 127;
      const int brow = group * 2 * Id + jn * Id + block_col + rr;
      __pipeline_memcpy_async(b + jn * 128 * BK * 2 + sw128_off(rr, c8),
                              &W12[(size_t)brow * Hd + kb * BK + c8], 16);
    }
    __pipeline_commit();
  };

  float acc[2][64];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

  for (int s = 0; s < STAGES - 1 && s < nblk; ++s) load(s, s);
  if (nblk < STAGES - 1) __pipeline_commit();

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int kb = 0; kb < nblk; ++kb) {
    const int stage = kb % STAGES, nxt = kb + STAGES - 1;
    if (nxt < nblk) load(nxt, nxt % STAGES);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();
    char* a = As + (size_t)stage * BM * BK * 2 + (size_t)wg * 64 * BK * 2;
    char* b = Bs + (size_t)stage * 256 * BK * 2;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < 2; ++jn) {
      char* bj = b + (size_t)jn * 128 * BK * 2;
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
  for (int j = 0; j < 16; ++j) {
    const int col = j * 8 + (lane & 3) * 2;
    const int r0 = block_row + row0, r1 = r0 + 8;
    const int c0 = block_col + col;
    const float g0 = acc[0][j * 4 + 0], g1 = acc[0][j * 4 + 1];
    const float g2 = acc[0][j * 4 + 2], g3 = acc[0][j * 4 + 3];
    const float u0 = acc[1][j * 4 + 0], u1 = acc[1][j * 4 + 1];
    const float u2 = acc[1][j * 4 + 2], u3 = acc[1][j * 4 + 3];
    if (FUSE) {
      // 相邻两列是同一线程写的 → 必须打包成一次 4B 存储（否则 nvcc 合并丢高 16 位）
      *reinterpret_cast<__nv_bfloat162*>(&A2[(size_t)r0 * Id + c0]) =
          __floats2bfloat162_rn(silu_f(g0) * u0, silu_f(g1) * u1);
      *reinterpret_cast<__nv_bfloat162*>(&A2[(size_t)r1 * Id + c0]) =
          __floats2bfloat162_rn(silu_f(g2) * u2, silu_f(g3) * u3);
    } else {
      G[(size_t)r0 * Id + c0] = g0; G[(size_t)r0 * Id + c0 + 1] = g1;
      G[(size_t)r1 * Id + c0] = g2; G[(size_t)r1 * Id + c0 + 1] = g3;
      U[(size_t)r0 * Id + c0] = u0; U[(size_t)r0 * Id + c0 + 1] = u1;
      U[(size_t)r1 * Id + c0] = u2; U[(size_t)r1 * Id + c0 + 1] = u3;
    }
  }
  (void)P;
  (void)Ed;
}

// ---------------------------------------------------------------------------
// K2：grouped down GEMM + unpermute
//     A = A2 [P,I]（permuted，连续）；B = Wd[E,H,I]
//     FUSE=1: epilogue w*acc atomicAdd 到 Yf[row_tok[p], :]；FUSE=0: 写 D [P,H] fp32
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool FUSE>
__global__ void __launch_bounds__((BM / 64) * 128)
down_kernel(const bf16* __restrict__ A2, const bf16* __restrict__ Wd,
            const int* __restrict__ row_tok, const int* __restrict__ gl,
            const float* __restrict__ row_w, float* __restrict__ Out,
            int P, int Hd, int Id, int Ed) {
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NT = NWG * 128;
  constexpr int SBO = 1024;
  extern __shared__ __align__(1024) char smem[];
  char* As = smem;
  char* Bs = As + (size_t)STAGES * BM * BK * 2;

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;   // over H
  const int group = gl[block_row];
  const int nblk = Id / BK;

  auto load = [&](int kb, int stage) {
    char* a = As + (size_t)stage * BM * BK * 2;
    char* b = Bs + (size_t)stage * BN * BK * 2;
    for (int i = tid; i < BM * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      __pipeline_memcpy_async(a + sw128_off(r, c8),
                              &A2[(size_t)(block_row + r) * Id + kb * BK + c8], 16);
    }
    for (int i = tid; i < BN * (BK / 8); i += NT) {
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      const int brow = group * Hd + block_col + r;
      __pipeline_memcpy_async(b + sw128_off(r, c8), &Wd[(size_t)brow * Id + kb * BK + c8], 16);
    }
    __pipeline_commit();
  };

  float acc[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

  for (int s = 0; s < STAGES - 1 && s < nblk; ++s) load(s, s);
  if (nblk < STAGES - 1) __pipeline_commit();

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int kb = 0; kb < nblk; ++kb) {
    const int stage = kb % STAGES, nxt = kb + STAGES - 1;
    if (nxt < nblk) load(nxt, nxt % STAGES);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();
    char* a = As + (size_t)stage * BM * BK * 2 + (size_t)wg * 64 * BK * 2;
    char* b = Bs + (size_t)stage * BN * BK * 2;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * BK * 2;
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
  const int p0 = block_row + row0, p1 = p0 + 8;
  if (FUSE) {
    const int t0 = row_tok[p0], t1 = row_tok[p1];
    const float w0 = row_w[p0], w1 = row_w[p1];
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int cc = block_col + jn * 128 + j * 8 + (lane & 3) * 2;
        const float a0 = acc[jn][j * 4 + 0], a1 = acc[jn][j * 4 + 1];
        const float a2 = acc[jn][j * 4 + 2], a3 = acc[jn][j * 4 + 3];
        if (w0 != 0.f) { red_add_f32(&Out[(size_t)t0 * Hd + cc], w0 * a0); red_add_f32(&Out[(size_t)t0 * Hd + cc + 1], w0 * a1); }
        if (w1 != 0.f) { red_add_f32(&Out[(size_t)t1 * Hd + cc], w1 * a2); red_add_f32(&Out[(size_t)t1 * Hd + cc + 1], w1 * a3); }
      }
  } else {
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int cc = block_col + jn * 128 + j * 8 + (lane & 3) * 2;
        Out[(size_t)p0 * Hd + cc] = acc[jn][j * 4 + 0];
        Out[(size_t)p0 * Hd + cc + 1] = acc[jn][j * 4 + 1];
        Out[(size_t)p1 * Hd + cc] = acc[jn][j * 4 + 2];
        Out[(size_t)p1 * Hd + cc + 1] = acc[jn][j * 4 + 3];
      }
  }
  (void)P;
  (void)Ed;
}

// ---------------------------------------------------------------------------
// 辅助 kernel（只在不融合路径里出现）
// ---------------------------------------------------------------------------
__global__ void fill_bf16_kernel(bf16* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)(i ^ (i >> 32)) ^ (seed * 0x9E3779B9u);
    x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    p[i] = __float2bfloat16(0.6f * ((float)((x >> 8) & 0xFFFF) / 65535.f - 0.5f));
  }
}

// permuted_x[pos,:] = x[row_tok[pos],:]，16B 向量化（按 token 分组，x 每 token 只读一次）
__global__ void permute_kernel(const bf16* __restrict__ x, const int* __restrict__ row_tok,
                               const int* __restrict__ tok_ptr, const int* __restrict__ tok_row,
                               bf16* __restrict__ px, int M) {
  const int t = blockIdx.x;
  if (t >= M) return;
  const int n8 = H / 8;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * H);
  for (int q = tok_ptr[t]; q < tok_ptr[t + 1]; ++q) {
    const int p = tok_row[q];
    uint4* dst = reinterpret_cast<uint4*>(px + (size_t)p * H);
    for (int i = threadIdx.x; i < n8; i += blockDim.x) dst[i] = src[i];
  }
}

// A2[p,c] = silu(G[p,c]) * U[p,c]   （8 元素/线程，打包 bf16 写出）
__global__ void swiglu_kernel(const float* __restrict__ G, const float* __restrict__ U,
                              bf16* __restrict__ A2, size_t n) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i + 7 >= n) return;
  bf16 out[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) out[q] = __float2bfloat16(silu_f(G[i + q]) * U[i + q]);
  __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(A2 + i);
#pragma unroll
  for (int q = 0; q < 4; ++q)
    d[q] = __nv_bfloat162(out[2 * q], out[2 * q + 1]);
}

// Yf[t,:] = Σ_{rows p of t} w[p] * D[p,:]   （fp32 累加）
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

__global__ void cast_kernel(const float* __restrict__ Yf, bf16* __restrict__ Y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) Y[i] = __float2bfloat16(Yf[i]);
}

// ---------------------------------------------------------------------------
// host：构造路由分布（token→experts），专家段对齐到 BM
// ---------------------------------------------------------------------------
struct Dist {
  int M, Pp, nrows_actual = 0;
  std::vector<int> gl, row_tok;      // [Pp]
  std::vector<float> row_w;          // [Pp]，padding 行 = 0
  std::vector<int> tok_ptr, tok_row; // [M+1], [Pp]（仅 actual 行）
  double padding_waste() const { return 100.0 * (Pp - nrows_actual) / Pp; }
};

static Dist build_dist(int M, int BM, unsigned seed, bool balanced = false) {
  Dist d;
  d.M = M;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ur(0.f, 1.f);
  std::uniform_int_distribution<int> ue(0, E - 1);
  std::vector<std::vector<int>> tok_experts(M);
  std::vector<int> count(E, 0);
  for (int t = 0; t < M; ++t) {
    if (balanced) {
      // 负载均衡路由：第 t 个 token 的 topk 槽轮流落到不同 expert（aux-loss 收敛后的近似）
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
  for (int g = 0; g < E; ++g) {
    off[g] = acc;
    acc += (count[g] + BM - 1) / BM * BM;  // 对齐
  }
  d.Pp = acc;
  d.gl.assign(d.Pp, -1);
  d.row_tok.assign(d.Pp, 0);
  d.row_w.assign(d.Pp, 0.f);
  std::vector<std::vector<std::pair<int, int>>> slots(E);  // (token, slot)
  for (int t = 0; t < M; ++t)
    for (size_t j = 0; j < tok_experts[t].size(); ++j) slots[tok_experts[t][j]].push_back({t, (int)j});
  std::vector<int> fill_pos(E, 0);
  for (int g = 0; g < E; ++g) {
    for (auto& s : slots[g]) {
      const int p = off[g] + fill_pos[g]++;
      d.gl[p] = g;
      d.row_tok[p] = s.first;
      d.row_w[p] = ur(rng);
      d.nrows_actual++;
    }
  }
  // token → rows（只含有非零权重的 actual 行）
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

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const int M = (argc > 1) ? std::atoi(argv[1]) : 8192;
  const char* which = (argc > 2) ? argv[2] : "all";
  const bool balanced = (argc > 3) && (std::strcmp(argv[3], "bal") == 0);
  constexpr int BM = 128, BK = 64, ST1 = 3, ST2 = 4;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nFused MoE expert FFN (bf16): M=%d E=%d topk=%d H=%d I=%d\n", M, E, TOPK, H, I);

  Dist dist = build_dist(M, BM, 1234, balanced);
  const int Pp = dist.Pp;
  std::printf("routing=%s  permuted rows: Pp=%d (actual=%d)  padding waste=%.1f%%\n",
              balanced ? "balanced" : "random", Pp, dist.nrows_actual, dist.padding_waste());
  if (Pp % BM) { std::printf("Pp %% BM != 0, abort\n"); return 1; }

  // ---- 分配 ----
  bf16 *X, *W12, *Wd, *PX, *A2, *Y;
  float *G, *U, *D, *Yf;
  int *rtok, *gl, *tokp, *tokr;
  float* rw;
  CUDA_CHECK(cudaMalloc(&X, (size_t)M * H * 2));
  CUDA_CHECK(cudaMalloc(&W12, (size_t)E * 2 * I * H * 2));
  CUDA_CHECK(cudaMalloc(&Wd, (size_t)E * H * I * 2));
  CUDA_CHECK(cudaMalloc(&PX, (size_t)Pp * H * 2));
  CUDA_CHECK(cudaMalloc(&A2, (size_t)Pp * I * 2));
  CUDA_CHECK(cudaMalloc(&G, (size_t)Pp * I * 4));
  CUDA_CHECK(cudaMalloc(&U, (size_t)Pp * I * 4));
  CUDA_CHECK(cudaMalloc(&D, (size_t)Pp * H * 4));
  CUDA_CHECK(cudaMalloc(&Yf, (size_t)M * H * 4));
  CUDA_CHECK(cudaMalloc(&Y, (size_t)M * H * 2));
  CUDA_CHECK(cudaMalloc(&rtok, (size_t)Pp * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&gl, (size_t)Pp * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&rw, (size_t)Pp * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&tokp, (size_t)(M + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&tokr, (size_t)Pp * sizeof(int)));

  fill_bf16_kernel<<<2048, 256>>>(X, (size_t)M * H, 11u);
  fill_bf16_kernel<<<8192, 256>>>(W12, (size_t)E * 2 * I * H, 22u);
  fill_bf16_kernel<<<8192, 256>>>(Wd, (size_t)E * H * I, 33u);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(rtok, dist.row_tok.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gl, dist.gl.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rw, dist.row_w.data(), (size_t)Pp * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tokp, dist.tok_ptr.data(), (size_t)(M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tokr, dist.tok_row.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));

  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  // ---- kernel 启动器 ----
  auto k1 = [&](bool fuse) {
    if (fuse) {
      auto fn = up_gate_kernel<BM, BK, ST1, true>;
      size_t shm = (size_t)ST1 * (BM + 256) * BK * 2;
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      dim3 grid(I / 128, Pp / BM);
      fn<<<grid, (BM / 64) * 128, shm>>>(X, W12, rtok, gl, A2, nullptr, nullptr, Pp, H, I, E);
    } else {
      auto fn = up_gate_kernel<BM, BK, ST1, false>;
      size_t shm = (size_t)ST1 * (BM + 256) * BK * 2;
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      dim3 grid(I / 128, Pp / BM);
      fn<<<grid, (BM / 64) * 128, shm>>>(X, W12, rtok, gl, nullptr, G, U, Pp, H, I, E);
    }
  };
  auto k2 = [&](bool fuse, int bn = 128) {
    if (bn == 256) {
      size_t shm = (size_t)ST2 * (BM + 256) * BK * 2;
      if (fuse) {
        auto fn = down_kernel<BM, 256, BK, ST2, true>;
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        fn<<<dim3(H / 256, Pp / BM), (BM / 64) * 128, shm>>>(A2, Wd, rtok, gl, rw, Yf, Pp, H, I, E);
      } else {
        auto fn = down_kernel<BM, 256, BK, ST2, false>;
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        fn<<<dim3(H / 256, Pp / BM), (BM / 64) * 128, shm>>>(A2, Wd, rtok, gl, rw, D, Pp, H, I, E);
      }
      return;
    }
    size_t shm = (size_t)ST2 * (BM + 128) * BK * 2;
    if (fuse) {
      auto fn = down_kernel<BM, 128, BK, ST2, true>;
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      fn<<<dim3(H / 128, Pp / BM), (BM / 64) * 128, shm>>>(A2, Wd, rtok, gl, rw, Yf, Pp, H, I, E);
    } else {
      auto fn = down_kernel<BM, 128, BK, ST2, false>;
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      fn<<<dim3(H / 128, Pp / BM), (BM / 64) * 128, shm>>>(A2, Wd, rtok, gl, rw, D, Pp, H, I, E);
    }
  };
  auto kperm = [&] { permute_kernel<<<M, 128>>>(X, rtok, tokp, tokr, PX, M); };
  auto kswi = [&] { swiglu_kernel<<<div_up((size_t)Pp * I / 8, 256), 256>>>(G, U, A2, (size_t)Pp * I); };
  auto kunt = [&] { unpermute_kernel<<<M, 256>>>(D, tokp, tokr, rw, Yf, M); };
  auto kcast = [&] { cast_kernel<<<2048, 256>>>(Yf, Y, (size_t)M * H); };

  // ---- 正确性：K1（gather + swiglu）对 CPU ----
  auto check_k1 = [&] {
    k1(true);
    CUDA_CHECK_LAST();
    std::vector<bf16> hx(H), hb0(H), hb1(H);
    double err = 0, ref = 0;
    for (int s = 0; s < 6; ++s) {
      const int p = (s * 7919 + 3) % Pp;
      if (dist.row_w[p] == 0.f) continue;
      const int g = dist.gl[p], tok = dist.row_tok[p];
      CUDA_CHECK(cudaMemcpy(hx.data(), X + (size_t)tok * H, H * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hb0.data(), W12 + ((size_t)(g * 2 * I + 0) * H), H * 2, cudaMemcpyDeviceToHost));
      for (int c = 0; c < 6; ++c) {
        const int cc = (c * 991 + 7) % I;
        CUDA_CHECK(cudaMemcpy(hb0.data(), W12 + ((size_t)(g * 2 * I + cc) * H), H * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb1.data(), W12 + ((size_t)(g * 2 * I + I + cc) * H), H * 2, cudaMemcpyDeviceToHost));
        double gs = 0, us = 0;
        for (int k = 0; k < H; ++k) {
          const double xv = __bfloat162float(hx[k]);
          gs += xv * __bfloat162float(hb0[k]);
          us += xv * __bfloat162float(hb1[k]);
        }
        const double e = (double)silu_f((float)gs) * us;
        bf16 got_b;
        CUDA_CHECK(cudaMemcpy(&got_b, A2 + (size_t)p * I + cc, 2, cudaMemcpyDeviceToHost));
        const double got = __bfloat162float(got_b);
        err = std::max(err, std::fabs(got - e));
        ref = std::max(ref, std::fabs(e));
      }
    }
    std::printf("  [K1 gather+swiglu ] max_abs_err=%.3e (ref~%.2f) %s\n", err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
  };

  // ---- 正确性：K2（down + unpermute）对 CPU（基于 A2） ----
  auto check_k2 = [&] {
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    k1(true);
    k2(true);
    CUDA_CHECK_LAST();
    std::vector<bf16> ha(I), hw(I);
    double err = 0, ref = 0;
    for (int s = 0; s < 6; ++s) {
      const int t = (s * 1009 + 5) % M;
      std::vector<int> rows;
      for (int q = dist.tok_ptr[t]; q < dist.tok_ptr[t + 1]; ++q) rows.push_back(dist.tok_row[q]);
      for (int c = 0; c < 4 && !rows.empty(); ++c) {
        const int cc = (c * 1009 + 11) % H;
        double acc = 0;
        for (int p : rows) {
          const int g = dist.gl[p];
          CUDA_CHECK(cudaMemcpy(ha.data(), A2 + (size_t)p * I, I * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(hw.data(), Wd + ((size_t)g * H + cc) * I, I * 2, cudaMemcpyDeviceToHost));
          double dot = 0;
          for (int k = 0; k < I; ++k) dot += (double)__bfloat162float(ha[k]) * __bfloat162float(hw[k]);
          acc += (double)dist.row_w[p] * dot;
        }
        float got;
        CUDA_CHECK(cudaMemcpy(&got, Yf + (size_t)t * H + cc, 4, cudaMemcpyDeviceToHost));
        err = std::max(err, std::fabs((double)got - acc));
        ref = std::max(ref, std::fabs(acc));
      }
    }
    std::printf("  [K2 down+unperm   ] max_abs_err=%.3e (ref~%.2f) %s\n", err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
  };

  // ---- 不融合 vs 融合：一致性 ----
  auto run_unfused = [&] {
    kperm(); k1(false); kswi(); k2(false, 256); kunt();
  };
  auto run_fused = [&] {
    CUDA_CHECK(cudaMemsetAsync(Yf, 0, (size_t)M * H * 4));
    k1(true); k2(true, 256); kcast();
  };

  if (want("check") || want("all")) {
    std::printf("\n-- 正确性 --\n");
    check_k1();
    check_k2();
    // 融合 vs 不融合 端到端一致（同一 A2 由各自路径产生）
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    run_unfused();
    CUDA_CHECK_LAST();
    std::vector<float> y_unf((size_t)M * H);
    CUDA_CHECK(cudaMemcpy(y_unf.data(), Yf, (size_t)M * H * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    run_fused();
    CUDA_CHECK_LAST();
    std::vector<float> y_fus((size_t)M * H);
    CUDA_CHECK(cudaMemcpy(y_fus.data(), Yf, (size_t)M * H * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (size_t i = 0; i < y_fus.size(); i += 9973) {
      err = std::max(err, std::fabs((double)y_fus[i] - (double)y_unf[i]));
      ref = std::max(ref, std::fabs((double)y_unf[i]));
    }
    std::printf("  [fused vs unfused ] max_abs_err=%.3e (ref~%.2f) %s\n", err, ref,
                err / std::max(ref, 1.0) < 2e-2 ? "OK" : "FAIL");
  }

  // ---- 计时 ----
  const double flops = 2.0 * (double)Pp * (2.0 * I) * H + 2.0 * (double)Pp * H * I;
  const double flops_useful = flops * (double)dist.nrows_actual / Pp;
  auto report = [&](const char* tag, double ms) {
    double tf = to_tflops(flops_useful, ms);
    std::printf("%-16s %9.4f ms  %8.2f TFLOPS(useful)  (%5.1f%% of bf16 peak)\n", tag, ms, tf,
                100.0 * tf / BF16_PEAK);
  };

  // 每段单独计时（用融合路径的输入做稳态）
  run_fused();
  CUDA_CHECK_LAST();
  if (want("k1") || want("all")) {
    double t = bench_ms([&] { k1(true); }, 3, 20);
    report("K1 fuse", t);
    double t0 = bench_ms([&] { k1(false); }, 3, 20);
    report("K1 nofuse", t0);
  }
  if (want("k2") || want("all")) {
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    double t = bench_ms([&] { k2(true, 128); }, 3, 20);
    report("K2 fuse bn128", t);
    double t0 = bench_ms([&] { k2(false, 128); }, 3, 20);
    report("K2 nofuse bn128", t0);
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    double t2 = bench_ms([&] { k2(true, 256); }, 3, 20);
    report("K2 fuse bn256", t2);
    double t3 = bench_ms([&] { k2(false, 256); }, 3, 20);
    report("K2 nofuse bn256", t3);
  }
  if (want("aux") || want("all")) {
    CUDA_CHECK(cudaMemset(Yf, 0, (size_t)M * H * 4));
    double tp = bench_ms([&] { kperm(); }, 3, 20);
    double ts = bench_ms([&] { kswi(); }, 3, 20);
    double tu = bench_ms([&] { kunt(); }, 3, 20);
    double tc = bench_ms([&] { kcast(); }, 3, 20);
    std::printf("aux: permute %.4f  swiglu %.4f  unpermute %.4f  cast %.4f ms\n", tp, ts, tu, tc);
  }

  if (want("all") || want("fused") || want("unfused")) {
    run_fused();
    CUDA_CHECK_LAST();
    double tf = bench_ms(run_fused, 3, 20);
    run_unfused();
    CUDA_CHECK_LAST();
    double tu = bench_ms(run_unfused, 3, 20);
    std::printf("\n[unfused 5-kernel ] %9.4f ms\n", tu);
    std::printf("[fused   3-kernel ] %9.4f ms   speedup %.3fx\n", tf, tu / tf);
    // 流量账（读+写，bf16 2B / fp32 4B）
    double B_unf = (double)Pp * H * 2 * 2                    // permute: 读X + 写PX
                 + (double)Pp * H * 2                          // K1 读 PX
                 + (double)Pp * I * 4 * 2 + (double)Pp * I * 4 * 2 // 写 G,U
                 + (double)Pp * I * 4 * 2                         // swiglu 读 G,U（≈）
                 + (double)Pp * I * 2                              // swiglu 写 A2
                 + (double)Pp * I * 2 + (double)Pp * I * 4 * 2     // K2 读 A2, 写 D
                 + (double)Pp * H * 4 + (double)M * H * 4          // unpermute 读 D, 写 Yf
                 + (double)M * H * 4;                             // cast 读 Yf
    double B_fus = (double)Pp * H * 2                             // K1 gather 读 X
                 + (double)Pp * I * 2                             // K1 写 A2
                 + (double)Pp * I * 2 + (double)M * H * 4         // K2 读 A2, atomic 写 Yf
                 + (double)M * H * 4;                             // cast 读 Yf
    double B_weights = ((double)(Pp / BM) * (I / 128) * 256 * H * 2)   // K1 的 B 重读
              + ((double)(Pp / BM) * (H / 128) * 128 * I * 2);  // K2 的 B 重读
    std::printf("act bytes: unfused=%.2f GB  fused=%.2f GB  (-%.0f%%)\n", B_unf / 1e9,
                B_fus / 1e9, 100.0 * (B_unf - B_fus) / B_unf);
    std::printf("all bytes: unfused=%.2f GB  fused=%.2f GB  (-%.0f%%);  B 重读占 %.1f GB\n",
                (B_unf + B_weights) / 1e9, (B_fus + B_weights) / 1e9,
                100.0 * (B_unf - B_fus) / (B_unf + B_weights), B_weights / 1e9);
  }

  CUDA_CHECK(cudaFree(X)); CUDA_CHECK(cudaFree(W12)); CUDA_CHECK(cudaFree(Wd));
  CUDA_CHECK(cudaFree(PX)); CUDA_CHECK(cudaFree(A2)); CUDA_CHECK(cudaFree(G));
  CUDA_CHECK(cudaFree(U)); CUDA_CHECK(cudaFree(D)); CUDA_CHECK(cudaFree(Yf));
  CUDA_CHECK(cudaFree(Y)); CUDA_CHECK(cudaFree(rtok)); CUDA_CHECK(cudaFree(gl));
  CUDA_CHECK(cudaFree(rw)); CUDA_CHECK(cudaFree(tokp)); CUDA_CHECK(cudaFree(tokr));
  return 0;
}
