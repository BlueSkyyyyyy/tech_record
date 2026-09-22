// 57（主题 21h）：V4-Pro 的 FP4 专家权重接回 grouped MoE —— 现场折 vs 预折叠
//
// 承接：
//   56 篇用真实 checkpoint 证明 routed experts 的权重是 **FP4 e2m1 + E8M0 block-32**
//        （`int8[I,H/2] + scale[I,H/32]`），且因 e2m1 ⊂ e4m3 且 scale 是 2 的幂，
//        FP4 × 2^e 可以**逐位无损**折进 e4m3（relRMS=0）。
//   34 篇把 31 篇的 per-block FP8 grouped GEMM 接进 MoE FFN 端到端。但 34 假设权重
//        已经是 FP8 —— 真实 checkpoint 里是 FP4，字节只有一半。
//
// 本篇的问题：既然放大权重带宽是 prefill FFN 的唯一杠杆（30 篇 K1 DRAM 80%），
//   把「读 FP8 权重」换成「读 FP4 权重 + 在 kernel 里现场折成 FP8」，
//   能不能把 K1 的权重带宽真正砍半？
//
// 设计（一次干净的消融）：
//   两个路径的 consumer（wgmma + per-128 激活 scale 折算）逐字节相同，只差 B 的格式：
//     MODE 0（预折叠 / pf）：全局读 FP8 权重 [N][K]，cp.async 搬进 smem，
//         仅做 SW128 置换（行主序 -> K-major SW128）。
//     MODE 1（现场折 / fp4）：全局读 FP4 packed [N][K/2] + E8M0 [N][K/32]（字节
//         一半），4096 项 LUT（(exp,nibble)->e4m3）现场折 + 置换。
//   A（激活）也走 cp.async + 手写 SW128；二者共用同一份 double-buffer 骨架。
//
// 正确性：折叠无损 ⇒ MODE 1 与 MODE 0 的 GEMM 结果应当**逐位相等**（最强验证）。
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, n_routed_experts=384,
//   num_experts_per_tok=6, e4m3 + ue8m0 + weight_block 128x128。
//
// 运行（K1 up/gate）：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//     scripts/run.sh 57-v4-fp4-moe-fold/fold_moe.cu [M] [bal|rand] [only] [BN] [kstop]
#include "../common/cuda_utils.cuh"

#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;

constexpr int H = 7168;   // hidden
constexpr int I = 3072;   // moe_intermediate_size
constexpr int E = 384;    // n_routed_experts
constexpr int TOPK = 6;   // num_experts_per_tok

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（同 20/22/31/34）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

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
// fp8 SW128（K-major）元素字节偏移；一个 1024B atom = 8 行 × 128B。
__device__ __forceinline__ int sw128_off(int r, int k, int K) {
  (void)K;
  const int rr = r & 7;
  return (r >> 3) * 1024 + (rr * 8 + (((k >> 4) & 7) ^ rr)) * 16 + (k & 15);
}

// ---------------------------------------------------------------------------
// cp.async helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// e2m1 值（幅值表）；e2m1 ⊂ e4m3
__host__ __device__ __forceinline__ float e2m1f(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  float v = (e == 0) ? (m * 0.5f) : ((1.f + m * 0.5f) * exp2f((int)e - 1));
  return s ? -v : v;
}
__device__ __forceinline__ unsigned char to_fp8(float x) {
  return (unsigned char)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__host__ __device__ __forceinline__ unsigned char to_fp8_hd(float x) {
#if defined(__CUDA_ARCH__)
  return (unsigned char)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
#else
  __nv_fp8_e4m3 v = __nv_fp8_e4m3(x);
  return *reinterpret_cast<unsigned char*>(&v);
#endif
}
__host__ __device__ __forceinline__ float fp8_to_float(unsigned char b) {
#if defined(__CUDA_ARCH__)
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
#else
  // 手写 e4m3 解码（host 侧 __nv_fp8_e4m3 的转换在本工具链下会给出原始位模式）
  const unsigned s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
  float v = (e == 0) ? (m / 8.0f) * (1.0f / 64.0f) : (1.0f + m / 8.0f) * exp2f((int)e - 7);
  return s ? -v : v;
#endif
}

// ---------------------------------------------------------------------------
// grouped GEMM：A=(P,K) fp8 + sa=(P,K/128)；权重两种格式
//   MODE 0：W8=(E*N,K) fp8（已折叠，per-tensor）
//   MODE 1：Wp=(E*N,K/2) packed fp4 + WsT=(K/32,E*N) E8M0（host 转置）
//   D[p,n] = Σ_k sa[p,k/128]·A[p,k]·B[g,n,k]
// 全部走 cp.async + 手写 SW128，双缓冲（STAGES=2）。
// ---------------------------------------------------------------------------
constexpr int STAGES = 2;
template <int BM, int BN, int BK, int MODE>
__global__ void __launch_bounds__((BM / 64) * 128)
grouped_fold_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ W8,
                    const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Ws,
                    float* __restrict__ D, const int* __restrict__ gl,
                    const float* __restrict__ sa, int N, int K, int mlim, int KBLK, int Wrow, int kstop, int diag) {
  static_assert(BK == 128, "TMA/SW128 内维固定 128 字节");
  constexpr int NWG = BM / 64;
  constexpr int NT = NWG * 128;
  constexpr int SBO = 1024;
  constexpr bool FP4 = (MODE == 1);
  constexpr int BROW = FP4 ? (BK / 2) : BK;
  constexpr int ASC = BM * BK, BSC = BN * BK, ARWC = BM * BK, BRWC = BN * BROW;
  constexpr int SCC = FP4 ? BN * (BK / 32) : 0;

  extern __shared__ __align__(1024) char smem[];
  char* As = smem;
  char* Bs = As + (size_t)STAGES * ASC;
  char* Araw = Bs + (size_t)STAGES * BSC;
  char* Braw = Araw + (size_t)STAGES * ARWC;
  char* Bsc = Braw + (size_t)STAGES * BRWC;
  uint8_t* foldlut = reinterpret_cast<uint8_t*>(Bsc + (size_t)STAGES * SCC);

  const int tid = threadIdx.x;
  const int block_col = blockIdx.x * BN;
  const int block_row = blockIdx.y * BM;
  const int nblk = (kstop > 0 && kstop < K / BK) ? kstop : (K / BK);
  const int group = gl[block_row];
  const int brow = group * N + block_col;

  for (int i = tid; i < 4096; i += NT) {
    const int exp = i >> 4, nib = i & 15;
    foldlut[i] = to_fp8(e2m1f(nib) * exp2f((int)exp - 127));
  }
  __syncthreads();

  auto load_stage = [&](int kb, int st) {
    for (int i = tid; i < BM * BK / 16; i += NT) {
      const int r = i / (BK / 16), c = i % (BK / 16);
      cp_async16(Araw + (size_t)st * ARWC + (size_t)r * BK + c * 16,
                 A + (size_t)(block_row + r) * K + (size_t)kb * BK + c * 16);
    }
    if constexpr (FP4) {
      for (int i = tid; i < BN * (BK / 2) / 16; i += NT) {
        const int r = i / (BK / 2 / 16), c = i % (BK / 2 / 16);
        cp_async16(Braw + (size_t)st * BRWC + (size_t)r * (BK / 2) + c * 16,
                   Wp + (size_t)(brow + r) * (K / 2) + (size_t)kb * (BK / 2) + c * 16);
      }
      for (int i = tid; i < BN * (BK / 32) / 16; i += NT) {
        const int kg = i / (BN / 16), c = i % (BN / 16);
        cp_async16(Bsc + (size_t)st * SCC + (size_t)kg * BN + c * 16,
                   Ws + (size_t)((size_t)kb * (BK / 32) + kg) * Wrow + brow + c * 16);
      }
    } else {
      for (int i = tid; i < BN * BK / 16; i += NT) {
        const int r = i / (BK / 16), c = i % (BK / 16);
        cp_async16(Braw + (size_t)st * BRWC + (size_t)r * BK + c * 16,
                   W8 + (size_t)(brow + r) * K + (size_t)kb * BK + c * 16);
      }
    }
    cp_async_commit();
  };
  auto swz_stage = [&](int st) {
    if (diag & 1) { fence_proxy_async(); return; }  // 诊断：跳过置换/折叠（结果无效）
    // A：16B chunk 粒度的 SW128 置换（chunk c -> c^(r&7)）
    {
      const uint8_t* ar = reinterpret_cast<const uint8_t*>(Araw + (size_t)st * ARWC);
      uint8_t* as = reinterpret_cast<uint8_t*>(As + (size_t)st * ASC);
      const int nc = BM * (BK / 16);
      for (int i = tid; i < nc; i += NT) {
        const int r = i / (BK / 16), c = i % (BK / 16);
        const int dst = (r >> 3) * 1024 + (r & 7) * 128 + (((c ^ (r & 7)) & 7) << 4);
        *reinterpret_cast<uint4*>(as + dst) = *reinterpret_cast<const uint4*>(ar + (size_t)i * 16);
      }
    }
    uint8_t* bs = reinterpret_cast<uint8_t*>(Bs + (size_t)st * BSC);
    if constexpr (FP4) {
      // FP4：每 16B packed（32 个 fp4 = 32 个 k）-> 32B e4m3，分落 2 个 SW128 chunk
      const uint8_t* raw = reinterpret_cast<const uint8_t*>(Braw + (size_t)st * BRWC);
      const uint8_t* scv = reinterpret_cast<const uint8_t*>(Bsc + (size_t)st * SCC);
      const int nc = BN * (BK / 32);
      for (int i = tid; i < nc; i += NT) {
        const int n = i / (BK / 32), t = i % (BK / 32);
        const uint4 w = *reinterpret_cast<const uint4*>(raw + ((size_t)n * (BK / 32) + t) * 16);
        const int exp = scv[(size_t)t * BN + n];
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        uint8_t o[32];
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int b = 0; b < 8; ++b)
            o[q * 8 + b] = foldlut[(exp << 4) | ((ww[q] >> (4 * b)) & 0xF)];
        const int base = (n >> 3) * 1024 + (n & 7) * 128;
        const int c0 = 2 * t, c1 = 2 * t + 1;
        *reinterpret_cast<uint4*>(bs + base + (((c0 ^ (n & 7)) & 7) << 4)) =
            *reinterpret_cast<const uint4*>(o);
        *reinterpret_cast<uint4*>(bs + base + (((c1 ^ (n & 7)) & 7) << 4)) =
            *reinterpret_cast<const uint4*>(o + 16);
      }
    } else {
      const uint8_t* raw = reinterpret_cast<const uint8_t*>(Braw + (size_t)st * BRWC);
      const int nc = BN * (BK / 16);
      for (int i = tid; i < nc; i += NT) {
        const int r = i / (BK / 16), c = i % (BK / 16);
        const int dst = (r >> 3) * 1024 + (r & 7) * 128 + (((c ^ (r & 7)) & 7) << 4);
        *reinterpret_cast<uint4*>(bs + dst) = *reinterpret_cast<const uint4*>(raw + (size_t)i * 16);
      }
    }
    fence_proxy_async();
  };

  const int lane = tid & 31, warp = tid >> 5;
  const int wg = tid >> 7;
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  const int r0g = block_row + row0;
  const int r1g = r0g + 8;
  float acc[64], fin[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) { acc[i] = 0.f; fin[i] = 0.f; }

  load_stage(0, 0);
  if (nblk > 1) load_stage(1, 1);

  for (int kb = 0; kb < nblk; ++kb) {
    const int st = kb & 1;
    if (kb + 1 < nblk) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();
    swz_stage(st);
    __syncthreads();
    if (kb + 2 < nblk) load_stage(kb + 2, st);  // raw[st] 已读完，可覆盖
    char* a = As + (size_t)st * ASC + (size_t)wg * 64 * BK;
    char* b = Bs + (size_t)st * BSC;
    wgmma_fence();
#pragma unroll
    for (int s = 0; s < BK / 32; ++s) {
      uint64_t da = make_desc_sw128(k32_addr(smem_u32(a), s), SBO);
      uint64_t db = make_desc_sw128(k32_addr(smem_u32(b), s), SBO);
      wgmma_m64n128k32(acc, da, db);
    }
    wgmma_commit();
    wgmma_wait0();
    const float sa0 = (r0g < mlim) ? sa[(size_t)r0g * KBLK + kb] : 0.f;
    const float sa1 = (r1g < mlim) ? sa[(size_t)r1g * KBLK + kb] : 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      fin[j * 4 + 0] += sa0 * acc[j * 4 + 0];
      fin[j * 4 + 1] += sa0 * acc[j * 4 + 1];
      fin[j * 4 + 2] += sa1 * acc[j * 4 + 2];
      fin[j * 4 + 3] += sa1 * acc[j * 4 + 3];
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int col = j * 8 + (lane & 3) * 2;
    const int cc = block_col + col;
    if (r0g < mlim)
      *reinterpret_cast<float2*>(&D[(size_t)r0g * N + cc]) = make_float2(fin[j * 4 + 0], fin[j * 4 + 1]);
    if (r1g < mlim)
      *reinterpret_cast<float2*>(&D[(size_t)r1g * N + cc]) = make_float2(fin[j * 4 + 2], fin[j * 4 + 3]);
  }
}

// ---------------------------------------------------------------------------
// 参考 kernel：从 FP4+scale 现场折成 row-major fp8（验证无损 + 生成「预折叠」）
// ---------------------------------------------------------------------------
__global__ void fold_fp4_to_fp8_kernel(const uint8_t* __restrict__ Wp,
                                       const uint8_t* __restrict__ Ws, uint8_t* __restrict__ W8,
                                       int R, int K) {
  const size_t tot = (size_t)R * K;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < tot;
       i += (size_t)gridDim.x * blockDim.x) {
    const int r = i / K, k = i % K;
    const uint8_t b = Wp[(size_t)r * (K / 2) + (k >> 1)];
    const int nib = (k & 1) ? (b >> 4) : (b & 0xF);
    const int exp = Ws[(size_t)r * (K / 32) + (k >> 5)];
    W8[i] = to_fp8(e2m1f(nib) * exp2f((int)exp - 127));
  }
}

__global__ void fill_fp4_kernel(uint8_t* __restrict__ Wp, uint8_t* __restrict__ Ws, size_t nw,
                                size_t ns, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < nw; i += stride) {
    unsigned x = (unsigned)i ^ (seed * 0x9E3779B9u);
    x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    Wp[i] = (uint8_t)x;
  }
  for (i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ns; i += stride) {
    unsigned x = (unsigned)i ^ (seed * 0x85EBCA6Bu);
    x *= 0x9E3779B9u; x ^= x >> 15; x *= 0xC2B2AE35u; x ^= x >> 16;
    Ws[i] = (uint8_t)(120 + (x % 15));  // 指数 120..134 -> 2^-7..2^7
  }
}
// e8m0 scale 转置：Ws[R][K/32] -> WsT[K/32][R]（host 侧一次性重排，部署友好）
__global__ void repack_scale_kernel(const uint8_t* __restrict__ Ws, uint8_t* __restrict__ WsT,
                                    int R, int KG) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)R * KG;
       i += (size_t)gridDim.x * blockDim.x) {
    const int r = i / KG, kg = i % KG;
    WsT[(size_t)kg * R + r] = Ws[i];
  }
}
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
// 路由分布（同 30/34）
// ---------------------------------------------------------------------------
struct Dist {
  int M, Pp, nrows_actual = 0;
  std::vector<int> gl, row_tok;
  std::vector<float> row_w;
  std::vector<int> tok_ptr, tok_row;
};
static Dist build_dist(int M, int BM, unsigned seed, bool balanced) {
  Dist d; d.M = M;
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
  std::vector<int> off(E); int acc = 0;
  for (int g = 0; g < E; ++g) { off[g] = acc; acc += (count[g] + BM - 1) / BM * BM; }
  d.Pp = acc;
  d.gl.assign(d.Pp, -1); d.row_tok.assign(d.Pp, 0); d.row_w.assign(d.Pp, 0.f);
  std::vector<std::vector<std::pair<int, int>>> slots(E);
  for (int t = 0; t < M; ++t)
    for (size_t j = 0; j < tok_experts[t].size(); ++j) slots[tok_experts[t][j]].push_back({t, (int)j});
  std::vector<int> fill_pos(E, 0);
  for (int g = 0; g < E; ++g)
    for (auto& s : slots[g]) {
      const int p = off[g] + fill_pos[g]++;
      d.gl[p] = g; d.row_tok[p] = s.first; d.row_w[p] = ur(rng); d.nrows_actual++;
    }
  return d;
}

// ---------------------------------------------------------------------------
// 启动一个 grouped 投影
// ---------------------------------------------------------------------------
struct GemmCfg { int BM, BN, BK; };

template <int MODE>
static void launch_grouped(const GemmCfg& c, const uint8_t* A, const uint8_t* W8,
                           const uint8_t* Wp, const uint8_t* Ws, float* D, const int* gl,
                           const float* sa, int N, int K, int mlim, int KBLK, int grid_y, int Wrow,
                           int kstop, int diag) {
  const size_t asc = (size_t)c.BM * c.BK, bsc = (size_t)c.BN * c.BK;
  const size_t brwc = (size_t)c.BN * ((MODE == 1) ? (c.BK / 2) : c.BK);
  const size_t scc = (MODE == 1) ? (size_t)c.BN * (c.BK / 32) : 0;
  const size_t shm = (size_t)STAGES * (asc + bsc + asc + brwc + scc) + 4096;
  const int nt = (c.BM / 64) * 128;
#define LAUNCH(BM_, BN_, BK_)                                                                    \
  do {                                                                                            \
    if (c.BM == BM_ && c.BN == BN_ && c.BK == BK_) {                                               \
      auto fn = grouped_fold_kernel<BM_, BN_, BK_, MODE>;                                          \
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);             \
      fn<<<dim3(N / BN_, grid_y), nt, shm>>>(A, W8, Wp, Ws, D, gl, sa, N, K, mlim, KBLK, Wrow, kstop, diag); \
      return;                                                                                      \
    }                                                                                              \
  } while (0)
  LAUNCH(64, 128, 128);
  LAUNCH(128, 128, 128);
  LAUNCH(128, 256, 128);
#undef LAUNCH
  std::fprintf(stderr, "no kernel for cfg BM=%d BN=%d BK=%d\n", c.BM, c.BN, c.BK);
  std::exit(1);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int M = (argc > 1) ? std::atoi(argv[1]) : 8192;
  const bool balanced = (argc > 2) && (std::strcmp(argv[2], "bal") == 0);
  const char* only = (argc > 3) ? argv[3] : "all";
  const int BN = (argc > 4) ? std::atoi(argv[4]) : 128;
  const int KSTOP = (argc > 5) ? std::atoi(argv[5]) : 0;
  const int DIAG = getenv("DIAG") ? std::atoi(getenv("DIAG")) : 0;
  const int BM = 128;
  constexpr int BK = 128;
  const int KBLK_H = H / 128;  // 56
  const int N1 = 2 * I;        // up/gate 输出维

  DeviceInfo d0 = device_info(0);
  print_device_info(d0);
  std::printf("\n[57] FP4 experts -> grouped MoE fold on-the-fly vs pre-fold\tM=%d %s\n", M,
              balanced ? "balanced" : "random");

  Dist dist = build_dist(M, BM, 1234, balanced);
  const int Pp = dist.Pp;
  std::printf("Pp=%d actual=%d padding=%.1f%%\n", Pp, dist.nrows_actual,
              100.0 * (Pp - dist.nrows_actual) / Pp);
  if (Pp % BM) { std::printf("Pp %% BM != 0\n"); return 1; }
  const int grid_y = Pp / BM;

  const size_t R = (size_t)E * N1;
  const size_t wf8c = R * H, wpc = R * (H / 2), wsc = R * (H / 32);
  std::printf("weights up/gate [E*N=%zu, K=%d]: fp8=%.2f GB  fp4=%.2f GB (+scale %.2f GB)\n", R, H,
              wf8c / 1e9, wpc / 1e9, wsc / 1e9);
  uint8_t *dWp, *dWs, *dW8, *dWsT;
  CUDA_CHECK(cudaMalloc(&dWp, wpc));
  CUDA_CHECK(cudaMalloc(&dWs, wsc));
  CUDA_CHECK(cudaMalloc(&dWsT, wsc));
  CUDA_CHECK(cudaMalloc(&dW8, wf8c));
  fill_fp4_kernel<<<8192, 256>>>(dWp, dWs, wpc, wsc, 22u);
  if (getenv("FIXSC")) cudaMemset(dWs, 127, wsc);
  if (getenv("ZERONIB")) cudaMemset(dWp, 0x00, wpc);  // 全 0 nibble，e2m1=0
  fold_fp4_to_fp8_kernel<<<8192, 256>>>(dWp, dWs, dW8, (int)R, H);
  repack_scale_kernel<<<8192, 256>>>(dWs, dWsT, (int)R, H / 32);
  CUDA_CHECK_LAST();

  fp8* A1; float* sa1;
  CUDA_CHECK(cudaMalloc(&A1, (size_t)Pp * H));
  CUDA_CHECK(cudaMalloc(&sa1, (size_t)Pp * KBLK_H * 4));
  fill_fp8_kernel<<<2048, 256>>>(A1, (size_t)Pp * H, 11u);
  fill_scale_kernel<<<1024, 256>>>(sa1, (size_t)Pp * KBLK_H, 44u);
  if (getenv("SA1")) { std::vector<float> ones((size_t)Pp*KBLK_H, 1.0f); cudaMemcpy(sa1, ones.data(), ones.size()*4, cudaMemcpyHostToDevice); }
  if (getenv("ONES")) { cudaMemset(A1, 0x38, (size_t)Pp*H); cudaMemset(dW8, 0x38, wf8c); }
  CUDA_CHECK_LAST();

  int* gl; float* GU;
  CUDA_CHECK(cudaMalloc(&gl, (size_t)Pp * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&GU, (size_t)Pp * N1 * 4));
  CUDA_CHECK(cudaMemcpy(gl, dist.gl.data(), (size_t)Pp * sizeof(int), cudaMemcpyHostToDevice));

  {  // 无损性抽样
    const size_t n = std::min<size_t>(wf8c, (size_t)1 << 20);
    std::vector<uint8_t> hwp(wpc < n ? wpc : n), hws(wsc < n ? wsc : n), hb(n);
    CUDA_CHECK(cudaMemcpy(hwp.data(), dWp, hwp.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hws.data(), dWs, hws.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), dW8, n, cudaMemcpyDeviceToHost));
    double maxd = 0;
    for (int k = 0; k < (int)n; ++k) {
      const int r = k / H, kk = k % H;
      const uint8_t b = hwp[(size_t)r * (H / 2) + (kk >> 1)];
      const int nib = (kk & 1) ? (b >> 4) : (b & 0xF);
      const int exp = hws[(size_t)r * (H / 32) + (kk >> 5)];
      const uint8_t ref = to_fp8_hd(e2m1f(nib) * exp2f((int)exp - 127));
      maxd = std::max(maxd, std::fabs((double)(int)ref - (int)hb[k]));
    }
    std::printf("[fold check] 抽样 %zu 元素 fp4->fp8：最大位差 = %.0f  %s\n", n, maxd,
                maxd == 0 ? "OK (逐位无损)" : "FAIL");
  }

  GemmCfg cfg{BM, BN, BK};
  auto run0 = [&] { launch_grouped<0>(cfg, reinterpret_cast<const uint8_t*>(A1), dW8, nullptr, nullptr, GU, gl, sa1, N1, H, Pp, KBLK_H, grid_y, (int)R, KSTOP, DIAG); };
  auto run1 = [&] { launch_grouped<1>(cfg, reinterpret_cast<const uint8_t*>(A1), nullptr, dWp, dWsT, GU, gl, sa1, N1, H, Pp, KBLK_H, grid_y, (int)R, KSTOP, DIAG); };

  if (std::strcmp(only, "n0") == 0 || std::strcmp(only, "n1") == 0) {
    for (int i = 0; i < 3; ++i) { if (only[1] == '0') run0(); else run1(); }
    CUDA_CHECK_LAST(); CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }
  std::vector<float> h0((size_t)Pp * N1), h1((size_t)Pp * N1);
  CUDA_CHECK(cudaMemset(GU, 0, (size_t)Pp * N1 * 4));
  run0(); CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h0.data(), GU, h0.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemset(GU, 0, (size_t)Pp * N1 * 4));
  run1(); CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h1.data(), GU, h1.size() * 4, cudaMemcpyDeviceToHost));
  {
    double mx = 0; int bad = 0, shown = 0;
    for (size_t i = 0; i < h0.size(); ++i) {
      const double dd = std::fabs((double)h0[i] - (double)h1[i]);
      if (dd > 1e-3 * std::max(1.0, std::fabs((double)h0[i]))) {
        bad++;
        if (shown < 8) { printf("    mismatch p=%zu cc=%zu M0=%.4f M1=%.4f\n", i/N1, i%N1, h0[i], h1[i]); shown++; }
      }
      mx = std::max(mx, dd);
    }
    std::printf("[bitwise] MODE0(pre-fold fp8) vs MODE1(fp4 on-the-fly): max_abs_diff=%.3e bad=%d %s\n",
                mx, bad, bad == 0 ? "OK" : "FAIL");
  }
  {
    std::vector<unsigned char> ha(H);
    std::vector<float> hsa(KBLK_H);
    std::vector<uint8_t> hw8(H);
    double mrel = 0, mref = 0;
    const int kbref = (KSTOP > 0 && KSTOP < KBLK_H) ? KSTOP : KBLK_H;
    int nbad=0, ntot=0; for (int s = 0; s < 200; ++s) {
      const int p = (s * 7919 + 3) % Pp;
      const int g = dist.gl[p];
      if (g < 0) continue;
      const int cc = (s * 2137 + 7) % N1;
      CUDA_CHECK(cudaMemcpy(ha.data(), A1 + (size_t)p * H, H, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hsa.data(), sa1 + (size_t)p * KBLK_H, KBLK_H * 4, cudaMemcpyDeviceToHost));
      if (s==0) { double ss=0; for(int q=0;q<KBLK_H;++q) ss+=hsa[q]; printf("  hsa sum=%.4f first=%.4f %.4f A1[0]=%.4f W8[0]=%.4f\n", ss, hsa[0], hsa[1], fp8_to_float(ha[0]), fp8_to_float(hw8[0])); }
      const size_t row = (size_t)g * N1 + cc;
      CUDA_CHECK(cudaMemcpy(hw8.data(), dW8 + row * H, H, cudaMemcpyDeviceToHost));
      double e = 0;
      for (int kb = 0; kb < kbref; ++kb) {
        double blk = 0;
        for (int k = kb * 128; k < (kb + 1) * 128; ++k)
          blk += (double)fp8_to_float(ha[k]) * (double)fp8_to_float(hw8[k]);
        e += blk * (double)hsa[kb];
      }
      const double got0 = (double)h0[(size_t)p * N1 + cc];
      const double got1 = (double)h1[(size_t)p * N1 + cc];
      const double rel = std::fabs(got0 - e) / std::max(std::fabs(e), 1.0);
      const double rel1 = std::fabs(got1 - e) / std::max(std::fabs(e), 1.0);
      ntot++;
      if (rel > 3e-2) { nbad++; if (nbad<6) std::printf("    M0 bad p=%d cc=%d got=%.4f ref=%.4f\n", p, cc, got0, e); }
      if (rel1 > 3e-2 && nbad>=0 && ntot<6) std::printf("    M1 bad p=%d cc=%d got=%.4f ref=%.4f\n", p, cc, got1, e);
      if (s<4) std::printf("    sample p=%d g=%d cc=%d  M0=%.4f M1=%.4f ref=%.4f\n", p, g, cc, got0, got1, e);
      if (std::fabs(e) > 3.0) { mrel = std::max(mrel, rel); mref = std::max(mref, std::fabs(e)); }
    }
    std::printf("[vs CPU ref] M0 bad=%d/%d max_rel=%.3e (ref~%.1f) %s\n", nbad, ntot, mrel, mref, mrel < 3e-2 ? "OK" : "FAIL");
  }

  const double flops = 2.0 * (double)Pp * N1 * H;
  const double tiles = (double)grid_y * (N1 / BN);
  const double b_read_fp8 = tiles * (double)BN * H;
  const double b_read_fp4 = tiles * ((double)BN * H / 2 + (double)BN * H / 32);
  auto report = [&](const char* tag, double ms, double bytes) {
    std::printf("%-26s %9.4f ms  %7.2f TFLOPS (%.1f%% fp8 peak)  B-read %6.1f GB/s\n", tag, ms,
                to_tflops(flops, ms), 100.0 * to_tflops(flops, ms) / FP8_PEAK, to_gbps(bytes, ms));
  };
  std::printf("\n-- 计时（BN=%d）--\n", BN);
  run0(); run1(); CUDA_CHECK_LAST();
  double t0 = bench_ms([&] { run0(); }, 3, 50);
  report("MODE0 pre-fold fp8", t0, b_read_fp8);
  double t1 = bench_ms([&] { run1(); }, 3, 50);
  report("MODE1 fp4 on-the-fly", t1, b_read_fp4);
  std::printf("speedup fp4/pre-fold = %.3fx   (fp8 B=%.2f GB, fp4 B=%.2f GB)\n", t0 / t1,
              b_read_fp8 / 1e9, b_read_fp4 / 1e9);

  CUDA_CHECK(cudaFree(dWp)); CUDA_CHECK(cudaFree(dWs)); CUDA_CHECK(cudaFree(dWsT)); CUDA_CHECK(cudaFree(dW8));
  CUDA_CHECK(cudaFree(A1)); CUDA_CHECK(cudaFree(sa1)); CUDA_CHECK(cudaFree(gl)); CUDA_CHECK(cudaFree(GU));
  return 0;
}
