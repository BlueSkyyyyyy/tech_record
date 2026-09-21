// 36 DSA compressor 投影的 L2 墙：TMA cluster multicast / L2 persisting
//
// 承接 35 篇（`35-dsa-compressor/compressor.cu`）。35 的 ncu 判决合并投影 GEMM
//   Y[M,N] = X[M,D] @ Wm[N,D]^T   （N = 2C，C = coff*512）
// 是 **L2 受限**：`L2 Cache Throughput` 85–88%、`Compute (SM)` 仅 57–59%、`DRAM` 26–29%。
// 根因是 L2→SM 的**重读放大**：
//   * A tile（BM×D）只由 m-tile 决定，却被 N/BN 个 n-tile 各拉一遍 → A 流量 ×(N/BN)；
//   * B tile（BN×D）只由 n-tile 决定，却被 M/BM 个 m-tile 各拉一遍 → B 流量 ×(M/BM)。
// ratio=128 时 N=1024、BN=128 → A 重读 8×；BM=256、M=32768 → B 重读 128×。
// 两者各自的 L2 字节都约 3.76 GB（A 470MB×8、B 14.7MB×128），加起来 ~7.5 GB / 0.74ms ≈ 10 TB/s。
//
// 本文件在 35 的 TMA + mbarrier + warp specialization + wgmma SW128 骨架上加 **cluster multicast**：
//   AXIS=0：无广播（基线，复现 35）
//   AXIS=1：沿 cluster.x 广播 A（同 m-tile 的 N/BN 个 n-tile 共享 A）→ A 的 L2 字节 ÷CN
//   AXIS=2：沿 cluster.y 广播 B（同 n-tile 的相邻 m-tile 共享 B）→ B 的 L2 字节 ÷CN
// 协议与 26/29 篇一致：leader 发 `...multicast::cluster` + mask；每 CTA 各自 `arrive.expect_tx`；
// 私有 operand 的 empty 本地 arrive，共享 operand 的 empty 由每个消费者 warp lane0 用 `mapa` 投到 rank0。
//
// 另测 **L2 persisting window**（cudaAccessPolicyWindow 把 Wm 钉进 L2）：ratio=128 时 Wm 仅 14.7MB、
// ratio=4 时 29.4MB，都 < 50MB L2，理论上可常驻，看能否压 L2 抖动。
//
// 真实 shape（/ssd/models/DeepSeek-V4-Pro/config.json）：hidden_size=7168, head_dim=512,
//   qk_rope_head_dim=64, compress_ratios ∈ {128,4,0}。
//
// 运行：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//     scripts/run.sh 36-dsa-compressor-mcast/compressor_mcast.cu [ratio] [M] [which] [skipcorr]
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

// 消费者在 mbar_wait 后是否需要 fence.proxy.async（实验开关）
// 判据：TMA 写 smem 与 wgmma 读 smem 都走 async proxy，mbarrier 已完成两者排序，
// 消费者再插 fence.proxy.async（generic↔async）是多余的；关掉后 256x128s4 +5.6%。
#ifndef CFENCE
#define CFENCE 0
#endif

// DeepSeek-V4-Pro compressor shape
constexpr int D  = 7168;  // hidden_size (K)
constexpr int HD = 512;   // head_dim (compressed dim d)
constexpr int RD = 64;    // qk_rope_head_dim

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（同 20/28/35）
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

// ---------------------------------------------------------------------------
// mbarrier + TMA helpers（同 23/26/35）
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
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t* bar, uint32_t cta_id) {
  asm volatile(
      "{\n.reg .b32 rem;\n"
      "mapa.shared::cluster.u32 rem, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64 _, [rem];\n}\n" ::"r"(smem_u32(bar)), "r"(cta_id));
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
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d_mcast(const CUtensorMap* tmap, void* dst, int c0, int c1,
                                                  uint64_t* bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}

// ---------------------------------------------------------------------------
// (1) 合并投影 GEMM：Y[M,N] = X[M,K] @ Wm[N,K]^T
//     TMA + mbarrier + warp specialization + wgmma SS SW128（+ 可选 cluster multicast）
//     AXIS=0 无广播；AXIS=1 沿 cluster.x 广播 A；AXIS=2 沿 cluster.y 广播 B
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int AXIS = 0, int CN = 1, bool BF16OUT = false,
          int KSPLIT = 1>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
proj_mcast_kernel(const __grid_constant__ CUtensorMap tmA,
                  const __grid_constant__ CUtensorMap tmB,
                  void* __restrict__ C, int M, int N, int K) {
  static_assert(BK == 64, "bf16 SW128 内维固定 64 元素");
  static_assert(KSPLIT == 1 || KSPLIT == 2, "KSPLIT 只支持 1/2");
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NCONS = NWG * 128;
  constexpr int NCW = NCONS / 32;  // 消费者 warp 数
  constexpr int SBO = 1024;
  constexpr bool SHARE_A = (AXIS == 1) && (CN > 1);
  constexpr bool SHARE_B = (AXIS == 2) && (CN > 1);
  static_assert(!(SHARE_A && SHARE_B), "一次只广播一个操作数");

  constexpr int BARR = ((int)(3 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;    // 私有 operand 释放：本 CTA 消费者线程本地 arrive
  uint64_t* sempty = empty + STAGES;  // 共享（广播）operand 释放：每 warp lane0 投到 rank0
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * 128;

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  const uint32_t rank = (SHARE_A || SHARE_B) ? cluster_rank() : 0u;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full + s, 1);
      mbar_init(empty + s, NCONS);
      mbar_init(sempty + s, CN * NCW);
    }
    fence_mbar_init();
  }
  __syncthreads();
  if constexpr (SHARE_A || SHARE_B) cluster_sync();

  float acc[KSPLIT][NSPLIT][64];  // 消费者 epilogue 要用，声明在分支外
  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int q = 0; q < nblk; ++q) {
        const int st = q % STAGES;
        if (q >= STAGES) {
          const uint32_t ph = (uint32_t)((q / STAGES - 1) & 1);
          mbar_wait(empty + st, ph);
          if ((SHARE_A || SHARE_B) && rank == 0) mbar_wait(sempty + st, ph);
        }
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * 128 + BN * 128);
        if constexpr (SHARE_A) {
          if (rank == 0)
            tma_load_2d_mcast(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st,
                              (uint16_t)((1u << CN) - 1));
        } else {
          tma_load_2d(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st);
        }
        if constexpr (SHARE_B) {
          if (rank == 0)
            tma_load_2d_mcast(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st,
                              (uint16_t)((1u << CN) - 1));
        } else {
          tma_load_2d(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st);
        }
      }
    }
  } else {
    const int wg = tid >> 7;
    const int lane = tid & 31;
#pragma unroll
    for (int ks = 0; ks < KSPLIT; ++ks)
#pragma unroll
      for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
        for (int i = 0; i < 64; ++i) acc[ks][j][i] = 0.f;

    for (int qb = 0; qb < nblk; qb += KSPLIT) {
#pragma unroll
      for (int kk = 0; kk < KSPLIT; ++kk) {
        const int q = qb + kk;
        const int st = q % STAGES;
        mbar_wait(full + st, (uint32_t)((q / STAGES) & 1));
#if CFENCE
        fence_proxy_async();
#endif
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
            wgmma_m64n128k16(acc[kk][jn], da, db);  // kk 编译期 → 独立累加链
          }
        }
        wgmma_commit();
        if (q >= STAGES - 2) {
          wgmma_wait_group<STAGES - 2>();
          const int rs = (q - (STAGES - 2)) % STAGES;
          mbar_arrive(empty + rs);
          if constexpr (SHARE_A || SHARE_B)
            if (lane == 0) mbar_arrive_cluster(sempty + rs, 0);
        }
      }
    }
    wgmma_wait0();
  }
  if constexpr (SHARE_A || SHARE_B) {
    __syncthreads();
    cluster_sync();
  }
  if (tid >= NCONS) return;

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  float* Cf = reinterpret_cast<float*>(C);
  bf16* Cb = reinterpret_cast<bf16*>(C);
  if constexpr (KSPLIT > 1) {
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int i = 0; i < 64; ++i) {
#pragma unroll
        for (int ks = 1; ks < KSPLIT; ++ks) acc[0][jn][i] += acc[ks][jn][i];
      }
  }
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if constexpr (BF16OUT) {
        if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r0 * N + cc]) =
            __floats2bfloat162_rn(acc[0][jn][j * 4 + 0], acc[0][jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r1 * N + cc]) =
            __floats2bfloat162_rn(acc[0][jn][j * 4 + 2], acc[0][jn][j * 4 + 3]);
      } else {
        if (r0 < M) *reinterpret_cast<float2*>(&Cf[(size_t)r0 * N + cc]) = make_float2(acc[0][jn][j * 4 + 0], acc[0][jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&Cf[(size_t)r1 * N + cc]) = make_float2(acc[0][jn][j * 4 + 2], acc[0][jn][j * 4 + 3]);
      }
    }
}

// ---------------------------------------------------------------------------
// (2) 门控池化 + RMSNorm + RoPE（同 35，仅用于端到端口径）
// ---------------------------------------------------------------------------
template <int RATIO, bool BF16IN = false>
__global__ void __launch_bounds__(HD)
pool_kernel(const void* __restrict__ Y, const float* __restrict__ ape, const float* __restrict__ nw,
            const float* __restrict__ cos_t, const float* __restrict__ sin_t,
            bf16* __restrict__ out, int M, int N, int C) {
  const int t = blockIdx.x;
  const int j = threadIdx.x;
  const int tok0 = t * RATIO;
  const float* Yf = reinterpret_cast<const float*>(Y);
  const bf16* Yb = reinterpret_cast<const bf16*>(Y);
  auto rdy = [&](size_t i) { return BF16IN ? __bfloat162float(Yb[i]) : Yf[i]; };

  float m = -1e30f, l = 0.f, acc = 0.f;
  auto add_slot = [&](float kv_v, float sc_v) {
    const float mn = fmaxf(m, sc_v);
    const float a = __expf(m - mn);
    const float p = __expf(sc_v - mn);
    l = l * a + p;
    acc = acc * a + p * kv_v;
    m = mn;
  };
  if constexpr (RATIO == 128) {
#pragma unroll 4
    for (int i = 0; i < RATIO; ++i) {
      const size_t base = (size_t)(tok0 + i) * N;
      add_slot(rdy(base + j), rdy(base + C + j) + ape[i * C + j]);
    }
  } else {
    if (t > 0) {
#pragma unroll
      for (int i = 0; i < RATIO; ++i) {
        const size_t base = (size_t)(tok0 - RATIO + i) * N;
        add_slot(rdy(base + j), rdy(base + C + j) + ape[i * C + j]);
      }
    }
#pragma unroll
    for (int i = 0; i < RATIO; ++i) {
      const size_t base = (size_t)(tok0 + i) * N;
      add_slot(rdy(base + HD + j), rdy(base + C + HD + j) + ape[i * C + HD + j]);
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
  const float rstd = rsqrtf(sred[0] / HD + 1e-6f);
  float v = v0 * rstd * nw[j];

  __shared__ float srow[HD];
  srow[j] = v;
  __syncthreads();
  if (j >= HD - RD) {
    const int p = (j - (HD - RD)) >> 1;
    const int par = (j - (HD - RD)) & 1;
    const float a0 = srow[HD - RD + 2 * p];
    const float a1 = srow[HD - RD + 2 * p + 1];
    const int ft = t * (RD / 2) + p;
    const float c = cos_t[ft], s = sin_t[ft];
    v = par == 0 ? (a0 * c - a1 * s) : (a0 * s + a1 * c);
  }
  out[(size_t)t * HD + j] = __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// host：tensor map / yarn / CPU 参考
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

static double yarn_freq(int m, int rope_dim, double base, double factor,
                        double original_seq_len, int beta_fast, int beta_slow) {
  auto fdim = [&](double num_rot) {
    return rope_dim * std::log(original_seq_len / (num_rot * 2 * M_PI)) / (2 * std::log(base));
  };
  double lo = std::floor(fdim(beta_fast)), hi = std::ceil(fdim(beta_slow));
  lo = std::max(lo, 0.0); hi = std::min(hi, (double)rope_dim - 1);
  double low = lo, high = hi;
  if (low == high) high += 0.001;
  double ramp = std::min(1.0, std::max(0.0, (m - low) / (high - low)));
  double smooth = 1.0 - ramp;
  double f = 1.0 / std::pow(base, (double)(2 * m) / rope_dim);
  return f / factor * (1.0 - smooth) + f * smooth;
}

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
        double mn = std::max((double)m, sc);
        double a = std::exp((double)m - mn);
        double p = std::exp(sc - mn);
        l = (float)(l * a + p);
        acc = (float)(acc * a + p * kv);
        m = (float)mn;
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

// L2 persisting window：把一块 buffer 钉进 L2（hitProp=persisting）
static void set_persist(const void* ptr, size_t bytes, bool on) {
  int maxpersist = 0, maxwin = 0;
  cudaDeviceGetAttribute(&maxpersist, cudaDevAttrMaxPersistingL2CacheSize, 0);
  cudaDeviceGetAttribute(&maxwin, cudaDevAttrMaxAccessPolicyWindowSize, 0);
  cudaStreamAttrValue attr{};
  if (on) {
    size_t nb = std::min(bytes, (size_t)maxwin);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, std::min((size_t)maxpersist, nb)));
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = nb;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  } else {
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio = 0.f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  }
  CUDA_CHECK(cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr));
}

int main(int argc, char** argv) {
  int RATIO = (argc > 1) ? std::atoi(argv[1]) : 128;
  int M     = (argc > 2) ? std::atoi(argv[2]) : 32768;
  const char* which = (argc > 3) ? argv[3] : "all";
  const bool skipcorr = (argc > 4) ? (std::atoi(argv[4]) != 0) : false;
  if (RATIO != 128 && RATIO != 4) { std::fprintf(stderr, "ratio must be 128 or 4\n"); return 1; }
  const int COFF = (RATIO == 4) ? 2 : 1;
  const int C = COFF * HD;
  const int N = 2 * C;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * D;
  std::printf("\nDSA Compressor proj (ratio=%d, coff=%d): M=%d  Y[M,%d]=X[M,%d]@Wm[%d,%d]^T\n",
              RATIO, COFF, M, N, D, N, D);
  std::printf("proj FLOPs=%.2f GFLOP  Wm=%.1f MB  X=%.1f MB  out=[%d, %d]\n\n", flops / 1e9,
              (double)N * D * 2 / 1e6, (double)M * D * 2 / 1e6, M / RATIO, HD);

  std::vector<bf16> hX((size_t)M * D), hWm((size_t)N * D);
  std::vector<float> hApe((size_t)RATIO * C), hNw(HD);
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : hX) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hWm) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hApe) v = 0.25f * nd(rng);
  for (auto& v : hNw) v = 1.f + 0.1f * nd(rng);

  bf16 *dX, *dWm, *dOut;
  float *dY, *dApe, *dNw;
  CUDA_CHECK(cudaMalloc(&dX, (size_t)M * D * 2));
  CUDA_CHECK(cudaMalloc(&dWm, (size_t)N * D * 2));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&dApe, (size_t)RATIO * C * 4));
  CUDA_CHECK(cudaMalloc(&dNw, HD * 4));
  CUDA_CHECK(cudaMalloc(&dOut, (size_t)(M / RATIO) * HD * 2));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), (size_t)M * D * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWm, hWm.data(), (size_t)N * D * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dApe, hApe.data(), (size_t)RATIO * C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dNw, hNw.data(), HD * 4, cudaMemcpyHostToDevice));

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

  auto want = [&](const char* name) { return std::strcmp(which, name) == 0; };
  const bool named_cfg = !(want("all") || want("sweep") || want("gemm") || want("pool") ||
                           want("mcast") || want("persist"));

  // ---- 正确性：CPU 全流程参考（小 M）----
  if (!skipcorr) {
    const int MC = 2 * RATIO;
    std::vector<float> xf((size_t)MC * D), wf((size_t)N * D);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = __bfloat162float(hX[i]);
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = __bfloat162float(hWm[i]);
    std::vector<float> ref;
    cpu_ref(xf, wf, hApe, hNw, MC, N, C, RATIO, ref);

    CUtensorMap tmA = make_tmap(dX, (uint64_t)D * 2, M > 256 ? M : 256, 128);
    CUtensorMap tmB = make_tmap(dWm, (uint64_t)D * 2, N, 128);
    auto fn = proj_mcast_kernel<128, 128, 64, 3, 0, 1>;
    const int nt = 128 * 2 + 32;
    const size_t shm = ((size_t)(3 * 3 * 8) + 1023) / 1024 * 1024 + (size_t)3 * (128 + 128) * 128;
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    dim3 g(div_up(N, 128), div_up(MC, 128));
    fn<<<g, nt, shm>>>(tmA, tmB, dY, MC, N, D);
    CUDA_CHECK_LAST();
    if (RATIO == 128) pool_kernel<128><<<MC / 128, HD>>>(dY, dApe, dNw, dcos, dsin, dOut, MC, N, C);
    else pool_kernel<4><<<MC / 4, HD>>>(dY, dApe, dNw, dcos, dsin, dOut, MC, N, C);
    CUDA_CHECK_LAST();
    std::vector<bf16> ho((size_t)(MC / RATIO) * HD);
    CUDA_CHECK(cudaMemcpy(ho.data(), dOut, ho.size() * 2, cudaMemcpyDeviceToHost));
    double err = 0, renorm = 0;
    for (size_t i = 0; i < ho.size(); ++i) {
      double a = __bfloat162float(ho[i]), b = ref[i];
      err = std::max(err, std::fabs(a - b));
      renorm = std::max(renorm, std::fabs(b));
    }
    std::printf("correctness (M=%d): max_abs_err=%.3e (ref~%.3f) %s\n\n", MC, err, renorm,
                err / std::max(renorm, 1.0) < 3e-2 ? "OK" : "FAIL");

    // KSPLIT=2 与 KSPLIT=1 的投影输出对比（验证交替累加链的索引/求和正确）
    {
      float* dY2;
      CUDA_CHECK(cudaMalloc(&dY2, (size_t)MC * N * 4));
      auto fn2 = proj_mcast_kernel<128, 128, 64, 3, 0, 1, false, 2>;
      CUDA_CHECK(cudaFuncSetAttribute(fn2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      fn2<<<g, nt, shm>>>(tmA, tmB, dY2, MC, N, D);
      CUDA_CHECK_LAST();
      std::vector<float> y1((size_t)MC * N), y2((size_t)MC * N);
      CUDA_CHECK(cudaMemcpy(y1.data(), dY, y1.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(y2.data(), dY2, y2.size() * 4, cudaMemcpyDeviceToHost));
      double dy = 0, yr = 0;
      for (size_t i = 0; i < y1.size(); ++i) {
        dy = std::max(dy, (double)std::fabs(y1[i] - y2[i]));
        yr = std::max(yr, (double)std::fabs(y1[i]));
      }
      std::printf("KSPLIT=2 vs =1 proj: max_abs_diff=%.3e (ref~%.3f) %s\n\n", dy, yr,
                  dy / std::max(yr, 1e-6) < 1e-3 ? "OK" : "FAIL");
      CUDA_CHECK(cudaFree(dY2));
    }
  }

  // 启动器：按 AXIS/CN 选 cluster dim
  auto launch = [&](auto fn, int BM, int BN, int ST, int AXIS, int CN, dim3 g, int nt, size_t shm,
                    bool persist, void* outbuf) {
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    if (CN > 8)
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUtensorMap tmA = make_tmap(dX, (uint64_t)D * 2, M, BM);
    CUtensorMap tmB = make_tmap(dWm, (uint64_t)D * 2, N, BN);
    auto run = [&] {
      if (CN > 1) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = g;
        cfg.blockDim = dim3(nt);
        cfg.dynamicSmemBytes = shm;
        cfg.stream = 0;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = (AXIS == 1) ? CN : 1;
        attr[0].val.clusterDim.y = (AXIS == 2) ? CN : 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, fn, tmA, tmB, outbuf, M, N, D));
      } else {
        fn<<<g, nt, shm>>>(tmA, tmB, outbuf, M, N, D);
      }
    };
    if (persist) set_persist(dWm, (size_t)N * D * 2, true);
    run(); CUDA_CHECK_LAST();
    double t = bench_ms(run, 200, 100);
    if (persist) set_persist(dWm, (size_t)N * D * 2, false);
    return t;
  };

  auto report = [&](const char* name, double t) {
    std::printf("  %-22s %8.4f ms  %8.2f TFLOPS  (%5.1f%%)\n", name, t, to_tflops(flops, t),
                100.0 * to_tflops(flops, t) / BF16_PEAK);
  };

  // ---- sweep：35 的基线配置（AXIS=0）----
  if (want("sweep") || want("all") || named_cfg) {
    if (want("sweep") || want("all"))
      std::printf("-- proj GEMM sweep, no multicast (BM x BN, BK=64) --\n");
#define RUN0(NAME, BM, BN, ST)                                                                     \
    do {                                                                                           \
      if (want("sweep") || want("all") || (named_cfg && want(NAME))) {                             \
        const int nt = (BM / 64) * 128 + 32;                                                       \
        const size_t shm = ((size_t)(3 * (ST) * 8) + 1023) / 1024 * 1024 +                         \
                           (size_t)(ST) * ((BM) + (BN)) * 128;                                     \
        dim3 g(div_up(N, BN), div_up(M, BM));                                                      \
        auto fn = proj_mcast_kernel<BM, BN, 64, ST, 0, 1>;                                         \
        report(NAME, launch(fn, BM, BN, ST, 0, 1, g, nt, shm, false, dY));                         \
      }                                                                                            \
    } while (0)
    RUN0("128x128s2", 128, 128, 2);
    RUN0("128x128s3", 128, 128, 3);
    RUN0("128x256s3", 128, 256, 3);
    RUN0("256x128s2", 256, 128, 2);
    RUN0("256x128s3", 256, 128, 3);
    RUN0("256x128s4", 256, 128, 4);
#undef RUN0
#define RUNK(NAME, BM, BN, ST)                                                                     \
    do {                                                                                           \
      if (want("sweep") || want("all") || (named_cfg && want(NAME))) {                             \
        const int nt = (BM / 64) * 128 + 32;                                                       \
        const size_t shm = ((size_t)(3 * (ST) * 8) + 1023) / 1024 * 1024 +                         \
                           (size_t)(ST) * ((BM) + (BN)) * 128;                                     \
        dim3 g(div_up(N, BN), div_up(M, BM));                                                      \
        auto fn = proj_mcast_kernel<BM, BN, 64, ST, 0, 1, false, 2>;                               \
        report(NAME, launch(fn, BM, BN, ST, 0, 1, g, nt, shm, false, dY));                         \
      }                                                                                            \
    } while (0)
    RUNK("128x128s3 K2", 128, 128, 3);
    RUNK("128x256s3 K2", 128, 256, 3);
    RUNK("256x128s4 K2", 256, 128, 4);
#undef RUNK
  }

  // ---- multicast 扫描：A/B 轴 × CN ----
  if (want("mcast") || want("all")) {
    std::printf("-- TMA cluster multicast (A=cluster.x / B=cluster.y) --\n");
  }
  if (want("mcast") || want("all") || named_cfg) {
#define RUNM(NAME, BM, BN, ST, AXIS, CN)                                                           \
    do {                                                                                           \
      const int nx = div_up(N, BN), ny = div_up(M, BM);                                            \
      if ((want("mcast") || want("all") || (named_cfg && want(NAME))) &&                            \
          (((AXIS) == 1 && nx % (CN) == 0) || ((AXIS) == 2 && ny % (CN) == 0) || (AXIS) == 0)) {    \
        const int nt = (BM / 64) * 128 + 32;                                                       \
        const size_t shm = ((size_t)(3 * (ST) * 8) + 1023) / 1024 * 1024 +                         \
                           (size_t)(ST) * ((BM) + (BN)) * 128;                                     \
        dim3 g(nx, ny);                                                                            \
        if ((AXIS) == 1) {                                                                         \
          auto fn = proj_mcast_kernel<BM, BN, 64, ST, 1, CN>;                                      \
          report(NAME, launch(fn, BM, BN, ST, 1, CN, g, nt, shm, false, dY));                       \
        } else {                                                                                   \
          auto fn = proj_mcast_kernel<BM, BN, 64, ST, 2, CN>;                                      \
          report(NAME, launch(fn, BM, BN, ST, 2, CN, g, nt, shm, false, dY));                       \
        }                                                                                          \
      }                                                                                            \
    } while (0)
    RUNM("256x128s4 A-cx2", 256, 128, 4, 1, 2);
    RUNM("256x128s4 A-cx4", 256, 128, 4, 1, 4);
    RUNM("256x128s4 A-cx8", 256, 128, 4, 1, 8);
    RUNM("256x128s4 B-cy2", 256, 128, 4, 2, 2);
    RUNM("256x128s4 B-cy4", 256, 128, 4, 2, 4);
    RUNM("256x128s4 B-cy8", 256, 128, 4, 2, 8);
    RUNM("128x128s3 A-cx2", 128, 128, 3, 1, 2);
    RUNM("128x128s3 A-cx4", 128, 128, 3, 1, 4);
    RUNM("128x128s3 B-cy2", 128, 128, 3, 2, 2);
    RUNM("128x128s3 B-cy4", 128, 128, 3, 2, 4);
    RUNM("128x256s3 A-cx2", 128, 256, 3, 1, 2);
    RUNM("128x256s3 B-cy2", 128, 256, 3, 2, 2);
#undef RUNM

  // ---- L2 persisting window（把 Wm 钉进 L2）----
  if (want("mcast") || want("all")) std::printf("-- L2 persisting window on Wm (%zu MB) --\n", (size_t)N * D * 2 / 1000000);
#define RUNP(NAME, BM, BN, ST)                                                                     \
    do {                                                                                           \
      if (want("mcast") || want("all") || (named_cfg && want(NAME))) {                             \
        const int nt = (BM / 64) * 128 + 32;                                                       \
        const size_t shm = ((size_t)(3 * (ST) * 8) + 1023) / 1024 * 1024 +                         \
                           (size_t)(ST) * ((BM) + (BN)) * 128;                                     \
        dim3 g(div_up(N, BN), div_up(M, BM));                                                      \
        auto fn = proj_mcast_kernel<BM, BN, 64, ST, 0, 1>;                                         \
        report(NAME, launch(fn, BM, BN, ST, 0, 1, g, nt, shm, true, dY));                          \
      }                                                                                            \
    } while (0)
    RUNP("256x128s4 persist", 256, 128, 4);
    RUNP("128x128s3 persist", 128, 128, 3);
#undef RUNP
  }

  CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dWm)); CUDA_CHECK(cudaFree(dY));
  CUDA_CHECK(cudaFree(dApe)); CUDA_CHECK(cudaFree(dNw)); CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaFree(dcos)); CUDA_CHECK(cudaFree(dsin));
  return 0;
}
