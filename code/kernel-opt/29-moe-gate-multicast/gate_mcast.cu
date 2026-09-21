// 29 MoE（一·续二）：gate GEMM + TMA cluster multicast
//
// 承接 28（gate_wgmma.cu）：DeepSeek-V4-Pro 的 router 前门 gate GEMM
//   logits[M, E] = X[M, H] @ Wg[E, H]^T
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：hidden=7168, n_routed=384。
//
// 28 的 ncu（M=32768，最佳 256×128 s4 = 600 TFLOPS）显示：
//   Compute 69.6%、tensor pipe 68.2%、L2 **79.1%**、DRAM 53.4%、occ 26%。
// 瓶颈是 L2→SM 的 A 流量：A 只由 m-tile(blockIdx.y) 决定，而 N=384 / BN=128 = 3
// 个 n-tile 都读同一块 A tile → A 沿 N 方向被重读 3×。28 的结论是「下一堵墙是
// A 被 3 个 n-tile 重读（L2 79%）」。
//
// 本文把 26（grouped GEMM）的 **A-multicast** 搬过来：gate 的 grid.x 恰好 = 3
// （同一 m-tile 的 3 个 n-tile），令 cluster.x = CN（=2 或 3）沿 x 组队，
// rank0 用 `cp.async.bulk.tensor.2d...multicast::cluster` 把 A 只从 L2 取一次、
// 广播给整组，A 的 L2 读放大从 N/BN=3× 降到 ceil(3/CN)×。
//
// 与 26 完全相同的协议（踩坑记录见 TECHNIQUES）：
//   1. cluster=CN，rank = cluster_ctarank；
//   2. rank0 发 A multicast + mask=(1<<CN)-1；B 每个 CTA 各自发；
//   3. 每个 CTA 仍对自己的 full barrier arrive.expect_tx(A_bytes+B_bytes)；
//   4. A 的释放用独立 aempty barrier（count = CN × 消费者 warp 数），
//      每 warp lane0 用 mapa 投到 rank0；私有 B 的 empty 保持本地 arrive；
//   5. init 后与退出前各一次 cluster_sync。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 29-moe-gate-multicast/gate_mcast.cu [M] [which]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;
constexpr double BF16_PEAK = 989.0;

// DeepSeek-V4-Pro
constexpr int H = 7168;  // K = hidden
constexpr int E = 384;   // N = n_routed_experts

// ---------------------------------------------------------------------------
// wgmma helpers（同 20/23/28）—— bf16 m64n128k16
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

// bf16 K-major SW128：8 行 × 64 元素（128B）atom；tile 宽固定 64 → SBO=1024。
__device__ __forceinline__ int sw128_off(int row, int k) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * 1 + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
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
// mbarrier + TMA helpers（同 23/26/28）
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
      "mbarrier.arrive.shared::cluster.b64 _, [rem];\n}\n" ::"r"(smem_u32(bar)),
      "r"(cta_id));
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
// gate GEMM：wgmma SS + TMA + mbarrier + warp specialization (+ operand multicast)
//   A = X[M, K] (bf16), B = Wg[E, K]，B 天然 K-major。
//   grid = (E/BN, M/BM, 1)。
//   AXIS=0: 无 multicast（CN=1，同 28）。
//   AXIS=1: cluster.x=CN，沿 N 组队，同一 m-tile 的 CN 个 n-tile 共享同一块 A
//           → 广播 A（A 读放大 N/BN=3×）。CN 必须整除 grid.x=3。
//   AXIS=2: cluster.y=CN，沿 M 组队，同一 n-tile 的 CN 个 m-tile 共享同一块 B
//           → 广播 B（B 权重只 5.5MB、常驻 L2，但被 M/BM 个 m-tile 各读一遍，
//              L2 读流量 ~1.4GB，是真正的大头）。CN 必须整除 grid.y。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int CN, int AXIS>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
gate_mcast_kernel(const __grid_constant__ CUtensorMap tmA,
                  const __grid_constant__ CUtensorMap tmB,
                  float* __restrict__ C, int M, int N, int K) {
  static_assert(BK == 64, "bf16 SW128 内维固定 64 元素");
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NCONS = NWG * 128;
  constexpr int NCW = NCONS / 32;  // 消费者 warp 数
  constexpr int SBO = 1024;
  constexpr bool MCAST = (CN > 1);

  constexpr int BARR = ((int)(3 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;    // 私有 operand 的释放（本地）
  uint64_t* sempty = empty + STAGES;  // 共享 operand 的释放（全 cluster 到 rank0）
  char* As = smem + BARR;             // [STAGES][BM][64] SW128
  char* Bs = As + (size_t)STAGES * BM * 128;

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  const uint32_t rank = MCAST ? cluster_rank() : 0u;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full + s, 1);
      mbar_init(empty + s, NCONS);
      mbar_init(sempty + s, MCAST ? CN * NCW : 1);
    }
    fence_mbar_init();
  }
  __syncthreads();
  if constexpr (MCAST) cluster_sync();

  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int q = 0; q < nblk; ++q) {
        const int st = q % STAGES;
        if (q >= STAGES) {
          const uint32_t ph = (uint32_t)((q / STAGES - 1) & 1);
          mbar_wait(empty + st, ph);  // 私有 operand 只等本 CTA
          if constexpr (MCAST)
            if (rank == 0) mbar_wait(sempty + st, ph);  // 广播方需等全 cluster
        }
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * 128 + BN * 128);
        if constexpr (AXIS == 1) {
          // 共享 = A（沿 x）；私有 = B
          if (rank == 0)
            tma_load_2d_mcast(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st,
                              (uint16_t)((1u << CN) - 1));
          tma_load_2d(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st);
        } else if constexpr (AXIS == 2) {
          // 共享 = B（沿 y）；私有 = A
          tma_load_2d(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st);
          if (rank == 0)
            tma_load_2d_mcast(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st,
                              (uint16_t)((1u << CN) - 1));
        } else {
          tma_load_2d(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st);
          tma_load_2d(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st);
        }
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

  for (int q = 0; q < nblk; ++q) {
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
      if constexpr (MCAST)
        if (lane == 0) mbar_arrive_cluster(sempty + rs, 0);
    }
  }
  wgmma_wait0();
  if constexpr (MCAST) {
    __syncthreads();   // 与 producer 汇合
    cluster_sync();    // 保护远端 barrier 反构
  }
  if (tid >= NCONS) return;

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
// (d) "wide-N" kernel：一个 CTA 算完整 N=384（3 个 n128），A 只读一遍
//   免 multicast、免 cluster 耦合：x 方向不再切 n-tile（grid.x=1），
//   每个 m-tile 的 CTA 把 A tile 读一次，用它同时算 3 个 expert 列块，
//   A 的 L2 读放大从 3× 降到 1×。B（权重 5.5MB，L2 常驻）每 CTA 全读。
//   TMA box 单维上限 256 → B 的 384 行拆成 256+128 两次 TMA。
// ---------------------------------------------------------------------------
constexpr int NW_TOTAL = E / 128;  // 3

template <int BM, int STAGES>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
gate_wide_kernel(const __grid_constant__ CUtensorMap tmA,
                 const __grid_constant__ CUtensorMap tmB,
                 float* __restrict__ C, int M, int N, int K) {
  constexpr int BK = 64;
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = NW_TOTAL;  // 3
  constexpr int NCONS = NWG * 128;
  constexpr int SBO = 1024;
  constexpr int NE = 384;  // 固定 N=384

  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;                       // [STAGES][BM][64]
  char* Bs = As + (size_t)STAGES * BM * 128;    // [STAGES][384][64]

  const int tid = threadIdx.x;
  const int block_row = blockIdx.x * BM;        // grid.x = M/BM
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
        mbar_arrive_expect_tx(full + st, (BM + E) * 128);
        tma_load_2d(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st);
        // B: 384 行 = 3×128（TMA box ≤ 256 → 用 boxR=128 的 map 发 3 次）
        char* bst = Bs + (size_t)st * E * 128;
#pragma unroll
        for (int jn = 0; jn < NSPLIT; ++jn)
          tma_load_2d(&tmB, bst + jn * 128 * 128, q * 128, jn * 128, full + st);
      }
    }
    return;
  }

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc[NSPLIT][64];
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[jn][i] = 0.f;

  for (int q = 0; q < nblk; ++q) {
    const int st = q % STAGES;
    mbar_wait(full + st, (uint32_t)((q / STAGES) & 1));
    fence_proxy_async();
    char* a = As + (size_t)st * BM * 128 + (size_t)wg * 64 * 128;
    char* b = Bs + (size_t)st * E * 128;
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
      mbar_arrive(empty + (q - (STAGES - 2)) % STAGES);
    }
  }
  wgmma_wait0();

  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = col;
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + cc]) = make_float2(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + cc]) = make_float2(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
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

template <int BM, int BN, int BK, int STAGES, int CN, int AXIS>
static void launch_gate(const CUtensorMap& tmA, const CUtensorMap& tmB, float* C, int M, int N, int K) {
  auto fn = gate_mcast_kernel<BM, BN, BK, STAGES, CN, AXIS>;
  const int nt = (BM / 64) * 128 + 32;
  const int BARR = ((int)(3 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  const size_t shm = (size_t)BARR + (size_t)STAGES * (BM + BN) * 128;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  dim3 grid(N / BN, div_up(M, BM), 1);
  if (CN == 1) {
    fn<<<grid, nt, shm>>>(tmA, tmB, C, M, N, K);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
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
  cudaError_t err = cudaLaunchKernelEx(&cfg, fn, tmA, tmB, C, M, N, K);
  cuda_check(err, "cudaLaunchKernelEx", __FILE__, __LINE__);
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 32768;
  const char* which = (argc > 2) ? argv[2] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * E * H;
  std::printf("\nMoE gate GEMM + TMA cluster multicast (bf16): M=%d N=%d K=%d  FLOPs=%.2f GFLOP\n\n",
              M, E, H, flops / 1e9);

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
    std::printf("%-20s %8.4f ms  %8.2f TFLOPS  (%5.2f%% of bf16 peak)\n", tag, ms, tf,
                100.0 * tf / BF16_PEAK);
  };
  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0 ||
           std::strncmp(name, which, std::strlen(which)) == 0;
  };

#define RUN_WIDE(NAME, BM, ST)                                                                     \
  do {                                                                                             \
    if (want(NAME)) {                                                                              \
      CUtensorMap tmA = make_tmap(X, (uint64_t)H * 2, M > BM ? M : BM, BM);                        \
      CUtensorMap tmB = make_tmap(Wg, (uint64_t)H * 2, E, 128);                                    \
      auto fn = gate_wide_kernel<BM, ST>;                                                          \
      const int nt = (BM / 64) * 128 + 32;                                                         \
      const int BARR = ((int)(2 * ST * sizeof(uint64_t)) + 1023) / 1024 * 1024;                    \
      const size_t shm = (size_t)BARR + (size_t)ST * ((BM) + E) * 128;                             \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
      dim3 grid(div_up(M, BM));                                                                    \
      auto run = [&] { fn<<<grid, nt, shm>>>(tmA, tmB, C, M, E, H); };                             \
      run();                                                                                       \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      double t = bench_ms(run, 5, 50);                                                             \
      report(NAME, t);                                                                             \
    }                                                                                              \
  } while (0)

#define RUN_GATE(NAME, BM, BN, ST, CN, AX)                                                         \
  do {                                                                                             \
    if (want(NAME)) {                                                                              \
      CUtensorMap tmA = make_tmap(X, (uint64_t)H * 2, M > BM ? M : BM, BM);                        \
      CUtensorMap tmB = make_tmap(Wg, (uint64_t)H * 2, E, BN);                                     \
      auto run = [&] { launch_gate<BM, BN, 64, ST, CN, AX>(tmA, tmB, C, M, E, H); };               \
      run();                                                                                       \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      double t = bench_ms(run, 5, 50);                                                             \
      report(NAME, t);                                                                             \
    }                                                                                              \
  } while (0)

  std::printf("-- (a) baseline: TMA + warp specialization (CN=1，同 28) --\n");
  RUN_GATE("ws64x128s3_c1", 64, 128, 3, 1, 0);
  RUN_GATE("ws128x128s2_c1", 128, 128, 2, 1, 0);
  RUN_GATE("ws128x128s3_c1", 128, 128, 3, 1, 0);
  RUN_GATE("ws128x128s4_c1", 128, 128, 4, 1, 0);
  RUN_GATE("ws128x128s5_c1", 128, 128, 5, 1, 0);
  RUN_GATE("ws128x128s6_c1", 128, 128, 6, 1, 0);
  RUN_GATE("ws256x128s3_c1", 256, 128, 3, 1, 0);
  RUN_GATE("ws256x128s4_c1", 256, 128, 4, 1, 0);

  std::printf("\n-- (b) + A multicast (cluster.x=CN; grid.x=3 → CN 只能取 3) --\n");
  RUN_GATE("ws64x128s3_ax3", 64, 128, 3, 3, 1);
  RUN_GATE("ws128x128s2_ax3", 128, 128, 2, 3, 1);
  RUN_GATE("ws128x128s3_ax3", 128, 128, 3, 3, 1);
  RUN_GATE("ws256x128s4_ax3", 256, 128, 4, 3, 1);

  std::printf("\n-- (c) + B multicast (cluster.y=CM; B 权重被 M/BM 个 m-tile 重读) --\n");
  RUN_GATE("ws128x128s2_by2", 128, 128, 2, 2, 2);
  RUN_GATE("ws128x128s3_by2", 128, 128, 3, 2, 2);
  RUN_GATE("ws128x128s3_by4", 128, 128, 3, 4, 2);
  RUN_GATE("ws128x128s3_by8", 128, 128, 3, 8, 2);
  RUN_GATE("ws256x128s3_by2", 256, 128, 3, 2, 2);
  RUN_GATE("ws256x128s4_by2", 256, 128, 4, 2, 2);
  RUN_GATE("ws256x128s4_by4", 256, 128, 4, 4, 2);

  std::printf("\n-- (d) wide-N：一个 CTA 算完 N=384，A 只读一遍（免 cluster）--\n");
  RUN_WIDE("wide64s3", 64, 3);
  RUN_WIDE("wide64s4", 64, 4);
  RUN_WIDE("wide128s2", 128, 2);
  RUN_WIDE("wide128s3", 128, 3);
  RUN_WIDE("wide256s2", 256, 2);

  CUDA_CHECK(cudaFree(X));
  CUDA_CHECK(cudaFree(Wg));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
