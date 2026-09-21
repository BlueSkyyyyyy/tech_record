// 26 MoE（二·续）：grouped GEMM + TMA cluster multicast 压 L2
//
// 承接 25 的 TMA + mbarrier + warp specialization + wgmma grouped GEMM。
// 25 的 ncu（prefill 最佳，128x256 s3）显示 L2 吞吐 81.9%、DRAM 52%、tensor 61.3%
// —— 瓶颈在 L2→SM 带宽，而不是 DRAM。根因：每个 CTA 都要把整块 A/B tile 从 L2 拉进
// 自己的 smem，而同一 cluster 内相邻 CTA 需要的 tile 大量重叠：
//
//   * A tile 只由 m-tile(blockIdx.y) 决定 → 同一「N 方向相邻」的一组 CTA
//     （blockIdx.y 相同、blockIdx.x 不同）**共享同一块 A**；
//   * B tile 由 (group, n-tile) 决定 → 同一「M 方向相邻」的一组 CTA
//     （blockIdx.x 相同、blockIdx.y 不同且同 expert）**共享同一块 B**。
//
// 本文先做**沿 N 方向的 A multicast**（cluster.x = CN）：把 cluster 内 A tile 只从
// L2 取一次，广播给同一组的 CN 个 CTA。A 的读放大 = N/BN（BN=128 时 24×，
// BN=256 时 12×），且它只依赖 m-tile、不受 expert 分组边界影响，因此对所有 token
// 规模都稳定有效。B 的 multicast 需要处理「相邻 m-tile 跨 expert」的退化，留待后续。
//
// 关键实现（对照 DeepGEMM sm90 的 TMA multicast）：
//   1. cluster 维度 (CN,1,1)，rank = blockIdx.x % CN；
//   2. A 只由 rank0 发 `...multicast::cluster` + mask=(1<<CN)-1；B 每个 CTA 各发各的；
//   3. 每个 CTA 仍对自己的 full barrier `arrive.expect_tx(A_bytes+B_bytes)`，
//      multicast 会把 A 的完成信号送到每个目标 CTA 的同偏移 full barrier；
//   4. empty barrier count = CN × 消费者 warp 数；消费者 lane<CN 用 `mapa` 把 arrive
//      投递到目标 CTA 的 empty barrier（rank0 覆盖 A 前必须等到全 cluster 读完）。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 26-moe-cluster-multicast/moe_cluster.cu [mode] [tokens]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <vector>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（同 20/22/23/25）
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
// 把 arrive 投递到同 cluster 内 cta_id 号 CTA 的（同偏移）barrier
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
// grouped kernel + A multicast（cluster.x = CN）
//   MASKED=false：contiguous。A=(M_total,K)，B=(G*N,K)。gl[row]=expert id。
//   MASKED=true ：masked。A=(G*max_m,K)，B=(G*N,K)。gl[g]=masked_m[g]。
// cluster 沿 x（N 方向）：同 cluster 内 blockIdx.y 相同 → a_row 相同 → A 可广播。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool MASKED, int CN>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
moe_kernel(const __grid_constant__ CUtensorMap tmA,
           const __grid_constant__ CUtensorMap tmB,
           float* __restrict__ D, const int* __restrict__ gl,
           int N, int K, int max_m, int m_glob_limit, float scale, int d_row_base) {
  constexpr int NWG = BM / 64;
  constexpr int NCONS = NWG * 128;
  constexpr int NCW = NCONS / 32;  // 消费者 warp 数
  constexpr int NSPLIT = BN / 128;
  constexpr int SBO = 1024;

  constexpr int BARR = ((int)(3 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;      // B 的释放：本 CTA 消费者本地 arrive
  uint64_t* aempty = empty + STAGES;    // A 的释放：全 cluster 消费者 arrive 到 rank0
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * BK;

  const int tid = threadIdx.x;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  const uint32_t rank = (CN > 1) ? cluster_rank() : 0u;

  int group, block_row, a_row, b_row, dbase, mlim;
  if constexpr (MASKED) {
    group = blockIdx.z;
    block_row = blockIdx.y * BM;
    if (block_row >= gl[group]) return;  // 整块无效 → 早退
    a_row = group * max_m + block_row;
    b_row = group * N + block_col;
    dbase = group * max_m;
    mlim = dbase + gl[group];
  } else {
    block_row = blockIdx.y * BM;
    group = (gl != nullptr) ? gl[block_row] : 0;
    if (group < 0) group = 0;
    a_row = block_row;
    b_row = group * N + block_col;
    dbase = d_row_base;
    mlim = m_glob_limit;
  }

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full + s, 1);
      mbar_init(empty + s, NCONS);          // 本 CTA 所有消费者线程各 arrive 一次
      mbar_init(aempty + s, CN * NCW);      // 全 cluster 每个消费者 warp 的 lane0 各 arrive 一次
    }
    fence_mbar_init();
  }
  __syncthreads();
  if constexpr (CN > 1) cluster_sync();  // 远端 barrier 必须已 init 才可被 arrive

  float acc[NSPLIT][64];  // 声明在分支外，供消费者 epilogue 使用

  // ---- producer warp：rank0 发 A multicast，各 CTA 各发自己的 B ----
  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int kb = 0; kb < nblk; ++kb) {
        const int st = kb % STAGES;
        if (kb >= STAGES) {
          const uint32_t ph = (uint32_t)((kb / STAGES - 1) & 1);
          mbar_wait(empty + st, ph);                        // B 只等本 CTA
          if constexpr (CN > 1)
            if (rank == 0) mbar_wait(aempty + st, ph);      // A 广播方需等全 cluster
        }
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * BK + BN * BK);
        if constexpr (CN == 1) {
          tma_load_2d(&tmA, As + (size_t)st * BM * BK, kb * BK, a_row, full + st);
        } else {
          // A 广播（沿 x 的 cluster），B 每个 CTA 各自装载
          if (rank == 0)
            tma_load_2d_mcast(&tmA, As + (size_t)st * BM * BK, kb * BK, a_row, full + st,
                              (uint16_t)((1u << CN) - 1));
        }
        tma_load_2d(&tmB, Bs + (size_t)st * BN * BK, kb * BK, b_row, full + st);
      }
    }
  } else {
    // ---- consumer warpgroups ----
    const int wg = tid >> 7;
    const int lane = tid & 31;
#pragma unroll
    for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
      for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

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
      if (kb >= STAGES - 2) {
        wgmma_wait_group<STAGES - 2>();
        const int rs = (kb - (STAGES - 2)) % STAGES;
        mbar_arrive(empty + rs);  // B：本 CTA 消费者线程本地 arrive
        if constexpr (CN > 1) {
          if (lane == 0) mbar_arrive_cluster(aempty + rs, 0);  // A：每 warp lane0 投到 rank0
        }
      }
    }
    wgmma_wait0();
  }
  if constexpr (CN > 1) {
    __syncthreads();      // 与 producer 汇合
    cluster_sync();       // 保护远端 barrier 反构前所有 arrive 已落地
  }
  if (tid >= NCONS) return;

  {
    // ---- epilogue（仅消费者） ----
    const int wg = tid >> 7;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int col = jn * 128 + j * 8 + (lane & 3) * 2;
        const int r0 = dbase + block_row + row0;
        const int r1 = r0 + 8, cc = block_col + col;
        if (r0 < mlim)
          *reinterpret_cast<float2*>(&D[(size_t)r0 * N + cc]) =
              make_float2(acc[jn][j * 4 + 0] * scale, acc[jn][j * 4 + 1] * scale);
        if (r1 < mlim)
          *reinterpret_cast<float2*>(&D[(size_t)r1 * N + cc]) =
              make_float2(acc[jn][j * 4 + 2] * scale, acc[jn][j * 4 + 3] * scale);
      }
  }
  (void)max_m;
  (void)d_row_base;
}

// ---------------------------------------------------------------------------
// host：tensor map
// ---------------------------------------------------------------------------
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

#include "moe_cluster_launch.h"

int main(int argc, char** argv) { return moe_main(argc, argv); }
