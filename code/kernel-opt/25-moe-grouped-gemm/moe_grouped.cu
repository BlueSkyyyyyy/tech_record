// 25 MoE（二）：grouped GEMM（contiguous / masked）
//
// 承接 23/24 的 TMA + mbarrier + warp specialization + wgmma FP8 GEMM。
// MoE 的第二个 GEMM（expert FFN）是「G 个形状相同、M 各不相同的 GEMM 共享 N/K」：
//   - prefill/training：contiguous 布局，A 按 expert 连续拼接 (M_total,K)，
//     B 堆叠成 (G,N,K)，grouped_layout[row] 给出该行属于哪个 expert；
//   - decode：masked 布局，A=(G,max_m,K)，masked_m[g] 给出每组有效行数，
//     小 M 组里超出 masked_m 的 tile 直接退出。
//
// 关键：一个 kernel 覆盖全部 expert。B tile 的行坐标 = group*N + n_tile*BN
// （把 (G,N) 拉平成一行），于是每个 CTA 自己算出该用哪个 expert 的 B，
// 省掉 G 次 launch、G 张 descriptor，也避免了每 expert 小 GEMM 的 SM 空转与尾部。
//
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 25-moe-grouped-gemm/moe_grouped.cu [mode] [tokens]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <vector>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（同 20/22/23）
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

// ---------------------------------------------------------------------------
// grouped kernel（A/B 都是 2D tensor map，expert 编码进行坐标）
//   MASKED=false：contiguous。A=(M_total,K)，B=(G*N,K)。gl[row]=expert id。
//                 grid=(N/BN, M_total/BM, 1)。要求 BM<=128 且每组 M 对齐到 BM。
//   MASKED=true ：masked。A=(G*max_m,K)，B=(G*N,K)。gl[g]=masked_m[g]。
//                 grid=(N/BN, ceil(max_m/BM), G)；超出 masked_m 的 tile 早退。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool MASKED, int BROW_SCALE = 0>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
moe_kernel(const __grid_constant__ CUtensorMap tmA,
           const __grid_constant__ CUtensorMap tmB,
           float* __restrict__ D, const int* __restrict__ gl,
           int N, int K, int max_m, int m_glob_limit, float scale, int d_row_base) {
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
  const int nblk = K / BK;

  // expert / 行坐标
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
    a_row = block_row;            // 连续布局：A 行就是全局行
    b_row = group * N + block_col; // B 拉平成 (G*N,K)
    dbase = d_row_base;            // 单组 loop 时 = 该组行偏移；grouped 时 = 0
    mlim = m_glob_limit;
  }

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  // ---- producer warp：专职发 TMA ----
  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int kb = 0; kb < nblk; ++kb) {
        const int st = kb % STAGES;
        if (kb >= STAGES) { mbar_wait(empty + st, (uint32_t)((kb / STAGES - 1) & 1)); }
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * BK + BN * BK);
        tma_load_2d(&tmA, As + (size_t)st * BM * BK, kb * BK, a_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * BK, kb * BK, b_row, full + st);
      }
    }
    return;
  }

  // ---- consumer warpgroups ----
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc[NSPLIT][64];
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
      const int r0 = dbase + block_row + row0;
      const int r1 = r0 + 8, cc = block_col + col;
      if (r0 < mlim)
        *reinterpret_cast<float2*>(&D[(size_t)r0 * N + cc]) =
            make_float2(acc[jn][j * 4 + 0] * scale, acc[jn][j * 4 + 1] * scale);
      if (r1 < mlim)
        *reinterpret_cast<float2*>(&D[(size_t)r1 * N + cc]) =
            make_float2(acc[jn][j * 4 + 2] * scale, acc[jn][j * 4 + 3] * scale);
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

#include "moe_grouped_launch.h"

int main(int argc, char** argv) { return moe_main(argc, argv); }
