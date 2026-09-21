// 54 融合 cross-entropy / logits（LM head）
//
//   logits[M,V] = X[M,H] @ W[V,H]^T
//   loss        = mean_m ( logsumexp_v(logits[m,v]) - logits[m, target[m]] )
//
// 传统两 kernel：cuBLAS 把 logits 物化到显存（bf16 [M,V]），再由 CE kernel 读一遍。
//   logits 的「写 + 读」是纯浪费：M=8192,V=129280 时 2×2.12 GB。
//
// 本文件：
//   (1) fce_kernel   : TMA + mbarrier + warp specialization + wgmma SS SW128，
//                      epilogue 就地做 tile 级 online logsumexp，只写
//                      partial_max[M][T] / partial_sum[M][T]（T=V/BN）+ target_logit[M]；
//                      FUSE=false 时退化成「普通 GEMM 写 bf16 logits」用于测 GEMM 裸吞吐。
//   (2) combine_kernel : 跨 T 个 n-tile 合并成每行 lse，算 row loss 并归约。
//   (3) 基线：cuBLAS gemm + ce_baseline_kernel（读物化 logits，单行 online logsumexp）。
//
// 编译：NVCC_FLAGS="-lcuda" scripts/run.sh 54-fused-ce/fused_ce.cu
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstring>
#include <vector>

using bf16 = __nv_bfloat16;
constexpr double BF16_PEAK = 989.0;

// ---------------------------------------------------------------------------
// wgmma / SW128 / TMA helpers（同 20/23/28/35 篇）
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
// (1) 融合 GEMM + tile-level logsumexp
//     A[M,K] bf16, W[N,K] bf16（K-major），N=V。grid=(N/BN, M/BM)。
//     FUSE=true ：写 partial_max/partial_sum[M,T] + target_logit[M]
//     FUSE=false：写 bf16 logits（测 GEMM 裸吞吐的对照）
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool FUSE, int GM = 0, int GN = 0>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
fce_kernel(const __grid_constant__ CUtensorMap tmA,
           const __grid_constant__ CUtensorMap tmB,
           void* __restrict__ C, float* __restrict__ partial_max,
           float* __restrict__ partial_sum, float* __restrict__ target_logit,
           const int* __restrict__ target, int M, int N, int K) {
  static_assert(BK == 64, "bf16 SW128 内维固定 64 元素");
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NCONS = NWG * 128;
  constexpr int SBO = 1024;
  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;

  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * 128;

  const int tid = threadIdx.x;
  int block_row, block_col;
  if constexpr (GM > 0) {
    // L2 友好的 superblock 调度：把 GM×GN 个 tile 编成一组，组内面板 (GM·BM + GN·BN)·K·2
    // 字节可常驻 L2；组间顺序扫描，A/B 各只回读一次。
    const int Mt = M / BM, Nt = N / BN;
    const int GNs = (Nt + GN - 1) / GN;
    const int id = blockIdx.x;
    const int sb = id / (GM * GN);
    const int in = id % (GM * GN);
    const int ms = sb / GNs, ns = sb % GNs;
    const int mi = in % GM, ni = in / GM;  // m 变化最快 → 相邻 CTA 共享 B tile
    const int mt = ms * GM + mi, nt = ns * GN + ni;
    if (mt >= Mt || nt >= Nt) return;
    block_row = mt * BM;
    block_col = nt * BN;
  } else {
    block_row = blockIdx.y * BM;
    block_col = blockIdx.x * BN;
  }
  const int nblk = K / BK;
  const int T = N / BN;

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
    }
  }
  wgmma_wait0();

  const int g = lane >> 2;                 // 0..7：行内偏移
  const int tig = lane & 3;                // 0..3：列对
  const int row0 = wg * 64 + (warp & 3) * 16 + g;   // CTA 内行号
  const int r0 = block_row + row0, r1 = r0 + 8;

  if constexpr (FUSE) {
    // ---- 每行 tile 级 online logsumexp ----
    float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        rmax0 = fmaxf(rmax0, fmaxf(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]));
        rmax1 = fmaxf(rmax1, fmaxf(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]));
      }
    // 4-lane group 内 all-reduce（同一行的 4 个 tig）
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

    float rsum0 = 0.f, rsum1 = 0.f;
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        rsum0 += __expf(acc[jn][j * 4 + 0] - rmax0) + __expf(acc[jn][j * 4 + 1] - rmax0);
        rsum1 += __expf(acc[jn][j * 4 + 2] - rmax1) + __expf(acc[jn][j * 4 + 3] - rmax1);
      }
    rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
    rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
    rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
    rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);

    if (tig == 0) {
      const int nt = block_col / BN;
      if (r0 < M) { partial_max[(size_t)r0 * T + nt] = rmax0; partial_sum[(size_t)r0 * T + nt] = rsum0; }
      if (r1 < M) { partial_max[(size_t)r1 * T + nt] = rmax1; partial_sum[(size_t)r1 * T + nt] = rsum1; }
    }
    // ---- target logit 提取（含 target 的 tile 才写）----
    const int tg0 = (r0 < M) ? target[r0] : -1;
    const int tg1 = (r1 < M) ? target[r1] : -1;
    const bool hit0 = (tg0 >= block_col && tg0 < block_col + BN);
    const bool hit1 = (tg1 >= block_col && tg1 < block_col + BN);
    if (hit0 || hit1) {
#pragma unroll
      for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int c0 = block_col + jn * 128 + j * 8 + tig * 2;
          if (hit0 && c0 == tg0) target_logit[r0] = acc[jn][j * 4 + 0];
          if (hit0 && c0 + 1 == tg0) target_logit[r0] = acc[jn][j * 4 + 1];
          if (hit1 && c0 == tg1) target_logit[r1] = acc[jn][j * 4 + 2];
          if (hit1 && c0 + 1 == tg1) target_logit[r1] = acc[jn][j * 4 + 3];
        }
    }
  } else {
    bf16* Cb = reinterpret_cast<bf16*>(C);
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int col = jn * 128 + j * 8 + tig * 2;
        const int cc = block_col + col;
        if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r0 * N + cc]) =
            __floats2bfloat162_rn(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r1 * N + cc]) =
            __floats2bfloat162_rn(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      }
  }
}

// ---------------------------------------------------------------------------
// (2) 跨 n-tile 合并 logsumexp，算 loss
// ---------------------------------------------------------------------------
__global__ void combine_kernel(const float* __restrict__ partial_max,
                               const float* __restrict__ partial_sum,
                               const float* __restrict__ target_logit,
                               float* __restrict__ loss_sum, int M, int T) {
  float local = 0.f;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += gridDim.x * blockDim.x) {
    float m = -INFINITY;
    for (int t = 0; t < T; ++t) m = fmaxf(m, partial_max[(size_t)row * T + t]);
    float s = 0.f;
    for (int t = 0; t < T; ++t) s += partial_sum[(size_t)row * T + t] * __expf(partial_max[(size_t)row * T + t] - m);
    const float lse = m + logf(s);
    local += (lse - target_logit[row]);
  }
  // block reduce
  __shared__ float red[256];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(loss_sum, red[0]);
}

// ---------------------------------------------------------------------------
// (3) 基线 CE：读物化 logits，单行 online logsumexp
// ---------------------------------------------------------------------------
__global__ void ce_baseline_kernel(const bf16* __restrict__ logits,
                                   const int* __restrict__ target,
                                   float* __restrict__ row_loss, int M, int N) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const bf16* rp = logits + (size_t)row * N;
  float m = -INFINITY, s = 0.f;
  for (int v = threadIdx.x; v < N; v += blockDim.x) {
    const float x = __bfloat162float(rp[v]);
    const float nm = fmaxf(m, x);
    s = s * __expf(m - nm) + __expf(x - nm);
    m = nm;
  }
  // block 内合并 (m,s)
  __shared__ float sm[256], ss[256];
  sm[threadIdx.x] = m; ss[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      const float a = sm[threadIdx.x], b = sm[threadIdx.x + st];
      const float nm = fmaxf(a, b);
      ss[threadIdx.x] = ss[threadIdx.x] * __expf(a - nm) + ss[threadIdx.x + st] * __expf(b - nm);
      sm[threadIdx.x] = nm;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float lse = sm[0] + logf(ss[0]);
    const float tl = __bfloat162float(rp[target[row]]);
    row_loss[row] = lse - tl;
  }
}

__global__ void sum_reduce_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
  float local = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) local += x[i];
  __shared__ float red[256];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, red[0]);
}

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

// 快速 host 填充（fill_random 对 9 亿个元素太慢）
template <class T>
static void fill_fast(T* p, size_t n, uint32_t seed) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)(i * 2654435761u) + seed;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = static_cast<T>(((x & 0xffffu) / 65535.0f - 0.5f) * 0.05f);
  }
}

template <int BM, int BN, int BK, int STAGES, bool FUSE, int GM = 0, int GN = 0>
static void launch_fce(const void* dA, const void* dW, void* C,
                       float* pm, float* ps, float* tl, const int* target, int M, int N, int K,
                       cudaStream_t st) {
  CUtensorMap tmA = make_tmap(dA, (uint64_t)K * 2, M, BM);
  CUtensorMap tmB = make_tmap(dW, (uint64_t)K * 2, N, BN);
  constexpr int NT = (BM / 64) * 128 + 32;
  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  const size_t smem = BARR + (size_t)STAGES * (BM + BN) * 128;
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(fce_kernel<BM, BN, BK, STAGES, FUSE, GM, GN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    done = true;
  }
  dim3 grid;
  if constexpr (GM > 0) {
    const int Mt = M / BM, Nt = N / BN;
    const int GMs = (Mt + GM - 1) / GM, GNs = (Nt + GN - 1) / GN;
    grid = dim3(GMs * GNs * GM * GN, 1, 1);
  } else {
    grid = dim3(N / BN, M / BM);
  }
  fce_kernel<BM, BN, BK, STAGES, FUSE, GM, GN><<<grid, NT, smem, st>>>(tmA, tmB, C, pm, ps, tl, target, M, N, K);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int M = 8192;
  const char* which = "all";
  if (argc > 1) M = std::atoi(argv[1]);
  if (argc > 2) which = argv[2];
  const int H = 7168, V = 129280;
  const size_t WBYTES = (size_t)V * H * 2;
  const size_t LOGITBYTES = (size_t)M * V * 2;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * V * H;
  std::printf("\nFused CE: X[M,%d] @ W[%d,%d]^T  logits[%d,%d]  (%.1f MB bf16 if materialized)\n",
              H, V, H, M, V, LOGITBYTES / 1e6);
  std::printf("GEMM FLOPs=%.2f GFLOP  peak bf16=%.0f TFLOPS\n\n", flops / 1e9, BF16_PEAK);

  std::vector<bf16> hA((size_t)M * H), hW(WBYTES / 2);
  fill_fast(hA.data(), hA.size(), 1234);
  fill_fast(hW.data(), hW.size(), 5678);
  std::vector<int> hT(M);
  for (int i = 0; i < M; ++i) hT[i] = rand() % V;

  bf16 *dA, *dW, *dC;
  int* dT;
  float *dpm, *dps, *dtl, *drowloss, *dloss;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&dW, WBYTES));
  CUDA_CHECK(cudaMalloc(&dC, LOGITBYTES));
  CUDA_CHECK(cudaMalloc(&dT, M * 4));
  const int TMAX = V / 128;  // 最细 BN=128
  CUDA_CHECK(cudaMalloc(&dpm, (size_t)M * TMAX * 4));
  CUDA_CHECK(cudaMalloc(&dps, (size_t)M * TMAX * 4));
  CUDA_CHECK(cudaMalloc(&dtl, M * 4));
  CUDA_CHECK(cudaMalloc(&drowloss, M * 4));
  CUDA_CHECK(cudaMalloc(&dloss, 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), WBYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dT, hT.data(), M * 4, cudaMemcpyHostToDevice));

  cublasHandle_t hb;
  cublasCreate(&hb);

  auto run_cublas = [&]() {
    const float alpha = 1.f, beta = 0.f;
    // row-major C[M,V] = A[M,H] * W[V,H]^T
    //   → col-major C'[V,M] = Wc^T * Ac，Wc=[K,N] col-major（即 W row-major），Ac=[K,M]（即 A row-major）
    cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, V, M, H, &alpha, dW, CUDA_R_16BF, H, dA,
                 CUDA_R_16BF, H, &beta, dC, CUDA_R_16BF, V, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };
  auto run_ce = [&]() {
    ce_baseline_kernel<<<M, 256>>>(dC, dT, drowloss, M, V);
    CUDA_CHECK(cudaMemset(dloss, 0, 4));
    sum_reduce_kernel<<<32, 256>>>(drowloss, dloss, M);
  };
  float loss_cublas = 0, loss_fused = 0;
  double t_gemm, t_ce, t_gemm_ce;

  if (std::strcmp(which, "all") == 0 || std::strcmp(which, "base") == 0) {
    run_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());
    t_gemm = bench_ms(run_cublas, 5, 20);
    t_ce = bench_ms(run_ce, 5, 20);
    t_gemm_ce = bench_ms([&]() { run_cublas(); run_ce(); }, 5, 20);
    CUDA_CHECK(cudaMemcpy(&loss_cublas, dloss, 4, cudaMemcpyDeviceToHost));
    loss_cublas /= M;
    std::printf("[baseline cuBLAS+CE]\n");
    std::printf("  gemm          %8.3f ms  %7.1f TFLOPS  (%4.1f%% peak)\n", t_gemm,
                flops / t_gemm / 1e9, 100 * flops / t_gemm / 1e9 / BF16_PEAK);
    std::printf("  ce            %8.3f ms\n", t_ce);
    std::printf("  total         %8.3f ms   loss=%.6f\n\n", t_gemm_ce, loss_cublas);
  } else {
    t_gemm = t_ce = t_gemm_ce = 0;
  }

  // ---- fused：plain（FUSE=false 写 bf16 logits，测 GEMM 裸吞吐）/ fused（partial lse）----
  struct G256 { enum { BM = 256, BN = 128, S = 4, GM = 0, GN = 0 }; };
  struct G128 { enum { BM = 128, BN = 128, S = 4, GM = 0, GN = 0 }; };
  struct G128n { enum { BM = 128, BN = 256, S = 3, GM = 0, GN = 0 }; };
  struct Gsw28 { enum { BM = 256, BN = 128, S = 4, GM = 2, GN = 8 }; };
  struct Gsw216 { enum { BM = 256, BN = 128, S = 4, GM = 2, GN = 16 }; };
  struct Gsw48 { enum { BM = 256, BN = 128, S = 4, GM = 4, GN = 8 }; };
  struct Gsw416 { enum { BM = 256, BN = 128, S = 4, GM = 4, GN = 16 }; };
  struct Gsw116 { enum { BM = 256, BN = 128, S = 4, GM = 1, GN = 16 }; };
  struct Gsw48_s3 { enum { BM = 256, BN = 128, S = 3, GM = 4, GN = 8 }; };
  struct Gsw48_s2 { enum { BM = 256, BN = 128, S = 2, GM = 4, GN = 8 }; };
  struct Gsw128_256 { enum { BM = 128, BN = 256, S = 3, GM = 4, GN = 8 }; };
  struct G256x256 { enum { BM = 256, BN = 256, S = 2, GM = 0, GN = 0 }; };

  if (std::strcmp(which, "prof") == 0) {
    constexpr int BM = 256, BN = 128, S = 4, GM = 4, GN = 8, T = V / BN;
    launch_fce<BM, BN, 64, S, true, GM, GN>(dA, dW, dC, dpm, dps, dtl, dT, M, V, H, 0);
    combine_kernel<<<32, 256>>>(dpm, dps, dtl, dloss, M, T);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[dbg] prof fused sw4x8 synced OK\n");
    return 0;
  }
  if (std::strcmp(which, "profnosw") == 0) {
    constexpr int BM = 256, BN = 128, S = 4, T = V / BN;
    launch_fce<BM, BN, 64, S, true>(dA, dW, dC, dpm, dps, dtl, dT, M, V, H, 0);
    combine_kernel<<<32, 256>>>(dpm, dps, dtl, dloss, M, T);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[dbg] prof nosw synced OK\n");
    return 0;
  }
  if (std::strcmp(which, "profplain") == 0) {
    constexpr int BM = 256, BN = 128, S = 4, GM = 4, GN = 8;
    launch_fce<BM, BN, 64, S, false, GM, GN>(dA, dW, dC, dpm, dps, dtl, dT, M, V, H, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[dbg] prof plain sw4x8 synced OK\n");
    return 0;
  }

  auto do_cfg = [&](const char* name, auto cfg) {
    constexpr int BM = decltype(cfg)::BM, BN = decltype(cfg)::BN, S = decltype(cfg)::S;
    constexpr int GM = decltype(cfg)::GM, GN = decltype(cfg)::GN;
    const int T = V / BN;
    double tp = bench_ms([&]() { launch_fce<BM, BN, 64, S, false, GM, GN>(dA, dW, dC, dpm, dps, dtl, dT, M, V, H, 0); }, 5, 20);
    CUDA_CHECK(cudaMemset(dloss, 0, 4));
    launch_fce<BM, BN, 64, S, true, GM, GN>(dA, dW, dC, dpm, dps, dtl, dT, M, V, H, 0);
    combine_kernel<<<std::min(1024, (M + 255) / 256), 256>>>(dpm, dps, dtl, dloss, M, T);
    CUDA_CHECK(cudaDeviceSynchronize());
    float lf = 0; CUDA_CHECK(cudaMemcpy(&lf, dloss, 4, cudaMemcpyDeviceToHost)); lf /= M;
    double tf = bench_ms([&]() {
      launch_fce<BM, BN, 64, S, true, GM, GN>(dA, dW, dC, dpm, dps, dtl, dT, M, V, H, 0);
      combine_kernel<<<std::min(1024, (M + 255) / 256), 256>>>(dpm, dps, dtl, dloss, M, T);
    }, 5, 20);
    std::printf("[%-16s] gemm %7.3f ms %6.1f TFLOPS | fused %7.3f ms %6.1f TFLOPS (%4.1f%% peak) | loss=%.6f\n",
                name, tp, flops / tp / 1e9, tf, flops / tf / 1e9, 100 * flops / tf / 1e9 / BF16_PEAK, lf);
  };

  std::printf("[fused sweep]  (gemm = FUSE=false 写 bf16 logits)\n");
  do_cfg("128x128 s4", G128{});
  do_cfg("128x256 s3", G128n{});
  do_cfg("256x128 s4", G256{});
  do_cfg("sw 1x16", Gsw116{});
  do_cfg("sw 2x8", Gsw28{});
  do_cfg("sw 2x16", Gsw216{});
  do_cfg("sw 4x8", Gsw48{});
  do_cfg("sw 4x16", Gsw416{});
  do_cfg("sw 4x8 s3", Gsw48_s3{});
  do_cfg("sw 4x8 s2", Gsw48_s2{});
  do_cfg("128x256 s3 sw", Gsw128_256{});
  do_cfg("256x256 s2", G256x256{});
  if (t_gemm_ce > 0)
    std::printf("baseline cuBLAS+CE total = %.3f ms (gemm %.3f + ce %.3f, loss %.6f)\n",
                t_gemm_ce, t_gemm, t_ce, loss_cublas);

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
