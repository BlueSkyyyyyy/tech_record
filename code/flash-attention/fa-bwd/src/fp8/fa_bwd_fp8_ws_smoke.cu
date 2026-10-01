// =============================================================================
// fa_bwd_fp8_ws_smoke.cu —— O120（第 214 轮）：fp8 主 kernel 的 **warp specialization**
//   de-risk（F3b② 的「producer/consumer + mbarrier 流水」）
// =============================================================================
// 动机（ROADMAP『fp8 专项冲刺』F6/F3b，及「阻塞」里的 O91 收口）：
//   fp8 默认 main 的墙是 dK/dV 跨 CTA 的 L2 `red`（105.4M 扇区、占 L2 ~80%）。唯一能改
//   「每 KV 元素被多少 m-block 归约」的软件杠杆是 **放大 BM**（BM 64→128/192 ⇒ red 砍半/砍到
//   1/3）。但放大 BM 后 CTA 的 smem/寄存器翻倍 ⇒ **1 CTA/SM**；O90/O91 实测：`wgmma2`(BM=128,
//   8 warp) 0.66–0.68× 默认、`wg3`(BM=192, 12 warp, 384 线程) **0.52×**——**red 确实降到
//   0.43×，但时间反而翻倍**。O91 的结论是「卡点不是 warp 数（wg3 的 12 warp/SM 已追平默认 3
//   CTA×4 warp），而是 **1 CTA/SM 的单一 barrier 域**：每 tile 的 5 个 `__syncthreads` 把
//   12 个 warp 串成依赖链，K/V 搬运（TMA/global 读）与 wgmma 计算无法重叠，跨 tile 延迟没有
//   别的独立 CTA 来填。」
//
//   这条结论**只被『改成多 warpgroup 后变慢』间接支持**，还没有被「把同步机制本身换掉」直接
//   检验过。本冒烟就做这件事：在与主 kernel **相同的 fp8 K/V 几何**（UINT8 / SW128 / box
//   {128,32} / KS_SZ=4096B）上，比较两种同步机制下的 **K/V TMA → wgmma 流水吞吐**：
//
//     (A) SYNC（现状 = O91 的形态）：每 tile 一条 TMA + `__syncthreads` 全 CTA 等 + 消费；
//         无跨 tile 预取，搬运与计算严格串行。
//     (B) WS（producer/consumer）：**独立 producer warpgroup** 跑 ring buffer 预取 K，
//         **consumer warpgroup** 只等 `mbarrier`（full）后做 wgmma、再 `arrive`（empty）；
//         搬运与计算解耦，流水深度 = NSTAGE。
//
//   判据：1 CTA/SM 下，(B) 的有效读带宽随 NSTAGE 上升能否逼近 L2 峰值（即延迟被彻底藏住）。
//   若能 ⇒ 说明「单一 barrier 域」确是 O91 的病因、warp specialization 是 BM≥128 的解锁路径
//   （下一轮可把它落进 `wgmma2`/`wg3`）；若 NSTAGE 再深也上不去 ⇒ WS 在 1 CTA/SM 上同样不
//   够（受 smem/寄存器墙 + L2 延迟限制），F3b② 收口。
//
// 本文件**不动任何默认路径**（独立冒烟），只产出「机制可行性」的证据。
//
// 编译：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//        scripts/run.sh src/fp8/fa_bwd_fp8_ws_smoke.cu [R]
//       `R` = 每个 CTA 扫多少遍 K（默认 32；总读 = grid*NT*KS_SZ）。
// =============================================================================

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---- 与主 kernel（O41 kvtma）一致的几何 ----
constexpr int D_ = 128;                       // head_dim（fp8 一行 128B = SW128 atom 整行）
constexpr int BN_ = 32;                       // 每 K/V tile 行数（= TMA boxR）
constexpr int KS_SZ = (BN_ / 8) * 1024;       // 4096B（[32][128] SW128）
constexpr int PROD_TH = 128;                  // producer warpgroup
constexpr int CONS_TH = 128;                  // consumer warpgroup
constexpr int THREADS = PROD_TH + CONS_TH;    // 256

// SW128 K-major（与 src/fp8/fa_bwd_fp8_kernels.cuh 的 sw128_off_fp8 逐字一致，K=128）
__host__ __device__ __forceinline__ int sw128_off_fp8(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(smem_u32(bar)), "r"(phase));
  }
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void tma_load_4d(const CUtensorMap* map, void* dst, int k0, int r0,
                                            int hd, int b, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(hd), "r"(b), "r"(smem_u32(bar))
      : "memory");
}

// ---- wgmma 数据通路（与 kernels.cuh 逐字同款；fp8 m64n32k32 SW128 直读）----
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define WS_HAS_WGMMA 1
#else
#define WS_HAS_WGMMA 0
#endif
__device__ __forceinline__ void wgmma_fence() {
#if WS_HAS_WGMMA
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if WS_HAS_WGMMA
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_wait0() {
#if WS_HAS_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ uint32_t sw128_k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint64_t make_desc_sw128_fp8(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ void wgmma_m64n32k32_e4e4(float (&d)[16], uint64_t da, uint64_t db) {
#if WS_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "%16, %17, p, %19, %20;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
#else
  (void)da; (void)db;
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
#endif
}
// Q[64][128]·Kᵀ[128][32]（A/B 均 SW128 K-major，K 归约维 128）。
__device__ __forceinline__ void wgmma_qkt(const char* Asw, const char* Bsw, float (&d)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
#pragma unroll
  for (int s = 0; s < D_ / 32; ++s) {
    uint64_t da = make_desc_sw128_fp8(sw128_k32_addr(aa, s), 1024);
    uint64_t db = make_desc_sw128_fp8(sw128_k32_addr(ba, s), 1024);
    wgmma_m64n32k32_e4e4(d, da, db);
  }
  wgmma_commit();
  wgmma_wait0();
}

// =============================================================================
// (A) SYNC：每 tile 一条 TMA，全 CTA `__syncthreads` 等；无跨 tile 预取（O91 形态）。
//     256 线程：producer WG(0-3) 与 consumer WG(4-7) 都参与 __syncthreads，但只有
//     consumer WG 跑 wgmma（对齐 O91：GEMM3/4/5 只 8 warp，其余 warp 空转但被同步串住）。
// =============================================================================
__global__ void __launch_bounds__(THREADS, 1) ws_sync_kernel(
    const __grid_constant__ CUtensorMap kmap, float* __restrict__ sink, int NT, int Hkv) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  char* Ks = smem + 1024;      // 1 stage
  char* As = smem + 1024 + KS_SZ;  // Q tile（A 操作数，固定）

  const int tid = threadIdx.x, wid = tid >> 5;
  const int h = blockIdx.x % Hkv;
  if (tid == 0) {
    mbar_init(full, 1);
    fence_mbar_init();
    // A 填常数（e4m3 的 0x30≈1.0），只需非零以免 wgmma 全 0。
    for (int i = 0; i < 8192; ++i) As[i] = (char)0x30;
  }
  __syncthreads();

  uint32_t ph = 0;
  float acc = 0.f;
  for (int t = 0; t < NT; ++t) {
    if (tid == 0) {
      mbar_arrive_expect(full, KS_SZ);
      tma_load_4d(&kmap, Ks, 0, t * BN_, h, 0, full);
    }
    mbar_wait(full, ph);
    ph ^= 1;
    __syncthreads();                       // 全 CTA 等 TMA（O91 的 barrier 域）
    if (wid >= 4) {                        // consumer WG
      float d[16];
      wgmma_qkt(As, Ks, d);
#pragma unroll
      for (int i = 0; i < 16; ++i) acc += d[i];
    }
    __syncthreads();                       // 消费完才允许下一条 TMA 覆写
  }
  if (wid >= 4) atomicAdd(sink, acc);
}

// =============================================================================
// (A2) SYNC-PREFETCH：ring buffer（NSTAGE）+ thread0 提前发 TMA，但**沿用全 CTA
//      `__syncthreads`**（无 producer/consumer warp specialization）。隔离「只要预取深度」
//      与「WS 解耦」两个因素：若 (A2)≈(B) ⇒ 真杠杆是「别把 TMA 串在 __syncthreads 里」。
// =============================================================================
template <int NSTAGE>
__global__ void __launch_bounds__(THREADS, 1) ws_syncpref_kernel(
    const __grid_constant__ CUtensorMap kmap, float* __restrict__ sink, int NT, int Hkv) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  char* Ks = smem + 1024;
  char* As = Ks + NSTAGE * KS_SZ;

  const int tid = threadIdx.x, wid = tid >> 5;
  const int h = blockIdx.x % Hkv;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) mbar_init(full + s, 1);
    for (int i = 0; i < 8192; ++i) As[i] = (char)0x30;
    fence_mbar_init();
  }
  __syncthreads();

  int pf[NSTAGE];
#pragma unroll
  for (int s = 0; s < NSTAGE; ++s) pf[s] = 0;
  // prologue：thread0 预取前 NSTAGE 个 tile
  if (tid == 0) {
    for (int tt = 0; tt < NSTAGE && tt < NT; ++tt) {
      const int s = tt % NSTAGE;
      mbar_arrive_expect(full + s, KS_SZ);
      tma_load_4d(&kmap, Ks + s * KS_SZ, 0, tt * BN_, h, 0, full + s);
    }
  }
  float acc = 0.f;
  for (int t = 0; t < NT; ++t) {
    const int s = t % NSTAGE;
    mbar_wait(full + s, (uint32_t)pf[s]);
    pf[s] ^= 1;
    if (wid >= 4) {
      float d[16];
      wgmma_qkt(As, Ks + s * KS_SZ, d);
#pragma unroll
      for (int i = 0; i < 16; ++i) acc += d[i];
    }
    __syncthreads();  // 全体消费完 stage s 才对 s 复用
    if (tid == 0 && t + NSTAGE < NT) {
      mbar_arrive_expect(full + s, KS_SZ);
      tma_load_4d(&kmap, Ks + s * KS_SZ, 0, (t + NSTAGE) * BN_, h, 0, full + s);
    }
  }
  if (wid >= 4) atomicAdd(sink, acc);
}

// =============================================================================
// (B) WS：producer WG(0-3) 跑 ring buffer（NSTAGE）预取 K；consumer WG(4-7) 只等
//     mbarrier(full) → wgmma → arrive(empty)。搬运与计算解耦，无全 CTA `__syncthreads`。
// =============================================================================
template <int NSTAGE>
__global__ void __launch_bounds__(THREADS, 1) ws_pipe_kernel(
    const __grid_constant__ CUtensorMap kmap, float* __restrict__ sink, int NT, int Hkv) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);          // NSTAGE
  uint64_t* empty = full + NSTAGE;                              // NSTAGE
  char* Ks = smem + 1024;                                       // NSTAGE * KS_SZ
  char* As = Ks + NSTAGE * KS_SZ;

  const int tid = threadIdx.x, wid = tid >> 5;
  const int h = blockIdx.x % Hkv;
  if (tid == 0) {
    // full[s]：producer 1 次 arrive + tx；empty[s]：consumer 128 线程各 arrive 一次。
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) mbar_init(full + s, 1);
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) mbar_init(empty + s, CONS_TH);
    for (int i = 0; i < 8192; ++i) As[i] = (char)0x30;
    fence_mbar_init();
  }
  __syncthreads();

  float acc = 0.f;
  if (wid < 4) {
    // ---- producer ----
    int pe[NSTAGE];
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) pe[s] = 0;
    for (int t = 0; t < NT; ++t) {
      const int s = t % NSTAGE;
      if (t >= NSTAGE) { mbar_wait(empty + s, (uint32_t)pe[s]); pe[s] ^= 1; }
      if (tid == 0) {  // 单线程发 TMA（lane==0 会命中每个 warp 各一，重复发 4 次）
        mbar_arrive_expect(full + s, KS_SZ);
        tma_load_4d(&kmap, Ks + s * KS_SZ, 0, t * BN_, h, 0, full + s);
      }
    }
  } else {
    // ---- consumer ----
    int pf[NSTAGE];
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) pf[s] = 0;
    for (int t = 0; t < NT; ++t) {
      const int s = t % NSTAGE;
      mbar_wait(full + s, (uint32_t)pf[s]);
      pf[s] ^= 1;
      float d[16];
      wgmma_qkt(As, Ks + s * KS_SZ, d);
#pragma unroll
      for (int i = 0; i < 16; ++i) acc += d[i];
      mbar_arrive(empty + s);
    }
    atomicAdd(sink, acc);
  }
}

// =============================================================================
// host
// =============================================================================
static CUtensorMap make_kmap(const void* ptr, long long Hkv, long long S) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D_, (uint64_t)S, (uint64_t)Hkv, 1};
  uint64_t strides[3] = {(uint64_t)(Hkv * D_), (uint64_t)D_, (uint64_t)(S * Hkv * D_)};
  uint32_t box[4] = {128, BN_, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s);
    std::exit(1);
  }
  return map;
}

template <int NS>
static float time_syncpref(const CUtensorMap& kmap, float* dsink, dim3 grid, int NT, int reps,
                           float* out_sink) {
  const int smem = 1024 + NS * KS_SZ + 8192;
  auto launch = [&]() {
    cudaError_t e =
        cudaFuncSetAttribute(ws_syncpref_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) { fprintf(stderr, "smem attr: %s\n", cudaGetErrorString(e)); std::exit(1); }
    ws_syncpref_kernel<NS><<<grid, THREADS, smem>>>(kmap, dsink, NT, /*Hkv=*/16);
    e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
  };
  if (out_sink) CUDA_CHECK(cudaMemset(dsink, 0, sizeof(float)));
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  if (out_sink) CUDA_CHECK(cudaMemcpy(out_sink, dsink, sizeof(float), cudaMemcpyDeviceToHost));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms / reps;
}

template <int NS>
static float time_pipe(const CUtensorMap& kmap, float* dsink, dim3 grid, int NT, int reps,
                       float* out_sink) {
  const int smem = 1024 + NS * KS_SZ + 8192;
  auto launch = [&]() {
    cudaError_t e = cudaFuncSetAttribute(ws_pipe_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) { fprintf(stderr, "smem attr: %s\n", cudaGetErrorString(e)); std::exit(1); }
    ws_pipe_kernel<NS><<<grid, THREADS, smem>>>(kmap, dsink, NT, /*Hkv=*/16);
    e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
  };
  if (out_sink) CUDA_CHECK(cudaMemset(dsink, 0, sizeof(float)));
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  if (out_sink) CUDA_CHECK(cudaMemcpy(out_sink, dsink, sizeof(float), cudaMemcpyDeviceToHost));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms / reps;
}

int main(int argc, char** argv) {
  const int S = 4096, Hkv = 16;
  int R = (argc > 1) ? atoi(argv[1]) : 32;   // 每 CTA 扫 K 的遍数
  int reps = 20;
  if (argc > 2) reps = atoi(argv[2]);
  const int NT = (S / BN_) * R;              // 每 CTA 处理的 K tile 数
  (void)cuInit(0);

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int nsm = prop.multiProcessorCount;

  const size_t kn = (size_t)S * Hkv * D_;
  std::vector<unsigned char> hK(kn);
  std::mt19937 rng(1234);
  // e4m3 的 0x7F/0xFF 是 NaN；限制到 [0,0x7E) 全是有限值，便于 WS/SYNC 数值一致性检查。
  for (size_t i = 0; i < kn; ++i) hK[i] = (unsigned char)(rng() % 0x7E);
  unsigned char* dK = nullptr;
  CUDA_CHECK(cudaMalloc(&dK, kn));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), kn, cudaMemcpyHostToDevice));
  CUtensorMap kmap = make_kmap(dK, Hkv, S);

  float* dsink = nullptr;
  CUDA_CHECK(cudaMalloc(&dsink, sizeof(float)));

  // 1 CTA/SM：grid = nsm（每 CTA 各自扫 NT 个 tile，K 常驻 L2 ⇒ 测的是 L2 读带宽 + 流水）。
  const dim3 grid(nsm);
  const double bytes = (double)nsm * NT * KS_SZ;   // 总读字节
  printf("=== fa_bwd_fp8_ws_smoke (O120) ===\n");
  printf("S=%d Hkv=%d D=%d BN=%d KS_SZ=%d  grid=%d(1 CTA/SM)  NT=%d/CTA  reps=%d\n", S, Hkv, D_,
         BN_, KS_SZ, nsm, NT, reps);
  printf("read bytes/launch = %.1f MB ; L2 峰值按 ~5.0 TB/s 参考\n\n", bytes / 1e6);

  // ---- (A) SYNC ----
  {
    const int smem = 1024 + KS_SZ + 8192;
    cudaFuncSetAttribute(ws_sync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    auto launch = [&]() {
      ws_sync_kernel<<<grid, THREADS, smem>>>(kmap, dsink, NT, Hkv);
    };
    CUDA_CHECK(cudaMemset(dsink, 0, sizeof(float)));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float sink = 0.f;
    CUDA_CHECK(cudaMemcpy(&sink, dsink, sizeof(float), cudaMemcpyDeviceToHost));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    ms /= reps;
    printf("[A] SYNC (__syncthreads, 1 stage)     : %.4f ms  %8.1f GB/s   sink=%.4e\n", ms,
           bytes / (ms * 1e-3) / 1e9, sink);
  }

  // ---- (A2) SYNC-PREFETCH（全 CTA __syncthreads，无 WS），各深度 ----
  {
    float s = 0.f;
    float ms = time_syncpref<2>(kmap, dsink, grid, NT, reps, &s);
    printf("[A2] SYNC-PREFETCH NSTAGE=2 (syncth)  : %.4f ms  %8.1f GB/s   sink=%.4e\n", ms,
           bytes / (ms * 1e-3) / 1e9, s);
  }
  {
    float s = 0.f;
    float ms = time_syncpref<4>(kmap, dsink, grid, NT, reps, &s);
    printf("[A2] SYNC-PREFETCH NSTAGE=4           : %.4f ms  %8.1f GB/s   sink=%.4e\n", ms,
           bytes / (ms * 1e-3) / 1e9, s);
  }
  {
    float s = 0.f;
    float ms = time_syncpref<8>(kmap, dsink, grid, NT, reps, &s);
    printf("[A2] SYNC-PREFETCH NSTAGE=8           : %.4f ms  %8.1f GB/s   sink=%.4e\n", ms,
           bytes / (ms * 1e-3) / 1e9, s);
  }

  // ---- (B) WS，各深度 ----
  float ref_sink = 0.f;
  {
    float ms = time_pipe<2>(kmap, dsink, grid, NT, reps, &ref_sink);
    printf("[B] WS NSTAGE=2 (producer/consumer)   : %.4f ms  %8.1f GB/s   sink=%.4e\n", ms,
           bytes / (ms * 1e-3) / 1e9, ref_sink);
  }
  {
    float s2 = 0.f;
    float ms = time_pipe<4>(kmap, dsink, grid, NT, reps, &s2);
    printf("[B] WS NSTAGE=4                       : %.4f ms  %8.1f GB/s   sink=%.4e  (dvsN2=%.3e)\n",
           ms, bytes / (ms * 1e-3) / 1e9, s2, s2 - ref_sink);
  }
  {
    float s2 = 0.f;
    float ms = time_pipe<8>(kmap, dsink, grid, NT, reps, &s2);
    printf("[B] WS NSTAGE=8                       : %.4f ms  %8.1f GB/s   sink=%.4e  (dvsN2=%.3e)\n",
           ms, bytes / (ms * 1e-3) / 1e9, s2, s2 - ref_sink);
  }

  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dsink));
  return 0;
}
