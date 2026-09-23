// =============================================================================
// fa_bwd_fp16_tma_smoke.cu —— O15 前置：TMA（cp.async.bulk.tensor）+ SW128 冒烟
// =============================================================================
// 目的：fp16/bf16 反向的 main kernel 目前用 `cp.async.cg` 手工把 Q/K/V/dO 写成 SW128
// （逐 16B unit + 地址计算）。O15 想换成 Hopper 的 **TMA**（bulk tensor）+ mbarrier：
// 省掉 load 指令/地址运算、让 bulk 引擎搬运，并允许更深的跨-tile 流水。
//
// 上整条反向之前，必须先用最小复现证明两件事：
//   1) `cuTensorMapEncodeTiled(SWIZZLE_128B)` 硬件写出的 smem 布局 **逐字节等于** kernel
//      里 `sw128_off()` 描述的 K-major SW128（布局 `[row/8][k/64][8][64]`，atom 1024B）。
//      注意：一个 TMA box 的内维 (128B) 只覆盖 64 个 fp16，所以 HD=128 的 tile 要拆成
//      **2 个 k-chunk（{k0,k64}）**，各自 8KB；这在 wgmma 侧用两个 `SBO=1024` 的描述符读。
//   2) TMA 搬进来的 tile 能被 `wgmma.m64n64k16` 直接消费（QKᵀ 对拍）。
//
// 参考：A[64][128]、B[64][128]（K-major），算 C=A·Bᵀ。
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#define CU_CHECK(call)                                                          \
  do {                                                                          \
    CUresult _r = (call);                                                       \
    if (_r != CUDA_SUCCESS) {                                                   \
      const char* s = "?";                                                      \
      cuGetErrorString(_r, &s);                                                 \
      fprintf(stderr, "CU error %s at %s:%d\n", s, __FILE__, __LINE__);         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ uint32_t sw128_k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
      "r"(bytes));
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
// 一条 2D TMA：把全局 [rows][HD] 里 (r0,k0) 起的 [64 行][64 列] 搬进 dst（SW128）。
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int k0,
                                             int r0, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(smem_u32(bar)));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_m64n64k16_f16(float (&d)[32], uint64_t da,
                                                    uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1));
}

constexpr int M = 64, N = 64, HD = 128;
// 每个 k-chunk（64 列）的 SW128 tile = (M/8)*1024 = 8KB
constexpr int CH = (M / 8) * 1024;

// qdst/kdst：两个 8KB 区域（kchunk 0 / 1）；同时把原始行主序 tile 存进 ref 供比对。
__global__ void __launch_bounds__(128) tma_qkt(const __grid_constant__ CUtensorMap qmap,
                                               const __grid_constant__ CUtensorMap kmap,
                                               const __half* __restrict__ Q,
                                               const __half* __restrict__ Kt,
                                               float* C, int* mismatch) {
  extern __shared__ __align__(1024) char smem[];
  char* sQ0 = smem;            // Q kchunk0 (8KB)
  char* sQ1 = sQ0 + CH;        // Q kchunk1 (8KB)
  char* sK0 = sQ1 + CH;        // K kchunk0
  char* sK1 = sK0 + CH;        // K kchunk1
  uint64_t* bar = reinterpret_cast<uint64_t*>(sK1 + CH);

  const int tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(bar, 1);
    mbar_arrive_expect(bar, 4 * CH);  // 4 个 8KB box
    tma_load_2d(sQ0, &qmap, 0, 0, bar);
    tma_load_2d(sQ1, &qmap, 64, 0, bar);
    tma_load_2d(sK0, &kmap, 0, 0, bar);
    tma_load_2d(sK1, &kmap, 64, 0, bar);
  }
  __syncthreads();
  mbar_wait(bar, 0);
  __syncthreads();

  // 校验：TMA 写出的 4 个 tile 逐字节 == 软件 sw128_store16(Q/Kt) 布局
  {
    int bad = 0;
    for (int i = tid; i < M * HD / 8; i += 128) {
      const int row = i / (HD / 8), k0 = (i % (HD / 8)) * 8;
      uint4 v = *reinterpret_cast<const uint4*>(Q + row * HD + k0);
      const char* tile = (k0 < 64) ? sQ0 : sQ1;
      const int kk = k0 % 64;
      uint4 got = *reinterpret_cast<const uint4*>(tile + sw128_off(row, kk, 64));
      if (got.x != v.x || got.y != v.y || got.z != v.z || got.w != v.w) bad++;
    }
    for (int i = tid; i < N * HD / 8; i += 128) {
      const int row = i / (HD / 8), k0 = (i % (HD / 8)) * 8;
      uint4 v = *reinterpret_cast<const uint4*>(Kt + row * HD + k0);
      const char* tile = (k0 < 64) ? sK0 : sK1;
      const int kk = k0 % 64;
      uint4 got = *reinterpret_cast<const uint4*>(tile + sw128_off(row, kk, 64));
      if (got.x != v.x || got.y != v.y || got.z != v.z || got.w != v.w) bad++;
    }
    if (bad) atomicAdd(mismatch, bad);
  }

  // 用 wgmma 消费这 4 个 tile：C = Q·Kᵀ，每个 k-chunk 用 SBO=1024 的描述符
  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t q0 = smem_u32(sQ0), q1 = smem_u32(sQ1);
  const uint32_t k0 = smem_u32(sK0), k1 = smem_u32(sK1);
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(q0, s), 1024),
                        make_desc_sw128(sw128_k16_addr(k0, s), 1024));
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(q1, s), 1024),
                        make_desc_sw128(sw128_k16_addr(k1, s), 1024));
  }
  wgmma_commit();
  wgmma_wait0();

  const int wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = wid * 16 + g + (q >= 2 ? 8 : 0);
      const int col = j * 8 + c2 + (q & 1);
      C[row * N + col] = d[j * 4 + q];
    }
}

static CUtensorMap make_map(const __half* ptr) {
  CUtensorMap map;
  uint64_t dims[2] = {(uint64_t)HD, (uint64_t)M};       // inner (k), outer (rows)
  uint64_t strides[1] = {(uint64_t)HD * sizeof(__half)};  // row stride bytes
  uint32_t box[2] = {64, 64};                            // 64 k × 64 rows
  uint32_t estr[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

int main() {
  std::vector<float> Q(M * HD), Kt(N * HD), C(M * N), Cref(M * N);
  srand(7);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : Q) x = rnd();
  for (auto& x : Kt) x = rnd();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < HD; ++k) acc += Q[m * HD + k] * Kt[n * HD + k];
      Cref[m * N + n] = acc;
    }
  std::vector<__half> Qh(M * HD), Kh(N * HD);
  for (int i = 0; i < M * HD; ++i) Qh[i] = __float2half(Q[i]);
  for (int i = 0; i < N * HD; ++i) Kh[i] = __float2half(Kt[i]);

  __half *dQ, *dK;
  float* dC;
  int* dBad;
  CUDA_CHECK(cudaMalloc(&dQ, Qh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dK, Kh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dBad, sizeof(int)));
  CUDA_CHECK(cudaMemset(dBad, 0, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dQ, Qh.data(), Qh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, Kh.data(), Kh.size() * sizeof(__half), cudaMemcpyHostToDevice));

  CUtensorMap qmap = make_map(dQ);
  CUtensorMap kmap = make_map(dK);

  int smem = 4 * CH + 16;
  CUDA_CHECK(cudaFuncSetAttribute(tma_qkt, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  tma_qkt<<<1, 128, smem>>>(qmap, kmap, dQ, dK, dC, dBad);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  int bad = 0;
  CUDA_CHECK(cudaMemcpy(&bad, dBad, sizeof(int), cudaMemcpyDeviceToHost));

  double e = 0;
  for (int i = 0; i < M * N; ++i) e = std::max(e, (double)std::fabs(C[i] - Cref[i]));
  printf("TMA layout byte-mismatch = %d (expect 0)\n", bad);
  printf("TMA->wgmma QKt vs CPU: max_abs=%.3e\n", e);
  bool ok = (bad == 0) && (e < 1e-2);
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dQ); cudaFree(dK); cudaFree(dC); cudaFree(dBad);
  return ok ? 0 : 1;
}
