// =============================================================================
// fa_bwd_fp16_wgmma_smoke.cu —— O9 前置：wgmma.m64n64k16 SS + SW128 布局冒烟
// =============================================================================
// 目的：fp16/bf16 反向的 5 个 GEMM 要上 Hopper `wgmma`（SS，A/B 均来自 smem 描述符）。
// 上整条反向之前，先用最小 GEMM 证明：
//   1) SW128（128B swizzle）K-major smem 布局与 wgmma 描述符自洽（`sw128_off` 存、
//      `make_desc_sw128` 读）；
//   2) `wgmma.mma_async.m64n64k16.f32.f16.f16` 的 **accumulator 布局**：warp w 持行
//      `[16w,16w+16)`，warp 内 `d[j*4+q]` ↔ `row=16w+g+(q>=2?8:0)`、`col=j*8+2*(lane%4)+(q&1)`
//      （与 mma.m16n8 的 `acc[i][j][q]` 同构，方便直接套现有 epilogue）。
// 这两个映射正好对应反向里的 GEMM1/2/5（B=[N][K]，K-major）。
// 参考：C = Q·Kᵀ（Q/K 都是 [64][128]，K-major 归约维 128）。
// =============================================================================

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

// ---------- SW128 布局（与 code/kernel-opt/20-mla-wgmma-sw128/wgmma_sw128.cuh 一致） ----------
// 元素 (row, k) -> 字节偏移；K = tile 行宽（沿 K 的元素数，须为 64 的倍数）。
// 布局 [row/8][k/64][8 行][64 元素]，atom 1024B；atom 内 16B chunk c' = c ^ r。
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
// 存一个 16B（8 个 fp16）到 SW128 tile：要求 k0 % 8 == 0。
__device__ __forceinline__ void sw128_store16(char* tile, int row, int k0, int K,
                                              uint4 v) {
  *reinterpret_cast<uint4*>(tile + sw128_off(row, k0, K)) = v;
}
// k16 块索引 s（沿 K，从 0 起）对应的描述符地址增量（字节）。
__device__ __forceinline__ uint32_t sw128_k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr,
                                                    uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;  // LBO = 1（B128 下硬件忽略）
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;  // base_offset
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}

// ---------- wgmma ----------
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

constexpr int M = 64, N = 64, K = 128;

__global__ void __launch_bounds__(128) kqkt_wgmma(const __half* Q,
                                                  const __half* Kt, float* C) {
  __shared__ __align__(1024) char sQ[(M / 8) * (K / 64) * 1024];
  __shared__ __align__(1024) char sK[(N / 8) * (K / 64) * 1024];
  const int tid = threadIdx.x;
  // 行主序 [row][k] 整数元素 -> SW128 存（每线程搬 16B = 8 个 fp16）
  for (int i = tid; i < M * K / 8; i += 128) {
    const int row = i / (K / 8), k0 = (i % (K / 8)) * 8;
    uint4 v = *reinterpret_cast<const uint4*>(Q + row * K + k0);
    sw128_store16(sQ, row, k0, K, v);
  }
  for (int i = tid; i < N * K / 8; i += 128) {
    const int row = i / (K / 8), k0 = (i % (K / 8)) * 8;
    uint4 v = *reinterpret_cast<const uint4*>(Kt + row * K + k0);
    sw128_store16(sK, row, k0, K, v);
  }
  __syncthreads();

  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;

  wgmma_fence();
  const uint32_t qa = smem_u32(sQ), ka = smem_u32(sK);
  const uint32_t sbo = (uint32_t)((K / 64) * 1024);
#pragma unroll
  for (int s = 0; s < K / 16; ++s) {
    uint64_t da = make_desc_sw128(sw128_k16_addr(qa, s), sbo);
    uint64_t db = make_desc_sw128(sw128_k16_addr(ka, s), sbo);
    wgmma_m64n64k16_f16(d, da, db);
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

int main() {
  std::vector<float> Q(M * K), Kt(N * K), C(M * N), Cref(M * N);
  srand(123);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };
  for (auto& x : Q) x = rnd();
  for (auto& x : Kt) x = rnd();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += Q[m * K + k] * Kt[n * K + k];
      Cref[m * N + n] = acc;
    }
  std::vector<__half> Qh(M * K), Kh(N * K);
  for (int i = 0; i < M * K; ++i) Qh[i] = __float2half(Q[i]);
  for (int i = 0; i < N * K; ++i) Kh[i] = __float2half(Kt[i]);

  __half *dQ, *dK;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dQ, Qh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dK, Kh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, Qh.data(), Qh.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, Kh.data(), Kh.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  kqkt_wgmma<<<1, 128>>>(dQ, dK, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(C.data(), dC, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double e = 0;
  for (int i = 0; i < M * N; ++i)
    e = std::max(e, (double)std::fabs(C[i] - Cref[i]));
  printf("wgmma.m64n64k16 SW128 vs CPU: max_abs=%.3e\n", e);
  bool ok = e < 1e-2;
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dQ); cudaFree(dK); cudaFree(dC);
  return ok ? 0 : 1;
}
