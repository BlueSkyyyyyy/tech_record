// nvcc 相邻 bf16 标量存储被合并的坑：小复现
//
// 场景：SW128 布局里同一行的相邻元素 (col, col+1) 落在同一 16B 单元内、字节相邻。
// 直觉写法是两个 bf16 标量存储；但 nvcc 会把它们**合并成一个 32-bit st.shared**
// 并只写低 16 位（丢掉 col+1），导致 P 矩阵一半元素变 0。
// 修法：显式打包成一次 u32 存储（`__floats2bfloat162_rn`）。
//
// 本程序用三条路径写入同一份 P 的 SW128 tile，两两对比字节：
//   ref : vectorized sw128_store16（16B 存储，正确布局基准）
//   bad : 两个 bf16 标量存储（触发 nvcc 合并）
//   fix : 打包 u32 存储
// 运行：ARCH=sm_90a scripts/run.sh 20-mla-wgmma-sw128/bf16_store_pitfall.cu
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
constexpr int BM = 64, KT = 64;

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void k(const bf16* P, char* ra, char* rb, char* rc, int* bad) {
  extern __shared__ __align__(1024) char smem[];
  char* pa = smem;                 // bad
  char* pb = smem + BM * KT * 2;   // ref
  char* pc = smem + 2 * BM * KT * 2;  // fix
  const int tid = threadIdx.x, lane = tid & 31, W = tid >> 5;
  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;
  if (tid < 128)
    for (int t = 0; t < 8; ++t) {
      const int col = t * 8 + (lane & 3) * 2;
      // bad：两个 bf16 标量存储
      *reinterpret_cast<bf16*>(pa + sw128_off(r0, col, KT)) = P[r0 * KT + col];
      *reinterpret_cast<bf16*>(pa + sw128_off(r0, col + 1, KT)) = P[r0 * KT + col + 1];
      *reinterpret_cast<bf16*>(pa + sw128_off(r1, col, KT)) = P[r1 * KT + col];
      *reinterpret_cast<bf16*>(pa + sw128_off(r1, col + 1, KT)) = P[r1 * KT + col + 1];
      // fix：打包 u32（col 偶数 -> sw128_off 4B 对齐）
      *reinterpret_cast<uint32_t*>(pc + sw128_off(r0, col, KT)) =
          pack2(P[r0 * KT + col], P[r0 * KT + col + 1]);
      *reinterpret_cast<uint32_t*>(pc + sw128_off(r1, col, KT)) =
          pack2(P[r1 * KT + col], P[r1 * KT + col + 1]);
    }
  for (int idx = tid; idx < BM * (KT / 8); idx += 256) {
    const int row = idx / (KT / 8), cu = idx % (KT / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&P[row * KT + cu * 8]);
    sw128_store16(pb, row, cu * 8, KT, v);
  }
  __syncthreads();
  if (tid == 0) {
    int nb = 0, nf = 0;
    for (int i = 0; i < BM * KT * 2; ++i) {
      if (pa[i] != pb[i]) ++nb;
      if (pc[i] != pb[i]) ++nf;
    }
    *bad = nb * 1000 + nf;
  }
}

int main() {
  std::vector<bf16> hP(BM * KT);
  srand(5);
  for (auto& x : hP) x = __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f));
  bf16* P;
  char *ra, *rb, *rc;
  int* bad;
  CUDA_CHECK(cudaMalloc(&P, BM * KT * 2));
  CUDA_CHECK(cudaMemcpy(P, hP.data(), BM * KT * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&ra, BM * KT * 2));
  CUDA_CHECK(cudaMalloc(&rb, BM * KT * 2));
  CUDA_CHECK(cudaMalloc(&rc, BM * KT * 2));
  CUDA_CHECK(cudaMalloc(&bad, 4));
  const size_t shm = 3 * BM * KT * 2;
  CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  k<<<1, 256, shm>>>(P, ra, rb, rc, bad);
  CUDA_CHECK_LAST();
  int v;
  CUDA_CHECK(cudaMemcpy(&v, bad, 4, cudaMemcpyDeviceToHost));
  int nb = v / 1000, nf = v % 1000;
  std::printf("bad (2x bf16 store) : byte diff = %d %s\n", nb, nb == 0 ? "OK" : "MISMATCH <- nvcc 合并掉高 16 位");
  std::printf("fix (packed u32)    : byte diff = %d %s\n", nf, nf == 0 ? "OK" : "MISMATCH");
  return 0;
}
