// 48 附：`mma.sync.m16n8k32` int8（IMMA）片段布局最小验证。
//
// 在做完整 W4A8 GEMM 之前，先用一个 16x32x8 的小 mma 验证
//   · A 片段（4×.b32 = 16×int8）：a0/a1 覆盖 k=0..15，a2/a3 覆盖 k=16..31
//     row = lane>>2（a0/a2）、(lane>>2)+8（a1/a3）
//     col = (lane%4)*4 + i
//   · B 片段（2×.b32 = 8×int8，B 存成 [n][k]）：b0 覆盖 k=0..15，b1 覆盖 k=16..31
//     row(k) = (lane%4)*4 + i，col(n) = lane>>2
//   · C（4×.s32）：c0/c1 = row lane>>2, col (lane%4)*2+{0,1}；c2/c3 = row+8
// 用「直接从行主序 smem 取 32-bit word」的方式喂片段，一次走通就说明
// 后面可以完全不用 ldmatrix（int8 的 mma A/B 片段每 lane 恰好一个 word）。
//
// 运行：
//   scripts/run.sh 48-w4a8-imma/imma_test.cu
#include "../common/cuda_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ void imma_m16n8k32(int c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A: [16][32] int8（m,k 行主序）；B: [8][32] int8（n,k 行主序）；C: [16][8] int32。
// 每 lane 用 32-bit load 取片段：A 的 a0..a3，B 的 b0/b1。
__global__ void imma_test_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 int32_t* __restrict__ C) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;      // groupID
  const int t = lane & 3;       // threadID in group
  const uint32_t* A32 = reinterpret_cast<const uint32_t*>(A);
  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  uint32_t a[4], b[2];
  // A row-major [16][32] -> word index row*8 + colword
  a[0] = A32[g * 8 + t];            // row g,   cols 4t..4t+3
  a[1] = A32[(g + 8) * 8 + t];      // row g+8, cols 4t..4t+3
  a[2] = A32[g * 8 + 4 + t];        // row g,   cols 16+4t..
  a[3] = A32[(g + 8) * 8 + 4 + t];  // row g+8, cols 16+4t..
  // B row-major [8][32] (n,k) -> word index n*8 + kword
  b[0] = B32[g * 8 + t];        // n=g, k 4t..4t+3
  b[1] = B32[g * 8 + 4 + t];    // n=g, k 16+4t..
  int c[4] = {0, 0, 0, 0};
  imma_m16n8k32(c, a, b);
  // C row-major [16][8]
  C[g * 8 + t * 2 + 0] = c[0];
  C[g * 8 + t * 2 + 1] = c[1];
  C[(g + 8) * 8 + t * 2 + 0] = c[2];
  C[(g + 8) * 8 + t * 2 + 1] = c[3];
}

int main() {
  int8_t hA[16 * 32], hB[8 * 32];
  int32_t hC[16 * 8], ref[16 * 8];
  srand(7);
  for (int i = 0; i < 16 * 32; ++i) hA[i] = (int8_t)(rand() % 255 - 127);
  for (int i = 0; i < 8 * 32; ++i) hB[i] = (int8_t)(rand() % 255 - 127);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      int s = 0;
      for (int k = 0; k < 32; ++k) s += (int)hA[m * 32 + k] * (int)hB[n * 32 + k];
      ref[m * 8 + n] = s;
    }
  int8_t *dA, *dB;
  int32_t* dC;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
  CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, sizeof(hC)));
  imma_test_kernel<<<1, 32>>>(dA, dB, dC);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < 16 * 8; ++i)
    if (hC[i] != ref[i]) {
      if (bad < 8)
        std::printf("  MISMATCH i=%d (m=%d,n=%d) got %d ref %d\n", i, i / 8, i % 8, hC[i], ref[i]);
      ++bad;
    }
  std::printf("IMMA m16n8k32 layout test: %s (%d/128 mismatches)\n", bad ? "FAIL" : "OK", bad);
  return bad ? 1 : 0;
}
