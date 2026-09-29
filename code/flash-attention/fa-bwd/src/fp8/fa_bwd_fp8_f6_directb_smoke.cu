// =============================================================================
// fa_bwd_fp8_f6_directb_smoke.cu —— F6 判定：能否从「K-major（N 连续）」的 fp8 tile
//   （含 wgmma 用的 SW128）**直接** ldmatrix 取出 mma.m16n8k32 的 B 片段？
// =============================================================================
// 背景（ROADMAP『fp8 专项冲刺』F6「剩余/下一步候选 ①」）：
//   第 141 轮定位 F6（BM=128 双 warpgroup）要冲 2 CTA/SM 的**最大障碍是 Qp/dOp（34.8KB
//   smem）**——Qp/dOp 是为 GEMM4/3（dK=scale·dSᵀ·Q、dV=Pᵀ·dO）准备的两份「K 配对」B 副本。
//   当时的判断是「可去掉 Qp/dOp：从 SW128 的 Q/dO tile 直接 `ldmatrix` 读出 B（kernel-opt
//   42 篇已证 SW128 16B chunk 可转置读）」。本文件**用最小复现判决这一前提是否成立**。
//
// 结论（本文件实测 + ptxas 约束，均为负结果）：
//   1) fp8 `mma.sync.aligned.m16n8k32` **只存在 `.row.col` 布局**：`.col.row`/`.row.row`/
//      `.col.col` 都被 ptxas 拒绝（见同目录 `fa_bwd_fp8_mma_variant_probe.*.out.txt`）。
//      即 B 必须是 **col-major**（对逻辑 B[K][N]，就是存成 [N][K]、K 连续）⇒ 必须转置。
//   2) fp8 的 `ldmatrix` 以 b16 为单位转置，而 **1 个 b16 = 2 个相邻 fp8**。K-major tile
//      （Q/dO 的 [m][d]，d=N 连续）里每个「16B 行」的 2-fp8 配对**沿 N**；`.trans` 只交换
//      8×8 矩阵的行列、**不改变 b16 内部的配对方向**，因此寄存器里得到的仍是「沿 N 的
//      4 个 fp8」，而不是 mma 需要的「沿 K 的 4 个 fp8」。O4b 的 `fa_bwd_fp8_trans_smoke.cu`
//      已给出同一结论（那轮改用「K 配对布局」绕开）。
//   3) 本文件进一步**穷举**从 K-major tile 用 `ldmatrix.x4.trans` 取回的 4 个寄存器里，
//      **任意 2 个**都拼不出与「col-major + `ldmatrix.x2`」逐位相同的 B 片段 ⇒ 不存在可用的
//      地址/寄存器重排方案。
//   ⇒ F6「去掉 Qp/dOp」在 fp8/`.row.col`/b16-pairing 的前提下**不可行**；F6 只能另寻出路
//     （物理转置 SW128 B：+16KB smem + scatter，O4b 已判净负；或换卡）。
//
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_f6_directb_smoke.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>

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

static constexpr int M = 16;   // mma 输出行
static constexpr int N = 8;    // mma 输出列（B 的逻辑列 = head_dim 方向）
static constexpr int K = 64;   // 归约维（= GEMM4 的 m 方向），2 个 mma k-step
static constexpr int NT = 32;  // 一个 warp

__host__ __device__ __forceinline__ unsigned char cvt_e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__host__ __device__ __forceinline__ float deq_e4m3(unsigned char q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half2float(__half(h));
}

__device__ __forceinline__ void mma_e4e4(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t d[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(addr));
}

// 布局：
//   * BC[n][k]  —— col-major（K 连续）参考，字节行距 BKLD；`ldmatrix.x2`（非 trans）读。
//   * BR[k][n]  —— K-major（N 连续，本文件要检验的「Q/dO 原布局」），字节行距 BKLD，
//                  每行 n=0..7 + 8 字节 padding（凑满 16B 的 ldmatrix 行）。
// 输出：ref（BC 路径）与 att（BR + x4.trans 的全部 4 个寄存器），供 host 穷举比对。
static constexpr int BKLD = K + 16;      // 字节行距（BC/BR 都用，n 只有 8 ⇒ 够）
static constexpr int BRSLD = 16;         // BR 一行 16B（8 个 n + 8B padding）

__global__ void directb_kernel(const unsigned char* __restrict__ B,
                               float* __restrict__ out_ref,
                               uint32_t* __restrict__ frag_att) {
  extern __shared__ __align__(16) char smem[];
  unsigned char* BC = reinterpret_cast<unsigned char*>(smem);          // [N][BKLD]
  unsigned char* BR = BC + N * BKLD;                                   // [K][BRSLD]
  const int lane = threadIdx.x & 31, t = threadIdx.x;
  for (int i = t; i < N * K; i += NT) {
    int n = i / K, k = i % K;
    BC[n * BKLD + k] = B[n * K + k];
  }
  for (int i = t; i < K * N; i += NT) {
    int k = i / N, n = i % N;
    BR[k * BRSLD + n] = B[n * K + k];
  }
  __syncthreads();

  // ---- 参考片段：直接从 col-major [N][K] 用 `ldmatrix.x2`（mma_block 的 B 路径）----
  // 一次 k-step 的 B 片段（K=32）。lane 0..7 给 n 行地址，lane 8..15 给 k+16 的 n 行。
  {
    const int brow = lane & 7;
    const int bcol = ((lane >> 3) & 1) * 16;
    uint32_t d[2];
    ldmatrix_x2(smem_u32(&BC[brow * BKLD + bcol]), d);
    out_ref[lane * 2 + 0] = d[0];
    out_ref[lane * 2 + 1] = d[1];
  }
  // ---- 尝试：从 K-major [K][N] 用 `ldmatrix.x4.trans`（lane 0..31 给 32 个 k 行）----
  {
    uint32_t d[4];
    ldmatrix_x4_trans(smem_u32(&BR[lane * BRSLD]), d);
#pragma unroll
    for (int i = 0; i < 4; ++i) frag_att[lane * 4 + i] = d[i];
  }
}

// host 参考：C[m][n] = sum_k A[m][k] B[n][k]
__global__ void ref_kernel(const unsigned char* __restrict__ A,
                           const unsigned char* __restrict__ B, float* __restrict__ C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int m = idx / N, n = idx % N;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += deq_e4m3(A[m * K + k]) * deq_e4m3(B[n * K + k]);
  C[m * N + n] = acc;
}

int main(int argc, char** argv) {
  printf("=== F6 判定：fp8 mma B 片段能否从 K-major (N 连续) tile 直接 ldmatrix ===\n");
  printf("M=%d N=%d K=%d  BKLD=%d BRSLD=%d\n", M, N, K, BKLD, BRSLD);

  std::vector<unsigned char> hA(M * K), hB(N * K);
  srand(20240929u);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = cvt_e4m3(2.f * ((float)rand() / RAND_MAX - 0.5f));
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = cvt_e4m3(2.f * ((float)rand() / RAND_MAX - 0.5f));

  unsigned char *dA, *dB;
  float *dC, *dRefFrag;
  uint32_t* dAttFrag;
  CUDA_CHECK(cudaMalloc(&dA, M * K));
  CUDA_CHECK(cudaMalloc(&dB, N * K));
  CUDA_CHECK(cudaMalloc(&dC, M * N * 4));
  CUDA_CHECK(cudaMalloc(&dRefFrag, 32 * 2 * 4));
  CUDA_CHECK(cudaMalloc(&dAttFrag, 32 * 4 * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K, cudaMemcpyHostToDevice));

  int smem = N * BKLD + K * BRSLD;
  directb_kernel<<<1, NT, smem>>>(dB, dRefFrag, dAttFrag);
  ref_kernel<<<(M * N + 255) / 256, 256>>>(dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> hRef(64), hAtt(128);
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRefFrag, 64 * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hAtt.data(), dAttFrag, 128 * 4, cudaMemcpyDeviceToHost));

  // 参考片段是 lane=l 的 (d0,d1) = hRef[2l],hRef[2l+1]；尝试片段是 lane=l 的 4 个 reg。
  // 穷举：对每个 lane，看 frag_att[l][0..3] 中**任意有序两个**是否等于 (d0,d1)。
  int lanes_match = 0;
  for (int l = 0; l < 32; ++l) {
    uint32_t want0 = hRef[2 * l], want1 = hRef[2 * l + 1];
    bool ok = false;
    for (int i = 0; i < 4 && !ok; ++i)
      for (int j = 0; j < 4; ++j)
        if (i != j && hAtt[l * 4 + i] == want0 && hAtt[l * 4 + j] == want1) ok = true;
    if (ok) ++lanes_match;
  }
  printf("  lane 级片段匹配（x4.trans 任选 2 reg vs col-major x2）：%d/32\n", lanes_match);
  printf("  lane0  ref = %08x %08x\n", hRef[0], hRef[1]);
  printf("  lane0  att = %08x %08x %08x %08x\n", hAtt[0], hAtt[1], hAtt[2], hAtt[3]);
  printf("=== %s ===\n",
         (lanes_match == 32)
             ? "PASS：K-major 直读可行（前提成立）"
             : "FAIL：K-major (N 连续) tile 无法用 ldmatrix.trans 拼出 mma 的 B 片段 ⇒ F6『去 Qp/dOp』不可行");

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRefFrag); cudaFree(dAttFrag);
  return (lanes_match == 32) ? 1 : 0;
}
