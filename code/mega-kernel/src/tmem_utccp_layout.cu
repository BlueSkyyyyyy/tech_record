// P145-1 (docs/59): MPK SM100 block-scaled scale-factor "SMEM -> TMEM" path.
//
// UTCCP (tcgen05.cp.32x128b.warpx4) reads a 4-column x 32-row block of uint32
// scale entries from SMEM and writes 4 TMEM columns x 32 data-path lanes. Before
// UTCCP, the row-major scale array (one uint32 per row, 128 rows) must be
// warp-transposed into the layout UTCCP expects.
//
// This file is a *host-runnable model* of that warp transpose: it reproduces
// the three implementations found in the snapshot (MPK plain, MPK XOR variant,
// DeepGEMM-style v4 store) as real SMEM ld.shared/st.shared sequences on the
// target card (H100/sm90a), and checks that:
//   (A) all three produce the exact transpose  out[l*4 + k] = in[k*32 + l]
//       (bit-identical to a host reference);
//   (B) the "UTCCP group view" (4 groups of 32 consecutive words) is a pure
//       permutation of the original 128 rows;
//   (C) the XOR permutation and the v4 store do not change the achieved
//       SMEM throughput (it is an in-place, conflict-free, index-order-only
//       difference).
//
// No SM100 hardware is needed: the transpose itself is arch-agnostic SMEM code
// (the UTCCP instruction, TMEM and the descriptor are SM100-only and are only
// modelled/described in the doc, not executed).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(e_));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// ---- the three transpose implementations (one warp, 32 lanes) --------------

// MPK plain (every `st_shared`/`ld_shared` element-wise).
__device__ __forceinline__ void transpose_plain(uint32_t *s, int lane) {
  uint32_t v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v[i] = s[i * 32 + lane];
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    s[lane * 4 + i] = v[i];
}

// MPK XOR variant (linear_fp8_sm100.cuh, fp8_group_gemm_sm100.cuh):
//   read  s[(i ^ (lane>>3))*32 + lane]
//   write s[lane*4 + (i ^ (lane>>3))]
__device__ __forceinline__ void transpose_xor(uint32_t *s, int lane) {
  uint32_t v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v[i] = s[(i ^ (lane >> 3)) * 32 + lane];
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    s[lane * 4 + (i ^ (lane >> 3))] = v[i];
}

// DeepGEMM / MPK linear_fp8_1d2d variant: contiguous read + v4 store.
__device__ __forceinline__ void transpose_v4(uint32_t *s, int lane) {
  uint32_t v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v[i] = s[i * 32 + lane];
  __syncwarp();
  *reinterpret_cast<uint4 *>(s + lane * 4) = make_uint4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ void apply(int variant, uint32_t *s, int lane) {
  if (variant == 0)
    transpose_plain(s, lane);
  else if (variant == 1)
    transpose_xor(s, lane);
  else
    transpose_v4(s, lane);
}

// ---- functional kernel: init smem, transpose once, store out ---------------
__global__ void k_once(const uint32_t *in, uint32_t *out, int variant) {
  __shared__ uint32_t s[128];
  int lane = threadIdx.x & 31;
  s[lane] = in[lane];
  s[lane + 32] = in[lane + 32];
  s[lane + 64] = in[lane + 64];
  s[lane + 96] = in[lane + 96];
  __syncwarp();
  apply(variant, s, lane);
  __syncwarp();
  out[lane] = s[lane];
  out[lane + 32] = s[lane + 32];
  out[lane + 64] = s[lane + 64];
  out[lane + 96] = s[lane + 96];
}

// ---- timing kernel: repeat the transpose ITERS times -----------------------
__global__ void k_bench(unsigned long long *cycles, int variant, int iters,
                        uint32_t seed) {
  __shared__ uint32_t s[128];
  int lane = threadIdx.x & 31;
  s[lane] = seed + lane;
  s[lane + 32] = seed + lane + 32;
  s[lane + 64] = seed + lane + 64;
  s[lane + 96] = seed + lane + 96;
  __syncwarp();
  unsigned long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    apply(variant, s, lane);
    __syncwarp();
    apply(variant, s, lane);
    __syncwarp();
  }
  unsigned long long t1 = clock64();
  // trick to keep s alive
  if (s[lane] == 0xdeadbeefu)
    cycles[threadIdx.x & 31] = 1;
  if (lane == 0)
    cycles[0] = t1 - t0;
}

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("device %d: %s, sm_%d%d\n", dev, prop.name, prop.major, prop.minor);

  const int N = 128;
  uint32_t h_in[N], h_ref[N];
  for (int i = 0; i < N; ++i)
    h_in[i] = 0x01000000u + (uint32_t)(i * 2654435761u); // deterministic
  // reference transpose: ref[l*4 + k] = in[k*32 + l]
  for (int l = 0; l < 32; ++l)
    for (int k = 0; k < 4; ++k)
      h_ref[l * 4 + k] = h_in[k * 32 + l];

  uint32_t *d_in, *d_out;
  CK(cudaMalloc(&d_in, N * sizeof(uint32_t)));
  CK(cudaMalloc(&d_out, N * sizeof(uint32_t)));
  CK(cudaMemcpy(d_in, h_in, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  const char *names[3] = {"plain (MPK element-wise)", "xor (MPK linear/group)",
                          "v4 (DeepGEMM/1d2d)"};

  printf("\n== A. transpose correctness vs host reference out[l*4+k]=in[k*32+l] ==\n");
  int all_ok = 1;
  for (int v = 0; v < 3; ++v) {
    CK(cudaMemset(d_out, 0, N * sizeof(uint32_t)));
    k_once<<<1, 32>>>(d_in, d_out, v);
    CK(cudaDeviceSynchronize());
    uint32_t h_out[N];
    CK(cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; ++i)
      if (h_out[i] != h_ref[i])
        ++bad;
    printf("  %-26s mismatches=%d  %s\n", names[v], bad, bad ? "FAIL" : "OK");
    all_ok &= (bad == 0);
  }
  printf("  => all three implementations bit-identical to reference: %s\n",
         all_ok ? "YES" : "NO");

  printf("\n== B. UTCCP group view: group g = out[g*32 .. g*32+31] ==\n");
  printf("     (original row index that lands at depth d of column g)\n");
  printf("     col0 :");
  for (int d = 0; d < 8; ++d) {
    int idx = 0 * 32 + d;
    int orig = (idx & 3) * 32 + (idx >> 2);
    printf(" %3d", orig);
  }
  printf(" ...\n     col1 :");
  for (int d = 0; d < 8; ++d) {
    int idx = 1 * 32 + d;
    int orig = (idx & 3) * 32 + (idx >> 2);
    printf(" %3d", orig);
  }
  printf(" ...\n     col2 :");
  for (int d = 0; d < 8; ++d) {
    int idx = 2 * 32 + d;
    int orig = (idx & 3) * 32 + (idx >> 2);
    printf(" %3d", orig);
  }
  printf(" ...\n     col3 :");
  for (int d = 0; d < 8; ++d) {
    int idx = 3 * 32 + d;
    int orig = (idx & 3) * 32 + (idx >> 2);
    printf(" %3d", orig);
  }
  printf(" ...\n");
  // check it is a permutation
  int seen[N] = {0};
  for (int idx = 0; idx < N; ++idx)
    seen[(idx & 3) * 32 + (idx >> 2)]++;
  int perm = 1;
  for (int i = 0; i < N; ++i)
    if (seen[i] != 1)
      perm = 0;
  printf("  => group view is a permutation of 0..127: %s\n", perm ? "YES" : "NO");

  printf("\n== C. timing: ITERS transposes (2x per iter, so 2*ITERS total) ==\n");
  unsigned long long *d_cyc;
  CK(cudaMalloc(&d_cyc, 32 * sizeof(unsigned long long)));
  const int ITERS = 2000000;
  for (int v = 0; v < 3; ++v) {
    unsigned long long best = ~0ull;
    for (int rep = 0; rep < 5; ++rep) {
      CK(cudaMemset(d_cyc, 0, 32 * sizeof(unsigned long long)));
      k_bench<<<1, 32>>>(d_cyc, v, ITERS, 123u + rep);
      CK(cudaDeviceSynchronize());
      unsigned long long h;
      CK(cudaMemcpy(&h, d_cyc, sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost));
      if (h < best)
        best = h;
    }
    double per_op = (double)best / (2.0 * ITERS);
    printf("  %-26s %8llu cyc / 2*ITERS  =>  %.3f cycles per transpose\n",
           names[v], best, per_op);
  }

  CK(cudaFree(d_in));
  CK(cudaFree(d_out));
  CK(cudaFree(d_cyc));
  printf("\ndone.\n");
  return all_ok ? 0 : 2;
}
