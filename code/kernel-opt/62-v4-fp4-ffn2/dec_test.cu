// 最小复现：验证 PRMT 版 e2m1->int8 解码（decode_word_prmt）与 PRMT selector 语义。
//   g++ 风格自测在 device 上穷举：
//     ① 16 个 nibble 单独解码 == e2m1_i8
//     ② 一个 uint32 的全部 8 个 nibble（自然顺序）== e2m1_i8
//     ③ selector field 0..15 的行为（0-3 取 a、4-7 取 b、8-15 回绕到 v&7）
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ int e2m1_i8(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  int v = (int)(((e ? 2u : 0u) + m) << (e ? e - 1 : 0u));
  return s ? -v : v;
}

__device__ __forceinline__ void decode_word_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  constexpr uint32_t T0 = 0x03020100u, T1 = 0x0C080604u;
  constexpr uint32_t T2 = 0xFDFEFF00u, T3 = 0xF4F8FAFCu;
  const uint32_t hi = w >> 16;
  const uint32_t mlo = __byte_perm(T0, T1, w), mhi = __byte_perm(T0, T1, hi);
  const uint32_t slo = __byte_perm(T2, T3, w), shi = __byte_perm(T2, T3, hi);
  const uint32_t klo = __byte_perm(0u, ~0u, ((w >> 3) & 0x11111111u) * 5u);
  const uint32_t khi = __byte_perm(0u, ~0u, ((hi >> 3) & 0x11111111u) * 5u);
  a = mlo ^ ((mlo ^ slo) & klo);
  b = mhi ^ ((mhi ^ shi) & khi);
}

__global__ void test(int* bad, int* sel_wrap) {
  int nb = 0;
  // ① 单 nibble：把它放在 w 的最低字节，检查 out_a 的最低字节
  for (int n = 0; n < 16; ++n) {
    uint32_t a, b;
    decode_word_prmt((uint32_t)n, a, b);
    if ((int)(int8_t)(a & 0xFF) != e2m1_i8(n)) ++nb;
  }
  // ② 一个 uint32 的 8 个 nibble（自然顺序）
  for (int t = 0; t < 256; ++t) {
    uint32_t w = 0;
    for (int k = 0; k < 8; ++k) {
      int n = (t * 7 + k * 13) & 0xF;
      w |= (uint32_t)n << (4 * k);
    }
    uint32_t a, b;
    decode_word_prmt(w, a, b);
    for (int k = 0; k < 4; ++k)
      if ((int)(int8_t)((a >> (8 * k)) & 0xFF) != e2m1_i8((w >> (4 * k)) & 0xF)) ++nb;
    for (int k = 0; k < 4; ++k)
      if ((int)(int8_t)((b >> (8 * k)) & 0xFF) != e2m1_i8((w >> (4 * (k + 4))) & 0xF)) ++nb;
  }
  bad[0] = nb;
  // ③ selector field 行为
  for (int v = 0; v < 16; ++v) {
    uint32_t sel = v | (v << 4) | (v << 8) | (v << 12);
    uint32_t r = __byte_perm(0x33323130u, 0x37363534u, sel);  // a=0x30..0x33, b=0x34..0x37
    // 期望：v&7 -> 字节 0x30+(v&7)
    uint32_t want = 0x30303030u + (uint32_t)(v & 7) * 0x01010101u;
    sel_wrap[v] = (r == want) ? 1 : 0;
  }
}

int main() {
  int *d, *dw;
  cudaMalloc(&d, 4);
  cudaMalloc(&dw, 16 * 4);
  test<<<1, 1>>>(d, dw);
  int bad, wrap[16];
  cudaMemcpy(&bad, d, 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(wrap, dw, 16 * 4, cudaMemcpyDeviceToHost);
  printf("decode_word_prmt exhaustive: bad=%d %s\n", bad, bad == 0 ? "OK" : "FAIL");
  printf("PRMT selector field behavior (1 = 按 v&7 回绕):\n  ");
  for (int v = 0; v < 16; ++v) printf("%d:%s ", v, wrap[v] ? "wrap" : "??");
  printf("\n  -> field 0-3 取 a、4-7 取 b、8-15 回绕到 v&7（bit3 被忽略）\n");
  return bad == 0 ? 0 : 1;
}
