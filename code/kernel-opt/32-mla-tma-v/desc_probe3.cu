// 打印 cute canonical MN-major SW128 布局的 (n,k) -> 元素偏移，与手写公式对比。
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;
using bf16 = cutlass::bfloat16_t;

// 手写公式（元素单位）
static int mn_off(int k, int n, int N, int K) {
  const int ch = ((n & 63) >> 3) ^ (k & 7);
  return (k >> 3) * (N / 64) * 64 + (n >> 6) * 64 + (k & 7) * 64 + ch * 8 + (n & 7);
}

int main() {
  constexpr int N = 256, K = 16;
  auto lay = tile_to_shape(SM90::GMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<_256, _16>{});
  printf("cute layout: "); print(lay); printf("\n");
  int bad = 0;
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) {
      int c = (int)lay(make_coord(n, k));
      int m = mn_off(k, n, N, K);
      if (c != m) {
        if (bad < 20) printf("  (n=%d,k=%d) cute=%d mine=%d\n", n, k, c, m);
        ++bad;
      }
    }
  printf("mismatches = %d / %d\n", bad, N * K);
  // 打印前几个
  for (int n = 0; n < 8; ++n) printf("n=%d k=0: cute=%d mine=%d\n", n, (int)lay(make_coord(n, 0)), mn_off(0, n, N, K));
  return 0;
}
