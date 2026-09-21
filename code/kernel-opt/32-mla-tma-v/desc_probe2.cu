// 反推 MN-major 描述符字段：直接构造满足 cute canonical 检查的 u128 布局并打印描述符。
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cstdio>

using namespace cute;
using u128 = uint128_t;

template <class L>
static void dump(const char* name, L l) {
  auto t = make_tensor(make_smem_ptr((u128*)nullptr), l);
  auto d = SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(t);
  printf("%-28s lbo=%u sbo=%u layout=%u raw=0x%llx\n", name,
         d.bitfield.leading_byte_offset_, d.bitfield.stride_byte_offset_,
         d.bitfield.layout_type_, (unsigned long long)d.desc_);
}

int main() {
  // canonical MN u128 布局：N 连续 8 个 u128，K 每 8 行，n-group / k-group 分别给不同 stride
  // 约定 ((8,n),(8,k)) : ((1,LBO),(8,SBO))  —— 这里直接以 u128 为单位
  dump("ngroup=64,kgroup=32", Layout<Shape<Shape<_8,_4>, Shape<_8,_2>>,
                                   Stride<Stride<_1,_64>, Stride<_8,_32>>>{});
  dump("ngroup=32,kgroup=64", Layout<Shape<Shape<_8,_4>, Shape<_8,_2>>,
                                   Stride<Stride<_1,_32>, Stride<_8,_64>>>{});
  dump("ngroup=16,kgroup=16", Layout<Shape<Shape<_8,_4>, Shape<_8,_2>>,
                                   Stride<Stride<_1,_16>, Stride<_8,_16>>>{});
  // 8x8 全 1024B / 4096B（对应我们 V tile N=256 的候选）
  dump("ngroup=64,kgroup=256", Layout<Shape<Shape<_8,_4>, Shape<_8,_2>>,
                                    Stride<Stride<_1,_64>, Stride<_8,_256>>>{});
  return 0;
}
