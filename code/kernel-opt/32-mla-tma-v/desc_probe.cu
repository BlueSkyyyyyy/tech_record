// 用 CUTLASS CuTe 生成 canonical GMMA 描述符，打印字段与布局，反推 MN-major。
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cstdio>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

template <class Layout, class Atom>
static void dump(const char* name, Layout layout, Atom atom) {
  auto t = make_tensor(make_smem_ptr((bf16*)nullptr), layout);
  auto d = cute::SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(t);
  printf("%s\n", name);
  printf("  strat=%u lbo=%u sbo=%u baseoff=%u layout=%u (raw=0x%llx)\n",
         d.bitfield.start_address_, d.bitfield.leading_byte_offset_,
         d.bitfield.stride_byte_offset_, d.bitfield.base_offset_,
         d.bitfield.layout_type_, (unsigned long long)d.desc_);
}

int main() {
  // MN-major SW128 atom，tile 到 Shape<N=256, K=64>
  {
    auto lay = tile_to_shape(SM90::GMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<_256, _16>{});
    printf("MN layout: "); print(lay); printf("\n");
    auto t = make_tensor(make_smem_ptr((bf16*)nullptr), lay);
    auto d = cute::SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(t);
    printf("  lbo=%u sbo=%u layout=%u raw=0x%llx\n", d.bitfield.leading_byte_offset_,
           d.bitfield.stride_byte_offset_, d.bitfield.layout_type_,
           (unsigned long long)d.desc_);
  }
  return 0;
}
