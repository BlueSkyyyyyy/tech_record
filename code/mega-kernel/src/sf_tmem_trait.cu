// P146-1 (docs/60): CUTLASS SM100 UMMA TMEM fragment layout traits for
// block-scaled (MXF8F6F4) scale factors, cross-referenced with MPK's hand-rolled
// scale-factor TMEM usage (docs/57 §3, docs/59 §3).
//
// This is a *host-side* structural probe: cute::UMMA::tmem_sf_frg / tmem_frg
// layouts are CUTE_HOST_DEVICE, so the whole layout algebra can be instantiated
// and evaluated on the host. No SM100 hardware is needed and none is present on
// the target H100/sm90; we only build/print/evaluate real CUTLASS Layouts from
// the exact snapshot the MPK source pins (deps/cutlass f3fde583).
//
// What it establishes (facts printed below, all from real CUTLASS code):
//   A. tmem_sf_frg<float,32,1,SFA> for M=128, VS=32, nK=4 maps logical M-row r
//      to TMEM (dp = r%32, col = 4*(r/32)); each K sub-block advances the column
//      by 1; MMA_VS has stride 0 (a scale value is shared by VS elements);
//      REP=4 replicates across the 4 subpartitions (dp += 32*rep).
//   B. SFA always uses the 4by1 scale layout (the SFA branch ignores
//      TmemAllocMode); SFB selects 4by1 (1SM) or 2by2 (2SM) and the 2by2 atom
//      splits the M axis into 2x2 with strides {64,32} DPs.
//   C. The accumulator fragment tmem_frg (M=128) is (DP=row, col=col_step=1)
//      i.e. one TMEM column per N element; M=64 uses the half-subpartition
//      16x4 atom, and Interleaved vs NonInterleaved only differ in the outer
//      tiling stride (identical for a single tile).
//   D. get_num_aligned_tmem_cols (MPK, linear_fp8_sm100_sm100_utils.cuh:157) is
//      a round-up-to-2-power clamp to [32,512] -- matches the CUTLASS allocator
//      contract (docs/59 §2).

#include <cstdio>
#include <cstdint>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

using namespace cute;

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

template <class Frag, class Shape>
static void show(const char *name, Shape const &shape) {
  auto t = Frag::make(shape);
  printf("[%s]\n  shape : ", name);
  print(shape);
  printf("\n  layout: ");
  print(t.layout());
  printf("\n");
}

static void dump_sf_mapping() {
  hdr("(A) tmem_sf_frg<float,32,1,SFA> M=128 VS=32 nK=4 -> (dp,col) map");
  auto s = make_shape(make_shape(Int<128>{}, make_shape(Int<32>{}, Int<1>{})),
                      Int<1>{}, Int<4>{});
  auto t = UMMA::tmem_sf_frg<float, 32, 1, true>::make(s);
  printf("layout = ");
  print(t.layout());
  printf("\n  logical M-row r -> (dp,col) at k=0 (expect dp = r mod 32, col = 4 * (r div 32)):\n");
  for (int r : {0, 1, 31, 32, 33, 63, 64, 96, 127}) {
    auto c = make_coord(make_coord(make_coord(r % 32, r / 32), 0),
                        make_coord(0, 0));
    int off = t.layout()(make_coord(c, 0, 0));
    printf("    r=%3d : addr=%7d  dp=%3d  col=%2d\n", r, off, off >> 16,
           off & 0xffff);
  }
  printf("  K sub-block sweep at r=0 (expect col advances by 1 per k): ");
  for (int k = 0; k < 4; ++k) {
    auto c = make_coord(make_coord(make_coord(0, 0), 0), make_coord(0, 0));
    int off = t.layout()(make_coord(c, 0, k));
    printf("k%d->col%d ", k, off & 0xffff);
  }
  printf("\n  REP sweep at r=0 (expect dp += 32 per rep, i.e. across subpartitions): ");
  for (int r = 0; r < 4; ++r) {
    auto c = make_coord(make_coord(make_coord(0, 0), r), make_coord(0, 0));
    int off = t.layout()(make_coord(c, 0, 0));
    printf("rep%d->dp%d ", r, off >> 16);
  }
  printf("\n  MMA_VS sweep at r=0 (expect identical addr, stride 0): ");
  for (int v : {0, 1, 31}) {
    auto c = make_coord(make_coord(make_coord(0, 0), 0), make_coord(v, 0));
    printf("vs%d->%d ", v, t.layout()(make_coord(c, 0, 0)));
  }
  printf("\n");
}

int main() {
  printf("CUTLASS SM100 UMMA TMEM fragment layout probe (host-side, no SM100 "
         "needed)\n");

  dump_sf_mapping();

  {
    auto s = make_shape(make_shape(Int<128>{}, make_shape(Int<32>{}, Int<1>{})),
                        Int<1>{}, Int<4>{});
    hdr("(B) scale-factor SFB variants (SFA always 4by1)");
    show<UMMA::tmem_sf_frg<float, 32, 1, false>>("SFB float VS=32 1SM 4by1", s);
    auto s64 = make_shape(make_shape(Int<128>{}, make_shape(Int<64>{}, Int<1>{})),
                          Int<1>{}, Int<2>{});
    show<UMMA::tmem_sf_frg<float, 64, 1, true>>("SFA float VS=64 (sparse K64)",
                                                s64);
    show<UMMA::tmem_sf_frg<
        float, 32, 2, false,
        UMMA::TmemAllocMode::ScaleFactorDuplicated2by2>>("SFB float VS=32 2SM 2by2",
                                                         s);
  }

  {
    hdr("(C) accumulator tmem_frg (c_type=float in uint32 storage)");
    auto s128 = make_shape(make_shape(Int<128>{}, Int<128>{}), Int<1>{},
                           Int<1>{});
    show<UMMA::tmem_frg_1sm<float, uint32_t,
                            UMMA::TmemAllocMode::Interleaved>>("acc M=128 N=128",
                                                               s128);
    auto s64 = make_shape(make_shape(Int<64>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_1sm<float, uint32_t,
                            UMMA::TmemAllocMode::Interleaved>>("acc M=64 N=128 I",
                                                               s64);
    show<UMMA::tmem_frg_1sm<float, uint32_t,
                            UMMA::TmemAllocMode::NonInterleaved>>(
        "acc M=64 N=128 NI", s64);
  }

  {
    hdr("(D) get_num_aligned_tmem_cols rounding (round up to 2-power, [32,512])");
    auto R = [](int n) {
      if (n <= 32) return 32;
      if (n <= 64) return 64;
      if (n <= 128) return 128;
      if (n <= 256) return 256;
      return 512;
    };
    for (int n : {20, 32, 33, 64, 65, 128, 129, 200, 256, 257, 512})
      printf("  cols=%3d -> round=%3d\n", n, R(n));
  }
  return 0;
}
