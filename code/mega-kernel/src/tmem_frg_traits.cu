// P148-1 (docs/62): the *rest* of the CUTLASS SM100 UMMA TMEM fragment layout
// trait family in mma_traits_sm100.hpp --
//   (i)  tmem_e_frg / tmem_e_frg_ws : the sparse-MMA metadata "E" layout, and
//   (ii) tmem_frg_ws               : the weight-stationary C/D layout.
// P146-1 (docs/60) covered the accumulator tmem_frg and the block-scaled
// scale-factor tmem_sf_frg; this probe completes the family and audits which of
// these traits actually have consumers.
//
// Host-side structural probe: every UMMA::tmem_*_frg::make is CUTE_HOST_DEVICE,
// so the layout algebra instantiates/evaluates on the host. No SM100 hardware is
// needed (and the target H100/sm90 does not have any). We use the exact CUTLASS
// snapshot MPK pins: deps/cutlass f3fde583.

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

// ValueType must carry the right sizeof_bits; the E metadata atom dims are tuned
// to the A/B element width (TF32=32b -> 128x16, FP16=16b -> 128x32, 8b -> 128x64,
// F4 -> 128x128).
static void dump_e_frg() {
  hdr("(A) tmem_e_frg: sparse metadata E layout vs A/B element width");

  {
    auto s = make_shape(make_shape(Int<128>{}, Int<16>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg<float>>("E tf32 (32b, 128x16 atom)", s);
  }
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<32>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg<half_t>>("E fp16 (16b, 128x32 atom)", s);
  }
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<64>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg<uint8_t>>("E int8/f8 (8b, 128x64 atom)", s);
  }
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg<float_e2m1_t>>("E f4 (4b, 128x128 atom)", s);
  }

  // fp16 E atom has shape ((8,2,8),(16,2)); decompose the logical (r,n) into the
  // atom's nested coordinates: r=(r%8,(r/8)%2,r/16), n=(n%16,n/16).
  auto s = make_shape(make_shape(Int<128>{}, Int<32>{}), Int<1>{}, Int<1>{});
  auto t = UMMA::tmem_e_frg<half_t>::make(s);
  printf("\n  fp16 E: logical (M-row r, N-col n) -> layout offset (DP_b=1<<21 bit "
         "addressing):\n");
  for (int r : {0, 1, 7, 8, 15, 16, 127}) {
    auto c = make_coord(make_coord(make_coord(r % 8, (r / 8) % 2, r / 16),
                                   make_coord(0, 0)),
                        0, 0);
    int off = t.layout()(c);
    printf("    r=%3d n=0 : off=%9d  (off>>21=%d dp-of-2^21)\n", r, off,
           off >> 21);
  }
  for (int n : {0, 1, 15, 16, 31}) {
    auto c = make_coord(make_coord(make_coord(0, 0, 0), make_coord(n % 16, n / 16)),
                        0, 0);
    int off = t.layout()(c);
    printf("    r=0 n=%2d : off=%9d\n", n, off);
  }
}

static void dump_e_frg_ws() {
  hdr("(B) tmem_e_frg_ws: weight-stationary sparse metadata (M=32/64/128)");
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<32>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg_ws<half_t>>("E_ws fp16 M=128", s);
  }
  {
    auto s = make_shape(make_shape(Int<64>{}, Int<32>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg_ws<half_t>>("E_ws fp16 M=64", s);
  }
  {
    auto s = make_shape(make_shape(Int<32>{}, Int<32>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg_ws<half_t>>("E_ws fp16 M=32", s);
  }
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<16>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_e_frg_ws<float>>("E_ws tf32 M=128", s);
  }
}

static void dump_frg_ws() {
  hdr("(C) tmem_frg_ws: weight-stationary C/D layout (M=32/64/128)");
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=128 N=128 (4x1)",
                                                 s);
  }
  {
    auto s = make_shape(make_shape(Int<64>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=64 N=128 (2x2)", s);
  }
  {
    auto s = make_shape(make_shape(Int<32>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=32 N=128 (1x4)", s);
  }
  {
    auto s = make_shape(make_shape(Int<32>{}, Int<256>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=32 N=256 (1x4)", s);
  }

  // ws M=64 shape (64,(64,2)): the 2nd N-half lands 64 DPs away (2x2 datapath).
  {
    auto s = make_shape(make_shape(Int<64>{}, Int<128>{}), Int<1>{}, Int<1>{});
    auto t = UMMA::tmem_frg_ws_1sm<float, uint32_t>::make(s);
    printf("\n  ws M=64 (2x2): (r, n) -> (dp, col):\n");
    for (int r : {0, 1, 31, 32, 63}) {
      auto c = make_coord(make_coord(r, make_coord(0, 0)), 0, 0);
      int off = t.layout()(c);
      printf("    r=%2d n=0   : dp=%3d col=%3d\n", r, off >> 16, off & 0xffff);
    }
    for (int n : {0, 1, 63, 64, 65, 127}) {
      auto c = make_coord(make_coord(0, make_coord(n % 64, n / 64)), 0, 0);
      int off = t.layout()(c);
      printf("    r=0 n=%3d  : dp=%3d col=%3d\n", n, off >> 16, off & 0xffff);
    }
  }
  // ws M=32 shape (32,(32,4)): 1x4 datapath -> 4 N-quarters each 32 DPs apart.
  {
    auto s = make_shape(make_shape(Int<32>{}, Int<128>{}), Int<1>{}, Int<1>{});
    auto t = UMMA::tmem_frg_ws_1sm<float, uint32_t>::make(s);
    printf("\n  ws M=32 (1x4): (r, n) -> (dp, col):\n");
    for (int n : {0, 1, 31, 32, 33, 63, 64, 96, 127}) {
      auto c = make_coord(make_coord(0, make_coord(n % 32, n / 32)), 0, 0);
      int off = t.layout()(c);
      printf("    r=0 n=%3d  : dp=%3d col=%3d\n", n, off >> 16, off & 0xffff);
    }
  }
}

int main() {
  printf("CUTLASS SM100 UMMA TMEM fragment trait probe pt.2 -- sparse E and "
         "weight-stationary C/D (host-side, no SM100 needed)\n");
  dump_e_frg();
  dump_e_frg_ws();
  dump_frg_ws();
  return 0;
}
