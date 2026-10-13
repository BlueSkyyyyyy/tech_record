// P157-1 (docs/71): weight-stationary (WS) UMMA instruction status probe.
//
// docs/62 (P148-1) established, against MPK's CUTLASS snapshot (`deps/cutlass`
// f3fde583), that the WS TMEM fragment layouts (`tmem_frg_ws` / `tmem_e_frg_ws`)
// are written but have no consumers, and that no `SM100_MMA_*_WS*` atom exists --
// only the `max_shift_ : 2` descriptor bit is reserved. This probe cross-checks
// that claim on a *second* checkout of the same CUTLASS (DeepGEMM's vendored
// `deep_gemm/include`), and pins the `max_shift_` descriptor encoding on the host.
//
// Host-side structural probe: every UMMA::tmem_*_frg::make is CUTE_HOST_DEVICE,
// so the layout algebra instantiates/evaluates on the host. No SM100 hardware is
// needed (and the target H100/sm90 does not have any).

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

// (A) The only WS-specific field in the UMMA instruction descriptor is
//     max_shift_ : 2 at bit [30,32). Pin its encoding on the host.
static void dump_max_shift() {
  hdr("(A) InstrDescriptor.max_shift_ : 2 at bit [30,32) (the only WS trace)");
  struct Row { const char *name; UMMA::MaxShift v; };
  const Row rows[] = {
      {"NoShift (0)", UMMA::MaxShift::NoShift},
      {"MaxShift8 (1)", UMMA::MaxShift::MaxShift8},
      {"MaxShift16 (2)", UMMA::MaxShift::MaxShift16},
      {"MaxShift32 (3)", UMMA::MaxShift::MaxShift32},
  };
  for (auto const &r : rows) {
    UMMA::InstrDescriptor d{};
    d.max_shift_ = uint8_t(r.v);
    uint32_t u = uint32_t(d);
    printf("  %-14s -> desc=0x%08x  bits[30,32)=%u\n", r.name, u,
           (u >> 30) & 0x3u);
  }
  UMMA::InstrDescriptor base{};
  printf("  default InstrDescriptor = 0x%08x (max_shift=0)\n", uint32_t(base));
}

// (B) WS C/D layout (tmem_frg_ws_1sm): 1x4 / 2x2 / 4x1 datapath over M.
static void dump_frg_ws() {
  hdr("(B) tmem_frg_ws: WS C/D layout (M=32/64/128, N=128)");
  {
    auto s = make_shape(make_shape(Int<128>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=128 (4x1)", s);
  }
  {
    auto s = make_shape(make_shape(Int<64>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=64 (2x2)", s);
  }
  {
    auto s = make_shape(make_shape(Int<32>{}, Int<128>{}), Int<1>{}, Int<1>{});
    show<UMMA::tmem_frg_ws_1sm<float, uint32_t>>("C/D ws M=32 (1x4)", s);
  }
}

// (C) WS sparse metadata E layout (tmem_e_frg_ws): same as tmem_e_frg at M=128,
//     gains an M sub-block dim at M=32/64.
static void dump_e_frg_ws() {
  hdr("(C) tmem_e_frg_ws: WS sparse metadata E (M=32/64/128)");
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
}

int main() {
  printf("WS UMMA instruction status probe -- DeepGEMM vendored CUTLASS "
         "(same commit f3fde583 as MPK), host-side, no SM100 needed\n");
  dump_max_shift();
  dump_frg_ws();
  dump_e_frg_ws();
  return 0;
}
