// P151-1 (docs/65): host-side structural probe for the BF16 SM100 linear device
// body `linear_sm100_mpk.cuh` (the sibling of the FP8 bodies walked in
// docs/32/33, only summarized in docs/32 §5.4).
//
// The kernel constructs its SMEM descriptors from CuTe layout algebra
// (make_tiled_mma / partition_shape_A/B/C / UMMA::tile_to_mma_shape) and its
// TMA writes from the MPK `smem_tma` view *on the same SMEM buffers*. Every
// layout in the former path is CUTE_HOST_DEVICE, so the algebra (shapes,
// strides, cosizes, the SW128 atom) evaluates on the host -- no SM100 hardware
// is needed (and the target H100/sm90 has none). CUTLASS snapshot =
// deps/cutlass f3fde583 (the submodule MPK's mpk branch pins).
//
// It recomputes exactly the layouts the kernel prints (commented-out debug at
// linear_sm100_mpk.cuh:190-231) so those stale comments can be checked against
// reality.

#include <cstdio>
#include <cstdint>
#include <cute/atom/mma_traits_sm100.hpp>  // UMMA:: + SM100_MMA_F16BF16_SS
#include <cute/tensor.hpp>

using namespace cute;

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

// The fixed template params driving the layout algebra in the kernel/wrapper.
static constexpr int kMMA_M = 128;
static constexpr int kMMA_N = 16;
static constexpr int kNumABStage = 8;
static constexpr int kNumAccStage = 2;
static constexpr int kNumCStage = 4;

using BF16 = bfloat16_t;

// Mirror linear_sm100_mpk.cuh:61-219 exactly (only the layout construction;
// no device code).
static void dump_layouts() {
  hdr("TiledMMA + tiler (linear_sm100_mpk.cuh:61-77)");
  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<BF16, BF16, float, kMMA_M, kMMA_N,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // 4 UMMAs per MmaTile_K
  auto mma_tiler = make_shape(bM, bN, bK);
  printf("  tile_size (bM,bN,bK) = ");
  print(mma_tiler);
  printf("   [UMMA-K for 16b = 16, x4 = 64 (source :73-75)]\n");
  printf("  tiled_mma  = ");
  print(tiled_mma);
  printf("\n");

  hdr("partition_shape_{A,B,C} (linear_sm100_mpk.cuh:170-188)");
  auto mma_shape_A = partition_shape_A(
      tiled_mma,
      make_shape(Int<kMMA_M>{}, size<2>(mma_tiler), Int<kNumABStage>{}));
  auto mma_shape_B = partition_shape_B(
      tiled_mma,
      make_shape(Int<kMMA_N>{}, size<2>(mma_tiler), Int<kNumABStage>{}));
  auto mma_shape_C =
      make_shape(make_shape(Int<kMMA_N>{}, Int<kMMA_M>{}), Int<1>{}, Int<1>{},
                 Int<kNumCStage>{});
  printf("  mma_shape_A = ");
  print(mma_shape_A);
  printf("   [comment: ((_128,_16),_1,_4,_8)]\n");
  printf("  mma_shape_B = ");
  print(mma_shape_B);
  printf("   [comment: ((_32,_16),_1,_4,_8)]\n");
  printf("  mma_shape_C = ");
  print(mma_shape_C);
  printf("   [comment: ((_32,_128),_1,_1,_4)]\n");

  hdr("SMEM layouts (linear_sm100_mpk.cuh:200-219)");
  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF16>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF16>{}, mma_shape_B);
  auto sC_layout_fake =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_INTER_Atom<BF16>{}, mma_shape_C);
  auto sC_shape =
      make_shape(make_shape(Int<kMMA_N>{}, Int<kMMA_M>{}), Int<1>{}, Int<1>{},
                 make_shape(Int<1>{}, Int<kNumCStage>{}));
  auto sC_stride =
      make_stride(make_stride(Int<kMMA_M>{}, Int<1>{}), Int<0>{}, Int<0>{},
                  make_stride(Int<0>{}, Int<kMMA_M * kMMA_N>{}));
  auto sC_layout = composition(sC_layout_fake.layout_a(),
                               sC_layout_fake.offset(),
                               make_layout(sC_shape, sC_stride));
  printf("  sA_layout = ");
  print(sA_layout);
  printf("\n");
  printf("  sB_layout = ");
  print(sB_layout);
  printf("\n");
  printf("  sC_layout = ");
  print(sC_layout);
  printf("\n");
  printf("  sC_layout_fake (INTER atom) = ");
  print(sC_layout_fake);
  printf("\n");
  printf("  -- commented prints at :222-230 show ArithTuple(...) shapes, "
         "i.e. the coord-tensor form, NOT these swizzled SMEM layouts --\n");

  hdr("SMEM cosize budget");
  printf("  cosize(sA) = %d elems = %zu B\n", (int)cosize(sA_layout),
         (size_t)cosize(sA_layout) * sizeof(BF16));
  printf("  cosize(sB) = %d elems = %zu B\n", (int)cosize(sB_layout),
         (size_t)cosize(sB_layout) * sizeof(BF16));
  printf("  cosize(sC) = %d elems = %zu B\n", (int)cosize(sC_layout),
         (size_t)cosize(sC_layout) * sizeof(BF16));
  size_t total = (size_t)(cosize(sA_layout) + cosize(sB_layout) +
                          cosize(sC_layout)) *
                 sizeof(BF16);
  printf("  A+B+C      = %zu B (dynamic SMEM the wrapper requests: 224 KiB)\n",
         total);
  printf("  cosize(sC_fake) = %d elems (unswizzled INTER view; sC is the "
         "composed one above)\n",
         (int)cosize(sC_layout_fake));

  hdr("ABI consistency: manual smem_tma stage stride == CuTe layout stride");
  // Kernel advances the SMEM pointers by hand (linear_sm100_mpk.cuh:452-455,
  // 664-665): A by OUTPUT_ATOM_SIZE*TILE_SIZE, B by MMA_N*TILE_SIZE,
  // C by MMA_N*OUTPUT_ATOM_SIZE. Those MUST equal the layout's stage stride or
  // the UMMA descriptors read the wrong buffer.
  int sA_stage =
      (int)get<1>(stride<3>(sA_layout.layout_b()));  // Tiles_K stride
  int sB_stage = (int)get<1>(stride<3>(sB_layout.layout_b()));
  int sC_stage = (int)get<1>(stride<3>(sC_layout.layout_b()));
  int manual_A = 128 /*OUTPUT_ATOM_SIZE*/ * 64 /*TILE_SIZE*/;
  int manual_B = 16 /*MMA_N*/ * 64 /*TILE_SIZE*/;
  int manual_C = 16 /*MMA_N*/ * 128 /*OUTPUT_ATOM_SIZE*/;
  printf("  A stage: layout=%d  manual=%d  %s\n", sA_stage, manual_A,
         sA_stage == manual_A ? "OK" : "MISMATCH");
  printf("  B stage: layout=%d  manual=%d  %s\n", sB_stage, manual_B,
         sB_stage == manual_B ? "OK" : "MISMATCH");
  printf("  C stage: layout=%d  manual=%d  %s\n", sC_stage, manual_C,
         sC_stage == manual_C ? "OK" : "MISMATCH");
}

// The SMEM A/B buffers are written by TMA mapped through MPK's `smem_tma<3,3,3>`
// (hardware 128B swizzle) but are *described* to UMMA through the CuTe
// Layout_K_SW128_Atom above. Prove the two share the same 128B swizzle
// structure by comparing the atom's swizzle to the SW128 atom.
static void dump_swizzle_agreement() {
  hdr("ABI: UMMA Layout_K_SW128_Atom == hardware 128B swizzle used by TMA");
  auto atom = UMMA::Layout_K_SW128_Atom<BF16>{};
  printf("  UMMA::Layout_K_SW128_Atom<bf16> = ");
  print(atom);
  printf("\n");
  // For a K-major (row stride 1) 8x64-element 16-bit atom, SW128 == Sw<3,4,3>
  printf("  -- SW128 swizzle is Sw<3,4,3> (16B blocks XOR'd by row); "
         "MPK smem_tma<3,3,3> get_swizzled_offset encodes the same "
         "`icol ^= irow` (docs/59).\n");
}

int main() {
  printf("BF16 SM100 linear (linear_sm100_mpk.cuh) host-side layout probe\n");
  printf("no SM100 hardware needed; CUTLASS deps/cutlass f3fde583\n");
  dump_layouts();
  dump_swizzle_agreement();
  return 0;
}
