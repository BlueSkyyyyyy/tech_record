// P152-1 (docs/66): host-side structural probe for the BF16 SM100 *MoE* linear
// device body `moe_linear_sm100.cuh` (827 lines) -- the MoE sibling of the
// dense BF16 body `linear_sm100_mpk.cuh` walked in P151-1/docs/65, and the
// BF16 sibling of the FP8 MoE group-GEMM `fp8_group_gemm_sm100.cuh`
// walked in P119-1/docs/33.
//
// Like the dense body, the kernel constructs its SMEM descriptors from CuTe
// layout algebra (make_tiled_mma / partition_shape_A/B/C / UMMA::tile_to_mma_shape)
// and its TMA writes from the MPK `smem_tma` view *on the same SMEM buffers*.
// Every layout in the former path is CUTE_HOST_DEVICE, so the algebra (shapes,
// strides, cosizes, the SW128 atom) evaluates on the host -- no SM100 hardware
// is needed (and the target H100/sm90 has none). CUTLASS snapshot =
// deps/cutlass f3fde583 (the submodule MPK's mpk branch pins).
//
// The MoE variant adds, relative to the dense body:
//   * an extra `sB_cp_layout` (the cp.async gather view) composed over sB; and
//   * `MoESharedStorage` (A + B + 5 barrier arrays + expert_mask + tmem ptr).
// Both are recomputed here. It also prints the cp.async copy-atom geometry
// (`copyB`) and the A TMA transaction bytes.

#include <cstdio>
#include <cstdint>
#include <cute/atom/mma_traits_sm100.hpp>  // UMMA:: + SM100_MMA_F16BF16_SS
#include <cute/tensor.hpp>

using namespace cute;

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

// The fixed template params driving the layout algebra in the kernel/wrapper
// (moe_linear_sm100.cuh:38-51; production hardcodes them, task_register.cc).
static constexpr int kMMA_M = 128;
static constexpr int kMMA_N = 16;
static constexpr int kNumABStage = 8;
static constexpr int kNumAccStage = 2;
static constexpr int kNumCStage = 4;
static constexpr int kTILE_SIZE = 64;         // :349
static constexpr int kWeightTile = 64;        // :350
static constexpr int kOutputAtom = 128;       // :351
static constexpr int kNumExperts = 128;       // test wrapper anchor

using BF16 = bfloat16_t;

static void dump_layouts() {
  hdr("TiledMMA + tiler (moe_linear_sm100.cuh:71-87)");
  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<BF16, BF16, float, kMMA_M, kMMA_N,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);
  printf("  tile_size (bM,bN,bK) = ");
  print(mma_tiler);
  printf("   [UMMA-K for 16b = 16, x4 = 64 (source :83-85)]\n");

  hdr("partition_shape_{A,B,C} (moe_linear_sm100.cuh:166-184)");
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
  printf("   [comment :190: ((_128,_16),_1,_4,_8)]\n");
  printf("  mma_shape_B = ");
  print(mma_shape_B);
  printf("   [comment :193: ((_32,_16),_1,_4,_8)]\n");
  printf("  mma_shape_C = ");
  print(mma_shape_C);
  printf("   [comment :196: ((_32,_128),_1,_1,_4)]\n");

  hdr("SMEM layouts (moe_linear_sm100.cuh:200-230)");
  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF16>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF16>{}, mma_shape_B);
  // sB_cp_layout: composition over sB's swizzle atom, laid out for cp.async.
  // The kernel builds it from shape(gB) modes 0/1 (16,64) and NUM_AB_STAGE
  // (:204-214). Here gB's (M,N) tile is (MMA_N, bK) = (16,64).
  auto sB_cp_layout = composition(
      sB_layout.layout_a(), sB_layout.offset(),
      make_layout(
          make_shape(Int<kMMA_N>{}, Int<kTILE_SIZE>{}, Int<1>{},
                     Int<kNumABStage>{}),
          make_stride(Int<kTILE_SIZE>{}, Int<1>{}, Int<0>{},
                      Int<kMMA_N * kTILE_SIZE>{})));
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
  printf("  sB_cp_layout = ");
  print(sB_cp_layout);
  printf("   [cp.async gather view; MoE-only, :204-214]\n");
  printf("  sC_layout = ");
  print(sC_layout);
  printf("\n");

  hdr("SMEM cosize budget (dense body vs MoE body)");
  size_t bA = (size_t)cosize(sA_layout) * sizeof(BF16);
  size_t bB = (size_t)cosize(sB_layout) * sizeof(BF16);
  printf("  cosize(sA) = %d elems = %zu B\n", (int)cosize(sA_layout), bA);
  printf("  cosize(sB) = %d elems = %zu B\n", (int)cosize(sB_layout), bB);
  printf("  cosize(sB_cp)(same buffer as sB) = %d elems\n",
         (int)cosize(sB_cp_layout));
  printf("  A+B = %zu B (dense body also sizes a C buffer; MoE reuses TMEM for "
         "C and scatters from regs)\n",
         bA + bB);
  // MoESharedStorage overhead (storage.cuh:141-152): 3*NUM_AB_STAGE + 2*NUM_ACC_STAGE
  // uint64 barriers (alignas 16 each), expert_mask[Num_Experts] uint32, tmem ptr.
  size_t barr = (size_t)(3 * kNumABStage + 2 * kNumAccStage) * sizeof(uint64_t);
  size_t mask = (size_t)kNumExperts * sizeof(uint32_t) + sizeof(uint32_t);
  printf("  barriers = %zu B, expert_mask+tmem = %zu B, layout B = %zu B\n",
         barr, mask, bA + bB);
  printf("  MoESharedStorage ~= %zu B (test wrapper requests 224 KiB dynamic "
         "SMEM; gap = conservative reserve, not derived in-source)\n",
         barr + mask + bA + bB);

  hdr("ABI consistency: manual smem_tma / cp.async stage stride == CuTe stride");
  printf("  weight_smem = smem_tma<T,3,3,3,OUTPUT_ATOM=%d,WEIGHT_TILE=%d,1> "
         "(:362-369)\n",
         kOutputAtom, kWeightTile);
  int sA_stage = (int)get<1>(stride<3>(sA_layout.layout_b()));
  int sB_stage = (int)get<1>(stride<3>(sB_layout.layout_b()));
  int manual_A = kOutputAtom * kTILE_SIZE;  // :528-530 weight_smem stage step
  int manual_B = kMMA_N * kTILE_SIZE;       // cp.async B stage step (sB_cp)
  printf("  A stage: layout=%d  manual(OUTPUT_ATOM*TILE)=%d  %s\n", sA_stage,
         manual_A, sA_stage == manual_A ? "OK" : "MISMATCH");
  printf("  B stage: layout=%d  manual(MMA_N*TILE)=%d  %s\n", sB_stage,
         manual_B, sB_stage == manual_B ? "OK" : "MISMATCH");
  // sB_cp_layout's Tiles_K stride should equal the ring stage step too.
  int sBcp_stage = (int)sB_cp_layout(Int<0>{}, Int<0>{}, Int<0>{}, Int<1>{});
  printf("  sB_cp Tiles_K stride (layout(0,0,0,1)) = %d  %s\n", sBcp_stage,
         sBcp_stage == manual_B ? "OK" : "MISMATCH");

  hdr("cp.async copy atom geometry (moe_linear_sm100.cuh:355/372-376)");
  constexpr int cp_group = 32 / kMMA_N;
  printf("  cp_async_group_size = 32/MMA_N = %d\n", cp_group);
  auto copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, BF16>{},
      Layout<Shape<Int<kMMA_N>, Int<cp_group>>,
             Stride<Int<cp_group>, _1>>{},
      Layout<Shape<_1, _8>>{});
  printf("  copyB = ");
  print(copyB);
  printf("\n");
  printf("  -> thread layout (MMA_N=%d, cp_group=%d) = %d threads = one warp "
         "(warp5); 16 B (uint128) per lane per k-tile; b_full barrier arrival "
         "count = 32 (one cpasync_barrier_arrive_noinc per lane, :555)\n",
         kMMA_N, cp_group, kMMA_N * cp_group);

  hdr("A TMA transaction bytes (moe_linear_sm100.cuh:341-342)");
  printf("  tma_transaction_bytes_A = sizeof(T)*size<0>(mma_tiler)*size<2>(mma_tiler)"
         " = 2*%d*%d = %d B\n",
         (int)size<0>(mma_tiler), (int)size<2>(mma_tiler),
         2 * (int)size<0>(mma_tiler) * (int)size<2>(mma_tiler));
}

int main() {
  printf("BF16 SM100 MoE linear (moe_linear_sm100.cuh) host-side layout probe\n");
  printf("no SM100 hardware needed; CUTLASS deps/cutlass f3fde583\n");
  dump_layouts();
  return 0;
}
