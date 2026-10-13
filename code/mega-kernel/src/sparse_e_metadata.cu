// P156-1 (docs/70): the *content* of the structured-sparse metadata "E" and the
// A-slot permutation it carries, completing docs/62 (P148-1, only the TMEM E
// trait shape) and docs/63 (P149-1, only the A-operand *type/SMEM layout*).
//
// docs/62 §6 and docs/63 §7 both explicitly left one thing unopened:
//   - "the hardware encoding of E (2:4 index bit order, per-atom bit packing)"
//   - "which elements are dropped, and how E's index selects even/odd A slots"
//
// This probe answers it from CUTLASS source, with a *host-side replay* of the
// compressor arithmetic (every value below is either a CUTE_HOST_DEVICE
// constexpr or the exact switch/loop of SM90StructuredSparseCompressor):
//   (i)   Sm1xxGemmSparseConfig constants per dtype (chunk size, E bits)   [SM100]
//   (ii)  Sm90GemmSparseConfig constants + E atom (16/32/8-bit)            [SM90]
//   (iii) the gmem E layout atom shapes/strides (SM100 F16/TF32/F8/F4)
//   (iv)  deduce_layoutA / deduce_layoutE (compressed-A layout vs E layout)
//   (v)   the E encoder replay: MetadataOneChunk2to4 (2:4) + MetadataOneChunk1to2
//         (TF32) + the single-nonzero corner cases
//   (vi)  a worked row: dense A -> compressed A order + one metadata nibble
//
// All CUTLASS headers are CUTE_HOST_DEVICE, so the whole thing evaluates on the
// host; the SM90/SM100 tensor cores themselves are unavailable on the target card
// (H100/sm90). CUTLASS snapshot = deps/cutlass f3fde583 (pinned by MPK's mpk branch).

#include <cstdio>
#include <cstdint>

#include <cutlass/gemm/collective/builders/sm1xx_sparse_config.inl>  // Sm1xxGemmSparseConfig (SM100)
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>   // Sm90GemmSparseConfig (SM90)
#include <cutlass/transform/kernel/sparse_gemm_compressor.hpp>       // StructuredSparseCompressorUtility
#include <cute/tensor.hpp>

using namespace cute;
using namespace cutlass;

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

// ---- helpers (function templates; C++17) ----------------------------------
template <class AMma, class EMma>
static void row_sm1xx(const char *name) {
  using Cfg = Sm1xxGemmSparseConfig<AMma, cutlass::layout::RowMajor, EMma>;
  using lAPC = typename Cfg::LogicalElemsAPerChunk;
  using pAPC = typename Cfg::PhysicalElemsAPerChunk;
  using ebc = typename Cfg::ElementEBitsPerChunk;
  using ebe = typename Cfg::ElementEBitsPerElementAMma;
  using eam = typename Cfg::TensorEAtomM;
  using eak = typename Cfg::TensorEAtomK;
  using ealk = typename Cfg::TensorEAlignmentK;
  printf("  %-8s %3d | %16d | %17d | %11d | %10d | %6d | %6d | %7d\n", name,
         (int)Cfg::IsKMajor, (int)lAPC{}, (int)pAPC{}, (int)ebc{}, (int)ebe{},
         (int)eam{}, (int)eak{}, (int)ealk{});
}

template <class AMma, class EMma>
static void row_sm90(const char *name) {
  using Cfg = Sm90GemmSparseConfig<AMma, GMMA::Major::K, EMma, Int<128>>;
  using lAPC = typename Cfg::LogicalElemsAPerChunk;
  using pAPC = typename Cfg::PhysicalElemsAPerChunk;
  using ebc = typename Cfg::ElementEBitsPerChunk;
  using ebe = typename Cfg::ElementEBitsPerElementAMma;
  using bk = typename Cfg::BlockK;
  using nk = typename Cfg::NumK;
  using eam = typename Cfg::TensorEAtomM;
  using eak = typename Cfg::TensorEAtomK;
  using EAtom = typename Cfg::TensorEAtom;
  printf("  %-6s lAPerChunk=%d pAPerChunk=%d EBits/chunk=%d EBits/elem=%d "
         "BlockK=%d NumK=%d EAtomM=%d EAtomK=%d\n",
         name, (int)lAPC{}, (int)pAPC{}, (int)ebc{}, (int)ebe{},
         (int)bk{}, (int)nk{}, (int)eam{}, (int)eak{});
  printf("          EAtom = ");
  print(EAtom{});
  printf("\n");
}

template <class AMma, class EMma>
static void atom_sm1xx(const char *name) {
  using Cfg = Sm1xxGemmSparseConfig<AMma, cutlass::layout::RowMajor, EMma>;
  using EAtom = typename Cfg::TensorEAtom;
  printf("  %-6s EAtom = ", name);
  print(EAtom{});
  printf("\n");
  printf("         cosize(EAtom)=%d (E elems) ; raw E elem = %d bits\n",
         (int)cosize(EAtom{}), (int)sizeof_bits_v<typename Cfg::ElementEMmaRaw>);
}

// ---------------------------------------------------------------------------
// (i) SM100 config constants
// ---------------------------------------------------------------------------
static void dump_sm1xx() {
  hdr("(i) SM100 Sm1xxGemmSparseConfig: chunk sizes and E bit budget");
  printf("  dtype    IsK | LogicalAPerChunk | PhysicalAPerChunk | EBits/chunk | EBits/Aelem |"
         " EAtomM | EAtomK | EAlignK\n");
  row_sm1xx<sparse_elem<2, half_t>, sparse_elem<8, uint8_t>>("F16");
  row_sm1xx<sparse_elem<2, bfloat16_t>, sparse_elem<8, uint8_t>>("BF16");
  row_sm1xx<sparse_elem<2, tfloat32_t>, sparse_elem<4, uint8_t>>("TF32");
  row_sm1xx<sparse_elem<2, float_e4m3_t>, sparse_elem<8, uint8_t>>("F8");
  row_sm1xx<sparse_elem<2, int8_t>, sparse_elem<8, uint8_t>>("I8");
  row_sm1xx<sparse_elem<4, uint8_t>, sparse_elem<16, uint8_t>>("F4");
  printf("  note: LogicalAPerChunk = logical A elems per 2:4 chunk (F4=8 nibbles, TF32=2, else 4);\n");
  printf("        PhysicalAPerChunk = Logical/2; EBits/Aelem = 2 (4-bit pos code per kept elem)\n");
  printf("        except TF32 = 4 (a 4-bit code per chunk of 2 logical elems -- see (v)).\n");
}

// ---------------------------------------------------------------------------
// (ii) SM90 config constants + E atom
// ---------------------------------------------------------------------------
static void dump_sm90() {
  hdr("(ii) SM90 Sm90GemmSparseConfig: constants + gmem E atom");
  row_sm90<sparse_elem<2, half_t>, sparse_elem<8, uint8_t>>("16bit");
  row_sm90<sparse_elem<2, tfloat32_t>, sparse_elem<4, uint8_t>>("32bit");
  row_sm90<sparse_elem<2, int8_t>, sparse_elem<8, uint8_t>>("8bit");
  printf("  source cites the PTX def:\n");
  printf("    docs.nvidia.com/cuda/parallel-thread-execution/...#sparse-wgmma-metadata-64n32-f16bf16\n");
}

// ---------------------------------------------------------------------------
// (iii) SM100 gmem E layout atoms
// ---------------------------------------------------------------------------
static void dump_atoms() {
  hdr("(iii) SM100 gmem E layout atom (TensorEAtom), in units of E logical elems");
  atom_sm1xx<sparse_elem<2, half_t>, sparse_elem<8, uint8_t>>("F16");
  atom_sm1xx<sparse_elem<2, tfloat32_t>, sparse_elem<4, uint8_t>>("TF32");
  atom_sm1xx<sparse_elem<2, float_e4m3_t>, sparse_elem<8, uint8_t>>("F8");
  atom_sm1xx<sparse_elem<4, uint8_t>, sparse_elem<16, uint8_t>>("F4");
}

// ---------------------------------------------------------------------------
// (iv) compressed-A layout vs E layout
// ---------------------------------------------------------------------------
static void dump_layouts() {
  hdr("(iv) deduce_layoutA / deduce_layoutE (SM100), concretized with M=64,N=1,K=128,L=1");
  {
    using Cfg = Sm1xxGemmSparseConfig<sparse_elem<2, half_t>, cutlass::layout::RowMajor, sparse_elem<8, uint8_t>>;
    auto la = Cfg::fill_layoutA(cute::make_tuple(64, 1, 128, 1));
    auto le = Cfg::fill_layoutE(cute::make_tuple(64, 1, 128, 1));
    printf("  F16 K-major  layoutA = "); print(la); printf("\n");
    printf("  F16 K-major  layoutE = "); print(le); printf("\n");
  }
  {
    using Cfg = Sm1xxGemmSparseConfig<sparse_elem<2, half_t>, cutlass::layout::ColumnMajor, sparse_elem<8, uint8_t>>;
    auto la = Cfg::fill_layoutA(cute::make_tuple(64, 1, 128, 1));
    printf("  F16 MN-major layoutA = "); print(la); printf("\n");
    printf("      (K axis = (2, K/2): physical pair-of-kept-slots x chunk index; A is half as wide)\n");
  }
}

// ---------------------------------------------------------------------------
// (ivb) byte accounting: compressed A really is half, plus E size
// ---------------------------------------------------------------------------
static void dump_bytes() {
  hdr("(ivb) byte accounting (StructuredSparseCompressorUtility, M=64 N=1 K=128 L=1)");
  using SC = Sm1xxGemmSparseConfig<sparse_elem<2, half_t>, cutlass::layout::RowMajor, sparse_elem<8, uint8_t>>;
  using Util = cutlass::transform::kernel::StructuredSparseCompressorUtility<
      cute::tuple<int, int, int, int>, half_t, cutlass::layout::RowMajor, SC>;
  Util u(cute::make_tuple(64, 1, 128, 1),
         cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>{});
  printf("  A logical: M=64 K=128 ; physical: m=%d k=%d\n",
         u.get_tensorA_m_physical(), u.get_tensorA_k_physical());
  printf("  A bytes: raw=%llu  compressed=%llu  (ratio %.2f)\n",
         (unsigned long long)u.get_raw_tensor_A_bytes(),
         (unsigned long long)u.get_compressed_tensor_A_bytes(),
         (double)u.get_compressed_tensor_A_bytes() / (double)u.get_raw_tensor_A_bytes());
  printf("  E bytes=%llu  (metadata M=%d K=%d)\n",
         (unsigned long long)u.get_tensor_E_bytes(),
         u.get_metadata_m_physical(), u.get_metadata_k_physical());
}

// ---------------------------------------------------------------------------
// (v) the E encoder replay
// ---------------------------------------------------------------------------
// Faithful replay of SM90StructuredSparseCompressor::MetadataOneChunk2to4
// (sm90_sparse_gemm_compressor.hpp:255-286): code == index of the kept elem,
// 2 bits, physical-slot 0 -> bits[0:2], slot 1 -> bits[2:4].
static int code_2to4(int elt_log_idx) {
  switch (elt_log_idx) { case 0: return 0b00; case 1: return 0b01;
                         case 2: return 0b10; case 3: return 0b11; }
  return 0;
}
// Faithful replay of MetadataOneChunk1to2 (op.cit.:226-252): 4-bit code, slot 0.
static int code_1to2(int elt_log_idx) {
  switch (elt_log_idx) { case 0: return 0b0100; case 1: return 0b1110; }
  return 0;
}

// Replay structure_sparse_compress's per-chunk body (:395-455) on a 4-elem chunk.
static void compress_chunk_2to4(const int a[4], bool *keep, int pos[2], uint8_t *nib) {
  int nz[2] = {0, 0};
  int n = 0;
  // tACsAC[] : kept values, in encounter order
  for (int i = 0; i < 4; ++i)
    if (a[i] != 0) { nz[n] = i; keep[n] = true; ++n; }
  // corner case (sizeof<32): single nonzero
  if (n == 1 && nz[0] == 3) { keep[0] = false; keep[1] = true; nz[0] = 0; nz[1] = 3; }
  else if (n == 1)          { nz[1] = 3; }
  pos[0] = nz[0]; pos[1] = nz[1];
  uint8_t s = 0;
  s |= (uint8_t)(code_2to4(nz[0]) << (2 * 0));
  s |= (uint8_t)(code_2to4(nz[1]) << (2 * 1));
  *nib = s;
}

static void dump_encoder() {
  hdr("(v) E encoder replay: SM90StructuredSparseCompressor (2:4 -> one nibble)");
  printf("  MetadataOneChunk2to4: code(elt_log_idx) = elt_log_idx (2 bits), packed [pos1:pos0]\n");
  printf("  dense chunk [a b c d]   kept(logical pos)   char  nibble  (binary)\n");
  const char *name[4] = {"a", "b", "c", "d"};
  // all C(4,2)=6 two-nonzero patterns
  int pats[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  for (int p = 0; p < 6; ++p) {
    int a[4] = {10,20,30,40};
    for (int i = 0; i < 4; ++i) if (i != pats[p][0] && i != pats[p][1]) a[i] = 0;
    bool keep[2] = {false, false}; int pos[2]; uint8_t nib;
    compress_chunk_2to4(a, keep, pos, &nib);
    printf("  [%2d %2d %2d %2d]  {%d,%d}         %s%s  0x%X  0b%c%c%c%c\n",
           a[0], a[1], a[2], a[3], pos[0], pos[1],
           keep[0] ? name[pats[p][0]] : "_", keep[1] ? name[pats[p][1]] : "_",
           nib,
           (nib & 8) ? '1' : '0', (nib & 4) ? '1' : '0',
           (nib & 2) ? '1' : '0', (nib & 1) ? '1' : '0');
  }
  printf("  single-nonzero corner cases (sizeof<32):\n");
  for (int only = 0; only < 4; ++only) {
    int a[4] = {0,0,0,0}; a[only] = 10;
    bool keep[2] = {false, false}; int pos[2]; uint8_t nib;
    compress_chunk_2to4(a, keep, pos, &nib);
    printf("  only %s at pos %d: [%2d %2d %2d %2d] -> recorded pos {%d,%d}  nibble 0x%X\n",
           name[only], only, a[0], a[1], a[2], a[3], pos[0], pos[1], nib);
  }
  printf("  MetadataOneChunk1to2 (TF32: chunk = 2 logical elems, 1 physical, 4-bit code):\n");
  for (int only = 0; only < 2; ++only) {
    int pos = only;
    int c = code_1to2(pos);
    printf("  chunk [%s %s] -> keep logical %d -> code 0b%c%c%c%c (0x%X)\n",
           only == 0 ? "x" : "0", only == 1 ? "x" : "0", pos,
           (c & 8) ? '1' : '0', (c & 4) ? '1' : '0',
           (c & 2) ? '1' : '0', (c & 1) ? '1' : '0', c);
  }
  printf("     (4-bit code; its exact hardware meaning is not derivable from source -- only-noted;\n");
  printf("      TF32 is 32-bit so the single-nonzero corner case above does not apply to it.)\n");
}

// ---------------------------------------------------------------------------
// (vi) a worked row
// ---------------------------------------------------------------------------
static void dump_worked() {
  hdr("(vi) worked row: dense A row (2:4) -> compressed A order + E nibble");
  const int a[8] = {7, 0, 0, 9,  0, 5, 0, 0};  // two chunks of 4
  printf("  dense A row (K=8) = [7 0 0 9 | 0 5 0 0]\n");
  for (int c = 0; c < 2; ++c) {
    bool keep[2] = {false, false}; int pos[2]; uint8_t nib;
    compress_chunk_2to4(&a[c*4], keep, pos, &nib);
    printf("    chunk %d: [%d %d %d %d] -> compressed slots [%s %s]  E nibble 0x%X  (pos {%d,%d})\n",
           c, a[c*4], a[c*4+1], a[c*4+2], a[c*4+3],
           keep[0] ? "x" : "0", keep[1] ? "x" : "0", nib, pos[0], pos[1]);
  }
  printf("  => physical A is half as wide; E holds one 4-bit nibble per chunk.\n");
}

int main() {
  printf("CUTLASS structured-sparse E metadata probe (host-side, no SM90/SM100 tensor core)\n");
  printf("completes docs/62 (E trait shape) and docs/63 (A operand type/layout)\n");
  dump_sm1xx();
  dump_sm90();
  dump_atoms();
  dump_layouts();
  dump_bytes();
  dump_encoder();
  dump_worked();
  return 0;
}
