// P149-1 (docs/63): the *A-operand* half of the CUTLASS SM100 sparse UMMA path,
// completing docs/62 (P148-1), which only walked the *metadata E* half.
//
// docs/62 covered the sparse metadata "E" trait (tmem_e_frg / tmem_e_frg_ws);
// this probe walks the *other* operand of tcgen05.mma.sp:
//   (i)   sparse_elem<2,T> / sizeof_bits (the "one physical elem == S logical elems")
//   (ii)  sparse_ptr + make_sparse_ptr (physical indexing contract)
//   (iii) the sparse SMEM layout atoms (Layout_*_SpAtom + smem_sparse_ptr_flag_bits)
//   (iv)  UMMA::sparse_smem_desc (bit-identical to the dense smem_desc; only the
//         interface validates is_sparse / is_sparse_ptr)
//   (v)   the instruction-descriptor sparse_flag_ bit and the K = 512/bits rule
//
// Host-side structural probe: every cute::Layout and the InstrDescriptor are
// CUTE_HOST_DEVICE, so the algebra evaluates on the host. No SM100 hardware is
// needed (and the target H100/sm90 has none). CUTLASS snapshot = deps/cutlass
// f3fde583 (the submodule MPK's mpk branch pins).

#include <cstdio>
#include <cstdint>
#include <cute/atom/mma_traits_sm100.hpp>  // UMMA:: + pulls in sparse layouts
#include <cute/tensor.hpp>

using namespace cute;

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

// ---- (i) sparse_elem<2,T> and its effective sizeof_bits ------------------
template <class Raw, int S>
static void show_sparse_elem(const char *name) {
  using SE = sparse_elem<S, Raw>;
  constexpr auto bits = sizeof_bits_v<SE>;  // == ratio(sizeof_bits<Raw>, S)
  printf("  %-28s raw=%2d bits, S=%d -> sizeof_bits(sparse_elem)=%d "
         "(is_sparse=%d)\n",
         name, (int)sizeof_bits_v<Raw>, S, (int)bits, (int)is_sparse_v<SE>);
  (void)bits;
}

static void dump_sparse_elem() {
  hdr("(i) sparse_elem<S,T>: one physical elem represents S logical elems");
  show_sparse_elem<half_t, 2>("sparse_elem<2,half_t> (fp16 A)");
  show_sparse_elem<float, 2>("sparse_elem<2,float>  (tf32 A)");
  show_sparse_elem<uint8_t, 2>("sparse_elem<2,uint8_t> (i8/f8 A)");
  // E metadata is *also* carried as sparse_elem, but the sparsity there is the
  // E atom packing (docs/62 §2.1), not the A 2:4 factor:
  show_sparse_elem<uint8_t, 4>("sparse_elem<4,uint8_t> (E tf32)");
  show_sparse_elem<uint8_t, 8>("sparse_elem<8,uint8_t> (E fp16)");
  printf("  -> dense half_t sizeof_bits=%d; dense float sizeof_bits=%d "
         "(contrast)\n",
         (int)sizeof_bits_v<half_t>, (int)sizeof_bits_v<float>);
}

// ---- (ii) sparse_ptr physical-indexing contract -------------------------
static void dump_sparse_ptr() {
  hdr("(ii) sparse_ptr<S,Iter>: op+ rounds to a multiple of S, op[] derefs i/S");
  // The iterator's value_type must itself be sparse (sparse_ptr asserts it).
  using SE = sparse_elem<2, half_t>;
  static SE buf[4] = {SE{half_t(10)}, SE{half_t(11)}, SE{half_t(12)},
                      SE{half_t(13)}};
  using It = SE *;
  auto ps = make_sparse_ptr<2>(It{buf});  // sparse_ptr<2, sparse_elem<2,half_t>*>
  printf("  is_sparse_ptr<sparse_ptr<2,SE*>> = %d ; "
         "iter_value raw sizeof_bits = %d\n",
         (int)is_sparse_ptr<decltype(ps)>::value,
         (int)sizeof_bits_v<iter_value_t<decltype(ps)>>);
  // op+ : allowed offsets are multiples of S, and they advance physical by i/S
  auto p0 = ps + 0;
  auto p2 = ps + 2;  // two *logical* elems ahead -> one physical sparse_elem
  auto p4 = ps + 4;  // -> two physical sparse_elems ahead
  printf("  (ps+0)-buf=%ld  (ps+2)-buf=%ld  (ps+4)-buf=%ld   (physical slots)\n",
         (long)(p0.get() - buf), (long)(p2.get() - buf), (long)(p4.get() - buf));
  // op[] : allows any index, derefs at i/S (so 0 and 1 alias the same physical)
  printf("  ps[0].elem_=%d ps[1].elem_=%d ps[2].elem_=%d ps[3].elem_=%d "
         "(0,1 alias; 2,3 alias)\n",
         (int)(float)ps[0].elem_, (int)(float)ps[1].elem_,
         (int)(float)ps[2].elem_, (int)(float)ps[3].elem_);
}

// ---- (iii) sparse SMEM layout atoms --------------------------------------
template <class Atom>
static void show_atom(const char *name, Atom const &a) {
  printf("  %-34s ", name);
  print(a);
  printf("\n");
}

static void dump_sparse_atoms() {
  hdr("(iii) sparse SMEM layout atoms: same swizzle, S-extended MN-axis");
  show_atom("dense  Layout_MN_SW128_Atom<half_t>",
            UMMA::Layout_MN_SW128_Atom<half_t>{});
  show_atom("sparse Layout_MN_SW128_SpAtom<half_t,2>",
            UMMA::Layout_MN_SW128_SpAtom<half_t, 2>{});
  show_atom("sparse Layout_MN_SW128_SpAtom<half_t,4>",
            UMMA::Layout_MN_SW128_SpAtom<half_t, 4>{});
  show_atom("sparse Layout_K_SW128_SpAtom<half_t,2>",
            UMMA::Layout_K_SW128_SpAtom<half_t, 2>{});
  show_atom("sparse Layout_MN_INTER_SpAtom<half_t,2>",
            UMMA::Layout_MN_INTER_SpAtom<half_t, 2>{});
  show_atom("sparse Layout_MN_SW128_32B_SpAtom<half_t,2>",
            UMMA::Layout_MN_SW128_32B_SpAtom<half_t, 2>{});
  // The A tensor's MN-axis logical extent grows by S while the physical extent
  // is unchanged: compare atom logical shapes.
  printf("  dense  MN_SW128 atom  shape = ");
  print(shape(UMMA::Layout_MN_SW128_Atom<half_t>{}.layout_b()));
  printf("\n  sparse MN_SW128 atom  shape = ");
  print(shape(UMMA::Layout_MN_SW128_SpAtom<half_t, 2>{}.layout_b()));
  printf("\n");
}

// ---- (iv) sparse_smem_desc vs dense smem_desc ----------------------------
// Both are DescriptorIterator wrappers; MakeTensor<sparse_smem_desc> literally
// reuses the dense construction, adding only is_sparse/is_sparse_ptr asserts.
static void dump_sparse_desc() {
  hdr("(iv) UMMA::sparse_smem_desc is bit-identical to UMMA::smem_desc");
  printf("  UMMA::smem_desc<Major::K>         is DescriptorIterator subclass = %d\n",
         (int)is_base_of_v<UMMA::DescriptorIterator,
                           UMMA::smem_desc<UMMA::Major::K>>);
  printf("  UMMA::sparse_smem_desc<Major::K>  is DescriptorIterator subclass = %d\n",
         (int)is_base_of_v<UMMA::DescriptorIterator,
                           UMMA::sparse_smem_desc<UMMA::Major::K>>);
  printf("  MakeTensor<sparse_smem_desc<Major::K>> == MakeTensor<smem_desc<Major::K>> "
         "body (only static_asserts differ)   [source fact: mma_traits_sm100.hpp:379-397]\n");
}

// ---- (v) instruction descriptor sparse_flag_ + K --------------------------
static void dump_instr_desc() {
  hdr("(v) instruction descriptor: sparse_flag_ bit[2] and K = 512/bits");
  using Dense =
      decltype(UMMA::make_instr_desc<half_t, half_t, float, 128, 128,
                                     UMMA::Major::K, UMMA::Major::K>());
  auto dense = UMMA::make_instr_desc<half_t, half_t, float, 128, 128,
                                     UMMA::Major::K, UMMA::Major::K>();
  auto sparse = UMMA::make_instr_desc<half_t, half_t, float, 128, 128,
                                      UMMA::Major::K, UMMA::Major::K,
                                      UMMA::ScaleIn::One, UMMA::ScaleIn::One,
                                      UMMA::Saturate::False, true>();
  (void)sizeof(Dense);
  printf("  dense  desc = 0x%08x  (sparse_flag_=%d)\n", (uint32_t)dense.desc_,
         dense.sparse_flag_);
  printf("  sparse desc = 0x%08x  (sparse_flag_=%d)  -> bit[2] set\n",
         (uint32_t)sparse.desc_, sparse.sparse_flag_);
  printf("  sparse_id2_ = %d (bitfield [0,2)); max_shift_ = %d (bitfield [30,32))\n",
         (int)sparse.sparse_id2_, (int)sparse.max_shift_);
  // K = 512/bits rule from the sparse MMA_Traits.
  printf("  sparse MMA K = 512/sizeof_bits<A>: half(16b)->%d, tf32(32b)->%d, "
         "i8(8b)->%d (dense is half of these)\n",
         512 / (int)sizeof_bits_v<half_t>, 512 / (int)sizeof_bits_v<float>,
         512 / (int)sizeof_bits_v<uint8_t>);
}

int main() {
  printf("CUTLASS SM100 sparse UMMA A-operand probe (host-side, no SM100 needed)\n");
  printf("completes docs/62 (P148-1), which walked only the metadata E half\n");
  dump_sparse_elem();
  dump_sparse_ptr();
  dump_sparse_atoms();
  dump_sparse_desc();
  dump_instr_desc();
  return 0;
}
