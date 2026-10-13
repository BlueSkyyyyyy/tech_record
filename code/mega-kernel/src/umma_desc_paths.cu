// P159-1 (docs/73): UMMA SMEM 描述符构造的「多路并置」——
//   [母本] DeepGEMM 自建  deep_gemm/mma/sm100.cuh        (make_smem_desc / make_umma_desc / advance_umma_desc_lo)
//   [全量] MPK linear_fp8_sm100_sm100_utils.cuh           (同一批 helper 的完整副本, 含 MN-major/BASE32)
//   [子集] MPK sm100_utils.cuh                            (K-major-only 副本)
//   [手写] MPK sm100_ptx.cuh::make_desc                   (固定 128B swizzle / f16 的裸描述符)
// 把四者的「构造算术」在 host 上逐字重放，打印真实 64 位描述符并逐位对比。
//
// 动机：docs/72 §7 局限明留 + ROADMAP「下一步候选」第一条——
//   "MPK `sm100_ptx::make_desc`（固定 128B swizzle 描述符）与 `sm100_utils::make_umma_desc` /
//    DeepGEMM `advance_umma_desc_lo` 的描述符构造并置（docs/69 提过 MPK 侧、未与 DeepGEMM 自建层对走）"
//
// 无需 SM100 硬件（目标卡 H100/sm90 也没有）：SmemDescriptor 是 union + bitfield（CUTE_HOST_DEVICE），
// 各层 make_smem_desc/make_umma_desc/advance_umma_desc_lo 是 __device__，这里逐字重放同一算术。
// 快照: mirage-project/mirage @ f9eb70c;  CUTLASS deps/cutlass @ f3fde583;  DeepGEMM @ 057ca59.

#include <cstdio>
#include <cstdint>
#include <string>

#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

// ============================ 位域自助打印 ============================
static std::string bits64(uint64_t v) {
  std::string s(64, '0');
  for (int i = 0; i < 64; ++i) s[63 - i] = ((v >> i) & 1) ? '1' : '0';
  return s;
}
static void dump_desc(const char* tag, UMMA::SmemDescriptor d) {
  printf("  %-22s desc=0x%016llx  start=%-6u lbo=%-5u sbo=%-5u ver=%u base=%u lbo_mode=%u layout=%u\n",
         tag, (unsigned long long)(uint64_t)d,
         (unsigned)d.start_address_, (unsigned)d.leading_byte_offset_,
         (unsigned)d.stride_byte_offset_, (unsigned)d.version_,
         (unsigned)d.base_offset_, (unsigned)d.lbo_mode_, (unsigned)d.layout_type_);
}

// ============================ [母本] DeepGEMM mma/sm100.cuh ============================
// common/packing.cuh:9-12
template <typename dtype_t>
static constexpr uint32_t dg_pack_factor() {
  return is_same_v<dtype_t, cutlass::float_e2m1_t> ? 2 : 1;
}
// common/tma_copy.cuh:12-15 (注意 4 个模板参: BLOCK_INNER, kSwizzleMode, kPackFactor, dtype)
template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, uint32_t kPackFactor, typename dtype_t>
static constexpr uint32_t dg_inner_block_atom_size() {
  return kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor : kSwizzleMode / sizeof(dtype_t);
}
// mma/sm100.cuh:14-40
static UMMA::SmemDescriptor dg_make_smem_desc(UMMA::LayoutType layout, uint32_t uint_ptr,
                                              uint32_t stride_byte_offset,
                                              uint32_t leading_byte_offset) {
  UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.leading_byte_offset_ = leading_byte_offset >> 4;
  return desc;
}
// mma/sm100.cuh:57-60
static constexpr uint32_t dg_atom_base(UMMA::LayoutType lt) {
  return lt == UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}
// mma/sm100.cuh:64-81
template <UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
static constexpr UMMA::LayoutType dg_to_umma_layout_type() {
  if constexpr ((is_same_v<dtype_t, float> && kMajorMode == UMMA::Major::MN) || kUseBase32) {
    return UMMA::LayoutType::SWIZZLE_128B_BASE32B;
  }
  if constexpr (kSwizzleMode == 0 || kSwizzleMode == 16) return UMMA::LayoutType::SWIZZLE_NONE;
  if constexpr (kSwizzleMode == 32) return UMMA::LayoutType::SWIZZLE_32B;
  if constexpr (kSwizzleMode == 64) return UMMA::LayoutType::SWIZZLE_64B;
  if constexpr (kSwizzleMode == 128) return UMMA::LayoutType::SWIZZLE_128B;
}
// mma/sm100.cuh:83-87
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
static constexpr uint32_t dg_stride_k() {
  return kMajorMode == UMMA::Major::K
             ? 1
             : dg_inner_block_atom_size<BLOCK_MN, kSwizzleMode, 1, dtype_t>();
}
// mma/sm100.cuh:89-96
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
static uint32_t dg_advance_lo(uint32_t base, uint32_t offset, uint32_t k_idx) {
  constexpr uint32_t kPackFactor = dg_pack_factor<dtype_t>();
  const uint32_t byte_offset =
      (offset + k_idx * dg_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>()) *
      static_cast<uint32_t>(sizeof(dtype_t)) / kPackFactor;
  return base + (byte_offset >> 4u);
}
// mma/sm100.cuh:98-142
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          bool kUseBase32 = false, typename dtype_t>
static UMMA::SmemDescriptor dg_make_umma_desc(uint32_t uint_smem_base, uint32_t mn_idx, uint32_t k_idx) {
  constexpr uint32_t kPackFactor = dg_pack_factor<dtype_t>();
  const uint32_t stride_k = dg_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
  const auto layout_type = dg_to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
  const auto num_non_contiguous = 128 / dg_atom_base(layout_type);
  if constexpr (kMajorMode == UMMA::Major::K) {
    const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t) / kPackFactor;
    const uint32_t leading_byte_offset = 0;
    const uint32_t byte_ptr =
        uint_smem_base + (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t) / kPackFactor;
    return dg_make_smem_desc(layout_type, byte_ptr, stride_byte_offset, leading_byte_offset);
  } else {
    constexpr uint32_t BLOCK_MN_ATOM = dg_inner_block_atom_size<BLOCK_MN, kSwizzleMode, 1, dtype_t>();
    uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
    uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
    if constexpr (kSwizzleMode == 16) { uint32_t t = stride_byte_offset; stride_byte_offset = leading_byte_offset; leading_byte_offset = t; }
    return dg_make_smem_desc(layout_type,
                             uint_smem_base + (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t),
                             stride_byte_offset, leading_byte_offset);
  }
}

// ============================ [全量/子集] MPK 两份副本 ============================
// [全量] linear_fp8_sm100_tma_utils.cuh:11-14
template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
static constexpr uint32_t mpk_inner_block_atom_size() {
  return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}
// [全量] linear_fp8_sm100_sm100_utils.cuh:13-28
static UMMA::SmemDescriptor mpk_make_smem_desc(UMMA::LayoutType layout, uint32_t uint_ptr,
                                               uint32_t stride_byte_offset,
                                               uint32_t leading_byte_offset) {
  UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.leading_byte_offset_ = leading_byte_offset >> 4;
  return desc;
}
// [全量] linear_fp8_sm100_sm100_utils.cuh:48-89
template <UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
static constexpr UMMA::LayoutType mpk_to_umma_layout_type() {
  if constexpr ((is_same_v<dtype_t, float> && kMajorMode == UMMA::Major::MN) || kUseBase32)
    return UMMA::LayoutType::SWIZZLE_128B_BASE32B;
  if constexpr (kSwizzleMode == 0 || kSwizzleMode == 16) return UMMA::LayoutType::SWIZZLE_NONE;
  if constexpr (kSwizzleMode == 32) return UMMA::LayoutType::SWIZZLE_32B;
  if constexpr (kSwizzleMode == 64) return UMMA::LayoutType::SWIZZLE_64B;
  if constexpr (kSwizzleMode == 128) return UMMA::LayoutType::SWIZZLE_128B;
}
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
static constexpr uint32_t mpk_stride_k() {
  return kMajorMode == UMMA::Major::K ? 1 : mpk_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}
// [全量] linear_fp8_sm100_sm100_utils.cuh:91-103 (无 pack factor)
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
static uint32_t mpk_advance_lo(uint32_t base, uint32_t offset, uint32_t k_idx) {
  return base + (((offset + k_idx * mpk_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>()) *
                  static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}
// [全量] linear_fp8_sm100_sm100_utils.cuh:105-145
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          bool kUseBase32 = false, typename dtype_t>
static UMMA::SmemDescriptor mpk_make_umma_desc(uint32_t uint_smem_base, uint32_t mn_idx, uint32_t k_idx) {
  const uint32_t stride_k = mpk_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
  const auto layout_type = mpk_to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
  const auto num_non_contiguous = 128 / dg_atom_base(layout_type);
  if constexpr (kMajorMode == UMMA::Major::K) {
    const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
    const uint32_t leading_byte_offset = 0;
    return mpk_make_smem_desc(layout_type,
                              uint_smem_base + (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t),
                              stride_byte_offset, leading_byte_offset);
  } else {
    constexpr uint32_t BLOCK_MN_ATOM = mpk_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
    uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
    uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
    if constexpr (kSwizzleMode == 16) { uint32_t t = stride_byte_offset; stride_byte_offset = leading_byte_offset; leading_byte_offset = t; }
    return mpk_make_smem_desc(layout_type,
                              uint_smem_base + (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t),
                              stride_byte_offset, leading_byte_offset);
  }
}
// [子集] sm100_utils.cuh:79-115 (K-major-only, 硬编码 SWIZZLE_128B)
template <uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
static uint32_t sub_advance_lo(uint32_t base, uint32_t offset, uint32_t k_idx) {
  (void)BLOCK_MN; (void)kSwizzleMode;
  return base + (((offset + k_idx) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}
template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
static UMMA::SmemDescriptor sub_make_umma_desc(uint32_t uint_smem_base, uint32_t mn_idx, uint32_t k_idx) {
  constexpr auto layout_type = UMMA::LayoutType::SWIZZLE_128B;
  const uint32_t stride_byte_offset = 8 * BLOCK_K * sizeof(dtype_t);
  const uint32_t leading_byte_offset = 0;
  return mpk_make_smem_desc(layout_type,
                            uint_smem_base + (mn_idx * BLOCK_K + k_idx) * sizeof(dtype_t),
                            stride_byte_offset, leading_byte_offset);
}

// ============================ [手写] MPK sm100_ptx.cuh:59-67 ============================
static constexpr uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFULL) >> 4; }
static uint64_t hand_make_desc(uint32_t smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) | (2ULL << 61);
}

// ============================ 工具 ============================
static void cmpres(const char* what, uint64_t a, uint64_t b) {
  printf("  %-46s %s  (0x%016llx vs 0x%016llx)\n", what, a == b ? "MATCH" : "DIFF ",
         (unsigned long long)a, (unsigned long long)b);
}

int main() {
  printf("== UMMA SMEM 描述符构造多路并置 (host 重放) ==\n");
  printf("   CUTLASS SmemDescriptor = union{bitfield, {lo,hi}}  (CUTE_HOST_DEVICE, host 可精确求值)\n");
  printf("   bitfield: start[0,14) lbo[16,30) sbo[32,46) version[46,48) base[49,52) lbo_mode[52) layout[61,64)\n\n");

  const uint32_t SMEM = 0x400;  // 假 SMEM 基址 (uint, 需 1024B 对齐见 docs/58)

  // ---------- §A: make_smem_desc —— 三份 body 逐位相同 ----------
  printf("§A  make_smem_desc 三份对照 (SWIZZLE_128B, sbo=1024, lbo=0, ptr=0x%x)\n", SMEM);
  {
    auto a = dg_make_smem_desc(UMMA::LayoutType::SWIZZLE_128B, SMEM, 1024, 0);
    auto b = mpk_make_smem_desc(UMMA::LayoutType::SWIZZLE_128B, SMEM, 1024, 0);
    dump_desc("DeepGEMM (母本)", a);
    dump_desc("MPK full (_sm100_utils)", b);
    cmpres("dg == mpk(full)", (uint64_t)a, (uint64_t)b);
  }

  // ---------- §B: K-major fp8 SW128 —— 三层 make_umma_desc 逐位相同 ----------
  printf("\n§B  make_umma_desc: K-major fp8(e4m3) BLOCK_MN=128 BLOCK_K=128 swizzle=128, mn=1 k=0\n");
  printf("    (swizzle_bytes == BLOCK_K*sizeof(dtype) 的 K-major 契约)\n");
  {
    using T = cutlass::float_e4m3_t;
    auto a = dg_make_umma_desc<UMMA::Major::K, 128, 128, 128, false, T>(SMEM, 1, 0);
    auto b = mpk_make_umma_desc<UMMA::Major::K, 128, 128, 128, false, T>(SMEM, 1, 0);
    auto c = sub_make_umma_desc<128, 128, 128, T>(SMEM, 1, 0);
    dump_desc("DeepGEMM", a);
    dump_desc("MPK full", b);
    dump_desc("MPK subset (sm100_utils)", c);
    cmpres("dg == mpk(full)", (uint64_t)a, (uint64_t)b);
    cmpres("dg == mpk(subset)", (uint64_t)a, (uint64_t)c);
  }

  // ---------- §C: MN-major bf16 —— 只有 [全量] 能走 ----------
  printf("\n§C  make_umma_desc: MN-major bf16 BLOCK_MN=64 BLOCK_K=64 swizzle=64, mn=0 k=0/1\n");
  {
    using T = cutlass::bfloat16_t;
    auto a0 = dg_make_umma_desc<UMMA::Major::MN, 64, 64, 64, false, T>(SMEM, 0, 0);
    auto b0 = mpk_make_umma_desc<UMMA::Major::MN, 64, 64, 64, false, T>(SMEM, 0, 0);
    auto a1 = dg_make_umma_desc<UMMA::Major::MN, 64, 64, 64, false, T>(SMEM, 0, 1);
    dump_desc("DeepGEMM   k=0", a0);
    dump_desc("MPK full   k=0", b0);
    dump_desc("DeepGEMM   k=1", a1);
    cmpres("dg == mpk(full) @k0", (uint64_t)a0, (uint64_t)b0);
    printf("    (MPK subset 无 MN-major 分支: static_assert(K-major) —— 不参与此处)\n");
  }

  // ---------- §D: 同一 MN-major bf16 的 advance_umma_desc_lo 步进 ----------
  printf("\n§D  advance_umma_desc_lo 步进 (K-major fp8, base=0x40, offset=0, k_idx 0..3; 单位=>>4 粒度)\n");
  {
    using T = cutlass::float_e4m3_t;
    printf("    k_idx:");
    for (int k = 0; k < 4; ++k) printf(" %6d", k);
    printf("\n    dg   :");
    for (int k = 0; k < 4; ++k) printf(" %6u", dg_advance_lo<UMMA::Major::K, 128, 128, T>(0x40, 0, k));
    printf("\n    mpk  :");
    for (int k = 0; k < 4; ++k) printf(" %6u", mpk_advance_lo<UMMA::Major::K, 128, 128, T>(0x40, 0, k));
    printf("\n    sub  :");
    for (int k = 0; k < 4; ++k) printf(" %6u", sub_advance_lo<128, 128, T>(0x40, 0, k));
    printf("\n");
    bool same = true;
    for (int k = 0; k < 4; ++k) {
      uint32_t a = dg_advance_lo<UMMA::Major::K, 128, 128, T>(0x40, 0, k);
      uint32_t b = mpk_advance_lo<UMMA::Major::K, 128, 128, T>(0x40, 0, k);
      uint32_t c = sub_advance_lo<128, 128, T>(0x40, 0, k);
      if (a != b || a != c) same = false;
    }
    printf("    三层步进逐点一致: %s\n", same ? "YES" : "NO");
  }

  // ---------- §E: pack-factor 分叉 —— fp4 暴露「母本 ÷kPackFactor、MPK 不除」 ----------
  printf("\n§E  pack-factor 分叉 (K-major fp4 e2m1, sizeof=1, pack=2, BLOCK_MN=128 BLOCK_K=128 swizzle=128)\n");
  printf("    DeepGEMM 母本对 sub-byte packed 类型除 kPackFactor; MPK 两份副本均无 pack factor\n");
  {
    using T = cutlass::float_e2m1_t;
    constexpr uint32_t pack = dg_pack_factor<T>();
    printf("    sizeof(e2m1)=%zu  pack_factor=%u\n", sizeof(T), pack);
    // 母本: sbo = num_non_contig(8)*128*1/2 = 512 -> >>4 = 32
    auto a = dg_make_umma_desc<UMMA::Major::K, 128, 128, 128, false, T>(SMEM, 0, 0);
    // MPK: sbo = 8*128*1 = 1024 -> >>4 = 64
    auto b = mpk_make_umma_desc<UMMA::Major::K, 128, 128, 128, false, T>(SMEM, 0, 0);
    dump_desc("DeepGEMM (÷pack)", a);
    dump_desc("MPK full (no pack)", b);
    cmpres("dg == mpk(full) for fp4", (uint64_t)a, (uint64_t)b);
    printf("    => 差异仅在 sub-byte packed 类型; 生产用 fp8/bf16 (pack=1) 时无差异 (§B/§C)\n");
  }

  // ---------- §F: 手写 make_desc vs 通用 K-major f16 SW128 描述符 ----------
  printf("\n§F  sm100_ptx::make_desc(smem) vs 通用 make_umma_desc: K-major f16 BLOCK_K=64 swizzle=128\n");
  printf("    (f16: swizzle_bytes 128 == BLOCK_K*2 => BLOCK_K=64; SBO=8*128=1024)\n");
  {
    using T = cutlass::half_t;
    uint64_t hand = hand_make_desc(SMEM);
    UMMA::SmemDescriptor g = dg_make_umma_desc<UMMA::Major::K, /*BLOCK_MN*/128, /*BLOCK_K*/64, 128, false, T>(SMEM, 0, 0);
    UMMA::SmemDescriptor gk = sub_make_umma_desc<128, 64, 128, T>(SMEM, 0, 0);
    printf("    hand  desc=0x%016llx\n", (unsigned long long)hand);
    printf("    bits=%s\n", bits64(hand).c_str());
    dump_desc("generic dg (f16 SW128)", g);
    dump_desc("generic mpk-sub", gk);
    cmpres("hand == generic(DeepGEMM)", hand, (uint64_t)g);
    cmpres("hand == generic(MPK subset)", hand, (uint64_t)gk);
    // 直接解手写描述符的四个字段
    UMMA::SmemDescriptor h; h.desc_ = hand;
    printf("    decode hand: start=%u sbo=%u ver=%u layout=%u  (== 0x%x>>4=%u, 1024>>4=%u)\n",
           (unsigned)h.start_address_, (unsigned)h.stride_byte_offset_, (unsigned)h.version_,
           (unsigned)h.layout_type_, SMEM, SMEM >> 4, 1024u >> 4);
  }

  printf("\n== done ==\n");
  return 0;
}
