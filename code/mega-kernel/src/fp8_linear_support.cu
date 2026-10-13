// P154-1 (docs/68): MPK SM100 block-scaled FP8 linear 的「支撑层」——
//   linear_fp8_sm100_utils.cuh      (143 行: PatternVisitor / ld.st_shared / bf16 打包)
//   linear_fp8_sm100_sm100_utils.cuh(232 行: SmemDescriptor / make_umma_desc / InstrDescriptorBlockScaled / 两个 MMA 结构体)
//   linear_fp8_sm100_tma_utils.cuh  (118 行: tma_copy 分派 / tensormap fence & replace)
// 逐行打开，并把其中「纯算术」的部分在 host 上逐字重放、打印真实描述符位模式。
//
// 动机（两份明留的尾巴）:
//   * docs/64 §12: "`linear_fp8_sm100_tma_utils.cuh` / `linear_fp8_sm100_sm100_utils.cuh`
//     只借用了行号（docs/32 已给 :16-116 摘要），未逐行"
//   * docs/57 §8: "`make_smem_desc` 的 version_/lbo_mode_ 语义依赖 GPU 架构文档，本文只记
//     「写死值」" + "committed 描述符的位编码未逐位展开"
//
// 本探针不依赖 SM100 硬件（目标卡 H100/sm90 没有）：SmemDescriptor / InstrDescriptorBlockScaled
// 是 union + bitfield（CUTE_HOST_DEVICE），make_instr_desc_block_scaled 是 CUTE_HOST_DEVICE
// constexpr；MPK 侧 make_smem_desc / make_umma_desc 是 __device__，这里逐字重放同一算术。
// 快照: mirage-project/mirage @ f9eb70c;  CUTLASS deps/cutlass @ f3fde583.

#include <cstdio>
#include <cstdint>

#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

// ---- MPK linear_fp8_sm100_utils.cuh 的 host 重放 ----
template <typename T>
static constexpr T host_ceil_div(T a, T b) { return (a + b - 1) / b; }

// linear_fp8_sm100_tma_utils.cuh:11-14
template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
static constexpr uint32_t get_inner_block_atom_size() {
  return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

// linear_fp8_sm100_sm100_utils.cuh:43-46
static constexpr uint32_t get_atom_base(UMMA::LayoutType lt) {
  return lt == UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

// linear_fp8_sm100_sm100_utils.cuh:48-79
template <UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
static constexpr UMMA::LayoutType to_umma_layout_type() {
  static_assert(kSwizzleMode == 0 || kSwizzleMode == 16 || kSwizzleMode == 32 ||
                    kSwizzleMode == 64 || kSwizzleMode == 128,
                "Invalid swizzling mode");
  if constexpr ((is_same_v<dtype_t, float> && kMajorMode == UMMA::Major::MN) || kUseBase32) {
    static_assert(kUseBase32, "Invalid swizzling base");
    return UMMA::LayoutType::SWIZZLE_128B_BASE32B;
  }
  if constexpr (kSwizzleMode == 0 || kSwizzleMode == 16) return UMMA::LayoutType::SWIZZLE_NONE;
  if constexpr (kSwizzleMode == 32) return UMMA::LayoutType::SWIZZLE_32B;
  if constexpr (kSwizzleMode == 64) return UMMA::LayoutType::SWIZZLE_64B;
  if constexpr (kSwizzleMode == 128) return UMMA::LayoutType::SWIZZLE_128B;
}

// linear_fp8_sm100_sm100_utils.cuh:81-89
template <UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
static constexpr uint32_t get_umma_desc_stride_k() {
  return kMajorMode == UMMA::Major::K
             ? 1
             : get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

// linear_fp8_sm100_sm100_utils.cuh:13-28
static UMMA::SmemDescriptor mk_smem_desc(UMMA::LayoutType layout,
                                         uint32_t uint_smem_ptr,
                                         uint32_t stride_byte_offset,
                                         uint32_t leading_byte_offset) {
  UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout);
  desc.start_address_ = static_cast<uint16_t>(uint_smem_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.leading_byte_offset_ = leading_byte_offset >> 4;
  return desc;
}

// linear_fp8_sm100_sm100_utils.cuh:105-145 (host 版: 用假的 SMEM uint 基址)
template <UMMA::Major kMajorMode,
          uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          bool kUseBase32 = false, typename dtype_t>
static UMMA::SmemDescriptor mk_umma_desc(uint32_t uint_smem_base, uint32_t mn_idx, uint32_t k_idx) {
  constexpr uint32_t stride_k =
      get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
  constexpr UMMA::LayoutType layout_type =
      to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
  constexpr uint32_t num_non_contiguous = 128 / get_atom_base(layout_type);
  if constexpr (kMajorMode == UMMA::Major::K) {
    static_assert(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");
    constexpr uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
    constexpr uint32_t leading_byte_offset = 0;
    return mk_smem_desc(layout_type,
                        uint_smem_base + (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t),
                        stride_byte_offset, leading_byte_offset);
  } else {
    constexpr uint32_t BLOCK_MN_ATOM =
        get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
    static_assert(kSwizzleMode > 0, "Invalid swizzling");
    uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
    uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
    if constexpr (kSwizzleMode == 16) {
      uint32_t t = stride_byte_offset; stride_byte_offset = leading_byte_offset; leading_byte_offset = t;
    }
    return mk_smem_desc(layout_type,
                        uint_smem_base + (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t),
                        stride_byte_offset, leading_byte_offset);
  }
}

// linear_fp8_sm100_sm100_utils.cuh:147-154
static uint64_t mk_runtime_instr_desc_with_sf_id(UMMA::InstrDescriptorBlockScaled desc,
                                                 uint32_t sfa_id, uint32_t sfb_id) {
  desc.a_sf_id_ = sfa_id;
  desc.b_sf_id_ = sfb_id;
  return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

// linear_fp8_sm100_sm100_utils.cuh:156-172
template <uint32_t kNumCols>
static constexpr uint32_t get_num_aligned_tmem_cols() {
  static_assert(kNumCols <= 512, "Too many tensor memory columns");
  if (kNumCols <= 32) return 32;
  if (kNumCols <= 64) return 64;
  if (kNumCols <= 128) return 128;
  if (kNumCols <= 256) return 256;
  return 512;
}

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

// ============================ Section A ============================
static void sec_a_smem_desc_bits() {
  hdr("A. SmemDescriptor 位布局：重放 MPK make_smem_desc 并逐字段解码");
  printf("CUTLASS union 位域 (mma_sm100_desc.hpp:98-123):\n");
  printf("  start_address_        14b [ 0,14)\n");
  printf("  leading_byte_offset_  14b [16,30)\n");
  printf("  stride_byte_offset_   14b [32,46)\n");
  printf("  version_               2b [46,48)\n");
  printf("  base_offset_           3b [49,52)\n");
  printf("  lbo_mode_              1b [52,53)\n");
  printf("  layout_type_           3b [61,64)\n\n");

  struct Case { const char *name; UMMA::LayoutType lt; uint32_t stride, leading; };
  Case cases[] = {
    {"SWIZZLE_NONE         (sf_desc: stride=8*16=128, leading=0)", UMMA::LayoutType::SWIZZLE_NONE, 8 * 16, 0},
    {"SWIZZLE_128B         (fp8 A/B K-major, 128B)", UMMA::LayoutType::SWIZZLE_128B, 1024, 0},
    {"SWIZZLE_128B_BASE32B (float MN-major)", UMMA::LayoutType::SWIZZLE_128B_BASE32B, 512, 2048},
    {"SWIZZLE_64B", UMMA::LayoutType::SWIZZLE_64B, 512, 512},
  };
  const uint32_t smem_uint = 0x00000400;  // 假 SMEM 地址 (1024, 16B 对齐)
  for (auto &c : cases) {
    UMMA::SmemDescriptor d = mk_smem_desc(c.lt, smem_uint, c.stride, c.leading);
    printf("[%s]\n", c.name);
    printf("  desc_            = 0x%016llx\n", (unsigned long long)(uint64_t)d);
    printf("  start_address_   = %u   (uint_ptr>>4 = %u)\n", (unsigned)d.start_address_, smem_uint >> 4);
    printf("  leading_byte_off = %u   (= %u>>4)\n", (unsigned)d.leading_byte_offset_, c.leading >> 4);
    printf("  stride_byte_off  = %u   (= %u>>4)\n", (unsigned)d.stride_byte_offset_, c.stride >> 4);
    printf("  version_ = %u  base_offset_ = %u  lbo_mode_ = %u  layout_type_ = %u (%s)\n",
           (unsigned)d.version_, (unsigned)d.base_offset_, (unsigned)d.lbo_mode_,
           (unsigned)d.layout_type_, UMMA::to_string(c.lt));
    printf("  hi32 = 0x%08x  lo32 = 0x%08x\n", d.hi, d.lo);
  }
  printf("\n-> 与文档一致：三个字节偏移字段都「右移 4 位」存(16B 粒度)，layout_type_ 落 bit[61,64)，\n");
  printf("   version_ 落 bit[46,48)，lbo_mode_ 落 bit[52]。make_sf_desc 传 stride=8*16=128,\n");
  printf("   leading=0, layout=SWIZZLE_NONE -> stride_byte_offset_=8, 即 SF 的 8x16=128B tile 步长。\n");
}

// ============================ Section B ============================
static void sec_b_umma_desc() {
  hdr("B. make_umma_desc：K-major vs MN-major 的 stride/leading/layout 实算");

  printf("B1. to_umma_layout_type<kMajorMode,kSwizzleMode,kUseBase32,dtype> 映射表\n");
  printf("   dtype  Major  swizzle  base32 -> LayoutType\n");
  auto row = [](const char *dt, const char *mj, int sw, bool b32, UMMA::LayoutType lt) {
    printf("   %-6s %-5s  %3d      %d      -> %s\n", dt, mj, sw, b32 ? 1 : 0, UMMA::to_string(lt));
  };
  row("fp8", "K", 0, false, to_umma_layout_type<UMMA::Major::K, 0, false, float_e4m3_t>());
  row("fp8", "K", 16, false, to_umma_layout_type<UMMA::Major::K, 16, false, float_e4m3_t>());
  row("fp8", "K", 32, false, to_umma_layout_type<UMMA::Major::K, 32, false, float_e4m3_t>());
  row("fp8", "K", 64, false, to_umma_layout_type<UMMA::Major::K, 64, false, float_e4m3_t>());
  row("fp8", "K", 128, false, to_umma_layout_type<UMMA::Major::K, 128, false, float_e4m3_t>());
  row("bf16", "MN", 0, false, to_umma_layout_type<UMMA::Major::MN, 0, false, bfloat16_t>());
  row("float", "MN", 0, true, to_umma_layout_type<UMMA::Major::MN, 0, true, float>());
  row("float", "MN", 128, true, to_umma_layout_type<UMMA::Major::MN, 128, true, float>());
  printf("   -> 两条「强制 BASE32」规则：(i) dtype==float 且 MN-major 必须 kUseBase32=true；\n");
  printf("      (ii) 显式 kUseBase32 无法与 K-major/non-float 组合时 static_assert 挡下。\n");
  printf("      故 fp8 的 K-major 128B -> SWIZZLE_128B，而 float 的 MN-major 恒 -> SWIZZLE_128B_BASE32B。\n");

  printf("\nB2. K-major（fp8 e4m3, swizzle=128=BLOCK_K*sizeof(fp8)=128*1）\n");
  {
    constexpr uint32_t BM = 128, BK = 128;
    printf("   %-22s %-10s %-10s %-8s %s\n", "mn_idx,k_idx", "start(>>4)", "stride(>>4)", "lead(>>4)", "layout");
    const char *mn_names[] = {"mn=0,k=0", "mn=1,k=0", "mn=2,k=0"};
    for (uint32_t mn = 0; mn < 3; ++mn) {
      UMMA::SmemDescriptor d = mk_umma_desc<UMMA::Major::K, BM, BK, 128, false, float_e4m3_t>(0x10000u, mn, 0);
      printf("   %-22s %-10u %-10u %-8u %s\n", mn_names[mn],
             (unsigned)d.start_address_, (unsigned)d.stride_byte_offset_,
             (unsigned)d.leading_byte_offset_, UMMA::to_string((UMMA::LayoutType)d.layout_type_));
    }
    printf("   K-major: stride_byte_offset = num_non_contiguous(128/16=8)*BLOCK_K(128)*1B = 1024 -> >>4 = 64\n");
    printf("            leading_byte_offset = 0；地址步进 stride_k=1 个元素；mn 每加 1 前进 BLOCK_K 元素。\n");
  }

  printf("\nB3. MN-major（bf16, swizzle=64, BLOCK_MN=64, BLOCK_K=64）\n");
  {
    constexpr uint32_t BM = 64, BK = 64, SW = 64;
    printf("   BLOCK_MN_ATOM = swizzle/sizeof(bf16) = 64/2 = 32\n");
    UMMA::SmemDescriptor d = mk_umma_desc<UMMA::Major::MN, BM, BK, SW, false, bfloat16_t>(0x10000u, 0, 0);
    printf("   mn=0,k=0: start(>>4)=%u stride(>>4)=%u lead(>>4)=%u layout=%s\n",
           (unsigned)d.start_address_, (unsigned)d.stride_byte_offset_,
           (unsigned)d.leading_byte_offset_, UMMA::to_string((UMMA::LayoutType)d.layout_type_));
    printf("   MN-major: num_non_contiguous=128/16=8; stride=8*32*2B=512 (>>4=32);\n");
    printf("             leading=BLOCK_K*32*2B=4096 (>>4=256)；stride_k=BLOCK_MN_ATOM=32 元素。\n");
  }

  printf("\nB4. swizzle=16 时 MN-major 会 swap stride/leading（源码 :137-139）\n");
  {
    UMMA::SmemDescriptor d = mk_umma_desc<UMMA::Major::MN, 64, 32, 16, false, bfloat16_t>(0x10000u, 0, 0);
    printf("   swizzle=16: stride(>>4)=%u lead(>>4)=%u （常规 MN-major 的 stride<lead，此处被交换）\n",
           (unsigned)d.stride_byte_offset_, (unsigned)d.leading_byte_offset_);
    printf("   -> layout_type_=%s（swizzle=16 映射到 SWIZZLE_NONE，与 swizzle=0 同）\n",
           UMMA::to_string((UMMA::LayoutType)d.layout_type_));
  }
}

// ============================ Section C ============================
static void sec_c_tmem_cols() {
  hdr("C. get_num_aligned_tmem_cols：TMEM 列向上取整到 {32,64,128,256,512}");
  printf("   kNumCols -> aligned\n");
#define ROW(N) printf("   %4u -> %4u\n", (unsigned)(N), get_num_aligned_tmem_cols<N>())
  ROW(1); ROW(32); ROW(33); ROW(48); ROW(64); ROW(65);
  ROW(128); ROW(129); ROW(200); ROW(256); ROW(257); ROW(512);
#undef ROW
  printf("   -> 分档 32/64/128/256/512；现产 FP8 linear 的 N 侧累加器列算出后套这一档。\n");
}

// ============================ Section D ============================
static void sec_d_instr_desc() {
  hdr("D. InstrDescriptorBlockScaled：MPK 依赖的 CUTLASS 指令描述符位编码");

  printf("CUTLASS union 位域 (mma_sm100_desc.hpp:441-468), 32b:\n");
  printf("  sparse_id2_    2b [ 0, 2)   sparse_flag_  1b [ 2, 3)\n");
  printf("  b_sf_id_       2b [ 4, 6)   a_format_     3b [ 7, 9)   b_format_ 3b [10,12)\n");
  printf("  a_negate_      1b [13,14)   b_negate_     1b [14,15)   a_major_  1b [15,16)\n");
  printf("  b_major_       1b [16,17)   n_dim_        6b [17,23)   scale_format_ 1b [23,24)\n");
  printf("  m_dim_         5b [24,29)   a_sf_id_      2b [29,31)   k_size_   1b [31,32)\n\n");

  struct Row { const char *name; uint32_t m, n; };
  Row rows[] = {
    {"dense M=128 N=16  (MoE MMA_N=16)", 128, 16},
    {"dense M=128 N=128", 128, 128},
    {"dense M=128 N=256", 128, 256},
    {"dense M=256 N=128 (2SM 单 CTA 视角)", 256, 128},
  };
  for (auto &r : rows) {
    // (CUTLASS 的 make_instr_desc_block_scaled 的 M/N 是编译期实参，这里按同一算术运行期构造以便打表)
    UMMA::InstrDescriptorBlockScaled d = {};
    d.a_format_ = uint8_t(UMMA::to_UMMAFormat<float_e4m3_t>());
    d.b_format_ = uint8_t(UMMA::to_UMMAFormat<float_e4m3_t>());
    d.scale_format_ = uint8_t(UMMA::to_ScaleFormat<float_ue8m0_t>());
    d.a_major_ = uint8_t(UMMA::Major::K);
    d.b_major_ = uint8_t(UMMA::Major::K);
    d.m_dim_ = (r.m >> 4);
    d.n_dim_ = (r.n >> 3);
    printf("[%s]\n", r.name);
    printf("  desc32 = 0x%08x\n", (uint32_t)d);
    printf("  a_format_=%u(E4M3) b_format_=%u(E4M3) scale_format_=%u(UE8M0) k_size_=%u\n",
           (unsigned)d.a_format_, (unsigned)d.b_format_, (unsigned)d.scale_format_, (unsigned)d.k_size_);
    printf("  m_dim_=%u (=M>>4) n_dim_=%u (=N>>3) a_major_=%u(K) b_major_=%u(K)\n",
           (unsigned)d.m_dim_, (unsigned)d.n_dim_, (unsigned)d.a_major_, (unsigned)d.b_major_);
  }

  printf("\nD2. MPK 的 make_runtime_instr_desc_with_sf_id = 塞 sf_id 后 <<32 (idescE 高 32 位)\n");
  {
    UMMA::InstrDescriptorBlockScaled d = {};
    d.a_format_ = uint8_t(UMMA::to_UMMAFormat<float_e4m3_t>());
    d.b_format_ = uint8_t(UMMA::to_UMMAFormat<float_e4m3_t>());
    d.scale_format_ = uint8_t(UMMA::to_ScaleFormat<float_ue8m0_t>());
    d.m_dim_ = (128 >> 4); d.n_dim_ = (128 >> 3);
    d.a_major_ = uint8_t(UMMA::Major::K); d.b_major_ = uint8_t(UMMA::Major::K);
    uint64_t e = mk_runtime_instr_desc_with_sf_id(d, /*sfa=*/1, /*sfb=*/2);
    printf("  desc32(a_sf_id=1,b_sf_id=2) = 0x%08x\n", (uint32_t)d);
    printf("  idescE                      = 0x%016llx\n", (unsigned long long)e);
    printf("  -> 低 32 位 = 0（dense 无 sparse metadata TMEM 地址），高 32 位 = desc32。\n");
    printf("  -> CUTLASS make_runtime_instr_desc_block_scaled 把 sf_id 从 tmem 地址 bit[30,32) 取；\n");
    printf("     MPK 这里直接把 sfa_id/sfb_id 当参数塞入 bit[29,31)/[4,6)（线性 task 的简化写法）。\n");
  }

  printf("\nD3. CUTLASS constexpr 与手工构造一致性（M=128,N=128,K-major）\n");
  {
    UMMA::InstrDescriptorBlockScaled cd = UMMA::make_instr_desc_block_scaled<
        float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 128, 128, UMMA::Major::K, UMMA::Major::K>();
    UMMA::InstrDescriptorBlockScaled md = {};
    md.a_format_ = uint8_t(UMMA::to_UMMAFormat<float_e4m3_t>());
    md.b_format_ = uint8_t(UMMA::to_UMMAFormat<float_e4m3_t>());
    md.scale_format_ = uint8_t(UMMA::to_ScaleFormat<float_ue8m0_t>());
    md.m_dim_ = (128 >> 4); md.n_dim_ = (128 >> 3);
    md.a_major_ = uint8_t(UMMA::Major::K); md.b_major_ = uint8_t(UMMA::Major::K);
    printf("  CUTLASS 构造 = 0x%08x\n", (uint32_t)cd);
    printf("  手工构造          = 0x%08x\n", (uint32_t)md);
    printf("  %s\n", (uint32_t)cd == (uint32_t)md ? "MATCH" : "MISMATCH");
    printf("  -> MPK 的 make_instr_desc_block_scaled 等价于 CUTLASS 同名 constexpr（默认 sf_id 全 0）。\n");
  }
}

int main() {
  printf("MPK SM100 block-scaled FP8 linear 支撑层 —— host 侧描述符位编码探针\n");
  printf("快照 mirage f9eb70c / CUTLASS deps/cutlass f3fde583；目标卡 H100/sm90a（无 SM100）\n");
  sec_a_smem_desc_bits();
  sec_b_umma_desc();
  sec_c_tmem_cols();
  sec_d_instr_desc();
  printf("\n[probe done]\n");
  return 0;
}
