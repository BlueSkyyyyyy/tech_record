// P86-1 experiment: H100(sm90) verification of the two remaining "mega family" JIT
// launch wrappers by directly including the real DeepGEMM host headers.
//   csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp   (FP8/FP4 Mega MoE)
//   csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp     (BF16 Mega Gate)
// plus the two heuristics modules they consume:
//   csrc/jit_kernels/heuristics/mega_moe.hpp
//   csrc/jit_kernels/heuristics/mega_gate.hpp
//
// Everything exercised here is pure host C++ + torch: config derivation, TMA descriptor
// composition, launch geometry, RoutingArgs ABI and the auto-generated JIT source arity.
// The SM100 device kernels themselves stay uninstantiated (only analysed in the doc).
//
// Build/run: harness/exp_mega_moe_gate_jit_wrapper.sh

#include <cassert>
// In an nvcc single-TU host pass the real headers' DG_UNIFIED_ASSERT falls through to
// asm("trap;"), which the host assembler rejects. Pin it to the host assert first.
#ifndef DG_UNIFIED_ASSERT
#define DG_UNIFIED_ASSERT(cond) assert(cond)
#endif

#include <torch/extension.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <deep_jit/backend/cuda/driver.hpp>
#include <deep_gemm/scheduler/mega_moe.cuh>

#include <csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp>
#include <csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp>
#include <csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp>

using namespace deep_gemm;
namespace mgl = layout::mega_gate;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { ++g_pass; std::printf("  PASS  %s\n", msg); }               \
        else      { ++g_fail; std::printf("  FAIL  %s\n", msg); }               \
    } while (0)

#define CHECK_DESC_EQ(a, b, msg)                                                \
    do {                                                                        \
        const bool eq = std::memcmp(&(a), &(b), sizeof(CUtensorMap)) == 0;      \
        CHECK(eq, msg);                                                         \
        if (not eq) {                                                           \
            const auto* pa = reinterpret_cast<const unsigned char*>(&(a));      \
            const auto* pb = reinterpret_cast<const unsigned char*>(&(b));      \
            for (int i = 0; i < (int)sizeof(CUtensorMap); ++i)                  \
                if (pa[i] != pb[i]) { std::printf("        first diff byte %d: %02x vs %02x\n", i, pa[i], pb[i]); break; } \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Independent TMA encoder (from contract-level args), cf. P56's reference.
// ---------------------------------------------------------------------------
static CUtensorMap indep_encode_tiled(CUtensorMapDataType dt, void* ptr,
                                      uint64_t g_d0, uint64_t g_d1, uint64_t stride1_bytes,
                                      uint32_t b_d0, uint32_t b_d1, CUtensorMapSwizzle sw) {
    CUtensorMap tm{};
    const cuuint64_t dims[2] = {g_d0, g_d1};
    const cuuint64_t strides[1] = {stride1_bytes};
    const cuuint32_t box[2] = {b_d0, b_d1};
    const cuuint32_t es[2] = {1, 1};
    const auto rc = deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &tm, dt, 2, ptr, dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rc == CUDA_SUCCESS, "indep cuTensorMapEncodeTiled failed rc=", (int)rc);
    return tm;
}

// Count the `{}` placeholders inside the first `std::format(R"( ... )"` of a header.
static int count_format_placeholders(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss; ss << f.rdbuf();
    const std::string s = ss.str();
    const auto p = s.find("std::format(R\"(");
    if (p == std::string::npos) return -1;
    const auto start = p + std::strlen("std::format(R\"(");
    const auto end = s.find(")\"", start);
    if (end == std::string::npos) return -1;
    const std::string body = s.substr(start, end - start);
    int n = 0;
    for (std::size_t i = 0; i + 1 < body.size(); ++i)
        if (body[i] == '{' and body[i + 1] == '}') { ++n; ++i; }
    return n;
}

// ---------------------------------------------------------------------------
// Part 1: FP8/FP4 vs BF16 Mega MoE config derivation
// ---------------------------------------------------------------------------
struct MoeShape { const char* name; int ranks, epr, max_tok, topk, hidden, inter, tokens, ring, sf_ring; };

static void part1_moe_configs() {
    std::printf("\n== Part 1: Mega MoE config derivation (get_mega_moe_config) ==\n");
    const MoeShape shapes[] = {
        {"deepseek-v3-like", 8, 32, 4096, 8, 7168, 2048, 4096, 1920, 61440},
        {"inter=1536/topk8", 8, 32, 4096, 8, 7168, 1536, 4096, 1920, 61440},
        {"few-local-experts", 8,  8, 4096, 8, 7168, 2048, 4096, 1920, 61440},
        {"single-rank-small", 1, 32,  256, 4, 7168, 2048,  256, 1920, 61440},
    };
    for (const auto& s : shapes) {
        for (const auto& mma : {MmaKind::MXFP8FP4, MmaKind::BF16}) {
            const int elem = mma == MmaKind::BF16 ? 2 : 1;
            const char* tag = mma == MmaKind::BF16 ? "BF16" : "FP8";
            const auto c = get_mega_moe_config(
                s.ranks, s.epr * s.ranks, s.epr, s.max_tok, s.tokens, s.topk,
                s.hidden, s.inter, s.ring, s.sf_ring, mma);
            std::printf("[%s/%s] %s: %s\n", s.name, tag, tag, ([&]{ std::ostringstream os; os << c; return os.str(); })().c_str());

            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s/%s block_k == 128/elem_bytes", s.name, tag);
            CHECK(c.block_k == 128 / elem, buf);
            std::snprintf(buf, sizeof(buf), "%s/%s block_n == 128", s.name, tag);
            CHECK(c.block_n == 128, buf);
            std::snprintf(buf, sizeof(buf), "%s/%s load_block_m == block_m/2 (multicast A)", s.name, tag);
            CHECK(c.load_block_m == c.block_m / 2 and c.load_block_n == 128, buf);
            std::snprintf(buf, sizeof(buf), "%s/%s thread split 128/128/256", s.name, tag);
            CHECK(c.num_dispatch_threads == 128 and c.num_non_epilogue_threads == 128 and c.num_epilogue_threads == 256, buf);
            std::snprintf(buf, sizeof(buf), "%s/%s swizzle acts/weights == 128", s.name, tag);
            CHECK(c.swizzle_acts_mode == 128 and c.swizzle_weights_mode == 128, buf);
            std::snprintf(buf, sizeof(buf), "%s/%s num_ring_tokens == ring", s.name, tag);
            CHECK(c.num_ring_tokens == s.ring, buf);
            if (mma == MmaKind::BF16) {
                std::snprintf(buf, sizeof(buf), "%s/BF16 has no SF (sf_block_m/n==0, sf_ring==0)", s.name);
                CHECK(c.sf_block_m == 0 and c.sf_block_n == 0 and c.num_sf_ring_tokens == 0, buf);
            } else {
                std::snprintf(buf, sizeof(buf), "%s/FP8 SF blocks UTCCP-aligned (align(block_m,128),128)", s.name);
                CHECK(c.sf_block_m == align(c.block_m, 128) and c.sf_block_n == 128 and c.num_sf_ring_tokens == s.sf_ring, buf);
            }
            // store_block_m_l1 ladder: <=16 -> 8, <=64 -> 16, <=192 -> 32, else fp8=24 / bf16=40
            const int expect_l1 = c.block_m <= 16 ? 8 : c.block_m <= 64 ? 16 : c.block_m <= 192 ? 32 :
                                  (mma == MmaKind::MXFP8FP4 ? 24 : 40);
            const int expect_l2 = (c.block_m == 240 and mma == MmaKind::MXFP8FP4) ? 8 : expect_l1;
            std::snprintf(buf, sizeof(buf), "%s/%s store_block_m ladder (%d/%d)", s.name, tag, expect_l1, expect_l2);
            CHECK(c.store_block_m_l1 == expect_l1 and c.store_block_m_l2 == expect_l2, buf);
            // num_bytes_per_pull: hidden*elem halved until <= 8192(fp8)/4096(bf16)
            int expect_pull = s.hidden * elem;
            const int cap = mma == MmaKind::MXFP8FP4 ? 8192 : 4096;
            while (expect_pull > cap) expect_pull /= 2;
            std::snprintf(buf, sizeof(buf), "%s/%s num_bytes_per_pull == %d", s.name, tag, expect_pull);
            CHECK(c.num_bytes_per_pull == expect_pull, buf);
            std::snprintf(buf, sizeof(buf), "%s/%s smem fits (%d <= 232448) and stages>=2", s.name, tag, c.smem_size);
            CHECK(c.smem_size <= SM100ArchSpec::smem_capacity and c.num_stages >= 2, buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Part 2: descriptor-set composition + shared fallback (real make_tma_*)
// ---------------------------------------------------------------------------
struct Fp8Descs {
    CUtensorMap l1_acts, l1_acts_sf, l1_weights, l1_weights_sf, l1_output;
    CUtensorMap l2_acts, l2_acts_sf, l2_weights, l2_weights_sf;
    CUtensorMap sh_l1_acts, sh_l1_acts_sf, sh_l1_weights, sh_l1_weights_sf, sh_l1_output;
    CUtensorMap sh_l2_acts, sh_l2_acts_sf, sh_l2_weights, sh_l2_weights_sf;
};

// Mirror the exact descriptor block of sm100_fp8_fp4_mega_moe.hpp:50-139.
static Fp8Descs build_fp8_descs(const MegaMoEConfig& config, int num_experts_per_rank,
                                int num_shared_experts, int max_tok, int hidden, int inter,
                                const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
                                const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
                                const torch::Tensor& sh_l1_acts, const torch::Tensor& sh_l1_acts_sf,
                                const torch::Tensor& sh_l2_acts, const torch::Tensor& sh_l2_acts_sf,
                                const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
                                const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
                                const torch::Tensor& sh_l1_weights, const torch::Tensor& sh_l2_weights,
                                const torch::Tensor& sh_l1_weights_sf, const torch::Tensor& sh_l2_weights_sf) {
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const int sh_inter = inter * num_shared_experts;
    Fp8Descs d{};
    d.l1_acts      = make_tma_2d_desc(l1_acts, hidden, config.num_ring_tokens, config.block_k, config.load_block_m, (int)l1_acts.stride(-2), config.swizzle_acts_mode);
    d.l1_acts_sf   = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf, config.num_sf_ring_tokens, hidden, config.sf_block_m, kGranK, 1, 0, 0, false, sf_smem_outer_dim, (int)l1_acts_sf.stride(-1));
    d.l1_weights   = make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    d.l1_weights_sf= make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf, inter * 2, hidden, config.block_n, kGranK, num_experts_per_rank, 0, 0, false, sf_smem_outer_dim, (int)l1_weights_sf.stride(-1));
    d.l1_output    = make_tma_2d_desc(l2_acts, inter, config.num_ring_tokens, config.block_n / 2, config.store_block_m_l1, (int)l2_acts.stride(-2), config.swizzle_acts_mode / 2);
    d.l2_acts      = make_tma_2d_desc(l2_acts, inter, config.num_ring_tokens, config.block_k, config.load_block_m, (int)l2_acts.stride(-2), config.swizzle_acts_mode);
    d.l2_acts_sf   = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf, config.num_sf_ring_tokens, inter, config.sf_block_m, kGranK, 1, 0, 0, false, sf_smem_outer_dim, (int)l2_acts_sf.stride(-1));
    d.l2_weights   = make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    d.l2_weights_sf= make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf, hidden, inter, config.block_n, kGranK, num_experts_per_rank, 0, 0, false, sf_smem_outer_dim, (int)l2_weights_sf.stride(-1));

    const bool has = num_shared_experts > 0;
    d.sh_l1_acts      = has ? make_tma_2d_desc(sh_l1_acts, hidden, max_tok, config.block_k, config.load_block_m, (int)sh_l1_acts.stride(-2), config.swizzle_acts_mode) : d.l1_acts;
    d.sh_l1_acts_sf   = has ? make_tma_sf_desc(cute::UMMA::Major::MN, sh_l1_acts_sf, (int)sh_l1_acts_sf.size(0), hidden, config.sf_block_m, kGranK, 1, 0, 0, false, sf_smem_outer_dim, (int)sh_l1_acts_sf.stride(-1)) : d.l1_acts_sf;
    d.sh_l1_weights   = has ? make_tma_weights_2d_desc(sh_l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : d.l1_weights;
    d.sh_l1_weights_sf= has ? make_tma_sf_desc(cute::UMMA::Major::MN, sh_l1_weights_sf, sh_inter * 2, hidden, config.block_n, kGranK, 1, 0, 0, false, sf_smem_outer_dim, (int)sh_l1_weights_sf.stride(-1)) : d.l1_weights_sf;
    d.sh_l1_output    = has ? make_tma_2d_desc(sh_l2_acts, sh_inter, max_tok, config.block_n / 2, config.store_block_m_l1, (int)sh_l2_acts.stride(-2), config.swizzle_acts_mode / 2) : d.l1_output;
    d.sh_l2_acts      = has ? make_tma_2d_desc(sh_l2_acts, sh_inter, max_tok, config.block_k, config.load_block_m, (int)sh_l2_acts.stride(-2), config.swizzle_acts_mode) : d.l2_acts;
    d.sh_l2_acts_sf   = has ? make_tma_sf_desc(cute::UMMA::Major::MN, sh_l2_acts_sf, (int)sh_l2_acts_sf.size(0), sh_inter, config.sf_block_m, kGranK, 1, 0, 0, false, sf_smem_outer_dim, (int)sh_l2_acts_sf.stride(-1)) : d.l2_acts_sf;
    d.sh_l2_weights   = has ? make_tma_weights_2d_desc(sh_l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : d.l2_weights;
    d.sh_l2_weights_sf= has ? make_tma_sf_desc(cute::UMMA::Major::MN, sh_l2_weights_sf, hidden, sh_inter, config.block_n, kGranK, 1, 0, 0, false, sf_smem_outer_dim, (int)sh_l2_weights_sf.stride(-1)) : d.l2_weights_sf;
    return d;
}

static void part2_descriptors() {
    std::printf("\n== Part 2: FP8 Mega MoE TMA descriptor composition + shared fallback ==\n");
    const int ranks = 8, epr = 32, topk = 8, hidden = 7168, inter = 2048, tokens = 4096, ring = 1920, sf_ring = 61440;
    const auto config = get_mega_moe_config(ranks, epr * ranks, epr, 4096, tokens, topk, hidden, inter, ring, sf_ring, MmaKind::MXFP8FP4);
    const int sf_smem_outer_dim = config.block_k / (32 * 4);
    CHECK(sf_smem_outer_dim == 1, "sf_smem_outer_dim == block_k/(gran_k*4) == 1 for FP8");

    auto opts = torch::TensorOptions().device(torch::kCUDA, 0);
    auto e4m3 = opts.dtype(torch::kFloat8_e4m3fn);
    auto i32 = opts.dtype(torch::kInt32);
    const auto l1_acts = torch::zeros({ring, hidden}, e4m3);
    const auto l2_acts = torch::zeros({ring, inter}, e4m3);
    // SF tensors are MN-major (M contiguous, last-dim stride == M), cf. mega_moe.hpp:104-115.
    const auto l1_acts_sf = torch::empty_strided({sf_ring, hidden / 128}, {1, sf_ring}, i32);
    const auto l2_acts_sf = torch::empty_strided({sf_ring, inter / 128}, {1, sf_ring}, i32);
    const auto l1_w = torch::zeros({epr, inter * 2, hidden}, e4m3);
    const auto l2_w = torch::zeros({epr, hidden, inter}, e4m3);
    // 3D (grouped) weight SF: stride(-2)==1, stride(-1)>=align(mn,4); cf. check_sf_layout layout.hpp:126-129.
    const auto l1_w_sf = torch::empty_strided({epr, inter * 2, hidden / 128}, {align(inter * 2, 4) * (hidden / 128), 1, align(inter * 2, 4)}, i32);
    const auto l2_w_sf = torch::empty_strided({epr, hidden, inter / 128}, {align(hidden, 4) * (inter / 128), 1, align(hidden, 4)}, i32);

    const auto d0 = build_fp8_descs(config, epr, 0, 4096, hidden, inter,
                                    l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                                    l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                                    l1_w, l2_w, l1_w_sf, l2_w_sf, l1_w, l2_w, l1_w_sf, l2_w_sf);
    CHECK_DESC_EQ(d0.sh_l1_acts, d0.l1_acts, "num_shared==0 -> shared_l1_acts aliases routed (byte-equal)");
    CHECK_DESC_EQ(d0.sh_l1_weights, d0.l1_weights, "num_shared==0 -> shared_l1_weights aliases routed");
    CHECK_DESC_EQ(d0.sh_l2_acts_sf, d0.l2_acts_sf, "num_shared==0 -> shared_l2_acts_sf aliases routed");
    CHECK_DESC_EQ(d0.sh_l2_weights, d0.l2_weights, "num_shared==0 -> shared_l2_weights aliases routed");

    // With S=2 shared experts the shared L1 N-axis / L2 K-axis fold in E_s = S.
    const int S = 2, sh_inter = inter * S;
    const int max_tok = 4096;
    const int M_sh = (int)layout::get_num_max_shared_sf_tokens(max_tok);
    const auto sh_l1_acts = torch::zeros({max_tok, hidden}, e4m3);
    const auto sh_l2_acts = torch::zeros({max_tok, sh_inter}, e4m3);
    const auto sh_l1_acts_sf = torch::empty_strided({M_sh, hidden / 128}, {1, M_sh}, i32);
    const auto sh_l2_acts_sf = torch::empty_strided({M_sh, sh_inter / 128}, {1, M_sh}, i32);
    const auto sh_l1_w = torch::zeros({sh_inter * 2, hidden}, e4m3);
    const auto sh_l2_w = torch::zeros({hidden, sh_inter}, e4m3);
    const auto sh_l1_w_sf = torch::empty_strided({sh_inter * 2, hidden / 128}, {1, align(sh_inter * 2, 4)}, i32);
    const auto sh_l2_w_sf = torch::empty_strided({hidden, sh_inter / 128}, {1, align(hidden, 4)}, i32);
    const auto dS = build_fp8_descs(config, epr, S, max_tok, hidden, inter,
                                    l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                                    sh_l1_acts, sh_l1_acts_sf, sh_l2_acts, sh_l2_acts_sf,
                                    l1_w, l2_w, l1_w_sf, l2_w_sf, sh_l1_w, sh_l2_w, sh_l1_w_sf, sh_l2_w_sf);

    // Independent re-encode (2D cases); weights are 4D descriptors already covered by P56/P79.
    {
        const auto indep = indep_encode_tiled(
            CU_TENSOR_MAP_DATA_TYPE_UINT8, l1_acts.data_ptr(),
            hidden, ring, (uint64_t)l1_acts.stride(-2),
            config.block_k, config.load_block_m, CU_TENSOR_MAP_SWIZZLE_128B);
        CHECK_DESC_EQ(dS.l1_acts, indep, "FP8 l1_acts == independent encode (hidden, ring, sw128)");
    }
    {
        const auto indep = indep_encode_tiled(
            CU_TENSOR_MAP_DATA_TYPE_UINT8, sh_l2_acts.data_ptr(),
            sh_inter, max_tok, (uint64_t)sh_l2_acts.stride(-2),
            config.block_k, config.load_block_m, CU_TENSOR_MAP_SWIZZLE_128B);
        CHECK_DESC_EQ(dS.sh_l2_acts, indep, "FP8 shared_l2_acts == independent encode (sh_inter, T)");
    }
    {
        // SF descriptor: block dim 0 == sf_block_m (UTCCP-aligned), outer == hidden/128, no swizzle.
        const int aligned_mn = align(sf_ring, 4);
        const auto indep = indep_encode_tiled(
            CU_TENSOR_MAP_DATA_TYPE_INT32, l1_acts_sf.data_ptr(),
            aligned_mn, hidden / (32 * 4), (uint64_t)l1_acts_sf.stride(-1) * 4,
            config.sf_block_m, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
        CHECK_DESC_EQ(dS.l1_acts_sf, indep, "FP8 l1_acts_sf == independent encode (M-major, no swizzle)");
    }

    // FP8 wrapper has 18 descriptors (9 routed + 9 shared); BF16 has 10 (5 routed + 5 shared).
    std::printf("  INFO  FP8 descriptor count = 18 (9 routed + 9 shared); BF16 = 10 (5 + 5)\n");
    const int fp8_placeholders = count_format_placeholders("~/github/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp");
    const int bf16_placeholders = count_format_placeholders("~/github/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp");
    const int gate_placeholders = count_format_placeholders("~/github/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp");
    char buf[256];
    std::snprintf(buf, sizeof(buf), "FP8 template arity from source == 25 (got %d)", fp8_placeholders);
    CHECK(fp8_placeholders == 25, buf);
    std::snprintf(buf, sizeof(buf), "BF16 template arity from source == 20 (got %d)", bf16_placeholders);
    CHECK(bf16_placeholders == 20, buf);
    std::snprintf(buf, sizeof(buf), "Gate template arity from source == 15 (got %d)", gate_placeholders);
    CHECK(gate_placeholders == 15, buf);
}

// ---------------------------------------------------------------------------
// Part 3: Mega Gate config + constants + RoutingArgs ABI
// ---------------------------------------------------------------------------
static void part3_gate() {
    std::printf("\n== Part 3: Mega Gate config / constants / RoutingArgs ABI ==\n");
    CHECK(mgl::BLOCK_K == 64 and mgl::kSwizzleMode == 128, "gate BLOCK_K==64, kSwizzleMode==128");
    CHECK(mgl::kNumNonEpilogueThreads == 128 and mgl::kNumMaxBlockTokens == 256 and mgl::kNumMinBlockTokens == 16,
          "gate thread/blocks constants (128/256/16)");
    CHECK(mgl::kExpertAlignment == 128 and mgl::kNumMaxLogicalCtas == 64 and mgl::kNumMaxStages == 32,
          "gate alignment/logical-ctas/stages constants (128/64/32)");
    CHECK(mgl::kNumExpertsPerWave == mgl::kExpertAlignment, "gate kNumExpertsPerWave == kExpertAlignment");

    // RoutingArgs ABI
    CHECK(sizeof(mgl::RoutingArgs) == 72, "sizeof(RoutingArgs) == 72");
    CHECK(offsetof(mgl::RoutingArgs, to_physical_map) == 0 and offsetof(mgl::RoutingArgs, unmapped_topk_idx_stride) == 40,
          "RoutingArgs ptr block @0..40");
    CHECK(offsetof(mgl::RoutingArgs, num_routed_experts) == 48 and offsetof(mgl::RoutingArgs, routed_scaling_factor) == 64,
          "RoutingArgs int/uint block @48..64");

    // SharedStorage / Workspace
    std::printf("  INFO  sizeof(SharedStorage)=%zu kStageDataOffset=%u\n",
                sizeof(mgl::SharedStorage), mgl::kStageDataOffset);
    CHECK(mgl::kStageDataOffset % mgl::kSharedMemoryAlignment == 0, "kStageDataOffset 1024-aligned");

    const int blocks = 16, splits = 8, block_tokens = 256, aligned_experts = 256;
    const auto scratch = mgl::Workspace<>::get_num_scratch_bytes(blocks, splits, block_tokens, aligned_experts);
    const uint64_t expect_scratch = (uint64_t)blocks * splits * block_tokens * aligned_experts * sizeof(float);
    CHECK(scratch == expect_scratch, "Workspace scratch bytes == nb*split*block_tokens*E*4");

    // caches_logical_count truth table
    CHECK(mgl::caches_logical_count(256, false, false, true) == true, "caches_logical_count(no bias,map)=true");
    CHECK(mgl::caches_logical_count(256, true, true, true) ==
          (256 * 2 * 4 + 256 * 4 <= (int)mgl::kNumMaxMetadataCacheBytes),
          "caches_logical_count(bias+image,map) matches 4KiB budget");
    CHECK(mgl::caches_logical_count(256, false, false, false) == false, "caches_logical_count(no map)=false");

    // Config across shapes, deterministic off vs on
    struct GateShape { const char* name; int tokens, hidden, experts, sms; };
    const GateShape gs[] = {
        {"T=4096,H=7168,E=256", 4096, 7168, 256, 132},
        {"T=64,H=7168,E=256",     64, 7168, 256, 132},
        {"T=32768,H=4096,E=128", 32768, 4096, 128, 132},
        {"T=1024,H=7168,E=384",  1024, 7168, 384, 132},
    };
    for (const auto& g : gs) {
        for (const bool det : {false, true}) {
            heuristics_runtime->use_deterministic_algorithms(det);
            const auto c = get_mega_gate_config(g.tokens, g.hidden, g.experts, g.sms);
            std::printf("[gate/%s det=%d] %s\n", g.name, (int)det, ([&]{ std::ostringstream os; os << c; return os.str(); })().c_str());
            const int aligned = align(g.experts, 128);
            const int nepg = aligned / c.num_expert_groups;
            char buf[256];
            std::snprintf(buf, sizeof(buf), "gate/%s det=%d load blocks (%d/%d)", g.name, (int)det, c.load_block_m, c.load_block_n);
            CHECK(c.load_block_m == c.block_tokens / c.num_mma_ctas and c.load_block_n == nepg / c.num_mma_ctas, buf);
            const int ntb = ceil_div(g.tokens, c.block_tokens);
            std::snprintf(buf, sizeof(buf), "gate/%s det=%d launch sms/waves", g.name, (int)det);
            CHECK(c.num_launch_sms == std::min(c.num_worker_groups, ntb) * c.num_logical_ctas and
                  c.num_waves == ceil_div(ntb, c.num_worker_groups), buf);
            std::snprintf(buf, sizeof(buf), "gate/%s det=%d gate threads multiple of 128 and <=7 warpgroup", g.name, (int)det);
            CHECK(c.num_gate_threads % 128 == 0 and c.num_gate_threads <= 7 * 128, buf);
            std::snprintf(buf, sizeof(buf), "gate/%s det=%d smem/stages", g.name, (int)det);
            CHECK(c.smem_size <= SM100ArchSpec::smem_capacity and c.num_stages >= 2, buf);
            std::snprintf(buf, sizeof(buf), "gate/%s det=%d split_k==1 under determinism", g.name, (int)det);
            CHECK(not det or c.num_split_k == 1, buf);
        }
    }
    heuristics_runtime->use_deterministic_algorithms(false);
}

int main() {
    std::printf("== P86-1: FP8 Mega MoE + Mega Gate JIT launch wrappers (host side) ==\n");
    part1_moe_configs();
    part2_descriptors();
    part3_gate();
    std::printf("\n==== TOTAL: PASS=%d FAIL=%d ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
