// P93-1 experiment: H100(sm90) verification of the SM100 BF16 `kSwapAB` (dgrad) branch's
// HOST half by directly including the real DeepGEMM heuristics headers.
//   csrc/jit_kernels/heuristics/sm100.hpp   (SM100ArchSpec::get_layout_candidates / storage / pipeline / launch)
//   csrc/jit_kernels/heuristics/sm90.hpp    (SM90ArchSpec, the no-swap contrast)
//   csrc/jit_kernels/heuristics/{common,config,runtime}.hpp
//
// Everything exercised here is pure host C++ + torch: layout selection, storage/pipeline/launch
// derivation, and the "swap == transpose" numerical identity. The SM100 device kernel
// (UMMA / TMEM / transposed STSM) stays uninstantiated and is analysed only in the doc.
//
// Build/run: harness/exp_sm100_swap_ab_dgrad.sh

#include <cassert>
// In an nvcc single-TU host pass the real headers' DG_UNIFIED_ASSERT falls through to
// asm("trap;"), which the host assembler rejects. Pin it to the host assert first.
#ifndef DG_UNIFIED_ASSERT
#define DG_UNIFIED_ASSERT(cond) assert(cond)
#endif

#include <torch/extension.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#include <deep_jit/backend/cuda/driver.hpp>

#include <csrc/jit_kernels/heuristics/sm100.hpp>
#include <csrc/jit_kernels/heuristics/sm90.hpp>

using namespace deep_gemm;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { ++g_pass; std::printf("  PASS  %s\n", msg); }               \
        else      { ++g_fail; std::printf("  FAIL  %s\n", msg); }               \
    } while (0)

#define CHECK_EQ(a, b, msg)                                                     \
    do {                                                                        \
        auto va = (a); auto vb = (b);                                           \
        if (va == vb) { ++g_pass; std::printf("  PASS  %s (%ld)\n", msg, (long) va); } \
        else { ++g_fail; std::printf("  FAIL  %s: got %ld expected %ld\n", msg, (long) va, (long) vb); } \
    } while (0)

// ---------------------------------------------------------------------------------------------
// A m-grouped contiguous BF16 descriptor, matching impls/sm100_bf16_gemm.hpp:
// sm100_m_grouped_bf16_gemm_contiguous.
// ---------------------------------------------------------------------------------------------
static GemmDesc make_mgrouped_bf16(int m, int n, int k, int num_groups,
                                   GemmType gemm_type = GemmType::MGroupedContiguous) {
    return GemmDesc{
        .gemm_type = gemm_type,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = torch::kBFloat16, .b_dtype = torch::kBFloat16, .cd_dtype = torch::kBFloat16,
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = 132, .tc_util = 100,
        .compiled_dims = "nk",
    };
}

static GemmDesc make_dense_bf16(int m, int n, int k) {
    return GemmDesc{
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = torch::kBFloat16, .b_dtype = torch::kBFloat16, .cd_dtype = torch::kBFloat16,
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = 132, .tc_util = 100,
        .compiled_dims = "nk",
    };
}

static int expect_cluster_n(int n) {
    // heuristics/sm100.hpp:42
    return (ceil_div(n, 128) % 2 == 0 and 132 % 2 == 0) ? 2 : 1;
}

// =============================================================================================
int main() {
    std::printf("== SM100 kSwapAB (dgrad) host-layer experiment (H100 / sm_90a) ==\n");
    (void) heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    std::printf("default mk_alignment_for_contiguous_layout = %d\n",
                heuristics_runtime->get_mk_alignment_for_contiguous_layout());
    std::printf("default block_m_multiple_of = %d, block_n_multiple_of = %d\n",
                heuristics_runtime->get_block_m_multiple_of(),
                heuristics_runtime->get_block_n_multiple_of());

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part A: m-grouped -> unique swap_ab layout, block_n == 128 ==\n");
    {
        struct Case { int m, n, k, G; };
        std::vector<Case> cases = {
            {4096, 7168, 2048, 8},   // DeepSeek-ish dgrad: 56 block-n -> even -> cluster_n=2
            {2048, 7296, 2048, 8},   // 57 block-n -> odd -> cluster_n=1
            {1024, 1024, 1024, 4},   // tiny square
        };
        for (const auto& c : cases) {
            auto desc = make_mgrouped_bf16(c.m, c.n, c.k, c.G);
            const auto cands = SM100ArchSpec::get_layout_candidates(desc);
            char buf[160];
            std::snprintf(buf, sizeof(buf), "m=%d n=%d k=%d G=%d: exactly one candidate", c.m, c.n, c.k, c.G);
            CHECK_EQ((long) cands.size(), 1, buf);
            const auto& L = cands[0];
            std::snprintf(buf, sizeof(buf), "  swap_ab==1 (m=%d n=%d)", c.m, c.n);
            CHECK_EQ(L.swap_ab, 1, buf);
            std::snprintf(buf, sizeof(buf), "  block_n==128 (m=%d n=%d)", c.m, c.n);
            CHECK_EQ(L.block_n, 128, buf);
            std::snprintf(buf, sizeof(buf), "  block_k==64 (BF16) (m=%d n=%d)", c.m, c.n);
            CHECK_EQ(L.block_k, 64, buf);
            std::snprintf(buf, sizeof(buf), "  cluster_m==1 (m=%d n=%d)", c.m, c.n);
            CHECK_EQ(L.cluster_m, 1, buf);
            std::snprintf(buf, sizeof(buf), "  cluster_n == formula (m=%d n=%d)", c.m, c.n);
            CHECK_EQ(L.cluster_n, expect_cluster_n(c.n), buf);
            std::snprintf(buf, sizeof(buf), "  block_m == mk_alignment (m=%d n=%d)", c.m, c.n);
            CHECK_EQ(L.block_m, heuristics_runtime->get_mk_alignment_for_contiguous_layout(), buf);
        }
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part A2: block_m tracks mk_alignment_for_contiguous_layout (128 vs 256) ==\n");
    {
        auto desc = make_mgrouped_bf16(4096, 7168, 2048, 8);
        for (int al : {128, 256}) {
            heuristics_runtime->set_mk_alignment_for_contiguous_layout(al);
            const auto cands = SM100ArchSpec::get_layout_candidates(desc);
            char buf[128];
            std::snprintf(buf, sizeof(buf), "mk_alignment=%d -> block_m=%d", al, al);
            CHECK_EQ(cands[0].block_m, al, buf);
            const auto sc = SM100ArchSpec::get_storage_config(desc, cands[0]);
            std::snprintf(buf, sizeof(buf), "  load_block_m = block_m/cluster_n = %d", al / cands[0].cluster_n);
            CHECK_EQ(sc.load_block_m, al / cands[0].cluster_n, buf);
            std::snprintf(buf, sizeof(buf), "  store_block_m == 16 (UMMA_STEP_N) for mk=%d", al);
            CHECK_EQ(sc.store_block_m, 16, buf);
            std::snprintf(buf, sizeof(buf), "  store_block_n == 128 for mk=%d", al);
            CHECK_EQ(sc.store_block_n, 128, buf);
        }
        heuristics_runtime->set_mk_alignment_for_contiguous_layout(128);
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part A3: mask / psum m-grouped also force swap_ab ==\n");
    {
        for (auto gt : {GemmType::MGroupedMasked, GemmType::MGroupedContiguousWithPsumLayout}) {
            auto desc = make_mgrouped_bf16(4096, 7168, 2048, 8, gt);
            const auto cands = SM100ArchSpec::get_layout_candidates(desc);
            char buf[128];
            std::snprintf(buf, sizeof(buf), "gemm_type=%d -> swap_ab==1, block_n==128",
                          static_cast<int>(gt));
            CHECK(cands.size() == 1 and cands[0].swap_ab == 1 and cands[0].block_n == 128, buf);
        }
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part B: storage / pipeline / launch derivations ==\n");
    {
        auto desc = make_mgrouped_bf16(4096, 7168, 2048, 8);
        const auto cands = SM100ArchSpec::get_layout_candidates(desc);
        const auto& L = cands[0];
        const auto sc = SM100ArchSpec::get_storage_config(desc, L);
        const auto pc = SM100ArchSpec::get_pipeline_config(desc, L, sc);
        const auto lc = SM100ArchSpec::get_launch_config(desc, L);

        CHECK_EQ(sc.swizzle_a_mode, 128, "swizzle_a == 128 (K-major, block_k*2 bytes)");
        CHECK_EQ(sc.swizzle_b_mode, 128, "swizzle_b == 128");
        CHECK_EQ(sc.swizzle_cd_mode, 128, "swizzle_cd == 128 (128B swizzled D)");
        CHECK_EQ(sc.load_block_n, 128, "load_block_n == 128 (cluster_m==1)");

        // smem_cd = store_block_m * store_block_n * sizeof(bf16) * num_tma_store_stages
        const int num_tma_store_stages = SM100ArchSpec::get_num_tma_store_stages(desc, L);
        CHECK_EQ(num_tma_store_stages, 2, "BF16 -> num_tma_store_stages == 2");
        const int smem_cd = 16 * 128 * 2 * num_tma_store_stages;
        CHECK_EQ(smem_cd, 8192, "smem_cd == 16*128*2*2 == 8192");
        // smem_extra = smem_cd + 824 barriers + 4 tmem ptr
        const int smem_extra = smem_cd + (32 * 8 * 3 + 2 * 8 * 3 + 8) + 4;
        CHECK_EQ(smem_extra, 9020, "smem_extra == 9020");
        // smem_per_stage = load_block_m*block_k*2 + load_block_n*block_k*2
        const int smem_per_stage = sc.load_block_m * L.block_k * 2 + sc.load_block_n * L.block_k * 2;
        CHECK_EQ(smem_per_stage, 64 * 64 * 2 + 128 * 64 * 2, "smem_per_stage == 24576");
        const int want_stages = std::min((232448 - smem_extra) / smem_per_stage, 32);
        CHECK_EQ(pc.num_stages, want_stages, "num_stages == (232448-extra)/per_stage");
        CHECK_EQ(pc.smem_size, smem_extra + want_stages * smem_per_stage, "smem_size == extra + stages*per_stage");

        CHECK_EQ(lc.num_sms, 132, "launch.num_sms");
        CHECK_EQ(lc.num_sms_per_cluster, 2, "launch.cluster_size == 2");
        CHECK_EQ(lc.num_threads, 256, "launch.num_threads == 256");
        CHECK_EQ(lc.num_non_epilogue_threads, 128, "launch.num_non_epilogue_threads == 128");
        CHECK_EQ(lc.num_epilogue_threads, 128, "launch.num_epilogue_threads == 128 (=kNumUMMAStoreThreads)");
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part B2: mk_alignment=256 changes smem budget (fewer stages) ==\n");
    {
        auto desc = make_mgrouped_bf16(4096, 7168, 2048, 8);
        heuristics_runtime->set_mk_alignment_for_contiguous_layout(256);
        const auto L = SM100ArchSpec::get_layout_candidates(desc)[0];
        const auto sc = SM100ArchSpec::get_storage_config(desc, L);
        const auto pc = SM100ArchSpec::get_pipeline_config(desc, L, sc);
        CHECK_EQ(sc.load_block_m, 128, "mk=256 -> load_block_m == 128");
        const int smem_per_stage = sc.load_block_m * L.block_k * 2 + sc.load_block_n * L.block_k * 2;
        const int want_stages = std::min((232448 - 9020) / smem_per_stage, 32);
        CHECK_EQ(pc.num_stages, want_stages, "mk=256 -> fewer stages (larger smem/stage)");
        std::printf("  (mk=256: num_stages=%d, smem_size=%d)\n", pc.num_stages, pc.smem_size);
        heuristics_runtime->set_mk_alignment_for_contiguous_layout(128);
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part B3: dense GEMM enumerates swap_ab in {0,1} (contrast) ==\n");
    {
        auto desc = make_dense_bf16(4096, 7168, 2048);
        const auto cands = SM100ArchSpec::get_layout_candidates(desc);
        bool has0 = false, has1 = false;
        for (const auto& L : cands) { if (L.swap_ab == 0) has0 = true; if (L.swap_ab == 1) has1 = true; }
        std::printf("  dense candidates = %zu\n", cands.size());
        CHECK(has0, "dense: contains a swap_ab==0 candidate");
        CHECK(has1, "dense: contains a swap_ab==1 candidate");
        // swap_ab==1 candidates must have block_n==128 (heuristics/sm100.hpp:121-123)
        bool all_ok = true;
        for (const auto& L : cands) if (L.swap_ab == 1 and L.block_n != 128) all_ok = false;
        CHECK(all_ok, "dense: every swap_ab==1 candidate has block_n == 128");
        // and cluster_m==1 for swap_ab==1
        bool cm_ok = true;
        for (const auto& L : cands) if (L.swap_ab == 1 and L.cluster_m != 1) cm_ok = false;
        CHECK(cm_ok, "dense: every swap_ab==1 candidate has cluster_m == 1");
        const auto best = get_best_config<SM100ArchSpec>(desc);
        std::printf("  dense best layout: swap_ab=%d block_m=%d block_n=%d cluster=(%d,%d)\n",
                    best.layout.swap_ab, best.layout.block_m, best.layout.block_n,
                    best.layout.cluster_m, best.layout.cluster_n);
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part C: swap == transpose (numerical, real torch) ==\n");
    {
        const bool cuda = torch::cuda::is_available();
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(cuda ? torch::kCUDA : torch::kCPU);
        std::printf("  torch device = %s\n", cuda ? "cuda" : "cpu");

        const int M = 257, N = 129, K = 96;   // deliberately non-power-of-two
        auto a = torch::randn({M, K}, opts);
        auto b = torch::randn({N, K}, opts);
        auto lhs = a.matmul(b.transpose(0, 1)).transpose(0, 1);   // (A @ B^T)^T
        auto rhs = b.matmul(a.transpose(0, 1));                   // B @ A^T
        CHECK(torch::equal(lhs, rhs), "dense: (A @ B^T)^T == B @ A^T (bitwise)");

        // grouped form: per-expert D_e = A_e @ B_e^T, verify (D_e)^T == B_e @ A_e^T.
        // me varies per expert, so cuBLAS may pick different tilings -> compare with a
        // tolerance (the dense case above already showed bitwise equality at a fixed shape).
        const int G = 4;
        double max_diff = 0.0;
        for (int e = 0; e < G; ++e) {
            int me = 32 + e * 37;
            auto ae = torch::randn({me, K}, opts);
            auto be = torch::randn({N, K}, opts);
            auto de_t = ae.matmul(be.transpose(0, 1)).transpose(0, 1);
            auto de_t_alt = be.matmul(ae.transpose(0, 1));
            max_diff = std::max(max_diff, (de_t - de_t_alt).abs().max().item<double>());
        }
        std::printf("  grouped max|diff| = %.3e\n", max_diff);
        CHECK(max_diff < 1e-4, "grouped: per-expert (A_e @ B_e^T)^T == B_e @ A_e^T (fp32 tolerance)");

        // and the shapes confirm the roles: D is [M,N] while the swap form is produced as [N,M]
        std::printf("  D shape (M,N) = (%d,%d); swap computes the (N,M) operand then transposes\n", M, N);
    }

    // ---------------------------------------------------------------------------------------
    std::printf("\n== Part D: SM90 has no swap_ab (contrast) ==\n");
    {
        auto desc = make_mgrouped_bf16(4096, 7168, 2048, 8);
        const auto cands = SM90ArchSpec::get_layout_candidates(desc);
        bool all_zero = true;
        for (const auto& L : cands) if (L.swap_ab != 0) all_zero = false;
        std::printf("  SM90 m-grouped candidates = %zu\n", cands.size());
        CHECK(all_zero, "SM90: every candidate has swap_ab == 0 (heuristics/sm90.hpp:125 asserts it)");
        CHECK(cands[0].block_m == heuristics_runtime->get_mk_alignment_for_contiguous_layout(),
              "SM90: m-grouped block_m also tracks mk_alignment");
    }

    std::printf("\n== SUMMARY: %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
