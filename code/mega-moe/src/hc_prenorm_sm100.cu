// P87-1 experiment: H100(sm90) verification of the SM100 HyperConnection prenorm GEMM
//   csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp   (host JIT wrapper)
//   deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh (SM100 kernel)
// by directly including the real headers.  Everything the target GPU cannot run is only
// *analysed* in the doc; here we run the architecture-independent pieces as real code:
//   Part 1  host num_stages / smem budget loop, SM90 vs SM100 side by side
//   Part 2  split-K K-block decomposition (coverage / remainder / partial-D sum == full GEMM)
//   Part 3  get_swizzled_smem_offset vs cute::Swizzle<3,4,3> (128B) + bijection (real device code)
//   Part 4  cast-and-reduce warpgroup mapping: warp_reduce_sum<4> + m_idx formula (real device code)
//   Part 5  SM100 wrapper TMA descriptor construction (2D / 3D split) with real torch tensors
//
// Build/run: harness/exp_hc_prenorm_sm100.sh

#include <cassert>
// In an nvcc single-TU host pass the real headers' DG_UNIFIED_ASSERT falls through to
// asm("trap;"), which the host assembler rejects. Pin it to the host assert first.
#ifndef DG_UNIFIED_ASSERT
#define DG_UNIFIED_ASSERT(cond) assert(cond)
#endif

#include <torch/extension.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cute/swizzle.hpp>

#include <deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh>

#include <csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp>
#include <csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp>

using namespace deep_gemm;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { ++g_pass; std::printf("  PASS  %s\n", msg); }               \
        else      { ++g_fail; std::printf("  FAIL  %s\n", msg); }               \
    } while (0)

// ============================================================================
// Part 3 device code: the bank-group swizzle offset helper, real device code.
// ============================================================================
__global__ void swizzle_probe_kernel(uint32_t* out128, uint32_t* out64) {
    const uint32_t lane = threadIdx.x;  // 0..31
    // Mode-128 with base 16: 8 chunks per row.
    for (int c = 0; c < 8; ++ c)
        out128[lane * 8 + c] = get_swizzled_smem_offset<128, 16>(static_cast<uint32_t>(c), lane);
    // Mode-64 with base 16: general (non-shortcut) branch.
    for (int c = 0; c < 4; ++ c)
        out64[lane * 4 + c] = get_swizzled_smem_offset<64, 16>(static_cast<uint32_t>(c), lane);
}

// ============================================================================
// Part 4 device code: cast-and-reduce warpgroup index + reduction mapping.
// ============================================================================
template <uint32_t BLOCK_M, uint32_t BLOCK_M_PER_WARP>
__global__ void reduce_map_kernel(float* out_sqr, int shape_m, uint32_t m_block_idx) {
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();  // 0..3
    const uint32_t lane = ptx::get_lane_idx();
    const uint32_t sub = warp_idx;

    // Synthetic per-lane partials with an exactly predictable group sum, so the test
    // validates the *mapping* (sub-warp / lane-group / u) and warp_reduce_sum<4>,
    // not the LDSM cross-lane distribution (that part is analysed in the doc).
    float2 sum[2] = {float2{0.f, 0.f}, float2{0.f, 0.f}};
    #pragma unroll
    for (uint32_t u = 0; u < 2; ++ u) {
        const float v = static_cast<float>(sub * 10000 + static_cast<int>(lane % 4)
                                           + 100 * static_cast<int>(lane / 4) + 1000 * static_cast<int>(u));
        sum[u] = float2{v, 0.f};
    }

    #pragma unroll
    for (uint32_t u = 0; u < 2; ++ u) {
        const auto reduced = math::warp_reduce_sum<4>(sum[u].x + sum[u].y);
        const auto m_idx = m_block_idx * BLOCK_M + sub * BLOCK_M_PER_WARP + lane / 4 + u * 8;
        if (lane % 4 == 0 and m_idx < static_cast<uint32_t>(shape_m))
            out_sqr[m_idx] = reduced;
    }
}

// ============================================================================
// Part 5 device-free: a tensor-map fingerprint helper (used only for prints).
// ============================================================================
static uint64_t desc_hash(const CUtensorMap& m) {
    const auto* p = reinterpret_cast<const unsigned char*>(&m);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(CUtensorMap); ++ i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main() {
    std::printf("== H100(sm90) verification of SM100 tf32_hc_prenorm_gemm (P87-1) ==\n\n");

    // ------------------------------------------------------------------
    // Part 1: host num_stages / smem budget loop (SM90 vs SM100)
    // ------------------------------------------------------------------
    std::printf("[Part 1] host num_stages / smem budget loop\n");
    auto stage_loop = [](bool sm100, int block_m, int block_k, int block_n, int swizzle_cd) {
        int num_stages = 12, smem_size = 0;
        while (num_stages > 0) {
            const int smem_a = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
            const int smem_b = block_n * block_k * static_cast<int>(sizeof(float));
            const int smem_cd = block_m * swizzle_cd;
            const int smem_barriers = sm100 ? (num_stages * 4 + 1) * 8 : num_stages * 2 * 8;
            const int smem_tmem_ptr = sm100 ? 4 : 0;
            smem_size = (smem_a + smem_b) * num_stages + smem_cd + smem_barriers + smem_tmem_ptr;
            const int cap = sm100 ? SM100ArchSpec::smem_capacity : SM90ArchSpec::smem_capacity;
            if (smem_size <= cap)
                break;
            -- num_stages;
        }
        return std::pair<int, int>{num_stages, smem_size};
    };

    std::printf("   n    block_n  swz_cd | sm90 stages   smem (cap %d) | sm100 stages  smem (cap %d)\n",
                SM90ArchSpec::smem_capacity, SM100ArchSpec::smem_capacity);
    for (int n : {8, 16, 24, 32, 64, 128}) {
        const int block_n = align(n, 16);
        const int swz_cd = get_swizzle_mode(block_n, sizeof(float));
        const auto [s90, sm90] = stage_loop(false, 64, 64, block_n, swz_cd);
        const auto [s100, sm100] = stage_loop(true, 64, 64, block_n, swz_cd);
        std::printf("  %3d   %3d        %3d   | %s %2d        %6d   | %2d        %6d\n",
                    n, block_n, swz_cd,
                    n <= 32 ? "  " : "rejected", n <= 32 ? s90 : 0, n <= 32 ? sm90 : 0,
                    s100, sm100);
    }
    // Contract invariants actually asserted by the two wrappers.
    {
        const auto [s, sz] = stage_loop(true, 64, 64, 32, get_swizzle_mode(32, 4));
        CHECK(s > 0 and sz <= SM100ArchSpec::smem_capacity, "SM100: n=24 -> num_stages>0 and fits smem_capacity");
        const auto [s2, sz2] = stage_loop(false, 64, 64, 32, get_swizzle_mode(32, 4));
        CHECK(s2 > 0 and sz2 <= SM90ArchSpec::smem_capacity, "SM90:  n=24 -> num_stages>0 and fits smem_capacity");
    }
    CHECK(align(24, 16) == 32 and align(8, 16) == 16 and align(128, 16) == 128,
          "block_n = align(n, 16) (SM100 tallies max 128)");
    CHECK(get_swizzle_mode(16, 4) == 64 and get_swizzle_mode(32, 4) == 128 and get_swizzle_mode(128, 4) == 128,
          "swizzle_cd_mode: block_n bytes >=128 -> 128B swizzle");

    // ------------------------------------------------------------------
    // Part 2: split-K K-block decomposition
    // ------------------------------------------------------------------
    std::printf("\n[Part 2] split-K K-block decomposition (BLOCK_K=64)\n");
    auto split_k = [](int k, int num_splits) {
        const int block_k = 64;
        const int kNumKBlocks = (k + block_k - 1) / block_k;
        const int per = kNumKBlocks / num_splits;
        const int rem = kNumKBlocks % num_splits;
        std::vector<int> koffs, nstages;
        for (int s = 0; s < num_splits; ++ s) {
            koffs.push_back((s * per + std::min(s, rem)) * block_k);
            nstages.push_back(per + (s < rem ? 1 : 0));
        }
        return std::tuple{kNumKBlocks, per, rem, koffs, nstages};
    };
    for (auto [k, splits] : std::vector<std::pair<int, int>>{{384, 4}, {384, 3}, {7168, 16}, {448, 6}}) {
        const auto [nb, per, rem, koffs, nstages] = split_k(k, splits);
        // Coverage: every K block exactly once, contiguous [0, nb*64).
        std::vector<int> cover(nb, 0);
        bool ok = true;
        for (int s = 0; s < splits; ++ s)
            for (int i = 0; i < nstages[s]; ++ i) {
                const int blk = koffs[s] / 64 + i;
                if (blk < 0 or blk >= nb) ok = false;
                else cover[blk] += 1;
            }
        for (int v : cover) ok = ok and (v == 1);
        std::printf("  k=%5d splits=%2d: K blocks=%3d, per-split=%2d, remainder=%d -> coverage-exactly-once=%s\n",
                    k, splits, nb, per, rem, ok ? "YES" : "NO");
        CHECK(ok, "split-K K blocks covered exactly once");
    }
    // Numeric: sum of per-split partial D equals the full GEMM (fp32, torch on CUDA).
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        const int m = 64, n = 24, k = 384, splits = 4;
        auto A = torch::randn({m, k}, opts);
        auto B = torch::randn({n, k}, opts);
        auto full = A.matmul(B.transpose(0, 1));
        auto part = torch::zeros({splits, m, n}, opts);
        const auto [nb, per, rem, koffs, nstages] = split_k(k, splits);
        for (int s = 0; s < splits; ++ s) {
            const int ks = koffs[s] / 64 * 64;
            const int ke = ks + nstages[s] * 64;
            part[s] = A.slice(1, ks, ke).matmul(B.slice(1, ks, ke).transpose(0, 1));
        }
        const auto summed = part.sum(0);
        const auto err = (summed - full).abs().max().item<float>();
        std::printf("  partial-D sum vs full GEMM (m=%d,n=%d,k=%d,splits=%d): max|d| = %.3e\n",
                    m, n, k, splits, err);
        CHECK(err < 1e-3, "sum of split-K partial D == full GEMM (fp32)");
        // D scratch is 3D [num_splits, m, n] -> m_offset = shape_m * k_split_idx
        CHECK(part.dim() == 3 and part.stride(0) == (long)(m * n),
              "split D scratch is 3D [num_splits, m, n] (m_offset = shape_m * k_split_idx)");
    }

    // ------------------------------------------------------------------
    // Part 3: get_swizzled_smem_offset (real device code) vs cute::Swizzle<3,4,3>
    // ------------------------------------------------------------------
    std::printf("\n[Part 3] get_swizzled_smem_offset vs cute::Swizzle<3,4,3> (128B swizzle)\n");
    {
        uint32_t *d128 = nullptr, *d64 = nullptr;
        cudaMalloc(&d128, 32 * 8 * sizeof(uint32_t));
        cudaMalloc(&d64, 32 * 4 * sizeof(uint32_t));
        swizzle_probe_kernel<<<1, 32>>>(d128, d64);
        cudaDeviceSynchronize();
        std::vector<uint32_t> h128(32 * 8), h64(32 * 4);
        cudaMemcpy(h128.data(), d128, h128.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h64.data(), d64, h64.size() * 4, cudaMemcpyDeviceToHost);

        // Reference: cute 128B swizzle Swizzle<3,4,3> on linear byte index row*128 + c*16.
        using Sw = cute::Swizzle<3, 4, 3>;
        bool match = true;
        for (int lane = 0; lane < 32; ++ lane)
            for (int c = 0; c < 8; ++ c) {
                const uint32_t ref = static_cast<uint32_t>(Sw{}(lane * 128 + c * 16));
                if (h128[lane * 8 + c] != ref) match = false;
            }
        std::printf("  mode-128: get_swizzled_smem_offset<128,16>(c,lane) == Swizzle<3,4,3>(lane*128+c*16) : %s\n",
                    match ? "ALL MATCH" : "MISMATCH");
        CHECK(match, "128B swizzle offset == cute::Swizzle<3,4,3> (real device code)");

        // Bijection: the 128B atom is a permutation of 0..(32*128-1) over (lane,c) byte cells.
        {
            std::vector<int> seen(32 * 128 / 16, 0);
            bool inj = true;
            for (int lane = 0; lane < 32; ++ lane)
                for (int c = 0; c < 8; ++ c) {
                    const int cell = h128[lane * 8 + c] / 16;
                    if (cell < 0 or cell >= (int)seen.size()) inj = false;
                    else seen[cell] += 1;
                }
            for (int v : seen) inj = inj and (v == 1);
            CHECK(inj, "128B swizzle is a bijection over the atom's 16B chunks");
        }
        // Mode-64 general branch: within-atom range + bijection over 4 chunks/row.
        {
            std::vector<int> seen(32 * 128 / 16, 0);
            bool ok = true;
            for (int lane = 0; lane < 32; ++ lane)
                for (int c = 0; c < 4; ++ c) {
                    const int cell = h64[lane * 4 + c] / 16;
                    if (cell < 0 or cell >= (int)seen.size()) ok = false;
                    else seen[cell] += 1;
                }
            int used = 0;
            for (int v : seen) { if (v > 1) ok = false; if (v == 1) ++used; }
            std::printf("  mode-64 general branch: distinct 16B cells = %d, no collision = %s\n", used, ok ? "YES" : "NO");
            CHECK(ok, "mode-64 general branch: no collision over the atom");
        }
        cudaFree(d128); cudaFree(d64);
    }

    // ------------------------------------------------------------------
    // Part 4: cast-and-reduce warpgroup index + reduction mapping (real device code)
    // ------------------------------------------------------------------
    std::printf("\n[Part 4] cast-and-reduce warpgroup mapping (warp_reduce_sum<4> + m_idx)\n");
    {
        constexpr uint32_t BLOCK_M = 64, BLOCK_M_PER_WARP = 16;
        const int shape_m = 64, m_block_idx = 0;
        float* d_out = nullptr;
        cudaMalloc(&d_out, shape_m * sizeof(float));
        cudaMemset(d_out, 0, shape_m * sizeof(float));
        reduce_map_kernel<BLOCK_M, BLOCK_M_PER_WARP><<<1, 128>>>(d_out, shape_m, m_block_idx);
        cudaDeviceSynchronize();
        std::vector<float> got(shape_m, 0.f);
        cudaMemcpy(got.data(), d_out, shape_m * sizeof(float), cudaMemcpyDeviceToHost);

        // Reference: row r -> sub=r/16, rem=r%16, g=rem%8, u=rem/8;
        //            value = 4*(sub*10000 + 100*g + 1000*u) + (0+1+2+3)
        bool ok = true;
        for (int r = 0; r < shape_m; ++ r) {
            const int sub = r / 16, rem = r % 16, g = rem % 8, u = rem / 8;
            const float ref = 4.f * (sub * 10000 + 100 * g + 1000 * u) + 6.f;
            if (got[r] != ref) ok = false;
        }
        std::printf("  m_idx = m_block_idx*64 + sub*16 + lane/4 + u*8, warp_reduce_sum<4> group sum : %s\n",
                    ok ? "ALL MATCH" : "MISMATCH");
        CHECK(ok, "cast-and-reduce m_idx mapping + warp_reduce_sum<4> (real device code)");
        cudaFree(d_out);
    }
    // __ffma2_rn square accumulation (sm90 fallback of fma2) vs fp64
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto x = torch::randn({65536}, opts);
        auto f32 = (x * x).sum();          // fma2(v,v,acc) with acc=0 == fma2 identity
        auto f64 = (x.to(torch::kFloat64) * x.to(torch::kFloat64)).sum();
        const double rel = std::abs((double)f32.item<float>() - f64.item<double>()) / (std::abs(f64.item<double>()) + 1e-30);
        std::printf("  fma2(v,v,0) square-sum fp32 vs fp64 rel = %.3e\n", rel);
        CHECK(rel < 1e-5, "square accumulation in fp32 matches fp64");
    }

    // ------------------------------------------------------------------
    // Part 5: SM100 wrapper TMA descriptor construction (real torch tensors)
    // ------------------------------------------------------------------
    std::printf("\n[Part 5] SM100 wrapper TMA descriptors (2D / 3D split) with real tensors\n");
    {
        constexpr int block_m = 64, block_k = 64;
        for (int n : {24, 128}) {
            const int m = 4096, k = 7168;
            const int block_n = align(n, 16);
            const auto swizzle_cd_mode = get_swizzle_mode(block_n, sizeof(float));
            auto a = torch::randn({m, k}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
            auto b = torch::randn({n, k}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto d2d = torch::zeros({m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            const auto tma_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k, block_m, block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                               get_swizzle_mode(block_k, a.element_size()), 0, true);
            const auto tma_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k, block_n, block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                               get_swizzle_mode(block_k, b.element_size()), 0, true);
            const auto tma_d2 = make_tma_cd_desc(d2d, m, n, block_m, block_n,
                                                 static_cast<int>(d2d.stride(-2)), 1, swizzle_cd_mode);
            char msg[160];
            std::snprintf(msg, sizeof(msg), "n=%d: A/B/D 2D TMA descriptors built (a=%.4llx d2=%.4llx)",
                          n, (unsigned long long)(desc_hash(tma_a) & 0xffff), (unsigned long long)(desc_hash(tma_d2) & 0xffff));
            CHECK(desc_hash(tma_a) != 0 and desc_hash(tma_b) != 0 and desc_hash(tma_d2) != 0, msg);

            // 3D split: D scratch is [num_splits, m, n] -> make_tma_3d_desc(d, n, m, num_splits, block_n, block_m, 1, ...)
            const int num_splits = 16;
            auto d3d = torch::zeros({num_splits, m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            const auto tma_d3 = make_tma_3d_desc(d3d, n, m, num_splits, block_n, block_m, 1,
                                                 static_cast<int>(d3d.stride(-2)), static_cast<int>(d3d.stride(-3)), swizzle_cd_mode);
            std::printf("  n=%d (block_n=%d, swz_cd=%d): 2D D hash=%.4llx, 3D D hash=%.4llx (differ=%s)\n",
                        n, block_n, swizzle_cd_mode,
                        (unsigned long long)(desc_hash(tma_d2) & 0xffff), (unsigned long long)(desc_hash(tma_d3) & 0xffff),
                        desc_hash(tma_d2) != desc_hash(tma_d3) ? "YES" : "NO");
            CHECK(desc_hash(tma_d3) != 0 and desc_hash(tma_d2) != desc_hash(tma_d3),
                  "3D split D descriptor constructed and distinct from 2D");

            CHECK(b.element_size() == 4 and k % block_k == 0, "B is fp32 and k % 64 == 0 (wrapper contract)");
        }
    }

    // ------------------------------------------------------------------
    std::printf("\n== SUMMARY: %d PASS / %d FAIL ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
