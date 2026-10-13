// P79-1 两个「Mega 家族」JIT 启动包装层的 host 侧验证实验（H100 / sm90 真机）
//
// 覆盖前面从没单独拆过的两个文件：
//   csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp   （P10-1 拆了 kernel，没拆 wrapper）
//   csrc/jit_kernels/impls/sm100_mega_mhc.hpp        （P5-2 拆了 kernel，没拆 wrapper）
//
// 主 kernel 都是 SM100 专属（跑不了），但这两个 wrapper 是**纯 host C++**：
//   它们只做「host 张量 → 张量形状推导 → TMA 描述符 → JIT 模板实参 → launch 参数」
//   最后一公里。本实验把其中架构无关的部分抽出来真机跑：
//   Part 1  to_string(float) 的十六进制浮点字面量编码（activation_clamp 进模板源）
//   Part 2  MixArgs / NormArgs 的 host<->device ABI 布局（offsetof / sizeof）
//   Part 3  mHC wrapper 的三个 TMA lambda 的参数推导 + 真实 3D TMA 载入（route 当第三维）
//   Part 4  BF16 Mega MoE wrapper 的描述符推导：与 FP8 wrapper 的关键差异 + 共享专家回退
//
// 编译：见 harness/exp_mega_moe_mhc_jit_wrapper.sh（nvcc 直接 include 真实头文件 + torch）
// 真实头文件里的 DG_UNIFIED_ASSERT 在「nvcc 单 TU 的 host pass」会落成 asm("trap;")，
// 这里在任何 deep_gemm 头之前把它钉到 host 版（普通 abort），否则 host 汇编器不认识 trap。
#include <cassert>
#ifndef DG_UNIFIED_ASSERT
#define DG_UNIFIED_ASSERT(cond) assert(cond)
#endif

#include <torch/python.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "utils/exception.hpp"                       // DG_HOST_ASSERT / DG_HOST_UNREACHABLE

#include <deep_gemm/scheduler/mega_moe.cuh>          // sched::TaskInfo<>（heuristics 需要 sizeof）
#include "jit_kernels/impls/runtime_utils.hpp"       // make_tma_* + to_string(float) + dtype/swizzle
#include "jit_kernels/heuristics/mega_moe.hpp"       // get_mega_moe_config + MmaKind
#include <deep_gemm/layout/mega_mhc.cuh>             // MixArgs / NormArgs / BLOCK_* / Workspace
#include <deep_gemm/layout/mega_moe.cuh>             // kCandidateBlockM

using deep_gemm::make_tma_2d_desc;
using deep_gemm::make_tma_3d_desc;
using deep_gemm::make_tma_weights_3d_desc;
namespace mhc_layout = deep_gemm::layout::mega_mhc;

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& tag) {
    if (ok) { g_pass++; } else { g_fail++; printf("  !! FAIL %s\n", tag.c_str()); }
}
static bool eq128(const CUtensorMap& a, const CUtensorMap& b) {
    return memcmp(&a, &b, 128) == 0;
}
static uint32_t fbits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }

// 独立复刻：直接调 driver 编码（不借道 make_tma_*）
static CUtensorMap ref_encode(const torch::Tensor& t, const int rank,
                              const std::vector<cuuint64_t>& gd,
                              const std::vector<cuuint64_t>& gs_elem,
                              const std::vector<cuuint32_t>& sd,
                              const int swz, const int base,
                              const bool tf32, const bool fp4_un) {
    const int elem = static_cast<int>(t.element_size());
    std::vector<cuuint64_t> gsb;
    for (auto s : gs_elem) gsb.push_back(s * static_cast<cuuint64_t>(elem));
    std::vector<cuuint32_t> es(rank, 1);
    CUtensorMap m;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &m, deep_gemm::aten_dtype_to_tensor_map_dtype(t.scalar_type(), tf32, fp4_un),
        rank, t.data_ptr(), gd.data(), rank > 1 ? gsb.data() : nullptr, sd.data(), es.data(),
        CU_TENSOR_MAP_INTERLEAVE_NONE, deep_gemm::mode_into_tensor_map_swizzle(swz, base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}
static std::vector<cuuint32_t> ref_2d_smem(const torch::Tensor& t, int si, int so, int swz) {
    const int elem = static_cast<int>(t.element_size());
    if (swz != 0) si = swz / elem;
    return {(cuuint32_t)si, (cuuint32_t)so};
}

// ---------------------------------------------------------------------------
// Part 1: to_string(float) —— activation_clamp 编进 JIT 源码的十六进制字面量
// ---------------------------------------------------------------------------
static void part1_float_literal() {
    printf("== Part 1: to_string(float) 十六进制浮点字面量往返 ==\n");
    printf("  来源 runtime_utils.hpp:75-83  std::format(\"{{}}0x{{:a}}f\", sign, abs(v))\n");
    struct V { const char* name; float v; };
    V vs[] = {
        {"one",      1.0f},
        {"half",     0.5f},
        {"neg_zero", -0.0f},
        {"tiny",     1e-5f},
        {"fmax",     3.4028235e38f},
        {"min_norm", 1.17549435e-38f},
        {"neg_val",  -2.7182817f},
        {"+inf",     INFINITY},
        {"-inf",     -INFINITY},
    };
    for (auto& x : vs) {
        std::string lit = deep_gemm::to_string(x.v);
        // 有限值：字面量再喂回 C 解析器应逐位回到原值；±inf：编成 cute 的 infinity() 表达式
        bool ok;
        if (std::isinf(x.v)) {
            ok = lit.find("infinity()") != std::string::npos &&
                 ((x.v > 0) == (lit[0] != '-'));
        } else {
            float back = std::strtof(lit.c_str(), nullptr);
            ok = (fbits(back) == fbits(x.v));
        }
        check(ok, std::string("float_lit_") + x.name);
        printf("  %-9s bits=0x%08x  literal=%-38s  %s\n",
               x.name, fbits(x.v), lit.c_str(), ok ? "OK" : "DIFF");
    }
    // NaN 必须在 wrapper 里被拒绝（DG_HOST_UNREACHABLE），否则会编出非法字面量
    bool rejected = false;
    try { (void) deep_gemm::to_string(NAN); } catch (...) { rejected = true; }
    check(rejected, "float_lit_nan_rejected");
    printf("  %-9s -> %s\n", "NaN", rejected ? "拒绝（DG_HOST_UNREACHABLE）" : "未拒绝 !!");
}

// ---------------------------------------------------------------------------
// Part 2: MixArgs / NormArgs 的 ABI 布局（kernel 以 __grid_constant__ 接收）
// ---------------------------------------------------------------------------
static void part2_abi() {
    printf("== Part 2: MixArgs / NormArgs 的 host<->device ABI 布局 ==\n");
    using deep_gemm::layout::mega_mhc::MixArgs;
    using deep_gemm::layout::mega_mhc::NormArgs;

    struct Exp { size_t off; };
    struct Field { const char* name; size_t got; size_t exp; };
    // MixArgs：5 指针(40B) + 4 float(16B) + 1 uint32(4B) = 60 → align8 → 64
    std::vector<Field> mix = {
        {"scales",            offsetof(MixArgs, scales),            0},
        {"bases",             offsetof(MixArgs, bases),             8},
        {"new_prev_mix",      offsetof(MixArgs, new_prev_mix),      16},
        {"new_post_mix",      offsetof(MixArgs, new_post_mix),      24},
        {"new_comb_res_mix",  offsetof(MixArgs, new_comb_res_mix),  32},
        {"hc_norm_eps",       offsetof(MixArgs, hc_norm_eps),       40},
        {"hc_pre_eps",        offsetof(MixArgs, hc_pre_eps),        44},
        {"hc_post_scale",     offsetof(MixArgs, hc_post_scale),     48},
        {"sinkhorn_eps",      offsetof(MixArgs, sinkhorn_eps),      52},
        {"num_sinkhorn_iters",offsetof(MixArgs, num_sinkhorn_iters),56},
    };
    for (auto& f : mix) {
        check(f.got == f.exp, std::string("MixArgs.") + f.name);
        printf("  MixArgs.%-18s off=%2zu exp=%2zu %s\n", f.name, f.got, f.exp,
               f.got == f.exp ? "OK" : "DIFF");
    }
    check(sizeof(MixArgs) == 64, "sizeof(MixArgs)");
    printf("  sizeof(MixArgs)=%zu (exp 64) align=%zu\n", sizeof(MixArgs), alignof(MixArgs));

    std::vector<Field> norm = {
        {"num_tokens",                 offsetof(NormArgs, num_tokens),                 0},
        {"weight",                     offsetof(NormArgs, weight),                     8},
        {"new_residual",               offsetof(NormArgs, new_residual),               16},
        {"eps",                        offsetof(NormArgs, eps),                        24},
        {"scale",                      offsetof(NormArgs, scale),                      28},
        {"y_bf16",                     offsetof(NormArgs, y_bf16),                     32},
        {"y_fp8",                      offsetof(NormArgs, y_fp8),                      40},
        {"y_primary_sf",               offsetof(NormArgs, y_primary_sf),               48},
        {"y_primary_sf_stride_token",  offsetof(NormArgs, y_primary_sf_stride_token),  56},
        {"y_primary_sf_stride_word",   offsetof(NormArgs, y_primary_sf_stride_word),   64},
        {"y_shared_sf",                offsetof(NormArgs, y_shared_sf),                72},
        {"y_shared_sf_stride_word",    offsetof(NormArgs, y_shared_sf_stride_word),    80},
    };
    for (auto& f : norm) {
        check(f.got == f.exp, std::string("NormArgs.") + f.name);
        printf("  NormArgs.%-26s off=%2zu exp=%2zu %s\n", f.name, f.got, f.exp,
               f.got == f.exp ? "OK" : "DIFF");
    }
    check(sizeof(NormArgs) == 88, "sizeof(NormArgs)");
    printf("  sizeof(NormArgs)=%zu (exp 88) align=%zu\n", sizeof(NormArgs), alignof(NormArgs));
    // 指针可空（y_fp8==nullptr / new_prev_mix==nullptr）正是模板布尔的来源
    printf("  wrapper 把 new_prev_mix!=nullptr / y_fp8!=nullptr 编成模板 bool（见 §mHC）\n");
}

// ---------------------------------------------------------------------------
// Part 3: mHC wrapper 的三个 TMA lambda（sm100_mega_mhc.hpp:37-51）
//   真机做一次 3D TMA 载入，验证「route 当第三维」的取址语义
// ---------------------------------------------------------------------------
__global__ void tma_load_3d_kernel(const __grid_constant__ CUtensorMap tmap,
                                   int d0, int d1, int d2, int elem_bytes,
                                   int c0, int c1, int c2, unsigned char* out) {
    __shared__ __align__(1024) unsigned char smem[32768];
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int box_bytes = d0 * d1 * d2 * elem_bytes;
    if (tid == 0) {
        const uint32_t sp = (uint32_t)__cvta_generic_to_shared(smem);
        const uint32_t mp = (uint32_t)__cvta_generic_to_shared(&mbar);
        const uint64_t dp = reinterpret_cast<uint64_t>(&tmap);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mp), "r"(box_bytes));
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5}], [%2];" ::
            "r"(sp), "l"(dp), "r"(mp), "r"(c0), "r"(c1), "r"(c2) : "memory");
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t mp = (uint32_t)__cvta_generic_to_shared(&mbar);
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
            "@!p bra WAIT;\n\t}" :: "r"(mp));
    }
    __syncthreads();
    for (int i = tid; i < box_bytes; i += blockDim.x) out[i] = smem[i];
}

// TMA swizzle 的 XOR 键取自 smem 绝对地址 bits[7:10)（smem 对齐 1024 → base_row_bits 恒 0）
static int swizzle_offset(const int linear_byte_offset) {
    const int key = (linear_byte_offset >> 7) & 7;   // swizzle 128B → 3 位
    return linear_byte_offset ^ (key << 4);
}

static void part3_mhc_lambdas() {
    printf("== Part 3: mHC wrapper 的 TMA lambda（输入维/步长/swizzle 推导）==\n");
    printf("  来源 sm100_mega_mhc.hpp:37-80，常量 layout/mega_mhc.cuh:11-31\n");
    const int H = 512, T = 128;
    const auto R  = mhc_layout::kNumRoutes;            // 4
    const auto HC = mhc_layout::kNumHCOutputs;         // 24
    const auto BM = mhc_layout::BLOCK_M;               // 64
    const auto BK = mhc_layout::BLOCK_K;               // 64
    const auto SW = mhc_layout::kSwizzleMode;          // 128
    printf("  geometry: kNumRoutes=%u kNumHCOutputs=%u BLOCK_M=%u BLOCK_K=%u kSwizzleMode=%u\n",
           R, HC, BM, BK, SW);

    auto x = torch::ones({T, H}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto residual = torch::ones({T, R, H}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto post_mix = torch::ones({T, R, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto comb_res_mix = torch::ones({T, R, R}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto fn = torch::ones({(long)HC, (long)R * H}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto new_residual = torch::ones({T, R, H}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

    // ---- make_hidden_tma_desc(x / y_bf16 / new_residual 不用，因为它 0 维不是 hidden）----
    auto hidden_desc = [&](const torch::Tensor& t) {
        return make_tma_2d_desc(t, H, T, BK, BM, (int)t.stride(0), SW);
    };
    // ---- make_residual_tma_desc: 第三维 = route，两个显式 stride ----
    auto residual_desc = [&](const torch::Tensor& t) {
        return make_tma_3d_desc(t, H, T, (int)R, BK, BM, (int)R,
                                (int)t.stride(0), (int)t.stride(1), SW);
    };
    // ---- make_coeff_tma_desc: channels 为内层宽度，per-token 布局 ----
    auto coeff_desc = [&](const torch::Tensor& t, int channels, int swz) {
        return make_tma_2d_desc(t, channels, T, channels, BM, (int)t.stride(0), swz);
    };

    auto d_x  = hidden_desc(x);
    auto d_res = residual_desc(residual);
    auto d_nres = residual_desc(new_residual);
    auto d_post = coeff_desc(post_mix, R, 0);
    auto d_comb = coeff_desc(comb_res_mix, (int)(R * R), SW / 2);
    auto d_fn = make_tma_3d_desc(fn, H, HC, (int)R, BK, HC, (int)R,
                                 (int)fn.stride(0), H, SW, 0, true);

    // --- 参数推导对拍（独立手算）---
    // post_mix: gmem=(4,128) smem inner: swizzle0→channels=4, outer=BM; stride=4
    check(eq128(d_post, ref_encode(post_mix, 2, {4, T}, {4}, ref_2d_smem(post_mix, 4, BM, 0), 0, 0, false, true)),
          "mhc_post_mix_2d");
    // comb_res_mix: gmem=(16,128) smem inner: swizzle64→64/4=16, outer=BM; stride=16
    check(eq128(d_comb, ref_encode(comb_res_mix, 2, {16, T}, {16}, ref_2d_smem(comb_res_mix, 16, BM, 64), 64, 0, false, true)),
          "mhc_comb_res_2d");
    // fn: 3D，dtype 因 allow_tf32=true 编成 TFLOAT32
    check(deep_gemm::aten_dtype_to_tensor_map_dtype(torch::kFloat, true, true) == CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
          "mhc_fn_tf32_dtype");
    check(eq128(d_fn, ref_encode(fn, 3, {H, HC, R}, {(cuuint64_t)R * H, (cuuint64_t)H},
                                 {(cuuint32_t)(SW / 4), HC, R}, SW, 0, true, true)),
          "mhc_fn_3d_tf32");
    // x: 2D hidden desc，inner=swizzle/2=64
    check(eq128(d_x, ref_encode(x, 2, {H, T}, {H}, ref_2d_smem(x, 64, BM, SW), SW, 0, false, true)),
          "mhc_x_2d");

    printf("  post_mix_desc    inner=%d outer=%d stride0=%lld swizzle=%d\n",
           R, T, (long long)post_mix.stride(0), 0);
    printf("  comb_res_desc    inner=%d outer=%d stride0=%lld swizzle=%d (kSwizzleMode/2)\n",
           (int)(R * R), T, (long long)comb_res_mix.stride(0), SW / 2);
    printf("  fn_desc          gmem=(%d,%u,%u) strides=(%lld,%d) smem=(%d,%u,%u) swizzle=%d allow_tf32=1\n",
           H, HC, R, (long long)fn.stride(0), H, BK, HC, R, SW);
    printf("  post/comb swizzle 0 vs %d：小数通道不 swizzle、16 通道宽才够 64B swizzle\n", SW / 2);
    check(!eq128(d_post, d_comb), "mhc_post_ne_comb");

    // --- 真实 3D TMA 载入：验证 route 当第三维的取址 ---
    // 真 kernel 一次载入把全部 kNumRoutes=4 个 route 的 [BLOCK_M x BLOCK_K] 都取进来，
    // smem 布局 route-major、每个 64x64 连续且内层 128B 做 swizzle。
    {
        const int m_block = 1, k_block = 1;
        const int num_routes = (int)R;
        auto host = torch::arange((long)T * R * H, torch::dtype(torch::kInt32))
                        .reshape({T, R, H}).to(torch::kFloat32);
        auto res2 = host.to(torch::kBFloat16).contiguous().to(torch::kCUDA);
        auto d_res2 = residual_desc(res2);
        const int box_bytes = BK * BM * num_routes * 2;
        auto out = torch::zeros({box_bytes}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        tma_load_3d_kernel<<<1, 128, 0>>>(d_res2, BK, BM, num_routes, 2,
                                          k_block * BK, m_block * BM, 0, out.data_ptr<unsigned char>());
        const bool run_ok = (cudaDeviceSynchronize() == cudaSuccess);
        check(run_ok, "mhc_3d_tma_run");

        // host 期望：smem 线性序 [route][m][k] + 128B swizzle
        auto host16 = res2.to(torch::kCPU).contiguous();
        const auto* hp = reinterpret_cast<const uint16_t*>(host16.data_ptr());
        std::vector<unsigned char> exp(box_bytes, 0);
        for (int r = 0; r < num_routes; r++)
            for (int m = 0; m < (int)BM; m++)
                for (int k = 0; k < (int)BK; k++) {
                    const int lin = ((r * (int)BM + m) * (int)BK + k) * 2;
                    const int phys = swizzle_offset(lin);
                    const uint16_t v = *reinterpret_cast<const uint16_t*>(
                        hp + ((m_block * BM + m) * R + r) * H + (k_block * BK + k));
                    exp[phys] = v & 0xff; exp[phys + 1] = v >> 8;
                }
        auto got = out.to(torch::kCPU);
        const auto* gp = got.data_ptr<unsigned char>();
        int mism = 0;
        for (int i = 0; i < box_bytes; i++) if (gp[i] != exp[i]) mism++;
        check(mism == 0, "mhc_3d_tma_route_swizzle_layout");
        printf("  residual 3D TMA load @(k=%d,m=%d) box=[64x64x4 route] swizzle128: mismatch=%d/%d  %s\n",
               k_block, m_block, mism, box_bytes, mism == 0 ? "OK" : "DIFF");
    }
}

// ---------------------------------------------------------------------------
// Part 4: BF16 Mega MoE wrapper（sm100_bf16_mega_moe.hpp:46-130）的描述符推导
// ---------------------------------------------------------------------------
static void part4_bf16_mega_moe() {
    printf("== Part 4: BF16 Mega MoE wrapper 描述符推导 + 与 FP8 的差异 ==\n");
    printf("  来源 sm100_bf16_mega_moe.hpp:39-130，对照 sm100_fp8_fp4_mega_moe.hpp:44-139\n");
    const int num_ranks = 2, num_experts = 8, epr = 4;
    const int max_tok = 128, num_tok = 128, topk = 2;
    const int H = 512, I = 256;
    const int num_ring_tokens = 1920;
    const int num_sf_ring_tokens = 1920;

    const auto cfg_bf = deep_gemm::get_mega_moe_config(
        num_ranks, num_experts, epr, max_tok, num_tok, topk, H, I,
        num_ring_tokens, 0, deep_gemm::MmaKind::BF16);
    const auto cfg_fp8 = deep_gemm::get_mega_moe_config(
        num_ranks, num_experts, epr, max_tok, num_tok, topk, H, I,
        num_ring_tokens, num_sf_ring_tokens, deep_gemm::MmaKind::MXFP8FP4);

    printf("  BF16 config: block_m=%d block_n=%d block_k=%d store_l1=%d store_l2=%d "
           "num_ring_tokens=%d num_sf_ring_tokens=%d num_stages=%d smem=%d threads=%d/%d/%d\n",
           cfg_bf.block_m, cfg_bf.block_n, cfg_bf.block_k, cfg_bf.store_block_m_l1,
           cfg_bf.store_block_m_l2, cfg_bf.num_ring_tokens, cfg_bf.num_sf_ring_tokens,
           cfg_bf.num_stages, cfg_bf.smem_size, cfg_bf.num_dispatch_threads,
           cfg_bf.num_non_epilogue_threads, cfg_bf.num_epilogue_threads);
    printf("  FP8  config: block_m=%d block_n=%d block_k=%d store_l1=%d store_l2=%d "
           "num_ring_tokens=%d num_sf_ring_tokens=%d num_stages=%d smem=%d\n",
           cfg_fp8.block_m, cfg_fp8.block_n, cfg_fp8.block_k, cfg_fp8.store_block_m_l1,
           cfg_fp8.store_block_m_l2, cfg_fp8.num_ring_tokens, cfg_fp8.num_sf_ring_tokens,
           cfg_fp8.num_stages, cfg_fp8.smem_size);

    // wrapper:44 断言 BF16 只有一个 store block
    check(cfg_bf.store_block_m_l1 == cfg_bf.store_block_m_l2, "bf16_store_block_m_l1_eq_l2");
    check(cfg_bf.block_k == 64, "bf16_block_k_64");
    check(cfg_fp8.block_k == 128, "fp8_block_k_128");
    check(cfg_bf.num_sf_ring_tokens == 0, "bf16_no_sf_ring");
    printf("  差异① block_k: BF16=64(128/2B) vs FP8=128(128/1B)；差异② BF16 num_sf_ring_tokens=0\n");

    const int hidden = H, intermediate_hidden = I;
    auto l1_acts = torch::ones({cfg_bf.num_ring_tokens, hidden}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto l2_acts = torch::ones({cfg_bf.num_ring_tokens, hidden}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

    // wrapper:53-57 BF16 l1_output 用 config.swizzle_acts_mode（=128）
    auto bf16_l1_out = make_tma_2d_desc(l2_acts, intermediate_hidden, cfg_bf.num_ring_tokens,
                                        cfg_bf.block_n / 2, cfg_bf.store_block_m_l1,
                                        (int)l2_acts.stride(-2), cfg_bf.swizzle_acts_mode);
    // 对照 FP8 wrapper:72-76 用 config.swizzle_acts_mode / 2（=64）
    auto fp8_l1_out = make_tma_2d_desc(l2_acts, intermediate_hidden, cfg_bf.num_ring_tokens,
                                       cfg_bf.block_n / 2, cfg_bf.store_block_m_l1,
                                       (int)l2_acts.stride(-2), cfg_bf.swizzle_acts_mode / 2);
    const int bf16_smem_inner = cfg_bf.swizzle_acts_mode / 2;          // make_tma_2d_desc 覆盖
    const int fp8_smem_inner = (cfg_bf.swizzle_acts_mode / 2) / 1;     // elem=1 → 64
    printf("  l1_output: BF16 swizzle=%d smem_inner=%d (≈%d elem)  FP8 swizzle=%d smem_inner=%d (≈%d elem)\n",
           cfg_bf.swizzle_acts_mode, bf16_smem_inner, cfg_bf.swizzle_acts_mode / 2,
           cfg_bf.swizzle_acts_mode / 2, fp8_smem_inner, cfg_bf.swizzle_acts_mode / 2);
    check(bf16_smem_inner == cfg_bf.block_n / 2, "bf16_l1out_smem_inner_64");
    // 反例：若 BF16 误用 FP8 的 swizzle/2，则 smem_inner=32 < block_n/2=64 → 覆盖错
    printf("  差异③ l1_output 的 swizzle：BF16 取 128（64 elem×2B）/ FP8 取 64（64 elem×1B）——\n");
    printf("          两者 smem_inner 都是 64 元素，但 swizzle 字节数随元素大小减半（否则内层会被压成 32）\n");
    check(bf16_smem_inner != 32, "bf16_l1out_not_32");

    // 共享专家回退：num_shared_experts==0 时 wrapper 直接复用 routed 描述符
    const int num_shared = 0;
    auto l1_acts_desc = make_tma_2d_desc(l1_acts, hidden, cfg_bf.num_ring_tokens,
                                         cfg_bf.block_k, cfg_bf.load_block_m,
                                         (int)l1_acts.stride(-2), cfg_bf.swizzle_acts_mode);
    auto shared_l1_acts = torch::ones({max_tok, hidden}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto shared_l1_acts_desc = num_shared > 0 ? make_tma_2d_desc(
        shared_l1_acts, hidden, max_tok, cfg_bf.block_k, cfg_bf.load_block_m,
        (int)shared_l1_acts.stride(-2), cfg_bf.swizzle_acts_mode) : l1_acts_desc;
    check(eq128(shared_l1_acts_desc, l1_acts_desc), "bf16_shared_fallback_identity");
    printf("  差异④ num_shared_experts==0：shared_* 描述符逐字节复用 routed 描述符（指针别名，非重编码）\n");

    // weights：BF16 用 make_tma_weights_3d_desc（无 SF 描述符）；FP8 还要额外 SF 描述符
    auto l1_w = torch::ones({epr, 2 * intermediate_hidden, hidden},
                            torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto w_desc = make_tma_weights_3d_desc(l1_w, cfg_bf.block_k, cfg_bf.load_block_n,
                                           cfg_bf.swizzle_weights_mode);
    const int nd = deep_gemm::LocalityDomainAllocator::get_num_locality_domains();
    check(eq128(w_desc, ref_encode(l1_w, 4,
              {(cuuint64_t)hidden, (cuuint64_t)(2 * intermediate_hidden / nd), (cuuint64_t)nd, (cuuint64_t)epr},
              {(cuuint64_t)hidden, (cuuint64_t)(hidden * (2 * intermediate_hidden / nd)),
               (cuuint64_t)(hidden * (2 * intermediate_hidden / nd) * nd)},
              {(cuuint32_t)cfg_bf.block_k, (cuuint32_t)cfg_bf.load_block_n, 1, 1},
              cfg_bf.swizzle_weights_mode, 0, false, true)), "bf16_l1w_3d");
    printf("  l1_weights: 4D(%d,%d,%d,%d) num_domains=%d；BF16 无 SF 描述符，FP8 另需 2 个 SF 描述符\n",
           hidden, 2 * intermediate_hidden / nd, nd, epr, nd);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("P79-1 Mega MoE / mHC JIT 包装层 host 侧验证（H100 sm90）\n");
    printf("DeepGEMM commit: ");
    fflush(stdout);
    system("cd ~/github/DeepGEMM && git -c safe.directory='*' log --oneline -1");
    printf("\n");
    part1_float_literal();   printf("\n");
    part2_abi();             printf("\n");
    part3_mhc_lambdas();     printf("\n");
    part4_bf16_mega_moe();   printf("\n");
    printf("总判定: %s (pass=%d fail=%d)\n", g_fail == 0 ? "ALL PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
