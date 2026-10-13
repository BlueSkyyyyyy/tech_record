// P128-1 实验：DeepGEMM 启发式共用的「元素位宽 / TMA swizzle 模式」公共选择器 +
//   GemmDesc 的 dtype→MmaKind 派生管道 + Python 公开布局别名层。
// 直接 include 真实头文件（heuristics/utils.hpp / heuristics/config.hpp），H100(sm90) 纯 host 运行。
// 对应 harness/exp_mma_kind_swizzle.{sh,out.txt}。

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include <deep_gemm/common/types.cuh>

#include "csrc/jit_kernels/heuristics/utils.hpp"
#include "csrc/jit_kernels/heuristics/config.hpp"
#include "csrc/utils/math.hpp"

using namespace deep_gemm;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& what) {
    if (cond) { ++ g_pass; }
    else { ++ g_fail; printf("  !! FAIL: %s\n", what.c_str()); }
}

static const char* kind_name(MmaKind k) {
    switch (k) {
        case MmaKind::BF16:     return "BF16";
        case MmaKind::MXFP8FP4: return "MXFP8FP4";
        case MmaKind::MXF4:     return "MXF4";
    }
    return "?";
}

// ---- Part A: get_num_element_bits 与「block_k = 128*8/bits」的 1024-bit 不变量 ----
static void part_a() {
    printf("== Part A: get_num_element_bits / block_k=128*8/bits 不变量 ==\n");
    for (auto k : {MmaKind::BF16, MmaKind::MXFP8FP4, MmaKind::MXF4}) {
        const int bits = get_num_element_bits(k);
        const int block_k = 128 * 8 / bits;
        printf("PART_A kind=%s bits=%d block_k=%d block_k*bits=%d\n",
               kind_name(k), bits, block_k, block_k * bits);
        check(block_k * bits == 1024, "block_k*bits should be 1024 for " + std::string(kind_name(k)));
        check(block_k == 64 or block_k == 128 or block_k == 256, "block_k in {64,128,256}");
    }
    // Mega MoE 的局部对偶： get_num_mma_elem_bytes = BF16?2:1  => block_k = 128 / that
    for (auto k : {MmaKind::BF16, MmaKind::MXFP8FP4, MmaKind::MXF4}) {
        const int eb = (k == MmaKind::BF16) ? 2 : 1;
        const int mega_block_k = 128 / eb;
        printf("PART_A mega_local kind=%s elem_bytes=%d block_k=%d\n", kind_name(k), eb, mega_block_k);
        // 与共享函数只在 MXF4 上分歧：共享给 256（逻辑 4-bit），Mega 给 128（smem 按 8-bit 展开）
        if (k == MmaKind::MXF4)
            check(mega_block_k == 128 and (128 * 8 / get_num_element_bits(k)) == 256,
                  "MXF4 divergence 128 vs 256");
        else
            check(mega_block_k == (128 * 8 / get_num_element_bits(k)), "agree for non-MXF4");
    }
    printf("\n");
}

// ---- Part B: get_swizzle_mode 穷举 = {128,64,32,16} 里最大的可整除者 ----
static int expected_swizzle(int block_size, int elem) {
    for (int mode : {128, 64, 32, 16})
        if ((block_size * elem) % mode == 0) return mode;
    return 0;  // 会抛 DG_HOST_UNREACHABLE
}
static void part_b() {
    printf("== Part B: get_swizzle_mode 穷举 vs 闭式定义 ==\n");
    int total = 0, mismatch = 0, unreachable = 0, unreachable_ok = 0;
    for (int bs = 1; bs <= 1024; ++ bs) {
        for (int elem : {1, 2, 4, 8}) {
            ++ total;
            const int exp = expected_swizzle(bs, elem);
            int got = -1;
            try {
                got = get_swizzle_mode(bs, elem);
            } catch (const DGException&) {
                ++ unreachable;
                if (exp == 0) ++ unreachable_ok;
                continue;
            }
            if (exp == 0) { ++ mismatch; continue; }  // 该抛却没抛
            if (got != exp) {
                ++ mismatch;
                if (mismatch <= 5)
                    printf("  MISMATCH bs=%d elem=%d got=%d exp=%d\n", bs, elem, got, exp);
            }
        }
    }
    printf("PART_B total=%d mismatch=%d unreachable=%d unreachable_expected=%d\n",
           total, mismatch, unreachable, unreachable_ok);
    check(mismatch == 0, "swizzle closed-form mismatch");
    check(unreachable == unreachable_ok, "unreachable exactly when product not divisible by 16");

    // 语义要点：128=最大 swizzle，16=「非 swizzle 仅交错」，返回 >0 表示交错
    for (int bs : {16, 32, 64, 128, 256}) {
        printf("PART_B row bs=%d e1=%d e2=%d e4=%d e8=%d\n",
               bs, get_swizzle_mode(bs, 1), get_swizzle_mode(bs, 2),
               get_swizzle_mode(bs, 4), get_swizzle_mode(bs, 8));
    }
    check(get_swizzle_mode(64, 2) == 128, "64*2 -> 128");
    check(get_swizzle_mode(64, 1) == 64,  "64*1 -> 64");
    check(get_swizzle_mode(16, 1) == 16,  "16*1 -> 16 (non-swizzle interleave)");
    printf("\n");
}

static GemmDesc make_desc(torch::ScalarType a, torch::ScalarType b, torch::ScalarType cd,
                          GemmType gt = GemmType::Normal, bool acc = false, int num_sms = 132) {
    GemmDesc d = {};
    d.a_dtype = a; d.b_dtype = b; d.cd_dtype = cd;
    d.gemm_type = gt; d.kernel_type = KernelType::Kernel1D1D;
    d.with_accumulation = acc; d.num_sms = num_sms;
    d.major_a = cute::UMMA::Major::K; d.major_b = cute::UMMA::Major::K;
    d.m = 128; d.n = 128; d.k = 128; d.num_groups = 1;
    return d;
}

// ---- Part C: GemmDesc 的 dtype→MmaKind 派生管道 + check_validity 接受/拒绝矩阵 ----
static void part_c() {
    printf("== Part C: GemmDesc 派生量 + check_validity 矩阵 ==\n");
    struct Case { const char* name; torch::ScalarType a, b, cd; };
    const std::vector<Case> cases = {
        {"bf16bf16", torch::kBFloat16, torch::kBFloat16, torch::kBFloat16},
        {"fp8fp8",   torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, torch::kBFloat16},
        {"fp8fp4",   torch::kFloat8_e4m3fn, kPackedFP4, torch::kBFloat16},
        {"fp4fp4",   kPackedFP4, kPackedFP4, torch::kBFloat16},
    };
    for (const auto& c : cases) {
        const auto d = make_desc(c.a, c.b, c.cd);
        printf("PART_C name=%s mma_kind=%d is_mxf4=%d pack_factor=%d\n",
               c.name, (int)d.get_mma_kind(), (int)d.is_mxf4_mma(), d.get_smem_pack_factor());
    }
    check(make_desc(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16).get_mma_kind() == MmaKind::BF16, "bf16 kind");
    check(make_desc(kPackedFP4, kPackedFP4, torch::kBFloat16).get_mma_kind() == MmaKind::MXF4, "fp4 kind");
    check(make_desc(kPackedFP4, kPackedFP4, torch::kBFloat16).get_smem_pack_factor() == 2, "fp4 pack 2");
    check(make_desc(torch::kFloat8_e4m3fn, kPackedFP4, torch::kBFloat16).get_mma_kind() == MmaKind::MXFP8FP4, "fp8fp4 kind");
    check(make_desc(torch::kFloat8_e4m3fn, kPackedFP4, torch::kBFloat16).get_smem_pack_factor() == 1, "fp8fp4 pack 1");

    // check_validity 的接受/拒绝矩阵
    struct VCase { const char* name; GemmDesc d; bool valid; };
    std::vector<VCase> vc = {
        {"bf16/bf16/bf16",        make_desc(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16), true},
        {"fp8/fp8/bf16",          make_desc(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, torch::kBFloat16), true},
        {"fp8/fp8/float",         make_desc(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, torch::kFloat), true},
        {"fp4/fp4/bf16",          make_desc(kPackedFP4, kPackedFP4, torch::kBFloat16), true},
        {"fp8/fp4/bf16",          make_desc(torch::kFloat8_e4m3fn, kPackedFP4, torch::kBFloat16), true},
        {"fp8d Normal",           make_desc(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn), false},
        {"fp8d Batched acc=F",    make_desc(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, GemmType::Batched, false), true},
        {"fp8d Batched acc=T",    make_desc(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn, GemmType::Batched, true), false},
        {"bf16/fp8/bf16",         make_desc(torch::kBFloat16, torch::kFloat8_e4m3fn, torch::kBFloat16), false},
        {"odd num_sms",           make_desc(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16, GemmType::Normal, false, 131), false},
    };
    for (auto& c : vc) {
        bool ok = true; std::string msg;
        try { c.d.check_validity(); }
        catch (const DGException& e) { ok = false; msg = e.what(); }
        printf("PART_C validity name=%s expect_valid=%d got_valid=%d\n",
               c.name, (int)c.valid, (int)ok);
        if (!ok and !msg.empty()) printf("PART_C reject_msg=%s\n", msg.c_str());
        check(ok == c.valid, std::string("validity ") + c.name);
    }
    // expected_* 的 0=回退语义
    auto d = make_desc(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16);
    d.m = 256; d.expected_m = 0;
    check(d.get_expected_m() == 256, "expected_m 0 -> m");
    d.expected_m = 512;
    check(d.get_expected_m() == 512, "expected_m set");
    printf("\n");
}

// ---- Part D: SM90/SM100 storage swizzle 复刻 = 恒为 128B（block_k*bits=1024） ----
static void part_d() {
    printf("== Part D: SM90/SM100 storage swizzle 复刻 ==\n");
    struct Cfg { const char* name; MmaKind kind; torch::ScalarType dtype; int block_m; };
    const std::vector<Cfg> cfgs = {
        {"bf16", MmaKind::BF16, torch::kBFloat16, 128},
        {"fp8",  MmaKind::MXFP8FP4, torch::kFloat8_e4m3fn, 128},
        {"fp4",  MmaKind::MXF4, kPackedFP4, 128},
    };
    for (const auto& c : cfgs) {
        const int block_k = 128 * 8 / get_num_element_bits(c.kind);
        const int elem = (int) c10::elementSize(c.dtype);   // packed FP4 存成 int8 => 1
        // SM90：不除 pack factor
        const int s90_k = get_swizzle_mode(block_k, elem);
        // SM100：K-major 时先除 pack factor（把逻辑 block_k 换成物理 smem 字节数）
        const auto d = make_desc(c.dtype, c.dtype, c.dtype);
        const int pf = d.get_smem_pack_factor();
        const int s100_k = get_swizzle_mode(block_k / pf, elem);
        // SM100 MN-major：用 load_block_m / pf
        const int s100_mn = get_swizzle_mode(c.block_m / pf, elem);
        printf("PART_D name=%s block_k=%d elem=%d pack=%d sm90_swizzle_k=%d sm100_swizzle_k=%d sm100_swizzle_mn=%d\n",
               c.name, block_k, elem, pf, s90_k, s100_k, s100_mn);
        check(s90_k == 128, std::string("sm90 K swizzle 128 for ") + c.name);
        check(s100_k == 128, std::string("sm100 K swizzle 128 for ") + c.name);
    }
    // 宽度减半 => swizzle 减半（Mega MoE L1 输出 128->64 的同款效应）
    check(get_swizzle_mode(128, 1) == 128 and get_swizzle_mode(64, 1) == 64, "half width -> half swizzle");
    printf("PART_D halfline swizzle_full=%d swizzle_half=%d\n",
           get_swizzle_mode(128, 1), get_swizzle_mode(64, 1));
    printf("\n");
}

// ---- Part E: 与 swizzle 配套的 16B TMA 对齐算术（csrc/utils/math.hpp） ----
static void part_e() {
    printf("== Part E: get_tma_aligned_size / align / ceil_div ==\n");
    for (int elem : {1, 2, 4}) {
        const int al = get_tma_aligned_size(100, elem);   // align(100, 16/elem)
        const int want = align(100, 16 / elem);
        printf("PART_E elem=%d aligned=%d want=%d\n", elem, al, want);
        check(al == want, "get_tma_aligned_size matches align(x,16/elem)");
    }
    check(ceil_div(101, 128) == 1 and ceil_div(128, 128) == 1 and ceil_div(129, 128) == 2, "ceil_div");
    check(align(129, 128) == 256, "align");
    printf("\n");
}

int main() {
    printf("== P128-1 mma-kind / swizzle selector ==\n");
    part_a();
    part_b();
    part_c();
    part_d();
    part_e();
    printf("SUMMARY pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
