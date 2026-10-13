// P81-1 实验：DeepGEMM 官方 SF 契约（docs/scaling-factor-format.md）与 host 契约代码的一致性核验。
//
// 本篇直接 include DeepGEMM 的**真实 host 头文件**，在 H100(sm90) 上运行契约校验函数本身：
//   - csrc/utils/layout.hpp                     （check_sf_layout，shape / TMA stride / type_check）
//   - csrc/jit_kernels/impls/smxx_layout.hpp    （check_k_grouped_packed_ue8m0_tensor）
//   - csrc/utils/math.hpp                       （get_tma_aligned_size）
//
// 关键点：这两个契约函数**不含任何 arch 门控**（arch 门控在其调用者 transform_sf_into_required_layout
// / transform_k_grouped_sf_into_required_layout 里）。因此虽然 int32 packed UE8M0 在功能上是 SM100
// 专属、kernel 路径 H100 跑不了，但其**契约校验逻辑**可以在 H100 上对真实代码逐条对拍。这正是
// 官方 spec §3 / §5.2 所描述、而前面 25/26 篇未系统核验的那一层。
//
// 编译与运行见 harness/exp_sf_contract_conformance.sh。

#include <cstdio>
#include <string>
#include <vector>
#include <optional>
#include <tuple>

#include <torch/torch.h>

// ---- 真实头文件（-I <DeepGEMM>/csrc -I <DeepGEMM> -I <DeepGEMM>/deep_gemm/include）----
#include "utils/layout.hpp"                      // deep_gemm::check_sf_layout / get_tma_aligned_size
#include "jit_kernels/impls/smxx_layout.hpp"     // deep_gemm::check_k_grouped_packed_ue8m0_tensor

using namespace deep_gemm;

static int g_pass = 0, g_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " — ", detail.c_str());
    ok ? ++ g_pass : ++ g_fail;
}

// 期望「抛出 DGException」：返回错误信息里的 file:line（用于确认命中了哪一条断言）。
template <class F>
static std::string expect_throw(F&& f) {
    try {
        f();
    } catch (const DGException& e) {
        return std::string(e.what());
    } catch (const std::exception& e) {
        return std::string("OTHER:") + e.what();
    }
    return "NO_THROW";
}

static bool contains(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}

// 造一个 int32 的 MN-major TMA 对齐 SF 张量：shape [mn, packed_sf_k]，stride (pk*align(mn,4), 1, align(mn,4))
static torch::Tensor make_int32_mn_major(int64_t mn, int64_t packed_sf_k) {
    const auto align_mn = get_tma_aligned_size(static_cast<int>(mn), 4);
    return torch::empty_strided({mn, packed_sf_k},
                                {1, align_mn},
                                at::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
}

// ============================================================================
// Part A：spec §1.1 形状公式（fp32 未变换 SF）  ↔ check_sf_layout:112-119
// ============================================================================
static void part_a_shape() {
    std::printf("\n== Part A: spec §1.1 形状公式（fp32 未变换 SF）==\n");
    const int mn = 1000, k = 512, gran_mn = 1, gran_k = 128;

    // 正确：ceil_div(1000,1)=1000, ceil_div(512,128)=4
    {
        const auto sf = torch::zeros({mn, 4}, at::TensorOptions().dtype(torch::kFloat));
        const auto r = expect_throw([&] { check_sf_layout(sf, mn, k, gran_mn, gran_k, std::nullopt); });
        report("A1 fp32 [1000,4] 通过（size(-1)=ceil_div(512,128)=4）", r == "NO_THROW", r == "NO_THROW" ? "" : r);
    }
    // 错误末维：5 != 4 → :119
    {
        const auto sf = torch::zeros({mn, 5}, at::TensorOptions().dtype(torch::kFloat));
        const auto r = expect_throw([&] { check_sf_layout(sf, mn, k, gran_mn, gran_k, std::nullopt); });
        report("A2 fp32 末维=5 被拒（layout.hpp:119 ceil_div(k,gran_k)）", contains(r, ":119"), r);
    }
    // gran_mn=128：ceil_div(1000,128)=8
    {
        const auto sf = torch::zeros({8, 4}, at::TensorOptions().dtype(torch::kFloat));
        const auto r = expect_throw([&] { check_sf_layout(sf, mn, k, 128, 128, std::nullopt); });
        report("A3 gran_mn=128 → size(-2)=ceil_div(1000,128)=8 通过", r == "NO_THROW", r == "NO_THROW" ? "" : r);
    }
    // num_groups=3：dim 必须 == 3（:115），且 size(-3)==3（:117）
    {
        const auto ok = torch::zeros({3, mn, 4}, at::TensorOptions().dtype(torch::kFloat));
        const auto r1 = expect_throw([&] { check_sf_layout(ok, mn, k, gran_mn, gran_k, 3); });
        report("A4 num_groups=3 且 dim=3 通过（layout.hpp:115/117）", r1 == "NO_THROW", r1 == "NO_THROW" ? "" : r1);
        const auto bad = torch::zeros({2, mn, 4}, at::TensorOptions().dtype(torch::kFloat));
        const auto r2 = expect_throw([&] { check_sf_layout(bad, mn, k, gran_mn, gran_k, 3); });
        report("A5 size(-3)=2 != num_groups=3 被拒（layout.hpp:117）", contains(r2, ":117"), r2);
        const auto dim2 = torch::zeros({mn, 4}, at::TensorOptions().dtype(torch::kFloat));
        const auto r3 = expect_throw([&] { check_sf_layout(dim2, mn, k, gran_mn, gran_k, 3); });
        report("A6 dim=2 != num_groups+2 被拒（layout.hpp:115）", contains(r3, ":115"), r3);
    }
}

// ============================================================================
// Part B：spec §3 预变换（int32 packed UE8M0）TMA stride 契约 ↔ check_sf_layout:122-130
// ============================================================================
static void part_b_pretransformed() {
    std::printf("\n== Part B: spec §3 预变换 int32 契约（MN-major + TMA 对齐）==\n");
    const int mn = 6, k = 128, gran_k = 32;
    const int packed_sf_k = ceil_div(k, gran_k * 4);   // = 1
    const int align_mn = get_tma_aligned_size(mn, 4);  // = 8
    std::printf("  (mn=%d, k=%d, gran_k=%d → packed_sf_k=%d, align(mn,4)=%d)\n",
                mn, k, gran_k, packed_sf_k, align_mn);

    // 正确：stride (pk*align, 1, align)，type_check=kInt，tma_stride_check=true
    {
        const auto sf = make_int32_mn_major(mn, packed_sf_k);
        std::printf("    strides = (%ld, %ld)\n", sf.stride(0), sf.stride(1));
        const auto r = expect_throw([&] {
            check_sf_layout(sf, mn, k, 1, gran_k, std::nullopt, true, false, torch::kInt);
        });
        report("B1 int32 MN-major [6,1] stride(1,8) 通过（:122-129）", r == "NO_THROW", r == "NO_THROW" ? "" : r);
    }
    // type_check=kInt 但喂 fp32 → :110
    {
        const auto sf = torch::zeros({mn, ceil_div(k, gran_k)}, at::TensorOptions().dtype(torch::kFloat));
        const auto r = expect_throw([&] {
            check_sf_layout(sf, mn, k, 1, gran_k, std::nullopt, true, false, torch::kInt);
        });
        report("B2 type_check=kInt 拒绝 fp32（layout.hpp:110；MegaMoE 用同一断言）", contains(r, ":110"), r);
    }
    // stride(-2)!=1（K-major 而非 MN-major）→ :126
    {
        // 一个明确的 K-major 张量：shape [mn, 4], contiguous strides {4,1}（k=512,gran_k=32）
        const auto km = torch::zeros({mn, 4}, at::TensorOptions().dtype(torch::kInt));
        const auto r2 = expect_throw([&] {
            check_sf_layout(km, mn, 512, 1, gran_k, std::nullopt, true, false, torch::kInt);
        });
        report("B3 stride(-2)!=1（K-major 的 [mn,4] 连续）被拒（:126）", contains(r2, ":126"), r2);
    }
    // stride(-1) < align(mn,4) → :128
    {
        const auto sf = torch::empty_strided({mn, packed_sf_k}, {1, 6},
                                             at::TensorOptions().dtype(torch::kInt));
        const auto r = expect_throw([&] {
            check_sf_layout(sf, mn, k, 1, gran_k, std::nullopt, true, false, torch::kInt);
        });
        report("B4 stride(-1)=6 < align(6,4)=8 被拒（:128）", contains(r, ":128"), r);
    }
    // stride(-1) % 4 != 0 → :129
    {
        const auto sf = torch::empty_strided({mn, packed_sf_k}, {1, 10},
                                             at::TensorOptions().dtype(torch::kInt));
        const auto r = expect_throw([&] {
            check_sf_layout(sf, mn, k, 1, gran_k, std::nullopt, true, false, torch::kInt);
        });
        report("B5 stride(-1)=10 % 4 != 0 被拒（:129）", contains(r, ":129"), r);
    }
}

// ============================================================================
// Part C：spec §5.2 K-grouped 预打包 int32 契约 ↔ check_k_grouped_packed_ue8m0_tensor:286-317
// ============================================================================
static void part_c_kgrouped() {
    std::printf("\n== Part C: spec §5.2 K-grouped 预打包 int32（gran_k 必须=32）==\n");
    const std::vector<int> ks = {2048, 4096, 1024};
    const int k_alignment = 128, gran_k = 32, mn = 8;
    int required = 0;
    for (auto k : ks) required += ceil_div(k, gran_k * 4);
    std::printf("  ks={2048,4096,1024}, gran_k=32 → required packed_sf_k=sum ceil_div(k,128)=%d\n", required);

    const auto layout = torch::tensor({2048, 4096, 1024}, at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));

    // 正确：sf [required, mn] 连续，mn%4==0，stride(0)=mn%4==0
    {
        const auto sf = torch::zeros({required, mn}, at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        const auto r = expect_throw([&] {
            check_k_grouped_packed_ue8m0_tensor(sf, layout, std::optional<std::vector<int>>(ks), gran_k, k_alignment, false);
        });
        report("C1 int32 [56,8] 连续 通过（:299 stride(0)>=mn, :300 %4==0）", r == "NO_THROW", r == "NO_THROW" ? "" : r);
    }
    // gran_k!=32 → :293
    {
        const auto sf = torch::zeros({28, mn}, at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        const auto r = expect_throw([&] {
            check_k_grouped_packed_ue8m0_tensor(sf, layout, std::optional<std::vector<int>>(ks), 128, k_alignment, false);
        });
        report("C2 gran_k=128 被拒（:293；gran_k=128 必须走 float32）", contains(r, ":293"), r);
    }
    // stride(0) < mn → :299
    {
        const auto sf = torch::empty_strided({required, mn}, {mn - 4, 1},
                                             at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        const auto r = expect_throw([&] {
            check_k_grouped_packed_ue8m0_tensor(sf, layout, std::optional<std::vector<int>>(ks), gran_k, k_alignment, false);
        });
        report("C3 stride(0)=mn-4 < mn 被拒（:299）", contains(r, ":299"), r);
    }
    // packed_sf_k < required → :312
    {
        const auto sf = torch::zeros({required - 1, mn}, at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        const auto r = expect_throw([&] {
            check_k_grouped_packed_ue8m0_tensor(sf, layout, std::optional<std::vector<int>>(ks), gran_k, k_alignment, false);
        });
        report("C4 packed_sf_k=55 < required=56 被拒（:312）", contains(r, ":312"), r);
    }
    // 更大 buffer 允许（trailing 行忽略）→ 通过（spec §5.2「larger is OK」）
    {
        const auto sf = torch::zeros({required + 4, mn}, at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        const auto r = expect_throw([&] {
            check_k_grouped_packed_ue8m0_tensor(sf, layout, std::optional<std::vector<int>>(ks), gran_k, k_alignment, false);
        });
        report("C5 packed_sf_k=60 > required（预分配更大 buffer）通过（:312 是 >=）", r == "NO_THROW", r == "NO_THROW" ? "" : r);
    }
    // ks_cpu 为空且 use_psum=false → :314
    {
        const auto sf = torch::zeros({required, mn}, at::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
        const auto r = expect_throw([&] {
            check_k_grouped_packed_ue8m0_tensor(sf, layout, std::optional<std::vector<int>>(), gran_k, k_alignment, false);
        });
        report("C6 ks_cpu 空且 use_psum=false 被拒（:314）", contains(r, ":314"), r);
    }
}

// ============================================================================
// Part D：spec §6.2 fp8_einsum 的「按 post-permute 坐标变换再换回」契约
//   —— 接收端最终仍调用 check_sf_layout(..., tma_stride_check=true, type_check=kInt)
// ============================================================================
static void part_d_einsum_permute() {
    std::printf("\n== Part D: spec §6.2 einsum 预变换 SFA 的「换回」契约 ==\n");
    // 'bhr,hdr->bhd'：einsum 内部把 SFA 从 [b,h,·] permute(1,0,2) 成 [h,b,·]，mn=b, num_groups=h
    const int b = 5, h = 4, r = 512, gran_k = 32;
    const int packed_sf_k = ceil_div(r, gran_k * 4);   // = 4
    const int align_b = get_tma_aligned_size(b, 4);    // = 8
    std::printf("  b=%d h=%d r=%d → packed_sf_k=%d, align(b,4)=%d\n", b, h, r, packed_sf_k, align_b);

    // 正确：按 post-permute 坐标 [h,b,packed] MN-major 建张量，再 permute(1,0,2) 回到 [b,h,packed]
    auto grouped = torch::empty_strided({h, b, packed_sf_k},
                                        {packed_sf_k * align_b, 1, align_b},
                                        at::TensorOptions().dtype(torch::kInt));
    auto user = grouped.permute({1, 0, 2});            // [b,h,packed] view
    // 接收端 permute(1,0,2) 回 [h,b,packed]，等价于在 grouped 上校验
    {
        const auto r_ok = expect_throw([&] {
            check_sf_layout(grouped, b, r, 1, gran_k, h, true, false, torch::kInt);
        });
        report("D1 换回后的 [h,b,packed] 满足 stride(-3)==stride(-1)*size(-1)（:124）", r_ok == "NO_THROW", r_ok == "NO_THROW" ? "" : r_ok);
    }
    // 错误：2D transform 后 reshape 成 [b,h,packed] 连续 → 内部 permute 后 stride(-2)!=1
    {
        auto flat = torch::empty_strided({b * h, packed_sf_k},
                                         {1, get_tma_aligned_size(b * h, 4)},
                                         at::TensorOptions().dtype(torch::kInt));
        auto reshaped = flat.reshape({b, h, packed_sf_k});   // 变连续，破坏 MN-major
        auto post = reshaped.permute({1, 0, 2});             // einsum 内部 permute(1,0,2)
        std::printf("    naive reshape 后 post-permute strides = (%ld, %ld, %ld)\n",
                    post.stride(0), post.stride(1), post.stride(2));
        const auto r_bad = expect_throw([&] {
            check_sf_layout(post, b, r, 1, gran_k, h, true, false, torch::kInt);
        });
        report("D2 2D transform + reshape 破坏 MN-major，被拒（:124/:126）",
               contains(r_bad, ":124") or contains(r_bad, ":126"), r_bad);
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("== P81-1 SF 契约一致性核验（H100/sm90，真实 host 头文件）==\n");
    std::printf("  csrc/utils/layout.hpp / csrc/jit_kernels/impls/smxx_layout.hpp\n");
    part_a_shape();
    part_b_pretransformed();
    part_c_kgrouped();
    part_d_einsum_permute();
    std::printf("\n== 汇总：%d PASS / %d FAIL ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
