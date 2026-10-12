// P16-1 实验：DeepGEMM **K-grouped SF（块级 scale）打包** 在 H100(sm90) 上直接编译运行真实设备代码。
//
// 背景：MoE 反向的 wgrad 是「K 轴分组」GEMM（`k_grouped_fp8_gemm_*`，每组专家收缩维 token 数不等长，
// 见 docs/12）。它的 FP8 操作数需要一个**按组拼接、按组独立打包**的 UE8M0 scale：
//   host `transform_k_grouped_sf_into_required_layout`（csrc/apis/layout.hpp:89-116）
//     -> SM100 FP32 分支 `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`
//        （csrc/jit_kernels/impls/smxx_layout.hpp:218-283）
//     -> device `pack_fp32_into_ue8m0<kNumGroups,...>`
//        （deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:193-293）
//   host 入口在 SM90 被 `layout.hpp:100-101` 直接拒绝（arch==10 专属），**但 device kernel 本身
//   与 UMMA/TMEM 无关**：它只做「找 owner 组 -> 逐 4 行打包指数」。因此本机可以在**绕过 host 断言**的
//   前提下，直接实例化并 launch 真实 kernel（smxx_layout.cuh:195），用真实数字回答：
//
//   A) num_groups=1 基线：k-grouped kernel 与「无分组」语义一致（退化情形）。
//   B) contiguous `ks_cpu` 路径（gran_k=32）：多组、组内 SF 行数非 4 倍数 -> **尾部 padding**、
//      下一组整体后移（num_padding_sf_rows），逐行对拍 host 参考。
//   C) contiguous `ks_cpu` 路径（gran_k=128）：不同 k_alignment、含空组（k=0）。
//   D) PSUM 路径：grouped_layout 存「逻辑 K 结束」前缀和、组间 gap 由 `math::align` 补齐，
//      逐行对拍；并验证 host 上界公式 packed_sf_k=(sf_k+num_groups*3)/4 足够大（尾部未用行不写）。
//   E) 输出布局：k-grouped 打包是 **[packed_sf_k, mn] 连续**（行距=mn），与「无分组打包」的
//      TMA 对齐行距（tma_aligned_mn）不同（smxx_layout.hpp:250 vs :136）。
//
// 参考实现全部在 host 侧按 kernel 逐行复刻（同一位运算），独立于 device 结果。
//
// 编译/运行见 harness/exp_kgrouped_sf_pack.sh（nvcc -arch=sm_90）。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>

// 为什么不是直接 `#include <deep_gemm/impls/smxx_layout.cuh>`：
// 该头文件里的 `transpose_fp32` 与 `transpose_and_pack_fp32_into_ue8m0` 各自声明了同名但
// 不同类型的 `extern __shared__ ... smem_buffer[]`（smxx_layout.cuh:20 vs :80），在同一个
// 翻译单元里 nvcc 会报 "declaration is incompatible"。DeepGEMM 的 JIT 每次只实例化一个 kernel，
// 且走的是更宽松的 NVRTC，故不受影响。这里由 harness/exp_kgrouped_sf_pack.sh 在编译前把
// **真实 kernel 逐行原样抽取**（smxx_layout.cuh:193-293）到生成文件，保证编译的就是仓库里的设备代码。
#include <kgrouped_pack_kernel.inc>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// 与 kernel 一致的打包（smxx_layout.cuh:287）：4 个 fp32 的 8 位指数 -> 1 个 int32，LSB-first
static inline uint32_t pack4(const uint32_t v[4]) {
    return (v[0] >> 23) | (v[1] >> 15) | (v[2] >> 7) | (v[3] << 1);
}

static inline int align_up(int a, int b) { return (a + b - 1) / b * b; }

// 构造只用指数的合法 SF：符号 0、尾数 0（kernel 用 DG_TRAP_ONLY_DEVICE_ASSERT 检查）
static std::vector<uint32_t> make_sf(int rows, int mn, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(120, 140);
    std::vector<uint32_t> v((size_t)rows * mn);
    for (auto& x : v) x = (uint32_t)dist(rng) << 23;
    return v;
}

// ---------------------------------------------------------------------------
// host 参考：逐行复刻 pack_fp32_into_ue8m0 的组归属 + padding + 打包（smxx_layout.cuh:224-292）
// grouped = contiguous: 每组 k_g；use_psum: 每组「逻辑 K 结束」前缀和。
// 返回 out[packed_sf_k*mn] 与「真实被写的 packed 行数 actual_rows」。
// ---------------------------------------------------------------------------
struct RefOut { std::vector<uint32_t> out; int actual_rows; int total_input_rows; };

static RefOut reference(const std::vector<uint32_t>& sf, int sf_k, int mn,
                        const std::vector<int>& grouped, int gran_k, int k_alignment,
                        bool use_psum, int packed_sf_k) {
    const int ng = (int)grouped.size();
    std::vector<int> g_start(ng), g_end(ng), g_npr(ng);
    int prefix_rows = 0, prefix_packed = 0, prev_end = 0;
    for (int g = 0; g < ng; ++g) {
        int group_k = use_psum ? (grouped[g] - align_up(prev_end, k_alignment)) : grouped[g];
        int aligned = use_psum ? align_up(group_k, k_alignment) : group_k;
        int rows = aligned / gran_k;
        g_start[g] = prefix_rows;
        g_end[g] = prefix_rows + rows;
        g_npr[g] = (rows + 3) / 4;
        prefix_rows += rows;
        prefix_packed += g_npr[g];
        if (use_psum) prev_end = grouped[g];
    }
    std::vector<uint32_t> out((size_t)packed_sf_k * mn, 0);
    int pad = 0, prefix_packed2 = 0;
    for (int g = 0; g < ng; ++g) {
        for (int local = 0; local < g_npr[g]; ++local) {
            const int pidx = prefix_packed2 + local;
            if (pidx >= packed_sf_k) continue;
            for (int m = 0; m < mn; ++m) {
                uint32_t vals[4];
                for (int j = 0; j < 4; ++j) {
                    int row = pidx * 4 + j - pad;
                    vals[j] = (row >= g_start[g] && row < g_end[g]) ? sf[(size_t)row * mn + m] : 0u;
                }
                out[(size_t)pidx * mn + m] = pack4(vals);
            }
        }
        prefix_packed2 += g_npr[g];
        int rem = (g_end[g] - g_start[g]) % 4;
        if (rem) pad += 4 - rem;
    }
    return {out, prefix_packed, prefix_rows};
}

// ---------------------------------------------------------------------------
// launch 真实 kernel（复刻 PackFP32IntoUE8M0Runtime::compile_and_launch 的 launch 配置，
// smxx_layout.hpp:15-42 / :245-274：block_mn=128、block_packed_sf_k=16、num_threads=512）
// ---------------------------------------------------------------------------
template <uint32_t NG, bool PSUM>
static std::vector<uint32_t> run_kernel(const std::vector<uint32_t>& sfbits, int sf_k, int mn,
                                        const std::vector<int>& grouped, int gran_k,
                                        int k_alignment, int packed_sf_k) {
    float* d_sf = nullptr;
    uint32_t* d_go = nullptr;
    uint32_t* d_out = nullptr;
    std::vector<uint32_t> go((size_t)std::max<size_t>(NG, grouped.size()), 0);
    for (size_t i = 0; i < grouped.size(); ++i) go[i] = (uint32_t)grouped[i];

    CK(cudaMalloc(&d_sf, sfbits.size() * 4));
    CK(cudaMalloc(&d_go, std::max<size_t>(NG, 1) * 4));
    CK(cudaMalloc(&d_out, (size_t)packed_sf_k * mn * 4));
    CK(cudaMemcpy(d_sf, sfbits.data(), sfbits.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_go, go.data(), std::max<size_t>(NG, 1) * 4, cudaMemcpyHostToDevice));
    // poison：未被任何组认领的尾部 packed 行应保持 0xAAAAAAAA（kernel 提前 return 不写）
    CK(cudaMemset(d_out, 0xAA, (size_t)packed_sf_k * mn * 4));

    constexpr int BLOCK_MN = 128, BLOCK_PACKED_SF_K = 16, NT = 512;
    dim3 grid((mn + BLOCK_MN - 1) / BLOCK_MN, (packed_sf_k + BLOCK_PACKED_SF_K - 1) / BLOCK_PACKED_SF_K);
    pack_fp32_into_ue8m0<NG, NT, BLOCK_MN, BLOCK_PACKED_SF_K, true, PSUM>
        <<<grid, NT>>>(reinterpret_cast<float*>(d_sf), d_out, d_go,
                       (uint32_t)mn, (uint32_t)sf_k, (uint32_t)packed_sf_k,
                       (uint32_t)gran_k, (uint32_t)k_alignment);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> out((size_t)packed_sf_k * mn);
    CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_sf); cudaFree(d_go); cudaFree(d_out);
    return out;
}

// ---------------------------------------------------------------------------
// 一个测试用例：跑 kernel + 参考，逐行比较
// ---------------------------------------------------------------------------
static bool compare_case(const char* name, int mn, int gran_k, int k_alignment,
                         const std::vector<int>& grouped, bool use_psum) {
    // 用参考先算出 input 行数与 packed 行数
    const int ng = (int)grouped.size();
    // 先算 input rows / actual packed rows
    int prefix_rows = 0, prefix_packed = 0, prev_end = 0;
    std::vector<int> g_rows(ng);
    for (int g = 0; g < ng; ++g) {
        int group_k = use_psum ? (grouped[g] - align_up(prev_end, k_alignment)) : grouped[g];
        int aligned = use_psum ? align_up(group_k, k_alignment) : group_k;
        g_rows[g] = aligned / gran_k;
        prefix_rows += g_rows[g];
        prefix_packed += (g_rows[g] + 3) / 4;
        if (use_psum) prev_end = grouped[g];
    }
    const int sf_k = prefix_rows;
    // host 计算的 packed_sf_k
    int packed_sf_k;
    if (use_psum) packed_sf_k = (sf_k + ng * 3) / 4;   // smxx_layout.hpp:248 上界
    else          packed_sf_k = prefix_packed;          // smxx_layout.hpp:243-247 精确值

    std::mt19937 rng(1234 + sf_k);
    auto sf = make_sf(sf_k, mn, rng);
    RefOut ref = reference(sf, sf_k, mn, grouped, gran_k, k_alignment, use_psum, packed_sf_k);

    auto got = [&]() {
        if (use_psum) {
            if (ng == 1) return run_kernel<1, true>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
            if (ng == 3) return run_kernel<3, true>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
            if (ng == 4) return run_kernel<4, true>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
            return run_kernel<8, true>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
        } else {
            if (ng == 1) return run_kernel<1, false>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
            if (ng == 3) return run_kernel<3, false>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
            if (ng == 4) return run_kernel<4, false>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
            return run_kernel<8, false>(sf, sf_k, mn, grouped, gran_k, k_alignment, packed_sf_k);
        }
    }();

    // 比较范围：kernel 实际会写 pidx < prefix_packed 的行；但 warp 是按 block 启动的，
    // blockIdx.y 覆盖到 packed_sf_k，owner 找不到就 return -> 尾部保持 poison。
    int mism = 0, poison_rows = 0, checked = 0;
    for (int p = 0; p < packed_sf_k; ++p) {
        bool authored = p < prefix_packed;
        for (int m = 0; m < mn; ++m) {
            uint32_t g = got[(size_t)p * mn + m];
            if (g == 0xAAAAAAAAu) {
                if (authored) ++mism;           // 应写却没写
                else ++poison_rows;
                continue;
            }
            if (authored) { ++checked; if (g != ref.out[(size_t)p * mn + m]) ++mism; }
        }
    }

    printf("  [%-34s] mn=%-4d gran_k=%-3d kalign=%-4d ng=%d %-6s sf_k=%-4d packed=%-3d(actual %d) "
           "mism=%d poison_rows=%d %s\n",
           name, mn, gran_k, k_alignment, ng, use_psum ? "PSUM" : "contig",
           sf_k, packed_sf_k, prefix_packed, mism, poison_rows, mism == 0 ? "PASS" : "FAIL");
    bool ok = (mism == 0);
    // 上界公式若宽松，尾部未用行必须保持 poison（kernel 不写）
    if (use_psum) {
        int expect_poison = (packed_sf_k - prefix_packed) * mn;
        if (poison_rows != expect_poison) {
            printf("      ^ 尾部 poison 行数不符：got %d want %d\n", poison_rows, expect_poison);
            ok = false;
        }
        printf("      PSUM 上界 packed=%d 实际用 %d -> 尾部 %d 个 int32 保持未初始化（poison）\n",
               packed_sf_k, prefix_packed, packed_sf_k - prefix_packed);
    }
    return ok;
}

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device=%s compute_cap=%d.%d\n", prop.name, prop.major, prop.minor);

    bool all = true;

    printf("\n== A) num_groups=1 基线（k-grouped kernel 退化 = 无分组语义）==\n");
    // 单组：sf_k 行，恰好 ceil(sf_k/4) 个 packed 行，无 padding
    all &= compare_case("A1 single full", 512, 32, 128, {256}, false);      // 256/32=8 行, ceil(8/4)=2
    all &= compare_case("A2 single rows%4!=0", 512, 32, 128, {96}, false);  // 3 行 -> 1 packed 行 + 1 pad

    printf("\n== B) contiguous ks_cpu，gran_k=32，组内行数非 4 倍数 -> padding 后移 ==\n");
    // k/gran_k: 8,12,4 -> packed: 2,3,1（恰好无 pad）；改成 8,10,6 -> 2,3(4-2=2 pad),2
    all &= compare_case("B1 no padding", 512, 32, 128, {256, 384, 128}, false);
    all &= compare_case("B2 with padding", 512, 32, 128, {256, 320, 192}, false); // 8,10,6 行
    all &= compare_case("B3 empty group", 512, 32, 128, {256, 0, 192}, false);     // 中间空组
    all &= compare_case("B4 8 groups", 512, 32, 128, {256, 384, 128, 512, 96, 128, 256, 64}, false);

    printf("\n== C) contiguous ks_cpu，gran_k=128，不同 k_alignment ==\n");
    all &= compare_case("C1 gran=128 kalign=128", 512, 128, 128, {384, 256, 512}, false);
    all &= compare_case("C2 gran=128 kalign=256", 512, 128, 256, {512, 256, 768}, false);
    all &= compare_case("C3 rows%4!=0", 512, 128, 128, {128, 256, 384}, false); // 1,2,3 行 -> 大量 pad

    printf("\n== D) PSUM 路径：grouped_layout 存逻辑 K 结束，组间 gap 由 align 补齐 ==\n");
    // gran_k=32, kalign=128: ends=[400,800,1024] -> rows 16,12,4 (见推导)
    all &= compare_case("D1 gran=32 kalign=128", 512, 32, 128, {400, 800, 1024}, true);
    // gran_k=128, kalign=256: ends=[256,640,1024] -> rows 2,4,2 (含 pad)
    all &= compare_case("D2 gran=128 kalign=256", 512, 128, 256, {256, 640, 1024}, true);

    printf("\n== E) 输出布局对比（host 侧，无 device 参与）==\n");
    printf("  k-grouped : out = torch::empty({packed_sf_k, mn}) -> stride (mn, 1) 连续（smxx_layout.hpp:250）\n");
    printf("  非分组     : out = empty_strided({...,mn,packed_sf_k}, {..,1,tma_aligned_mn})（smxx_layout.hpp:136）\n");
    printf("  即：k-grouped 的 K pitch = mn，非分组为 get_tma_aligned_size(mn,4)（16B 对齐）。\n");

    printf("\n== 汇总: %s ==\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
