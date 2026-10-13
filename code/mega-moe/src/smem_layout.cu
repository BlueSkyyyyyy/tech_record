// P82-1 实验：DeepGEMM 的共享内存布局结构层。
//
// 直接 include 两个此前从未被系统核验过的 layout 头文件，在 H100(sm90) 上编译它们的**真实定义**：
//   - deep_gemm/include/deep_gemm/layout/gemm.cuh        （SM100BF16GemmSharedStorage / SM100FP8FP4GemmSharedStorage）
//   - deep_gemm/include/deep_gemm/layout/mqa_logits.cuh  （MQALogitsSharedStorage）
//
// 这些结构体是**纯 host 可编译**的：只含 constexpr 推导、`alignas` 数组、cutlass mbarrier，
// 不含任何 UMMA/TMA 设备指令。故虽然使用它们的 kernel 是 SM100 专属，其**布局本身**可在 H100
// 上逐字段对拍——正好回答「每个 kernel 的 `extern __shared__` 到底怎么切、host 怎么算 smem_size」。
//
// 三组核验：
//   A) MQA：`sizeof(struct)` == host 镜像 `MQALogitsConfig::smem_size()`（后者正是 JIT 源码里
//      `static_assert(sizeof(layout::MQALogitsSharedStorage<...>) == smem_size, ...)` 的右值）。
//   B) GEMM：`sizeof(struct)` <= host 保守预算 `get_pipeline_config` 的 `smem_extra + stages*per_stage`
//      （后者用 `kNumMaxStages=32` 为 barrier 预留，故对实际 stages 留出松弛）。
//   C) offsetof / alignas 约束：CD 1024 对齐、A/B 1024 对齐、Q/KV swizzle 对齐、SF/weights 128 对齐。
//
// 编译与运行见 harness/exp_smem_layout.sh。

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include <deep_gemm/layout/gemm.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>

using namespace deep_gemm;

static int g_pass = 0, g_fail = 0;

static void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  [%s] %-56s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    ok ? ++ g_pass : ++ g_fail;
}

static constexpr int ialign(int x, int a) { return (x + a - 1) / a * a; }

// ===========================================================================
// Part A：MQALogitsSharedStorage —— 与 host 镜像逐字节一致
// ===========================================================================

// 逐字复刻 csrc/jit_kernels/impls/sm100_mqa_logits.hpp:168-194 的 MQALogitsConfig::smem_size()
static int mqa_smem_size_mirror(int num_heads, int head_dim, bool is_fp4,
                                int block_q, int umma_n, int num_q_stages, int num_kv_stages,
                                int num_tmem_stages, int split_kv) {
    constexpr uint32_t kNumBarrierBytes = 8;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kTmaAlignment = 128;
    const uint32_t num_qk_bytes_per_token = is_fp4 ? head_dim / 2 : head_dim;
    const uint32_t swizzle_alignment = 8 * num_qk_bytes_per_token;
    const uint32_t num_sf_q = ialign(umma_n, kNumUTCCPAlignedElems);
    const uint32_t num_sf_kv = ialign(split_kv, kNumUTCCPAlignedElems);
    const uint32_t num_weight_elements_per_row = ialign(num_heads * 2, 16u) / 2;
    const uint32_t num_weight_bytes_per_stage =
        ialign(static_cast<uint32_t>(block_q * num_weight_elements_per_row * 2), kTmaAlignment);

    uint32_t n = 0;
    auto add_region = [&](const uint32_t& bytes, const uint32_t& alignment) {
        n = ialign(n, alignment) + bytes;
    };
    add_region(num_q_stages * umma_n * num_qk_bytes_per_token, swizzle_alignment);
    add_region(num_kv_stages * split_kv * num_qk_bytes_per_token, swizzle_alignment);
    add_region(num_q_stages * num_sf_q * sizeof(uint32_t), kTmaAlignment);
    add_region(num_kv_stages * num_sf_kv * sizeof(uint32_t), kTmaAlignment);
    add_region(num_q_stages * num_weight_bytes_per_stage, kTmaAlignment);
    const uint32_t num_barriers = 3 * (num_q_stages + num_kv_stages) + 2 * num_tmem_stages;
    add_region(num_barriers * kNumBarrierBytes, kNumBarrierBytes);
    add_region(sizeof(uint32_t), alignof(uint32_t));
    return static_cast<int>(ialign(n, swizzle_alignment));
}

template <int HEADS, int DIM, bool FP4>
static void mqa_case() {
    constexpr int block_q = 128 / HEADS;               // get_mqa_logits_block_q
    constexpr int umma_n = ialign(block_q * HEADS, 8);
    constexpr int q_stages = FP4 ? 2 : 3;
    constexpr int kv_stages = FP4 ? 8 : 3;
    constexpr int split_kv = 384;                      // kMQALogitsSplitKV
    constexpr int tmem_stages = 3;                     // kMQALogitsNumTmemStages
    using D = std::conditional_t<FP4, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
    using S = layout::MQALogitsSharedStorage<HEADS, DIM, block_q, split_kv, umma_n,
                                             q_stages, kv_stages, tmem_stages, D>;
    const int host = mqa_smem_size_mirror(HEADS, DIM, FP4, block_q, umma_n, q_stages, kv_stages,
                                          tmem_stages, split_kv);
    const std::string tag = std::string("H=") + std::to_string(HEADS) + (FP4 ? " FP4" : " FP8") +
                            " D=" + std::to_string(DIM);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "sizeof=%d host_mirror=%d align=%d swz=%d q=%d kv=%d",
                  (int) sizeof(S), host, (int) alignof(S), (int) S::kSwizzleAlignment,
                  q_stages, kv_stages);
    check("A.sizeof==mirror  " + tag, (int) sizeof(S) == host, buf);

    // 派生常量的 host 口径一致性
    bool ok = S::kNumSFQ == ialign(umma_n, 128) and S::kNumSFKV == ialign(split_kv, 128) and
              S::kNumQElementsPerStage == umma_n * (FP4 ? DIM / 2 : DIM) / (int) sizeof(D);
    check("A.derived consts    " + tag, ok,
          "kNumSFQ=" + std::to_string(S::kNumSFQ) + " kNumSFKV=" + std::to_string(S::kNumSFKV) +
          " qElems/stage=" + std::to_string(S::kNumQElementsPerStage));

    // alignas 约束（offsetof 必须落在声明的对齐上）
    const bool aligned = ((uintptr_t) offsetof(S, smem_q) % S::kSwizzleAlignment == 0) and
                         ((uintptr_t) offsetof(S, smem_kv) % S::kSwizzleAlignment == 0) and
                         ((uintptr_t) offsetof(S, smem_sf_q) % S::kTmaAlignment == 0) and
                         ((uintptr_t) offsetof(S, smem_sf_kv) % S::kTmaAlignment == 0) and
                         ((uintptr_t) offsetof(S, smem_weights) % S::kTmaAlignment == 0) and
                         ((uintptr_t) offsetof(S, full_q_barriers) % alignof(typename S::Barrier) == 0);
    std::snprintf(buf, sizeof(buf), "off q=%zu kv=%zu sfq=%zu w=%zu",
                  offsetof(S, smem_q), offsetof(S, smem_kv), offsetof(S, smem_sf_q),
                  offsetof(S, smem_weights));
    check("A.alignas offsets   " + tag, aligned, buf);
}

static void part_a() {
    std::printf("\n=== Part A：MQALogitsSharedStorage（sizeof == host 镜像） ===\n");
    mqa_case<64, 128, false>();
    mqa_case<128, 128, false>();
    mqa_case<64, 64, false>();
    mqa_case<32, 128, false>();
    mqa_case<16, 128, false>();
    mqa_case<64, 128, true>();
    mqa_case<128, 64, true>();
    mqa_case<64, 64, true>();
}

// ===========================================================================
// Part B：SM100 GEMM SharedStorage —— sizeof <= host 保守预算
// ===========================================================================

// 逐字复刻 csrc/jit_kernels/heuristics/sm100.hpp:209-252 的 get_pipeline_config 预算
static int gemm_host_budget(bool bf16, int load_m, int load_n, int block_k, int store_m,
                            int swizzle_cd_mode, int num_tma_store_stages, int a_elem, int b_elem,
                            int pack_factor, int sf_block_m, int sf_block_n, int num_stages) {
    constexpr int kNumMaxStages = 32;
    const int smem_cd = store_m * swizzle_cd_mode * num_tma_store_stages;
    const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 3 + 8;   // = 824
    const int smem_tmem_ptr = 4;
    const int smem_a_per_stage = load_m * block_k * a_elem / pack_factor;
    const int smem_b_per_stage = load_n * block_k * b_elem / pack_factor;
    int smem_sfa_per_stage = 0, smem_sfb_per_stage = 0;
    if (not bf16) {
        smem_sfa_per_stage = sf_block_m * block_k / 32;
        smem_sfb_per_stage = sf_block_n * block_k / 32;
    }
    const int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
    return smem_extra + num_stages * smem_per_stage;
}

template <int LOAD_M, int LOAD_N, int BLOCK_K, int STORE_M, int SWZ_CD, int STAGES,
          int EPI, int TMA_ST, int SWZ_AB>
static void bf16_case() {
    constexpr int STORE_N = SWZ_CD / (int) sizeof(cutlass::bfloat16_t);
    using S = layout::SM100BF16GemmSharedStorage<STAGES, EPI, TMA_ST, LOAD_M, LOAD_N, BLOCK_K,
                                                 STORE_M, STORE_N, cutlass::bfloat16_t>;
    const int host = gemm_host_budget(true, LOAD_M, LOAD_N, BLOCK_K, STORE_M, SWZ_CD, TMA_ST,
                                      2, 2, 1, 0, 0, STAGES);
    // 实际占用 = 最后一个成员结束处（不含尾部补齐到 alignas(1024) 的结构体 sizeof 填充）
    const int used = (int)(offsetof(S, tmem_ptr) + sizeof(uint32_t));
    const int actual_barrier_bytes = (int)(3 * STAGES * 8 + 2 * EPI * 8 + 8);
    char buf[220];
    std::snprintf(buf, sizeof(buf), "used=%d <= host=%d (slack=%d) | sizeof=%d 尾部pad=%d barriers=%d",
                  used, host, host - used, (int) sizeof(S), (int) sizeof(S) - used,
                  actual_barrier_bytes);
    const std::string tag = "BF16 L" + std::to_string(LOAD_M) + "x" + std::to_string(LOAD_N) +
                            " K" + std::to_string(BLOCK_K) + " S" + std::to_string(STAGES);
    check("B.used<=host         " + tag, used <= host, buf);

    const bool aligned = ((uintptr_t) offsetof(S, cd) % 1024 == 0) and
                         ((uintptr_t) offsetof(S, a) % 1024 == 0) and
                         ((uintptr_t) offsetof(S, b) % 1024 == 0) and
                         sizeof(typename S::a_stage_t) % 1024 == 0 and
                         sizeof(typename S::b_stage_t) % 1024 == 0;
    std::snprintf(buf, sizeof(buf), "off cd=%zu a=%zu b=%zu", offsetof(S, cd), offsetof(S, a),
                  offsetof(S, b));
    check("B.1024 alignment     " + tag, aligned, buf);
    (void) SWZ_AB;
}

template <int LOAD_M, int LOAD_N, int BLOCK_K, int STORE_M, int SWZ_CD, int STAGES,
          int EPI, int TMA_ST, bool FP4>
static void fp8fp4_case() {
    constexpr int STORE_N = SWZ_CD / (int) sizeof(cutlass::bfloat16_t);
    // 无 multicast 时 LOAD == BLOCK，故 SF_BLOCK = align(BLOCK, 128)
    constexpr int SF_BLOCK_M = ialign(LOAD_M, 128);
    constexpr int SF_BLOCK_N = ialign(LOAD_N, 128);
    constexpr int SF_BLOCK_K = BLOCK_K / 128;
    constexpr int PACK = FP4 ? 2 : 1;
    using A = std::conditional_t<FP4, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
    using S = layout::SM100FP8FP4GemmSharedStorage<STAGES, EPI, TMA_ST, LOAD_M, LOAD_N, BLOCK_K,
                                                   STORE_M, STORE_N, SF_BLOCK_M, SF_BLOCK_N,
                                                   SF_BLOCK_K, A, A, cutlass::bfloat16_t>;
    const int host = gemm_host_budget(false, LOAD_M, LOAD_N, BLOCK_K, STORE_M, SWZ_CD, TMA_ST,
                                      1, 1, PACK, SF_BLOCK_M, SF_BLOCK_N, STAGES);
    const int used = (int)(offsetof(S, tmem_ptr) + sizeof(uint32_t));
    char buf[240];
    std::snprintf(buf, sizeof(buf), "used=%d <= host=%d (slack=%d) | sizeof=%d pad=%d SF=%d,%d,%d pack=%d",
                  used, host, host - used, (int) sizeof(S), (int) sizeof(S) - used,
                  SF_BLOCK_M, SF_BLOCK_N, SF_BLOCK_K, PACK);
    const std::string tag = (FP4 ? "FP4 " : "FP8 ") + ("L" + std::to_string(LOAD_M) + "x" +
                            std::to_string(LOAD_N)) + " K" + std::to_string(BLOCK_K) +
                            " S" + std::to_string(STAGES);
    check("B.used<=host         " + tag, used <= host, buf);

    const bool sf_ok = (sizeof(typename S::sfa_stage_t) == (size_t)(SF_BLOCK_M * BLOCK_K / 32)) and
                       (sizeof(typename S::sfb_stage_t) == (size_t)(SF_BLOCK_N * BLOCK_K / 32));
    std::snprintf(buf, sizeof(buf), "sfa/stage=%zu sfb/stage=%zu (uint32 count %d/%d)",
                  sizeof(typename S::sfa_stage_t), sizeof(typename S::sfb_stage_t),
                  SF_BLOCK_M * SF_BLOCK_K, SF_BLOCK_N * SF_BLOCK_K);
    check("B.SF stage consistent " + tag, sf_ok, buf);
}

static void part_b() {
    std::printf("\n=== Part B：SM100 GEMM SharedStorage（实际占用 <= host 保守预算） ===\n");
    // BF16：block 128x128、BLOCK_K=64、STORE_M=128、swizzle_cd=128B → STORE_N=64
    bf16_case<128, 128, 64, 128, 128, 8, 2, 2, 0>();
    bf16_case<128, 128, 64, 128, 128, 16, 2, 2, 0>();
    bf16_case<128, 256, 64, 128, 128, 8, 2, 2, 0>();
    bf16_case<64, 128, 64, 64, 128, 12, 2, 2, 0>();
    // FP8：BLOCK_K=128、A/B e4m3、SF 128-aligned；pack=1
    fp8fp4_case<128, 128, 128, 128, 128, 8, 2, 2, false>();
    fp8fp4_case<128, 128, 128, 128, 128, 16, 2, 2, false>();
    fp8fp4_case<128, 256, 128, 128, 128, 8, 2, 2, false>();
    // FP4：BLOCK_K=256、pack=2
    fp8fp4_case<128, 128, 256, 128, 128, 4, 2, 2, true>();
}

// ===========================================================================
// Part C：reserved_barriers 的「barrier 布局对齐」效应
// ===========================================================================
static void part_c() {
    std::printf("\n=== Part C：BF16 vs FP8FP4 的 barrier 区域尺寸（reserved_barriers 的作用） ===\n");
    constexpr int S = 8, EPI = 2;
    using BF = layout::SM100BF16GemmSharedStorage<S, EPI, 2, 128, 128, 64, 128, 64, cutlass::bfloat16_t>;
    using FP = layout::SM100FP8FP4GemmSharedStorage<S, EPI, 2, 128, 128, 128, 128, 64, 128, 128, 1,
                                                    cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                    cutlass::bfloat16_t>;
    // 每 stage barrier 数：BF16 = full + empty + reserved；FP8 = full + sf_full + empty
    const int bf_per_stage = 3, fp_per_stage = 3;
    const int bf_total = 3 * S * 8 + 2 * EPI * 8 + 8;      // + tensor_core_full_barrier
    const int fp_total = 3 * S * 8 + 3 * EPI * 8;
    char buf[200];
    std::snprintf(buf, sizeof(buf), "per-stage BF16=%d FP8=%d; total bytes BF16=%d FP8=%d",
                  bf_per_stage, fp_per_stage, bf_total, fp_total);
    check("C.per-stage barrier count 3 both", bf_per_stage == fp_per_stage, buf);
    std::printf("  BF16 struct: sizeof=%d, offsetof(full_barriers)=%zu, offsetof(reserved)=%zu\n",
                (int) sizeof(BF), offsetof(BF, full_barriers), offsetof(BF, reserved_barriers));
    std::printf("  FP8  struct: sizeof=%d, offsetof(full_barriers)=%zu, offsetof(sf_full)=%zu\n",
                (int) sizeof(FP), offsetof(FP, full_barriers), offsetof(FP, sf_full_barriers));
    check("C.host barrier budget >= both", (3 * 32 * 8 + 2 * 8 * 3 + 8) >= bf_total and
                                            (3 * 32 * 8 + 2 * 8 * 3 + 8) >= fp_total,
          "host=824");
}

int main() {
    part_a();
    part_b();
    part_c();
    std::printf("\n==== 总判定：PASS=%d FAIL=%d ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
