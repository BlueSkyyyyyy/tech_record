// P5-1 实验：Mega Gate 的 warp 级 top-k 路由算法在 H100(sm90) 上的忠实移植与对拍。
//
// Mega Gate 主体（sm100_bf16_mega_gate.cuh、epilogue/sm100_mega_gate.cuh）是 SM100 专属，
// 但其**路由数值语义**与 warp 级选择算法是架构无关的。本文件把以下源码逐行搬到裸 CUDA：
//
//   - sqrt_softplus 标量 / 向量版            (epilogue/sm100_mega_gate.cuh:19-37)
//   - sort4_best_first / compare_swap        (epilogue:88-105)
//   - select_warp_topk（3-bit 游标归并）      (epilogue:107-152)
//   - warp_reduce_best（max + 最小专家号破平）(epilogue:80-86)
//   - select_warp_expert_value                (epilogue:154-170)
//   - sqrt_softplus 阈值 20 的近似分支        (epilogue:19-23)
//
// 参考实现（host 侧 Python/torch）见 harness/exp_mega_gate_routing.py。
// 编译/运行见 harness/exp_mega_gate_routing.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

#define DEV __device__ __forceinline__

// ---------------------------------------------------------------------------
// 源码移植：epilogue/sm100_mega_gate.cuh
// ---------------------------------------------------------------------------
DEV float sqrt_softplus(const float& value) {
    const auto softplus_precise = log1pf(expf(value));
    const auto softplus = value > 20.0f ? value : softplus_precise;
    return sqrtf(softplus);
}

template <uint32_t kNumValues>
DEV void sqrt_softplus(float (&values)[kNumValues]) {
    float exponentials[kNumValues], softplus[kNumValues];
    #pragma unroll
    for (uint32_t i = 0; i < kNumValues; ++i) exponentials[i] = expf(values[i]);
    #pragma unroll
    for (uint32_t i = 0; i < kNumValues; ++i) softplus[i] = log1pf(exponentials[i]);
    #pragma unroll
    for (uint32_t i = 0; i < kNumValues; ++i) values[i] = sqrtf(values[i] > 20.0f ? values[i] : softplus[i]);
}

DEV unsigned get_lane_idx() { unsigned l; asm("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }

// DeepGEMM 用 `redux.sync.max.f32`（`ptx/utils.cuh:49-53`）；该指令在 Hopper 上不可用，
// 这里用 shfl_xor 蝶式归约得到**同语义**的最大值（fmaxf 无 NaN 参与，结果一致）。
DEV float reduce_max_sync(const float& value) {
    float v = value;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

DEV void warp_reduce_best(const float& score, int& expert_idx) {
    const auto best_score = reduce_max_sync(score);
    const auto tied_expert_idx = (score == best_score and expert_idx >= 0)
                                     ? static_cast<uint32_t>(expert_idx) : 0xffffffffu;
    expert_idx = static_cast<int>(__reduce_min_sync(0xffffffffu, tied_expert_idx));
}

DEV void compare_swap_best_first(float& lhs_score, int& lhs_expert_idx, float& rhs_score, int& rhs_expert_idx) {
    if (rhs_score > lhs_score or (rhs_score == lhs_score and rhs_expert_idx < lhs_expert_idx)) {
        const auto score = lhs_score;
        const auto expert_idx = lhs_expert_idx;
        lhs_score = rhs_score, lhs_expert_idx = rhs_expert_idx;
        rhs_score = score, rhs_expert_idx = expert_idx;
    }
}

DEV uint32_t sort4_best_first(float& s0, float& s1, float& s2, float& s3) {
    int i0 = 0, i1 = 1, i2 = 2, i3 = 3;
    compare_swap_best_first(s0, i0, s1, i1);
    compare_swap_best_first(s2, i2, s3, i3);
    compare_swap_best_first(s0, i0, s2, i2);
    compare_swap_best_first(s1, i1, s3, i3);
    compare_swap_best_first(s1, i1, s2, i2);
    return static_cast<uint32_t>(i0 | i1 << 2 | i2 << 4 | i3 << 6);
}

static constexpr uint32_t kNumExpertsPerLaneVector = 4;
static constexpr uint32_t kNumExpertsPerWave = 32 * kNumExpertsPerLaneVector;  // 128

template <uint32_t kNumExpertWaves, uint32_t kNumTopk>
DEV void select_warp_topk(float (&scores)[kNumExpertWaves * kNumExpertsPerLaneVector], int& selected_expert_idx) {
    uint32_t permutations[kNumExpertWaves];
    #pragma unroll
    for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
        const auto base = w * kNumExpertsPerLaneVector;
        permutations[w] = sort4_best_first(scores[base + 0], scores[base + 1], scores[base + 2], scores[base + 3]);
    }
    const auto lane_idx = get_lane_idx();
    uint32_t packed_cursors = 0;
    #pragma unroll
    for (uint32_t output_idx = 0; output_idx < kNumTopk; ++output_idx) {
        auto best_score = -INFINITY;
        int best_expert_idx = -1;
        uint32_t best_expert_wave_idx = 0;
        #pragma unroll
        for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
            const auto cursor = packed_cursors >> (w * 3) & 7;
            const auto base = w * kNumExpertsPerLaneVector;
            const auto candidate_score = cursor == 4 ? -INFINITY :
                (cursor & 2 ? (cursor & 1 ? scores[base + 3] : scores[base + 2])
                            : (cursor & 1 ? scores[base + 1] : scores[base + 0]));
            const auto candidate_offset = permutations[w] >> (cursor * 2) & 3;
            const auto candidate_expert_idx = static_cast<int>(w * kNumExpertsPerWave +
                                                               lane_idx * kNumExpertsPerLaneVector + candidate_offset);
            if (candidate_score > best_score) {
                best_score = candidate_score;
                best_expert_idx = candidate_expert_idx;
                best_expert_wave_idx = w;
            }
        }
        const auto local_best_expert_idx = best_expert_idx;
        warp_reduce_best(best_score, best_expert_idx);
        if (local_best_expert_idx == best_expert_idx)
            packed_cursors += 1u << (best_expert_wave_idx * 3);
        if (lane_idx == output_idx)
            selected_expert_idx = best_expert_idx;
    }
}

template <uint32_t kNumExpertWaves>
DEV float select_warp_expert_value(const float (&values)[kNumExpertWaves * kNumExpertsPerLaneVector],
                                   const int& expert_idx) {
    const auto selected_value_idx = expert_idx >= 0 ?
        expert_idx / static_cast<int>(kNumExpertsPerWave) * static_cast<int>(kNumExpertsPerLaneVector) +
        expert_idx % static_cast<int>(kNumExpertsPerLaneVector) : -1;
    const auto source_lane_idx = expert_idx >= 0 ?
        static_cast<uint32_t>(expert_idx % static_cast<int>(kNumExpertsPerWave) / static_cast<int>(kNumExpertsPerLaneVector)) : 0u;
    auto selected_value = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertWaves * kNumExpertsPerLaneVector; ++i) {
        const auto exchanged = __shfl_sync(0xffffffffu, values[i], static_cast<int>(source_lane_idx));
        if (selected_value_idx == static_cast<int>(i)) selected_value = exchanged;
    }
    return selected_value;
}

template <uint32_t kNumLanesPerGroup>
DEV float warp_reduce_sum(float value) {
    #pragma unroll
    for (int off = kNumLanesPerGroup / 2; off > 0; off >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, off);
    return value;
}

// ---------------------------------------------------------------------------
// 实验 1：sqrt_softplus 数值（标量路径）——对拍 double 参考，标出阈值 20 的分支
// ---------------------------------------------------------------------------
__global__ void ssp_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sqrt_softplus(x[i]);
}

// ---------------------------------------------------------------------------
// 实验 2：完整路由——logits -> sqrt_softplus + bias 排名 -> warp 级 top-k -> 归一化权重
//   score   = sqrt_softplus(logit)          (unbiased, 用于权重)
//   ranking = score + bias                  (用于排名)
//   weight  = score_sel / sum(score_sel) * routed_scaling_factor
// 对应 epilogue:107-152（选择）、:154-216（取值与权重）。
// ---------------------------------------------------------------------------
template <uint32_t kNumExpertWaves, uint32_t kNumTopk>
__global__ void gate_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                  int num_tokens, int num_experts, float scaling,
                                  long long* __restrict__ out_idx, float* __restrict__ out_w) {
    const auto lane_idx = get_lane_idx();
    const int token = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (token >= num_tokens) return;

    float unbiased[kNumExpertWaves * kNumExpertsPerLaneVector];
    float ranking[kNumExpertWaves * kNumExpertsPerLaneVector];
    #pragma unroll
    for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
        const auto base = w * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
        const auto cell = w * kNumExpertsPerLaneVector;
        #pragma unroll
        for (uint32_t v = 0; v < kNumExpertsPerLaneVector; ++v) {
            const int e = static_cast<int>(base) + static_cast<int>(v);
            const float logit = e < num_experts ? logits[static_cast<long long>(token) * num_experts + e] : -INFINITY;
            const bool is_valid_expert = logit != -INFINITY;
            unbiased[cell + v] = sqrt_softplus(logit);
            // 对应 epilogue:397-398：无效（padding）专家的排名置 -inf，永不被选中
            ranking[cell + v]  = is_valid_expert
                ? unbiased[cell + v] + (bias ? bias[e] : 0.0f)
                : -INFINITY;
        }
    }

    int selected_expert_idx = -1;
    select_warp_topk<kNumExpertWaves, kNumTopk>(ranking, selected_expert_idx);
    float selected_unbiased = select_warp_expert_value<kNumExpertWaves>(unbiased, selected_expert_idx);

    constexpr uint32_t kSumWidth = kNumTopk <= 8 ? 8 : (kNumTopk <= 16 ? 16 : 32);
    const auto topk_sum = warp_reduce_sum<kSumWidth>(selected_unbiased) + 1e-20f;
    if (lane_idx < kNumTopk) {
        out_idx[static_cast<long long>(token) * kNumTopk + lane_idx] = selected_expert_idx;
        out_w[static_cast<long long>(token) * kNumTopk + lane_idx] =
            selected_unbiased / topk_sum * scaling;
    }
}

// ---------------------------------------------------------------------------
// host 侧
// ---------------------------------------------------------------------------
#define CUDA_CHECK(x) do { cudaError_t _e = (x); \
    if (_e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

static void exp_sqrt_softplus() {
    const int n = 1 << 22;
    std::vector<float> hx(n);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-60.0f, 60.0f);
    for (int i = 0; i < n; ++i) hx[i] = dist(rng);
    // 覆盖阈值附近的精确点
    const float probes[] = {-60.f, -20.f, -1.f, 0.f, 1.f, 19.9999f, 20.f, 20.0001f, 40.f, 60.f};
    for (int i = 0; i < 10; ++i) hx[i] = probes[i];

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    ssp_kernel<<<(n + 255) / 256, 256>>>(dx, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hy(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs = 0, max_rel = 0;
    int num_over_threshold = 0;
    for (int i = 0; i < n; ++i) {
        double x = hx[i];
        double ref = std::sqrt(std::log1p(std::exp(x)));   // 精确 double 参考
        double abs_err = std::fabs((double)hy[i] - ref);
        double rel_err = abs_err / std::max(std::fabs(ref), 1e-30);
        if (x > 20.0) { num_over_threshold++; }
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }
    printf("== sqrt_softplus (n=%d, x~U[-60,60]) ==\n", n);
    printf("   max_abs_err vs double = %.3e, max_rel_err = %.3e\n", max_abs, max_rel);
    printf("   x>20 (走阈值分支, 返回 sqrt(x)) : %d / %d (%.1f%%)\n",
           num_over_threshold, n, 100.0 * num_over_threshold / n);
    printf("   阈值探针 (x -> ssp):");
    for (int i = 0; i < 10; ++i) printf(" %.4f->%.6f", probes[i], hy[i]);
    printf("\n\n");
    cudaFree(dx); cudaFree(dy);
}

// 参考实现：在 host 上按 (ranking desc, expert_idx asc) 排序取 top-k
struct Candidate { double ranking; double score; int idx; };

static void reference_topk(const std::vector<float>& logits, const std::vector<float>& bias,
                           int E, int K, std::vector<int>& idx_out, std::vector<double>& w_out) {
    std::vector<Candidate> cand(E);
    for (int e = 0; e < E; ++e) {
        const double lg = logits[e];
        const double sp = std::sqrt(std::log1p(std::exp(lg)));
        const double s = lg > 20.0 ? std::sqrt(lg > 20.0 ? (double)lg : sp) : sp;  // 与 kernel 同分支
        cand[e] = { s + (bias.empty() ? 0.0 : (double)bias[e]), s, e };
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) {
        if (a.ranking != b.ranking) return a.ranking > b.ranking;
        return a.idx < b.idx;   // 破平：最小专家号
    });
    double sum = 0;
    for (int i = 0; i < K; ++i) sum += cand[i].score;
    idx_out.resize(K); w_out.resize(K);
    for (int i = 0; i < K; ++i) { idx_out[i] = cand[i].idx; w_out[i] = cand[i].score / (sum + 1e-20); }
}

template <uint32_t WAVES, uint32_t K>
static void run_route_case(const char* tag, const std::vector<float>& h_logits, const std::vector<float>& h_bias,
                           int T, int E, float scaling, int num_experts_real) {
    // 用真实 E 计算（WAVES*128 需 >= E）；超出真实专家的位置用 -inf。
    const int total_experts = WAVES * 128;
    float *d_logits, *d_bias; long long* d_idx; float* d_w;
    std::vector<float> logits(total_experts, -INFINITY);
    for (int e = 0; e < E; ++e) logits[e] = h_logits[e];
    std::vector<float> bias = h_bias; bias.resize(total_experts, 0.0f);
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)T * total_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, total_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)T * K * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_w, (size_t)T * K * sizeof(float)));
    std::vector<float> rep(T * total_experts);
    for (int t = 0; t < T; ++t) std::copy(logits.begin(), logits.end(), rep.begin() + (size_t)t * total_experts);
    CUDA_CHECK(cudaMemcpy(d_logits, rep.data(), rep.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), total_experts * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (T * 32 + threads - 1) / threads;
    gate_route_kernel<WAVES, K><<<blocks, threads>>>(d_logits, d_bias, T, total_experts, scaling, d_idx, d_w);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> h_idx(T * K); std::vector<float> h_w(T * K);
    CUDA_CHECK(cudaMemcpy(h_idx.data(), d_idx, h_idx.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(), d_w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int idx_mismatch = 0; double max_w_rel = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> ridx; std::vector<double> rw;
        reference_topk(logits, bias, num_experts_real, K, ridx, rw);
        for (int i = 0; i < K; ++i) {
            if ((int)h_idx[t * K + i] != ridx[i]) idx_mismatch++;
            double d = std::fabs((double)h_w[t * K + i] - rw[i] * scaling);
            max_w_rel = std::max(max_w_rel, d / std::max(std::fabs(rw[i] * scaling), 1e-30));
        }
    }
    printf("== %s (waves=%u, K=%u, E=%d, T=%d, scaling=%.3f) ==\n", tag, WAVES, K, num_experts_real, T, scaling);
    printf("   top-k 索引与参考不符: %d / %d\n", idx_mismatch, T * K);
    printf("   权重 max_rel_err(vs ref*scaling): %.3e\n\n", max_w_rel);
    cudaFree(d_logits); cudaFree(d_bias); cudaFree(d_idx); cudaFree(d_w);
}

// 实验 3：重复专家（physical map）的 token 分散公式
//   源码 epilogue:205 `duplicate_idx = (rank_idx + global_token_idx * 23333u) % num_duplicates`
//   对每个 logical 专家的 d 个 duplicate，验证各 duplicate 分到的 token 数是否均衡（按 rank）。
static void exp_duplicate_map() {
    const int T = 1 << 16;
    printf("== duplicate spread: (rank + t*23333) %% d  (epilogue:205) ==\n");
    for (int d : {1, 2, 4, 32}) {
        int worst_max = 0, worst_min = T;
        for (int rank = 0; rank < 8; ++rank) {
            std::vector<int> cnt(d, 0);
            for (int t = 0; t < T; ++t) cnt[(rank + t * 23333u) % d]++;
            int cmin = *std::min_element(cnt.begin(), cnt.end());
            int cmax = *std::max_element(cnt.begin(), cnt.end());
            worst_max = std::max(worst_max, cmax);
            worst_min = std::min(worst_min, cmin);
        }
        printf("   d=%2d : 每个 duplicate 分到 token 数 min=%d max=%d (理想 %d), 最大/最小=%.4f\n",
               d, worst_min, worst_max, T / d, (double)worst_max / std::max(worst_min, 1));
    }
    printf("\n");
}

int main() {
    printf("Mega Gate routing algorithm port -- H100(sm90) numeric checks\n");
    printf("(Mega Gate 主体 sm100_bf16_mega_gate.cuh 为 SM100 专属，本实验仅移植其架构无关的路由语义)\n\n");
    exp_sqrt_softplus();
    exp_duplicate_map();

    const int T = 4096;
    std::mt19937 rng(7);

    // 案例 A：128 专家（1 wave），K=6，无 bias，随机 logits（间隔大，选择唯一）
    {
        const int E = 128;
        std::vector<float> logits(E); std::uniform_real_distribution<float> d(-10.f, 10.f);
        for (auto& v : logits) v = d(rng);
        run_route_case<1, 6>("A random no-bias", logits, {}, T, E, 1.5f, E);
    }
    // 案例 B：128 专家，K=8，带 bias（排名用 score+bias，权重用 score）
    {
        const int E = 128;
        std::vector<float> logits(E), bias(E);
        std::uniform_real_distribution<float> d(-5.f, 5.f), db(-1.f, 1.f);
        for (auto& v : logits) v = d(rng);
        for (auto& v : bias) v = db(rng);
        run_route_case<1, 8>("B random with-bias", logits, bias, T, E, 1.5f, E);
    }
    // 案例 C：重平局——logits 取小整数，大量 ranking 相等，验证破平=最小专家号
    {
        const int E = 128;
        std::vector<float> logits(E);
        for (int e = 0; e < E; ++e) logits[e] = (float)(rng() % 3);  // {0,1,2} 大量并列
        run_route_case<1, 6>("C heavy-ties", logits, {}, T, E, 1.0f, E);
    }
    // 案例 D：3 waves（384 专家，模拟目标卡 256 路由专家 + padding），K=6，带 bias
    {
        const int E = 384;
        std::vector<float> logits(E), bias(E);
        std::uniform_real_distribution<float> d(-8.f, 8.f), db(-0.5f, 0.5f);
        for (auto& v : logits) v = d(rng);
        for (auto& v : bias) v = db(rng);
        run_route_case<3, 6>("D 3-waves with-bias", logits, bias, T / 4, E, 1.5f, E);
    }
    // 案例 E：padding——4 waves 但只填 300 个真实专家，其余 -inf，K=6
    {
        const int E = 300;
        std::vector<float> logits(4 * 128), bias(4 * 128, 0.0f);
        std::uniform_real_distribution<float> d(-8.f, 8.f);
        for (int e = 0; e < E; ++e) logits[e] = d(rng);
        for (int e = E; e < 4 * 128; ++e) logits[e] = -INFINITY;
        // run_route_case 会把 >E 的位置覆盖为 -inf，这里直接调用内部对其真实 E=384 的参考
        run_route_case<4, 6>("E 4-waves padded(300/512)", logits, bias, T / 4, 4 * 128, 1.5f, E);
    }
    return 0;
}
