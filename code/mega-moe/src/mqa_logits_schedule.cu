// P13-1 实验：SM100 dense MQA logits 的连续版 metadata 负载均衡调度器。
//
// 为什么能在 H100(sm90) 上跑：`deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh`
// 里的 `sm100_mqa_logits_metadata` kernel 与 `SM100MQALogitsScheduler` 只依赖 cute/cutlass
// 和 PDL（`cudaGridDependencySynchronize` / `launch_dependent_grids`，sm90 已有），
// 与 UMMA/tcgen05 无关，因此可以直接在 sm90 上编译运行真实设备代码。
// 主 kernel（`impls/sm100_mqa_logits.cuh` 的 UMMA 部分）是 SM100 专属，本实验不涉及。
//
// 本程序做的事：
//   1. 用真实 `sm100_mqa_logits_metadata<...>` 在 device 上生成 schedule_meta；
//   2. 用 host 端独立复刻（span 公式 / 前缀和 / cost 模型 / boundary locate）逐字段对拍；
//   3. 用真实 `SM100MQALogitsScheduler`（scheduled 与 grid-stride 两种模式）发射任务，
//      验证「所有有效 KV split 恰好被覆盖一次」；
//   4. 量化负载均衡收益：scheduled 模式 per-SM 工作量的最大/最小 vs grid-stride round-robin。
//
// 编译：见 harness/exp_mqa_logits_schedule.sh

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>

#include <cuda_runtime.h>

#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

using namespace deep_gemm;

// ---------------------------------------------------------------------------
// device：发射任务（scheduled / grid-stride 两种模式都实例化）
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs, bool kUseSchedule>
__global__ void emit_tasks_kernel(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                                  const uint32_t* __restrict__ cu_start,
                                  const uint32_t* __restrict__ cu_end,
                                  const uint32_t* __restrict__ schedule_meta,
                                  uint32_t* __restrict__ out_q_block,
                                  uint32_t* __restrict__ out_kv_base,
                                  uint32_t* __restrict__ out_num_splits,
                                  uint32_t* __restrict__ out_count,
                                  const uint32_t max_tasks_per_sm) {
    const uint32_t sm = blockIdx.x;
    sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs, kUseSchedule> scheduler(
        sm, num_q_tokens, num_kv_tokens, cu_start, cu_end, schedule_meta);
    sched::MQALogitsTask task;
    uint32_t cnt = 0;
    while (scheduler.next_q_block(task)) {
        const uint32_t base = sm * max_tasks_per_sm + cnt;
        out_q_block[base] = task.q_token_base / BLOCK_Q;
        out_kv_base[base] = task.kv_token_base;
        out_num_splits[base] = task.num_kv_splits;
        ++cnt;
    }
    out_count[sm] = cnt;
}

// ---------------------------------------------------------------------------
// host 端独立参考实现（不调用 device 代码，纯 C++）
// ---------------------------------------------------------------------------
struct HostSpan { uint32_t base; uint32_t splits; };

static HostSpan host_span(uint32_t q_block, uint32_t BLOCK_Q, uint32_t SPLIT_KV,
                          uint32_t num_q_tokens, uint32_t num_kv_tokens,
                          const std::vector<int>& ks, const std::vector<int>& ke) {
    uint32_t start = 0xffffffffu, end = 0u;
    for (uint32_t t = 0; t < BLOCK_Q; ++t) {
        const int r = std::min<int>(q_block * BLOCK_Q + t, num_q_tokens - 1);
        const uint32_t a = std::min<uint32_t>(ks[r], num_kv_tokens);
        const uint32_t b = std::min<uint32_t>(ke[r], num_kv_tokens);
        start = std::min(start, a);
        end = std::max(end, b);
    }
    const uint32_t base = start / 4 * 4;
    return HostSpan{base, (end > base) ? (uint32_t)((end - base + SPLIT_KV - 1) / SPLIT_KV) : 0u};
}

static uint32_t upper_bound_u32(const std::vector<uint32_t>& prefix, uint32_t target) {
    uint32_t lo = 0, hi = (uint32_t)prefix.size();
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (prefix[mid] <= target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

struct Boundary { uint32_t q_block; uint32_t split_offset; uint32_t work_offset; };

// ---------------------------------------------------------------------------
// 单个用例
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs, uint32_t SEG, uint32_t SPL>
static bool run_case(const std::string& name, uint32_t num_q_tokens, uint32_t num_kv_tokens,
                     const std::vector<int>& ks, const std::vector<int>& ke) {
    const uint32_t num_q_blocks = (num_q_tokens + BLOCK_Q - 1) / BLOCK_Q;
    const uint32_t span_word_offset = (3 * kNumSMs + 1) / 2 * 2;
    const uint32_t required_words = span_word_offset + 2 * num_q_blocks;

    // host 参考：spans / 前缀和 / cost 模型
    std::vector<HostSpan> spans(num_q_blocks);
    std::vector<uint32_t> work_prefix(num_q_blocks), cost_prefix(num_q_blocks);
    uint32_t work_acc = 0, cost_acc = 0;
    for (uint32_t b = 0; b < num_q_blocks; ++b) {
        spans[b] = host_span(b, BLOCK_Q, SPLIT_KV, num_q_tokens, num_kv_tokens, ks, ke);
        work_acc += spans[b].splits;
        cost_acc += SPL * spans[b].splits + (spans[b].splits == 0 ? 0 : SEG);
        work_prefix[b] = work_acc;
        cost_prefix[b] = cost_acc;
    }
    const uint32_t total_work = work_acc;
    const uint32_t total_cost = cost_acc;

    const uint32_t cost_per_sm = total_cost / kNumSMs;
    const uint32_t cost_remainder = total_cost % kNumSMs;
    auto locate = [&](uint32_t idx) -> Boundary {
        const uint32_t target = idx * cost_per_sm + std::min<uint32_t>(idx, cost_remainder);
        if (target == total_cost) return Boundary{num_q_blocks, 0, total_work};
        const uint32_t qb = upper_bound_u32(cost_prefix, target);
        const uint32_t cost_before = qb == 0 ? 0 : cost_prefix[qb - 1];
        const uint32_t work_before = qb == 0 ? 0 : work_prefix[qb - 1];
        const uint32_t split_offset = std::min<uint32_t>(
            (std::max<uint32_t>(target - cost_before, SEG) - SEG) / SPL, work_prefix[qb] - work_before - 1);
        return Boundary{qb, split_offset, work_before + split_offset};
    };

    // device 分配
    int *d_ks = nullptr, *d_ke = nullptr;
    uint32_t* d_meta = nullptr;
    cudaMalloc(&d_ks, sizeof(int) * ks.size());
    cudaMalloc(&d_ke, sizeof(int) * ke.size());
    cudaMalloc(&d_meta, sizeof(uint32_t) * required_words);
    cudaMemcpy(d_ks, ks.data(), sizeof(int) * ks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ke, ke.data(), sizeof(int) * ke.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_meta, 0, sizeof(uint32_t) * required_words);

    // 1) 真实 metadata kernel
    const uint32_t num_threads = 256;
    const int smem_size = (int)((2 * num_q_blocks + num_threads / 32) * sizeof(uint32_t));
    sched::sm100_mqa_logits_metadata<BLOCK_Q, SPLIT_KV, kNumSMs, SEG, SPL, num_threads>
        <<<1, num_threads, smem_size>>>(num_q_tokens, num_kv_tokens, (const uint32_t*)d_ks, (const uint32_t*)d_ke, d_meta);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("  [%s] metadata kernel FAILED: %s\n", name.c_str(), cudaGetErrorString(err)); return false; }

    std::vector<uint32_t> meta(required_words);
    cudaMemcpy(meta.data(), d_meta, sizeof(uint32_t) * required_words, cudaMemcpyDeviceToHost);

    bool ok = true;

    // 2) device spans 与 host 参考对拍
    int span_mismatch = 0;
    for (uint32_t b = 0; b < num_q_blocks; ++b) {
        const uint32_t* sp = &meta[span_word_offset + 2 * b];
        if (sp[0] != spans[b].base || sp[1] != spans[b].splits) ++span_mismatch;
    }
    ok &= (span_mismatch == 0);

    // 3) 每 SM 边界对拍
    int start_mismatch = 0, count_mismatch = 0;
    std::vector<uint32_t> sm_begin(kNumSMs), sm_count(kNumSMs);
    for (uint32_t sm = 0; sm < kNumSMs; ++sm) {
        const uint32_t s_qb = meta[2 * sm];
        const uint32_t s_off = meta[2 * sm + 1];
        const uint32_t s_cnt = meta[2 * kNumSMs + sm];
        const Boundary begin = locate(sm);
        const Boundary end = locate(sm + 1);
        if (s_qb != begin.q_block || s_off != begin.split_offset) ++start_mismatch;
        if (s_cnt != end.work_offset - begin.work_offset) ++count_mismatch;
        sm_begin[sm] = s_qb;
        sm_count[sm] = s_cnt;
    }
    ok &= (start_mismatch == 0 && count_mismatch == 0);

    // 4) 用真实 scheduler 发射任务并验证覆盖
    const uint32_t max_tasks = num_q_blocks;
    uint32_t *d_out_qb = nullptr, *d_out_kb = nullptr, *d_out_n = nullptr, *d_out_cnt = nullptr;
    const size_t out_elems = (size_t)kNumSMs * max_tasks;
    cudaMalloc(&d_out_qb, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_out_kb, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_out_n, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_out_cnt, sizeof(uint32_t) * kNumSMs);

    auto run_emit = [&](bool scheduled, std::vector<uint32_t>& hits, uint32_t& covered,
                        std::vector<uint32_t>& per_sm_work, uint32_t& tasks_total) {
        cudaMemset(d_out_cnt, 0, sizeof(uint32_t) * kNumSMs);
        if (scheduled)
            emit_tasks_kernel<BLOCK_Q, SPLIT_KV, kNumSMs, true><<<kNumSMs, 32>>>(
                num_q_tokens, num_kv_tokens, (const uint32_t*)d_ks, (const uint32_t*)d_ke, d_meta,
                d_out_qb, d_out_kb, d_out_n, d_out_cnt, max_tasks);
        else
            emit_tasks_kernel<BLOCK_Q, SPLIT_KV, kNumSMs, false><<<kNumSMs, 32>>>(
                num_q_tokens, num_kv_tokens, (const uint32_t*)d_ks, (const uint32_t*)d_ke, d_meta,
                d_out_qb, d_out_kb, d_out_n, d_out_cnt, max_tasks);
        cudaDeviceSynchronize();

        std::vector<uint32_t> host_qb(out_elems), host_kb(out_elems), host_n(out_elems), host_cnt(kNumSMs);
        cudaMemcpy(host_qb.data(), d_out_qb, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_kb.data(), d_out_kb, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_n.data(), d_out_n, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cnt.data(), d_out_cnt, sizeof(uint32_t) * kNumSMs, cudaMemcpyDeviceToHost);

        hits.assign(total_work ? total_work : 1, 0u);
        covered = 0;
        tasks_total = 0;
        per_sm_work.assign(kNumSMs, 0u);
        int bad_task = 0;
        for (uint32_t sm = 0; sm < kNumSMs; ++sm) {
            for (uint32_t i = 0; i < host_cnt[sm]; ++i) {
                const uint32_t base = sm * max_tasks + i;
                const uint32_t qb = host_qb[base];
                const uint32_t kvbase = host_kb[base];
                const uint32_t n = host_n[base];
                const HostSpan sp = spans[qb];
                const uint32_t gbase = qb == 0 ? 0 : work_prefix[qb - 1];
                if (n == 0) continue;
                if (kvbase < sp.base || (kvbase - sp.base) % SPLIT_KV != 0 ||
                    (kvbase - sp.base) / SPLIT_KV + n > sp.splits) { ++bad_task; continue; }
                const uint32_t off = (kvbase - sp.base) / SPLIT_KV;
                for (uint32_t j = 0; j < n; ++j) hits[gbase + off + j] += 1;
                per_sm_work[sm] += n;
                covered += n;
                ++tasks_total;
            }
        }
        int dup = 0, miss = 0;
        for (uint32_t g = 0; g < total_work; ++g) { if (hits[g] == 0) ++miss; else if (hits[g] > 1) ++dup; }
        return std::make_tuple(bad_task == 0, dup, miss);
    };

    std::vector<uint32_t> hits, per_sm_sched, per_sm_grid;
    uint32_t covered_sched = 0, covered_grid = 0, tasks_sched = 0, tasks_grid = 0;
    auto [good_sched, dup_s, miss_s] = run_emit(true, hits, covered_sched, per_sm_sched, tasks_sched);
    std::vector<uint32_t> hits2;
    auto [good_grid, dup_g, miss_g] = run_emit(false, hits2, covered_grid, per_sm_grid, tasks_grid);
    ok &= good_sched && dup_s == 0 && miss_s == 0 && covered_sched == total_work;
    ok &= good_grid && dup_g == 0 && miss_g == 0 && covered_grid == total_work;

    // 5) 均衡度
    auto maxmin = [](const std::vector<uint32_t>& v) {
        uint32_t mx = 0, mn = 0xffffffffu;
        for (auto x : v) { mx = std::max(mx, x); mn = std::min(mn, x); }
        return std::make_pair(mn, mx);
    };
    auto [smin, smax] = maxmin(per_sm_sched);
    auto [gmin, gmax] = maxmin(per_sm_grid);

    printf("  [%-22s] BQ=%u NQ=%u NKV=%u  blocks=%u  total_work=%u total_cost=%u seg:split=%u:%u\n",
           name.c_str(), BLOCK_Q, num_q_tokens, num_kv_tokens, num_q_blocks, total_work, total_cost, SEG, SPL);
    printf("      spans match=%s (%d mismatch)   per-SM boundary match=%s (start %d / count %d mismatch)\n",
           span_mismatch == 0 ? "YES" : "NO", span_mismatch,
           (start_mismatch == 0 && count_mismatch == 0) ? "YES" : "NO", start_mismatch, count_mismatch);
    printf("      scheduled : covered=%u/%u dup=%d miss=%d bad=%d  tasks=%u  per-SM work [%u..%u]\n",
           covered_sched, total_work, dup_s, miss_s, good_sched ? 0 : 1, tasks_sched, smin, smax);
    printf("      gridstride: covered=%u/%u dup=%d miss=%d bad=%d  tasks=%u  per-SM work [%u..%u]\n",
           covered_grid, total_work, dup_g, miss_g, good_grid ? 0 : 1, tasks_grid, gmin, gmax);
    printf("      => result: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_ks); cudaFree(d_ke); cudaFree(d_meta);
    cudaFree(d_out_qb); cudaFree(d_out_kb); cudaFree(d_out_n); cudaFree(d_out_cnt);
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("P13-1 SM100 dense MQA logits 调度器：H100 真实设备代码验证\n");
    printf("GPU: %s  sm_%d%d  SMs=%d\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // SM100 连续版常量（csrc/jit_kernels/impls/sm100_mqa_logits.hpp:13-15）
    constexpr uint32_t SEG = 3;   // kMQALogitsSegmentCost
    constexpr uint32_t SPL = 2;   // kMQALogitsSplitCost
    constexpr uint32_t SPLIT_KV = 384;

    // H100 有 132 个 SM；metadata 维度里 kNumSMs 必须与主 kernel 一致
    constexpr uint32_t SMS = 132;

    auto make_uniform = [](int T, int S, int align_off, std::vector<int>& ks, std::vector<int>& ke) {
        ks.assign(T, 0); ke.assign(T, S);
        for (int i = 0; i < T; ++i) ks[i] = align_off;   // 让起点非 4 对齐，测试 round-down
    };
    auto make_skew = [](int T, int S, std::vector<int>& ks, std::vector<int>& ke) {
        ks.assign(T, 0); ke.assign(T, 0);
        for (int i = 0; i < T; ++i) {
            if (i % 8 == 0) { ks[i] = 0; ke[i] = S; }                 // 长
            else if (i % 4 == 0) { ks[i] = S / 2; ke[i] = S; }        // 中
            else if (i % 7 == 0) { ks[i] = 0; ke[i] = 0; }            // 空
            else { ks[i] = S - 64; ke[i] = S; }                       // 短（<1 split）
        }
    };

    std::vector<int> ks, ke;
    bool all = true;

    printf("== 用例组 A：BLOCK_Q=2（H=64 的 Q tile）==\n");
    make_uniform(4096, 4096, 0, ks, ke);
    all &= run_case<2, SPLIT_KV, SMS, SEG, SPL>("uniform S=4096", 4096, 4096, ks, ke);

    make_uniform(4096, 4096, 5, ks, ke);
    all &= run_case<2, SPLIT_KV, SMS, SEG, SPL>("uniform kstart%4!=0", 4096, 4096, ks, ke);

    make_uniform(4096, 3970, 0, ks, ke);   // 尾部非 SPLIT_KV 整数倍
    all &= run_case<2, SPLIT_KV, SMS, SEG, SPL>("uniform tail", 4096, 3970, ks, ke);

    make_skew(4096, 8192, ks, ke);
    all &= run_case<2, SPLIT_KV, SMS, SEG, SPL>("skewed+empty S=8192", 4096, 8192, ks, ke);

    make_skew(4096, 700, ks, ke);
    all &= run_case<2, SPLIT_KV, SMS, SEG, SPL>("few-split skewed", 4096, 700, ks, ke);

    make_uniform(4103, 4096, 0, ks, ke);   // num_q_tokens 非 BLOCK_Q 整数倍（尾行 clamp）
    all &= run_case<2, SPLIT_KV, SMS, SEG, SPL>("ragged num_q_tokens", 4103, 4096, ks, ke);

    printf("\n== 用例组 B：BLOCK_Q=4（H=32 的 Q tile）==\n");
    make_uniform(4096, 4096, 0, ks, ke);
    all &= run_case<4, SPLIT_KV, SMS, SEG, SPL>("uniform S=4096", 4096, 4096, ks, ke);

    make_skew(4096, 8192, ks, ke);
    all &= run_case<4, SPLIT_KV, SMS, SEG, SPL>("skewed+empty S=8192", 4096, 8192, ks, ke);

    printf("\n== 用例组 C：少 SM（kNumSMs=8，便于观察边界）==\n");
    make_skew(128, 4096, ks, ke);
    all &= run_case<2, SPLIT_KV, 8, SEG, SPL>("skewed 8 SMs", 128, 4096, ks, ke);
    make_uniform(40, 3000, 1, ks, ke);
    all &= run_case<2, SPLIT_KV, 8, SEG, SPL>("tiny 8 SMs", 40, 3000, ks, ke);

    printf("\n== metadata 缓冲区字数公式 ==\n");
    for (uint32_t T : {4096u, 4103u}) {
        const uint32_t nqb = (T + 2 - 1) / 2;
        const uint32_t off = (3 * SMS + 1) / 2 * 2;
        printf("  T=%u: num_words = align(3*%u,2)+2*%u = %u+%u = %u words (%u bytes)\n",
               T, SMS, nqb, off, 2 * nqb, off + 2 * nqb, (off + 2 * nqb) * 4);
    }

    printf("\n%s\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
