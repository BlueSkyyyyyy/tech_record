// P14-1 实验：SM100 dense MQA logits 的【分页版】metadata 负载均衡调度器。
//
// 为什么能在 H100(sm90) 上跑：`deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh`
// 里的 `sm100_paged_mqa_logits_metadata` kernel 与 `SM100PagedMQALogitsScheduler`
// 只依赖 `scheduler/sm100_mqa_logits.cuh`（前缀和 / 二分 / `MQALogitsTask`）和 PDL，
// 与 UMMA/tcgen05 无关，因此可以直接在 sm90 上编译运行真实设备代码。
// 主 kernel（`impls/sm100_mqa_logits.cuh` 的 block-scaled UMMA 部分）是 SM100 专属，本实验不涉及。
//
// 与 P13-1（连续版）的区别：
//   1. varlen 请求：`indices` 把 token 划分成「请求」，同一请求共享一段 KV（block_table 同一行）；
//      请求的 context_len 取【该请求最后一个 token】的（`context_lens[end-1]`）。
//   2. 请求内 token 按 Q block **均分**（`base + remainder`），不是固定 BLOCK_Q。
//   3. 代价模型没有「段启动代价」，每请求代价 = num_kv_splits * split_cost；
//      split_cost = 2*BLOCK_Q*ceil(nq/BLOCK_Q) + nq（KV/MMA 随 Q block、epilogue 随 token）。
//   4. 代价可能溢出 32 位：高位标志位（借 num_requests 字最高位）触发回退到 token 代价。
//   5. 同一请求内多个 Q block **复用同一段 KV**（`kv_shared_with_prev`），按 `kSplitsPerChunk` 分块。
//   6. 分页坐标用「每 lane 缓存一页」的滑动窗口（`get_kv_page_coords`，kNumCachedPages=32）。
//
// 本程序做的事：
//   (A) 用真实 `sm100_paged_mqa_logits_metadata<...>` 在 device 上生成 schedule_meta；
//   (B) host 端独立复刻（请求分组 / 溢出回退 / 前缀和 / split 粒度边界）逐字段对拍；
//   (C) 用真实 `SM100PagedMQALogitsScheduler` 发射任务，验证「所有 (请求,Q-block,split) 恰好覆盖一次」、
//       块大小 / KV 复用标志 / 每 token KV 上界；
//   (D) 用真实 `get_kv_page_coords` 验证分页查表（含滑动缓存中途 refill）与 host 逐字段一致。
//
// 编译：见 harness/exp_mqa_logits_paged_schedule.sh

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>

#include <cuda_runtime.h>

#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

using namespace deep_gemm;

// ---------------------------------------------------------------------------
// device：发射分页任务（真实 SM100PagedMQALogitsScheduler）
// ---------------------------------------------------------------------------
template <uint32_t kNumHeads, uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV,
          uint32_t kSplitsPerChunk, uint32_t kNumSMs, uint32_t kPageTasks>
__global__ void emit_paged_kernel(
        const uint32_t num_q_tokens_total,
        const uint32_t* __restrict__ context_lens,
        const uint32_t* __restrict__ indices,
        const uint32_t* __restrict__ block_table,
        const uint32_t block_table_stride,
        const uint32_t* __restrict__ schedule_meta,
        uint32_t* __restrict__ out_q_base,
        uint32_t* __restrict__ out_num_q,
        uint32_t* __restrict__ out_kv_base,
        uint32_t* __restrict__ out_num_splits,
        uint32_t* __restrict__ out_shared,
        uint32_t* __restrict__ out_seq_end,
        uint32_t* __restrict__ out_page_coords,
        uint32_t* __restrict__ out_count,
        const uint32_t max_tasks_per_sm) {
    constexpr uint32_t kPagesPerSplit = SPLIT_KV / PAGE_KV;
    constexpr uint32_t kDumpSplits = kSplitsPerChunk;   // 每个抽样任务最多 dump 的 split 数

    const uint32_t sm = blockIdx.x;
    sched::SM100PagedMQALogitsScheduler<kNumHeads, BLOCK_Q, SPLIT_KV, PAGE_KV, kSplitsPerChunk>
        scheduler(sm, num_q_tokens_total, context_lens, indices, block_table,
                  block_table_stride, schedule_meta);
    sched::MQALogitsTask task;
    uint32_t seq_start[BLOCK_Q], seq_end[BLOCK_Q];
    int page_coords[kPagesPerSplit];
    uint32_t cnt = 0;
    while (scheduler.next_q_block(task, seq_start, seq_end)) {
        const uint32_t base = sm * max_tasks_per_sm + cnt;
        out_q_base[base] = task.q_token_base;
        out_num_q[base] = task.num_q_tokens;
        out_kv_base[base] = task.kv_token_base;
        out_num_splits[base] = task.num_kv_splits;
        out_shared[base] = task.kv_shared_with_prev ? 1u : 0u;
        #pragma unroll
        for (uint32_t t = 0; t < BLOCK_Q; ++t)
            out_seq_end[base * BLOCK_Q + t] = seq_end[t];
        if (cnt < kPageTasks) {
            const uint32_t dump_base = (sm * kPageTasks + cnt) * (kDumpSplits * kPagesPerSplit);
            #pragma unroll 1
            for (uint32_t s = 0; s < kDumpSplits and s < task.num_kv_splits; ++s) {
                scheduler.get_kv_page_coords(task, s, page_coords);
                #pragma unroll
                for (uint32_t p = 0; p < kPagesPerSplit; ++p)
                    out_page_coords[dump_base + s * kPagesPerSplit + p] = (uint32_t)page_coords[p];
            }
        }
        ++cnt;
    }
    out_count[sm] = cnt;
}

// ---------------------------------------------------------------------------
// host 端独立参考实现（纯 C++）
// ---------------------------------------------------------------------------
struct Req {
    uint32_t start, nq, ctx, nqb, nkv, npage;
};

static uint32_t host_ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

static std::vector<Req> build_requests(const std::vector<int>& indices,
                                       const std::vector<int>& context_lens,
                                       uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV) {
    std::vector<Req> reqs;
    const uint32_t T = (uint32_t)indices.size();
    uint32_t r_start = 0;
    while (r_start < T) {
        uint32_t r_end = r_start + 1;
        while (r_end < T and indices[r_end] == indices[r_start])
            ++r_end;
        Req r;
        r.start = r_start;
        r.nq = r_end - r_start;
        r.ctx = (uint32_t)context_lens[r_end - 1];    // 最后一个 token 的 context_len
        r.nqb = host_ceil_div(r.nq, BLOCK_Q);
        r.nkv = host_ceil_div(r.ctx, SPLIT_KV);
        r.npage = host_ceil_div(r.ctx, PAGE_KV);
        reqs.push_back(r);
        r_start = r_end;
    }
    return reqs;
}

// metadata 的 block_q 固定为 4（H32 调优），与主 kernel 的 BLOCK_Q 可能不同
static uint32_t split_cost(const Req& r, uint32_t BLOCK_Q) {
    return 2 * BLOCK_Q * host_ceil_div(r.nq, BLOCK_Q) + r.nq;
}

struct MetaRef {
    std::vector<uint32_t> prefix;   // 每请求 work 前缀和
    uint32_t total_cost;
    bool use_token_cost;
};

static MetaRef meta_reference(const std::vector<Req>& reqs, uint32_t BLOCK_Q) {
    MetaRef out;
    const uint32_t R = (uint32_t)reqs.size();
    out.prefix.assign(R, 0u);
    if (R == 0) { out.total_cost = 0; out.use_token_cost = false; return out; }
    bool overflow = false;
    std::vector<uint64_t> cost64(R);
    for (uint32_t r = 0; r < R; ++r) {
        const uint64_t c = (uint64_t)reqs[r].nkv * split_cost(reqs[r], BLOCK_Q);
        cost64[r] = c;
        overflow |= c > 0xffffffffull;
    }
    uint32_t acc = 0;
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t prev = acc;
        acc += (uint32_t)cost64[r];
        if (acc < prev) overflow = true;
        out.prefix[r] = acc;
    }
    out.use_token_cost = overflow;
    if (overflow) {
        acc = 0;
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t prev = acc;
            acc += (uint32_t)((uint64_t)reqs[r].nkv * reqs[r].nq);
            out.prefix[r] = acc;
            (void)prev;
        }
    }
    out.total_cost = out.prefix[R - 1];
    return out;
}

static uint32_t upper_bound_u32(const std::vector<uint32_t>& prefix, uint32_t target) {
    uint32_t lo = 0, hi = (uint32_t)prefix.size();
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (prefix[mid] <= target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

struct SmBoundary { uint32_t q_token; uint32_t kv_split; };

static SmBoundary locate(const std::vector<Req>& reqs, const MetaRef& meta,
                         uint32_t BLOCK_Q, uint32_t sm, uint32_t kNumSMs) {
    const uint32_t R = (uint32_t)reqs.size();
    const uint32_t T = R == 0 ? 0 : reqs.back().start + reqs.back().nq;
    const uint32_t cost_per_sm = meta.total_cost / kNumSMs;
    const uint32_t cost_rem = meta.total_cost % kNumSMs;
    const uint32_t target = sm * cost_per_sm + std::min(sm, cost_rem);
    const uint32_t ridx = upper_bound_u32(meta.prefix, target);
    if (ridx >= R) return SmBoundary{T, 0};
    const uint32_t cost_before = ridx == 0 ? 0 : meta.prefix[ridx - 1];
    const uint32_t per_split = meta.use_token_cost ? reqs[ridx].nq : split_cost(reqs[ridx], BLOCK_Q);
    return SmBoundary{reqs[ridx].start, (target - cost_before) / per_split};
}

// ---------------------------------------------------------------------------
// 单个用例
// ---------------------------------------------------------------------------
template <uint32_t kNumHeads, uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV,
          uint32_t kSplitsPerChunk, uint32_t kNumSMs, uint32_t META_BLOCK_Q>
static bool run_case(const std::string& name,
                     const std::vector<int>& indices, const std::vector<int>& context_lens,
                     const std::vector<int>& block_table_host, uint32_t max_pages,
                     bool run_scheduler) {
    const uint32_t T = (uint32_t)indices.size();
    constexpr uint32_t kPagesPerSplit = SPLIT_KV / PAGE_KV;
    constexpr uint32_t kDumpSplits = kSplitsPerChunk;
    constexpr uint32_t kPageTasks = 4;

    const std::vector<Req> reqs = build_requests(indices, context_lens, META_BLOCK_Q, SPLIT_KV, PAGE_KV);
    const MetaRef meta_ref = meta_reference(reqs, META_BLOCK_Q);
    const uint32_t R = (uint32_t)reqs.size();

    // metadata 的总 work（仅用于打印）与每 SM 任务数上界
    uint64_t total_slots_meta = 0;
    // 覆盖统计按【主 kernel 的 BLOCK_Q】分 Q block（metadata 的 block_q 仅用于代价模型）
    std::vector<uint32_t> sched_nqb(R), work_prefix(R, 0);
    uint64_t total_slots = 0, total_tasks_bound = 0;
    for (uint32_t r = 0; r < R; ++r) {
        total_slots_meta += (uint64_t)reqs[r].nqb * reqs[r].nkv;
        sched_nqb[r] = host_ceil_div(reqs[r].nq, BLOCK_Q);
        work_prefix[r] = (uint32_t)total_slots;
        total_slots += (uint64_t)sched_nqb[r] * reqs[r].nkv;
        // nqb>1 时按 kSplitsPerChunk 分块，否则单任务覆盖整段
        total_tasks_bound += sched_nqb[r] > 1
            ? (uint64_t)sched_nqb[r] * host_ceil_div(reqs[r].nkv, kSplitsPerChunk)
            : 1;
    }

    // ---- device 分配 ----
    const uint32_t meta_words = (kNumSMs + 1) * 2;
    const uint32_t max_tasks = std::max<uint32_t>(1u, (uint32_t)total_tasks_bound);

    int *d_ctx=nullptr, *d_idx=nullptr, *d_bt=nullptr;
    uint32_t* d_meta=nullptr;
    cudaMalloc(&d_ctx, sizeof(int) * std::max<size_t>(1, context_lens.size()));
    cudaMalloc(&d_idx, sizeof(int) * std::max<size_t>(1, indices.size()));
    cudaMalloc(&d_bt, sizeof(int) * std::max<size_t>(1, block_table_host.size()));
    cudaMalloc(&d_meta, sizeof(uint32_t) * meta_words);
    cudaMemcpy(d_ctx, context_lens.data(), sizeof(int) * context_lens.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, block_table_host.data(), sizeof(int) * block_table_host.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_meta, 0, sizeof(uint32_t) * meta_words);

    // (A) 真实 metadata kernel
    constexpr uint32_t num_threads = 256;   // 线程数只影响扫描划分，不影响语义
    const int smem_size = (int)((2 * T + num_threads / 32 + 1) * sizeof(uint32_t));
    sched::sm100_paged_mqa_logits_metadata<SPLIT_KV, kNumSMs, META_BLOCK_Q, num_threads>
        <<<1, num_threads, smem_size>>>(
            T, (const uint32_t*)d_ctx, (const uint32_t*)d_idx, d_meta);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [%s] metadata kernel FAILED: %s\n", name.c_str(), cudaGetErrorString(err));
        cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_bt); cudaFree(d_meta);
        return false;
    }
    std::vector<uint32_t> meta(meta_words);
    cudaMemcpy(meta.data(), d_meta, sizeof(uint32_t) * meta_words, cudaMemcpyDeviceToHost);

    bool ok = true;

    // (B) 每 SM 边界对拍
    int bnd_mismatch = 0;
    for (uint32_t sm = 0; sm <= kNumSMs; ++sm) {
        const uint32_t dq = meta[2 * sm], dkv = meta[2 * sm + 1];
        const SmBoundary ref = locate(reqs, meta_ref, META_BLOCK_Q, sm, kNumSMs);
        if (dq != ref.q_token or dkv != ref.kv_split) ++bnd_mismatch;
    }
    ok &= (bnd_mismatch == 0);

    // (C) 真实 scheduler 发射任务（代价溢出用例只测 metadata，跳过）
    uint32_t covered = 0, tasks_total = 0, bad_task = 0, seq_mismatch = 0,
             shared_mismatch = 0, chunk_mismatch = 0, page_mismatch = 0, page_checked = 0;
    uint32_t dup = 0, miss = 0;
    std::vector<uint32_t> per_sm(kNumSMs, 0u);

    if (run_scheduler) {
    const size_t out_elems = (size_t)kNumSMs * max_tasks;
    uint32_t *d_oq=nullptr,*d_on=nullptr,*d_ok=nullptr,*d_onk=nullptr,*d_os=nullptr,*d_oe=nullptr;
    uint32_t *d_oc=nullptr,*d_op=nullptr;
    cudaMalloc(&d_oq, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_on, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_ok, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_onk, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_os, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_oe, sizeof(uint32_t) * out_elems * BLOCK_Q);
    cudaMalloc(&d_oc, sizeof(uint32_t) * kNumSMs);
    const size_t page_dump_elems = (size_t)kNumSMs * kPageTasks * kDumpSplits * kPagesPerSplit;
    cudaMalloc(&d_op, sizeof(uint32_t) * page_dump_elems);

    emit_paged_kernel<kNumHeads, BLOCK_Q, SPLIT_KV, PAGE_KV, kSplitsPerChunk, kNumSMs, kPageTasks>
        <<<kNumSMs, 32>>>(
            T, (const uint32_t*)d_ctx, (const uint32_t*)d_idx, (const uint32_t*)d_bt,
            max_pages, d_meta, d_oq, d_on, d_ok, d_onk, d_os, d_oe, d_op, d_oc, max_tasks);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [%s] emit kernel FAILED: %s\n", name.c_str(), cudaGetErrorString(err));
        cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_bt); cudaFree(d_meta);
        return false;
    }

    std::vector<uint32_t> h_oq(out_elems), h_on(out_elems), h_ok(out_elems),
                          h_onk(out_elems), h_os(out_elems), h_oc(kNumSMs),
                          h_oe(out_elems * BLOCK_Q), h_op(page_dump_elems);
    cudaMemcpy(h_oq.data(), d_oq, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_on.data(), d_on, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ok.data(), d_ok, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_onk.data(), d_onk, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_os.data(), d_os, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oe.data(), d_oe, sizeof(uint32_t) * out_elems * BLOCK_Q, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oc.data(), d_oc, sizeof(uint32_t) * kNumSMs, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_op.data(), d_op, sizeof(uint32_t) * page_dump_elems, cudaMemcpyDeviceToHost);

    std::vector<uint32_t> hits(std::max<uint64_t>(total_slots, 1), 0u);

    std::vector<uint32_t> req_starts(R);
    for (uint32_t r = 0; r < R; ++r) req_starts[r] = reqs[r].start;

    for (uint32_t sm = 0; sm < kNumSMs; ++sm) {
        for (uint32_t i = 0; i < h_oc[sm]; ++i) {
            const uint32_t base = sm * max_tasks + i;
            const uint32_t qb_tok = h_oq[base];
            const uint32_t nq = h_on[base];
            const uint32_t kvb = h_ok[base];
            const uint32_t n = h_onk[base];
            const uint32_t shared = h_os[base];

            // 定位请求
            const uint32_t ridx = (uint32_t)(std::upper_bound(req_starts.begin(), req_starts.end(), qb_tok)
                                             - req_starts.begin()) - 1;
            const Req& r = reqs[ridx];

            // 定位请求内 Q block（按主 kernel 的 BLOCK_Q 均分）
            const uint32_t nqb_s = sched_nqb[ridx];
            uint32_t qb = 0; bool found = false;
            {
                const uint32_t qb_base = r.nq / nqb_s, qb_rem = r.nq % nqb_s;
                for (uint32_t b = 0; b < nqb_s; ++b) {
                    const uint32_t off = b * qb_base + std::min(b, qb_rem);
                    if (r.start + off == qb_tok) { qb = b; found = true; break; }
                }
            }
            if (not found or n == 0 or kvb % SPLIT_KV != 0) { ++bad_task; continue; }

            const uint32_t s0 = kvb / SPLIT_KV;
            if (s0 + n > r.nkv) { ++bad_task; continue; }

            // KV 复用标志：请求内非首个 Q block
            if (shared != (qb > 0 ? 1u : 0u)) ++shared_mismatch;

            // 分块大小：nqb>1 时不超过 kSplitsPerChunk
            if (nqb_s > 1 and n > kSplitsPerChunk) ++chunk_mismatch;

            // 每 token KV 上界
            const uint32_t dump_idx_in_sm = i < kPageTasks ? i : kPageTasks;
            for (uint32_t t = 0; t < BLOCK_Q; ++t) {
                const uint32_t tok = qb_tok + t;
                if (tok < T and h_oe[base * BLOCK_Q + t] != (uint32_t)context_lens[tok])
                    ++seq_mismatch;
            }

            // 覆盖统计
            const uint32_t gbase = work_prefix[ridx];
            for (uint32_t j = 0; j < n; ++j) hits[gbase + qb * r.nkv + s0 + j] += 1;
            covered += n; per_sm[sm] += n; ++tasks_total;

            // (D) 分页坐标对拍
            if (i < kPageTasks) {
                const uint32_t dump_base = (sm * kPageTasks + i) * (kDumpSplits * kPagesPerSplit);
                const uint32_t ndump = std::min<uint32_t>(n, kDumpSplits);
                for (uint32_t s = 0; s < ndump; ++s) {
                    const uint32_t page_base = (s0 + s) * kPagesPerSplit;
                    for (uint32_t p = 0; p < kPagesPerSplit; ++p) {
                        const uint32_t logical = page_base + p;
                        const uint32_t expect = logical < r.npage
                            ? (uint32_t)block_table_host[r.start * max_pages + logical] : 0u;
                        if (h_op[dump_base + s * kPagesPerSplit + p] != expect) ++page_mismatch;
                        ++page_checked;
                    }
                }
                (void)dump_idx_in_sm;
            }
        }
    }

    for (uint64_t g = 0; g < total_slots; ++g) {
        if (hits[g] == 0) ++miss; else if (hits[g] > 1) ++dup;
    }
    ok &= (bad_task == 0 and dup == 0 and miss == 0);
    ok &= (seq_mismatch == 0 and shared_mismatch == 0 and chunk_mismatch == 0 and page_mismatch == 0);
    ok &= (covered == total_slots);

    cudaFree(d_oq); cudaFree(d_on); cudaFree(d_ok); cudaFree(d_onk); cudaFree(d_os);
    cudaFree(d_oe); cudaFree(d_oc); cudaFree(d_op);
    }   // if (run_scheduler)

    uint32_t mn = 0xffffffffu, mx = 0;
    for (auto x : per_sm) { mn = std::min(mn, x); mx = std::max(mx, x); }

    printf("  [%-24s] BQ=%u H=%u nq=%u reqs=%u pkv=%u  slots(sched/meta)=%llu/%llu  use_token_cost=%s\n",
           name.c_str(), BLOCK_Q, kNumHeads, T, R, PAGE_KV,
           (unsigned long long)total_slots, (unsigned long long)total_slots_meta,
           meta_ref.use_token_cost ? "Y" : "n");
    printf("      meta boundary match=%s (%d mismatch)  total_cost=%u\n",
           bnd_mismatch == 0 ? "YES" : "NO", bnd_mismatch, meta_ref.total_cost);
    printf("      scheduler: tasks=%u slots=%u dup=%u miss=%u bad=%u  seq/shared/chunk bad=%u/%u/%u\n",
           tasks_total, covered, dup, miss, bad_task, seq_mismatch, shared_mismatch, chunk_mismatch);
    printf("      pages : checked=%u mismatch=%u   per-SM work [%u..%u]\n",
           page_checked, page_mismatch, mn, mx);
    printf("      => result: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_bt); cudaFree(d_meta);
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("P14-1 SM100 paged MQA logits 调度器：H100 真实设备代码验证\n");
    printf("GPU: %s  sm_%d%d  SMs=%d\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    constexpr uint32_t SPLIT_KV = 384;          // kMQALogitsSplitKV
    constexpr uint32_t CHUNK = 8;               // kMQALogitsSplitsPerChunk
    constexpr uint32_t SMS = 132;               // H100

    bool all = true;

    // 构造 varlen 请求：indices 给请求号，context_lens 每个 token 一个
    // req_lens: 各请求 token 数；ctx_per_req: 各请求最后 token 的 context_len
    auto build = [](const std::vector<int>& req_lens, const std::vector<int>& ctx_per_req,
                    std::vector<int>& indices, std::vector<int>& context_lens) {
        indices.clear(); context_lens.clear();
        for (size_t r = 0; r < req_lens.size(); ++r) {
            for (int t = 0; t < req_lens[r]; ++t) {
                indices.push_back((int)r);
                // 请求内前面的 token 给一个更小的 context，最后 token 给完整值，
                // 以验证 metadata 取的是【最后一个 token】的 context_len
                context_lens.push_back(t + 1 == req_lens[r] ? ctx_per_req[r] : (ctx_per_req[r] / 2));
            }
        }
    };
    auto make_bt = [](uint32_t T, uint32_t max_pages) {
        std::vector<int> bt((size_t)T * max_pages);
        for (uint32_t q = 0; q < T; ++q)
            for (uint32_t p = 0; p < max_pages; ++p)
                bt[(size_t)q * max_pages + p] = (int)((q * 1000003u + p * 7919u) % 999983u + 1u);
        return bt;
    };

    printf("== 用例组 A：BLOCK_Q=4（H=32），PAGE_KV=64，132 SM ==\n");
    {
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 64; ++r) { rl.push_back(8); cr.push_back(2048); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(2048, 64);
        all &= run_case<32, 4, SPLIT_KV, 64, CHUNK, SMS, 4>(
            "uniform req=8 ctx=2048", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }
    {
        std::vector<int> idx, cl, rl, cr;
        const int lens[] = {1,2,3,4,6,8,12,16,32};
        const int ctxs[] = {128,257,384,700,1025,2048,4096,8192,300};
        for (int r = 0; r < 9; ++r) { rl.push_back(lens[r]); cr.push_back(ctxs[r]); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(8192, 64);
        all &= run_case<32, 4, SPLIT_KV, 64, CHUNK, SMS, 4>(
            "skewed req/ctx", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }
    {
        // 大量小请求 + 少量长请求，制造明显不均衡
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 200; ++r) {
            if (r % 37 == 0) { rl.push_back(32); cr.push_back(8192); }
            else { rl.push_back(2); cr.push_back(196); }
        }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(8192, 64);
        all &= run_case<32, 4, SPLIT_KV, 64, CHUNK, SMS, 4>(
            "mixed heavy/light", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }

    printf("\n== 用例组 B：PAGE_KV=32 / 128（同一 metadata，不同页大小）==\n");
    {
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 40; ++r) { rl.push_back(8); cr.push_back(1500); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(1500, 32);
        all &= run_case<32, 4, SPLIT_KV, 32, CHUNK, SMS, 4>(
            "PAGE_KV=32", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }
    {
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 40; ++r) { rl.push_back(8); cr.push_back(3000); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(3000, 128);
        all &= run_case<32, 4, SPLIT_KV, 128, CHUNK, SMS, 4>(
            "PAGE_KV=128", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }

    printf("\n== 用例组 C：头数改变 BLOCK_Q（metadata 仍固定 block_q=4）==\n");
    {
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 48; ++r) { rl.push_back(6); cr.push_back(1024); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(1024, 64);
        all &= run_case<64, 2, SPLIT_KV, 64, CHUNK, SMS, 4>(
            "H=64 -> BLOCK_Q=2", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }
    {
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 48; ++r) { rl.push_back(6); cr.push_back(1024); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(1024, 64);
        all &= run_case<128, 1, SPLIT_KV, 64, CHUNK, SMS, 4>(
            "H=128 -> BLOCK_Q=1", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }

    printf("\n== 用例组 D：少 SM（8），便于观察边界与空 SM ==\n");
    {
        std::vector<int> idx, cl, rl, cr;
        const int lens[] = {8, 8, 16, 4, 32, 2};
        const int ctxs[] = {2048, 512, 4096, 128, 8192, 64};
        for (int r = 0; r < 6; ++r) { rl.push_back(lens[r]); cr.push_back(ctxs[r]); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(8192, 64);
        all &= run_case<32, 4, SPLIT_KV, 64, CHUNK, 8, 4>(
            "skewed 8 SM", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }
    {
        // 请求数少于 SM 数：很多 SM 分到空区间（q_token==T）
        std::vector<int> idx, cl, rl, cr;
        for (int r = 0; r < 3; ++r) { rl.push_back(8); cr.push_back(4096); }
        build(rl, cr, idx, cl);
        uint32_t max_pages = host_ceil_div(4096, 64);
        all &= run_case<32, 4, SPLIT_KV, 64, CHUNK, 8, 4>(
            "few reqs -> empty SM", idx, cl, make_bt((uint32_t)idx.size(), max_pages), max_pages, true);
    }

    printf("\n== 用例组 E：代价溢出 -> 回退到 token 代价（只验证 metadata 边界）==\n");
    {
        std::vector<int> idx, cl, rl, cr;
        rl.push_back(4096); cr.push_back(156000000);   // 单请求，超大 context
        build(rl, cr, idx, cl);
        uint32_t max_pages = 1;                        // 不测分页
        std::vector<int> bt((size_t)idx.size());
        all &= run_case<32, 4, SPLIT_KV, 64, CHUNK, SMS, 4>(
            "cost overflow -> token", idx, cl, bt, max_pages, false);
    }

    printf("\n%s\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
