// P20-1 实验：SM90 paged MQA logits 的 metadata 负载均衡调度器。
//
// 为什么能在 H100(sm90) 上跑：`deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh`
// 里的 `sm90_paged_mqa_logits_metadata` kernel 与 `SM90PagedMQALogitsScheduler`
// 只依赖 `<deep_gemm/common/math.cuh>` / `<deep_gemm/ptx/utils.cuh>` 与 PDL，
// 与 WGMMA/TMA 无关，因此可以直接在 sm90 上编译运行【真实设备代码】。
// 主 kernel（`impls/sm90_fp8_paged_mqa_logits.cuh`）是 SM90 WGMMA，本实验不涉及。
//
// 与 P14-1（SM100 分页版）的对照点：
//   1. SM90 metadata 的 work 单位是「split」= SPLIT_KV/KV 块（split_kv=256、block_kv=64 → 4 块），
//      SM100 的 work 单位是「Q block × KV split」，代价模型完全不同（SM90 纯按 ctx 段数均分）。
//   2. SM90 支持 **非 varlen**（batch 维度 + next_n∈{1,2} + 2D context lens），
//      SM100 分页只支持 varlen（indices 划分请求、next_n==1）。
//   3. SM90 用 `q_atom_idx` 直接当 token/atom 起点（varlen）或 q_idx（非 varlen，
//      `atom_to_token_idx` 再乘 next_n_atom），SM100 用 `(请求, 请求内 Q 均分)`。
//   4. SM90 无分页坐标滑动缓存（那是主 kernel 里逐 KV 块读 block_table），调度器只发 split。
//
// 本程序做的事：
//   (A) 用真实 `sm90_paged_mqa_logits_metadata<...>` 在 device 上生成 schedule_meta；
//   (B) host 端独立复刻（varlen atom 分组 / 前缀和 / 反序分配 / 二分）逐字段对拍；
//   (C) 用真实 `SM90PagedMQALogitsScheduler` 发射任务，验证「所有 (atom, split) 恰好覆盖一次」+
//       num_kv / advance / token 映射 / block_table 行映射。
//
// 编译：见 harness/exp_sm90_paged_mqa_logits_schedule.sh

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include <deep_gemm/scheduler/sm90_paged_mqa_logits.cuh>

using namespace deep_gemm;

static uint32_t host_ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// device：调用真实 scheduler 发射任务并 dump
// ---------------------------------------------------------------------------
template <bool kIsVarlen, uint32_t kNextN, bool kIsCtx2D,
          uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit, uint32_t kNumNextNAtoms>
__global__ void emit_kernel(
        const uint32_t batch_size,
        const uint32_t* __restrict__ context_lens,
        const uint32_t* __restrict__ indices,
        const uint32_t* __restrict__ schedule_meta,
        uint32_t* __restrict__ out_qraw,
        uint32_t* __restrict__ out_token,
        uint32_t* __restrict__ out_btrow,
        uint32_t* __restrict__ out_split,
        uint32_t* __restrict__ out_numkv,
        uint32_t* __restrict__ out_adv,
        uint32_t* __restrict__ out_count,
        const uint32_t max_tasks) {
    using Scheduler = sched::SM90PagedMQALogitsScheduler<
        kNextN, kIsCtx2D, kIsVarlen, BLOCK_KV, kNumBlocksPerSplit, kNumNextNAtoms>;
    const uint32_t sm = blockIdx.x;
    Scheduler scheduler(sm, batch_size, context_lens, schedule_meta, indices);

    uint32_t q, kv, num_kv;
    uint32_t cnt = 0;
    while (scheduler.fetch_next_task(q, kv, num_kv)) {
        const uint32_t base = sm * max_tasks + cnt;
        out_qraw[base]  = q;
        out_token[base] = Scheduler::atom_to_token_idx(q);
        out_btrow[base] = Scheduler::atom_to_block_table_row(q);
        out_split[base] = kv / kNumBlocksPerSplit;
        out_numkv[base] = num_kv;
        out_adv[base]   = scheduler.get_last_advance();
        ++cnt;
    }
    out_count[sm] = cnt;
}

// ---------------------------------------------------------------------------
// host 端独立参考实现
// ---------------------------------------------------------------------------
struct Atom {
    uint32_t token_start;   // varlen: token 起点；非 varlen: q_atom_idx（token 起点 = q_atom_idx * next_n_atom）
    uint32_t ctx;           // 该 atom 的 context_len
    uint32_t num_segs;      // ceil_div(ctx, SPLIT_KV)
};

struct MetaRef {
    std::vector<std::pair<uint32_t, uint32_t>> meta;  // size = num_sms + 1
};

// 构造 atom 列表。返回的 atom 的 token_start 对所有模式都代表「主 kernel 用的 q_atom 索引」
// （varlen = token 起点；非 varlen = q_idx → 主 kernel 的 atom_to_token_idx 再展开）。
static std::vector<Atom> build_atoms(bool is_varlen, uint32_t batch_size, uint32_t next_n,
                                     bool is_ctx_2d, const std::vector<int>& indices,
                                     const std::vector<int>& context_lens, uint32_t SPLIT_KV) {
    std::vector<Atom> atoms;
    if (is_varlen) {
        uint32_t t = 0;
        while (t < batch_size) {
            const bool paired = (t + 1 < batch_size and indices[t] == indices[t + 1]);
            const uint32_t ctx = (uint32_t)(paired ? context_lens[t + 1] : context_lens[t]);
            atoms.push_back({t, ctx, host_ceil_div(ctx, SPLIT_KV)});
            t += paired ? 2 : 1;
        }
    } else {
        // 每个 q_idx 一个 atom，context 取该行最后一个 token 的（2D context lens）
        for (uint32_t i = 0; i < batch_size; ++i) {
            const uint32_t lens_idx = is_ctx_2d ? i * next_n + next_n - 1 : i;
            const uint32_t ctx = (uint32_t)context_lens[lens_idx];
            atoms.push_back({i, ctx, host_ceil_div(ctx, SPLIT_KV)});
        }
    }
    return atoms;
}

static MetaRef metadata_reference(bool is_varlen, uint32_t batch_size, uint32_t next_n,
                                  bool is_ctx_2d, const std::vector<int>& indices,
                                  const std::vector<int>& context_lens,
                                  uint32_t SPLIT_KV, uint32_t num_sms) {
    const std::vector<Atom> atoms = build_atoms(is_varlen, batch_size, next_n, is_ctx_2d,
                                                indices, context_lens, SPLIT_KV);
    const uint32_t A = (uint32_t)atoms.size();

    // 前缀和（capped 到 batch_size 之外用总数，模拟 device padded 行为）
    std::vector<uint32_t> prefix(A);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < A; ++i) { sum += atoms[i].num_segs; prefix[i] = sum; }
    auto pref_at = [&](uint32_t i) -> uint32_t { return i < A ? prefix[i] : sum; };

    MetaRef out;
    out.meta.resize(num_sms + 1);

    if (is_varlen) {
        const uint32_t total = sum;
        const uint32_t q = total / num_sms, r = total % num_sms;
        const uint32_t pivot = num_sms - r;
        for (uint32_t sm = 0; sm <= num_sms; ++sm) {
            const uint32_t seg_starts = sm * q + (sm > pivot ? sm - pivot : 0);
            uint32_t lo = 0, hi = A;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (prefix[mid] <= seg_starts) lo = mid + 1; else hi = mid;
            }
            const uint32_t atom_idx = lo;
            const uint32_t kv_split = (atom_idx == 0 ? seg_starts : seg_starts - pref_at(atom_idx - 1));
            const uint32_t q_atom = (atom_idx < A ? atoms[atom_idx].token_start : batch_size);
            out.meta[sm] = {q_atom, kv_split};
        }
    } else {
        const uint32_t next_n_atom = (next_n >= 2) ? 2 : 1;
        const uint32_t nna = host_ceil_div(next_n, next_n_atom);
        const uint32_t total = sum * nna;
        const uint32_t q = total / num_sms, r = total % num_sms;
        const uint32_t pivot = num_sms - r;
        for (uint32_t sm = 0; sm < num_sms; ++sm) {
            const uint32_t seg_starts = sm * q + (sm > pivot ? sm - pivot : 0);
            uint32_t lo = 0, hi = A;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (prefix[mid] * nna <= seg_starts) lo = mid + 1; else hi = mid;
            }
            const uint32_t qi = lo;   // 可能与 A 相等（空区间）
            const uint32_t offset = (qi == 0 ? seg_starts : seg_starts - pref_at(qi - 1) * nna);
            const uint32_t num_segs = (qi == 0 ? pref_at(0) : pref_at(qi) - pref_at(qi - 1));
            const uint32_t atom_idx = num_segs > 0 ? offset / num_segs : 0;
            const uint32_t kv_split = num_segs > 0 ? offset % num_segs : 0;
            out.meta[sm] = {qi * nna + atom_idx, kv_split};
        }
        out.meta[num_sms] = {batch_size * nna, 0};
    }
    return out;
}

// 独立复刻 SM90PagedMQALogitsScheduler 的任务流（不含设备）
struct TaskHost { uint32_t qraw, token, btrow, split, numkv, adv; };

static std::vector<TaskHost> scheduler_reference(
        bool is_varlen, uint32_t batch_size, uint32_t next_n, bool is_ctx_2d,
        const std::vector<int>& indices, const std::vector<int>& context_lens,
        const MetaRef& meta, uint32_t sm,
        uint32_t BLOCK_KV, uint32_t KBPS, uint32_t kNumNextNAtoms) {
    std::vector<TaskHost> tasks;
    const bool kNextNPadOdd = (not is_varlen) and (next_n % 2 == 1) and (next_n >= 3);
    const uint32_t kNextNAtom = (is_varlen or next_n >= 2) ? 2 : 1;

    auto atom_to_token_idx = [&](uint32_t a) -> uint32_t {
        if (is_varlen) return a;
        if (kNextNPadOdd) return a / kNumNextNAtoms * next_n + a % kNumNextNAtoms * kNextNAtom;
        return a * kNextNAtom;
    };
    auto atom_to_bt_row = [&](uint32_t a) -> uint32_t {
        return is_varlen ? a : a / kNumNextNAtoms;
    };

    auto refresh = [&](uint32_t a, uint32_t& adv, uint32_t& num_kv) {
        if (is_varlen) {
            const bool paired = (a + 1 < batch_size and indices[a] == indices[a + 1]);
            adv = paired ? 2 : 1;
            const uint32_t ctx = (uint32_t)(paired ? context_lens[a + 1] : context_lens[a]);
            num_kv = host_ceil_div(ctx, BLOCK_KV);
        } else {
            adv = 1;
            const uint32_t q_idx = a / kNumNextNAtoms;
            const uint32_t lens_idx = is_ctx_2d ? q_idx * next_n + next_n - 1 : q_idx;
            num_kv = host_ceil_div((uint32_t)context_lens[lens_idx], BLOCK_KV);
        }
    };
    auto should_refresh = [&](uint32_t a) -> bool {
        return is_varlen ? true : (a % kNumNextNAtoms == 0);
    };

    uint32_t cur_q = meta.meta[sm].first;
    uint32_t cur_kv = meta.meta[sm].second * KBPS;
    const uint32_t end_q = meta.meta[sm + 1].first;
    const uint32_t end_kv = meta.meta[sm + 1].second * KBPS;
    auto exist = [&](uint32_t a) -> bool { return a < end_q or (a == end_q and 0 < end_kv); };

    uint32_t adv, num_kv;
    refresh(cur_q, adv, num_kv);

    while (true) {
        if (cur_q == end_q and cur_kv == end_kv) break;
        tasks.push_back({cur_q, atom_to_token_idx(cur_q), atom_to_bt_row(cur_q),
                         cur_kv / KBPS, num_kv, adv});
        cur_kv += KBPS;
        if (cur_kv >= num_kv) {
            cur_kv = 0;
            cur_q += adv;
            if (should_refresh(cur_q) and exist(cur_q))
                refresh(cur_q, adv, num_kv);
        }
    }
    return tasks;
}

// ---------------------------------------------------------------------------
// 单个用例
// ---------------------------------------------------------------------------
template <bool kIsVarlen, uint32_t kNextN, bool kIsCtx2D, uint32_t kAlignedBatchSize, uint32_t kNumSMs>
static bool run_case(const std::string& name, uint32_t batch_size,
                     const std::vector<int>& indices,
                     const std::vector<int>& context_lens) {
    constexpr uint32_t SPLIT_KV = 256;         // sm90_paged_mqa_logits_metadata split_kv
    constexpr uint32_t BLOCK_KV = 64;          // sm90 paged 只允许 64
    constexpr uint32_t KBPS = SPLIT_KV / BLOCK_KV;   // kNumBlocksPerSplit = 4
    constexpr uint32_t kNumNextNAtoms = 1;     // sm90 kernel 实例化时固定传 1

    const uint32_t aligned_bs = kAlignedBatchSize;

    // host 参考
    const MetaRef meta_ref = metadata_reference(kIsVarlen, batch_size, kNextN, kIsCtx2D,
                                                indices, context_lens, SPLIT_KV, kNumSMs);
    const std::vector<Atom> atoms = build_atoms(kIsVarlen, batch_size, kNextN, kIsCtx2D,
                                                indices, context_lens, SPLIT_KV);

    // ---- device ----
    const uint32_t meta_words = (kNumSMs + 1) * 2;
    int *d_ctx = nullptr, *d_idx = nullptr;
    uint32_t* d_meta = nullptr;
    cudaMalloc(&d_ctx, sizeof(int) * std::max<size_t>(1, context_lens.size()));
    cudaMalloc(&d_idx, sizeof(int) * std::max<size_t>(1, indices.size()));
    cudaMalloc(&d_meta, sizeof(uint32_t) * meta_words);
    cudaMemcpy(d_ctx, context_lens.data(), sizeof(int) * context_lens.size(), cudaMemcpyHostToDevice);
    if (not indices.empty())
        cudaMemcpy(d_idx, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_meta, 0, sizeof(uint32_t) * meta_words);

    // (A) 真实 metadata kernel
    const int smem_ints = kIsVarlen ? (int)(3 * aligned_bs + 1) : (int)aligned_bs;
    const int smem_size = smem_ints * (int)sizeof(int);
    sched::sm90_paged_mqa_logits_metadata<kAlignedBatchSize, SPLIT_KV, kNumSMs, kIsVarlen>
        <<<1, 32, smem_size>>>(
            batch_size, kNextN, kIsCtx2D,
            (const uint32_t*)d_ctx, (const uint32_t*)d_idx, d_meta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  [%s] metadata kernel launch FAILED: %s\n", name.c_str(), cudaGetErrorString(err));
        cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_meta);
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [%s] metadata kernel FAILED: %s\n", name.c_str(), cudaGetErrorString(err));
        cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_meta);
        return false;
    }
    std::vector<uint32_t> meta(meta_words);
    cudaMemcpy(meta.data(), d_meta, sizeof(uint32_t) * meta_words, cudaMemcpyDeviceToHost);

    bool ok = true;

    // (B) 每 SM 边界对拍
    int bnd_mismatch = 0;
    for (uint32_t sm = 0; sm <= kNumSMs; ++sm) {
        if (meta[2 * sm] != meta_ref.meta[sm].first or meta[2 * sm + 1] != meta_ref.meta[sm].second)
            ++bnd_mismatch;
    }
    ok &= (bnd_mismatch == 0);

    // (C) 真实 scheduler 发射任务
    // 计算任务数上界与全局槽
    uint64_t total_segs = 0, total_tasks_bound = 0;
    std::vector<uint32_t> atom_seg_prefix(atoms.size(), 0);
    for (size_t a = 0; a < atoms.size(); ++a) {
        atom_seg_prefix[a] = (uint32_t)total_segs;
        total_segs += atoms[a].num_segs;
        total_tasks_bound += atoms[a].num_segs;
    }
    const uint32_t max_tasks = std::max<uint32_t>(1u, (uint32_t)total_tasks_bound);

    const size_t out_elems = (size_t)kNumSMs * max_tasks;
    uint32_t *d_oq=nullptr,*d_ot=nullptr,*d_ob=nullptr,*d_osp=nullptr,*d_onk=nullptr,*d_oa=nullptr,*d_oc=nullptr;
    cudaMalloc(&d_oq, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_ot, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_ob, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_osp, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_onk, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_oa, sizeof(uint32_t) * out_elems);
    cudaMalloc(&d_oc, sizeof(uint32_t) * kNumSMs);

    emit_kernel<kIsVarlen, kNextN, kIsCtx2D, BLOCK_KV, KBPS, kNumNextNAtoms>
        <<<kNumSMs, 32>>>(
            batch_size, (const uint32_t*)d_ctx, (const uint32_t*)d_idx, d_meta,
            d_oq, d_ot, d_ob, d_osp, d_onk, d_oa, d_oc, max_tasks);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [%s] emit kernel FAILED: %s\n", name.c_str(), cudaGetErrorString(err));
        cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_meta);
        return false;
    }

    std::vector<uint32_t> h_oq(out_elems), h_ot(out_elems), h_ob(out_elems), h_osp(out_elems),
                          h_onk(out_elems), h_oa(out_elems), h_oc(kNumSMs);
    cudaMemcpy(h_oq.data(), d_oq, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ot.data(), d_ot, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ob.data(), d_ob, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_osp.data(), d_osp, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_onk.data(), d_onk, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oa.data(), d_oa, sizeof(uint32_t) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oc.data(), d_oc, sizeof(uint32_t) * kNumSMs, cudaMemcpyDeviceToHost);

    std::vector<uint32_t> hits(std::max<uint64_t>(total_segs, 1), 0u);
    uint32_t tasks_total = 0, dup = 0, miss = 0, bad = 0, ref_mismatch = 0;
    std::vector<uint32_t> per_sm(kNumSMs, 0u);

    // 用 host 参考逐 SM 复刻，并与 device 逐任务对拍
    for (uint32_t sm = 0; sm < kNumSMs; ++sm) {
        const std::vector<TaskHost> ref = scheduler_reference(
            kIsVarlen, batch_size, kNextN, kIsCtx2D, indices, context_lens, meta_ref, sm,
            BLOCK_KV, KBPS, kNumNextNAtoms);
        if (h_oc[sm] != ref.size()) { ++ref_mismatch; }
        for (uint32_t i = 0; i < h_oc[sm]; ++i) {
            const uint32_t base = sm * max_tasks + i;
            // device vs host 参考逐字段
            if (i < ref.size()) {
                if (h_oq[base] != ref[i].qraw or h_ot[base] != ref[i].token or
                    h_ob[base] != ref[i].btrow or h_osp[base] != ref[i].split or
                    h_onk[base] != ref[i].numkv or h_oa[base] != ref[i].adv)
                    ++ref_mismatch;
            }
            // 覆盖：定位 atom
            const std::vector<Atom> as = atoms;
            uint32_t a_idx = 0; bool found = false;
            for (uint32_t a = 0; a < as.size(); ++a) {
                if (as[a].token_start == h_oq[base]) { a_idx = a; found = true; break; }
            }
            if (not found) { ++bad; continue; }
            const uint32_t split = h_osp[base];
            if (split >= as[a_idx].num_segs) { ++bad; continue; }
            hits[atom_seg_prefix[a_idx] + split] += 1;
            per_sm[sm] += 1;
            ++tasks_total;
        }
    }
    for (uint64_t g = 0; g < total_segs; ++g) {
        if (hits[g] == 0) ++miss; else if (hits[g] > 1) ++dup;
    }
    ok &= (bad == 0 and dup == 0 and miss == 0 and ref_mismatch == 0);
    ok &= (tasks_total == total_segs);

    uint32_t mn = 0xffffffffu, mx = 0;
    for (auto x : per_sm) { mn = std::min(mn, x); mx = std::max(mx, x); }

    printf("  [%-28s] varlen=%d nextN=%u batch=%u atoms=%zu segs=%llu  smem=%dB\n",
           name.c_str(), kIsVarlen ? 1 : 0, kNextN, batch_size, atoms.size(),
           (unsigned long long)total_segs, smem_size);
    printf("      meta boundary match=%s (%d mismatch)\n",
           bnd_mismatch == 0 ? "YES" : "NO", bnd_mismatch);
    printf("      scheduler: tasks=%u segs=%llu dup=%u miss=%u bad=%u  ref_mismatch=%u\n",
           tasks_total, (unsigned long long)total_segs, dup, miss, bad, ref_mismatch);
    printf("      per-SM splits [%u..%u]  => %s\n", mn, mx, ok ? "PASS" : "FAIL");

    cudaFree(d_ctx); cudaFree(d_idx); cudaFree(d_meta);
    cudaFree(d_oq); cudaFree(d_ot); cudaFree(d_ob); cudaFree(d_osp);
    cudaFree(d_onk); cudaFree(d_oa); cudaFree(d_oc);
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("P20-1 SM90 paged MQA logits 调度器：H100 真实设备代码验证\n");
    printf("GPU: %s  sm_%d%d  SMs=%d\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    constexpr uint32_t SMS = 132;   // H100
    bool all = true;

    // 构造非 varlen 场景：2D context lens [batch, next_n]
    auto ctx_2d = [](uint32_t batch, uint32_t next_n, const std::vector<uint32_t>& lens) {
        std::vector<int> out;
        for (uint32_t i = 0; i < batch; ++i)
            for (uint32_t j = 0; j < next_n; ++j)
                out.push_back((int)lens[i]);
        return out;
    };
    auto idx_pairs = [](const std::vector<std::pair<int,int>>& reqs) {
        std::vector<int> idx;
        for (auto& r : reqs) for (int t = 0; t < r.second; ++t) idx.push_back(r.first);
        return idx;
    };
    auto ctx_from_reqs = [](const std::vector<std::pair<int,int>>& reqs,
                            const std::vector<uint32_t>& ctx_per_req) {
        std::vector<int> out;
        for (size_t r = 0; r < reqs.size(); ++r)
            for (int t = 0; t < reqs[r].second; ++t)
                out.push_back((int)ctx_per_req[r]);
        return out;
    };

    printf("== 用例组 A：非 varlen，next_n=1，2D context lens（batch x 1）==\n");
    {
        std::vector<uint32_t> lens;
        for (uint32_t i = 0; i < 64; ++i) lens.push_back(2048);
        all &= run_case<false, 1, true, 64, SMS>("uniform batch=64 ctx=2048", 64, {}, ctx_2d(64, 1, lens));
    }
    {
        std::vector<uint32_t> lens;
        const uint32_t L[] = {128,257,300,384,700,1024,1025,2048,4096,8192};
        for (uint32_t i = 0; i < 60; ++i) lens.push_back(L[i % 10]);
        all &= run_case<false, 1, true, 64, SMS>("skewed lens batch=60", 60, {}, ctx_2d(60, 1, lens));
    }
    {
        // batch 小于 SM 数：大量空 SM
        std::vector<uint32_t> lens = {64, 128, 4096};
        all &= run_case<false, 1, true, 32, SMS>("few batch=3 -> empty SM", 3, {}, ctx_2d(3, 1, lens));
    }
    {
        // batch 不是 32 的倍数（aligned_batch_size 补零）
        std::vector<uint32_t> lens;
        for (uint32_t i = 0; i < 37; ++i) lens.push_back(1000 + i * 37);
        all &= run_case<false, 1, true, 64, SMS>("batch=37 (unaligned)", 37, {}, ctx_2d(37, 1, lens));
    }

    printf("\n== 用例组 B：非 varlen，next_n=2（每 atom 覆盖 2 个 token，2D context lens）==\n");
    {
        std::vector<uint32_t> lens;
        for (uint32_t i = 0; i < 40; ++i) lens.push_back(1536 + (i % 5) * 512);
        all &= run_case<false, 2, true, 64, SMS>("next_n=2 batch=40", 40, {}, ctx_2d(40, 2, lens));
    }
    {
        // 2D context lens 只取最后一行（`q_idx*next_n + next_n-1`）：让同 batch 两行不同
        // 这里构造 batch=8 next_n=2，第二列给大值，验证取的是最后一列
        std::vector<int> cl;
        for (uint32_t i = 0; i < 8; ++i) { cl.push_back(64); cl.push_back(4096); }
        all &= run_case<false, 2, true, 32, SMS>("next_n=2 last-col wins", 8, {}, cl);
    }

    printf("\n== 用例组 C：varlen（indices 配对），next_n=1 ==\n");
    {
        std::vector<std::pair<int,int>> reqs;
        for (int r = 0; r < 64; ++r) reqs.push_back({r, 8});
        std::vector<uint32_t> cp;
        for (int r = 0; r < 64; ++r) cp.push_back(2048);
        all &= run_case<true, 1, true, 512, SMS>("varlen uniform req=8", 512,
                                                 idx_pairs(reqs), ctx_from_reqs(reqs, cp));
    }
    {
        // 配对：同一 request 连续两 token 的 context_lens 不同，metadata 取 [t+1]
        std::vector<std::pair<int,int>> reqs;
        const int lens[] = {1,2,3,4,6,8,12,16,32};
        const uint32_t ctxs[] = {128,257,384,700,1025,2048,4096,8192,300};
        for (int r = 0; r < 9; ++r) reqs.push_back({r, lens[r]});
        // context_len 对每个 token 给不同值：偶数 token 给小值、奇数 token 给大值，
        // 验证配对时 metadata 取的是 pair 的 [t+1]（较大者）
        std::vector<int> idx, cl;
        for (size_t r = 0; r < reqs.size(); ++r)
            for (int t = 0; t < reqs[r].second; ++t) {
                idx.push_back(reqs[r].first);
                cl.push_back((t % 2 == 1) ? (int)ctxs[r] : (int)(ctxs[r] / 2));
            }
        all &= run_case<true, 1, true, 96, SMS>("varlen paired ctx cols", (uint32_t)idx.size(), idx, cl);
    }
    {
        // 少请求：很多空 SM
        std::vector<std::pair<int,int>> reqs = {{0,8},{1,16},{2,8}};
        std::vector<uint32_t> cp = {4096, 1024, 8192};
        all &= run_case<true, 1, true, 32, 8>("varlen 3 req -> 8SM", 32,
                                              idx_pairs(reqs), ctx_from_reqs(reqs, cp));
    }

    printf("\n== 用例组 D：少 SM（8）观察边界 / 空 SM ==\n");
    {
        std::vector<uint32_t> lens;
        const uint32_t L[] = {2048, 512, 4096, 128, 8192, 64};
        for (uint32_t i = 0; i < 6; ++i) lens.push_back(L[i]);
        all &= run_case<false, 1, true, 32, 8>("nonvarlen skewed 8 SM", 6, {}, ctx_2d(6, 1, lens));
    }

    printf("\n%s\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
