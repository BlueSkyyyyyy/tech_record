// P127-1 实验：DeepGEMM 稀疏 MQA logits 的 **metadata kernel** 主循环
//   `deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh`（475 行）+
//   `layout/sparse_mqa_logits.cuh`（156 行）。
//
// 为什么能在 H100(sm90) 上跑**真实设备代码**：第 22 篇把这个 metadata kernel 只当「概览」讲，
// 但事实上它**完全架构无关**——只用 `ptx::` 整数/原子、`cp.async`(sm80)、`math::cta_exclusive_sum`、
// `cute::` 下标工具，**不碰任何 SM100 指令**（UMMA/tcgen05/UTCCP/TMA 都在主 kernel 里）。所以本篇
// 直接把真实头文件 include 进来、用真实 launch 参数在 H100 上跑，并把输出 metadata 与一份**独立的
// host oracle**（重写「有效块前缀 → 两 token 归并去重 → 槽打包 → 物理块映射 → split/调度」）逐字节对拍。
//
//   A) 连续对齐：归并去重、双 Q 槽 16+16 位打包、物理块 = logical*SB、split 头（contiguous/tail 标记）；
//   B) 连续非对齐 ks：logical 已含 `*SB + ks%SB`，物理块 = 绝对 token 起点；
//   C) 分页：物理块经 block_table 映射（sparse 块不跨页）、header q1_slot_base=~0 语义；
//   D) 负载均衡调度：`build_contiguous_schedule` / `build_paged_schedule`+`balance_wave_entries`
//      产出的 `[wave][SM]` 条目对所有 KV split **恰好覆盖一次**，并打印每 SM 工作量分布。
//
// 编译：见 harness/exp_sparse_mqa_logits_metadata.sh

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>

#include <deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using namespace deep_gemm::layout::sparse_mqa_logits;
namespace sched_ns = deep_gemm::sched::sparse_mqa_logits;

// ---------------------------------------------------------------------------
// 实验固化的模板常量（对照 host wrapper `sm100_sparse_mqa_logits.hpp:135-200`）
//   SPLIT_KV 取 128（而非线上 512/640）只是为了在可控规模下产生较多 KV split，
//   静态断言只要求 SPLIT_KV % 128 == 0 且 % SB == 0；KPS = SPLIT_KV/SB。
// ---------------------------------------------------------------------------
static constexpr int kBLOCK_Q = 2;
static constexpr int kNumThreads = 256;
static constexpr int kSPLIT_KV = 128;
static constexpr int kSB = 8;
static constexpr int kNB = 32;                       // num_max_sparse_blocks
static constexpr int kKPS = kSPLIT_KV / kSB;         // 16 blocks per split
static constexpr int kKVSplitsPerEntry = 8;
static constexpr int kNumSMs = 4;                    // schedule 按 4 个 SM 排布
static constexpr int kPAGE_KV = 64;
static constexpr int kBlocksPerPage = kPAGE_KV / kSB;  // 8

static int64_t metadata_bytes(int T, bool paged) {
    const int64_t split_bytes = sizeof(KVSplitHeader) + kKPS * sizeof(KVBlockInfo);
    const int64_t nmax = paged
        ? (int64_t)T * ((kNB + kKPS - 1) / kKPS)
        : (int64_t)((T + 1) / 2) * ((2 * kNB + kKPS - 1) / kKPS);
    const int64_t sched = (nmax + kNumSMs - 1) / kNumSMs * kNumSMs;
    return 16 + nmax * split_bytes + sched * 16 + 4096;
}

// ---------------------------------------------------------------------------
// 输入样例
// ---------------------------------------------------------------------------
struct CaseInput {
    bool paged;
    bool unaligned;
    int T;
    int Nkv;
    std::vector<int> sparse;      // [T*kNB] 升序唯一的逻辑块号（前缀有效）
    std::vector<int> start, end;  // [T] 连续版
    std::vector<int> ctx;         // [T] 分页版
    std::vector<int> indices;     // [T] 分页版 request id
    std::vector<int> block_table; // [T*stride] 分页版
    int block_table_stride;
};

// 独立 host oracle 的归并元素
struct M {
    int logical;
    bool in0, in1;
    int s0, s1;
};

static std::vector<M> merge_dedup(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<M> out;
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        M m{0, false, false, -1, -1};
        if (j >= b.size() || (i < a.size() && a[i] < b[j])) {
            m.logical = a[i]; m.in0 = true; m.s0 = (int)i; ++i;
        } else if (i >= a.size() || b[j] < a[i]) {
            m.logical = b[j]; m.in1 = true; m.s1 = (int)j; ++j;
        } else {
            m.logical = a[i]; m.in0 = m.in1 = true; m.s0 = (int)i; m.s1 = (int)j; ++i; ++j;
        }
        out.push_back(m);
    }
    return out;
}

// 每个 token 的有效块列表（contiguous/unaligned/paged 三种取值口径）
static std::vector<int> token_list(const CaseInput& in, int t) {
    int k;
    if (in.paged) {
        k = std::min(kNB, (in.ctx[t] + kSB - 1) / kSB);
    } else {
        int kb = std::min(in.start[t], in.Nkv);
        int ke = std::max(kb, std::min(in.end[t], in.Nkv));
        k = std::min(kNB, (ke - kb + kSB - 1) / kSB);
    }
    std::vector<int> list;
    for (int i = 0; i < k; ++i) {
        int raw = in.sparse[t * kNB + i];
        if (in.paged) list.push_back(raw);                                  // logical block
        else if (in.unaligned) list.push_back(raw * kSB + (in.start[t] % kSB));  // absolute token
        else list.push_back(raw);
    }
    return list;
}

static int expected_physical(const CaseInput& in, int qbase, int logical) {
    if (in.paged) {
        int page = logical / kBlocksPerPage;
        int within = logical % kBlocksPerPage;
        return in.block_table[qbase * in.block_table_stride + page] * kBlocksPerPage + within;
    }
    return in.unaligned ? logical : logical * kSB;
}

// ---------------------------------------------------------------------------
// 逐字段核对单个 q block：归并顺序、槽偏移、物理块、split 头
// ---------------------------------------------------------------------------
struct Checker {
    int mism = 0, checked_blocks = 0, checked_splits = 0;
    void bad(const char* what, int qb, int split, int k, int got, int exp) {
        if (got != exp) {
            printf("   [MISMATCH] %s qb=%d split=%d k=%d got=%d exp=%d\n", what, qb, split, k, got, exp);
            ++mism;
        }
    }
};

// contiguous（aligned/unaligned）
static void check_contiguous(const CaseInput& in, const uint8_t* meta, const uint8_t* ws, Checker& ck) {
    const auto* hdr = reinterpret_cast<const MetadataHeader*>(meta);
    const uint32_t total = hdr->num_kv_splits;
    const auto* splits = reinterpret_cast<const KVSplit<kKPS>*>(meta + sizeof(MetadataHeader));
    const auto* qbi = reinterpret_cast<const QBlockInfo*>(ws + sizeof(WorkspaceState));
    const int num_q_blocks = (in.T + kBLOCK_Q - 1) / kBLOCK_Q;
    for (int qb = 0; qb < num_q_blocks; ++qb) {
        const int qbase = qb * kBLOCK_Q;
        const int ntokens = std::min(kBLOCK_Q, in.T - qbase);
        std::vector<int> l0 = token_list(in, qbase);
        std::vector<int> l1 = ntokens == kBLOCK_Q ? token_list(in, qbase + 1) : std::vector<int>();
        std::vector<M> merged = merge_dedup(l0, l1);

        const QBlockInfo info = qbi[qbase];
        const int expect_splits = (int)((merged.size() + kKPS - 1) / kKPS);
        ck.bad("qbi.num_kv_splits", qb, -1, 0, (int)info.num_kv_splits, expect_splits);

        for (int sc = 0; sc < (int)info.num_kv_splits; ++sc) {
            const int sidx = info.kv_split_base + sc;
            if (sidx < 0 || (uint32_t)sidx >= total) { printf("   split idx OOB\n"); ++ck.mism; continue; }
            const auto& sp = splits[sidx];
            const uint32_t nblk = KVSplitHeader::get_num_kv_blocks(sp.header.packed_num_kv_blocks);
            const int begin = sc * kKPS;
            const int expect_n = std::min(kKPS, (int)merged.size() - begin);
            ck.bad("split.nblk", qb, sc, 0, (int)nblk, expect_n);
            ck.bad("split.qbase", qb, sc, 0, (int)sp.header.q_token_base, qbase);
            const M& first = merged[begin];
            ck.bad("split.q0_base", qb, sc, 0, (int)sp.header.q0_slot_base, first.in0 ? first.s0 : 0);
            int expect_q1base = (ntokens == kBLOCK_Q) ? (first.in1 ? first.s1 : 0) : (int)0xFFFFFFFFu;
            ck.bad("split.q1_base", qb, sc, 0, (int)sp.header.q1_slot_base, expect_q1base);

            // split 末块（用于 contiguous/partial-tail 标记）
            const M& last = merged[begin + expect_n - 1];
            int last_token_start = in.unaligned ? last.logical : last.logical * kSB;
            bool expect_contig = false;
            if (!in.unaligned)
                expect_contig = (expect_n == kKPS) && (last.logical == first.logical + expect_n - 1);
            bool expect_tail = (!expect_contig) && (last_token_start + kSB > in.Nkv);
            ck.bad("split.is_contiguous", qb, sc, 0,
                   KVSplitHeader::is_contiguous(sp.header.packed_num_kv_blocks) ? 1 : 0, expect_contig ? 1 : 0);
            ck.bad("split.has_partial_tail", qb, sc, 0,
                   KVSplitHeader::has_partial_tail(sp.header.packed_num_kv_blocks) ? 1 : 0, expect_tail ? 1 : 0);

            for (int k = 0; k < expect_n; ++k) {
                const M& m = merged[begin + k];
                const auto& bi = sp.kv_block_infos[k];
                const uint32_t q0off = bi.packed_slot_offsets & 0xFFFFu;
                const uint32_t q1off = (bi.packed_slot_offsets >> 16) & 0xFFFFu;
                int eq0base = sp.header.q0_slot_base, eq1base = (int)(sp.header.q1_slot_base == 0xFFFFFFFFu ? 0 : sp.header.q1_slot_base);
                ck.bad("blk.q0_off", qb, sc, k, (int)q0off, m.in0 ? (m.s0 - eq0base) : (int)kInvalidSparseSlot);
                ck.bad("blk.q1_off", qb, sc, k, (int)q1off, m.in1 ? (m.s1 - eq1base) : (int)kInvalidSparseSlot);
                ck.bad("blk.physical", qb, sc, k, (int)bi.physical_kv_block_idx, expected_physical(in, qbase, m.logical));
                ++ck.checked_blocks;
            }
            ++ck.checked_splits;
        }
    }
}

// paged（单 request）
static void check_paged(const CaseInput& in, const uint8_t* meta, const uint8_t* ws, Checker& ck) {
    const auto* hdr = reinterpret_cast<const MetadataHeader*>(meta);
    const uint32_t total = hdr->num_kv_splits;
    const auto* splits = reinterpret_cast<const KVSplit<kKPS>*>(meta + sizeof(MetadataHeader));
    const auto* qbi = reinterpret_cast<const QBlockInfo*>(ws + sizeof(WorkspaceState));
    for (int qbase = 0; qbase < in.T; qbase += kBLOCK_Q) {
        const int ntokens = std::min(kBLOCK_Q, in.T - qbase);
        std::vector<int> l0 = token_list(in, qbase);
        std::vector<int> l1 = ntokens == kBLOCK_Q ? token_list(in, qbase + 1) : std::vector<int>();
        std::vector<M> merged = merge_dedup(l0, l1);
        // 分页版第二个 token 的 q_block_info 被清零，schedule 由第一个 token 承载
        const QBlockInfo info = qbi[qbase];
        for (int sc = 0; sc < (int)info.num_kv_splits; ++sc) {
            const int sidx = info.kv_split_base + sc;
            if (sidx < 0 || (uint32_t)sidx >= total) { ++ck.mism; continue; }
            const auto& sp = splits[sidx];
            const uint32_t nblk = KVSplitHeader::get_num_kv_blocks(sp.header.packed_num_kv_blocks);
            const int begin = sc * kKPS;
            const int expect_n = std::min(kKPS, (int)merged.size() - begin);
            ck.bad("p.split.nblk", qbase, sc, 0, (int)nblk, expect_n);
            for (int k = 0; k < expect_n; ++k) {
                const M& m = merged[begin + k];
                const auto& bi = sp.kv_block_infos[k];
                ck.bad("p.blk.physical", qbase, sc, k, (int)bi.physical_kv_block_idx,
                       expected_physical(in, qbase, m.logical));
                ++ck.checked_blocks;
            }
            ++ck.checked_splits;
        }
    }
}

// 调度：非空条目恰好覆盖 [0,total) 一次；打印每 wave 每 SM 工作量
static void check_schedule(const uint8_t* meta, bool paged, int print) {
    const auto* hdr = reinterpret_cast<const MetadataHeader*>(meta);
    const uint32_t total = hdr->num_kv_splits;
    const uint32_t nwaves = hdr->num_waves;
    const auto* splits = reinterpret_cast<const KVSplit<kKPS>*>(meta + sizeof(MetadataHeader));
    const auto* sched = reinterpret_cast<const ScheduleEntry*>(splits + total);
    std::vector<int> cover(total, 0);
    int bad = 0, entries = 0;
    for (uint32_t w = 0; w < nwaves; ++w) {
        std::vector<int> load(kNumSMs, 0);
        for (int sm = 0; sm < kNumSMs; ++sm) {
            const auto& e = sched[w * kNumSMs + sm];
            if (e.num_q_tokens == 0) continue;
            ++entries;
            for (uint32_t s = e.kv_split_begin; s < e.kv_split_end; ++s) {
                if (s >= total) { ++bad; continue; }
                cover[s]++; load[sm] += (int)(e.kv_split_end - e.kv_split_begin);
            }
        }
        if (print) {
            int mn = *std::min_element(load.begin(), load.end());
            int mx = *std::max_element(load.begin(), load.end());
            printf("   wave %u: per-SM work=[", w);
            for (int sm = 0; sm < kNumSMs; ++sm) printf("%s%d", sm ? "," : "", load[sm]);
            printf("] spread=max-min=%d\n", mx - mn);
        }
    }
    int miss = 0, dup = 0;
    for (uint32_t s = 0; s < total; ++s) { if (cover[s] == 0) ++miss; else if (cover[s] > 1) ++dup; }
    printf("   schedule: total_kv_splits=%u waves=%u non_empty_entries=%d cover(miss=%d,dup=%d,bad=%d) -> %s\n",
           total, nwaves, entries, miss, dup, bad, (miss == 0 && dup == 0 && bad == 0) ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// 真实 kernel launch
// ---------------------------------------------------------------------------
template <bool kIsPaged, bool kUseUnalignedKs>
static void launch_real(const CaseInput& in, std::vector<uint8_t>& meta_out, std::vector<uint8_t>& ws_out) {
    const int T = in.T;
    const size_t msz = metadata_bytes(T, kIsPaged);
    const size_t wsz = sizeof(WorkspaceState) + (size_t)T * sizeof(QBlockInfo) + 4096;

    int *d_sparse = nullptr, *d_start = nullptr, *d_end = nullptr, *d_ctx = nullptr, *d_ind = nullptr, *d_bt = nullptr;
    uint8_t *d_meta = nullptr, *d_ws = nullptr;
    CK(cudaMalloc(&d_sparse, sizeof(int) * in.sparse.size()));
    CK(cudaMalloc(&d_start, sizeof(int) * std::max(1, T)));
    CK(cudaMalloc(&d_end, sizeof(int) * std::max(1, T)));
    CK(cudaMalloc(&d_ctx, sizeof(int) * std::max(1, T)));
    CK(cudaMalloc(&d_ind, sizeof(int) * std::max(1, T)));
    CK(cudaMalloc(&d_bt, sizeof(int) * std::max<size_t>(1, in.block_table.size())));
    CK(cudaMalloc(&d_meta, msz));
    CK(cudaMalloc(&d_ws, wsz));
    CK(cudaMemset(d_meta, 0, msz));
    CK(cudaMemset(d_ws, 0, wsz));

    CK(cudaMemcpy(d_sparse, in.sparse.data(), sizeof(int) * in.sparse.size(), cudaMemcpyHostToDevice));
    if (!in.paged && T > 0) {
        CK(cudaMemcpy(d_start, in.start.data(), sizeof(int) * T, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_end, in.end.data(), sizeof(int) * T, cudaMemcpyHostToDevice));
    }
    if (in.paged && T > 0) {
        CK(cudaMemcpy(d_ctx, in.ctx.data(), sizeof(int) * T, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ind, in.indices.data(), sizeof(int) * T, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_bt, in.block_table.data(), sizeof(int) * in.block_table.size(), cudaMemcpyHostToDevice));
    }

    const int smem = (int)((sizeof(sched_ns::SharedStorage<kBLOCK_Q, kNB, kKPS, kKVSplitsPerEntry, kNumThreads>) + 127) / 128 * 128);
    auto kfn = sched_ns::sm100_sparse_mqa_logits_metadata<
        kIsPaged, kUseUnalignedKs, kBLOCK_Q, kSPLIT_KV, kSB, kNB, kPAGE_KV,
        kKVSplitsPerEntry, kNumSMs, kNumThreads>;
    CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kfn<<<1, kNumThreads, smem>>>(T, in.Nkv,
                                  (const uint32_t*)d_start, (const uint32_t*)d_end,
                                  (const uint32_t*)d_ctx, (const uint32_t*)d_bt,
                                  (uint32_t)in.block_table_stride, (const uint32_t*)d_ind,
                                  (const uint32_t*)d_sparse, d_meta, d_ws);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    meta_out.assign(msz, 0);
    ws_out.assign(wsz, 0);
    CK(cudaMemcpy(meta_out.data(), d_meta, msz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ws_out.data(), d_ws, wsz, cudaMemcpyDeviceToHost));

    cudaFree(d_sparse); cudaFree(d_start); cudaFree(d_end); cudaFree(d_ctx);
    cudaFree(d_ind); cudaFree(d_bt); cudaFree(d_meta); cudaFree(d_ws);
}

// ---------------------------------------------------------------------------
// 样例构造
// ---------------------------------------------------------------------------
using case_list = std::vector<std::pair<const char*, CaseInput>>;

static case_list build_cases() {
    case_list v;

    // A. 连续对齐：T=6（3 个 Q 块），KV 足够长；两 token 不同长度、刻意共享若干块
    {
        CaseInput in; in.paged = false; in.unaligned = false; in.T = 6; in.Nkv = 512;
        // 每个 token 的 start/end（块 = ceil((end-start)/SB)）
        in.start = {0, 0, 8, 16, 8, 0};
        in.end   = {8 * 6, 8 * 4, 8 * 12, 8 * 6, 8 * 2, 8 * 8};  // 6,4,12,6,2,8 blocks
        // 行内升序唯一的逻辑块号
        in.sparse.assign(in.T * kNB, 0);
        int rows[6][16] = {
            {0,1,2,3,16,17},                       // 6
            {0,2,3,4},                             // 4
            {0,1,4,5,6,20,21,22,40,41,50,60},      // 12
            {0,3,4,5,30,31},                       // 6
            {3,7},                                 // 2
            {0,1,2,5,6,8,9,11},                    // 8
        };
        for (int t = 0; t < 6; ++t) for (int i = 0; i < 16; ++i) in.sparse[t * kNB + i] = rows[t][i];
        v.push_back({"A1 连续对齐 T=6 NB=32 SB=8", in});
    }
    // A2. 更小 KV，触发 has_partial_tail（尾块越过 KV 末尾）
    {
        CaseInput in; in.paged = false; in.unaligned = false; in.T = 4; in.Nkv = 100;
        in.start = {0, 0, 0, 0};
        in.end   = {8 * 20, 8 * 20, 8 * 20, 8 * 20};  // 20 blocks each
        in.sparse.assign(in.T * kNB, 0);
        for (int t = 0; t < 4; ++t) for (int i = 0; i < 20; ++i) in.sparse[t * kNB + i] = i * 5;  // 0,5,10,...
        v.push_back({"A2 连续对齐 尾块裁剪 T=4 Nkv=100", in});
    }
    // B. 连续非对齐：ks%SB=5，logical 变成绝对 token 起点
    {
        CaseInput in; in.paged = false; in.unaligned = true; in.T = 4; in.Nkv = 512;
        in.start = {5, 5, 5, 5};
        in.end   = {8 * 6 + 5, 8 * 10 + 5, 8 * 3 + 5, 8 * 6 + 5};
        in.sparse.assign(in.T * kNB, 0);
        int rows[4][16] = {{0,1,2,10,11,12},{0,2,3,4,11,20,21,30,31,40},{0,5,6},{0,1,2,3,4,5}};
        for (int t = 0; t < 4; ++t) for (int i = 0; i < 16; ++i) in.sparse[t * kNB + i] = rows[t][i];
        v.push_back({"B1 连续非对齐 ks%8=5 T=4", in});
    }
    // C. 分页：单 request、T=8、块经 block_table 映射
    {
        CaseInput in; in.paged = true; in.unaligned = false; in.T = 8; in.Nkv = 0;
        in.ctx = {8 * 10, 8 * 5, 8 * 10, 8 * 5, 8 * 12, 8 * 12, 8 * 4, 8 * 4};
        in.indices.assign(8, 0);
        in.block_table_stride = 4;
        in.block_table.assign(8 * 4, 0);
        // 每 token 的 block_table 行：页号 = 100 + t*4 + page
        for (int t = 0; t < 8; ++t) for (int p = 0; p < 4; ++p) in.block_table[t * 4 + p] = 100 + t * 4 + p;
        in.sparse.assign(in.T * kNB, 0);
        int rows[8][16] = {
            {0,1,2,3,8,9,10,11,16,17},
            {0,1,8,9,16},
            {0,1,2,3,8,9,10,11,16,17},
            {0,1,8,9,16},
            {0,1,2,3,4,8,9,10,11,12,16,17},
            {0,1,2,3,4,8,9,10,11,12,16,17},
            {0,1,2,3},
            {0,1,2,3},
        };
        for (int t = 0; t < 8; ++t) for (int i = 0; i < 16; ++i) in.sparse[t * kNB + i] = rows[t][i];
        v.push_back({"C1 分页 单request T=8", in});
    }
    // A3. 连续对齐、多 wave：每个 Q 块并集 24 块 -> 2 split，8 个 Q 块 -> 16 split（kNumSMs=4 -> 4 wave）
    {
        CaseInput in; in.paged = false; in.unaligned = false; in.T = 16; in.Nkv = 512;
        in.start.assign(16, 0); in.end.assign(16, 24 * 8);
        in.sparse.assign(in.T * kNB, 0);
        for (int t = 0; t < 16; ++t) for (int i = 0; i < 24; ++i) in.sparse[t * kNB + i] = i;
        v.push_back({"A3 连续对齐 多wave T=16", in});
    }
    // C2. 分页、负载不均：6 对，并集大小 32/16 交替 -> split 数 2/1 交替，entries=6 -> 2 wave（触发反向分支）
    {
        CaseInput in; in.paged = true; in.unaligned = false; in.T = 12; in.Nkv = 0;
        const int usz[6] = {32, 32, 16, 16, 32, 16};
        in.indices.assign(12, 0);
        in.block_table_stride = 4;
        in.block_table.assign(12 * 4, 0);
        for (int t = 0; t < 12; ++t) for (int p = 0; p < 4; ++p) in.block_table[t * 4 + p] = 100 + t * 4 + p;
        in.sparse.assign(in.T * kNB, 0);
        in.ctx.assign(12, 0);
        for (int t = 0; t < 12; ++t) {
            const int u = usz[t / 2];
            const int own = (t % 2 == 0) ? u : u / 2;   // 第二 token 是第一个 token 的前 half，并集 = u
            in.ctx[t] = own * kSB;
            for (int i = 0; i < own; ++i) in.sparse[t * kNB + i] = i;
        }
        v.push_back({"C2 分页 负载不均 T=12", in});
    }
    return v;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("== P127-1 稀疏 MQA logits metadata kernel 主循环（真实设备代码，H100/sm90）==\n");
    printf("GPU: %s  cc=%d.%d  | 模板: BLOCK_Q=%d SPLIT_KV=%d SB=%d NB=%d KPS=%d kNumSMs=%d\n\n",
           prop.name, prop.major, prop.minor, kBLOCK_Q, kSPLIT_KV, kSB, kNB, kKPS, kNumSMs);

    auto cases = build_cases();
    int total_mism = 0;
    int idx = 0;
    for (auto& pr : cases) {
        const char* name = pr.first;
        const CaseInput& in = pr.second;
        printf("---- %s ----\n", name);
        std::vector<uint8_t> meta, ws;
        if (!in.paged) {
            if (in.unaligned) launch_real<false, true>(in, meta, ws);
            else              launch_real<false, false>(in, meta, ws);
        } else {
            launch_real<true, false>(in, meta, ws);
        }
        const auto* hdr = reinterpret_cast<const MetadataHeader*>(meta.data());
        printf("   header: num_kv_splits=%u num_waves=%u use_unaligned_ks=%u num_sms=%u\n",
               hdr->num_kv_splits, hdr->num_waves, hdr->use_unaligned_ks, hdr->num_sms);
        Checker ck;
        if (in.paged) check_paged(in, meta.data(), ws.data(), ck);
        else          check_contiguous(in, meta.data(), ws.data(), ck);
        check_schedule(meta.data(), in.paged, /*print=*/1);
        printf("   结果: 核对 %d 块 / %d split，不一致=%d -> %s\n\n",
               ck.checked_blocks, ck.checked_splits, ck.mism, ck.mism == 0 ? "PASS" : "FAIL");
        total_mism += ck.mism;
        ++idx;
    }
    printf("=== 汇总: %s（%d 组样例，总不一致=%d）===\n",
           total_mism == 0 ? "ALL PASS" : "FAIL", idx, total_mism);
    return total_mism == 0 ? 0 : 1;
}
