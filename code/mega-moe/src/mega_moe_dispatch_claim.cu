// P184-1：Mega MoE dispatch 段的「round-robin min-peeling」token 认领协议。
//
// 目标：这是 `sm100_fp8_fp4_mega_moe.cuh:439-534` 里把「本 rank 第 token_idx_in_expert
// 个待拉取的 (expert) token」映射到「某个源 rank 的哪个 token_idx_in_rank」的那段**纯整数**算法。
// 它决定了 dispatch pull 的源坐标、源 topk 索引，进而决定 combine 反查与（前向）wgrad 的 token 顺序。
//
// 关键：该算法**与 SM100 无关**——只用 warp intrinsics（__reduce_add/min_sync、__ballot_sync、
// __fns），不碰 UMMA/TMA/NVLink。故在 H100(sm90a) 上直接编译运行真实设备代码即可。
//
// 本程序做三件事：
//   A. **忠实复刻**：把 kernel 里那段 while 循环逐行搬到 template<kNumRanks> 的 kernel，
//      在所有 token 上与一个独立的 host「聚合模型」oracle 逐位对拍（含 kNumRanksPerLane>=2）。
//   B. **双射/无锁唯一性**：验证 token_idx 全域枚举出的 (rank, token_idx_in_rank) 恰好是
//      {(r,t) : 0<=t<counts[r]} 的一次排列（每个远端槽位被唯一认领一次）；并按
//      grid-stride（kNumSMs*kNumDispatchWarps 个 warp）验证各 warp 的 token 集合不相交、
//      并集覆盖全部 token。
//   C. **交错性（为什么它能喂满 NVLink）**：量 min-peeling 的「源 rank 序列」相对朴素
//      「按 counts 前缀连续切段」在每 warp 内的 rank 覆盖数与最长同 rank 连跑，说明
//      min-peeling 把每个 warp 的 pull 摊到更多源 rank 上（并行 NVLink），而不是钉在 1-2 个 rank。
//
// 编译：nvcc -std=c++20 -O2 -arch=sm_90a src/mega_moe_dispatch_claim.cu -o /tmp/claim
// 运行：/tmp/claim

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <string>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("[FATAL] %s : %s\n", #x, cudaGetErrorString(e)); exit(2);} } while (0)

// ---------------------------------------------------------------------------
// 真实设备代码：逐行复刻 kernel:439-534 的 min-peeling 映射（一个 warp 认领一批 token）。
//   counts[r]      : 当前 expert 下源 rank r 的到达 token 数（对称缓冲里的 expert_recv_count）
//   total          : Σ counts
//   out_rank[t]    : 全局第 t 个 token 的源 rank
//   out_t[t]       : 该 token 在源 rank 缓冲里的行号 token_idx_in_rank
// 每个 warp 处理 token_idx = warp; warp + num_warps; ...（grid-stride，与 kernel 一致）。
// ---------------------------------------------------------------------------
template <uint32_t kNumRanks>
__global__ void claim_kernel(const uint32_t* __restrict__ counts,
                             uint32_t total,
                             uint32_t* __restrict__ out_rank,
                             uint32_t* __restrict__ out_t) {
    constexpr uint32_t kNumRanksPerLane = (kNumRanks + 31u) / 32u;  // = constexpr_ceil_div(kNumRanks,32)
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t num_warps = (gridDim.x * blockDim.x) >> 5;

    // kernel:441-478：本 lane 持有 rank 集合 {i*32+lane} 的计数（j>=kNumRanks 补 0）。
    uint32_t stored_rank_count[kNumRanksPerLane];
    #pragma unroll
    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
        const uint32_t j = i * 32u + lane;
        stored_rank_count[i] = (j < kNumRanks) ? counts[j] : 0u;
    }

    for (uint32_t token_idx = warp; token_idx < total; token_idx += num_warps) {
        // ---- kernel:480-528 逐行复刻 ----
        uint32_t remaining[kNumRanksPerLane];
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) remaining[i] = stored_rank_count[i];

        uint32_t offset = 0;
        uint32_t slot_idx = token_idx;            // token_idx_in_expert
        uint32_t current_rank_in_expert_idx = 0;
        uint32_t token_idx_in_rank = 0;
        while (true) {
            uint32_t num_actives_in_lane = 0;
            uint32_t min_in_lane = 0xffffffffu;
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                num_actives_in_lane += (remaining[i] > 0);
                if (remaining[i] > 0)
                    min_in_lane = (min_in_lane < remaining[i]) ? min_in_lane : remaining[i];
            }
            const uint32_t num_active_ranks = __reduce_add_sync(0xffffffffu, num_actives_in_lane);
            const uint32_t length = __reduce_min_sync(0xffffffffu, min_in_lane);
            const uint32_t num_round_tokens = length * num_active_ranks;
            if (slot_idx < num_round_tokens) {
                const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                uint32_t num_seen_ranks = 0;
                current_rank_in_expert_idx = 0;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t mask = __ballot_sync(0xffffffffu, remaining[i] > 0);
                    const uint32_t num_active_lanes = __popc(mask);
                    if (slot_idx_in_round >= num_seen_ranks &&
                        slot_idx_in_round < num_seen_ranks + num_active_lanes)
                        current_rank_in_expert_idx = i * 32u + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                    num_seen_ranks += num_active_lanes;
                }
                token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                break;
            }
            slot_idx -= num_round_tokens;
            offset += length;
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] -= ((remaining[i] < length) ? remaining[i] : length);
        }
        if (lane == 0) {  // current_rank / token_idx_in_rank 在全 warp 一致，lane0 落盘即可
            out_rank[token_idx] = current_rank_in_expert_idx;
            out_t[token_idx] = token_idx_in_rank;
        }
    }
}

// ---------------------------------------------------------------------------
// 独立 host「聚合模型」oracle：不模拟 lane/ballot，只按「每轮活跃 rank 各出 min(remaining) 个、
// 轮内 round-robin 摊」的语义，直接解析 token_idx 落点。用来和上面的真实 warp 实现逐位对拍。
// ---------------------------------------------------------------------------
struct Claim { uint32_t rank, t; };

static Claim claim_host(const std::vector<uint32_t>& counts, uint32_t token_idx) {
    const uint32_t R = (uint32_t)counts.size();
    std::vector<uint32_t> remaining = counts;
    uint32_t offset = 0, slot = token_idx;
    while (true) {
        uint32_t nactive = 0, len = 0xffffffffu;
        for (uint32_t r = 0; r < R; ++ r)
            if (remaining[r] > 0) { nactive ++; len = std::min(len, remaining[r]); }
        const uint64_t nround = (uint64_t)len * nactive;
        if ((uint64_t)slot < nround) {
            const uint32_t in_round = slot % nactive;
            // 第 in_round 个活跃 rank（活跃 rank 按升序）
            uint32_t seen = 0, rank = 0;
            for (uint32_t r = 0; r < R; ++ r)
                if (remaining[r] > 0) {
                    if (seen == in_round) { rank = r; break; }
                    seen ++;
                }
            return { rank, offset + (uint32_t)(slot / nactive) };
        }
        slot -= (uint32_t)nround;
        offset += len;
        for (uint32_t r = 0; r < R; ++ r)
            if (remaining[r] > 0) remaining[r] -= std::min(remaining[r], len);
    }
}

static int g_pass = 0, g_fail = 0;
static bool g_quiet = false;
static void chk(bool ok, const std::string& msg) {
    if (!g_quiet || !ok) printf("  [%s] %s\n", ok ? "OK" : "FAIL", msg.c_str());
    ok ? g_pass++ : g_fail++;
}

// 对给定 counts 与 [1..kMaxRanks] 的真实 kernel 全量对拍 + 双射检查。
template <uint32_t kNumRanks>
static int run_case(const std::vector<uint32_t>& counts, const std::string& tag, bool verbose) {
    const uint32_t total = std::accumulate(counts.begin(), counts.end(), 0u);
    std::vector<uint32_t> h_counts(kNumRanks, 0);
    for (uint32_t r = 0; r < kNumRanks; ++ r) h_counts[r] = counts[r];

    uint32_t *d_c, *d_rank, *d_t;
    CK(cudaMalloc(&d_c, kNumRanks * 4));
    CK(cudaMalloc(&d_rank, std::max(total, 1u) * 4));
    CK(cudaMalloc(&d_t, std::max(total, 1u) * 4));
    CK(cudaMemcpy(d_c, h_counts.data(), kNumRanks * 4, cudaMemcpyHostToDevice));

    const int threads = 32;                 // 一个 warp
    const int num_warps = 64;               // 64 个 warp 做 grid-stride 切分
    const int blocks = num_warps;           // 每块 1 warp
    claim_kernel<kNumRanks><<<blocks, threads>>>(d_c, total, d_rank, d_t);
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_rank(total), h_t(total);
    CK(cudaMemcpy(h_rank.data(), d_rank, total * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_t.data(), d_t, total * 4, cudaMemcpyDeviceToHost));

    int mism = 0;
    for (uint32_t i = 0; i < total; ++ i) {
        Claim c = claim_host(counts, i);
        if (c.rank != h_rank[i] || c.t != h_t[i]) {
            if (mism < 5) printf("      mismatch token %u: cuda=(%u,%u) oracle=(%u,%u)\n",
                                 i, h_rank[i], h_t[i], c.rank, c.t);
            mism ++;
        }
    }
    chk(mism == 0, tag + " : kernel vs host-oracle 逐位 (" + std::to_string(total) + " tokens, mism=" + std::to_string(mism) + ")");

    // 双射：每个 (rank,t) 恰好出现一次
    std::vector<uint8_t> seen(total, 0);   // 用 rank 连续区间编号
    std::vector<uint32_t> base(kNumRanks + 1, 0);
    for (uint32_t r = 0; r < kNumRanks; ++ r) base[r + 1] = base[r] + counts[r];
    int dup = 0, oob = 0;
    for (uint32_t i = 0; i < total; ++ i) {
        if (h_rank[i] >= kNumRanks || h_t[i] >= counts[h_rank[i]]) { oob ++; continue; }
        uint32_t id = base[h_rank[i]] + h_t[i];
        if (seen[id]) dup ++; else seen[id] = 1;
    }
    int uncovered = 0; for (uint32_t i = 0; i < total; ++ i) if (!seen[i]) uncovered ++;
    chk(dup == 0 && oob == 0 && uncovered == 0,
        tag + " : 双射 (dup=" + std::to_string(dup) + ", oob=" + std::to_string(oob) +
        ", uncovered=" + std::to_string(uncovered) + ")");

    // 频率公平：经验 rank 频率 == counts/total
    int freq_bad = 0;
    std::vector<uint32_t> freq(kNumRanks, 0);
    for (uint32_t i = 0; i < total; ++ i) freq[h_rank[i]] ++;
    for (uint32_t r = 0; r < kNumRanks; ++ r) if (freq[r] != counts[r]) freq_bad ++;
    chk(freq_bad == 0, tag + " : 经验 rank 频率 == counts 精确 (bad=" + std::to_string(freq_bad) + ")");

    if (verbose) {
        printf("    counts = [");
        for (uint32_t r = 0; r < kNumRanks; ++ r) printf("%s%u", r ? "," : "", counts[r]);
        printf("] (total=%u)\n", total);
        printf("    rank 序列 [前 64]: ");
        for (uint32_t i = 0; i < std::min(total, 64u); ++ i) printf("%u ", h_rank[i]);
        printf("\n");
        // 轮次分解
        std::vector<uint32_t> rem = counts; uint32_t off = 0; int round = 0;
        printf("    轮次: ");
        while (true) {
            uint32_t na = 0, len = 0xffffffffu;
            for (uint32_t r = 0; r < kNumRanks; ++ r) if (rem[r] > 0) { na ++; len = std::min(len, rem[r]); }
            if (na == 0) break;
            printf("#%d(len=%u,active=%u,tokens=%llu) ", round, len, na, (unsigned long long)(uint64_t)len * na);
            for (uint32_t r = 0; r < kNumRanks; ++ r) if (rem[r] > 0) rem[r] -= std::min(rem[r], len);
            off += len; round ++;
        }
        printf("\n");
    }

    cudaFree(d_c); cudaFree(d_rank); cudaFree(d_t);
    return mism + dup + oob + uncovered + freq_bad;
}

// 按 counts 前缀连续切段的「朴素映射」：token i → rank = 第 i 个 token 所属段，t = i - 段起点。
struct Plain { uint32_t rank, t; };
static Plain plain_map(const std::vector<uint32_t>& counts, uint32_t i) {
    uint32_t p = 0;
    for (uint32_t r = 0; r < counts.size(); ++ r) {
        if (i < p + counts[r]) return { r, i - p };
        p += counts[r];
    }
    return { 0, 0 };
}

// C. 交错性：把 token 按 grid-stride（num_warps 个 warp）切分，统计每个 warp 内
//    * distinct ranks*（覆盖多少源 rank）与 *最长同 rank 连跑*，对 min-peeling vs 朴素连续映射。
static void interleave_stats(const std::vector<uint32_t>& counts,
                             const std::vector<uint32_t>& h_rank,
                             const std::vector<uint32_t>& h_t,
                             uint32_t num_warps, const std::string& tag) {
    const uint32_t total = h_rank.size();
    auto warp_stats = [&](bool use_peel) {
        double sum_distinct = 0; long long max_run_all = 0; double sum_run = 0; int warps = 0;
        for (uint32_t w = 0; w < num_warps; ++ w) {
            uint32_t last = 0xffffffffu; long long run = 0; long long maxrun = 0;
            std::vector<uint8_t> rseen(counts.size(), 0); int distinct = 0;
            for (uint32_t i = w; i < total; i += num_warps) {
                uint32_t r = use_peel ? h_rank[i] : plain_map(counts, i).rank;
                if (!rseen[r]) { rseen[r] = 1; distinct ++; }
                if (r == last) run ++; else run = 1;
                maxrun = std::max(maxrun, run); last = r;
            }
            if ((w % num_warps) < total) { sum_distinct += distinct; sum_run += maxrun; max_run_all = std::max(max_run_all, maxrun); warps ++; }
        }
        return std::make_tuple(sum_distinct / warps, sum_run / warps, max_run_all);
    };
    auto [pd, pr, pmax] = warp_stats(true);
    auto [cd, cr, cmax] = warp_stats(false);
    printf("  %s (num_warps=%u):\n", tag.c_str(), num_warps);
    printf("    [每 warp 视角] min-peeling : 平均覆盖 %.1f 个源 rank / 平均最长同 rank 连跑 %.1f / 全局最长 %lld\n", pd, pr, pmax);
    printf("    [每 warp 视角] 朴素连续切段: 平均覆盖 %.1f 个源 rank / 平均最长同 rank 连跑 %.1f / 全局最长 %lld\n", cd, cr, cmax);

    // 并发窗口视角（才是充要的）：第 k 次迭代时 528 个 warp 处理的是 [k*W,(k+1)*W) 连续 token。
    auto window_stats = [&](bool use_peel) {
        double sum_distinct = 0, sum_run = 0; long long gmax = 0; int single = 0, windows = 0;
        for (uint32_t base = 0; base < total; base += num_warps) {
            uint32_t end = std::min<uint64_t>((uint64_t)base + num_warps, total);
            uint32_t last = 0xffffffffu; long long run = 0, mrun = 0;
            std::vector<uint8_t> seen(counts.size(), 0); int distinct = 0;
            for (uint32_t i = base; i < end; ++ i) {
                uint32_t r = use_peel ? h_rank[i] : plain_map(counts, i).rank;
                if (!seen[r]) { seen[r] = 1; distinct ++; }
                if (r == last) run ++; else run = 1;
                mrun = std::max(mrun, run); last = r;
            }
            sum_distinct += distinct; sum_run += mrun; gmax = std::max(gmax, mrun);
            if (distinct == 1) single ++;
            windows ++;
        }
        return std::make_tuple(sum_distinct / windows, sum_run / windows, gmax, single, windows);
    };
    auto [wd, wr, wmax, wsingle, wn] = window_stats(true);
    auto [zd, zr, zmax, zsingle, zn] = window_stats(false);
    printf("    [并发窗口视角] min-peeling : 每窗口平均覆盖 %.1f 个源 rank / 平均最长连跑 %.1f / 全局最长 %lld / 单 rank 窗口 %d/%d\n", wd, wr, wmax, wsingle, wn);
    printf("    [并发窗口视角] 朴素连续切段: 每窗口平均覆盖 %.1f 个源 rank / 平均最长连跑 %.1f / 全局最长 %lld / 单 rank 窗口 %d/%d\n", zd, zr, zmax, zsingle, zn);
}

int main() {
    printf("== P184-1 Mega MoE dispatch round-robin min-peeling 认领协议（H100 sm90a 真机）==\n\n");

    // ---- A/B：逐行对拍 + 双射（含零计数、极端偏斜、kNumRanksPerLane>=2）----
    printf("[A/B] 忠实复刻 + 双射/唯一认领\n");
    run_case<8>({10, 6, 0, 3, 8, 1, 0, 4}, "R=8 含零+偏斜", true);
    run_case<8>({7, 7, 7, 7, 7, 7, 7, 7}, "R=8 均匀", false);
    run_case<8>({100, 0, 0, 0, 0, 0, 0, 0}, "R=8 单 rank", false);
    run_case<8>({1, 1, 1, 1, 1, 1, 1, 1}, "R=8 全 1", false);
    run_case<4>({12, 0, 5, 3}, "R=4", false);
    run_case<16>({9, 0, 4, 7, 2, 0, 6, 3, 8, 1, 5, 0, 2, 9, 4, 6}, "R=16", false);
    run_case<32>({5, 3, 0, 8, 2, 7, 1, 0, 6, 4, 9, 2, 3, 5, 0, 8, 1, 7, 4, 6, 2, 0, 9, 3, 5, 1, 8, 4, 6, 2, 7, 0}, "R=32", false);
    run_case<64>({5,3,0,8,2,7,1,0,6,4,9,2,3,5,0,8,1,7,4,6,2,0,9,3,5,1,8,4,6,2,7,0,
                  3,6,1,0,4,8,2,5,7,0,9,1,3,6,4,2,8,0,5,7,1,9,3,4,6,2,0,8,5,1,7,3}, "R=64 (kNumRanksPerLane=2)", false);
    run_case<72>(std::vector<uint32_t>(72, 3), "R=72 (kNumRanksPerLane=3)", false);

    // 随机 sweep：200 组随机 (R, counts)
    printf("\n[B2] 随机 sweep（200 组，R∈{8,16,32,64,72}，含零/偏斜）\n");
    std::mt19937 rng(12345);
    int sweep_bad = 0;
    g_quiet = true;
    for (int it = 0; it < 200; ++ it) {
        static const uint32_t Rs[5] = {8, 16, 32, 64, 72};
        uint32_t R = Rs[rng() % 5];
        std::vector<uint32_t> c(R);
        // 一半均匀、一半偏斜（Dirichlet-ish）
        bool skew = (it % 2);
        for (uint32_t r = 0; r < R; ++ r) {
            uint32_t v = skew ? (uint32_t)(rng() % 40) : (uint32_t)(rng() % 8);
            if (rng() % 5 == 0) v = 0;  // 制造零
            c[r] = v;
        }
        if (std::accumulate(c.begin(), c.end(), 0u) == 0) c[0] = 1;
        int bad = 0;
        if (R == 8)  bad = run_case<8>(c, "sweep R=8", false);
        else if (R == 16) bad = run_case<16>(c, "sweep R=16", false);
        else if (R == 32) bad = run_case<32>(c, "sweep R=32", false);
        else if (R == 64) bad = run_case<64>(c, "sweep R=64", false);
        else bad = run_case<72>(c, "sweep R=72", false);
        if (bad) sweep_bad ++;
    }
    g_quiet = false;
    chk(sweep_bad == 0, "随机 sweep 200 组全部逐位+双射正确 (bad=" + std::to_string(sweep_bad) + ")");

    // ---- C：交错性 / NVLink 覆盖面 ----
    printf("\n[C] 交错性：min-peeling vs 朴素连续切段（生产配置 kNumSMs*kNumDispatchWarps = 132*4 = 528 warp）\n");
    std::vector<uint32_t> counts = {4096, 3072, 0, 1024, 5120, 2048, 256, 1280};  // 偏斜+零，sum=16896
    uint32_t total = std::accumulate(counts.begin(), counts.end(), 0u);
    // 用 R=8 kernel 生成 min-peeling 结果
    {
        uint32_t *d_c, *d_rank, *d_t;
        CK(cudaMalloc(&d_c, 8 * 4));
        CK(cudaMalloc(&d_rank, total * 4));
        CK(cudaMalloc(&d_t, total * 4));
        CK(cudaMemcpy(d_c, counts.data(), 8 * 4, cudaMemcpyHostToDevice));
        claim_kernel<8><<<528, 32>>>(d_c, total, d_rank, d_t);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> hr(total), ht(total);
        CK(cudaMemcpy(hr.data(), d_rank, total * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ht.data(), d_t, total * 4, cudaMemcpyDeviceToHost));
        interleave_stats(counts, hr, ht, 528, "counts=[4096,3072,0,1024,5120,2048,256,1280]");
        // 用真实设备结果核对 host oracle 一次（C 段结果的可信度）
        int mism = 0;
        for (uint32_t i = 0; i < total; ++ i) { Claim c = claim_host(counts, i); if (c.rank != hr[i] || c.t != ht[i]) mism ++; }
        chk(mism == 0, "C 段 rank 序列 kernel vs oracle 逐位 (mism=" + std::to_string(mism) + ")");
        cudaFree(d_c); cudaFree(d_rank); cudaFree(d_t);
    }

    printf("\n== 判定: PASS=%d FAIL=%d ==\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
