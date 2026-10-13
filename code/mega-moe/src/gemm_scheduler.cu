// P98-1 实验：DeepGEMM 的通用 GEMM 任务调度器 `scheduler/gemm.cuh`。
//
// 该文件（314 行）被**所有** GEMM / grouped GEMM / MoE 反向 wgrad-dgrad 复用，却被前面 107 篇
// 只以「`scheduler/gemm.cuh:xxx`」的行号引用过、从没被单独拆透。本篇补上，并在 H100(sm90) 上
// **直接 include 真实头文件、编译运行真实设备代码**。
//
// 为什么 H100(sm90) 可跑：`Scheduler` 的三个公开设备方法（`get_next_block` / `get_swizzled_block_idx` /
// `get_global_idx` / `get_aligned_effective_m_in_block` / `is_computation_valid` / `is_tma_multicast_valid`）
// 全是整数索引运算，不碰 UMMA/TMA/tcgen05。它唯一的架构分支是 `#if __CUDA_ARCH__ < 1000`（SM90 的
// 「不齐整 TMA multicast」修补）；在 sm90 上运行即走 SM90 分支，与 H100 上真实反向 grouped GEMM 一致。
// SM100 分支（无修补）只做**编译期**覆盖（同一个 .cu 同时编 `sm_90a` 与 `sm_100a`，H100 运行 sm90 cubin）。
//
// 七组核验：
//   A) `get_num_1d_blocks_per_group` 的枚举启发式 vs 暴力。
//   B) `get_swizzled_block_idx` 的 L2 swizzle + SM90 multicast 修补：穷举 block_idx vs 独立复刻。
//   C) 六种 GemmType 的 `get_next_block` 任务枚举：覆盖性是 [0,N) 的排列 + 逐 CTA 负载；
//      k-grouped 的物理 K 起点 / SF-K 前缀和 vs 规范。
//   D) `get_aligned_effective_m_in_block` 的 psum 尾块 16 对齐（EZ 开/关）。
//   E) `is_computation_valid`（masked / contiguous / psum）vs 独立复刻。
//   F) `is_tma_multicast_valid`（SM90 multicast 对端一致性）vs 独立复刻。
//   G) `get_global_idx` 对 7 种 GemmType × {MN,K,SF_K} 的偏移 vs 规范。
//
// 编译与运行见 harness/exp_gemm_scheduler.sh。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include <deep_gemm/scheduler/gemm.cuh>

using namespace deep_gemm;
using deep_gemm::sched::Scheduler;
using deep_gemm::sched::IndexType;

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1); } } while (0)

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& msg) {
    if (ok) ++g_pass;
    else { ++g_fail; printf("      FAIL: %s\n", msg.c_str()); }
}

// ---------------------------------------------------------------------------
// host 侧的独立复刻（只依赖「规范」，不依赖设备代码）
// ---------------------------------------------------------------------------
static uint32_t hceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static uint32_t halign(uint32_t a, uint32_t b) { return ((a + b - 1) / b) * b; }
static uint32_t hmin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// `get_num_1d_blocks_per_group` 的暴力版（枚举 {8,16}，同 usage 公式取最小）
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static uint32_t host_num_1d_blocks() {
    uint32_t best = 0, min_usage = 0xffffffffu;
    for (uint32_t cand : {8u, 16u}) {
        uint32_t usage = kIsMulticastOnA
            ? cand * BLOCK_N + hceil(kNumSMs, cand) * BLOCK_M
            : cand * BLOCK_M + hceil(kNumSMs, cand) * BLOCK_N;
        if (usage < min_usage) { min_usage = usage; best = cand; }
    }
    return best;
}

struct SwzOut { uint32_t m, n, nbig; };

// `get_swizzled_block_idx` 规范复刻。arch_lt_1000 = 是否走 SM90 multicast 修补。
static SwzOut host_swizzle(uint32_t block_idx, uint32_t num_m, uint32_t num_n,
                           uint32_t kMult, bool MA, uint32_t N1D, bool arch_lt_1000) {
    SwzOut o{};
    const uint32_t primary   = MA ? num_n : num_m;
    const uint32_t secondary = MA ? num_m : num_n;
    const uint32_t npg       = secondary * N1D;
    const uint32_t g         = block_idx / npg;
    uint32_t first           = g * N1D;
    uint32_t ig              = block_idx % npg;
    uint32_t nbig            = hmin(N1D, primary - first);
    if (arch_lt_1000 && kMult > 1 && (nbig % 2 != 0)) {
        if (ig < (nbig ^ 1) * secondary) {
            nbig = nbig ^ 1;
        } else {
            ig -= (nbig ^ 1) * secondary;
            first += (nbig ^ 1);
            nbig = 1;
        }
    }
    if (MA) { o.m = ig / nbig; o.n = first + ig % nbig; }
    else    { o.m = first + ig % nbig; o.n = ig / nbig; }
    o.nbig = nbig;
    return o;
}

// ---------------------------------------------------------------------------
// 设备侧：调用真实 Scheduler
// ---------------------------------------------------------------------------
struct Rec { uint32_t nbi, grp, m, n, kstart, sfcum, nbig; };

template <GemmType GT, uint32_t BM, uint32_t BN, uint32_t NG, uint32_t NM, bool MA,
          uint32_t NSM, bool EZ, uint32_t KA, uint32_t SFK>
__global__ void probe_kernel(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             int* gl, Rec* out, int* cnt, uint32_t cap) {
    using S = Scheduler<GT, BM, BN, NG, NM, MA, NSM, EZ, KA, SFK>;
    S s(shape_m, shape_n, shape_k, gl);
    uint32_t m = 0, n = 0;
    int c = 0;
    while (s.get_next_block(m, n)) {
        if ((uint32_t) c < cap) {
            Rec r;
            r.nbi    = (uint32_t) s.current_iter * NSM + blockIdx.x;
            r.grp    = s.current_group_idx;
            r.m      = m;
            r.n      = n;
            r.kstart = s.current_k_start;
            r.sfcum  = s.current_sf_k_cumsum;
            r.nbig   = s.num_blocks_in_group;
            out[(size_t) blockIdx.x * cap + c] = r;
        }
        ++c;
    }
    cnt[blockIdx.x] = c;
}

template <GemmType GT, uint32_t BM, uint32_t BN, uint32_t NG, uint32_t NM, bool MA,
          uint32_t NSM, bool EZ, uint32_t KA, uint32_t SFK>
__global__ void swizzle_kernel(uint32_t num_m, uint32_t num_n, SwzOut* out) {
    using S = Scheduler<GT, BM, BN, NG, NM, MA, NSM, EZ, KA, SFK>;
    S s(0, 0, 0, nullptr);
    s.num_m_blocks = num_m;
    s.num_n_blocks = num_n;
    const uint32_t total = num_m * num_n;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    SwzOut o{};
    uint32_t m = 0, n = 0;
    s.get_swizzled_block_idx(i, m, n);
    o.m = m; o.n = n; o.nbig = s.num_blocks_in_group;
    out[i] = o;
}

// Part D/E/F/G: 逐参数调用单个方法
template <GemmType GT, uint32_t BM, uint32_t BN, uint32_t NG, uint32_t NM, bool MA,
          uint32_t NSM, bool EZ, uint32_t KA, uint32_t SFK>
__global__ void method_kernel(int* gl, uint32_t* out, uint32_t shape_m, uint32_t shape_n,
                              uint32_t shape_k) {
    using S = Scheduler<GT, BM, BN, NG, NM, MA, NSM, EZ, KA, SFK>;
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    S s(shape_m, shape_n, shape_k, gl);
    if (t == 0) {
        // D) psum 尾块的有效 M：组 g 的 psum 结尾 400、上一组对齐起点 256、num_m_blocks=2
        //    尾块下标 = last_psum_m/BM + num_m_blocks - 1 = 2+2-1 = 3；400-3*128=16 -> 16
        s.num_m_blocks = 2;
        s.current_psum_m = 400;
        s.last_psum_m = 256;
        out[0] = s.get_aligned_effective_m_in_block(3);   // 尾块
        out[1] = s.get_aligned_effective_m_in_block(2);   // 非尾块 -> BLOCK_M
        // E) is_computation_valid（masked 用组计数、psum 用 current_psum_m）
        s.current_group_idx = 0;
        out[2] = s.is_computation_valid(0, 0) ? 1 : 0;
        out[3] = s.is_computation_valid(2, 0) ? 1 : 0;    // 2*128=256 < 400 -> valid
        // F) is_tma_multicast_valid：先 swizzle 设定 num_blocks_in_group
        uint32_t m = 0, n = 0;
        s.get_swizzled_block_idx(0, m, n);
        out[4] = s.is_tma_multicast_valid(m) ? 1 : 0;
        // G) get_global_idx
        out[5]  = s.template get_global_idx<false, IndexType::MN>(shape_m, BM, 2);
        // MGroupedContiguous：组偏移取第 4 参（m_block_idx）指向的 `grouped_layout[m*BLOCK_M]`
        out[6]  = s.template get_global_idx<true,  IndexType::MN>(shape_m, BM, 2, 3);
        out[7]  = s.template get_global_idx<true,  IndexType::K >(shape_m, 128, 1);
        out[8]  = s.template get_global_idx<true,  IndexType::SF_K>(shape_m, 1, 1);
        out[9]  = s.template get_global_idx<false, IndexType::SF_K>(shape_m, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// Part A
// ---------------------------------------------------------------------------
template <uint32_t BM, uint32_t BN, uint32_t NSM, bool MA, GemmType GT>
void part_A(const char* tag) {
    const uint32_t got = sched::get_num_1d_blocks_per_group<GT, BM, BN, NSM, MA>();
    const uint32_t exp = host_num_1d_blocks<BM, BN, NSM, MA>();
    char buf[256];
    snprintf(buf, sizeof buf, "A[%s] BM=%u BN=%u NSM=%u MA=%d -> N1D=%u (exp %u)", tag, BM, BN, NSM, MA, got, exp);
    check(got == exp, buf);
    printf("  %s\n", buf);
}

// ---------------------------------------------------------------------------
// Part B: swizzle 穷举
// ---------------------------------------------------------------------------
template <GemmType GT, uint32_t BM, uint32_t BN, uint32_t NM, bool MA, uint32_t NSM>
void part_B(const char* tag, uint32_t num_m, uint32_t num_n, bool arch_lt_1000) {
    using S = Scheduler<GT, BM, BN, 1, NM, MA, NSM, true, 128, 128>;
    const uint32_t N1D = sched::get_num_1d_blocks_per_group<GT, BM, BN, NSM, MA>();
    const uint32_t total = num_m * num_n;
    SwzOut* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(SwzOut) * total));
    const uint32_t th = 128, blk = hceil(total, th);
    swizzle_kernel<GT, BM, BN, 1, NM, MA, NSM, true, 128, 128><<<blk, th>>>(num_m, num_n, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<SwzOut> h(total);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, sizeof(SwzOut) * total, cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    uint32_t mismatch = 0, oob = 0, fixup_hits = 0;
    std::vector<int> seen(total, 0);
    for (uint32_t i = 0; i < total; ++i) {
        const auto e = host_swizzle(i, num_m, num_n, NM, MA, N1D, arch_lt_1000);
        if (h[i].m != e.m || h[i].n != e.n || h[i].nbig != e.nbig) ++mismatch;
        if (h[i].m >= num_m || h[i].n >= num_n) ++oob;
        seen[h[i].m * num_n + h[i].n]++;
        // multicast 修补是否真的改变了映射（对照「不修补」）
        const auto no_fix = host_swizzle(i, num_m, num_n, NM, MA, N1D, false);
        if (no_fix.m != e.m || no_fix.n != e.n || no_fix.nbig != e.nbig) ++fixup_hits;
    }
    uint32_t dup = 0;
    for (int s : seen) if (s != 1) ++dup;
    char buf[256];
    snprintf(buf, sizeof buf, "B[%s] %ux%u MA=%d N1D=%u -> mismatch=%u oob=%u non-bijective=%u fixup_hits=%u",
             tag, num_m, num_n, MA, N1D, mismatch, oob, dup, fixup_hits);
    check(mismatch == 0 && oob == 0 && dup == 0, buf);
    printf("  %s\n", buf);
}

// ---------------------------------------------------------------------------
// Part C: 任务覆盖
// ---------------------------------------------------------------------------
template <GemmType GT, uint32_t BM, uint32_t BN, uint32_t NG, uint32_t NM, bool MA,
          uint32_t NSM, uint32_t KA = 128u, uint32_t SFK = 128u>
void part_C(const char* tag, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
            const std::vector<int>& gl_host, uint32_t expect_N, bool verbose_kgroup = false) {
    using S = Scheduler<GT, BM, BN, NG, NM, MA, NSM, true, KA, SFK>;
    const uint32_t cap = expect_N;
    int* d_gl = nullptr;
    Rec* d_out = nullptr;
    int* d_cnt = nullptr;
    if (!gl_host.empty()) {
        CUDA_CHECK(cudaMalloc(&d_gl, sizeof(int) * gl_host.size()));
        CUDA_CHECK(cudaMemcpy(d_gl, gl_host.data(), sizeof(int) * gl_host.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(Rec) * cap * NSM));
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int) * NSM));
    probe_kernel<GT, BM, BN, NG, NM, MA, NSM, true, KA, SFK><<<NSM, 1>>>(
        shape_m, shape_n, shape_k, d_gl, d_out, d_cnt, cap);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> cnt(NSM);
    CUDA_CHECK(cudaMemcpy(cnt.data(), d_cnt, sizeof(int) * NSM, cudaMemcpyDeviceToHost));
    std::vector<Rec> all;
    all.reserve(expect_N);
    for (uint32_t sm = 0; sm < NSM; ++sm) {
        std::vector<Rec> r(cnt[sm]);
        CUDA_CHECK(cudaMemcpy(r.data(), d_out + (size_t) sm * cap,
                              sizeof(Rec) * cnt[sm], cudaMemcpyDeviceToHost));
        for (auto& x : r) all.push_back(x);
    }
    if (d_gl) cudaFree(d_gl);
    cudaFree(d_out); cudaFree(d_cnt);

    // 覆盖性：nbi 是 [0,N) 的排列
    std::vector<int> hit(expect_N, 0);
    uint32_t oob = 0;
    for (auto& r : all) { if (r.nbi < expect_N) hit[r.nbi] = 1; else ++oob; }
    uint32_t miss = 0;
    for (int v : hit) if (!v) ++miss;
    uint32_t total_records = (uint32_t) all.size();
    uint32_t dup = total_records > expect_N ? total_records - expect_N : 0;

    char buf[256];
    snprintf(buf, sizeof buf, "C[%s] tasks=%u/%u miss=%u oob=%u dup=%u", tag,
             total_records, expect_N, miss, oob, dup);
    check(total_records == expect_N && miss == 0 && oob == 0, buf);
    printf("  %s\n", buf);

    // k-grouped：物理 K 起点 / SF-K 前缀和 逐组 vs 规范
    if (verbose_kgroup) {
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> g2ks;   // group -> (kstart,sfcum)
        for (auto& r : all) g2ks[r.grp] = {r.kstart, r.sfcum};
        uint32_t bad = 0;
        const uint32_t num_mb = hceil(shape_m, BM);
        if (GT == GemmType::KGroupedContiguousWithPsumLayout) {
            uint32_t prev_end = 0, sf = 0;
            for (uint32_t g = 0; g < NG; ++g) {
                const uint32_t kstart = halign(prev_end, KA);
                const uint32_t shapek = (uint32_t) gl_host[g] - kstart;
                if (g2ks[g] != std::make_pair(kstart, sf)) ++bad;
                sf += hceil(halign(shapek, KA), SFK);
                prev_end = gl_host[g];
            }
        } else {
            uint32_t kstart = 0, sf = 0;
            for (uint32_t g = 0; g < NG; ++g) {
                const uint32_t shapek = gl_host[g];
                if (g2ks[g] != std::make_pair(kstart, sf)) ++bad;
                sf += hceil(halign(shapek, KA), SFK);
                kstart += shapek;
            }
        }
        (void) num_mb;
        char b2[256];
        snprintf(b2, sizeof b2, "C[%s] k-group (kstart,sfcum) bad=%u", tag, bad);
        check(bad == 0, b2);
        printf("      %s\n", b2);
    }
}

// ---------------------------------------------------------------------------
// Part D/G: 单方法
// ---------------------------------------------------------------------------
template <GemmType GT, uint32_t BM, uint32_t BN, uint32_t NG, uint32_t NM, bool MA,
          uint32_t NSM, bool EZ, bool PSUM_TAIL = false>
void part_methods(const char* tag, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                  const std::vector<int>& gl_host) {
    int* d_gl = nullptr;
    if (!gl_host.empty()) {
        CUDA_CHECK(cudaMalloc(&d_gl, sizeof(int) * gl_host.size()));
        CUDA_CHECK(cudaMemcpy(d_gl, gl_host.data(), sizeof(int) * gl_host.size(), cudaMemcpyHostToDevice));
    }
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t) * 16));
    method_kernel<GT, BM, BN, NG, NM, MA, NSM, EZ, 128, 128><<<1, 32>>>(d_gl, d_out, shape_m, shape_n, shape_k);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t o[16];
    CUDA_CHECK(cudaMemcpy(o, d_out, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost));
    if (d_gl) cudaFree(d_gl);
    cudaFree(d_out);

    char buf[256];
    // D) 尾块有效 M：只有 psum 布局 + 关零填充时才是 16，否则恒为 BLOCK_M
    const uint32_t exp_tail = PSUM_TAIL ? 16u : BM;
    snprintf(buf, sizeof buf, "D[%s] tail_eff_m=%u/%u head_eff_m=%u/%u", tag, o[0], exp_tail, o[1], BM);
    check(o[0] == exp_tail && o[1] == BM, buf);
    printf("  %s\n", buf);
    // E) is_computation_valid：Normal/Batched 恒真；masked 看组计数；psum 看 current_psum_m
    uint32_t exp_e2 = 1, exp_e3 = 1;
    if (GT == GemmType::MGroupedMasked) exp_e3 = (2 * BM < (uint32_t) gl_host[0]) ? 1 : 0;       // 300
    else if (GT == GemmType::MGroupedContiguousWithPsumLayout) exp_e3 = (2 * BM < 400u) ? 1 : 0; // 400
    else if (GT == GemmType::MGroupedContiguous)
        exp_e3 = (gl_host[2 * BM] >= 0) ? 1 : 0;   // 源码：grouped_layout[m_offset + m_block*BM] >= 0
    snprintf(buf, sizeof buf, "E[%s] valid(m0)=%u/%u valid(m2)=%u/%u", tag, o[2], exp_e2, o[3], exp_e3);
    check(o[2] == exp_e2 && o[3] == exp_e3, buf);
    printf("  %s\n", buf);
    // F) is_tma_multicast_valid：普通类型折进「num_blocks_in_group != 1」；
    //    MGroupedContiguous 额外要求 peer CTA 同组
    const uint32_t num_m = hceil(shape_m, BM), num_n = hceil(shape_n, BN);
    const uint32_t N1D = sched::get_num_1d_blocks_per_group<GT, BM, BN, NSM, MA>();
    const auto sz0 = host_swizzle(0, num_m, num_n, NM, MA, N1D, true);
    const uint32_t nbig0 = sz0.nbig;
    uint32_t exp_f;
    if (GT == GemmType::MGroupedContiguous) {
        if (MA) exp_f = (nbig0 != 1) ? 1u : 0u;
        else exp_f = (nbig0 != 1 &&
                      gl_host[sz0.m * BM] == gl_host[(sz0.m ^ 1) * BM]) ? 1u : 0u;
    } else {
        exp_f = (nbig0 != 1) ? 1u : 0u;
    }
    snprintf(buf, sizeof buf, "F[%s] tma_mcast_valid=%u/%u (nbig0=%u NM=%u m0=%u)", tag, o[4], exp_f, nbig0, NM, sz0.m);
    check(o[4] == exp_f, buf);
    printf("  %s\n", buf);
    // G: get_global_idx 的规范期望值
    uint32_t exp5, exp6, exp7, exp8, exp9;
    if (GT == GemmType::Normal) {
        exp5 = 2 * BM; exp6 = exp5; exp7 = 128; exp8 = 1; exp9 = 1;
    } else if (GT == GemmType::MGroupedContiguous) {
        const uint32_t off = std::max(0, gl_host[3 * BM]);   // 第 4 参 = 3
        exp5 = 2 * BM; exp6 = off * shape_m + 2 * BM; exp7 = 128; exp8 = 1; exp9 = 1;
    } else if (GT == GemmType::MGroupedMasked or GT == GemmType::MGroupedContiguousWithPsumLayout) {
        exp5 = 2 * BM; exp6 = 0 * shape_m + 2 * BM; exp7 = 128; exp8 = 1; exp9 = 1;
    } else if (is_k_grouped_contiguous(GT)) {
        exp5 = 2 * BM; exp6 = 0 * shape_m + 2 * BM; exp7 = 0 + 128; exp8 = 0 + 1; exp9 = 1;
    } else { // Batched
        exp5 = 2 * BM; exp6 = 2 * BM; exp7 = 128; exp8 = 0 * shape_m + 1; exp9 = 0 * shape_m + 1;
    }
    snprintf(buf, sizeof buf, "G[%s] MN(noff)=%u/%u MN(off)=%u/%u K=%u/%u SFK(off)=%u/%u SFK(noff)=%u/%u",
             tag, o[5], exp5, o[6], exp6, o[7], exp7, o[8], exp8, o[9], exp9);
    check(o[5] == exp5 && o[6] == exp6 && o[7] == exp7 && o[8] == exp8 && o[9] == exp9, buf);
    printf("  %s\n", buf);
}

int main() {
    printf("== DeepGEMM 通用 GEMM 调度器 scheduler/gemm.cuh（H100/sm90 真机）==\n\n");

    printf("== A) get_num_1d_blocks_per_group 枚举启发式 ==\n");
    part_A<128, 128, 132, false, GemmType::Normal>("BM128 BN128 132SM MA0");
    part_A<128, 256, 132, false, GemmType::Normal>("BM128 BN256 132SM MA0");
    part_A<128, 128, 132, true,  GemmType::Normal>("BM128 BN128 132SM MA1");
    part_A<128, 256, 132, true,  GemmType::Normal>("BM128 BN256 132SM MA1");
    part_A<64,  128, 78,  false, GemmType::Normal>("BM64  BN128  78SM MA0");
    part_A<128, 128, 8,   false, GemmType::KGroupedContiguous>("BM128 BN128   8SM MA0");

    printf("\n== B) get_swizzled_block_idx（SM90 multicast 修补路径）==\n");
    // NM=2（2-CTA multicast），SM90 路径
    part_B<GemmType::Normal, 128, 128, 2, false, 132>("Normal   onM NM2 r7c5", 7, 5, true);
    part_B<GemmType::Normal, 128, 128, 2, true,  132>("Normal   onN NM2 r5c7", 7, 5, true);
    part_B<GemmType::Normal, 128, 128, 2, false, 132>("Normal   onM NM2 r8c6", 8, 6, true);
    part_B<GemmType::Normal, 128, 128, 2, true,  132>("Normal   onN NM2 r6c8", 8, 6, true);
    part_B<GemmType::MGroupedContiguous, 128, 128, 2, false, 132>("MGrouped NM2 r13c3", 13, 3, true);
    part_B<GemmType::KGroupedContiguous, 128, 128, 2, false, 132>("KGrouped NM2 r5c13", 5, 13, true);
    part_B<GemmType::Normal, 128, 256, 2, false, 132>("Normal   onM NM2 r9c5 N1D16", 9, 5, true);
    part_B<GemmType::Normal, 128, 256, 2, true,  132>("Normal   onN NM2 r5c9 N1D16", 5, 9, true);
    // NM=1
    part_B<GemmType::Normal, 128, 128, 1, false, 132>("Normal   onM NM1 r9c7", 9, 7, true);
    part_B<GemmType::Normal, 128, 128, 1, true,  132>("Normal   onN NM1 r7c9", 7, 9, true);

    printf("\n== C) get_next_block 任务枚举（覆盖性 = [0,N) 的排列）==\n");
    // Normal: num_mb=ceil(1000/128)=8, num_nb=ceil(768/256)=3 -> 24
    part_C<GemmType::Normal, 128, 256, 1, 1, false, 8>("Normal 8SM", 1000, 768, 512, {}, 8 * 3);
    // Batched: num_blocks=8*3=24, groups=4 -> 96
    part_C<GemmType::Batched, 128, 256, 4, 1, false, 8>("Batched 8SM G4", 1000, 768, 512, {}, 24 * 4);
    // MGroupedContiguous: num_blocks=24, all groups same -> 24
    {
        std::vector<int> gl(1000, 0);
        for (int i = 0; i < 1000; ++i) gl[i] = (i / 256) * 256;   // 组行起点
        part_C<GemmType::MGroupedContiguous, 128, 256, 4, 1, false, 8>("MGroupedContig 8SM", 1000, 768, 512, gl, 8 * 3);
    }
    // MGroupedMasked: 组计数 [300,100,500,100] -> ceil/128 = [3,1,4,1] *3 = 27
    {
        std::vector<int> gl = {300, 100, 500, 100};
        uint32_t N = (hceil(300,128)+hceil(100,128)+hceil(500,128)+hceil(100,128))*3;
        part_C<GemmType::MGroupedMasked, 128, 256, 4, 1, false, 8>("MGroupedMasked 8SM", 1000, 768, 512, gl, N);
    }
    // MGroupedContiguousWithPsumLayout: psum ends [256,300,512,1000]
    {
        std::vector<int> gl = {256, 300, 512, 1000};
        // 组 g 的 m_blocks = ceil((psum[g]-align(psum[g-1],128))/128)，psum[-1]=0
        uint32_t N = 0; uint32_t last = 0;
        for (size_t g = 0; g < gl.size(); ++g) {
            N += hceil((uint32_t) gl[g] - halign(last, 128), 128) * 3;
            last = gl[g];
        }
        part_C<GemmType::MGroupedContiguousWithPsumLayout, 128, 256, 4, 1, false, 8, 128, 128>(
            "MGroupedPsum 8SM", 1000, 768, 512, gl, N);
    }
    // KGroupedContiguous: 每组 K [128,256,512,384] -> 4*24=96
    {
        std::vector<int> gl = {128, 256, 512, 384};
        part_C<GemmType::KGroupedContiguous, 128, 256, 4, 1, false, 8, 128, 128>(
            "KGroupedContig 8SM", 1000, 768, 0, gl, 24 * 4, true);
    }
    // KGroupedContiguousWithPsumLayout: psum K ends [128,512,640,1152]，KA=128
    {
        std::vector<int> gl = {128, 512, 640, 1152};
        part_C<GemmType::KGroupedContiguousWithPsumLayout, 128, 256, 4, 1, false, 8, 128, 128>(
            "KGroupedPsum 8SM KA=128", 1000, 768, 0, gl, 24 * 4, true);
    }
    // KGroupedContiguousWithPsumLayout 带不齐整端点（KA=256）以暴露 align 语义
    {
        std::vector<int> gl = {300, 900, 1100, 2000};
        part_C<GemmType::KGroupedContiguousWithPsumLayout, 128, 256, 4, 1, false, 8, 256, 128>(
            "KGroupedPsum 8SM KA=256", 1000, 768, 0, gl, 24 * 4, true);
    }

    printf("\n== D/E/F/G) 单方法：psum 尾块 / 有效性 / multicast / get_global_idx ==\n");
    part_methods<GemmType::Normal, 128, 128, 1, 1, false, 8, true>("Normal", 1000, 768, 512, {});
    {
        std::vector<int> gl(1000, 0);
        for (int i = 0; i < 1000; ++i) gl[i] = (i / 256) * 256;
        part_methods<GemmType::MGroupedContiguous, 128, 128, 4, 2, false, 8, true>("MGroupedContig NM2 peer-eq", 1000, 768, 512, gl);
    }
    {
        std::vector<int> gl(1000, 0);
        for (int i = 128; i < 1000; ++i) gl[i] = 1;   // 在 128 处跨组 -> peer 不同组
        part_methods<GemmType::MGroupedContiguous, 128, 128, 4, 2, false, 8, true>("MGroupedContig NM2 peer-neq", 1000, 768, 512, gl);
    }
    {
        std::vector<int> gl = {300, 100, 500, 100};
        part_methods<GemmType::MGroupedMasked, 128, 128, 4, 1, false, 8, true>("MGroupedMasked", 1000, 768, 512, gl);
    }
    {
        std::vector<int> gl = {256, 300, 512, 1000};
        part_methods<GemmType::MGroupedContiguousWithPsumLayout, 128, 128, 4, 1, false, 8, false, true>("MGroupedPsum", 1000, 768, 512, gl);
    }
    {
        std::vector<int> gl = {128, 256, 512, 384};
        part_methods<GemmType::Batched, 128, 128, 4, 1, false, 8, true>("Batched", 1000, 768, 512, gl);
    }

    printf("\n== 汇总 ==\n");
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    if (g_fail == 0) printf("ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
