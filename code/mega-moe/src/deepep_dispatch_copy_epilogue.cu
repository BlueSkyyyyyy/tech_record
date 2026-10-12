// P51-1 实验：DeepEP dispatch 的拷贝 epilogue（`dispatch_copy_epilogue_impl`）。
//
// 为什么 H100(sm90) 可跑：
//   * `deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh` 整段**不触碰 NCCL**（无
//     `ncclDevComm`/`NCCLGin`/`ncclTeam`），只用：
//       - `layout::TokenLayout` / `BufferLayout`（`layout/ep/token.cuh`）+ `EPWorkspaceLayout`
//         （`layout/ep/workspace.cuh`）——纯 host/device 结构体；
//       - `ptx::{tma_load_1d,tma_store_1d,tma_store_commit,tma_store_wait,st_bulk,mbarrier_*}`（第 59 篇
//         已逐条在 sm90 验证）+ 轻量 warp 内建（`exchange/gather/deduplicate/reduce_add/warp_*_sum`）；
//       - PDL（`cudaGridDependencySynchronize`，第 52 篇）。
//   因此本文件**直接 include 真实头文件、编译运行真实设备代码**，用真实 token 缓冲布局喂进去，
//   再与独立 host oracle 逐位/集合对拍。PDL 语义在本实验里退化为「无前序 kernel 时的空同步」。
//
//   nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
//     -I~/github/DeepEP/deep_ep/include -I/usr/local/cuda/include/cccl \
//     src/deepep_dispatch_copy_epilogue.cu -o /tmp/deepep_dispatch_copy_epilogue \
//     && /tmp/deepep_dispatch_copy_epilogue

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>

#include <deep_ep/impls/ep/dispatch_copy_epilogue.cuh>

using namespace deep_ep;
using namespace deep_ep::ep;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

// -----------------------------------------------------------------------------
// 固定配置（单 scaleout × 单 scaleup → 1 rank；token 缓冲 8 槽）
// -----------------------------------------------------------------------------
constexpr int kHiddenBytes = 128;
constexpr int kHiddenWords = kHiddenBytes / 4;
constexpr int kNumSFPacks = 4;                 // 16 bytes
constexpr int kNumTopk = 4;
constexpr int kNumExperts = 4;
constexpr int kExpertAlignment = 4;
constexpr int kNumMaxTokensPerRank = 8;
constexpr int kNumScaleup = 1, kNumScaleout = 1;
constexpr int kNumRanks = kNumScaleup * kNumScaleout;
constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
constexpr int kNumSMs = 1, kNumWarps = 4;
constexpr int kNumChannels = 1;
constexpr int kNumThreads = kNumWarps * 32;
constexpr int SF_TOKEN_STRIDE = 8, SF_HIDDEN_STRIDE = 1;
constexpr int kMetadataStride = 2 + kNumTopk;
constexpr int T = 6;

using kernel_t = void(*)(void*, void*, int*, int*, void*, sf_pack_t*, topk_idx_t*, float*, int*, int*, int*, int, int, int, int, int);

template <bool kDoExpand, bool kCachedMode, bool kDoZeroPadding>
static kernel_t get_kernel() {
    return reinterpret_cast<kernel_t>(&dispatch_copy_epilogue_impl<
        kDoExpand, kCachedMode, kDoZeroPadding,
        kNumSMs, kNumChannels, kNumWarps,
        kNumScaleout, kNumScaleup,
        kHiddenBytes, kNumSFPacks,
        kNumMaxTokensPerRank,
        kNumExperts, kNumTopk, kExpertAlignment>);
}

// 每个 token 的 top-k 目标专家（-1 = 无效），构造出带 padding 的每专家计数
static std::vector<std::vector<int>> make_topk() {
    return {
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {0, 1, -1, -1},
        {2, 3, -1, -1},
        {0, -1, -1, -1},
        {3, -1, -1, -1},
    };
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

// -----------------------------------------------------------------------------
// 构造 token 缓冲（模拟主 dispatch kernel 留在 buffer 里的内容）
// -----------------------------------------------------------------------------
struct TokenPayload {
    std::vector<int32_t> hidden;      // kHiddenWords
    std::vector<sf_pack_t> sf;        // kNumSFPacks
    std::vector<int> topk;            // kNumTopk
    std::vector<float> weights;       // kNumTopk
    int src_global_idx;               // 源全局 token id
};

static layout::TokenLayout token_layout() {
    return layout::TokenLayout(kHiddenBytes, kNumSFPacks * (int)sizeof(sf_pack_t), kNumTopk, /*with_metadata=*/true);
}

static void fill_token(const layout::TokenLayout& tok, int i, const std::vector<int>& topk) {
    int32_t* hp = reinterpret_cast<int32_t*>(tok.get_hidden_ptr());
    for (int h = 0; h < kHiddenWords; ++h) hp[h] = i * 1000 + h;
    sf_pack_t* sp = tok.get_sf_ptr();
    for (int k = 0; k < kNumSFPacks; ++k) sp[k].fp32 = 100.0f * i + k;
    int* ti = tok.get_topk_idx_ptr();
    float* tw = tok.get_topk_weights_ptr();
    int* src = tok.get_src_token_global_idx_ptr();
    int* ll = tok.get_linked_list_idx_ptr();
    for (int l = 0; l < kNumTopk; ++l) { ti[l] = topk[l]; tw[l] = 0.5f * i + 0.1f * l; }
    *src = 1000 + i;
    for (int l = 0; l < kNumTopk; ++l) ll[l] = 7000 + i * 10 + l;
}

// -----------------------------------------------------------------------------
// Part 0: TokenLayout / BufferLayout / EPWorkspaceLayout 字节布局（host 核对）
// -----------------------------------------------------------------------------
static void part0_layouts() {
    printf("\n=== Part 0: TokenLayout / BufferLayout / EPWorkspaceLayout 布局（token.cuh / workspace.cuh）===\n");
    const auto tl = token_layout();
    const int64_t n_gmem = tl.get_num_bytes<false>();
    const int64_t n_smem = tl.get_num_bytes<true>();
    printf("  TokenLayout: hidden=%dB sf=%dB topk=%d with_metadata=1\n", kHiddenBytes, kNumSFPacks*4, kNumTopk);
    printf("    get_num_bytes<false>=%lld  get_num_bytes<true>=%lld\n", (long long)n_gmem, (long long)n_smem);

    const auto tl_host = layout::TokenLayout(kHiddenBytes, kNumSFPacks * (int)sizeof(sf_pack_t), kNumTopk, true,
                                             (void*)0x100000);
    auto off = [](const void* p) { return (long long)((const int8_t*)p - (const int8_t*)0x100000); };
    printf("    hidden_off=%lld sf_off=%lld metadata_off=%lld topk_idx_off=%lld weights_off=%lld "
           "src_global_off=%lld linked_list_off=%lld mbarrier_off=%lld(=num_bytes<false>)\n",
           off(tl_host.get_hidden_ptr()), off(tl_host.get_sf_ptr()), off(tl_host.get_metadata_ptr()),
           off(tl_host.get_topk_idx_ptr()), off(tl_host.get_topk_weights_ptr()),
           off(tl_host.get_src_token_global_idx_ptr()), off(tl_host.get_linked_list_idx_ptr()),
           (long long)n_gmem);
    bool ok = off(tl_host.get_hidden_ptr()) == 0
           && off(tl_host.get_sf_ptr()) == align_up(kHiddenBytes, 32)
           && off(tl_host.get_topk_idx_ptr()) == align_up(kHiddenBytes, 32) + align_up(kNumSFPacks*4, 32)
           && off(tl_host.get_topk_weights_ptr()) == off(tl_host.get_topk_idx_ptr()) + kNumTopk*4
           && off(tl_host.get_src_token_global_idx_ptr()) == off(tl_host.get_topk_weights_ptr()) + kNumTopk*4
           && off(tl_host.get_linked_list_idx_ptr()) == off(tl_host.get_src_token_global_idx_ptr()) + 4
           && n_smem == n_gmem + align_up((int)sizeof(ptx::mbarrier), 32);
    check(ok, "TokenLayout 各字段偏移与 <true>/<false> 字节数");

    const auto bl = layout::BufferLayout<false>(tl, kNumScaleup, kNumScaleout * kNumMaxTokensPerRank, (void*)0x200000);
    const auto bl_true = layout::BufferLayout<true>(tl, kNumWarps, 1, (void*)0x300000);
    printf("    BufferLayout<false>: per_token=%lld per_rank=%lld total=%lld\n",
           (long long)bl.get_num_bytes_per_token(), (long long)bl.get_num_bytes_per_rank(), (long long)bl.get_num_bytes());
    bool ok2 = bl.get_num_bytes_per_token() == n_gmem
            && bl.get_num_bytes_per_rank() == n_gmem * kNumMaxTokensPerRank
            && bl.get_num_bytes() == n_gmem * kNumMaxTokensPerRank
            && bl_true.get_num_bytes() == n_smem * kNumWarps;
    check(ok2, "BufferLayout 每 rank / 总量与 smem 视图字节数");

    const auto ws = layout::EPWorkspaceLayout((void*)0x400000, kNumScaleout, kNumScaleup, kNumExperts);
    printf("    EPSignals::get_num_bytes()=%lld  sizeof(EPSignals)=%zu\n",
           (long long)layout::EPWorkspaceLayout::get_num_bytes(), sizeof(layout::EPSignals));
    check(ws.num_experts_per_rank == kNumExpertsPerRank, "EPWorkspaceLayout.num_experts_per_rank");
}

// -----------------------------------------------------------------------------
// 通用 launch 封装
// -----------------------------------------------------------------------------
struct HostInputs {
    std::vector<int8_t> buffer;         // scaleup recv buffer（gmem 端）
    std::vector<int> psum_scaleup;      // 长度 kNumScaleup（累积）
    std::vector<int> psum_expert;       // 长度 kNumExpertsPerRank（exclusive 基址）
    std::vector<int> unaligned_expert;  // 长度 kNumExpertsPerRank（每专家实际 token 数）
    std::vector<int> metadata;          // [T][2+topk]，cached 模式下提供 dst
    std::vector<int> channels;          // linked list 槽
    int num_recv_tokens = T;
    int scaleout_rank_idx = 0, scaleup_rank_idx = 0;
    const layout::EPSignals* signals = nullptr;  // 可选：非零 workspace（linked list 尾部）
};

struct DevBuf {
    void *d_buffer=nullptr, *d_workspace=nullptr, *d_recv_x=nullptr, *d_recv_sf=nullptr;
    int *d_psum_scaleup=nullptr, *d_psum_expert=nullptr, *d_unaligned=nullptr;
    topk_idx_t* d_recv_topk_idx=nullptr; float* d_recv_weights=nullptr;
    int *d_metadata=nullptr, *d_channels=nullptr;
    size_t recv_rows=0;
};

static DevBuf launch(const HostInputs& in, kernel_t kernel, int recv_rows, const std::vector<int>& topk_flat_gmem_src) {
    (void)topk_flat_gmem_src;
    DevBuf d;
    d.recv_rows = recv_rows;
    const auto tl = token_layout();

    // smem 尺寸：kNumWarps 个 token 槽（带 mbarrier）
    const int smem_bytes = kNumWarps * (int)tl.get_num_bytes<true>();
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    CK(cudaMalloc(&d.d_buffer, in.buffer.size()));
    CK(cudaMemcpy(d.d_buffer, in.buffer.data(), in.buffer.size(), cudaMemcpyHostToDevice));

    const int64_t ws_bytes = layout::EPWorkspaceLayout::get_num_bytes();
    CK(cudaMalloc(&d.d_workspace, ws_bytes)); CK(cudaMemset(d.d_workspace, 0, ws_bytes));
    if (in.signals != nullptr)
        CK(cudaMemcpy(d.d_workspace, in.signals, ws_bytes, cudaMemcpyHostToDevice));

    CK(cudaMalloc(&d.d_recv_x, (size_t)recv_rows * kHiddenBytes)); CK(cudaMemset(d.d_recv_x, 0, (size_t)recv_rows*kHiddenBytes));
    CK(cudaMalloc(&d.d_recv_sf, (size_t)recv_rows * SF_TOKEN_STRIDE * sizeof(sf_pack_t)));
    CK(cudaMemset(d.d_recv_sf, 0, (size_t)recv_rows*SF_TOKEN_STRIDE*sizeof(sf_pack_t)));

    CK(cudaMalloc(&d.d_psum_scaleup, in.psum_scaleup.size()*sizeof(int)));
    CK(cudaMemcpy(d.d_psum_scaleup, in.psum_scaleup.data(), in.psum_scaleup.size()*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.d_psum_expert, in.psum_expert.size()*sizeof(int)));
    CK(cudaMemcpy(d.d_psum_expert, in.psum_expert.data(), in.psum_expert.size()*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.d_unaligned, in.unaligned_expert.size()*sizeof(int)));
    CK(cudaMemcpy(d.d_unaligned, in.unaligned_expert.data(), in.unaligned_expert.size()*sizeof(int), cudaMemcpyHostToDevice));

    CK(cudaMalloc(&d.d_recv_topk_idx, (size_t)recv_rows*kNumTopk*sizeof(topk_idx_t)));
    CK(cudaMemset(d.d_recv_topk_idx, 0, (size_t)recv_rows*kNumTopk*sizeof(topk_idx_t)));
    CK(cudaMalloc(&d.d_recv_weights, (size_t)recv_rows*kNumTopk*sizeof(float)));
    CK(cudaMemset(d.d_recv_weights, 0, (size_t)recv_rows*kNumTopk*sizeof(float)));

    CK(cudaMalloc(&d.d_metadata, in.metadata.size()*sizeof(int)));
    CK(cudaMemcpy(d.d_metadata, in.metadata.data(), in.metadata.size()*sizeof(int), cudaMemcpyHostToDevice));

    CK(cudaMalloc(&d.d_channels, std::max<size_t>(1, in.channels.size())*sizeof(int)));
    if (!in.channels.empty()) CK(cudaMemcpy(d.d_channels, in.channels.data(), in.channels.size()*sizeof(int), cudaMemcpyHostToDevice));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(kNumSMs, 1, 1);
    cfg.blockDim = dim3(kNumThreads, 1, 1);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = 0;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;

    CK(cudaLaunchKernelEx(&cfg, kernel,
        d.d_buffer, d.d_workspace,
        d.d_psum_scaleup, d.d_psum_expert,
        d.d_recv_x, reinterpret_cast<sf_pack_t*>(d.d_recv_sf),
        d.d_recv_topk_idx, d.d_recv_weights,
        d.d_metadata, d.d_channels, d.d_unaligned,
        in.num_recv_tokens, SF_TOKEN_STRIDE, SF_HIDDEN_STRIDE,
        in.scaleout_rank_idx, in.scaleup_rank_idx));
    CK(cudaDeviceSynchronize());
    return d;
}

static void free_dev(DevBuf& d) {
    cudaFree(d.d_buffer); cudaFree(d.d_workspace);
    cudaFree(d.d_recv_x); cudaFree(d.d_recv_sf);
    cudaFree(d.d_psum_scaleup); cudaFree(d.d_psum_expert); cudaFree(d.d_unaligned);
    cudaFree(d.d_recv_topk_idx); cudaFree(d.d_recv_weights);
    cudaFree(d.d_metadata); cudaFree(d.d_channels);
}

// 构造标准输入（非 expand）
static HostInputs build_inputs(const std::vector<std::vector<int>>& topk) {
    HostInputs in;
    const auto tl = token_layout();
    const int64_t per_token = tl.get_num_bytes<false>();
    in.buffer.assign((size_t)per_token * kNumScaleup * kNumScaleout * kNumMaxTokensPerRank, 0);
    const auto bl = layout::BufferLayout<false>(tl, kNumScaleup, kNumScaleout * kNumMaxTokensPerRank, in.buffer.data());
    for (int i = 0; i < T; ++i)
        fill_token(bl.get_rank_buffer(0).get_token_buffer(i), i, topk[i]);
    in.psum_scaleup = {T};
    in.psum_expert.assign(kNumExpertsPerRank, 0);
    in.unaligned_expert.assign(kNumExpertsPerRank, 0);
    in.metadata.assign((size_t)T * kMetadataStride, -1);
    in.channels.assign(64, -12345);
    return in;
}

// -----------------------------------------------------------------------------
// Part 1: 非 expand 模式（regular dispatch）
// -----------------------------------------------------------------------------
static void part1_non_expand() {
    printf("\n=== Part 1: 非 expand 模式 dispatch_copy_epilogue_impl（cuh:70-209）===\n");
    auto topk = make_topk();
    HostInputs in = build_inputs(topk);
    DevBuf d = launch(in, get_kernel<false, false, false>(), /*recv_rows=*/T, {});

    std::vector<int32_t> hx((size_t)T * kHiddenWords);
    CK(cudaMemcpy(hx.data(), d.d_recv_x, hx.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::vector<sf_pack_t> hsf((size_t)T * SF_TOKEN_STRIDE);
    CK(cudaMemcpy(hsf.data(), d.d_recv_sf, hsf.size()*sizeof(sf_pack_t), cudaMemcpyDeviceToHost));
    std::vector<topk_idx_t> hidx((size_t)T*kNumTopk);
    CK(cudaMemcpy(hidx.data(), d.d_recv_topk_idx, hidx.size()*sizeof(topk_idx_t), cudaMemcpyDeviceToHost));
    std::vector<float> hw((size_t)T*kNumTopk);
    CK(cudaMemcpy(hw.data(), d.d_recv_weights, hw.size()*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<int> hmeta((size_t)T*kMetadataStride);
    CK(cudaMemcpy(hmeta.data(), d.d_metadata, hmeta.size()*sizeof(int), cudaMemcpyDeviceToHost));

    int hidden_mism = 0, sf_mism = 0, idx_mism = 0, w_mism = 0, meta_mism = 0;
    for (int i = 0; i < T; ++i) {
        for (int h = 0; h < kHiddenWords; ++h) hidden_mism += (hx[i*kHiddenWords+h] != i*1000+h);
        for (int k = 0; k < kNumSFPacks; ++k) sf_mism += (hsf[i*SF_TOKEN_STRIDE + k*SF_HIDDEN_STRIDE].fp32 != 100.0f*i + k);
        for (int l = 0; l < kNumTopk; ++l) {
            idx_mism += (hidx[i*kNumTopk+l] != topk[i][l]);
            w_mism += (hw[i*kNumTopk+l] != 0.5f*i + 0.1f*l);
        }
        // 单 rank → expert 全在范围内，(i*0) 的 [0]=src global, [1]=current_rank*topk+master
        int master = -1; for (int l = 0; l < kNumTopk; ++l) if (topk[i][l] >= 0) master = l;
        meta_mism += (hmeta[i*kMetadataStride+0] != 1000+i);
        meta_mism += (hmeta[i*kMetadataStride+1] != 0*kNumTopk + master);
    }
    printf("  hidden_mism=%d/%d  sf_mism=%d/%d  topk_idx_mism=%d  weights_mism=%d  metadata_mism=%d\n",
           hidden_mism, T*kHiddenWords, sf_mism, T*kNumSFPacks, idx_mism, w_mism, meta_mism);
    check(hidden_mism==0, "recv_x 行 i = token i 的 hidden（TMA 载入→TMA 存出）");
    check(sf_mism==0, "recv_sf[i*token_stride + k*hidden_stride] = SF（含两种 stride）");
    check(idx_mism==0, "recv_topk_idx[i, l] = 本地专家号（expert - expert_start）");
    check(w_mism==0, "recv_topk_weights[i, l] = 原始 top-k 权重");
    check(meta_mism==0, "recv_src_metadata[i,0]=源 token id、[i,1]=rank*topk+master_lane");

    // 主循环写了 metadata[0],[1]；[2+l] 只在 expand 写 → 非 expand 保持 -1
    int untouched = 0; for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l)
        untouched += (hmeta[i*kMetadataStride+2+l] == -1);
    printf("  metadata[2+l] 未被触碰: %d/%d\n", untouched, T*kNumTopk);
    check(untouched == T*kNumTopk, "非 expand 不写 metadata[2+l]（留待 expand 用）");

    free_dev(d);
}

// -----------------------------------------------------------------------------
// Part 2: expand 模式（atomicAdd 分配 + 零填充）
// -----------------------------------------------------------------------------
static void part2_expand_zero_pad() {
    printf("\n=== Part 2: expand + zero-padding（cuh:113-124, 232-325）===\n");
    auto topk = make_topk();
    HostInputs in = build_inputs(topk);

    // 每专家计数 → exclusive 基址（按对齐后的块布局）
    std::vector<int> counts(kNumExpertsPerRank, 0);
    for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l)
        if (topk[i][l] >= 0) counts[topk[i][l]]++;
    std::vector<int> base(kNumExpertsPerRank, 0), aligned(kNumExpertsPerRank, 0);
    int acc = 0;
    for (int e = 0; e < kNumExpertsPerRank; ++e) { base[e] = acc; aligned[e] = align_up(counts[e], kExpertAlignment); acc += aligned[e]; }
    const int recv_rows = acc;
    in.psum_expert = base;
    in.unaligned_expert = counts;

    printf("  counts=[%d,%d,%d,%d] aligned=[%d,%d,%d,%d] base=[%d,%d,%d,%d] rows=%d\n",
           counts[0],counts[1],counts[2],counts[3], aligned[0],aligned[1],aligned[2],aligned[3],
           base[0],base[1],base[2],base[3], recv_rows);

    DevBuf d = launch(in, get_kernel<true, false, true>(), recv_rows, {});

    std::vector<int32_t> hx((size_t)recv_rows*kHiddenWords, -7);
    CK(cudaMemcpy(hx.data(), d.d_recv_x, hx.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::vector<sf_pack_t> hsf((size_t)recv_rows*SF_TOKEN_STRIDE);
    CK(cudaMemcpy(hsf.data(), d.d_recv_sf, hsf.size()*sizeof(sf_pack_t), cudaMemcpyDeviceToHost));
    std::vector<float> hw((size_t)recv_rows*kNumTopk, -7.f);
    CK(cudaMemcpy(hw.data(), d.d_recv_weights, hw.size()*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<int> hmeta((size_t)T*kMetadataStride);
    CK(cudaMemcpy(hmeta.data(), d.d_metadata, hmeta.size()*sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> hpsum(kNumExpertsPerRank);
    CK(cudaMemcpy(hpsum.data(), d.d_psum_expert, hpsum.size()*sizeof(int), cudaMemcpyDeviceToHost));

    // (a) 每专家区内的 token 行集合 = {(i,l) | topk[i][l]==e}
    int region_bad = 0;
    std::set<std::pair<int,int>> all_expected;
    for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l) if (topk[i][l]>=0) all_expected.insert({topk[i][l], i});
    for (int e = 0; e < kNumExpertsPerRank; ++e) {
        std::map<int,int> got;  // token -> 出现次数（区内）
        for (int r = base[e]; r < base[e]+counts[e]; ++r) {
            int tok = hx[r*kHiddenWords+0] / 1000;
            if (hx[r*kHiddenWords+0] % 1000 != 0) { region_bad++; continue; }
            got[tok]++;
        }
        std::map<int,int> exp;
        for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l) if (topk[i][l]==e) exp[i]++;
        // 只保留有该专家的 token
        for (auto& kv : exp) if (kv.second == 0) exp.erase(kv.first);
        if (got != exp) { region_bad++; printf("    expert %d 区 token 集合不符 (got=%zu exp=%zu)\n", e, got.size(), exp.size()); }
    }
    check(region_bad==0, "expand: 每个专家对齐区内包含的 token 集合正确（atomicAdd 分配）");
    bool psum_ok = true;
    for (int e = 0; e < kNumExpertsPerRank; ++e) psum_ok &= (hpsum[e] == base[e] + counts[e]);
    check(psum_ok, "expand: psum 计数器被 atomicAdd 到 base+count");

    // (b) 每专家区内 token 行 hidden = token 的 hidden；padding 行全零
    int fill_bad = 0, pad_bad = 0, sf_pad_bad = 0, w_pad_bad = 0;
    for (int e = 0; e < kNumExpertsPerRank; ++e) {
        for (int r = base[e]; r < base[e]+counts[e]; ++r) {
            int tok = hx[r*kHiddenWords+0]/1000;
            for (int h = 0; h < kHiddenWords; ++h) fill_bad += (hx[r*kHiddenWords+h] != tok*1000+h);
        }
        for (int r = base[e]+counts[e]; r < base[e]+aligned[e]; ++r) {
            for (int h = 0; h < kHiddenWords; ++h) pad_bad += (hx[r*kHiddenWords+h] != 0);
            for (int k = 0; k < kNumSFPacks; ++k) sf_pad_bad += (hsf[r*SF_TOKEN_STRIDE+k*SF_HIDDEN_STRIDE].fp32 != 0.f);
            w_pad_bad += (hw[r*kNumTopk] != 0.f);
        }
    }
    printf("  fill_bad=%d (token 行 hidden), pad_zero_bad=%d (padding hidden), sf_pad_bad=%d, w_pad_bad=%d\n",
           fill_bad, pad_bad, sf_pad_bad, w_pad_bad);
    check(fill_bad==0, "expand: token 行 hidden 正确");
    check(pad_bad==0, "expand: 对齐 padding 行 hidden 被 st_bulk+TMA 清零");
    check(sf_pad_bad==0, "expand: padding 行 SF 清零");
    check(w_pad_bad==0, "expand: padding 行权重清零");

    // (c) metadata[2+l] 记录 dst_tensor_idx
    int meta_bad = 0;
    for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l) {
        int dst = hmeta[i*kMetadataStride+2+l];
        int e = topk[i][l];
        if (e < 0) { meta_bad += (dst != -1); continue; }
        meta_bad += !(dst >= base[e] && dst < base[e]+counts[e]);
    }
    check(meta_bad==0, "expand: recv_src_metadata[i,2+l] = 该 (i,l) 的 dst_tensor_idx（在专家区内）");

    free_dev(d);
}

// -----------------------------------------------------------------------------
// Part 3: expand + cached 模式（从 metadata 读 dst，确定性）
// -----------------------------------------------------------------------------
static void part3_cached() {
    printf("\n=== Part 3: expand + cached 模式（cuh:118-120, 189-208）===\n");
    auto topk = make_topk();
    HostInputs in = build_inputs(topk);

    std::vector<int> counts(kNumExpertsPerRank, 0);
    for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l) if (topk[i][l]>=0) counts[topk[i][l]]++;
    std::vector<int> base(kNumExpertsPerRank, 0), aligned(kNumExpertsPerRank,0);
    int acc = 0; for (int e = 0; e < kNumExpertsPerRank; ++e){ base[e]=acc; aligned[e]=align_up(counts[e],kExpertAlignment); acc+=aligned[e]; }
    const int recv_rows = acc;

    // cached 模式不读 psum；这里仍填上 base 以坐实「kernel 不写 psum」
    in.psum_expert = base;

    // 预计算确定性 dst（按 i,l 稳定顺序填入每专家区）
    std::vector<int> next = base;
    in.metadata.assign((size_t)T*kMetadataStride, -1);
    std::vector<int> dst_map((size_t)T*kNumTopk, -1);
    for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l) {
        int e = topk[i][l]; if (e < 0) continue;
        int dst = next[e]++;
        dst_map[i*kNumTopk+l] = dst;
        in.metadata[i*kMetadataStride+2+l] = dst;
        in.metadata[i*kMetadataStride+0] = 1000+i;
    }

    DevBuf d = launch(in, get_kernel<true, true, false>(), recv_rows, {});

    std::vector<int32_t> hx((size_t)recv_rows*kHiddenWords, -7);
    CK(cudaMemcpy(hx.data(), d.d_recv_x, hx.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::vector<int> hpsum(kNumExpertsPerRank);
    CK(cudaMemcpy(hpsum.data(), d.d_psum_expert, hpsum.size()*sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < T; ++i) for (int l = 0; l < kNumTopk; ++l) {
        int dst = dst_map[i*kNumTopk+l]; if (dst < 0) continue;
        for (int h = 0; h < kHiddenWords; ++h) bad += (hx[dst*kHiddenWords+h] != i*1000+h);
    }
    check(bad==0, "cached: 输出严格写到 metadata 指定的 dst 行");
    check(hpsum==base, "cached: psum 计数器保持不变（不 atomicAdd）");

    free_dev(d);
}

// -----------------------------------------------------------------------------
// Part 4: linked list（scaleout=2，T=1 使写序确定；cuh:131-136, 211-230）
// -----------------------------------------------------------------------------
static void part4_linked_list() {
    printf("\n=== Part 4: linked list 创建与尾部清理（scaleout=2，cuh:132-136 / 211-230）===\n");
    constexpr int kLLScaleout = 2, kLLScaleup = 1, kLLWarps = 4, kLLSMs = 1, kLLChannels = 1;
    constexpr int kLLExperts = 4;   // 每 rank 2 个专家
    using ll_kernel_t = kernel_t;
    auto kernel = reinterpret_cast<ll_kernel_t>(&dispatch_copy_epilogue_impl<
        false, false, false, kLLSMs, kLLChannels, kLLWarps,
        kLLScaleout, kLLScaleup, kHiddenBytes, kNumSFPacks,
        kNumMaxTokensPerRank, kLLExperts, kNumTopk, kExpertAlignment>);

    const auto tl = token_layout();
    const int64_t per_token = tl.get_num_bytes<false>();
    const int64_t slot_idx = 10;   // token 0 的 master lane 指向的链槽
    const int64_t tail_slot = 5;   // 该 channel 的尾部指针指向的槽

    HostInputs in;
    in.buffer.assign((size_t)per_token * kLLScaleup * kLLScaleout * kNumMaxTokensPerRank, 0);
    const auto bl = layout::BufferLayout<false>(tl, kLLScaleup, kLLScaleout*kNumMaxTokensPerRank, in.buffer.data());
    // 只有 token 0 有效；topk={0,1,-1,-1} → rank 0 的专家 [0,1]，master lane = 1
    {
        auto tok = bl.get_rank_buffer(0).get_token_buffer(0);
        fill_token(tok, 0, {0, 1, -1, -1});
        int* ll = tok.get_linked_list_idx_ptr();
        ll[1] = (int)slot_idx;   // master lane 的链槽
    }
    in.num_recv_tokens = 1;
    in.psum_scaleup = {1};
    in.psum_expert.assign(2, 0);
    in.unaligned_expert.assign(2, 0);
    in.metadata.assign((size_t)1*kMetadataStride, -1);
    in.channels.assign(64, -12345);
    in.scaleout_rank_idx = 0; in.scaleup_rank_idx = 0;

    // 非零 workspace：channel_scaleup_tail[0][0] = tail_slot
    layout::EPSignals sig; memset(&sig, 0, sizeof(sig));
    sig.channel_scaleup_tail[0][0] = (int)tail_slot;
    in.signals = &sig;

    DevBuf d = launch(in, kernel, /*recv_rows=*/1, {});

    std::vector<int> hch(64);
    CK(cudaMemcpy(hch.data(), d.d_channels, hch.size()*sizeof(int), cudaMemcpyDeviceToHost));
    layout::EPSignals sig_out; CK(cudaMemcpy(&sig_out, d.d_workspace, sizeof(sig_out), cudaMemcpyDeviceToHost));

    printf("  channels[slot=%lld]=%d (期望 0=token index)  channels[tail=%lld]=%d (期望 -1)  tail 清零后=%d\n",
           (long long)slot_idx, hch[slot_idx], (long long)tail_slot, hch[tail_slot], sig_out.channel_scaleup_tail[0][0]);
    check(hch[slot_idx] == 0, "linked list: channel_linked_list[linked_list_idx[master]] = i");
    check(hch[tail_slot] == -1, "linked list: 尾部槽写入 -1（链尾哨兵）");
    check(sig_out.channel_scaleup_tail[0][0] == 0, "linked list: channel_scaleup_tail 被清零（供 combine 复用）");

    free_dev(d);
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d), CUDA %d\n", prop.name, prop.major, prop.minor, CUDART_VERSION);
    printf("sizeof(topk_idx_t)=%zu  kNumTMAAlignmentBytes=%d  kNumRDMAAlignmentBytes=%d  kNumAlignedSFPacks=%d\n",
           sizeof(topk_idx_t), kNumTMAAlignmentBytes, kNumRDMAAlignmentBytes, kNumAlignedSFPacks);

    part0_layouts();
    part1_non_expand();
    part2_expand_zero_pad();
    part3_cached();
    part4_linked_list();

    printf("\n================ 汇总: %d PASS / %d FAIL ================\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
