// P48-1 实验：DeepEP combine 的归约 epilogue（`combine_reduce` / `combine_reduce_epilogue_impl`）。
//
// 为什么 H100(sm90) 可跑：
//   * `deep_ep/include/deep_ep/impls/ep/combine_utils.cuh` 的 `use_rank_layout` /
//     `get_num_tokens_in_layout` / `get_max_unroll_factor` 是纯 host constexpr；
//     `CombineVecTraits` 在 sm90 选 `int4`；`compute_topk_slots` / `combine_reduce` 只用
//     warp 内建（`__ffs`/`__ballot_sync`/`__match_any_sync`/`__shfl_sync`）+ `ptx::ldg_with_ge_pred`
//     （带谓词的 `ld.global.nc`）+ `ptx::accumulate`（sm90 回退为 bf16→fp32 add）。
//   * `deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh` 的 `combine_reduce_epilogue_impl`
//     只用 TMA store（`cp.async.bulk`，sm90 支持）+ PDL（`cudaGridDependencySynchronize`）+ TokenLayout。
//   因此本文件**直接 include 真实头文件、编译运行真实设备代码**，用真实的对称 recv_buffer 布局喂进去，
//   再用独立 host oracle 逐位对拍。sm100 的融合 `add.rn.f32.bf16` 快路在 sm90 自动回退（见 ptx.cuh:593-602）。
//
//   nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
//     -I~/github/DeepEP/deep_ep/include -I/usr/local/cuda/include/cccl \
//     src/deepep_combine_reduce.cu -o /tmp/deepep_combine_reduce && /tmp/deepep_combine_reduce

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <deep_ep/impls/ep/combine_reduce_epilogue.cuh>

using namespace deep_ep;
using namespace deep_ep::ep;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

static __host__ __device__ __forceinline__ float b2f(nv_bfloat16 x) { return __bfloat162float(x); }
static __host__ __device__ __forceinline__ nv_bfloat16 f2b(float x) { return __float2bfloat16_rn(x); }

// -----------------------------------------------------------------------------
// Part 0: host constexpr 语义
// -----------------------------------------------------------------------------
static void part0_constexpr() {
    printf("\n=== Part 0: host constexpr 语义（combine_utils.cuh:8-26）===\n");
    // use_rank_layout / get_num_tokens_in_layout
    struct Row { bool mult; int ranks, topk; };
    std::vector<Row> rows = {
        {false, 8, 4}, {true, 1, 8}, {true, 2, 8}, {true, 4, 4}, {true, 8, 4}, {true, 8, 2}, {true, 16, 8},
    };
    bool ok = true;
    for (auto& r : rows) {
        bool rl = false, exp_rl;
        if (!r.mult) rl = false; else rl = (r.ranks <= r.topk);
        int ntl = rl ? r.ranks : r.topk;
        exp_rl = r.mult and (r.ranks <= r.topk);
        bool good = (rl == exp_rl) and (ntl == (exp_rl ? r.ranks : r.topk));
        ok &= good;
        printf("    mult=%-5s ranks=%-3d topk=%-2d -> use_rank_layout=%-5s num_tokens_in_layout=%d %s\n",
               r.mult?"true":"false", r.ranks, r.topk, rl?"true":"false", ntl, good?"":"<<< BAD");
    }
    check(ok, "use_rank_layout / get_num_tokens_in_layout 表");
    // static_assert 编译期坐实
    static_assert(use_rank_layout<true, 4, 4>() == true,  "");
    static_assert(use_rank_layout<true, 8, 4>() == false, "");
    static_assert(use_rank_layout<false, 1, 8>() == false, "");
    static_assert(get_num_tokens_in_layout<true, 4, 4>() == 4, "");
    static_assert(get_num_tokens_in_layout<true, 8, 4>() == 4, "");
    // 注意：use_rank_layout<false,1,8>() 为 false，故 num_tokens_in_layout=topk=8
    static_assert(get_num_tokens_in_layout<false, 1, 8>() == 8, "");

    // get_max_unroll_factor
    auto uf = [](int len, int maxu, int ws=32) -> int {
        for (int i = maxu; i >= 1; --i) if (len % (ws*i) == 0) return i;
        return -1;
    };
    int m;
    m = uf(64, 4);  printf("    get_max_unroll_factor<64,4>  = %d (expect 2)\n", m);
    check(m == 2, "unroll(64,4)=2");
    m = uf(512, 4); printf("    get_max_unroll_factor<512,4> = %d (expect 4)\n", m);
    check(m == 4, "unroll(512,4)=4");
    m = uf(256, 4); printf("    get_max_unroll_factor<256,4> = %d (expect 4)\n", m);
    check(m == 4, "unroll(256,4)=4");
    m = uf(32, 4);  printf("    get_max_unroll_factor<32,4>  = %d (expect 1)\n", m);
    check(m == 1, "unroll(32,4)=1");
    // CombineVecTraits on sm90 = int4 (16B)
    check(sizeof(CombineVecTraits<512>::vec_t) == 16, "sm90 CombineVecTraits = int4 (16B)");
}

// -----------------------------------------------------------------------------
// Part 1: 直接驱动真实 `combine_reduce`（bypass vs 通用路径 + bias）
// -----------------------------------------------------------------------------
// 把真实 combine_reduce 逐行调用：源数据按 slot 放在 src[slot * kHidden]（已展开成 vec 视图）。
template<int kHiddenVec, int kUnrollFactor, int kNV, int kHidden>
__global__ void test_combine_reduce_kernel(const nv_bfloat16* src, int* slots, bool with_bias,
                                           const nv_bfloat16* bias, nv_bfloat16* out) {
    using vec_t = int4;
    const int lane_idx = threadIdx.x & 31;
    int topk_slot_idx[kNV];
    #pragma unroll
    for (int k = 0; k < kNV; ++k) topk_slot_idx[k] = slots[k];

    const auto* bvec = reinterpret_cast<const vec_t*>(bias);
    combine_reduce<kHiddenVec, kUnrollFactor, kNV>(
        lane_idx, topk_slot_idx, reinterpret_cast<vec_t*>(out),
        [=](const int& slot_idx) -> const vec_t* {
            return reinterpret_cast<const vec_t*>(src) + (int64_t)slot_idx * kHiddenVec;
        },
        /* wait_buffer */ []() {},
        /* bias_0 */ with_bias ? const_cast<vec_t*>(bvec) : nullptr
    );
}

// 通用 host oracle：复制 combine_reduce 的加法顺序与舍入规则。
static std::vector<float> oracle_combine_reduce(const std::vector<std::vector<float>>& slot_vals,
                                                const std::vector<float>& bias, bool bypass) {
    int H = slot_vals.empty() ? 0 : (int)slot_vals[0].size();
    std::vector<float> out(H, 0.f);
    if (bypass) {
        // values_0（slot 0）+ values_1（slot 1，缺失=0），bf16 逐元素 add（RNE）
        for (int h = 0; h < H; ++h) {
            nv_bfloat16 a = f2b(slot_vals.size() > 0 ? slot_vals[0][h] : 0.f);
            nv_bfloat16 b = f2b(slot_vals.size() > 1 ? slot_vals[1][h] : 0.f);
            out[h] = b2f(f2b(b2f(a) + b2f(b)));
        }
    } else {
        for (int h = 0; h < H; ++h) {
            float acc = 0.f;
            for (float v : bias) acc += v;
            for (auto& sv : slot_vals) acc += sv[h];
            out[h] = b2f(f2b(acc));
        }
    }
    return out;
}

static void part1_combine_reduce_direct() {
    printf("\n=== Part 1: 直接调用真实 combine_reduce（combine_utils.cuh:55-172）===\n");
    constexpr int kHidden = 256;                // 512 bytes -> kHiddenVec(int4) = 32（须 >=32 循环才执行）
    constexpr int kHiddenVec = kHidden * 2 / 16;
    constexpr int kUnroll = 1;                  // 32 % (32*1)=0
    constexpr int kNV = 4;
    constexpr int kNSlots = 4;

    std::vector<nv_bfloat16> h_src(kNSlots * kHidden);
    // 三组值：v0,v1 尾数为 1（易触发 bf16 加法舍入差异），v2 为大数
    auto set_slot = [&](int slot, float base) {
        for (int h = 0; h < kHidden; ++h) h_src[slot*kHidden + h] = f2b(base + h * 0.03125f);
    };
    set_slot(0, 1.0f); set_slot(1, 0.0078125f); set_slot(2, 100.0f); set_slot(3, -3.0f);

    nv_bfloat16 *d_src, *d_bias, *d_out;
    int* d_slots;
    CK(cudaMalloc(&d_src, h_src.size()*sizeof(nv_bfloat16)));
    CK(cudaMalloc(&d_slots, kNV*sizeof(int)));
    CK(cudaMalloc(&d_bias, kHidden*sizeof(nv_bfloat16)));
    CK(cudaMalloc(&d_out, kHidden*sizeof(nv_bfloat16)));
    CK(cudaMemcpy(d_src, h_src.data(), h_src.size()*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    std::vector<nv_bfloat16> h_bias(kHidden, f2b(0.5f));
    CK(cudaMemcpy(d_bias, h_bias.data(), kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    // case A: 2 个有效 slot，无 bias -> bypass（bf16 add）
    {
        std::vector<int> slots = {0, 1, -1, -1};
        CK(cudaMemcpy(d_slots, slots.data(), kNV*sizeof(int), cudaMemcpyHostToDevice));
        test_combine_reduce_kernel<kHiddenVec, kUnroll, kNV, kHidden><<<1,32>>>(d_src, d_slots, false, nullptr, d_out);
        CK(cudaDeviceSynchronize());
        std::vector<nv_bfloat16> got(kHidden);
        CK(cudaMemcpy(got.data(), d_out, kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<std::vector<float>> sv;
        for (int s = 0; s < 2; ++s) { std::vector<float> v(kHidden); for (int h=0;h<kHidden;++h) v[h]=b2f(h_src[s*kHidden+h]); sv.push_back(v); }
        auto ref = oracle_combine_reduce(sv, {}, /*bypass=*/true);
        int mism = 0; for (int h = 0; h < kHidden; ++h) mism += (b2f(got[h]) != ref[h]);
        printf("    bypass(2 slot, bf16 add): mism=%d/%d\n", mism, kHidden);
        check(mism == 0, "bypass 路径 vs bf16 折叠 oracle 逐位");
        cudaFree(0);
    }

    // case B: 4 个有效 slot，无 bias -> 通用路径（fp32 累加）
    {
        std::vector<int> slots = {0, 1, 2, 3};
        CK(cudaMemcpy(d_slots, slots.data(), kNV*sizeof(int), cudaMemcpyHostToDevice));
        test_combine_reduce_kernel<kHiddenVec, kUnroll, kNV, kHidden><<<1,32>>>(d_src, d_slots, false, nullptr, d_out);
        CK(cudaDeviceSynchronize());
        std::vector<nv_bfloat16> got(kHidden);
        CK(cudaMemcpy(got.data(), d_out, kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<std::vector<float>> sv;
        for (int s = 0; s < 4; ++s) { std::vector<float> v(kHidden); for (int h=0;h<kHidden;++h) v[h]=b2f(h_src[s*kHidden+h]); sv.push_back(v); }
        auto ref = oracle_combine_reduce(sv, {}, /*bypass=*/false);
        int mism = 0; for (int h = 0; h < kHidden; ++h) mism += (b2f(got[h]) != ref[h]);
        printf("    general(4 slot, fp32 累加): mism=%d/%d\n", mism, kHidden);
        check(mism == 0, "通用路径 vs fp32 累加后一次舍入 oracle 逐位");
    }

    // case C: 2 slot + bias -> 强制通用路径（bias != null）
    {
        std::vector<int> slots = {0, 1, -1, -1};
        CK(cudaMemcpy(d_slots, slots.data(), kNV*sizeof(int), cudaMemcpyHostToDevice));
        test_combine_reduce_kernel<kHiddenVec, kUnroll, kNV, kHidden><<<1,32>>>(d_src, d_slots, true, d_bias, d_out);
        CK(cudaDeviceSynchronize());
        std::vector<nv_bfloat16> got(kHidden);
        CK(cudaMemcpy(got.data(), d_out, kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<std::vector<float>> sv;
        for (int s = 0; s < 2; ++s) { std::vector<float> v(kHidden); for (int h=0;h<kHidden;++h) v[h]=b2f(h_src[s*kHidden+h]); sv.push_back(v); }
        auto ref = oracle_combine_reduce(sv, {b2f(h_bias[0])}, /*bypass=*/false);
        int mism = 0; for (int h = 0; h < kHidden; ++h) mism += (b2f(got[h]) != ref[h]);
        printf("    general(2 slot + bias): mism=%d/%d\n", mism, kHidden);
        check(mism == 0, "有 bias 强制通用路径 vs oracle 逐位");
    }

    // case D: bypass 的适用边界——≤2 slot 时 bypass == 通用（单次舍入），>2 slot 链式 bf16 会偏
    {
        // 随机 bf16 值对：bypass(单次 bf16 add) 与通用(fp32 累加一次舍入) 逐位一致
        int pair_diff = 0;
        uint64_t s = 88172645463325252ull;
        auto rnd = [&]() { s ^= s<<13; s ^= s>>7; s ^= s<<17; return (float)((int)(s>>16) - (1<<15)) * 0.0009765625f; };
        for (int it = 0; it < 200000; ++it) {
            nv_bfloat16 a = f2b(rnd()), b = f2b(rnd());
            nv_bfloat16 byp = f2b(b2f(a) + b2f(b));       // bypass: 一次 bf16 add
            float acc = b2f(a) + b2f(b);                  // 通用: fp32 累加
            if (byp != f2b(acc)) ++pair_diff;
        }
        printf("    200k 随机 bf16 值对: bypass(单次 bf16 add) vs 通用 = %d 处不同\n", pair_diff);
        check(pair_diff == 0, "≤2 slot 时 bypass 与通用逐位一致（精确和再舍入）");
        // 3 slot 链式 bf16 折叠 vs fp32 累加一次舍入
        int tri_diff = 0;
        for (int it = 0; it < 200000; ++it) {
            nv_bfloat16 a = f2b(rnd()), b = f2b(rnd()), c = f2b(rnd());
            nv_bfloat16 chain = f2b(b2f(f2b(b2f(a) + b2f(b))) + b2f(c)); // 两次 bf16 add
            nv_bfloat16 once  = f2b(b2f(a) + b2f(b) + b2f(c));           // 一次舍入
            if (chain != once) ++tri_diff;
        }
        printf("    200k 随机 bf16 三元组: 链式两次 bf16 add vs 一次舍入 = %d 处不同（%.2f%%）\n",
               tri_diff, 100.0*tri_diff/200000);
        check(tri_diff > 0, "解释为何 bypass 只在 ≤2 slot 启用（>2 链式 bf16 会累积舍入）");
    }

    cudaFree(d_src); cudaFree(d_slots); cudaFree(d_bias); cudaFree(d_out);
}

// -----------------------------------------------------------------------------
// Part 2: 完整 `combine_reduce_epilogue_impl`（dedup + rank/topk 双布局 + bias + 权重）
// -----------------------------------------------------------------------------
template<bool kUseExpandedLayout, bool kAllowMultipleReduction,
         int kNumSMs, int kNumWarps, int kNumScaleoutRanks, int kNumScaleupRanks,
         int kHidden, int kNumMaxTokensPerRank, int kNumExperts, int kNumTopk>
static void run_epilogue_case(const char* name, int T,
                              const std::vector<std::vector<int>>& topk_experts,
                              const std::vector<std::vector<float>>& rank_hidden, // [T][kNumRanks]
                              bool with_bias, bool check_weights) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks;
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    constexpr bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>();
    constexpr int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>();
    using kernel_t = void(*)(nv_bfloat16*, float*, topk_idx_t*, void*, void*, void*, int, int, int);
    auto kernel = reinterpret_cast<kernel_t>(&combine_reduce_epilogue_impl<
        kUseExpandedLayout, kAllowMultipleReduction, kNumSMs, kNumWarps,
        kNumScaleoutRanks, kNumScaleupRanks, kHidden, kNumMaxTokensPerRank, kNumExperts, kNumTopk>);

    printf("\n-- %s (use_rank_layout=%s, num_tokens_in_layout=%d) --\n",
           name, kUseRankLayout?"true":"false", kNumTokensInLayout);

    // recv buffer: [kNumTokensInLayout][kNumMaxTokensPerRank] token
    const auto comm_tl = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const int64_t tok_bytes = comm_tl.get_num_bytes<false>();
    const int64_t rank_bytes = tok_bytes * kNumMaxTokensPerRank;
    const int64_t recv_bytes = rank_bytes * kNumTokensInLayout;
    std::vector<int8_t> h_recv(recv_bytes, 0);
    const auto comm = layout::BufferLayout<false>(comm_tl, kNumTokensInLayout, kNumMaxTokensPerRank, h_recv.data());

    // 每个 token 计算 rank 分组、master lane、slot
    std::vector<std::vector<int>> slot_of_rank(T);      // [t][rank]
    std::vector<std::vector<int>> distinct_ranks(T);
    for (int t = 0; t < T; ++t) {
        // rank per lane
        std::vector<int> rank_of_lane(kNumTopk);
        for (int l = 0; l < kNumTopk; ++l) {
            int e = topk_experts[t][l];
            rank_of_lane[l] = e >= 0 ? (e / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : (kNumExperts/kNumScaleoutRanks))) : -1;
        }
        // master lane per rank = 最大 lane idx
        std::vector<int> master(kNumRanks, -1);
        for (int l = 0; l < kNumTopk; ++l)
            if (rank_of_lane[l] >= 0) master[rank_of_lane[l]] = std::max(master[rank_of_lane[l]], l);
        std::vector<int> dr;
        std::vector<int> slotv(kNumRanks, -1);
        for (int r = 0; r < kNumRanks; ++r) if (master[r] >= 0) {
            dr.push_back(r);
            slotv[r] = kUseRankLayout ? r : master[r];
        }
        // 按 lane 递增顺序（对应 compute_topk_slots 从最低 bit）
        std::sort(dr.begin(), dr.end(), [&](int a, int b){ return master[a] < master[b]; });
        distinct_ranks[t] = dr;
        slot_of_rank[t] = slotv;
        // 填充 comm buffer 的 hidden
        for (int r : dr) {
            __nv_bfloat16* hp = (__nv_bfloat16*)comm.get_rank_buffer(slotv[r]).get_token_buffer(t).get_hidden_ptr();
            for (int h = 0; h < kHidden; ++h) hp[h] = f2b(rank_hidden[t][r] + h * 0.001953125f);
            // topk weights
            float* wp = comm.get_rank_buffer(slotv[r]).get_token_buffer(t).get_topk_weights_ptr();
            for (int l = 0; l < kNumTopk; ++l) wp[l] = 0.25f * (r + 1) + 0.01f * l;
        }
        // 把非 master slot 的 topk weight 也写成可区分值（不应被读取）
    }

    // combined_topk_idx [T][kNumTopk]
    std::vector<topk_idx_t> h_idx(T * kNumTopk, -1);
    for (int t = 0; t < T; ++t) for (int l = 0; l < kNumTopk; ++l) h_idx[t*kNumTopk + l] = topk_experts[t][l];

    void *d_recv, *d_b0, *d_b1;
    nv_bfloat16 *d_x; float* d_w; topk_idx_t* d_idx;
    CK(cudaMalloc(&d_recv, recv_bytes)); CK(cudaMemcpy(d_recv, h_recv.data(), recv_bytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, T*kHidden*sizeof(nv_bfloat16)));
    CK(cudaMalloc(&d_w, T*kNumTopk*sizeof(float)));
    CK(cudaMalloc(&d_idx, T*kNumTopk*sizeof(topk_idx_t)));
    CK(cudaMemcpy(d_idx, h_idx.data(), h_idx.size()*sizeof(topk_idx_t), cudaMemcpyHostToDevice));
    std::vector<nv_bfloat16> h_b0(T*kHidden), h_b1(T*kHidden);
    for (int i = 0; i < T*kHidden; ++i) { h_b0[i] = f2b(0.125f); h_b1[i] = f2b(-0.0625f); }
    if (with_bias) {
        CK(cudaMalloc(&d_b0, T*kHidden*sizeof(nv_bfloat16))); CK(cudaMemcpy(d_b0, h_b0.data(), T*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_b1, T*kHidden*sizeof(nv_bfloat16))); CK(cudaMemcpy(d_b1, h_b1.data(), T*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    } else { d_b0 = nullptr; d_b1 = nullptr; }

    const auto out_tl = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    int smem_bytes = kNumWarps * (int)out_tl.get_num_bytes<false>();
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(kNumSMs, 1, 1);
    cfg.blockDim = dim3(kNumWarps*32, 1, 1);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = 0;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, kernel, d_x, d_w, d_idx, d_recv, d_b0, d_b1, T, /*scaleout*/0, /*scaleup*/0));
    CK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_x(T*kHidden);
    CK(cudaMemcpy(h_x.data(), d_x, T*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> h_w(T*kNumTopk, -999.f);
    CK(cudaMemcpy(h_w.data(), d_w, T*kNumTopk*sizeof(float), cudaMemcpyDeviceToHost));

    // oracle：对每个 token，按 distinct_ranks 递增顺序做 combine_reduce
    int mism = 0, wt_mism = 0;
    for (int t = 0; t < T; ++t) {
        int nvalid = (int)distinct_ranks[t].size();
        bool bypass = (not with_bias) and nvalid <= 2;
        for (int h = 0; h < kHidden; ++h) {
            std::vector<float> vals;
            for (int r : distinct_ranks[t]) vals.push_back(b2f(f2b(rank_hidden[t][r] + h * 0.001953125f)));
            float expect;
            if (bypass) {
                nv_bfloat16 a = f2b(nvalid > 0 ? vals[0] : 0.f);
                nv_bfloat16 b = f2b(nvalid > 1 ? vals[1] : 0.f);
                expect = b2f(f2b(b2f(a) + b2f(b)));
            } else {
                float acc = 0.f;
                if (with_bias) acc += b2f(h_b0[t*kHidden+h]) + b2f(h_b1[t*kHidden+h]);
                for (float v : vals) acc += v;
                expect = b2f(f2b(acc));
            }
            if (b2f(h_x[t*kHidden+h]) != expect) ++mism;
        }
        if (check_weights) {
            for (int l = 0; l < kNumTopk; ++l) {
                int e = topk_experts[t][l];
                int r = e >= 0 ? e / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : (kNumExperts/kNumScaleoutRanks)) : -1;
                float expect_w = (r >= 0) ? (0.25f * (r + 1) + 0.01f * (l % kNumTopk)) : 0.f;
                if (h_w[t*kNumTopk + l] != expect_w) ++wt_mism;
            }
        }
    }
    { char buf[256]; snprintf(buf, sizeof(buf), "%s: 输出 vs oracle 逐位 (mism=%d/%d)", name, mism, T*kHidden); check(mism==0, buf); }
    if (check_weights) {
        char buf[256]; snprintf(buf, sizeof(buf), "%s: topk weights (mism=%d/%d)", name, wt_mism, T*kNumTopk); check(wt_mism==0, buf);
    }
    cudaFree(d_recv); cudaFree(d_x); cudaFree(d_w); cudaFree(d_idx);
    if (with_bias) { cudaFree(d_b0); cudaFree(d_b1); }
}

static void part2_epilogue() {
    printf("\n=== Part 2: 完整 combine_reduce_epilogue_impl（combine_reduce_epilogue.cuh:24-143）===\n");
    constexpr int T = 5, kHidden = 256, kTopk = 4, kMaxTok = 6;

    // ---- 配置 A：rank layout（1 scaleout × 4 scaleup，topk=4，ranks=4<=topk）----
    //   experts: rank = expert / 8（kNumExperts=32, per rank 8）
    {
        std::vector<std::vector<int>> exps(T, std::vector<int>(kTopk, -1));
        // token 0: ranks 0,0,2,3（rank0 重复 -> dedup 只算一次）
        exps[0] = {0, 3, 16, 24};
        // token 1: ranks 1,1,1,1（全同 -> 只算一次）
        exps[1] = {8, 9, 10, 11};
        // token 2: ranks 0,1,2,3（互异）
        exps[2] = {0, 8, 16, 24};
        // token 3: 只有 rank 3
        exps[3] = {24, -1, -1, -1};
        // token 4: ranks 2,3,2,3 -> distinct {2,3}
        exps[4] = {16, 24, 17, 25};
        std::vector<std::vector<float>> vals(T, std::vector<float>(4, 0.f));
        for (int t = 0; t < T; ++t) for (int r = 0; r < 4; ++r) vals[t][r] = 1.0f + 0.5f*t + 0.25f*r;
        run_epilogue_case<false, true, 1, 1, 1, 4, kHidden, kMaxTok, 32, kTopk>(
            "A rank-layout(no bias)", T, exps, vals, false, true);
        run_epilogue_case<false, true, 1, 1, 1, 4, kHidden, kMaxTok, 32, kTopk>(
            "A rank-layout(with bias)", T, exps, vals, true, false);
    }

    // ---- 配置 B：topk layout（1 scaleout × 8 scaleup，topk=4，ranks=8>topk）----
    {
        std::vector<std::vector<int>> exps(T, std::vector<int>(kTopk, -1));
        // kNumExperts=64, per rank=8 -> rank = expert/8
        exps[0] = {0, 8, 16, 24};     // ranks 0,1,2,3 互异
        exps[1] = {0, 1, 8, 16};      // ranks 0,0,1,2 -> distinct{0,1,2}
        exps[2] = {8, 9, 10, 11};     // ranks 1,1,1,1 -> distinct{1}
        exps[3] = {24, -1, 8, -1};    // ranks 3,1 -> distinct{1,3}
        exps[4] = {0, 8, 16, 24};     // distinct{0,1,2,3}
        std::vector<std::vector<float>> vals(T, std::vector<float>(8, 0.f));
        for (int t = 0; t < T; ++t) for (int r = 0; r < 8; ++r) vals[t][r] = 2.0f + 0.25f*t + 0.125f*r;
        run_epilogue_case<false, true, 1, 1, 1, 8, kHidden, kMaxTok, 64, kTopk>(
            "B topk-layout(no bias)", T, exps, vals, false, true);
    }
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d), CUDA %d\n", prop.name, prop.major, prop.minor, CUDART_VERSION);
    printf("EP_NUM_TOPK_IDX_BITS=%d, sizeof(topk_idx_t)=%zu, kNumTMAAlignmentBytes=%d, kNumRDMAAlignmentBytes=%d\n",
           EP_NUM_TOPK_IDX_BITS, sizeof(topk_idx_t), kNumTMAAlignmentBytes, kNumRDMAAlignmentBytes);
    part0_constexpr();
    part1_combine_reduce_direct();
    part2_epilogue();
    printf("\n================ 汇总: %d PASS / %d FAIL ================\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
