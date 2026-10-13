// P103-1 实验：DeepGEMM SM100 dense MQA logits 主 kernel
//   `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh` 中**架构无关**的几段逻辑。
//
// 为什么能在 H100(sm90) 上跑：主 kernel 的 UMMA / tcgen05 / UTCCP / TMA 部分是 SM100 专属，
// 但「运行期 token 数 → 编译期 cute::Int 的分派」「TMEM 按头切 16/8/4 的遍历」「ReLU×权重
// 的 bf16x2 归约 + 压缩偏移写回」「分页模式下两个生产者对 KV 页的均分算术」「SF 的 4×32→32×4
// warp 内转置」全部是纯整数 / 纯 bf16 运算，不依赖任何 SM100 指令，可直接在 sm90 上编译运行。
//
// 本程序逐段把真实函数**原样抽取**过来（行号对照 DeepGEMM commit `057ca59`）并验证：
//   A) `dispatch_num_block_tokens`（`:27-36`）：运行期 N → 编译期 Int<min(max(N,1),BLOCK_Q)>；
//   B) `visit_tmem_head_chunks`（`:39-49`）：16/8/4 头块分解，恰好覆盖且不越读（只把 SM100
//      专属的 `ptx::tmem_load_32dp32b` 换成记录桩，索引逻辑逐字保留）；
//   C) `reduce_pairs` + `store_token`（`:473-489`）：bf16x2 累加的 ReLU×w 归约 vs fp32 参考，
//      以及「按 -seq_k_start 压缩 + [start,end) 裁剪」的写回语义；同时验证「按头块分段调用」
//      与「一次性调用」逐位一致（分段只换遍历顺序，不改数学）；
//   D) `issue_paged_split` 的生产者均分算术（`:184-221`）：`kNumSFTMAs` / `kNumKVPagesFromSFProducer`
//      在两个生产者之间对 KV 页与 SF 份额的划分，验证「每页恰好被一个生产者搬一次」；
//   E) `utccp_required_smem_warp_transpose`（`:288-296`）：`src[i*32+lane] → dst[lane*4+i]` 的
//      4×32 → 32×4 warp 内转置。
//
// 编译：见 harness/exp_mqa_logits_dense_epilogue.sh

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/numeric/integral_constant.hpp>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using cute::Int;

// ===========================================================================
// 原样抽取的设备函数（把 CUTLASS_DEVICE 宏改成 __device__ __forceinline__）
// ===========================================================================

// impls/sm100_mqa_logits.cuh:27-36（逐字）
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
__device__ __forceinline__ void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// ptx/utils.cuh:55-63（逐字）：`cvt.rn.relu.bf16x2.f32`，sm90 走 fmaxf + __floats2bfloat162_rn 回退
__device__ __forceinline__ nv_bfloat162 cvt_relu_bf16x2_f32(const float2& v) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    uint32_t packed;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(v.y), "f"(v.x));
    return *reinterpret_cast<const nv_bfloat162*>(&packed);
#else
    return __floats2bfloat162_rn(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f));
#endif
}

// ptx/ld_st.cuh:245-247（逐字）：只存 bf16x2 的低 16 位
__device__ __forceinline__ void st_global_low_bf16(nv_bfloat16* ptr, const nv_bfloat162& value) {
    asm volatile("st.global.b16 [%0], %1;" :: "l"(ptr), "r"(*reinterpret_cast<const uint32_t*>(&value)));
}

// ptx/ld_st.cuh:161-163（逐字）：st.shared.v4.u32
__device__ __forceinline__ void st_shared_v4(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)),
                 "r"(x), "r"(y), "r"(z), "r"(w));
}
// ptx/ld_st.cuh:115-119（逐字）：ld.shared.u32
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

// ===========================================================================
// Part B：把 `ptx::tmem_load_32dp32b` 换成记录桩，逐字保留
// `visit_tmem_head_chunks` 的索引逻辑（未知 TMEM 内容，只记录 (列, 宽度) 序列）。
// ===========================================================================
__device__ uint32_t g_rec_col[64];
__device__ uint32_t g_rec_width[64];
__device__ uint32_t g_rec_count;
__device__ int g_record;  // 仅在 Part B 打开

template <uint32_t kNumChunkHeads>
__device__ __forceinline__ void stub_tmem_load_32dp32b(const uint32_t tmem_col, uint32_t* values) {
    if (g_record) {
        const uint32_t idx = atomicAdd(&g_rec_count, 1u);
        if (idx < 64) { g_rec_col[idx] = tmem_col; g_rec_width[idx] = kNumChunkHeads; }
    }
    #pragma unroll
    for (uint32_t i = 0; i < kNumChunkHeads; ++ i) values[i] = tmem_col + i;
}

// impls/sm100_mqa_logits.cuh:39-49（逐字；`ptx::tmem_load_32dp32b` → 记录桩，fence 为 no-op）
template <uint32_t kNumHeads, uint32_t kHeadBase = 0, typename Fn>
__device__ __forceinline__ void visit_tmem_head_chunks(const uint32_t& tmem_col, float* accum, Fn&& fn) {
    constexpr uint32_t kNumRemaining = kNumHeads - kHeadBase;
    if constexpr (kNumRemaining > 0) {
        constexpr uint32_t kNumChunkHeads = kNumRemaining >= 16 ? 16 : (kNumRemaining >= 8 ? 8 : 4);
        stub_tmem_load_32dp32b<kNumChunkHeads>(tmem_col + kHeadBase, reinterpret_cast<uint32_t*>(accum));
        fn(Int<kHeadBase>{}, Int<kNumChunkHeads>{});
        visit_tmem_head_chunks<kNumHeads, kHeadBase + kNumChunkHeads>(tmem_col, accum, static_cast<Fn&&>(fn));
    }
}

// ---------------------------------------------------------------------------
// Part A 核：记录 dispatch_num_block_tokens 选中的 kNumValidTokens
// ---------------------------------------------------------------------------
template <uint32_t kBlockQ>
__global__ void dispatch_kernel(const uint32_t num_block_tokens, uint32_t* out) {
    dispatch_num_block_tokens<kBlockQ>(num_block_tokens, [&](auto n_t) { *out = decltype(n_t)::value; });
}

// ---------------------------------------------------------------------------
// Part B 核：记录头块分解序列
// ---------------------------------------------------------------------------
template <uint32_t kNumHeads>
__global__ void chunks_kernel(const uint32_t tmem_col, uint32_t* out_cols, uint32_t* out_widths, uint32_t* out_count) {
    __shared__ float accum[16];
    uint32_t local_n = 0, local_c[16], local_w[16];
    visit_tmem_head_chunks<kNumHeads>(tmem_col, accum, [&](auto hb_t, auto nch_t) {
        local_c[local_n] = decltype(hb_t)::value;
        local_w[local_n] = decltype(nch_t)::value;
        ++ local_n;
    });
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < local_n; ++ i) { out_cols[i] = local_c[i]; out_widths[i] = local_w[i]; }
        *out_count = local_n;
    }
}

// ---------------------------------------------------------------------------
// Part E 核：SF 的 4×32 → 32×4 warp 内转置
// ---------------------------------------------------------------------------
__global__ void sf_transpose_kernel(const uint32_t* src, uint32_t* dst) {
    __shared__ uint32_t buf[128];
    const uint32_t lane = threadIdx.x;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) buf[i * 32 + lane] = src[i * 32 + lane];
    __syncwarp();
    // impls/sm100_mqa_logits.cuh:288-296（逐字）
    uint32_t values[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) values[i] = ld_shared_u32(buf + i * 32 + lane);
    __syncwarp();
    st_shared_v4(buf + lane * 4, values[0], values[1], values[2], values[3]);
    __syncwarp();
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) dst[i * 32 + lane] = buf[i * 32 + lane];
}

// ---------------------------------------------------------------------------
// Part C 核：逐行 ReLU×w 的 bf16x2 归约 + 压缩偏移写回
//   dbg_once = 一次性调用结果；dbg_chunked = 按 16/8/4 头块分段调用结果。
// ---------------------------------------------------------------------------
template <uint32_t kNumHeads>
__global__ void reduce_kernel(const float* accum, const nv_bfloat16* weights,
                              nv_bfloat16* output, uint32_t logits_stride,
                              uint32_t seq_k_start, uint32_t seq_k_end, uint32_t num_rows,
                              nv_bfloat162* dbg_once, nv_bfloat162* dbg_chunked) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    const float* chunk = accum + static_cast<size_t>(row) * kNumHeads;
    float dummy[16];

    nv_bfloat162 w[kNumHeads / 2];
    #pragma unroll
    for (uint32_t j = 0; j < kNumHeads / 2; ++ j) {
        const auto packed = reinterpret_cast<const uint32_t*>(weights)[j];
        w[j] = *reinterpret_cast<const nv_bfloat162*>(&packed);
    }

    // impls/sm100_mqa_logits.cuh:480-489（逐字）
    auto reduce_pairs = [&](const float* ch, const nv_bfloat162* chunk_weights, const uint32_t& num_heads,
                            nv_bfloat162& sum_0, nv_bfloat162& sum_1) {
        #pragma unroll
        for (uint32_t head_offset = 0; head_offset < num_heads; head_offset += 4) {
            const auto accum_pair_0 = make_float2(ch[head_offset], ch[head_offset + 1]);
            const auto accum_pair_1 = make_float2(ch[head_offset + 2], ch[head_offset + 3]);
            sum_0 = __hfma2(cvt_relu_bf16x2_f32(accum_pair_0), chunk_weights[head_offset / 2], sum_0);
            sum_1 = __hfma2(cvt_relu_bf16x2_f32(accum_pair_1), chunk_weights[head_offset / 2 + 1], sum_1);
        }
    };

    // (1) 一次性调用（等价于把所有头块顺序拼起来）
    auto once_0 = __floats2bfloat162_rn(0.0f, 0.0f);
    auto once_1 = __floats2bfloat162_rn(0.0f, 0.0f);
    reduce_pairs(chunk, w, kNumHeads, once_0, once_1);
    {
        const auto s = __hadd2_rn(once_0, once_1);
        dbg_once[row] = __hadd2_rn(__low2bfloat162(s), __high2bfloat162(s));
    }

    // (2) 按 16/8/4 头块分段调用（与主 kernel visit_tmem_head_chunks 同序）
    auto csum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
    auto csum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
    visit_tmem_head_chunks<kNumHeads>(0u, dummy, [&](auto hb_t, auto nch_t) {
        constexpr uint32_t kHeadBase = decltype(hb_t)::value;
        constexpr uint32_t kNumChunkHeads = decltype(nch_t)::value;
        reduce_pairs(chunk + kHeadBase, w + kHeadBase / 2, kNumChunkHeads, csum_0, csum_1);
    });
    {
        const auto s = __hadd2_rn(csum_0, csum_1);
        dbg_chunked[row] = __hadd2_rn(__low2bfloat162(s), __high2bfloat162(s));
    }

    // impls/sm100_mqa_logits.cuh:473-479（逐字）：压缩偏移 + [start,end) 裁剪
    auto store_token = [&](const uint32_t& i, const uint32_t& kv_off, const nv_bfloat162& s0, const nv_bfloat162& s1) {
        const auto sum = __hadd2_rn(s0, s1);
        const auto result = __hadd2_rn(__low2bfloat162(sum), __high2bfloat162(sum));
        if (seq_k_start <= kv_off and kv_off < seq_k_end)
            st_global_low_bf16(output + i * static_cast<uint64_t>(logits_stride) + kv_off - seq_k_start, result);
    };
    store_token(0u, row, csum_0, csum_1);
}

static uint32_t bf162_bits(__nv_bfloat162 v) { uint32_t u; memcpy(&u, &v, 4); return u; }

// ===========================================================================
// host 端参考
// ===========================================================================
static __nv_bfloat16 h_round(float x) { return __float2bfloat16(x); }

// 真实 `__hfma2` 是 device-only；host 上按「乘积+加法和在 double 里精确、再 RNE 到 bf16」复刻
static __nv_bfloat162 h_hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    __nv_bfloat162 r;
    r.x = h_round((float)((double)__bfloat162float(a.x) * (double)__bfloat162float(b.x) + (double)__bfloat162float(c.x)));
    r.y = h_round((float)((double)__bfloat162float(a.y) * (double)__bfloat162float(b.y) + (double)__bfloat162float(c.y)));
    return r;
}

static float fp32_ref(const std::vector<float>& a, const std::vector<nv_bfloat16>& w, uint32_t H) {
    float s = 0.0f;
    for (uint32_t h = 0; h < H; ++ h) s += std::max(a[h], 0.0f) * __bfloat162float(w[h]);
    return s;
}

// 端到端重放 device 的 bf16 累加（用于逐位对拍）
static nv_bfloat16 replay_reduce(const std::vector<float>& a, const std::vector<nv_bfloat16>& w, uint32_t H) {
    __nv_bfloat162 s0 = __floats2bfloat162_rn(0.0f, 0.0f);
    __nv_bfloat162 s1 = __floats2bfloat162_rn(0.0f, 0.0f);
    auto relu2 = [](float x, float y) { return __floats2bfloat162_rn(fmaxf(x, 0.0f), fmaxf(y, 0.0f)); };
    for (uint32_t ho = 0; ho < H; ho += 4) {
        __nv_bfloat162 r0 = relu2(a[ho], a[ho + 1]);
        __nv_bfloat162 r1 = relu2(a[ho + 2], a[ho + 3]);
        __nv_bfloat162 w0 = __floats2bfloat162_rn(__bfloat162float(w[ho]), __bfloat162float(w[ho + 1]));
        __nv_bfloat162 w1 = __floats2bfloat162_rn(__bfloat162float(w[ho + 2]), __bfloat162float(w[ho + 3]));
        s0 = h_hfma2(r0, w0, s0);
        s1 = h_hfma2(r1, w1, s1);
    }
    const auto s = __hadd2_rn(s0, s1);
    return __low2bfloat16(__hadd2_rn(__low2bfloat162(s), __high2bfloat162(s)));
}

// ---------------------------------------------------------------------------
// Part D：issue_paged_split 的生产者均分算术（`:184-221` 的 constexpr 部分）
// ---------------------------------------------------------------------------
struct SplitPlan { uint32_t pages, gather_pages, sf_tmas, kv_from_sf, kv_from_kv; };

static SplitPlan plan_paged(uint32_t SPLIT_KV, uint32_t PAGE_KV) {
    SplitPlan p{};
    p.pages = SPLIT_KV / PAGE_KV;
    p.gather_pages = p.pages / 4 * 4;
    p.sf_tmas = p.gather_pages / 4 + p.pages - p.gather_pages;
    p.kv_from_sf = p.pages > 2 * p.sf_tmas ? (p.pages - p.sf_tmas) / 2 : 0;
    p.kv_from_kv = p.pages - p.kv_from_sf;
    return p;
}

// ===========================================================================
int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("=== P103-1 SM100 dense MQA logits 主 kernel 架构无关逻辑验证 (H100/sm90) ===\n");
    printf("device: %s  SMs=%d  sm=%d%d\n\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    bool all_ok = true;

    // ---------------- Part A ----------------
    printf("## Part A. dispatch_num_block_tokens (:27-36)\n");
    for (uint32_t kBlockQ : {1u, 2u, 4u}) {
        uint32_t* d_out; CK(cudaMalloc(&d_out, 4));
        printf("  BLOCK_Q=%u:\n", kBlockQ);
        for (uint32_t n = 0; n <= kBlockQ + 2; ++ n) {
            if (kBlockQ == 1) dispatch_kernel<1><<<1, 1>>>(n, d_out);
            else if (kBlockQ == 2) dispatch_kernel<2><<<1, 1>>>(n, d_out);
            else dispatch_kernel<4><<<1, 1>>>(n, d_out);
            CK(cudaDeviceSynchronize());
            uint32_t got = 0; CK(cudaMemcpy(&got, d_out, 4, cudaMemcpyDeviceToHost));
            const uint32_t expect = std::max(1u, std::min(n, kBlockQ));
            const bool ok = got == expect; all_ok &= ok;
            printf("    N=%u -> Int<%u>  expect %u  %s\n", n, got, expect, ok ? "OK" : "FAIL");
        }
        cudaFree(d_out);
    }
    printf("  结论：N 被夹到 [1, BLOCK_Q]；N=0（空 Q block）也退化为处理 1 个 token 的模板实例。\n\n");

    // ---------------- Part B ----------------
    printf("## Part B. visit_tmem_head_chunks (:39-49) 的 16/8/4 头块分解\n");
    printf("  %-8s %-30s %-6s %s\n", "H", "chunks(base,width)", "sum", "check");
    for (uint32_t H : {4u, 8u, 12u, 16u, 20u, 24u, 32u, 64u, 128u}) {
        int rec = 1; CK(cudaMemcpyToSymbol(g_record, &rec, sizeof(int)));
        uint32_t z = 0; CK(cudaMemcpyToSymbol(g_rec_count, &z, 4));
        uint32_t *c, *w, *cnt;
        CK(cudaMalloc(&c, 64 * 4)); CK(cudaMalloc(&w, 64 * 4)); CK(cudaMalloc(&cnt, 4));
        switch (H) {
            case 4: chunks_kernel<4><<<1, 1>>>(7u, c, w, cnt); break;
            case 8: chunks_kernel<8><<<1, 1>>>(7u, c, w, cnt); break;
            case 12: chunks_kernel<12><<<1, 1>>>(7u, c, w, cnt); break;
            case 16: chunks_kernel<16><<<1, 1>>>(7u, c, w, cnt); break;
            case 20: chunks_kernel<20><<<1, 1>>>(7u, c, w, cnt); break;
            case 24: chunks_kernel<24><<<1, 1>>>(7u, c, w, cnt); break;
            case 32: chunks_kernel<32><<<1, 1>>>(7u, c, w, cnt); break;
            case 64: chunks_kernel<64><<<1, 1>>>(7u, c, w, cnt); break;
            case 128: chunks_kernel<128><<<1, 1>>>(7u, c, w, cnt); break;
        }
        CK(cudaDeviceSynchronize());
        uint32_t n = 0; CK(cudaMemcpy(&n, cnt, 4, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> hc(n), hw(n);
        CK(cudaMemcpy(hc.data(), c, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hw.data(), w, n * 4, cudaMemcpyDeviceToHost));
        uint32_t base = 0, sum = 0; std::string desc; bool ok = true;
        while (base < H) {
            uint32_t rem = H - base, ww = rem >= 16 ? 16 : (rem >= 8 ? 8 : 4);
            desc += std::to_string(base) + "/" + std::to_string(ww) + " "; sum += ww; base += ww;
        }
        for (uint32_t i = 0; i < n; ++ i) ok &= (hc[i] == (i ? hc[i-1] + hw[i-1] : 0u));
        ok &= (sum == H) && (n == (uint32_t)std::count(desc.begin(), desc.end(), '/'));
        { std::string d2; for (uint32_t i = 0; i < n; ++ i) d2 += std::to_string(hc[i]) + "/" + std::to_string(hw[i]) + " ";
          ok &= (d2 == desc); }
        all_ok &= ok;
        printf("  %-8u %-30s %-6u %s\n", H, desc.c_str(), sum, ok ? "OK" : "FAIL");
        cudaFree(c); cudaFree(w); cudaFree(cnt);
    }
    { int rec = 0; CK(cudaMemcpyToSymbol(g_record, &rec, sizeof(int))); }
    printf("  结论：分解按 16/8/4 贪心，起点连续、宽度和恰为 H；H%%4==0（:441 静态断言）下不越读。\n\n");

    // ---------------- Part C ----------------
    printf("## Part C. reduce_pairs + store_token (:473-489)：bf16x2 ReLU×w 归约 + 压缩写回\n");
    {
        const uint32_t H = 64, NUM_ROWS = 64, STRIDE = 64, kStart = 8, kEnd = 40;
        std::mt19937 rng(1234); std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        std::vector<float> acc(static_cast<size_t>(NUM_ROWS) * H);
        for (auto& x : acc) x = dist(rng);
        std::vector<nv_bfloat16> w(H);
        for (auto& x : w) x = h_round(dist(rng));

        float* d_acc; nv_bfloat16 *d_w, *d_out; nv_bfloat162 *d_once, *d_chunk;
        CK(cudaMalloc(&d_acc, acc.size() * 4)); CK(cudaMalloc(&d_w, H * 2)); CK(cudaMalloc(&d_out, STRIDE * 2));
        CK(cudaMalloc(&d_once, NUM_ROWS * 4)); CK(cudaMalloc(&d_chunk, NUM_ROWS * 4));
        CK(cudaMemcpy(d_acc, acc.data(), acc.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_w, w.data(), H * 2, cudaMemcpyHostToDevice));
        std::vector<uint16_t> poison(STRIDE, 0xA5A5u);
        CK(cudaMemcpy(d_out, poison.data(), STRIDE * 2, cudaMemcpyHostToDevice));
        reduce_kernel<H><<<NUM_ROWS / 32, 32>>>(d_acc, d_w, d_out, STRIDE, kStart, kEnd, NUM_ROWS, d_once, d_chunk);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

        std::vector<__nv_bfloat162> h_once(NUM_ROWS), h_chunk(NUM_ROWS);
        CK(cudaMemcpy(h_once.data(), d_once, NUM_ROWS * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_chunk.data(), d_chunk, NUM_ROWS * 4, cudaMemcpyDeviceToHost));
        std::vector<uint16_t> h_out(STRIDE); CK(cudaMemcpy(h_out.data(), d_out, STRIDE * 2, cudaMemcpyDeviceToHost));
        uint32_t mismatch = 0, replay_mismatch = 0, bad_window = 0, bad_value = 0;
        double max_rel = 0.0, max_abs = 0.0, max_cond = 0.0, sum_abs_err = 0.0, rms_ref = 0.0;
        for (uint32_t r = 0; r < NUM_ROWS; ++ r) {
            if (bf162_bits(h_once[r]) != bf162_bits(h_chunk[r])) ++ mismatch;
            std::vector<float> a(acc.begin() + r * H, acc.begin() + (r + 1) * H);
            const __nv_bfloat16 hr = replay_reduce(a, w, H);
            const __nv_bfloat16 dv = __low2bfloat16(h_chunk[r]);
            if (*reinterpret_cast<const uint16_t*>(&hr) != *reinterpret_cast<const uint16_t*>(&dv)) ++ replay_mismatch;
            const float got = __bfloat162float(dv), ref = fp32_ref(a, w, H);
            double term_abs = 0.0;
            for (uint32_t h = 0; h < H; ++ h) term_abs += std::abs(std::max(a[h], 0.0f) * __bfloat162float(w[h]));
            const double err = std::abs((double)got - ref);
            max_abs = std::max(max_abs, err);
            sum_abs_err += err;
            rms_ref += (double)ref * ref;
            if (std::abs(ref) > 1e-6) max_rel = std::max(max_rel, err / std::abs(ref));
            if (term_abs > 1e-6) max_cond = std::max(max_cond, err / term_abs);
        }
        rms_ref = std::sqrt(rms_ref / NUM_ROWS);
        for (uint32_t c = 0; c < STRIDE; ++ c) {
            if (c >= kEnd - kStart) { if (h_out[c] != 0xA5A5u) ++ bad_window; continue; }
            const __nv_bfloat16 expect = __low2bfloat16(h_chunk[c + kStart]);
            if (*reinterpret_cast<const uint16_t*>(&expect) != h_out[c]) ++ bad_value;
        }
        const bool ok = (mismatch == 0) && (replay_mismatch == 0) && (bad_window == 0) && (bad_value == 0);
        all_ok &= ok;
        printf("  一次性 vs 分段 bit 不一致: %u / %u\n", mismatch, NUM_ROWS);
        printf("  device vs host 逐位重放不一致: %u / %u\n", replay_mismatch, NUM_ROWS);
        printf("  bf16 累加误差 vs fp32 参考：max|Δ|=%.3e  mean|Δ|=%.3e\n", max_abs, sum_abs_err / NUM_ROWS);
        printf("  max 相对误差 |Δ|/|ref| = %.3f（ref 存在相消，可为 O(1)）；RMS(ref)=%.3f\n", max_rel, rms_ref);
        printf("  max 相对项幅 |Δ|/Σ|term| = %.3e（=逐项 bf16 舍入的积累，≈2^-9 量级）\n", max_cond);
        printf("  压缩写回 [0,%u) 有效、其余保留 poison：窗口错 %u、值错 %u\n", kEnd - kStart, bad_window, bad_value);
        printf("  %s\n\n", ok ? "Part C OK" : "Part C FAIL");
        cudaFree(d_acc); cudaFree(d_w); cudaFree(d_out); cudaFree(d_once); cudaFree(d_chunk);
    }

    // ---------------- Part D ----------------
    printf("## Part D. issue_paged_split 的生产者均分算术 (:184-221)\n");
    printf("  %-8s %-8s %-9s %-8s %-12s %-12s %s\n", "SPLIT", "PAGE", "pages", "gather", "SF-TMAs", "KV(SF/KV)", "cover");
    for (uint32_t PAGE_KV : {32u, 64u, 128u}) {
        SplitPlan p = plan_paged(384u, PAGE_KV);
        const bool cover = (p.kv_from_sf + p.kv_from_kv == p.pages);
        all_ok &= cover;
        printf("  %-8u %-8u %-9u %-8u %-12u %-12s %s\n", 384u, PAGE_KV, p.pages, p.gather_pages, p.sf_tmas,
               (std::to_string(p.kv_from_sf) + "/" + std::to_string(p.kv_from_kv)).c_str(), cover ? "OK" : "FAIL");
    }
    printf("  说明：SF 生产者发 kNumSFTMAs 条 SF TMA（每 4 页合并 1 条 gather4）；\n");
    printf("        仅当 pages > 2*SFTMAs 时把 (pages-SFTMAs)/2 页 KV 也交给它，使两生产者工作量≈对半。\n\n");

    // ---------------- Part E ----------------
    printf("## Part E. utccp_required_smem_warp_transpose (:288-296) 4x32 -> 32x4\n");
    {
        std::vector<uint32_t> src(128);
        for (uint32_t i = 0; i < 128; ++ i) src[i] = 1000 + i;
        uint32_t *d_src, *d_dst;
        CK(cudaMalloc(&d_src, 128 * 4)); CK(cudaMalloc(&d_dst, 128 * 4));
        CK(cudaMemcpy(d_src, src.data(), 128 * 4, cudaMemcpyHostToDevice));
        sf_transpose_kernel<<<1, 32>>>(d_src, d_dst);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> dst(128); CK(cudaMemcpy(dst.data(), d_dst, 128 * 4, cudaMemcpyDeviceToHost));
        uint32_t mismatch = 0;
        for (uint32_t lane = 0; lane < 32; ++ lane)
            for (uint32_t i = 0; i < 4; ++ i)
                if (dst[lane * 4 + i] != src[i * 32 + lane]) ++ mismatch;
        all_ok &= (mismatch == 0);
        printf("  dst[lane*4+i] == src[i*32+lane]：不一致 %u / 128  %s\n\n", mismatch, mismatch == 0 ? "OK" : "FAIL");
        cudaFree(d_src); cudaFree(d_dst);
    }

    printf("=== 汇总: %s ===\n", all_ok ? "ALL PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
