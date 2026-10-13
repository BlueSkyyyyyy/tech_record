// P126-1 实验：DeepGEMM SM100 sparse MQA logits 主 kernel
//   `deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh`（688 行）+ `csrc/jit_kernels/impls/
//   sm100_sparse_mqa_logits.hpp`（202 行）+ `layout/sparse_mqa_logits.cuh`（156 行）中**架构无关**的逻辑。
//
// 为什么能在 H100(sm90) 上跑：主 kernel 的 UMMA（`SM100_MMA_MXF8F6F4_SS`/`MXF4_SS`）、tcgen05、UTCCP、
// 2-SM TMA 是 SM100 专属；但下列几段是纯整数 / 纯 bf16 运算，不依赖任何 SM100 指令，可直接在 sm90 上
// 编译运行（`SharedStorage` 只含 constexpr/alignas/cutlass::arch::ClusterTransactionBarrier，见第 82 篇）。
//
//   A) `get_swizzled_kv_chunk_idx`（`:28-32`）：KV smem 的 16B-chunk XOR 自定义 swizzle；
//   B) `KVSplitHeader`/`KVBlockInfo`（`layout:31-72`）：标记位打包 + 双 Q 槽 16+16 位打包/解码；
//   C) KV 访问几何：连续版 `get_num_valid_tokens`（`:96-98`）尾块裁剪 + 分页版
//      `resolve_kv_block`（`:121-131`）的页内映射（sparse 块不跨页）；
//   D) math warp 的压缩列映射（`:588-590`、`:627-633`）：slot 解码 + `(base+slot)*SB + t` 双射；
//   E) 资源账：`get_num_threads`、TMEM 列预算（`:180-196`）、`kNumKVBlocksPerWarp` 静态断言、
//      `SharedStorage` 4 组配置的 offsetof/sizeof 与独立字节模型；
//   F) SF warp 的 4x32 -> 32x4 `transpose_sf`（`:325-333`，与 dense 版同源）。
//
// 编译：见 harness/exp_sparse_mqa_logits_kernel.sh

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/float_subbyte.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/sparse_mqa_logits.cuh>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using namespace deep_gemm::layout::sparse_mqa_logits;

// ===========================================================================
// 原样抽取的设备函数（行号对照 DeepGEMM commit 057ca59）
// ===========================================================================

// impls/sm100_sparse_mqa_logits.cuh:28-32（逐字）
template <uint32_t kSwizzleMode>
__device__ __forceinline__ uint32_t get_swizzled_kv_chunk_idx(const uint32_t logical_chunk_idx) {
    constexpr uint32_t kMask = kSwizzleMode / 16 - 1;
    return (logical_chunk_idx & ~kMask) | ((logical_chunk_idx & kMask) ^ ((logical_chunk_idx >> 3u) & kMask));
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

// ---------------------------------------------------------------------------
// Part A 核：记录 swizzle 映射
// ---------------------------------------------------------------------------
__global__ void swizzle_kernel_rt(uint32_t N, uint32_t* out) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = (N == 64) ? get_swizzled_kv_chunk_idx<64>(idx) : get_swizzled_kv_chunk_idx<128>(idx);
}

// ---------------------------------------------------------------------------
// Part C 核：连续版尾块裁剪 + 分页版页内映射（抽取真实算术）
// ---------------------------------------------------------------------------
// impls/sm100_sparse_mqa_logits.cuh:96-98（逐字，cute::min -> min）
__device__ __forceinline__ uint32_t contiguous_num_valid_tokens(uint32_t SPARSE_BLOCK_KV,
                                                               uint32_t num_kv_tokens, uint32_t kv_block_ref) {
    return min(SPARSE_BLOCK_KV, num_kv_tokens - min(kv_block_ref, num_kv_tokens));
}

// impls/sm100_sparse_mqa_logits.cuh:121-131（逐字，只回页内字节偏移而非指针；
//   真实代码 `page = fused_kv_cache + physical_page_idx * kv_page_stride_bytes`）
__device__ __forceinline__ void paged_resolve(uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV,
                                              uint32_t kNumQKBytesPerToken, uint32_t kv_page_stride_bytes,
                                              uint32_t physical_kv_block_idx,
                                              uint64_t* kv_off, uint64_t* sf_off) {
    const uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
    const uint32_t physical_page_idx = physical_kv_block_idx / kNumKVBlocksPerPage;
    const uint32_t kv_block_idx_in_page = physical_kv_block_idx % kNumKVBlocksPerPage;
    const uint64_t page = static_cast<uint64_t>(physical_page_idx) * kv_page_stride_bytes;
    const uint32_t token_idx_in_page = kv_block_idx_in_page * SPARSE_BLOCK_KV;
    *kv_off = page + static_cast<uint64_t>(token_idx_in_page) * kNumQKBytesPerToken;
    *sf_off = page + static_cast<uint64_t>(PAGE_KV) * kNumQKBytesPerToken + token_idx_in_page;
}

__global__ void accessor_kernel(uint32_t SB, uint32_t num_kv, uint32_t block, uint32_t* out) {
    out[0] = contiguous_num_valid_tokens(SB, num_kv, block);
}
__global__ void paged_kernel(uint32_t PAGE, uint32_t SB, uint32_t bytes_per_tok, uint32_t page_stride, uint32_t block,
                             uint64_t* kv_off, uint64_t* sf_off) {
    paged_resolve(PAGE, SB, bytes_per_tok, page_stride, block, kv_off, sf_off);
}

// Part E：真实 TMEM 列对齐函数（utils.cuh:42-50）
template <uint32_t N>
__global__ void tmem_align_kernel(uint32_t* out) { *out = deep_gemm::utils::get_num_aligned_tmem_cols<N>(); }

// ---------------------------------------------------------------------------
// Part F 核：SF 的 4x32 -> 32x4 warp 内转置（impls:325-333 逐字）
// ---------------------------------------------------------------------------
__global__ void sf_transpose_kernel(const uint32_t* src, uint32_t* dst) {
    __shared__ uint32_t buf[128];
    const uint32_t lane = threadIdx.x;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) buf[i * 32 + lane] = src[i * 32 + lane];
    __syncwarp();
    // impls/sm100_sparse_mqa_logits.cuh:325-333（transpose_sf 的 4x32 一段，逐字）
    uint32_t values[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) values[i] = ld_shared_u32(buf + i * 32 + lane);
    __syncwarp();
    st_shared_v4(buf + lane * 4, values[0], values[1], values[2], values[3]);
    __syncwarp();
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) dst[i * 32 + lane] = buf[i * 32 + lane];
}

// ===========================================================================
// host 端独立模型
// ===========================================================================
static uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// 位域独立模型：把低 b 位与「右移 3 位后的同宽位域」XOR
static uint32_t swizzle_model(uint32_t N, uint32_t idx) {
    uint32_t mask = N / 16 - 1;
    uint32_t b = 0; while ((1u << b) <= mask) ++ b;  // b = log2(mask+1)
    uint32_t low = idx & mask, high = (idx >> 3) & mask;
    return (idx & ~mask) | (low ^ high);
}

// ===========================================================================
int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("=== P126-1 SM100 sparse MQA logits 主 kernel 架构无关逻辑验证 (H100/sm90) ===\n");
    printf("device: %s  SMs=%d  sm=%d%d\n\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    bool all_ok = true;

    printf("## 常量\n");
    printf("  kBlockQ=%u  kHeadDim=%u  kNumMaxHeads=%u  kNumSparseSlotBits=%u  kInvalidSparseSlot=0x%x\n",
           kBlockQ, kHeadDim, kNumMaxHeads, kNumSparseSlotBits, kInvalidSparseSlot);
    printf("  split_kv: fp8=512 fp4=640  num_kv_stages: fp8=3 fp4=5  kNumQStages=2 kNumTmemStages=5  (hpp:15-49)\n\n");

    // ---------------- Part A ----------------
    printf("## Part A. get_swizzled_kv_chunk_idx (:28-32) — KV smem 16B-chunk swizzle\n");
    printf("  %-8s %-8s %-10s %-12s %s\n", "N(fp8/4)", "kMask", "bijection", "involution", "=model");
    for (uint32_t N : {64u, 128u}) {   // fp4=64, fp8=128
        uint32_t* d_out; CK(cudaMalloc(&d_out, N * 4));
        swizzle_kernel_rt<<<1, (N + 31) / 32 * 32>>>(N, d_out);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> g(N); CK(cudaMemcpy(g.data(), d_out, N * 4, cudaMemcpyDeviceToHost));
        std::vector<int> seen(N, 0); bool bij = true, inv = true, mod = true;
        for (uint32_t i = 0; i < N; ++ i) { if (g[i] >= N || seen[g[i]]++) bij = false; if (g[g[i]] != i) inv = false;
                                            if (g[i] != swizzle_model(N, i)) mod = false; }
        all_ok &= bij && inv && mod;
        printf("  %-8u %-8u %-10s %-12s %s\n", N, N / 16 - 1, bij ? "yes" : "NO", inv ? "yes" : "NO", mod ? "yes" : "NO");
        cudaFree(d_out);
    }
    printf("  结论：把 chunk 低 3 位与该块内 bits[3,6) 做 XOR（精简版 Swizzle），是块内对合置换。\n\n");

    // ---------------- Part B ----------------
    printf("## Part B. KVSplitHeader / KVBlockInfo 打包解码 (layout:31-72)\n");
    {
        uint32_t mismatch = 0, trials = 0;
        for (uint32_t nb : {1u, 32u, 511u, 4095u}) {
            for (int contig = 0; contig <= 1; ++ contig) for (int tail = 0; tail <= 1; ++ tail) {
                KVSplitHeader h(1234u, nb, contig, tail, 7u, 9u);
                trials ++;
                if (KVSplitHeader::get_num_kv_blocks(h.packed_num_kv_blocks) != nb) ++ mismatch;
                if (KVSplitHeader::is_contiguous(h.packed_num_kv_blocks) != (bool)contig) ++ mismatch;
                if (KVSplitHeader::has_partial_tail(h.packed_num_kv_blocks) != (bool)tail) ++ mismatch;
                if (h.q_token_base != 1234u || h.q0_slot_base != 7u || h.q1_slot_base != 9u) ++ mismatch;
            }
        }
        for (uint32_t q0 : {0u, 1u, 4095u, kInvalidSparseSlot}) {
            for (uint32_t q1 : {0u, 63u, 4094u, kInvalidSparseSlot}) {
                KVBlockInfo bi(99u, q0, q1);
                trials ++;
                if (bi.physical_kv_block_idx != 99u) ++ mismatch;
                if (bi.packed_slot_offsets != (q0 | (q1 << kNumSparseSlotBits))) ++ mismatch;
                if (((bi.packed_slot_offsets >> 0) & kInvalidSparseSlot) != q0) ++ mismatch;
                if (((bi.packed_slot_offsets >> kNumSparseSlotBits) & kInvalidSparseSlot) != q1) ++ mismatch;
            }
        }
        all_ok &= (mismatch == 0);
        printf("  打包/解码 %u 例，不一致 %u  %s\n", trials, mismatch, mismatch == 0 ? "OK" : "FAIL");
    }
    printf("  说明：packed_num_kv_blocks 高 2 位是 contiguous/partial-tail 标记，低位是块数（:50-60）。\n\n");

    // ---------------- Part C ----------------
    printf("## Part C. KV 访问几何：连续尾块裁剪 + 分页页内映射\n");
    {
        printf("  C1) 连续版 get_num_valid_tokens (:96-98)：SB=16, num_kv=100（kv_block_ref 是 token 起点）\n");
        uint32_t* d_out; CK(cudaMalloc(&d_out, 4));
        uint32_t bad = 0;
        printf("     %-10s %-10s %-10s %s\n", "block_ref", "valid", "expect", "check");
        for (uint32_t blk : {0u, 5u, 84u, 88u, 90u, 100u, 200u}) {
            accessor_kernel<<<1, 1>>>(16u, 100u, blk, d_out);
            CK(cudaDeviceSynchronize());
            uint32_t got = 0; CK(cudaMemcpy(&got, d_out, 4, cudaMemcpyDeviceToHost));
            uint32_t expect = std::min(16u, blk >= 100u ? 0u : 100u - blk);
            bool ok = got == expect; if (!ok) ++ bad;
            printf("     %-10u %-10u %-10u %s\n", blk, got, expect, ok ? "OK" : "FAIL");
        }
        all_ok &= (bad == 0);
        cudaFree(d_out);
        printf("     结论：起点 0..84 满 16；起点 88 只剩 12、90 只剩 10；>=100 为 0（不越界读）。\n");

        printf("  C2) 分页版 resolve_kv_block (:121-131)：PAGE=64, SB=16, bytes/tok=128\n");
        uint64_t *d_kv, *d_sf; CK(cudaMalloc(&d_kv, 8)); CK(cudaMalloc(&d_sf, 8));
        uint32_t mismatch = 0, kv_region = 64u * 128u, sf_region = 64u * 4u, page_stride = kv_region + sf_region;
        for (uint32_t blk = 0; blk < 40; ++ blk) {
            paged_kernel<<<1, 1>>>(64u, 16u, 128u, page_stride, blk, d_kv, d_sf);
            CK(cudaDeviceSynchronize());
            uint64_t gkv, gsf; CK(cudaMemcpy(&gkv, d_kv, 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&gsf, d_sf, 8, cudaMemcpyDeviceToHost));
            uint32_t per_page = 64u / 16u;
            uint64_t page = blk / per_page, in_page = blk % per_page;
            uint64_t exp_kv = page * page_stride + in_page * 16u * 128u;
            uint64_t exp_sf = page * page_stride + kv_region + in_page * 16u;
            if (gkv != exp_kv || gsf != exp_sf) ++ mismatch;
            if (blk < 5) printf("     blk=%-3u -> page=%llu in_page=%llu  kv_off=%llu sf_off=%llu\n",
                                blk, (unsigned long long)page, (unsigned long long)in_page,
                                (unsigned long long)gkv, (unsigned long long)gsf);
        }
        all_ok &= (mismatch == 0);
        printf("     40 块映射不一致 %u  %s（sparse 块不跨页，页内偏移连续）\n", mismatch, mismatch == 0 ? "OK" : "FAIL");
        cudaFree(d_kv); cudaFree(d_sf);
        printf("     说明：SF 基址 = 页基址 + PAGE*bytes/tok（fused cache 里 KV 区在前、SF 区在后，:129）。\n\n");
    }

    // ---------------- Part D ----------------
    printf("## Part D. math warp 压缩列映射 (:588-590, :627-633) 双射\n");
    {
        const uint32_t SB = 16, SPLIT_KV = 512, UMMA_M = 128, NWG = 4;
        std::vector<int> seen(SPLIT_KV, 0); uint32_t dup = 0;
        for (uint32_t wg = 0; wg < NWG; ++ wg)
            for (uint32_t tid = 0; tid < 128; ++ tid) {
                uint32_t tok = wg * UMMA_M + tid;
                uint32_t blk = tok / SB, t = tok - blk * SB;
                if (blk * SB + t != tok) ++ dup;
                seen[tok] ++;
            }
        uint32_t cover = std::count(seen.begin(), seen.end(), 1);
        const uint32_t base = 5, n_valid = 20;
        std::vector<int> col((base + n_valid) * SB + 8, 0);
        for (uint32_t slot = 0; slot < n_valid; ++ slot)
            for (uint32_t t = 0; t < SB; ++ t) col[(base + slot) * SB + t] ++;
        uint32_t col_mismatch = 0;
        for (uint32_t c = 0; c < base * SB; ++ c) if (col[c] != 0) ++ col_mismatch;
        for (uint32_t c = base * SB; c < (base + n_valid) * SB; ++ c) if (col[c] != 1) ++ col_mismatch;
        for (size_t c = (base + n_valid) * SB; c < col.size(); ++ c) if (col[c] != 0) ++ col_mismatch;
        uint32_t packed = (3u) | (kInvalidSparseSlot << kNumSparseSlotBits);
        uint32_t q0 = (packed >> 0) & kInvalidSparseSlot;
        uint32_t q1 = (packed >> kNumSparseSlotBits) & kInvalidSparseSlot;
        bool invalid_ok = (q0 == 3u) && (q1 == kInvalidSparseSlot);
        bool ok = (dup == 0) && (cover == SPLIT_KV) && (col_mismatch == 0) && invalid_ok;
        all_ok &= ok;
        printf("  token 分解：%u/%u 唯一、分解错误 %u\n", cover, SPLIT_KV, dup);
        printf("  压缩列：base=%u valid=%u -> 有效列 [%u,%u) 每列恰写一次；不一致 %u\n",
               base, n_valid, base * SB, (base + n_valid) * SB, col_mismatch);
        printf("  slot 解码：q0=%u q1=0x%x（invalid=0x%x）  %s\n", q0, q1, kInvalidSparseSlot, invalid_ok ? "OK" : "FAIL");
        printf("  %s\n\n", ok ? "Part D OK" : "Part D FAIL");
    }

    // ---------------- Part E ----------------
    printf("## Part E. 资源账：线程数 / TMEM 列预算 / SharedStorage\n");
    {
        auto nthr = [](uint32_t nwg) { return (nwg + 1 + nwg / 4) * 128; };
        printf("  E1) get_num_threads (layout:19-21)：fp8 nwg=4 -> %u ; fp4 nwg=5 -> %u\n", nthr(4), nthr(5));
        all_ok &= (nthr(4) == 768 && nthr(5) == 896);

        printf("  E2) 真实 TMEM 列对齐 get_num_aligned_tmem_cols (utils.cuh:42-50)\n");
        uint32_t* d_out; CK(cudaMalloc(&d_out, 4));
        uint32_t bad_tmem = 0;
        // 真实设备函数逐档
        auto call_tmem = [&](uint32_t raw) -> uint32_t {
            uint32_t v = 0;
            switch (raw) {
                case 32:  tmem_align_kernel<32><<<1,1>>>(d_out); break;
                case 33:  tmem_align_kernel<33><<<1,1>>>(d_out); break;
                case 64:  tmem_align_kernel<64><<<1,1>>>(d_out); break;
                case 128: tmem_align_kernel<128><<<1,1>>>(d_out); break;
                case 256: tmem_align_kernel<256><<<1,1>>>(d_out); break;
                case 376: tmem_align_kernel<376><<<1,1>>>(d_out); break;
                case 428: tmem_align_kernel<428><<<1,1>>>(d_out); break;
                case 512: tmem_align_kernel<512><<<1,1>>>(d_out); break;
                default: return 0xffffffff;
            }
            CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&v, d_out, 4, cudaMemcpyDeviceToHost)); return v;
        };
        auto align_round = [](uint32_t raw) { return raw <= 32 ? 32 : raw <= 64 ? 64 : raw <= 128 ? 128 : raw <= 256 ? 256 : 512; };
        printf("     %-8s %-12s %-12s %s\n", "raw", "device(get_)", "model", "check");
        for (uint32_t raw : {32u, 33u, 64u, 128u, 256u, 376u, 428u, 512u}) {
            uint32_t g = call_tmem(raw), m = align_round(raw);
            bool ok = g == m; if (!ok) ++ bad_tmem;
            printf("     %-8u %-12u %-12u %s\n", raw, g, m, ok ? "OK" : "FAIL");
        }
        all_ok &= (bad_tmem == 0);
        cudaFree(d_out);
        // 各配置的 raw TMEM 列数
        printf("     各配置 kNumTmemCols = UMMA_N*5 + kNumQStages*SFQcols + kNumKVStages*(split/32)：\n");
        for (auto cfg : std::vector<std::tuple<std::string, uint32_t, uint32_t, uint32_t>>{
                 {"fp8/sb8", 512, 3, 32}, {"fp8/sb16", 512, 3, 32}, {"fp4/sb8", 640, 5, 32}, {"fp4/sb16", 640, 5, 32}}) {
            uint32_t heads = std::get<3>(cfg);
            uint32_t umma_n = kBlockQ * heads;
            uint32_t sfq_cols = align_up(kBlockQ * kNumMaxHeads, 128) / 32;
            uint32_t raw = umma_n * 5 + 2 * sfq_cols + std::get<2>(cfg) * (std::get<1>(cfg) / 32);
            printf("       %-8s heads=%u raw=%-4u aligned=%u  %s\n", std::get<0>(cfg).c_str(), heads, raw,
                   align_round(raw), raw <= 512 ? "<=512 OK" : "OVERFLOW");
            all_ok &= (raw <= 512);
        }

        printf("  E3) 每 copy warp 的 KV 块数（UMMA_M/SB）与 SF 块数（SB*4/16）\n");
        for (uint32_t SB : {8u, 16u}) {
            uint32_t kvb = 128u / SB, sfch = SB * 4u / 16u;
            bool ok = (kvb % 2 == 0) && (kvb * sfch == 32);
            all_ok &= ok;
            printf("     SB=%-3u kNumKVBlocksPerWarp=%u kNumSFChunksPerKVBlock=%u  kvb%%2==0 && kvb*sfch==32  %s\n",
                   SB, kvb, sfch, ok ? "OK" : "FAIL");
        }

        printf("  E4) SharedStorage (layout:103-138) offsetof/sizeof vs 独立字节模型\n");
        using S_fp8_16 = SharedStorage<2, 16, 512, 2, 3, 5, cutlass::float_e4m3_t>;
        using S_fp4_16 = SharedStorage<2, 16, 640, 2, 5, 5, cutlass::float_e2m1_t>;
        using S_fp8_8  = SharedStorage<2, 8, 512, 2, 3, 5, cutlass::float_e4m3_t>;
        using S_fp4_8  = SharedStorage<2, 8, 640, 2, 5, 5, cutlass::float_e2m1_t>;
        auto check_layout = [&](const char* name, uint32_t q_align, uint32_t q_bytes, uint32_t kv_bytes,
                                uint32_t sfq_bytes, uint32_t sfkv_bytes, uint32_t kvs, uint32_t split, uint32_t sb,
                                uint32_t q_off, uint32_t kv_off, uint32_t sfq_off, uint32_t sfkv_off,
                                uint32_t w_off, uint32_t kbi_off, uint32_t qb_off, uint32_t ksh_off, uint32_t sz) {
            uint32_t nblocks = split / sb;
            uint32_t off = 0, eq = 0, tot = 0;
            off = align_up(off, q_align); ++ tot; eq += (q_off == off); off += q_bytes;
            off = align_up(off, q_align); ++ tot; eq += (kv_off == off); off += kv_bytes;
            off = align_up(off, 128);     ++ tot; eq += (sfq_off == off); off += sfq_bytes;
            off = align_up(off, 128);     ++ tot; eq += (sfkv_off == off); off += sfkv_bytes;
            off = align_up(off, 128);     ++ tot; eq += (w_off == off); off += 2 * 64 * 2;
            off = align_up(off, 16);      ++ tot; eq += (kbi_off == off); off += kvs * nblocks * 8;
            off = align_up(off, 16);      ++ tot; eq += (qb_off == off); off += 2 * 12;
            off = align_up(off, 16);      ++ tot; eq += (ksh_off == off); off += kvs * 16;
            bool ok = (eq == tot);
            all_ok &= ok;
            printf("     %-8s sizeof=%-7u 8 个数据成员 offsetof 命中 %u/%u  %s\n", name, sz, eq, tot, ok ? "OK" : "FAIL");
        };
        check_layout("fp8/sb16", 8 * 128, 2 * 64 * 128, 3 * 512 * 128, 2 * 128 * 4, 3 * 512 * 4, 3, 512, 16,
                     offsetof(S_fp8_16, q), offsetof(S_fp8_16, kv), offsetof(S_fp8_16, sf_q), offsetof(S_fp8_16, sf_kv),
                     offsetof(S_fp8_16, weights), offsetof(S_fp8_16, kv_block_infos), offsetof(S_fp8_16, q_blocks),
                     offsetof(S_fp8_16, kv_split_headers), sizeof(S_fp8_16));
        check_layout("fp4/sb16", 8 * 128 / 2, 2 * 64 * 64, 5 * 640 * 64, 2 * 128 * 4, 5 * 640 * 4, 5, 640, 16,
                     offsetof(S_fp4_16, q), offsetof(S_fp4_16, kv), offsetof(S_fp4_16, sf_q), offsetof(S_fp4_16, sf_kv),
                     offsetof(S_fp4_16, weights), offsetof(S_fp4_16, kv_block_infos), offsetof(S_fp4_16, q_blocks),
                     offsetof(S_fp4_16, kv_split_headers), sizeof(S_fp4_16));
        check_layout("fp8/sb8", 8 * 128, 2 * 64 * 128, 3 * 512 * 128, 2 * 128 * 4, 3 * 512 * 4, 3, 512, 8,
                     offsetof(S_fp8_8, q), offsetof(S_fp8_8, kv), offsetof(S_fp8_8, sf_q), offsetof(S_fp8_8, sf_kv),
                     offsetof(S_fp8_8, weights), offsetof(S_fp8_8, kv_block_infos), offsetof(S_fp8_8, q_blocks),
                     offsetof(S_fp8_8, kv_split_headers), sizeof(S_fp8_8));
        check_layout("fp4/sb8", 8 * 128 / 2, 2 * 64 * 64, 5 * 640 * 64, 2 * 128 * 4, 5 * 640 * 4, 5, 640, 8,
                     offsetof(S_fp4_8, q), offsetof(S_fp4_8, kv), offsetof(S_fp4_8, sf_q), offsetof(S_fp4_8, sf_kv),
                     offsetof(S_fp4_8, weights), offsetof(S_fp4_8, kv_block_infos), offsetof(S_fp4_8, q_blocks),
                     offsetof(S_fp4_8, kv_split_headers), sizeof(S_fp4_8));
        printf("     结论：数据成员的 alignas/顺序就是共享内存布局本身（第 82 篇），host 直接 sizeof 即用。\n\n");
    }

    // ---------------- Part F ----------------
    printf("## Part F. SF warp 4x32 -> 32x4 transpose (:325-333)\n");
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
