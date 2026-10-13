// P133-1 实验：Mega MoE **前向收尾两段**（L2 bf16 epilogue 写回 + combine 归约）在 H100(sm90)
// 上的数值语义核对与**定时预算**，补全第 142 篇（P132-1）缺的「非 GEMM 段」。
//
// 背景：第 142 篇把前向 single-step 预算算成
//   `L1 m_grouped 204.1us + L1 epilogue 24.2us + L2 m_grouped 111.1us = 339.4us`，
// 但它**只到 L2 的 GEMM 输出**——融合 kernel 里 L2 输出还要经过两段才变成 `y`：
//   1. **L2 bf16 epilogue**（`sm100_fp8_fp4_mega_moe.cuh:1254-1357`）：把 TMEM 里的 L2 输出
//      `round->bf16`（`cast_into_bf16_and_pack`，`:1311-1314`），STSM 到 smem，再经
//      `sym_buffer.map` **直写远端** combine buffer（`:1346-1351`）——这是「combine 的写端」；
//   2. **combine 归约**（`:1365-1520`）：等 peer 就绪（grid-index 水位线 `:1405-1441`），逐 chunk
//      `tma_load_1d` 拉回 top-k 贡献（`:1459`），用 `ptx::accumulate` 做 **bf16->fp32 累加**
//      （`:1485`），`__float22bfloat162_rn` 回 bf16（`:1499`），TMA store 写 `y`（`:1514`）。
//
// Mega MoE 主体是 SM100 专属（`mega_moe.hpp:367` 只在 arch_major==10 派发；UMMA/tcgen05/TMEM 在
// sm90 不可用），H100 端到端跑不了。但这两段的**数值原语与数据布局**是架构无关的：
//   - `math::cast_into_bf16_and_pack`（RNE bf16 打包，`common/math.cuh:80-84`）；
//   - `ptx::accumulate`（`ptx/utils.cuh:65-74`，sm90 回退 = bf16->fp32 普通加）；
//   - combine buffer 的**邮箱布局** `combine_token_buffer[topk(+shared)][num_max_tokens][hidden]`
//     （`layout/mega_moe.cuh:447-449` + `get_rank_buffer` `:326-333`）+ `TokenSrcMetadata{rank,token,topk}`
//     （`layout/mega_moe.cuh:43-52`，写端从 `get_token_src_metadata_ptr` 反查 `:1287-1289`）。
// 本文件把这些搬到裸 CUDA（**include 真实头文件**），做两件事：
//
//   A) **语义核对**：在一个单 rank 的 MoE 替身上完整跑「专家 L2 输出 → 邮箱 scatter（写端）→
//      combine 归约（读端）」 vs 稠密参考 `y[t]=Σ_slot v`，逐位对拍（邮箱布局 + src metadata + 归约）。
//   B) **预算**：在 DeepSeek 量级（T=2048, topk=2, H=7168）量 L2 写回 + combine 归约的用时，
//      并算出写回/读回的**线上字节**，与第 11/132 篇的远端带宽数对接，补进前向 step 预算。
//
// 编译/运行见 harness/exp_moe_forward_combine.sh。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// --- 真实头文件（架构无关部分）---
#include <deep_gemm/common/math.cuh>   // cast_into_bf16_and_pack（RNE bf16 打包）
#include <deep_gemm/ptx/utils.cuh>     // ptx::accumulate（bf16->fp32 累加，sm90 回退）

using namespace deep_gemm;

#define DEV __device__ __forceinline__
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// 邮箱布局：`combine_token_buffer[slot][token][hidden]`（`layout/mega_moe.cuh:447-449`）
//   - slot 维 = `num_topk + (num_shared_experts>0)`；每个 slot 是「该 token 第 j 个 top-k 专家
//     的计算结果」的到达缓冲（不按 rank 切；远端 rank 通过 `sym_buffer.map` 写进来）。
//   - 写端：专家 e 产出 (pool_row -> token t) 的 L2 向量 v[H]，查 `TokenSrcMetadata` 得
//     `(dst_rank, dst_token=t, dst_slot)`，写到本 rank 的 `mailbox[dst_slot][t][:]`。
//   - 读端：combine 对每个 token t 读 `mailbox[0..K)[t][:]`，bf16->fp32 累加，回 bf16 写 y[t]。
// 这里用「单 rank 占位」的 mailbox（远端由写入地址的 rank 维模拟），把两段串起来验证。
// ---------------------------------------------------------------------------
struct Mailbox {
    nv_bfloat16* base;   // [num_slots][T][H]
    int num_slots, T, H;
    __host__ __device__ nv_bfloat16* at(int slot, int token) const {
        return base + ((size_t)slot * T + token) * H;
    }
};

// ---------------------------------------------------------------------------
// 写端（L2 epilogue）的等价内核：把每个专家的 L2 输出按 src metadata scatter 进邮箱。
// 真机里这一步是 epilogue warp 从 TMEM 读回 fp32、RNE 打包成 bf16、经 NVLink 远端直写；
// 这里把「RNE 打包」用真实的 `math::cast_into_bf16_and_pack` 实现，写到 host 给定的 dst 地址。
//   l2_out[g][H]  : 专家 g 的 L2 输出（已按 (pool_row) 排列的 bf16），只对「属于本 rank 的专家」存在
//   meta[g][rows] : 每个 pool_row 的 TokenSrcMetadata（dst_rank, dst_token, dst_slot）
// 这里 rank 维折叠掉（单 rank），dst_slot 即邮箱第一维。
// ---------------------------------------------------------------------------
__global__ void l2_scatter_kernel(const float* __restrict__ l2_out_f32,  // [G][R][H] fp32（TMEM 读回）
                                  const int* __restrict__ src_token,     // [G][R] dst token（-1 为 padding）
                                  const int* __restrict__ src_slot,      // [G][R] dst slot（-1 为 padding）
                                  int G, int R, int H,
                                  nv_bfloat16* __restrict__ mailbox, int num_slots, int T) {
    // 每个 (g, r) 行由一个 warp 处理；每 lane 处理 H/32 个元素（H 为 32 的倍数）
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp >= G * R) return;
    const int g = warp / R, r = warp % R;
    const int t = src_token[g * R + r];
    if (t < 0) return;                          // padding 行（:1333 的 `m_idx_in_block >= valid_m` 跳过）
    const int slot = src_slot[g * R + r];
    const float* src = l2_out_f32 + ((size_t)g * R + r) * H;
    nv_bfloat16* dst = mailbox + ((size_t)slot * T + t) * H;
    for (int j = lane * 2; j < H; j += 64) {
        // 真机 `:1311-1314` 每 8 个 fp32 打成 4 个 int（`cast_into_bf16_and_pack`），
        // 这里逐 pair 用同一函数（值语义等价）。
        float lo = src[j], hi = src[j + 1];
        int packed = math::cast_into_bf16_and_pack(lo, hi);
        *reinterpret_cast<int*>(dst + j) = packed;
    }
}

// ---------------------------------------------------------------------------
// 读端（combine 归约）的等价内核：镜像 `:1426-1520` 的主循环。
//   - 一个 warp 处理一个 token 的 H 个元素：`kNumUint4PerLane` 个 uint4/lane（H/8/32）；
//   - 逐个 slot `ptx::accumulate` 到 fp32（`:1485`），最后 `__float22bfloat162_rn` 回 bf16（`:1499`）；
//   - 真机用 TMA 两级流水拉取；这里直接读（同一累加语义/同一字节量）。
// ---------------------------------------------------------------------------
template <uint32_t kNumUint4PerLane>
__global__ void combine_reduce_kernel(const nv_bfloat16* __restrict__ mailbox,
                                      int num_slots, int T, int H,
                                      nv_bfloat16* __restrict__ y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp >= T) return;
    const int t = warp;

    float2 reduced[kNumUint4PerLane * 4];       // uint4 = 8 bf16 = 4 个 bf16x2
    #pragma unroll
    for (uint32_t i = 0; i < kNumUint4PerLane * 4; ++i)
        reduced[i] = make_float2(0.0f, 0.0f);

    for (int s = 0; s < num_slots; ++s) {
        const nv_bfloat16* src = mailbox + ((size_t)s * T + t) * H;
        #pragma unroll
        for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
            const auto uint4_values = *reinterpret_cast<const uint4*>(src + (j * 32 + lane) * 8);
            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
            #pragma unroll
            for (uint32_t l = 0; l < 4; ++l)
                ptx::accumulate(reduced[j * 4 + l], bf16_values[l]);
        }
    }

    // Cast 回 bf16 写 y（`:1493-1507`）
    nv_bfloat16* dst = y + (size_t)t * H;
    #pragma unroll
    for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
        nv_bfloat162 packed[4];
        #pragma unroll
        for (uint32_t l = 0; l < 4; ++l)
            packed[l] = __float22bfloat162_rn(reduced[j * 4 + l]);
        *reinterpret_cast<uint4*>(dst + (j * 32 + lane) * 8) =
            *reinterpret_cast<const uint4*>(packed);
    }
}

// ---------------------------------------------------------------------------
// L2 epilogue 的「打包段」微基准：从 fp32 读、RNE 打 bf16、写 bf16（真机还要经过 smem/STSM，
// 这里只量化「每字节搬运 + 打包」的带宽上限）。真机读是 TMEM（片上），写是远端 NVLink。
// ---------------------------------------------------------------------------
__global__ void l2_pack_kernel(const float* __restrict__ in, nv_bfloat16* __restrict__ out, size_t n_pairs) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n_pairs; i += stride) {
        float lo = in[2 * i], hi = in[2 * i + 1];
        int packed = math::cast_into_bf16_and_pack(lo, hi);
        *reinterpret_cast<int*>(out + 2 * i) = packed;
    }
}

// ---------------------------------------------------------------------------
// host 参考
// ---------------------------------------------------------------------------
static double bf16r_d(double x) {
    nv_bfloat16 h = __float2bfloat16_rn((float)x);
    return (double)__bfloat162float(h);
}

int main() {
    cudaFree(0);
    std::mt19937_64 rng(20241013);
    std::normal_distribution<float> nd(0.0f, 1.5f);

    printf("==== P133-1: Mega MoE 前向收尾两段（L2 写回 + combine 归约）H100(sm90) 复刻 ====\n\n");

    // ================= A) 语义核对：scatter -> reduce vs 稠密参考 =================
    {
        const int G = 8, R = 16, H = 256, T = 24, K = 2;   // 8 个专家，各 16 个 pool_row
        // 每个 pool_row 属于一个 token 的一个 top-k 槽；token 0..T-1，topk=2。
        // 构造一个合法的分配：每个 token 恰好被 2 个专家选中。
        std::vector<int> src_token(G * R, -1), src_slot(G * R, -1);
        std::vector<std::vector<int>> owner(T);   // token -> [expert_examples]
        // 轮转分配：token t 由专家 t%G 和 (t+1)%G 处理
        std::vector<int> next_row(G, 0);
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < K; ++s) {
                int e = (t + s) % G;
                int r = next_row[e]++;
                src_token[e * R + r] = t;
                src_slot[e * R + r] = s;
            }
        }
        // 每个 (g,r) 的 L2 输出 = 某个确定性向量（每 token 不同、每 slot 不同）
        std::vector<float> l2(G * R * H);
        for (int g = 0; g < G; ++g)
            for (int r = 0; r < R; ++r) {
                int t = src_token[g * R + r], s = src_slot[g * R + r];
                for (int h = 0; h < H; ++h)
                    l2[((size_t)g * R + r) * H + h] = nd(rng) + 0.1f * (t + 1) - 0.05f * s;
            }

        // 邮箱 + device 缓冲
        std::vector<nv_bfloat16> h_mail((size_t)K * T * H, __float2bfloat16(0.0f));
        nv_bfloat16 *d_mail, *d_y; float* d_l2; int *d_st, *d_ss;
        CK(cudaMalloc(&d_mail, (size_t)K * T * H * 2));
        CK(cudaMalloc(&d_y, (size_t)T * H * 2));
        CK(cudaMalloc(&d_l2, l2.size() * 4));
        CK(cudaMalloc(&d_st, G * R * 4));
        CK(cudaMalloc(&d_ss, G * R * 4));
        CK(cudaMemcpy(d_l2, l2.data(), l2.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_st, src_token.data(), G * R * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ss, src_slot.data(), G * R * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_mail, 0, (size_t)K * T * H * 2));

        l2_scatter_kernel<<<(G * R + 7) / 8, 256>>>(d_l2, d_st, d_ss, G, R, H, d_mail, K, T);
        // 读端：H=256 -> uint4/lane = 256/8/32 = 1
        combine_reduce_kernel<1><<<(T + 7) / 8, 256>>>(d_mail, K, T, H, d_y);
        CK(cudaDeviceSynchronize());

        CK(cudaMemcpy(h_mail.data(), d_mail, (size_t)K * T * H * 2, cudaMemcpyDeviceToHost));
        std::vector<nv_bfloat16> h_y((size_t)T * H);
        CK(cudaMemcpy(h_y.data(), d_y, (size_t)T * H * 2, cudaMemcpyDeviceToHost));

        // 参考：邮箱内容应为 RNE bf16 的 expert 输出；y[t] = bf16(Σ_slot fp32(mail))
        int mail_mism = 0, y_mism = 0; double y_maxabs = 0;
        std::vector<double> ref_y((size_t)T * H, 0.0);
        for (int g = 0; g < G; ++g)
            for (int r = 0; r < R; ++r) {
                int t = src_token[g * R + r], s = src_slot[g * R + r];
                if (t < 0) continue;
                for (int h = 0; h < H; ++h) {
                    double v = (double)l2[((size_t)g * R + r) * H + h];
                    double ref_mail = bf16r_d(v);
                    double got_mail = __bfloat162float(h_mail[((size_t)s * T + t) * H + h]);
                    if (got_mail != ref_mail) ++mail_mism;          // 逐位（都是 bf16 栅格）
                    ref_y[(size_t)t * H + h] += ref_mail;           // 累加 bf16 值（与 device 的 bf16->f32 同）
                }
            }
        for (size_t i = 0; i < (size_t)T * H; ++i) {
            double ref = bf16r_d(ref_y[i]);
            double got = __bfloat162float(h_y[i]);
            if (got != ref) ++y_mism;
            y_maxabs = std::max(y_maxabs, std::fabs(got - ref));
        }
        printf("A) mailbox scatter（RNE bf16）: mismatch=%d/%d\n", mail_mism, K * T * H);
        printf("   combine reduce y[t]=Σ_slot bf16: mismatch=%d/%d  max|Δ|=%.1e\n\n",
               y_mism, T * H, y_maxabs);
        cudaFree(d_mail); cudaFree(d_y); cudaFree(d_l2); cudaFree(d_st); cudaFree(d_ss);
    }

    // ================= B) 预算：DeepSeek 量级（T=2048, topk=2, H=7168）=================
    {
        const int T = 2048, K = 2, H = 7168;
        // uint4/lane = H/8/32 = 28
        constexpr uint32_t kU4 = 7168u / 8 / 32;    // 28
        nv_bfloat16* d_mail; nv_bfloat16* d_y;
        CK(cudaMalloc(&d_mail, (size_t)K * T * H * 2));
        CK(cudaMalloc(&d_y, (size_t)T * H * 2));
        CK(cudaMemset(d_mail, 0, (size_t)K * T * H * 2));
        int blocks = (T + 7) / 8;
        for (int i = 0; i < 5; ++i) combine_reduce_kernel<kU4><<<blocks, 256>>>(d_mail, K, T, H, d_y);
        CK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        const int iters = 200;
        cudaEventRecord(t0);
        for (int i = 0; i < iters; ++i) combine_reduce_kernel<kU4><<<blocks, 256>>>(d_mail, K, T, H, d_y);
        cudaEventRecord(t1); CK(cudaDeviceSynchronize());
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;

        // 字节：读 topk 份 bf16（本地邮箱）+ 写 1 份 bf16 y
        double rd_bytes = (double)K * T * H * 2;
        double wr_bytes = (double)T * H * 2;
        printf("B) combine reduce 定时（T=%d, topk=%d, H=%d, %d uint4/lane）：%.1f us  %.1f GB/s\n",
               T, K, H, kU4, ms * 1e3, (rd_bytes + wr_bytes) / (ms * 1e-3) / 1e9);
        printf("   读 %.1f MB（topk 份邮箱）+ 写 %.1f MB（y）= %.1f MB\n\n",
               rd_bytes / 1e6, wr_bytes / 1e6, (rd_bytes + wr_bytes) / 1e6);
        cudaFree(d_mail); cudaFree(d_y);
    }

    // ================= C) L2 打包段微基准（fp32 -> RNE bf16）=================
    {
        const size_t n = 1u << 26;   // 64Mi 个 pair = 512 MiB fp32 读 / 256 MiB bf16 写
        std::vector<float> h_in(2 * (1u << 24));
        for (auto& v : h_in) v = nd(rng);
        float* d_in; nv_bfloat16* d_out;
        CK(cudaMalloc(&d_in, h_in.size() * 4));
        CK(cudaMalloc(&d_out, h_in.size() * 2));
        CK(cudaMemcpy(d_in, h_in.data(), h_in.size() * 4, cudaMemcpyHostToDevice));
        size_t pairs = h_in.size() / 2;
        for (int i = 0; i < 3; ++i) l2_pack_kernel<<<2048, 256>>>(d_in, d_out, pairs);
        CK(cudaDeviceSynchronize());
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        const int iters = 20;
        cudaEventRecord(t0);
        for (int i = 0; i < iters; ++i) l2_pack_kernel<<<2048, 256>>>(d_in, d_out, pairs);
        cudaEventRecord(t1); CK(cudaDeviceSynchronize());
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;
        double bytes = (double)pairs * 4 + (double)pairs * 2;
        printf("C) L2 pack（fp32->RNE bf16，%zu pairs）：%.1f us  %.1f GB/s（读 %.1f MB + 写 %.1f MB）\n",
               pairs, ms * 1e3, bytes / (ms * 1e-3) / 1e9,
               (double)pairs * 4 / 1e6, (double)pairs * 2 / 1e6);
        (void)n;
        cudaFree(d_in); cudaFree(d_out);
    }

    printf("\nALL DONE\n");
    return 0;
}
