// P152-1 实验：DeepEP 真实 per-token 缓冲布局字节口径（直接 include 真实头文件）。
//
// 为什么 H100(sm90) 可跑：
//   * `deep_ep/layout/ep/token.cuh` 里的 `layout::TokenLayout` / `layout::BufferLayout`
//     全是 `__host__ __device__` 的纯算术（对齐、加法、指针推进），不含任何 SM100 专属指令，
//     也不依赖 NCCL / 运行时句柄；本文件**直接 include 真实头文件**编译运行。
//   * `get_dispatch_token_layout` / `get_combine_token_layout` 定义在
//     `csrc/kernels/ep/{dispatch,combine}.hpp`，那两个头 include 了 `<nccl.h>`；本实验按
//     真实函数体（dispatch.hpp:26-29、combine.hpp:19-22）**原样重述**这两行调用。
//
// 权威来源（DeepEP @ 93eb6eb，行号核对过）：
//   * `deep_ep/layout/ep/token.cuh:10-82`  TokenLayout：num_metadata_bytes 公式在 `:25-26`，
//     get_num_bytes 公式在 `:32-41`（TMA=32B、RDMA=64B 对齐）；各字段偏移 `:51-81`。
//   * `deep_ep/include/deep_ep/common/compiled.cuh:59-70`  sf_pack_t=4B（4 个 UE8M0）、
//     kNumTMAAlignmentBytes=32、kNumRDMAAlignmentBytes=64、kNumAlignedSFPacks=4。
//   * `csrc/kernels/ep/dispatch.hpp:26-29`  get_dispatch_token_layout(hidden,elem,sf_packs,topk)
//     = TokenLayout(hidden*elem, sf_packs*sizeof(sf_pack_t), topk, /*with_metadata=*/true)。
//   * `csrc/kernels/ep/combine.hpp:19-22`   get_combine_token_layout(hidden,elem,topk)
//     = TokenLayout(hidden*elem, 0, topk, /*with_metadata=*/false)。
//   * `csrc/buffers/ep.hpp:256-273`         dispatch 用 num_sf_packs=ceil_div(hidden,32)（近似）、
//     elem_size = fp8?1:2；combine 用 bf16(2B)。
//
// 编译：见 harness/exp_deepep_token_layout_bytes.sh
//   nvcc -arch=sm_90a -I$DeepEP/deep_ep/include -I/usr/local/cuda/include/cccl ...

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

#include <deep_ep/layout/ep/token.cuh>

using namespace deep_ep;

// ----------------------------------------------------------------------------------
// 复述 csrc/kernels/ep/dispatch.hpp:26-29 / combine.hpp:19-22（真实函数体）
// ----------------------------------------------------------------------------------
static layout::TokenLayout dispatch_layout(int hidden, int elem_size, int num_sf_packs, int num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * (int)sizeof(sf_pack_t), num_topk, true);
}
static layout::TokenLayout combine_layout(int hidden, int elem_size, int num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static int align(int x, int a) { return (x + a - 1) / a * a; }

struct Case { const char* name; int hidden; int num_topk; };

// 逐字段打印 dispatch 一个 token 的字节账
static void report_dispatch(const Case& c, bool fp8) {
    const int elem_size = fp8 ? 1 : 2;
    const int num_sf_packs = fp8 ? (c.hidden + 31) / 32 : 0;  // ep.hpp:269 近似公式
    auto L = dispatch_layout(c.hidden, elem_size, num_sf_packs, c.num_topk);

    const int hb = L.num_hidden_bytes;
    const int sb = L.num_sf_bytes;
    const int mb = L.num_metadata_bytes;
    // metadata 拆分（token.cuh:67-76）：topk_idx(topk*4) + topk_weights(topk*4) + src(1*4) + linked_list(topk*4)
    const int topk_idx_b = c.num_topk * (int)sizeof(int);
    const int topk_w_b = c.num_topk * (int)sizeof(float);
    const int src_b = 1 * (int)sizeof(int);
    const int ll_b = c.num_topk * (int)sizeof(int);
    const int64_t nbytes = L.get_num_bytes<true, int64_t>();  // 含 mbarrier
    const int64_t nbytes_nomb = L.get_num_bytes<false, int64_t>();

    printf("  dispatch %-4s hidden=%5d topk=%d | data=%5d  sf=%4d  meta=%3d (=idx %d + w %d + src %d + ll %d)"
           " | token_bytes=%6lld (no-mb %6lld) | TMA align data/sf/meta = %d/%d/%d\n",
           fp8 ? "FP8" : "BF16", c.hidden, c.num_topk, hb, sb, mb, topk_idx_b, topk_w_b, src_b, ll_b,
           (long long)nbytes, (long long)nbytes_nomb,
           align(hb, kNumTMAAlignmentBytes), align(sb, kNumTMAAlignmentBytes), align(mb, kNumTMAAlignmentBytes));
}

static void report_combine(const Case& c) {
    auto L = combine_layout(c.hidden, (int)sizeof(nv_bfloat16), c.num_topk);
    const int64_t nbytes = L.get_num_bytes<true, int64_t>();
    const int64_t nbytes_nomb = L.get_num_bytes<false, int64_t>();
    printf("  combine BF16 hidden=%5d topk=%d | data=%5d  sf=%d  meta=%3d (=w %d) | token_bytes=%6lld (no-mb %6lld)\n",
           c.hidden, c.num_topk, L.num_hidden_bytes, L.num_sf_bytes, L.num_metadata_bytes,
           c.num_topk * (int)sizeof(float), (long long)nbytes, (long long)nbytes_nomb);
}

int main() {
    printf("== DeepEP 真实 per-token 布局（include deep_ep/layout/ep/token.cuh）==\n");
    printf("sizeof(sf_pack_t)=%d  kNumTMAAlignmentBytes=%d  kNumRDMAAlignmentBytes=%d  kNumAlignedSFPacks=%d\n\n",
           (int)sizeof(sf_pack_t), kNumTMAAlignmentBytes, kNumRDMAAlignmentBytes, kNumAlignedSFPacks);

    std::vector<Case> cases = {
        {"DeepSeek", 7168, 2},
        {"H2048",    2048, 2},
        {"H7168-t8", 7168, 8},
        {"H512-t4",   512, 4},
    };
    for (auto& c : cases) {
        printf("[%s]\n", c.name);
        report_dispatch(c, /*fp8=*/true);
        report_dispatch(c, /*fp8=*/false);
        report_combine(c);
        printf("\n");
    }

    // 关键比值：FP8 dispatch vs BF16 dispatch 的 token 字节
    printf("== 关键比值（DeepSeek H=7168, topk=2）==\n");
    {
        Case c{"DeepSeek", 7168, 2};
        auto dfp8 = dispatch_layout(c.hidden, 1, (c.hidden + 31) / 32, c.num_topk).get_num_bytes<true, int64_t>();
        auto dbf  = dispatch_layout(c.hidden, 2, 0, c.num_topk).get_num_bytes<true, int64_t>();
        auto comb = combine_layout(c.hidden, (int)sizeof(nv_bfloat16), c.num_topk).get_num_bytes<true, int64_t>();
        printf("  dispatch FP8  token = %lld B\n", (long long)dfp8);
        printf("  dispatch BF16 token = %lld B\n", (long long)dbf);
        printf("  combine  BF16 token = %lld B\n", (long long)comb);
        printf("  BF16/FP8 dispatch   = %.4f\n", (double)dbf / (double)dfp8);
        printf("  combine/dispatch(FP8) = %.4f\n", (double)comb / (double)dfp8);
    }

    // 一致性自检：metadata 公式 token.cuh:25-26
    {
        Case c{"chk", 7168, 2};
        auto L = dispatch_layout(c.hidden, 1, (c.hidden + 31) / 32, c.num_topk);
        const int expect = c.num_topk * ((int)sizeof(int) + (int)sizeof(float)) + (1 + c.num_topk) * (int)sizeof(int);
        printf("\n[check] num_metadata_bytes=%d  expect=%d  %s\n",
               L.num_metadata_bytes, expect, L.num_metadata_bytes == expect ? "OK" : "FAIL");
        // 对齐自检：data/sf/meta 都按 TMA=32B 对齐后再整体按 RDMA=64B 对齐
        const int64_t raw = align(L.num_hidden_bytes, kNumTMAAlignmentBytes)
                          + align(L.num_sf_bytes, kNumTMAAlignmentBytes)
                          + align(L.num_metadata_bytes, kNumTMAAlignmentBytes);
        const int64_t expect_nomb = align((int)raw, kNumRDMAAlignmentBytes);
        printf("[check] get_num_bytes<false>=%lld expect=%lld  %s\n",
               (long long)L.get_num_bytes<false, int64_t>(), (long long)expect_nomb,
               L.get_num_bytes<false, int64_t>() == expect_nomb ? "OK" : "FAIL");
    }
    return 0;
}
