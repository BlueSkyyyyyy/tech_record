// P105-1: DeepEP `refs.dispatch` FP8 分支所依赖的【token 缓冲 SF 布局】。
//
// 为什么 H100(sm90) 可跑：
//   * `deep_ep/include/deep_ep/layout/ep/token.cuh` 的 `TokenLayout` / `BufferLayout` 全部方法都带
//     `__host__ __device__`，**host 可直接调用**——本文件不启动任何 kernel，只 include 真实头文件、
//     在 host 上调真实布局方法，把 `sf_pack_t` / `kNumAlignedSFPacks` / `get_sf_ptr` 偏移 /
//     `get_num_bytes` 的整数几何打印出来，交给 Python 独立公式逐字段对拍。
//   * 它正是参考实现 `refs.py` 里那条硬约束 `hidden_sf == ceil_div(hidden, 128)`（`refs.py:46-49`）
//     在设备侧的落地：SF 每 128 个 hidden 元素一个 `sf_pack_t`（4 字节），整行 SF 再按
//     `kNumTMAAlignmentBytes=32` 对齐，整 token（hidden+sf+metadata）按 `kNumRDMAAlignmentBytes=64` 对齐。
//
// 编译：bash harness/exp_deepep_refs_fp8.sh
#include <deep_ep/layout/ep/token.cuh>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace deep_ep;

struct Case {
    int hidden_bytes;   // FP8 e4m3: 1 byte/elem
    int num_topk;
    bool with_metadata;
};

static void emit(const char* tag, const Case& c) {
    const int hidden_sf_by_128 = (c.hidden_bytes + 127) / 128;       // refs.py: ceil_div(hidden, 128)
    const int num_sf_bytes = hidden_sf_by_128 * static_cast<int>(sizeof(sf_pack_t));
    layout::TokenLayout tl(c.hidden_bytes, num_sf_bytes, c.num_topk, c.with_metadata, nullptr);

    const int sf_off = static_cast<int>(reinterpret_cast<intptr_t>(tl.get_sf_ptr()) -
                                        reinterpret_cast<intptr_t>(tl.get_base_ptr()));
    const int md_off = static_cast<int>(reinterpret_cast<intptr_t>(tl.get_metadata_ptr()) -
                                        reinterpret_cast<intptr_t>(tl.get_base_ptr()));
    const int bytes_no_mb = tl.get_num_bytes<false, int>();
    const int bytes_mb = tl.get_num_bytes<true, int>();

    printf("%s hidden=%d topk=%d md=%d hidden_sf=%d num_sf_bytes=%d sf_off=%d md_off=%d "
           "bytes_no_mb=%d bytes_mb=%d\n",
           tag, c.hidden_bytes, c.num_topk, c.with_metadata ? 1 : 0, hidden_sf_by_128,
           num_sf_bytes, sf_off, md_off, bytes_no_mb, bytes_mb);
}

int main() {
    printf("CONST sizeof_sf_pack_t=%zu kNumAlignedSFPacks=%d kNumTMAAlignmentBytes=%d "
           "kNumRDMAAlignmentBytes=%d\n",
           sizeof(sf_pack_t), kNumAlignedSFPacks, kNumTMAAlignmentBytes, kNumRDMAAlignmentBytes);

    // union 成员同址
    {
        sf_pack_t p;
        p.ue8m0x4 = 0x01020304;
        const float as_f = p.fp32;
        unsigned back;
        std::memcpy(&back, &as_f, 4);
        printf("UNION roundtrip=%u (expect 16909060 == 0x01020304)\n", back);
    }

    // 覆盖 DeepSeek 量级 / 小形状 / 非 128 倍数（hidden 只需 32 字节对齐）
    const std::vector<Case> cases = {
        {2048, 8, true},
        {4096, 8, true},
        {7168, 8, true},   // DeepSeek-V3 hidden
        {8192, 4, true},
        {512, 2, true},
        {4096, 0, true},   // topk=0
        {4096, 8, false},  // 无 metadata（无 src_token 索引）
    };
    for (const auto& c : cases) emit("LAYOUT", c);

    // BF16 token：SF 段为空（num_sf_bytes=0），看它如何折叠
    for (int hidden : {2048, 7168}) {
        layout::TokenLayout tl(hidden * 2, 0, 8, true, nullptr);
        printf("BF16 hidden=%d num_sf_bytes=0 sf_off=%d num_bytes_no_mb=%d\n", hidden,
               static_cast<int>(reinterpret_cast<intptr_t>(tl.get_sf_ptr()) -
                                reinterpret_cast<intptr_t>(tl.get_base_ptr())),
               tl.get_num_bytes<false, int>());
    }

    // BufferLayout：整 rank 缓冲 = num_max_tokens * per-token * num_ranks
    {
        const int hidden = 7168, topk = 8;
        const int num_sf_bytes = ((hidden + 127) / 128) * static_cast<int>(sizeof(sf_pack_t));
        layout::TokenLayout tl(hidden, num_sf_bytes, topk, true, nullptr);
        layout::BufferLayout<true> bl(tl, /*num_ranks=*/8, /*num_max_tokens=*/1024, nullptr);
        printf("BUFFER per_token=%lld per_rank=%lld total=%lld\n",
               static_cast<long long>(bl.get_num_bytes_per_token()),
               static_cast<long long>(bl.get_num_bytes_per_rank()),
               static_cast<long long>(bl.get_num_bytes()));
    }
    return 0;
}
