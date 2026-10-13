# DeepEP `refs.dispatch` 的 FP8 分支：per-128 block scale 的传递契约与设备端 `sf_pack_t` 布局

> 本文对应路线图 **P105-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 115 篇。
>
> 第 65 篇拆了 DeepEP `deep_ep/utils/refs.py` 的 dispatch/combine **语义规范**，但 §2 只在
> 「SF 布局硬约束」一条 bullet 里点了 `refs.py:46-49` 的 `hidden_sf == ceil_div(hidden, 128)`，
> 没讲这条**硬约束为什么存在、SF 在真实参考实现里怎么随 token 一起搬、以及它在设备端落地成什么**。
> 第 114 篇拆了同目录 `math.py` 的**参考量化器**，指出它 `per_token_cast_to_fp8` 返回的是
> **fp32** scale（不是 packed UE8M0）。本篇把两边的接缝补上：**参考实现里那条 `(x, sf)` 元组通路，
> 到设备端 `sf_pack_t` / `TokenLayout` 到底怎么一一对应**。
>
> **可跑性**：`refs.py` / `math.py` 是纯 PyTorch（只用 torch 自带 NCCL 的 `all_to_all_single`），
> 设备端布局 `layout/ep/token.cuh` 的 `TokenLayout` 方法全部带 `__host__ __device__`、**host 可直接调用**。
> 所以两端都能在 H100 真机验证：8 卡真实 all_to_all + 独立 oracle 对拍，以及 include 真实头文件、
> host 调用真实布局方法逐字段对拍。**全部 PASS**（`harness/exp_deepep_refs_fp8.{sh,py,out.txt}` +
> `src/deepep_token_layout_sf.cu`）。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - 参考实现：`deep_ep/utils/refs.py:10-123`（dispatch 的 `(x, sf)` 分支）、`deep_ep/utils/math.py:30-56`
> - 设备布局：`deep_ep/include/deep_ep/common/compiled.cuh:59-68`、`layout/ep/token.cuh:10-143`
> - 设备 kernel：`impls/ep/dispatch.cuh:23-33/262/293-309`、`impls/ep/dispatch_copy_epilogue.cuh:16/47/146-178/304-317`
> - 契约锚点：`tests/engram/test_engram.py:19`

---

## 1. 一段话总览

MoE 用 FP8 时，dispatch 搬的不只是激活 `x [T,H]`，还有它的 **per-128 block scale** `sf`。
参考实现把它写成 **一个 `(x, sf)` 元组**：`x` 是 `[T,H]` 的 fp8、`sf` 是 `[T, ceil_div(H,128)]` 的
**4 字节单位**（测试里是 fp32）。参考实现给这条通路加了一条**它自己的**硬约束——**SF 的列数必须恰等于
`ceil_div(H,128)`**（gran-128 契约）；除此外 SF 被当作**不透明的 4 字节块**，与 token 同行、同排序、
同掩码地搬过去。

设备端把这条通路落成三件事：`sf_pack_t` 是「4 字节 SF 单位」的 union（`compiled.cuh:59-62`）；
token 缓冲的 SF 段大小 = `kNumSFPacks * sizeof(sf_pack_t)`，其中 `kNumSFPacks` 就是调用方
SF 张量的**列数**（设备不解释粒度）；参考实现取的是 `ceil_div(hidden,128)`（`tests/engram/test_engram.py:19`
的测试路径同理），于是 **参考的 `sf` 一列 = 设备的一个 `sf_pack_t` = 128 个 hidden 元素**，
`4 / 128 = 3.125%` 就是 FP8 dispatch 为缩放因子付出的固定带宽税。而设备**缓冲预留**用了更保守的
`ceil_div(hidden,32)`（gran-32 未打包的上界，`ep.hpp:269`），保证任何 recipe 都放得下。

---

## 2. FP8 元组的进入：检测与布局硬约束（`refs.py:38-49`）

```python
# Unpack SF
use_fp8 = isinstance(x, tuple)                       # :39   用「是不是元组」判 FP8
x, sf = x if use_fp8 else (x, None)                  # :40
...
if sf is not None:
    num_tokens__, hidden_sf = sf.size()              # :47
    assert num_tokens == num_tokens__                # :48
    assert hidden_sf == ceil_div(hidden, 128)        # :49   ← 唯一硬约束
```

三个「为什么」：

1. **用「元组」而不是 dtype 标志来判 FP8**（`:39`）。因为 FP8 dispatch 的输入天然是
   「激活 + 缩放」一对，`x` 单独一个张量无法表达 scale；把 `(x, sf)` 打包成元组，让
   `dispatch` 的**调用签名在 BF16 与 FP8 之间不变**（返回也是元组或单张量，`:120`）。
2. **`hidden_sf == ceil_div(hidden,128)`（`:49`）是参考实现选定的 gran-128 契约**。它把 SF 的
   **粒度假死**在 per-128：一个 SF 单位覆盖 128 个连续 hidden 元素。这个 128 来自 FP8 训练的
   block-scale 约定（DeepGEMM 的 `gran_k=128` 与 `gran_mn=128`），也正好让 SF 的列数与「hidden 除以
   128 向上取整」一一对应——**多一个、少一个都说明调用方的量化 recipe 和设备预期对不上**，直接在
   host 就拦住。注意这是**参考实现自己的约定**：设备端 `dispatch` 并不写死这个数（它取张量自己的
   `sf.size(1)`，见第 6.4 节），参考只是选了 gran-128 fp32 这一种最常见的 recipe。
3. **`sf` 本身不参与数值**。参考实现从头到尾只对 `sf` 做**搬运**（切片、cat、all_to_all），
   从不读它的值、从不做 UE8M0 打包。它把「SF 里到底装 fp32 还是 packed UE8M0」也当成调用方的自由
   ——这正是 `sf_pack_t` 用 union 的原因（第 6 节）。

---

## 3. 逐目标 rank 选 token 时，SF 与 x 同行（`refs.py:62-90`）

dispatch 逐 `dst_rank` 收集「至少有一个 top-k 专家落在该 rank」的 token（`:67` 的 `mask_to_send`），
把整行 `x` 发过去：

```python
x_to_send  = x[indices_to_send]                      # :72
sf_to_send = sf[indices_to_send] if use_fp8 else None  # :73  ← SF 用同一组下标
...
send_sf_list.append(sf_to_send)                      # :80
```

**为什么 SF 只需一次 `indices_to_send` 索引**：`sf` 的行与 `x` 的行严格对齐（第 0 维都是 token）。
所以 SF 不需要独立的路由计算——**谁能发，它的缩放因子就跟着发**，二者是同一行数据的两个部分。
这在设备端对应 `dispatch.cuh:262` 把 `kNumHiddenBytes` 与 `kNumSFPacks*sizeof(sf_pack_t)` 一起塞进
`TokenLayout`——**同一 token 结构里的两段**（第 6 节）。

`send_src_token_idx += rank_idx * num_max_tokens_per_rank`（`:88-90`）把所有副本标上来源全局索引，
SF 也随这份排序一起走，于是接收侧的顺序不变量（按 `src_token_global_idx` 升序）对 SF 同样成立。

---

## 4. 两次交换：为什么 SF 单独一次 `all_to_all_single`（`refs.py:92-110`）

```python
dist.all_to_all_single(num_recv_tokens_per_rank, num_send_tokens_per_rank)   # :94  先换计数
...
dist.all_to_all_single(recv_x, send_x, num_recv_tokens_per_rank, num_send_tokens_per_rank)   # :105
if use_fp8:
    dist.all_to_all_single(recv_sf, send_sf, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # :107
dist.all_to_all_single(recv_topk_idx, ...)          # :108
dist.all_to_all_single(recv_topk_weights, ...)      # :109
dist.all_to_all_single(recv_src_token_idx, ...)     # :110
```

要点：**SF 与 x 用同一份发/收计数、但发成两次 collective**。为什么不像 `x` 那样内含 SF（拼成
一列）？因为 `x` 是 fp8（1 字节/元素）、`sf` 是 4 字节单位，**两者 dtype 与行宽都不同**；torch 的
`all_to_all_single` 每次只能搬一个同 dtype 张量。参考实现选择了「同计数、分张量」的写法，
而不是 host 端拼接再切分——这让参考实现保持**逐字段可读**（你说得清每个 collective 搬了什么）。

设备端对应的是同一条思路的不同实现：`dispatch.cuh:288-290` 用 TMA 搬 hidden、
`:293-309` 用 `cp.async` 搬 SF（`_ca` = cache-at-all，SF 小而常驻 L2），两条路并行进同一个 smem token
缓冲。**SF 的搬运在主 kernel 里也是一条独立的路**，而不是和 hidden 混在一条 TMA 描述符里。

---

## 5. 接收侧掩码与打包回元组（`refs.py:112-120`）

```python
mask = (expert_start_idx <= recv_topk_idx) & (recv_topk_idx < expert_end_idx)   # :115
recv_topk_idx = recv_topk_idx - expert_start_idx                                # :116
recv_topk_idx.masked_fill_(~mask, -1)                                           # :117
recv_x = (recv_x, recv_sf) if use_fp8 else recv_x                               # :120
```

`recv_sf` 的形状在 `:101` 就按 `(num_recv_tokens, hidden_sf)` 分配、dtype 取 `sf.dtype`
（**原样保留**）——所以 FP8 进来、FP8 出去。`recv_topk_idx` 的本地化与 `-1` 掩码只作用于索引，
**不碰 SF**：一个 token 即使是「发给本 rank 但本 rank 只命中部分专家」，它的 SF 仍然是整行的
（SF 是 per-128 hidden 维的，与 top-k 维无关）。

---

## 6. 设备端落地：`sf_pack_t` 与 `TokenLayout` 的 SF 段

### 6.1 4 字节单位与两个对齐档（`compiled.cuh:59-68`）

```cuda
union sf_pack_t {            // :59-62
    float fp32;              //   单个 fp32 scale（gran 128，参考量化器的产物）
    int   ue8m0x4;           //   4 个 UE8M0 打包进一个 int32（gran 32，DeepGEMM 的产物）
};
constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);   // :64  = 4
constexpr int kNumTMAAlignmentBytes  = 32;                   // :67
constexpr int kNumRDMAAlignmentBytes = 64;                   // :68
```

`union` 的两个成员都是 4 字节、同址（host roundtrip 实测 `0x01020304` 原样读回）——
**「SF 单位」在设备侧就是不透明的 4 字节**。`:64` 的 `kNumAlignedSFPacks = 4` 表示 SF 的
token 维按 **4 个 pack = 16 字节**对齐（`ep.hpp:693` 用到它）。

### 6.2 token 结构：hidden / sf / metadata 三段（`token.cuh:10-60`）

```cuda
TokenLayout(num_hidden_bytes, num_sf_bytes, num_topk, with_metadata, base)   // :18
    ...
    EP_UNIFIED_ASSERT(num_hidden_bytes % kNumTMAAlignmentBytes == 0);        // :29 hidden 必须 32B 对齐

// :34-37 每 token 字节数 = 三段各自对齐到 TMA(32)，整体再对齐到 RDMA(64)
num_data_bytes = align(align(num_hidden_bytes,32) + align(num_sf_bytes,32)
                     + align(num_metadata_bytes,32), 64)

get_sf_ptr()      = base + align(num_hidden_bytes, 32)                        // :55-57
get_metadata_ptr()= get_sf_ptr() + align(num_sf_bytes, 32)                    // :59-60
```

也就是说 **SF 段紧跟在 hidden 段之后**（`get_sf_ptr` 偏移 = hidden 段对齐长度），metadata 再跟在 SF 之后。
`num_sf_bytes = kNumSFPacks * sizeof(sf_pack_t)`——`kNumSFPacks` 是调用方 SF 张量的列数，参考
路径下恰等于它的 `hidden_sf`（第 6.4 / 7 节）。

### 6.3 kernel 侧：SF 用 `cp.async` 单独搬（`dispatch.cuh:262/293-309`）

```cuda
const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t),
                                              kNumTopk, true);          // :262
...
if constexpr (kNumSFPacks > 0) {                                        // :294
    const auto gmem_src_ptr = advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t)); // :296
    constexpr auto kNumFullIters = kNumSFPacks / 32;                    // :299  32 lane 步进
    for (int k = 0; k < kNumFullIters; ++k)
        ptx::cp_async_ca(gmem_src_ptr + (k*32 + lane_idx) * sf_hidden_stride,
                         smem_dst_ptr + k*32 + lane_idx);               // :302-303
    if (kNumFullIters*32 + lane_idx < kNumSFPacks)                      // :305  尾块
        ptx::cp_async_ca(...);                                          // :306-307
}
```

落盘 epilogue 对称地在 `dispatch_copy_epilogue.cuh:146-178` 把 SF 从 smem 读回寄存器再写下游张量
（`kNumSFPacks/32` 次整 lane 循环 + 尾块），`kNumSFPacks == 0`（BF16）时整段被 `if constexpr` 消除
（`:147`）；零填充分支在 `:304-317` 写 4 字节的 `zero_sf_pack`。

### 6.4 host 侧：SF 列数取张量自己的，缓冲按 gran-32 上界预留（`csrc/buffers/ep.hpp`）

设备 kernel 的 `kNumSFPacks` 来自 host 编排层，而编排层**并不写死 gran-128**——

```cpp
// dispatch 调用点（csrc/buffers/ep.hpp:347-357）
if (sf.has_value()) {
    // SF must be FP32 or packed UE8M0x4                      // :351
    const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());  // :354
    EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));  // :355  每单位 4 字节
    num_sf_packs = num_sf_packs_;                             // :356  实际列数 = 张量给的
    ...
}

// 缓冲尺寸用 gran-32 未打包作【保守上界】（csrc/buffers/ep.hpp:269）
const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;
//                                                 ^^^ "An approximation for number of SF packs"
EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);    // :257  最坏 SF <= 主数据
```

三个「为什么」：

1. **设备对 recipe 开放，参考只是选定一种**。`dispatch` 的 SF 列数 = 调用方张量的 `sf.size(1)`
   （`:356`），只断言每个单位 4 字节（`:355`）与「必须是 FP32 或 packed UE8M0x4」（`:351` 注释）。
   所以参考的 gran-128 fp32（`ceil_div(hidden,128)`）只是**一种合法输入**；packed UE8M0 gran-32
   （第 25/26 篇）也是。
2. **缓冲按最坏情况预留**：`ceil_div(hidden,32)`（`:269`）对应「每 32 元素一个 4 字节单位」的
   未打包 gran-32 上界，是 `ceil_div(hidden,128)` 的 **4 倍**——只要最坏 SF 字节 ≤ 主数据（`:257`
   的断言），任意更细/更粗的 recipe 都塞得下同一个对称缓冲，无需按 recipe 重建。
3. **输出 SF 布局可切 TMA 列主序**（`:689-690`）：`use_tma_aligned_col_major_sf=True` 时
   `recv_sf_token_stride=1`、`recv_sf_hidden_stride=align(num_allocated_tokens, kNumAlignedSFPacks)`
   ——把 SF 直接排成**下一个 GEMM 输入**要的列主序（第 25 篇的 MN-major TMA 布局）。

---

## 7. 参考的 fp32 scale vs 设备的 packed UE8M0：同一个 4 字节

把第 6 节的 `num_sf_bytes` 与第 2 节的 `hidden_sf` 对上（**参考路径**）：

```
hidden_sf    = ceil_div(hidden, 128)          (refs.py:49)          → 每单位 1 个 fp32 scale(gran128)
kNumSFPacks  = sf.size(1)                     (csrc/buffers/ep.hpp:356)
参考路径     = ceil_div(hidden, 128)          (tests/engram/test_engram.py:19 同)
num_sf_bytes = kNumSFPacks * sizeof(sf_pack_t) = ceil_div(hidden,128) * 4
```

**一个 4 字节 SF 单位 = 128 个 hidden 元素**（gran-128 fp32 路径）。这个「128」对 union 的两个成员都成立：

- `fp32` 成员：**1 个 fp32 scale 覆盖 128 个元素**（gran 128）——这是参考量化器
  `per_token_cast_to_fp8`（`math.py:37-39` 返回 `x_amax/448`）和 DeepEP 测试路径实际传的东西。
- `ue8m0x4` 成员：**4 个 UE8M0 字节覆盖 128 个元素**（gran 32，每字节一个 scale）——
  这是 DeepGEMM 前向 GEMM 需要的 packed 形态（见第 25/26 篇）。此时 SF 列数会降到
  `ceil_div(hidden,128)`（4 个 scale 打包进 1 个 int32），仍落在缓冲预留的 `ceil_div(hidden,32)` 内。

`union` 让设备**对两种 recipe 一视同仁**：dispatch/combine 只管把 4 字节搬到位，**从不解释它**。
因此 DeepEP 里没有任何 UE8M0 打包代码——全仓唯一的 `ue8m0` 字样就是 `compiled.cuh:61` 的 union
成员与 `ep.hpp:351` 的注释；**打包是 DeepGEMM 的活**，DeepEP 只是运。这条「参考给 fp32 / 设备不解释 /
GEMM 要 packed」的分工，正是第 114 篇「`per_token_cast_back` 的 int 分支能把 packed UE8M0
重解释回 fp32」的**上游原因**。

---

## 8. H100 实测证据

脚本 `harness/exp_deepep_refs_fp8.sh`（`+ exp_deepep_refs_fp8.py` + `src/deepep_token_layout_sf.cu`，
输出 `harness/exp_deepep_refs_fp8.out.txt`）。真实源码路径加载（sha256：`refs.py=c5f551b4…
math.py=ed0dfb4b…`；DeepEP `93eb6eb`）。

**Part B —— `refs.dispatch` 的 `(x, sf)` 分支（8×H100 NCCL all_to_all，T=512, H=512, E=32, k=4）**
与**独立 oracle**（不调用 refs，按 `(src_rank, local_idx)` 升序重建 x 行 + SF 行）逐位对拍：

```
本 rank 收到 1765 行；recv_sf (1765, 4) torch.float32；hidden_sf = ceil_div(512,128) = 4
每来源 rank 计数 = [208, 224, 222, 218, 216, 231, 221, 225]
[PASS] recv_x 是 (x, sf) 元组 / recv_sf 形状 == [nrecv, ceil_div(H,128)] / recv_sf dtype 原样保留(fp32)
[PASS] recv_src == oracle / recv_topk_idx == oracle / recv_x(fp8) == oracle（逐位）/ recv_sf == oracle（逐位）
[PASS] num_recv_tokens_per_rank == oracle
[PASS] hidden_sf != ceil_div(H,128) 被 assert 拒绝          ← 坐实 refs.py:49
[PASS] per_token_cast_back(recv_x, recv_sf) == oracle 反量化 (rel=0.00e+00)   ← 参考量化器往返
```

即：**SF 与 token 同行、同排序、同掩码地搬过去**（`x`、`sf` 两条 collective 结果都对上独立 oracle），
且**用真实参考量化器反量化后与 oracle 逐位一致**。

**Part C —— 设备端 token 缓冲 SF 布局**（`src/deepep_token_layout_sf.cu` include 真实
`layout/ep/token.cuh`、host 调用真实 `TokenLayout`/`BufferLayout`；Python 独立公式复刻对拍）：

```
CONST sizeof_sf_pack_t=4 kNumAlignedSFPacks=4 kNumTMAAlignmentBytes=32 kNumRDMAAlignmentBytes=64
UNION roundtrip=16909060 (expect 16909060 == 0x01020304)
LAYOUT hidden=7168 topk=8 md=1 hidden_sf=56 num_sf_bytes=224 sf_off=7168 md_off=7392 bytes_no_mb=7552 ...
BF16   hidden=7168 num_sf_bytes=0 sf_off=14336 num_bytes_no_mb=14464
BUFFER per_token=7584 per_rank=7766016 total=62128128
→ Part C: 11/11 PASS
```

7 组 `(hidden, topk, with_metadata)`（含 DeepSeek `H=7168`、`hidden=512`、`topk=0`、无 metadata）
的 `hidden_sf / num_sf_bytes / sf_off / md_off / bytes_no_mb` 与 Python 独立公式**逐字段 0 不符**；
BF16（`num_sf_bytes=0`）时 SF 段折叠、`sf_off` 落到 hidden 之后；`TokenLayout.get_num_bytes<true>`
比 `<false>` 恰多 `align(sizeof(mbarrier),32)=32`。

**Part D —— SF 字节开销 / 参考契约 vs 设备缓冲上界 / combine 无 FP8 变体**：

```
hidden  | ref: ceil(H,128) | dev buf upper: ceil(H,32) | ratio | 4*ceil(H,32)<=H
   7168 |              56 |                    224 |   4.0 | True
  hidden  |  data/token | sf/token(ref) | sf% of data | 设备上界 sf% of data
    7168 |       7168 |       224 |      3.125% |         12.500%
[PASS] 参考 gran128 是设备缓冲上界(gran32)的 1/4，且最坏 SF 字节 <= 主数据（逐形状）
[PASS] 参考 gran128 无 padding 时 SF/数据 = 4/128 = 3.125%
[PASS] dispatch 用 sf.size(1)（ep.hpp:356）作为实际列数
[PASS] 输出 SF 可选 TMA 列主序（ep.hpp:689-690）
[PASS] refs.combine 签名无 sf/元组分支
[PASS] generate_pre_combine_data 恒返回 bf16（`:153`）
→ Part D: 6/6 PASS
```

> 即：参考实现固定 gran-128（SF 税 3.125%），而设备**缓冲**预留到 gran-32 上界（12.5%），
> 保证任何更细粒度的 recipe（含 packed UE8M0）都能塞进同一块对称缓冲——两者不是同一个数，
> 但后者严格覆盖前者。

---

## 9. 边界与下一步

- **`combine` 没有 FP8 变体**（本机实测）：`refs.combine`（`refs.py:177-180`）只收单个 `y` 张量、
  **没有 `(y, sf)` 元组分支**，`generate_pre_combine_data` 也**恒返回 bf16**（`:153`）。所以 FP8
  只在 **dispatch** 一条腿上出现；combine（以及反向的两个 expanded 相位）都是 BF16——这与第 104 篇
  「反向通信 ≈1.5× 前向」的模型一致（反向没有 FP8 去重收益）。
- **DeepEP 不做 UE8M0 打包**：全仓唯一的 `ue8m0` 字样就是 `compiled.cuh:61` 的 union 成员与
  `ep.hpp:351` 的注释（「SF must be FP32 or packed UE8M0x4」）；实际打包由 DeepGEMM 的
  `get_*_packed_ue8m0_tensor` 系列完成（第 25/26 篇）。
- **本机 `import deep_ep` 仍受阻**（NCCL 2.29.7 < 2.32.3，见路线图「阻塞」）：本篇验证的是
  **参考语义 + 布局算术**，设备端真实 `dispatch.cuh` 的端到端执行仍需 NCCL ≥ 2.32.3 环境。
- **可继续补的纯 host 语义层**：`refs.py` 的 dispatch 已覆盖 BF16 + FP8；`envs.py` 的带宽探测层
  （第 73 篇已系统拆过）。DeepEP 侧剩下的空档基本都要 NCCL 设备 API。

**参考链接**：DeepEP 仓库 <https://github.com/deepseek-ai/DeepEP>；参考文件
`deep_ep/utils/{refs,math}.py`、`deep_ep/include/deep_ep/{common/compiled.cuh,layout/ep/token.cuh,
impls/ep/dispatch.cuh,impls/ep/dispatch_copy_epilogue.cuh}`、`tests/engram/test_engram.py`。
