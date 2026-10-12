# BF16 Mega MoE：`bf16_mega_moe` 变体精读（H100 语义实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P10 增量（第 20 篇）。
> 前面第 2、3、5~8 篇逐段拆的是 **FP8/FP4** 版 Mega MoE（`fp8_fp4_mega_moe` /
> `sm100_fp8_fp4_mega_moe.cuh`）。但 DeepGEMM 里其实还有**第二个 Mega MoE 主 kernel**：
> `bf16_mega_moe`（`sm100_bf16_mega_moe.cuh`，1316 行），它把同一套「dispatch → L1 →
> SwiGLU → L2 → combine 单 kernel、通信计算重叠」的骨架原封不动复用，只把 **激活/权重换回
> BF16**、**去掉整条 block-scale（SF/UTCCP/UE8M0）通路的**。本篇补上这个变体的精读。
>
> **它是 SM100 专属**（`mega_moe.hpp:365-367` 只在 `arch_major == 10` 派发；UMMA
> `SM100_MMA_F16BF16_2x1SM_SS` / TMEM / tcgen05 在 sm90 都没有），H100 **端到端跑不了**，
> 因此以源码精读为主线。可跑的只有**架构无关**的部分：host 侧配置启发式、symmetric buffer
> 尺寸/切片、权重重排，以及 L1/L2 epilogue 的**数值原语**——这些在 H100 上留了真实数字。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。本文行号均在该 commit 上核对。

## 1. 为什么会有 BF16 变体

Mega MoE 的**结构性创新**（单 mega-kernel、对称显存 ring、跨 rank 前缀和认领、TMA/NVLink
重叠、warp specialization）与**数据精度**是两件正交的事。DeepGEMM 把前者做成一套骨架，后者
抽成 `MmaKind`：

```
MmaKind::BF16      -> a/b/d = cutlass::bfloat16_t        （本文）
MmaKind::MXFP8FP4  -> a = e4m3、b = fp8/fp4 解包到 8bit、含 SF/UTCCP/UE8M0（第 2/7 篇）
```

`heuristics/mega_moe.hpp:73-86` 就是这次分流：

```cpp
static MmaKind parse_mma_kind(const std::string& s) {
    if (s == "bf16xbf16") return MmaKind::BF16;
    DG_HOST_ASSERT(s == "fp8xfp4" or s == "fp8xfp8");
    return MmaKind::MXFP8FP4;
}
static int get_num_mma_elem_bytes(const MmaKind& k) { return k == MmaKind::BF16 ? 2 : 1; }
static bool is_mma_with_sf(const MmaKind& k) { return k == MmaKind::MXFP8FP4; }
```

于是 BF16 变体的取舍很直白：

- **好处**：无量化误差、无需 SF 的 UTCCP 转置与 UE8M0 打包；适合对精度敏感的 prefill/训练
  或参考对拍。
- **代价**：激活元素 2B（FP8 1B），片上/显存流量约翻倍；Tensor Core 峰值也更低；
  K 方向的 UMMA 块从 128 降到 64（`get_num_mma_elem_bytes` 直接决定 `block_k`）。

## 2. host 契约（`csrc/apis/mega_moe.hpp:285-369`）

`bf16_mega_moe` 与 `fp8_fp4_mega_moe` 的 host 入口几乎同构，差别集中在**只收 BF16 权重**、
**没有 `recipe`（无 SF 量化配方）**、**symmetric buffer 用 `"bf16xbf16"` 计算尺寸**：

- `:299` `DG_HOST_ASSERT(activation == "swiglu")`（BF16 变体只支持 SwiGLU）。
- `:302-304` `activation_clamp_opt.value_or(inf)`、`DG_HOST_ASSERT(activation_clamp >= 0)`：
  钳位对 gate 取 `min(+clamp)`、对 up 取 `clamp` 到 `[-clamp, +clamp]`（kernel `:953-957`）。
- `:308` `check_weights_layout_3d(l1_weights)` → `[E, 2I, H]`；`:309` `l2_weights` → `[E, H, I]`；
  随后 `:300` 断言 `intermediate_hidden_2 == 2 * intermediate_hidden`。
- `:313-314` `l1_weights.scalar_type() == torch::kBFloat16`（l2 同理 `:315`）——纯 BF16，无 SF 张量。
- `:352` `get_symm_buffer_size_for_mega_moe(..., "bf16xbf16", activation, num_shared_experts)`：
  注意 **`with_sf=false`**，切片里 `x_sf/l1_acts_sf/l2_acts_sf` 全是 `None`（§3.2 实测）。
- `:358` `sym_buffer.nbytes() >= num_required_bytes`；`:362-366` `slice(sym_buffer)` 取出
  `x / topk_idx / topk_weights / l1_acts / l2_acts`（BF16）。
- `:367-369` 唯有 `arch_major == 10` 才调 `sm100_bf16_mega_moe(...)`，否则 `DG_HOST_UNREACHABLE`。
  这就是「**SM100 专属，仅分析**」的直接证据。

shared expert 分支（`:318-337`）额外检查 2D 权重形状：`shared_l1 = [2·shared_I, H]`、
`shared_l2 = [H, shared_I]`，且 `shared_intermediate_hidden % intermediate_hidden == 0`
（shared 专家数由隐藏维整除推出）。

## 3. 配置启发式：与 FP8 变体的差异（H100 实测）

### 3.1 block config

`get_block_config_for_mega_moe`（`heuristics/mega_moe.hpp:88-127`）里唯一与 `mma_kind` 相关的
是 **`block_k` 与 `store_block_m`**：

```cpp
const int store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 :
                             mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;   // :113
const int store_block_m_l2 = (block_m == 240 and mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1; // :115
const int block_k = 128 / get_num_mma_elem_bytes(mma_kind);              // :117
```

即 **BF16 `block_k = 64`、`store_block_m` 在 `block_m=240` 取 40（FP8/FP4 取 24）；FP8/FP4
`block_k = 128`**。而 `block_m` 的公式（`:90-104`，基于「每 token 期望专家数 + √余量」和
「每个 M block 大约重载 128 行权重」的成本模型）**完全不含 `mma_kind`**，所以两种精度
**选出的 `block_m` 相同**。

H100 实测（`harness/exp_bf16_mega_moe.py`，用 `deep_gemm.get_block_m_for_mega_moe` 对拍复刻公式）：

```
R=8 E=64 T=8   K=2 : block_m API(bf16/fp8)=16/16  ; BF16 store_l1/l2=8/8   block_k=64 ; FP8 block_k=128
R=8 E=64 T=256 K=2 : block_m API(bf16/fp8)=128/128; BF16 store_l1/l2=32/32 block_k=64 ; FP8 block_k=128
R=8 E=64 T=4096 K=2: block_m API(bf16/fp8)=240/240; BF16 store_l1/l2=40/40 block_k=64 ; FP8 block_k=128
R=4 E=128 T=4096 K=4:block_m API(bf16/fp8)=192/192; BF16 store_l1/l2=32/32 block_k=64 ; FP8 block_k=128
（13 组形状 block_m 全部一致，仅 block_k / store_block_m 分化）
```

拉取块粒度 `num_bytes_per_pull`（`heuristics/mega_moe.hpp:223-229`，把 `hidden*elem_bytes`
反复除 2 到上限内，BF16 上限 4096B、FP8 上限 8192B）实测：

```
hidden=7168: BF16=3584B(cap 4096) FP8=3584B(cap 8192)
hidden=4096: BF16=4096B FP8=4096B
hidden=2048: BF16=4096B FP8=2048B
```

有趣的是 7168 下两者恰好都是 3584B：BF16 的 2× 元素被更小的 cap 抵消了一部分。

### 3.2 symmetric buffer 尺寸与切片

`get_symm_buffer_size_for_mega_moe`（`mega_moe.hpp:38-133`）先按 `kCandidateBlockM` 全枚举
求最大 ring 尺寸，再用 `layout::MegaMoEBuffer(..., with_sf, num_shared_experts)` 求字节。
BF16 的关键差异是**没有 SF 缓冲**（`with_sf = is_mma_with_sf(BF16) = false`，`:76`），
但激活元素翻倍。H100 实测（`R=8,E=64,maxT=4096,K=8,H=7168,I=2048`）：

| 配置 | bytes | 相对 FP8 | `x` dtype | `x_sf` | `l1_acts` | `l1_sf` |
|---|---|---|---|---|---|---|
| `bf16xbf16` | **2713.5 MB** | 1.265× | `bfloat16` | None | `(117120, 7168)` bf16 | None |
| `fp8xfp4` | 2145.3 MB | 1× | `float8_e4m3fn` | int32 `(229376)` | `(117120, 7168)` fp8 | `(1873920, 56)` int32 |

带 1 个 shared expert 时 BF16 2863.0MB vs FP8 2287.6MB（1.252×）。**BF16 约多 25~26%**：
激活从 1B 涨到 2B，而被去掉的 SF 缓冲（`l1_sf` 达 1.87M×56 int32）相对激活总量只是零头。
切片对象的 `x_sf=None` 也直接证明整条 SF 通路在 BF16 下不存在。

### 3.3 权重重排（`deep_gemm/mega/__init__.py:146-163`）

BF16 分支只做 L1 gate/up 的 gran-8 交错，L2 原样返回：

```python
if isinstance(l1_weights, tuple):
    ...  # FP8/FP4：权重+SF 都交错，SF 还要 UTCCP 转置
else:
    l1_transformed = _interleave_weights(l1_weights)   # BF16
    l2_transformed = l2_weights                        # 原样
```

H100 实测：3 组形状（含 `E=1` / `E=8`）交错与纯 torch 复刻 `diff=0.0`、反交错还原
`diff=0.0`、`l2` 返回**同一对象**（`data_ptr` 相等）、交错是纯置换（排序后集合逐位相等）；
2D 单专家输入走 squeeze 分支形状不变。作为对照，FP8 分支额外把 SF「先交错再做 4×32 的
UTCCP 转置」，BF16 无此步骤。

## 4. kernel 结构总览（`sm100_bf16_mega_moe.cuh`）

### 4.1 模板与常量

- `:21-51` 模板参数与 FP8 版同名同序（`kNumMaxTokensPerRank/Hidden/IntermediateHidden/...`、
  `BLOCK_M/N/K`、`STORE_BLOCK_M`、`kNumRingTokens/Stages`、三类线程数、`kActivationClamp`、
  `kFastMath`）；`kHasShared = kNumSharedExperts > 0`。
- `:68` `#if __CUDA_ARCH__ >= 1000` 整段包住，否则仅 `DG_DEVICE_ASSERT` 报「only support sm_100f」。
- `:105-113` `layout::MegaMoEBuffer(..., /*with_sf=*/ false, kNumSharedExperts)`——与 host 一致。
- `:119-121` `a_dtype_t / b_dtype_t / d_dtype_t = cutlass::bfloat16_t`（**三处同类型**，
  FP8 版这里是 `e4m3` 输入 + `fp8/fp4` 权重）。
- `:125-133` `LAYOUT_AD_M=128`、`UMMA_M=256`、`UMMA_N=BLOCK_M`（AB 互换）、`UMMA_BLOCK_K=64`、
  `UMMA_K=16`、`LOAD_BLOCK_M=BLOCK_M/2`（A 侧 multicast）。
- `:136-139` swizzle 全 128B；`:154-155` **L1 输出块宽 `L1_OUT_BLOCK_N = BLOCK_N/2`**
  （SwiGLU 把 gate/up 合并成一半），L2 输出无 TMA（直接 NVLink 写远端）。
- `:158-160` TMEM 只放累加器：`kNumAccumTmemCols = UMMA_N * kNumEpilogueStages`，
  对齐后 `kNumTmemCols`（FP8 版还要多出 SFA/SFB 的 TMEM 列）。

### 4.2 mbarrier 初始化（`:196-240`）

与 FP8 版几乎逐行相同：dispatch barrier `init(1)`（`:209`）、GEMM `full.init(2*2)` /
`empty.init(1)`（`:217-218`）、TMEM `tmem_full.init(1)` / `tmem_empty.init(2*kNumEpilogueThreads)`
（`:223-225`）、combine `init(1)`（`:228-229`）、task 槽 full/empty（`:231-234`）。细节见第 6 篇。

### 4.3 唯一的 MMA 路径（`:730-808`）

BF16 变体**没有**「routed FP8/FP4 与 shared FP8 两套 `block-scaled` descriptor」，只有一条 UMMA：

```cpp
auto instr_desc = cute::UMMA::make_instr_desc<b_dtype_t, a_dtype_t, float, UMMA_M, UMMA_N,
                                              Major::K, Major::K>();            // :734
...
ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(
    b_desc, a_desc, accum_stage_idx * UMMA_N,
    k_block_idx > 0 or umma_k_block_idx > 0 or k > 0,          // 首条非累加=清零
    static_cast<uint64_t>(static_cast<uint32_t>(instr_desc)) << 32);   // :795-798
```

对比 FP8 版 `:841-920`：那边每条 `umma_k_block` 之前要先 `SM100_UTCCP_4x32dp128bit_2cta::copy`
把 SFA/SFB 从 smem 搬进 TMEM（`:894-907`），再用 `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`
（`:918`）。BF16 里**这些全消失了**——只有裸 F16BF16 UMMA，内层 `UMMA_BLOCK_K/UMMA_K = 64/16 = 4`
条 `fma` 累加（`:787-800`）。`update_instr_desc_with_umma_n`（`:754`）按有效 M 动态缩 N。

## 5. L1 epilogue：in-place SwiGLU（`:883-1027`）

这是 BF16 与 FP8 变体**数值上差异最大**的一段。FP8 版 L1 epilogue 要把 SwiGLU 结果
**量化成 FP8 + UE8M0 SF** 再上屏（因为 L2 GEMM 吃 FP8）；BF16 版**直接产出 BF16**：

```cpp
// :944-972
auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k*4],   fp32_values[k*4+1]));
auto bf16_up   = __float22bfloat162_rn(make_float2(fp32_values[k*4+2], fp32_values[k*4+3]));
if (clamp != inf) {  // :953-957
    bf16_gate = __hmin2(bf16_gate, {clamp, clamp});
    bf16_up   = __hmax2(bf16_up, {-clamp,-clamp});
    bf16_up   = __hmin2(bf16_up, {clamp, clamp});
}
auto gate = __bfloat1622float2(bf16_gate);
auto neg_gate_exp = { kFastMath ? __expf(-gate.x) : expf(-gate.x), ... };   // :961-963
const auto denom = 1 + neg_gate_exp;
gate = kFastMath ? gate * fast_rcp(denom) : gate / denom;                   // :965-969
bf16x2_output = __float22bfloat162_rn(gate * up * weights);                 // :971-972
```

几个「为什么」：

- **gate/up 先 round 到 bf16 再做 silu**（`:949-950`）：因为 L1 权重的 gran-8 交错
  （`_interleave_weights`）让 TMEM 里的 gate/up 成对读回，而 L2 GEMM 吃的是 BF16 ——
  所以 gate/up 一旦被量化成 bf16 就**不可再复原**，SwiGLU 必须在 bf16 输入上算。这个
  「bf16 舍入 = 数值边界」和 mHC 里 `new_residual` 的 bf16 边界（第 13 篇）是同一思路。
- **top-k 权重在 fp32 上乘**（`:972` 的 `weights` 是 fp32，`:918-927` 每 32 token
  从 `l1_topk_weights_buffer` 读一次到寄存器缓存）；乘积最终 round 到 bf16 上屏。
- **clamp 只作用在 gate/up**（不是输出），为 SwiGLU 的指数提供数值护栏。
- **`kFastMath`**：`__expf` + `rcp.approx.ftz.f32`（`math.cuh:68-72`）替代 `expf` / 除法。

STSM（`:989-993`）把 bf16 写进 `smem_d.l1` 的 bank-conflict-free 布局，再由 warp0 发
`SM90_TMA_STORE_2D`（`:1002-1007`）写到 ring 里的 `l2_acts`（就是 L2 的输入）。之后
`:1014-1027` 用 `tma_store_wait<0>` 排空、`sync_aligned(kNumEpilogueThreads, ...)`、
再 `L2KBlockDependency::arrive` 置位 L1→L2 的 K-block 位图 + `red_add` L1 空计数。

**H100 数值实测**（`src/bf16_mega_moe_epilogue.cu`，2M 个 gate/up 对，与 fp64 参考对拍）：

```
accurate vs fp64(from fp32)  max_abs=1.059e+00  max_rel=5.923e-02   # 含 gate/up 的 bf16 舍入
accurate vs fp64(bf16-in)    max_abs=4.932e-01  max_rel=3.891e-03   # 只剩输出 bf16 舍入(≈2^-8)
fast_math vs accurate bf16   max_abs=1.250e-01  max_rel=7.042e-03   # __expf+rcp.approx 引入
clamp=2.0：gate>clamp 样本 1060065 个、up<-clamp 样本 1059746 个（已按 :953-957 钳位）
```

读法：最大相对 5.9e-2 几乎全来自 **gate/up 的 bf16 舍入**（把输入也 round 到 bf16 后误差
降到 3.9e-3，正好是输出 bf16 的 1 ulp 量级 2^-8）；`fast_math` 再叠加 ~7e-3。
这条链路「**bf16 in → fp32 silu → bf16 out**」的误差结构，与 FP8 版「bf16 in → UE8M0 量化
→ fp8 out」形成对照（第 7 篇实测 UE8M0 幂次 scale、0 溢出）。

## 6. L2 epilogue：BF16 直写远端 combine（`:1029-1144`）

L2 阶段（`else` 分支）把 TMEM 里的 fp32 累加读回、`cast_into_bf16_and_pack` 成 bf16、
经共享内存摆好 swizzle 布局后**直接 `sym_buffer.map` 写到目标 rank 的 combine buffer**
（`:1133-1138`）——中间**不落显存**。这一段与 FP8 版几乎逐行相同（FP8 版 `:1346-1351`），
因为两个变体的 **kernel 输出 `y` 都是 BF16**，combine 语义完全一致。

关键点：
- `:1053-1062` 提前读本 warp 负责行的 `TokenSrcMetadata`（`dst_rank/token/topk`），
  与 TMEM load 重叠以掩盖延迟。
- `:1097-1103` `SM90_U32x4_STSM_T` 写 smem；`:1130-1138` `ld_shared` 出 float4、
  `sym_buffer.map` 把地址映到远端 rank、`*dst = packed` 是**普通全局写**（NVLink）。
- 对 shared 专家，目标元数据直接构造为「本 rank 当前池位置」（`:1059-1061`），不查远端。

**H100 数值实测**：`cast_into_bf16_and_pack`（`common/math.cuh:81-84` →
`__float22bfloat162_rn`，`:73-77`）在 2,097,152 个 bf16 对上 **vs round-to-nearest-even
不符 0、vs 截断 不符 1,571,586**——证明是 RNE 舍入，不是截断。这与第 16 篇 TF32「实测 RNE」
是同一种「别猜舍入模式，实测」的套路。

## 7. combine 段（`:1152-1308`）

L2 epilogue 之后同一批 epilogue warp 继续做 combine：按 token×chunk 网格
（`:1214` `for chunk: epilogue_warp_idx*kNumSMs+sm_idx ... += kNumSMs*kNumEpilogueWarps`）
拉取该 token 的 top-k（+shared）贡献。与 FP8 版一致：

- `:1200-1203` 一个 `grid_sync` 保证本地 L2 写全部落完；
- `:1206-1207` SM0 把本 launch 的 `grid_idx` `st_rel_sys` 推到 peer 的 readiness 槽；
- `:1224-1229` 等所选专家的 rank 都到达该水位（`ld_acq_sys == grid_idx`）；
- `:1235-1278` 双缓冲 TMA 拉取 + **fp32 寄存器累加**（`float2 reduced[...]`，`ptx::accumulate`）；
- `:1280-1307` round 到 bf16、`tma_store_1d` 写回 `y`。

这一段的 chunk 数、`kNumHiddenBytes = kHidden*sizeof(nv_bfloat16)`（`:1154`）与 FP8 版相同
（输出都是 bf16），因此可视为**逐行复用**。

## 8. 与 FP8/FP4 变体对照

| 维度 | BF16（本文） | FP8/FP4（第 2/3/5~8 篇） |
|---|---|---|
| 输入/权重 dtype | `bfloat16`（`kernel:119-121`） | `e4m3` 激活 + `e4m3/e2m1` 权重 |
| `block_k` | **64**（`heuristics:117`） | **128** |
| UMMA 块 K | `UMMA_BLOCK_K=64` / `UMMA_K=16`（`:128-129`） | `128` / `16`（`sm100_fp8_fp4_mega_moe.cuh:163`） |
| MMA | `SM100_MMA_F16BF16_2x1SM_SS`（`:795`） | `SM100_MMA_MXF8F6F4_2x1SM_SS`（`sm100_fp8_fp4_mega_moe.cuh:918`） |
| block-scale（SF） | **无**（`with_sf=false`） | SFA/SFB + UTCCP 进 TMEM |
| L1 epilogue 输出 | **BF16**（`:971-972`） | FP8 + UE8M0 量化 + SF 打包 |
| L1→L2 依赖位图 | `L2KBlockDependency`（`:116/1021`） | 同 |
| L2 输出→combine | BF16 直写远端（`:1133-1138`） | 同 |
| combine 归约 | fp32（`:1259-1278`） | 同 |
| symmetric buffer | 大 ~25%（§3.2） | 基准 |
| 派发 | `arch_major==10`（`mega_moe.hpp:367`） | 同 |

一句话：**BF16 变体 = FP8/FP4 骨架 −（SF/UTCCP/UE8M0 通路）+（更宽的 bf16 数据路径）**。
通信、调度、ring、combine、warp specialization 完全共用；差异全部落在「数据怎么进 MMA」
和「L1 epilogue 产出什么」。

## 9. H100 实测证据台账（本轮）

| 项 | 结果 | 命令/文件 |
|---|---|---|
| `block_m` 启发式 BF16 vs FP8 | 13 组形状**全部一致**，仅 `block_k` 64/128 分化 | `harness/exp_bf16_mega_moe.py` |
| `num_bytes_per_pull` | 7168: 3584/3584；4096: 4096/4096；2048: 4096/2048 | 同上 |
| symm buffer 尺寸 | BF16 2713.5MB vs FP8 2145.3MB（**1.265×**）；`x_sf=None` | 同上 |
| 权重交错（BF16） | interleave/recover diff **0.0**，l2 同对象，纯置换 | 同上 |
| FP8 分支 SF | 确有 UTCCP 4×32 转置（`_transpose_sf_for_utccp`） | 同上 |
| L1 SwiGLU 语义 | bf16-in 后 vs fp64 rel **3.89e-3**；fast-math 叠加 7.0e-3；clamp 生效 | `src/bf16_mega_moe_epilogue.cu` |
| L2 cast RNE | 2.1M 对 vs RNE 不符 **0**，vs 截断 1.57M | 同上 |

原始输出：`harness/exp_bf16_mega_moe.out.txt`。

## 10. 结论与边界

- DeepGEMM 的 Mega MoE 是「**一套骨架 + 两种 MmaKind**」；第 2/3/5~8 篇讲的是骨架与 FP8/FP4
  特化，本篇补齐 BF16 变体。BF16 变体在 host 契约、调度、通信上完全复用，只在精度通路上做减法。
- **SM100 专属**：BF16 变体同样只在 `arch_major==10` 派发，H100 端到端不可跑。本轮能坐实的
  只有 host 侧配置/布局（真实数字）与 epilogue 数值原语（faithful 移植 + 独立 fp64 参考）。
- 数值上，BF16 变体的 L1 链路是「**bf16 in → fp32 silu → bf16 out**」，误差结构清晰
  （输入舍入 + 输出 1 ulp + fast-math），与 FP8 的量化误差是不同来源、可按需选择。
- 后续若继续本专题：可把 `transform_sf_into_required_layout`（通用 SF 布局）、
  `get_paged_mqa_logits_metadata` 的负载均衡调度、或 DeepEP 运行期实验补上（见 ROADMAP）。
