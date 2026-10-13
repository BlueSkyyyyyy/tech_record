# 105. Mega MoE 的 FP4 路由权重路径：`fp8xfp4` / `kPackedFP4` 打包表示、SMEM 解包与 H100 数值对拍

> 本专题第 3/9/10/11 篇逐段精读了 SM100 主 kernel `sm100_fp8_fp4_mega_moe.cuh`，第 20 篇拆了它的
> **BF16 变体**，第 35 篇拆了 SM100 稠密 GEMM 的 MXFP4/MXFP8，第 37 篇拆了参考量化器，第 96 篇拆了
> 它的 JIT 启动包装层。但**同一 kernel 的第二个 dtype 形态——路由权重走 FP4（`fp8xfp4`）**——一直只在
> 各处被一句「打包因子 2」「packed FP4」带过，从没被单独讲清：**FP4 权重到底以什么张量表示、host 怎么
> 断言、TMA/SMEM 怎么把 4-bit 变成能喂 UMMA 的形式、它和 FP8 权重共用哪些代码、又为什么共用、
> 精度/显存代价是多少**。本篇补上。
>
> **重要**：Mega MoE 主 kernel 是 **SM100（目标卡）专属**，本机 H100（sm90）跑不了端到端；主 kernel 的
> MMA/TMA 路径**仅分析**。但 FP4 的**张量表示契约、权重重排、参考量化、host 配置**全是架构无关的
> host + 张量语义，可以在 H100 上用真实 DeepGEMM API 直接核对——本篇给出真机数字。
>
> 目标卡代号敏感：全文用「目标卡」指代。

参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1），本机 `kernel_lab`（H100 sm90）。
实测脚本：`harness/exp_fp4_mega_moe_weights.{py,sh}`，原始输出 `harness/exp_fp4_mega_moe_weights.out.txt`
（`ALL_OK = True`）。

---

## 1. 一句话与全景

Mega MoE 的 FP8/FP4 kernel 是**同一个 kernel**：`mma_type` 字符串只有两个真值——`"bf16xbf16"` 走另一
个 kernel（第 20 篇），而 `"fp8xfp8"` 与 `"fp8xfp4"` **都映射到 `MmaKind::MXFP8FP4`**
（`heuristics/mega_moe.hpp:73-78`）：

```cpp
// heuristics/mega_moe.hpp:73-78
static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    DG_HOST_ASSERT(mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8");
    return MmaKind::MXFP8FP4;
}
```

也就是说，**FP4 与 FP8 的全部差异只落在「路由权重张量的 dtype」这一件事上**：激活恒为 FP8（e4m3）、
共享专家权重恒为 FP8、块缩放（SF）恒为 MN-major + packed UE8M0（gran 32）。所有分块、SF 粒度、
流水级数、线程划分、拉取粒度、通信缓冲尺寸都**逐字段相同**。真机 API 直接证实了这一点：

```
R=8 E=64 S=0:  fp8xfp8=2145.3MB  fp8xfp4=2145.3MB  bf16=2713.5MB   (fp8xfp4==fp8xfp8: True)
R=8 E=256 S=1: fp8xfp8=2287.6MB  fp8xfp4=2287.6MB  bf16=2863.0MB   (fp8xfp4==fp8xfp4: True)
R=8 E=8 T=8192 K=2:  block_m fp8xfp8=240  fp8xfp4=240  bf16=240
```

对称缓冲尺寸只装激活/SF（权重不在里面），FP4 与 FP8 因此逐字节相等；BF16 因激活 2B 而更大。

涉及的文件坐标：

| 位置 | 作用 |
| --- | --- |
| `csrc/apis/mega_moe.hpp:155-283` | host 入口、dtype/SF/recipe 契约、`"fp8xfp8"/"fp8xfp4"` 字符串 |
| `csrc/utils/math.hpp:11` | `kPackedFP4 = torch::kInt8` |
| `csrc/utils/layout.hpp:57-73,164-170` | 逻辑形状（FP4 K 翻倍）、权重形状 |
| `csrc/jit_kernels/impls/runtime_utils.hpp:63-100,311-323` | dtype→模板串、TMA dtype、权重 TMA 描述符 |
| `csrc/jit_kernels/heuristics/mega_moe.hpp:80-127,214-229` | 元素字节、store block、block_k、拉取粒度 |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:152-164,803-810,918` | 权重 dtype 分支、TMA 期望字节、MMA |
| `deep_gemm/mega/__init__.py:112-164` | 权重重排（FP8/FP4 共用 tuple 路径） |
| `deep_gemm/utils/math.py:99-167` | 参考 FP4 量化/打包/反量化 |
| `tests/test_mega_moe.py:102-119` | 官方测试怎么造 FP4 权重 |

---

## 2. FP4 的打包表示：`kPackedFP4` = int8、逻辑 K 翻倍

DeepGEMM 从不给 FP4 造一个「4-bit 张量」；FP4 权重的**物理存储就是一个 `int8` 张量**，每字节装 2 个
e2m1 码（低半字节在前）。这体现在三条契约上：

**(1) `kPackedFP4` 就是 `torch::kInt8`**（`csrc/utils/math.hpp:11`），所以 host 侧判断「这是不是 FP4」
用的是 `scalar_type() == kPackedFP4`。

**(2) 逻辑形状要把末维（K）翻倍**（`csrc/utils/layout.hpp:57-64`）：

```cpp
// csrc/utils/layout.hpp:57-64
template <int N>
static auto get_logical_shape(const torch::Tensor& t) {
    auto shape = get_shape<N>(t);
    if (t.scalar_type() == kPackedFP4)
        std::get<N - 1>(shape) *= 2;   // 物理末维是 K/2，逻辑 K 要 ×2
    return shape;
}
```

`check_weights_layout_3d`（`:164-170`）就用它把 `[E, N, K/2]` 的 int8 还原成逻辑 `(E, N, K)`。真机核对：

```
m=128 n=2048: packed=(128, 1024) dtype=torch.int8  logical_K=2048  roundtrip_bit_exact=True
m=64  n=7168: packed=(64, 3584)  dtype=torch.int8  logical_K=7168  roundtrip_bit_exact=True
```

**(3) 同一个「×2」在 TMA 描述符里再出现一次**（`runtime_utils.hpp:301,316`）：

```cpp
const auto k = static_cast<int>(t.size(-1)) * (t.scalar_type() == kPackedFP4 ? 2 : 1);
```

即：从权重张量的**最后一维**看，FP4 的物理 K 是 `K/2`，逻辑 K 用它 ×2 恢复。**这也是为什么 FP4 权重
不需要任何特殊的形状断言**——`check_weights_layout_3d`、`get_logical_shape`、TMA 描述符三处都把「×2」
收进了同一个概念里。

### 2.1 host 契约（`mega_moe.hpp:155-283`）

```cpp
// csrc/apis/mega_moe.hpp:176-204（节选）
const auto [rm, rn, rk] = recipe;
DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);                 // recipe 钉死 (1,1,32)
const auto weight_dtype = l1_weights.scalar_type();
DG_HOST_ASSERT(weight_dtype == torch::kFloat8_e4m3fn or weight_dtype == kPackedFP4);
DG_HOST_ASSERT(l2_weights.scalar_type() == weight_dtype);        // L1/L2 必须同 dtype
...
constexpr int kGranMN = 1, kGranK = 32;
check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                num_experts_per_rank, true, false, torch::kInt); // SF 必须是 int32（packed UE8M0）
```

四个硬约束一次到位：**recipe 恒 `(1,1,32)`**（FP4 与 FP8 都用 gran-32）、**L1/L2 同 dtype**、
**权重 SF 是 `torch::kInt`（packed UE8M0）**、**路由权重 FP4 时共享专家仍必须 FP8**（`:222-223` 单独断言
`shared_*_weights.scalar_type() == torch::kFloat8_e4m3fn`）。最后一条正是 §1「差异只落在路由权重」的
代码依据。

---

## 3. 从 global 到 SMEM：TMA 的「解包」与 UMMA 的混合精度

这是整篇最「为什么」的一段。FP4 权重从显存到 tensor core，跨了三层不同的表示：

### 3.1 全局内存：2 个 e2m1 挤在 1 字节

见 §2。打包/解包由参考实现给出（`deep_gemm/utils/math.py:84-141`，第 37 篇已拆）：`_quantize_to_fp4_e2m1`
按 `{0,0.5,1,1.5,2,3,4,6}` 的 e2m1 格点做 round-half-even，低 4 位在前；`per_token_cast_to_fp4`
（`:99-125`）先按 gran-32 每行取 amax、`ceil_to_ue8m0` 得 2 的幂 scale，再量化打包；`transpose_packed_fp4`
（`:128-141`）在保持 4-bit 码不变的前提下交换打包轴（用于权重按 K-major 的排布）。

### 3.2 TMA：`16U4_ALIGN16B` 把 4-bit 摊成可寻址形式

TMA 描述符的 dtype 选择在 `aten_dtype_to_tensor_map_dtype`（`runtime_utils.hpp:85-100`）：

```cpp
case kPackedFP4: return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                                          : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
```

Mega MoE 走的是 **`fp4_unpacked_smem = true`** 分支（默认值，见 `to_string`/`make_tma_*` 的默认参数
`runtime_utils.hpp:63,125`）。`16U4` 表示元素是 4-bit、数据是「unsigned 4-bit 打包」；`ALIGN16B` 让
每个输出元素按 16 字节边界对齐排进 SMEM，从而让后续 UMMA 把每个 FP4 当作**可以按字节寻址的操作数**读
——这就是 `float_e2m1_unpacksmem_t`（"unpack in smem"）这个名字的来历。启发式文件里也有一句原话
（`heuristics/mega_moe.hpp:214`）：

```cpp
// NOTES: FP8 activations, FP8 weights, and FP4 weights unpacked to 8-bit in smem all use 128B swizzle
```

**关键结论**：SMEM 里 FP4 权重是「每元素 1 字节」的未打包形式（`get_num_mma_elem_bytes(MXFP8FP4)==1`，
`heuristics:80-82`；`smem_b_size_per_stage = block_n*block_k*1`，`:180`），TMA 负责在搬运时把 4-bit
摊开。因此 **FP4 权重 tile 的 SMEM 占用与 FP8 完全相同**，流水级数不变；只有真正落到 DRAM↔SMEM 线上的
字节数减半。

### 3.3 UMMA：为什么要用 `MXF8F6F4` 而不是 `MXF4`

kernel 里唯一的 MMA 指令是（`sm100_fp8_fp4_mega_moe.cuh:918`）：

```cpp
ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(b_desc, a_desc, ...);
```

它**对 FP8 和 FP4 权重用同一条 MMA**，`UMMA_K = 32`（`:164`），内层循环 `UMMA_BLOCK_K / UMMA_K = 4` 条
（`:163,911`）。对比第 35 篇的稠密 SM100 kernel：那里有 **MXF4**（`UMMA_K = 64`）与 **MXF8F6F4**
（`UMMA_K = 32`）两种，是否走 MXF4 由 `is_mxf4_mma()`（两个操作数都是 packed e2m1）决定
（`heuristics/config.hpp:36`）。Mega MoE 里 **A 是 FP8 激活、B 是 FP4 权重**——**不是**「两侧都 e2m1」，
所以 MXF4 不可用，只能落在 `MXF8F6F4` 家族（它涵盖 E4M3/E5M2/E3M2/E2M1 全部尺寸）。这也解释了为什么
kernel 用 `float_e2m1_unpacksmem_t` 而非 `float_e2m1_t`：**MXF8F6F4 读的是 unpacked 操作数**，而
`float_e2m1_t` 是给 MXF4 的 packed 操作数用的（`to_string` 的 `fp4_unpacked_smem` 开关正是切这两个，
`runtime_utils.hpp:69-70`）。

模板实参里的 `weight_dtype_t` 就由这一处决定（`sm100_fp8_fp4_mega_moe.hpp:189`）：

```cpp
to_string(l1_weights.scalar_type())   // e4m3 -> "cutlass::float_e4m3_t"；kPackedFP4 -> "cutlass::detail::float_e2m1_unpacksmem_t"
```

（`to_string` 默认 `fp4_unpacked_smem = true`，所以 FP4 一定落 `unpacksmem` 串。）

### 3.4 一个会咬人的数字：TMA 期望字节减半

commit 级细节在权重 TMA 的 barrier 计数（`sm100_fp8_fp4_mega_moe.cuh:807-810`）：

```cpp
constexpr uint32_t kNumWeightBytes = sizeof(SharedStorage::smem_b[0]) * 2 /
    (kIsWeightFP8 ? 1 : 2);
shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(kNumWeightBytes + ...);
```

`sizeof(smem_b[0])` 是 SMEM 中权重 tile 的字节数（每元素 1B，FP4/FP8 相同）。FP4 时 `expect_tx` 是它的
1 倍、FP8 时是 2 倍——**恰好对应线上打包字节的 1:0.5**（FP8 每元素 1B、FP4 每元素 0.5B）。换言之，
barrier 等的是**真正移动的线上字节**，而不是 SMEM 里摊开后的容量。这也是 FP4 权重能让 TMA 流水「更深」
的物理来源：同样的 tile，走线字节少一半。

---

## 4. 权重重排：FP8/FP4 共用同一条 tuple 路径

`transform_weights_for_mega_moe`（`deep_gemm/mega/__init__.py:146-164`）只看「权重是不是 `(w, sf)` tuple」
来分流 FP8/FP4 与 BF16，**从不看 FP4 还是 FP8**：

```python
if isinstance(l1_weights, tuple):
    l1_w  = _interleave_weights(l1_weights[0])                       # L1 gate/up gran-8 交错
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))  # SF 交错 + UTCCP 4x32 转置
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
else:
    l1_transformed = _interleave_weights(l1_weights)                 # BF16 无 SF
    l2_transformed = l2_weights
```

对 FP4 权重（int8 packed），`_interleave_weights`（`:112-126`）沿 **N 轴**按 gran-8 把 gate/up 交错；
对 FP4 的 SF（int32），做同样的交错再 UTCCP 转置。真机核对（`E=4, l1=[1024,256]`）：

```
l1 w 仅交错=True；l1 sf 与「交错后 UTCCP 转置」一致=True；l2 sf 仅转置=True；l2 w 原样（同对象）=True
pack/unpack 与 N-交错可交换（逐位）: True
```

**一个容易担心的点**：交错发生在 N 轴、打包发生在 K 轴，两者会不会打架？答案是**不会，且逐位可交换**
（`unpack∘interleave == interleave∘unpack`，实测 True）——因为打包只在最后一维（K）上做相邻两元素合并，
而交错是沿倒数第二维（N）的置换，两者作用在不同轴。这正是 FP8 与 FP4 能共用同一段重排代码的根本原因。

### 4.1 官方测试怎么造 FP4 权重（`tests/test_mega_moe.py:102-110`）

```python
def _cast_weights_to_fp4(bf16_weights):
    num_groups, n, k = bf16_weights.shape
    w    = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)   # 物理 K/2
    w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
    w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)  # -> int32 MN-major
    return w, w_sf
```

注意与 FP8（`_cast_weights_to_fp8`，`:112-119`）逐行同构：唯一的差别是 `per_token_cast_to_fp4` 输出的
权重张量是 `[E,N,K/2]` int8、SF 仍是 `[E,N,K/32]` fp32（随后经 `transform_sf_into_required_layout`
变成 host 契约要求的 int32 packed UE8M0 MN-major）。**激活**（`:53-59`）恒用
`per_token_cast_to_fp8(..., use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`，与权重 dtype 无关。

---

## 5. host 配置面的 FP4 专属位（仅两处）

分块/流水启发式里真正区分 FP8 与 FP4 的只有 **store block**（`heuristics/mega_moe.hpp:112-115`）：

```cpp
// Use smaller, asymmetric store blocks at FP8xFP4 prefill to save shared memory for sufficient TMA pipeline depth
const int store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 :
                             mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;
const int store_block_m_l2 = (block_m == 240 and mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1;
```

对 `block_m ∈ (192,240]`，MXFP8FP4 取 `store_block_m_l1 = 24`（BF16 取 40），且 `block_m==240` 时
`store_block_m_l2 = 8`。注释给出了理由：**FP8xFP4 prefill 阶段把 store block 变小，省下的 SMEM 换成更深
的 TMA 流水**（FP4 权重线字节减半，正好把省下的预算用在 stage 数上）。注意这段对 `fp8xfp8` 与 `fp8xfp4`
**同样生效**（都走 `MXFP8FP4` 分支），实测 `block_m` 与 dtype 无关：

```
R=8 E=8 T=8192 K=2: block_m fp8xfp8=240 fp8xfp4=240 bf16=240
```

其余两处 FP4 影响（`heuristics:117,223-229`）：

- `block_k = 128 / get_num_mma_elem_bytes(mma_kind)` → MXFP8FP4 恒 128（BF16=64）；
- `num_bytes_per_pull = hidden * 1`，cap 8192（MXFP8FP4）vs BF16 的 `hidden*2`、cap 4096——因为激活恒
  FP8，**pull 粒度与权重是否 FP4 无关**。

---

## 6. H100 数值对拍：FP8 权重 vs FP4 权重

主 kernel 跑不了，但「FP4 权重相对 FP8 权重在 MoE 里代价多少」可以纯张量复刻。实验（`harness/exp_fp4_mega_moe_weights.py`
Part 4）搭一个小 MoE（T=256, H=512, I=512, E=8, topk=2），严格按 Mega MoE 的语义顺序：激活每 token
gran-32 UE8M0 FP8 量化 → L1 → SwiGLU（精确激活，只隔离 GEMM 量化）→ **L1 输出再 FP8 量化**（第 99 篇
钉死的语义）→ L2 → 按 topk 权重求和；权重分别做 FP8-MX 与 FP4-MX，与 fp64 精确参考比：

```
端到端 relerr vs fp64：  权重FP8 = 6.59e-2   权重FP4 = 2.11e-1   (激活都 FP8 gran32)
权重量化本身（W1，相对 bf16）：FP8 = 2.66e-2   FP4 = 1.18e-1
每专家权重字节： FP8 = 786432B   FP4 = 393216B   比 = 2.00x
```

结论：
- **FP4 权重把 MoE 输出误差从 ~6.6e-2 推到 ~2.1e-1**（≈3.2×），而**激活量化（gran 32 FP8）在两套配置里
  完全一样**——所以这个差距**纯粹来自权重的 4-bit 表示**（quantization 本身 1.18e-1 vs 2.66e-2，≈4.4×）。
- **权重显存恰好减半**（2.00×）。这正是「FP8 激活 + FP4 路由权重」在部署上诱人的原因：激活精度不变、
  专家权重减半，代价是 ~3× 的输出误差——对 prefill 这类对数值不敏感的阶段是可接受的交换（也解释了
  §5 注释里为什么专门为「FP8xFP4 prefill」调 store block）。

> 注意这是**极简**小形状（H=I=512）的定性对比，量化噪声随通道数、分布变化很大；这里量的是「FP4 的
> 相对代价量级」，不是目标卡上的最终精度。

---

## 7. 与相邻篇目/变体的关系

- **第 10 篇（MMA/epilogue）**：那里的 `SM100_MMA_MXF8F6F4_2x1SM_SS`、UTCCP、packed UE8M0 是**同一
  条路径**；本篇补的是「B 操作数何时是 FP4、它怎么被 TMA 摊开、为什么仍走 F8F6F4」。
- **第 20 篇（BF16 Mega MoE）**：BF16 是**另一个 kernel**（`sm100_bf16_mega_moe.cuh`），去掉了整条 SF
  通路、`block_k=64`、激活 2B；FP4 则是**同一个 kernel** 的权重 dtype 变体、`block_k=128`、激活 1B。
- **第 35 篇（SM100 FP8/FP4 稠密 GEMM）**：那里 A、B 都可能是 FP4 时会走 packed **MXF4**（`UMMA_K=64`）；
  Mega MoE 因 A=FP8 只能走 **MXF8F6F4**（`UMMA_K=32`）——同一个 `float_e2m1` 的两副面孔。
- **第 37 篇（参考量化器）**：FP4 的 `per_token_cast_to_fp4`/`cast_back_from_fp4` 的语义出处。

---

## 8. 小结

1. **`fp8xfp4` 与 `fp8xfp8` 是同一个 kernel、同一个 `MmaKind::MXFP8FP4`**；差异只在路由权重张量的 dtype
   （`kPackedFP4 = int8`）与 JIT 模板里 `weight_dtype_t` 的一处字符串。
2. **FP4 权重就是 `int8`**，每字节 2 个 e2m1；逻辑 K 靠 `get_logical_shape`/TMA 描述符的「×2」恢复。
3. **TMA 用 `16U4_ALIGN16B` 在搬运时把 4-bit 摊成可寻址的未打包形式**（`float_e2m1_unpacksmem_t`），
   SMEM 里每元素仍占 1 字节，故 tile 容量/流水级数不变，只有线字节减半——barrier 的 `expect_tx` 用
   `/(kIsWeightFP8?1:2)` 精确对上。
4. **因为 A=FP8、B=FP4，必须用 `MXF8F6F4`（unpacked）而非 packed `MXF4`**。
5. **权重重排的 N 轴 gran-8 交错与 K 轴打包可交换（逐位）**，所以 FP8/FP4 共用同一条 tuple 重排路径。
6. H100 定性对照：FP4 权重使 MoE 输出误差 ~3.2×（量化本身 ~4.4×），权重显存**恰好减半**。
7. 主 kernel 的 MMA/TMA 路径**SM100 专属、仅分析**；本篇所有 host 契约与张量语义均为 H100 真机实测。
