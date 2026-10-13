# 91. DeepGEMM 官方 SF 契约的代码对照与 H100 一致性核验（`docs/scaling-factor-format.md`）

> 第 25 篇讲了「块级 scale 怎么被变换」、第 26 篇讲了「K-grouped 的 SF 怎么按组打包」、第 78 篇讲了
> 「`tests/test_layout.py` 这份逐位 oracle 怎么定义对错」。但它们都从**实现侧**出发。DeepGEMM 仓库里
> 还有一份**官方契约文件** `docs/scaling-factor-format.md`（472 行），它是 SF 张量的
> **source of truth**：形状 / dtype / stride / 变换管线 / 各 API 的硬性要求，全都由它规定。
>
> 本篇换一个视角：**以官方 spec 为准，逐条对着代码核验**。重点放在 spec 里那些**前面几篇没系统覆盖**的
> 部分——预变换（pre-transformed）SF 的 stride 契约、K-grouped 预打包契约、以及
> 「被内部 permute 的 API（einsum）该在哪个坐标系变换」这条最容易踩坑的规则。
>
> **关键观察**：承载这些契约的两个 host 函数 `check_sf_layout` 与 `check_k_grouped_packed_ue8m0_tensor`
> **本身不含任何 arch 门控**（门控在它们的调用者里）。因此虽然 int32 packed UE8M0 在功能上属于
> SM100 专属、kernel 路径 H100 跑不了，但**契约校验这一层是纯 host、架构无关的**——本篇在 H100 上
> 直接 include 真实头文件、**运行真实契约函数**，对 spec 的每一条断言做正/负用例对拍。
>
> 核对版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（Public release 26/09/30），本机
> `kernel_lab`（H100 80GB HBM3, sm90）。实验：`harness/exp_sf_contract_conformance.{sh,out.txt}`
> + `src/sf_contract_conformance.cu`。

---

## 1. 官方 spec 在讲什么，本篇补哪一块

`docs/scaling-factor-format.md` 的目录（真实文件坐标）：

| § | 内容 | 代码落点 |
|---|---|---|
| §1.1 | quantization recipe `(gran_m, gran_n, gran_k)`；未变换 SF 的形状 | `csrc/utils/layout.hpp:112-119` |
| §1.2 | SF 值必须是**精确的 2 的幂** | device `smxx_layout.cuh:178/279-282` |
| §1.3 | 常量 `packed_sf_dtype=int32`、`ALIGN_MN=4` | `csrc/utils/math.hpp:23-27` |
| §2 | 两条路径：未变换 fp32（自动变换）vs 预变换 int32（只校验） | `csrc/apis/layout.hpp:38-55` |
| §3 | 预变换格式契约（MN-major + TMA 对齐 + packed） | `csrc/utils/layout.hpp:122-130` |
| §4 | `transform_sf_into_required_layout` 分派表 | `csrc/apis/layout.hpp:11-58` |
| §5.2 | K-grouped 预打包 int32 契约（**仅 gran_k=32**） | `smxx_layout.hpp:286-317` |
| §6.1 | MQA logits 的 `q_sf`/`kv_sf` **dtype 耦合** | `csrc/apis/attention.hpp:23-24/32/96` |
| §6.2 | `fp8_einsum` **内部 permute** SF，预变换要在哪个坐标系做 | `csrc/apis/einsum.hpp:262-294` |
| §6.3 | Mega MoE 的额外要求（recipe 钉死、权重 SF 必须 int32） | `csrc/apis/mega_moe.hpp:177/201-204` |

**分工**（避免重复）：§4 的分派表（哪条 recipe 走哪条路）第 25 篇已实测；§6.1 的 MQA 耦合第 19 篇
§5.4 已实测；§1.2/§7.1 的 UE8M0 打包位运算第 25 篇已逐位对拍。本篇补齐的是
**§1.1 / §2 / §3 / §5.2 / §6.2** 这五块**契约校验本身**——尤其「预变换 SF」这一路，前面几篇都没系统
验证过（因为 int32 packed 在功能上是 SM100 专属）。

---

## 2. 两条路径：自动变换 vs 预变换（spec §2）

spec §2 给出一个两行的表：同一个 GEMM API，SF 可以有两种给法（`:44-49`）：

- **路径 A（未变换）**：SF 是 `float32`，**每次调用** DeepGEMM 内部启动一个 transform kernel；
  用于原型 / 正确性测试。
- **路径 B（预变换）**：SF 已是 `int32`（packed UE8M0），**不再启动 kernel，只做布局校验**；
  用于生产（权重变换一次后缓存；激活由 cast kernel 直接产出）。

代码里这两条路径就是 `transform_sf_into_required_layout`（`csrc/apis/layout.hpp:11-58`）的四条
`if` 分支：前两条是 SM90 的 fp32 路（`:38-43`），第三条是 SM100 的 fp32→int32 路（`:46-51`，
真正启动 pack kernel），第四条是 **int32 的「只校验」路**：

```cpp
// csrc/apis/layout.hpp:54-55
if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 10)
    return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);
```

注意第四个参数（`tma_stride_check`）是 `true`、最后一个（`type_check`）是 `torch::kInt`。**路径 B 的
全部契约，就是这一次 `check_sf_layout` 调用**。这就是为什么本篇可以直接在 H100 上验证它——它不碰 arch。

> **为什么路径 B 是生产首选**：路径 A 每个 GEMM 调用都要为 A/B 各跑一次变换 kernel。对 MoE 权重
> （专家数 × 大 K）这是纯开销，所以生产把权重**变换一次缓存**；激活则让量化 cast kernel 直接吐出
> packed int32（spec `:125-139` 的 `per_token_cast` 例子）。

---

## 3. spec §1.1 形状公式 ↔ `check_sf_layout:112-119`（H100 A 组）

spec §1.1（`:21-24`）规定未变换 SF 的形状：

```
SFA: [·, ceil_div(M, gran_m), ceil_div(K, gran_k)]   dtype float32
SFB: [·, ceil_div(N, gran_n), ceil_div(K, gran_k)]   dtype float32
```

其中 `·` 是可选的 group/batch 维。对应到 `check_sf_layout`（`csrc/utils/layout.hpp`）：

```cpp
// csrc/utils/layout.hpp:112-119
DG_HOST_ASSERT(sf_dtype == torch::kFloat or sf_dtype == torch::kInt);         // :114
DG_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);     // :115
if (num_groups.has_value())
    DG_HOST_ASSERT(sf.size(-3) == num_groups.value());                        // :117
DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));                         // :118
DG_HOST_ASSERT(sf.size(-1) == ceil_div(k, gran_k * (sf_dtype == torch::kFloat ? 1 : 4)));  // :119
```

`:119` 里那个 `? 1 : 4` 是**同一个公式贯穿两条路径**的关键：fp32 时末维是 `ceil_div(k, gran_k)`；
int32（packed）时因为「4 个 UE8M0 指数打进 1 个 int32」，末维缩到 `ceil_div(k, gran_k*4)`。spec §1.3
把这两件事拆成常量（`packed_sf_dtype=int32` + 4-packed，`:39-40`），而代码用**一个三元表达式**表达。

**H100 实测（A 组）**：

| 用例 | 输入 | 结果 | 命中断言 |
|---|---|---|---|
| A1 | fp32 `[1000,4]`, mn=1000,k=512,gran=(1,128) | 通过 | — |
| A2 | fp32 末维=5 | 拒绝 | `layout.hpp:119` |
| A3 | fp32 `[8,4]`, gran_mn=128 → size(-2)=ceil_div(1000,128) | 通过 | — |
| A4 | fp32 `[3,1000,4]`, num_groups=3 | 通过 | `:115/117` |
| A5 | size(-3)=2 ≠ num_groups=3 | 拒绝 | `layout.hpp:117` |
| A6 | dim=2 但给了 num_groups | 拒绝 | `layout.hpp:115` |

---

## 4. spec §3 预变换 int32 契约 ↔ `check_sf_layout:122-130`（H100 B 组）

spec §3（`:55-63`）给出预变换 SF 的完整契约：

```
dtype:       int32 (packed UE8M0)
shape:       [·, mn, packed_sf_k]   packed_sf_k = ceil_div(k, gran_k * 4)
stride(-3):  stride(-1) * size(-1)          # group 维紧packed
stride(-2):  1                              # MN 方向连续（"MN-major"）
stride(-1):  align(mn, 4)                   # TMA 16 字节对齐
```

对应 `check_sf_layout` 的 `tma_stride_check` 分支：

```cpp
// csrc/utils/layout.hpp:122-130
if (tma_stride_check) {
    if (num_groups.has_value())
        DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));   // :124 group 维紧 packed
    DG_HOST_ASSERT(sf.stride(-2) == 1 or mn == 1);                       // :126 MN-major
    const auto compact_stride = get_tma_aligned_size(mn, sf.element_size());  // :127
    DG_HOST_ASSERT(sf.stride(-1) >= compact_stride);                     // :128 ≥ align(mn,4)
    DG_HOST_ASSERT(sf.stride(-1) % 4 == 0);                              // :129 4 的倍数
}
```

其中 `get_tma_aligned_size`（`csrc/utils/math.hpp:23-27`）= `align(x, 16/elem_size)`；对 int32
（elem_size=4）就是 `align(mn, 4)`——这就是 spec 里 `ALIGN_MN = 16/4 = 4` 的来源（§1.3 `:40`）。

**「为什么是 MN-major 而不是常见的 K-major」**：kernel 侧 SF 要经 TMA 载入、再被 UTCCP 搬进 TMEM
（第 10 篇）。TMA 描述符要求 `stride(-1)` 是 16 字节的倍数，且 kernel 沿 MN 方向做 UTCCP 转置，
所以 MN 维必须连续、K 维必须按 `align(mn,4)` 步进。`:126/:128/:129` 三条断言正是这些硬件约束的
host 侧镜像。

**H100 实测（B 组，mn=6, k=128, gran_k=32 → packed_sf_k=1, align(6,4)=8）**：

| 用例 | 输入 | 结果 | 命中断言 |
|---|---|---|---|
| B1 | int32 `[6,1]` stride `(1,8)`，`type_check=kInt` | 通过 | — |
| B2 | fp32 但 `type_check=kInt` | 拒绝 | `layout.hpp:110` |
| B3 | K-major `[6,4]` 连续 → `stride(-2)=4` | 拒绝 | `layout.hpp:126` |
| B4 | `stride(-1)=6 < align(6,4)=8` | 拒绝 | `layout.hpp:128` |
| B5 | `stride(-1)=10 % 4 != 0` | 拒绝 | `layout.hpp:129` |

> **B2 是本篇和 Mega MoE 的直连点**：`csrc/apis/mega_moe.hpp:201-204` 用的正是
> `check_sf_layout(..., true, false, torch::kInt)` 这同一个调用。spec §6.3 强调「Mega MoE 的权重 SF
> **必须已是 int32，没有 fp32 回退**」——原因就是这里的 `type_check=kInt`：喂 fp32 会在 `:110` 直接抛
> `DGException`。（Mega MoE 整个入口在 `mega_moe.hpp:367` 又有一道 `arch_major == 10` 门控，所以 H100
> 上走不到这条断言；但断言函数本身架构无关，B2 证明了它的行为。）

---

## 5. spec §5.2 K-grouped 预打包 ↔ `check_k_grouped_packed_ue8m0_tensor:286-317`（H100 C 组）

K-grouped GEMM（MoE 的 wgrad，第 12/26 篇）的 SF 也分 fp32（自动打包）和 int32（预打包）两条路。
spec §5.2（`:215-231`）规定 int32 预打包**只接受 `gran_k = 32`**，契约是 `[packed_sf_k, mn]` 连续，
**不插 MN 方向 padding**，改为硬性要求 `mn % 4 == 0`。对应
`check_k_grouped_packed_ue8m0_tensor`（`csrc/jit_kernels/impls/smxx_layout.hpp:286-317`）：

```cpp
// smxx_layout.hpp:292-302
DG_HOST_ASSERT(sf.scalar_type() == torch::kInt);                 // :292
DG_HOST_ASSERT(gran_k == 32);                                    // :293 只有 32 能预打包
DG_HOST_ASSERT(sf.dim() == 2);                                   // :294
const auto [packed_sf_k, mn] = get_shape<2>(sf);                 // :296（要求 is_cuda）
DG_HOST_ASSERT(mn % 4 == 0);                                     // :298
DG_HOST_ASSERT(sf.stride(0) >= mn and sf.stride(1) == 1);        // :299 pitch ≥ mn、MN 连续
DG_HOST_ASSERT(sf.stride(0) % 4 == 0);                           // :300
...
// :307-312：有 ks_cpu 时，packed_sf_k 必须 ≥ Σ ceil_div(k_i, gran_k*4)（larger is OK）
```

注意 `:299` 的 `stride(0) >= mn`（**大于等于**）：spec §5.2 明确「预分配一个按最大 K 的 buffer，
尾部多余行被忽略」。这与 §3 的单组契约不同——单组是 MN-major（`stride(-2)=1`），K-grouped 是
**普通连续 `[packed_sf_k, mn]`**（`stride(0)=mn`、`stride(1)=1`），因为 K-grouped 的输出层要把整组
拼接、不需要 TMA 在 MN 方向对齐。

**H100 实测（C 组，ks={2048,4096,1024}, gran_k=32 → required = 16+32+8 = 56）**：

| 用例 | 输入 | 结果 | 命中断言 |
|---|---|---|---|
| C1 | int32 `[56,8]` 连续（stride(0)=8≥mn、%4==0） | 通过 | — |
| C2 | `gran_k=128` | 拒绝 | `smxx_layout.hpp:293` |
| C3 | `stride(0)=mn-4 < mn` | 拒绝 | `smxx_layout.hpp:299` |
| C4 | `packed_sf_k=55 < required=56` | 拒绝 | `smxx_layout.hpp:312` |
| C5 | `packed_sf_k=60 > required`（预分配大 buffer） | 通过 | — |
| C6 | `ks_cpu` 空且 `use_psum_layout=false` | 拒绝 | `smxx_layout.hpp:314` |

> 这里有一个**契约细节值得记住**：C5 通过、C4 拒绝，说明校验是「`>=` 下界」而非「精确相等」。
> 第 26 篇讲 device 打包时提过「break 在累加 padding 之前」，本篇补充的是 **host 侧这条下界的确切语义**
> ——spec §5.2 用一句 "larger is OK, e.g. a buffer pre-allocated for the maximum K" 概括。

---

## 6. spec §6.2 einsum 的「按 post-permute 坐标变换再换回」（H100 D 组）

这是整份 spec 里**最容易踩、也最少被讲清**的一条规则。

`fp8_einsum` 在内部会把操作数和它们的 SF 一起 **permute** 成 `(batch, m, n, k)` 顺序，再转调内部
batched GEMM（`csrc/apis/einsum.hpp:262-294`）：

```cpp
// 以 'bhr,hdr->bhd' 为例（:268-269）
const auto perm_a   = a.first.permute({1, 0, 2});    // [b,h,r]  -> [h,b,r]
const auto perm_sfa = a.second.permute({1, 0, 2});   // [b,h,·]  -> [h,b,·]
```

对**路径 A（fp32）**的 SF，这没有任何影响——内部变换会按 permute 后的坐标自己处理。但对
**路径 B（预变换 int32）**的 SF 就出问题了：接收端最终会调用
`check_sf_layout(post_permute_sf, mn=b, ..., num_groups=h, tma_stride_check=true, type_check=kInt)`，
也就是拿**permute 之后的张量**去套 §3 的 stride 契约。所以 spec §6.2（`:373-387`）给出正确做法：

1. 按 **post-permute 坐标** `[h, b, packed]` 造一份 MN-major 的 int32 张量；
2. 把它 `permute(1,0,2)` 换回 `[b, h, packed]` 视图再喂给 `fp8_einsum`（einsum 内部再 permute 回去）。

**为什么要换回、不能直接 reshape**：变换后的张量是 **MN-major（非连续）**。若用 2D 变换输出再
`reshape(b, h, -1)`，PyTorch 会**静默 copy 成连续布局**，`stride(-2)` 从 1 变成 `packed_sf_k`，
于是进入 `check_sf_layout` 时在 `:124`（group 维紧 packed）或 `:126`（MN-major）处被拒。

代码上的确切约束就在 `csrc/utils/layout.hpp:122-128`（见 §4）。本篇用真实契约函数验证了两个方向：

**H100 实测（D 组，b=5, h=4, r=512 → packed_sf_k=4, align(b,4)=8）**：

| 用例 | 构造 | 结果 | 命中断言 |
|---|---|---|---|
| D1 | 按 `[h,b,packed]` MN-major 造、再 permute 回 `[b,h,packed]`（等价于在 `[h,b,packed]` 上校验） | 通过 | — |
| D2 | 2D 变换后 `reshape(b,h,packed)`（连续）→ 内部 permute 后 `stride(-2)=packed_sf_k` | 拒绝 | `layout.hpp:124` |

D2 里 naive reshape 后 post-permute 的 stride 打印为 `(1, 4, 20)`——`stride(-2)=4≠1`，正是 MN-major
被破坏的证据。这条规则对 Mega MoE 也有直接意义：`deep_gemm/mega/__init__.py` 的
`transform_weights_for_mega_moe` 同样是「在某个坐标里变换、kernel 按另一个坐标消费」的同一类问题
（第 10 篇讲过的 gate/up 交错 + UTCCP 4×32 转置）。

---

## 7. spec §6.3 Mega MoE 的额外要求（SM100 专属，仅分析）

spec §6.3（`:391-400`）列出 Mega MoE 相对普通 GEMM 的四条更硬的要求，逐条对代码：

1. **recipe 钉死 `(1,1,32)`**：`csrc/apis/mega_moe.hpp:177` 直接
   `DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32)`，`kGranK = 32` 也硬编码进 JIT 实例化。
2. **权重 SF 必须已是 packed int32，无 fp32 回退**：`mega_moe.hpp:201-204` 调
   `check_sf_layout(..., true, false, torch::kInt)`——和本篇 B2 命中的是同一个 `:110` 断言。路由 L1
   SF 形状 `[num_experts_per_rank, 2*intermediate_hidden, ceil_div(hidden,128)]`（`/128 = gran_k*4`）。
3. **必须先过 `transform_weights_for_mega_moe`**：L1 gate/up 按 8 粒度交错（权重 + SF）+ 对 L1/L2 SF
   做 UTCCP 4×32 转置（`deep_gemm/mega/__init__.py:112-149`，第 10 篇已逐位对拍）。spec 强调
   「单有 §3 布局还不够，kernel 不直接消费它」。
4. **激活 SF 是内部的**：`x_sf`、`l1/l2_acts_sf` 都住在对称 buffer 的切片里（`mega_moe.hpp:96-153`），
   由 kernel 流水线产出，不是用户参数。

第 1/2 条是 host 契约（本篇 B2 已间接覆盖 `:110`），第 3 条是第 10 篇的实测对象，第 4 条是第 42 篇的
buffer 布局。**四条合起来说明一件事**：Mega MoE 的 SF 契约比普通 GEMM **只紧不松**——它砍掉了路径 A
（没有 fp32 回退）、钉死了 recipe、还多要一步坐标系变换。

---

## 8. H100 实测汇总

`harness/exp_sf_contract_conformance.{sh,out.txt}`：`nvcc -arch=sm_90a` 直接 include
`csrc/utils/layout.hpp` + `csrc/jit_kernels/impls/smxx_layout.hpp`，在 H100 上编译并运行**真实契约函数**。

- **19 PASS / 0 FAIL**（A 组 6 + B 组 5 + C 组 6 + D 组 2）。
- 每个负用例都打印真实抛出的 `DGException` 里的 `file:line`，逐条对上 spec 的描述。
- 两个契约函数**架构无关**：int32 packed UE8M0 的 kernel 路径是 SM100 专属，但其 host 校验逻辑在
  H100 上原样可跑——这是「SM100 专属功能」里少数能在 sm90 上做**真机验证**的部分。

---

## 9. 与其它篇的分工

| spec 段 | 谁讲了 | 本篇 |
|---|---|---|
| §1.2 / §7.1 UE8M0 打包位运算 | 第 25 篇（逐位对拍） | 不重复 |
| §4 分派表（哪条 recipe 走哪条路） | 第 25 篇 §7 E 组 | 不重复 |
| §5 / §7.4 K-grouped device 打包算法 | 第 26 篇 | 只补 §5.2 host 契约 |
| §6.1 MQA 的 `q_sf`/`kv_sf` dtype 耦合 | 第 19 篇 §5.4 | 不重复 |
| §1.1 / §2 / §3 / §5.2 / §6.2 契约校验 | — | **本篇** |
| §6.3 Mega MoE 额外要求 | 第 10 篇（变换）+ 第 42 篇（buffer） | 本篇做代码对照 |

---

## 10. 线上资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM
- 官方 SF 契约文件（本篇的 spec）：https://github.com/deepseek-ai/DeepGEMM/blob/main/docs/scaling-factor-format.md
- 上游 PR #95（Mega MoE）：https://github.com/deepseek-ai/DeepGEMM/pull/95
- 上游 PR #316（Mega MoE EP8 基准）：https://github.com/deepseek-ai/DeepGEMM/pull/316
