# MoE 一个训练步（前向 + 反向）的分段成本账 + 共享专家反向的同图闭环

> 前面各篇把**前向**（第 99 篇 P89-1 路由、第 168 篇 P158-1 路由 + 共享）与**反向**
> （第 14 篇 P4-4 路由四 GEMM、第 15 篇前向归因、第 145/146/147/150 篇 wgrad 累积协议）
> 分别讲透，但留下两格空白：
>
> 1. **反向图里从来没有共享专家**——第 14 篇的图只有路由分支；共享专家的反向整条缺席。
>    第 168 篇刚把共享专家接进**前向**，却没有同图的反向。
> 2. **从来没有把前向和反向放进同一次计时**——量出「一个训练步里 前向 vs 反向、GEMM vs 非 GEMM」
>    的真实占比（第 142/143 篇只做前向 single-step，第 144 篇的反向预算只到四个 GEMM + 一段激活）。
>
> 本篇补这两格，全部用 H100 上可跑的真实 SM90 FP8 kernel。所有数字来自
> `harness/exp_moe_train_step.{py,sh,out.txt}`（H100 sm90，DeepGEMM 2.8.1）。
>
> **Mega MoE 主 kernel 是 SM100（Blackwell）专属、本机跑不了**，本篇只用其**架构无关的数学、
> 布局与量化语义**复刻；主 kernel 的 UMMA / tcgen05 / NVLink 部分**仅分析**。
> 目标卡代号敏感：全文用「目标卡」指代。

参考坐标：DeepGEMM commit `057ca59`（v2.8.1）；
`csrc/apis/mega_moe.hpp`、`csrc/apis/gemm.hpp`、
`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、
`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`、
`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`、`tests/test_mega_moe.py`。

---

## 1. 为什么需要这一篇

DeepSeek 的 MoE 层是

```
y = Σ_{i∈topk} w_i · Expert_i(x) + SharedExpert(x)
```

**前向**（第 168 篇）已经在 H100 上用真实 FP8 kernel 闭合：共享走稠密 `fp8_gemm_nt`、路由走
`m_grouped_fp8_gemm_nt_contiguous`、combine 把共享项按 `kNumTopk` 槽 fp32 相加一次。

**反向**却一直只有路由那一半。第 14 篇（`docs/14-moe-backward-kernels.md`）把四个路由大 GEMM
（`dW2`、`dA`、`dX`、`dW1`）换成 DeepGEMM 的 grouped kernel 并与 autograd 对拍，但**共享专家的
梯度整条没进图**——共享权重 `W1s`/`W2s` 的 `dW1s`/`dW2s` 从没在 kernel 上算过。

同时，「一个 MoE 训练步到底花在哪」也一直是拆开的：第 142/143 篇做前向，第 144 篇做反向的
GEMM + 一段激活。把两条腿放进**同一个计时**，才能回答「前向占多少、反向占多少、GEMM 占多少、
量化占多少」——这也是 Mega MoE 为什么要把前向融成单 kernel 的量化依据。

本篇做两件事：

* **补齐共享专家的反向**，并给出它和前向一样是「同一对 grouped kernel、只是组数 E→1」的结论；
* **给出一个训练步的分段成本账**（前向 + 反向，每段真实 kernel / 算子各自计时）。

---

## 2. 共享专家的反向：单组 grouped GEMM

### 2.1 共享专家的前向（无专家维）

共享专家是一个**对所有 token 恒定全量激活、没有专家维**的前馈层。设共享中间维
`S = num_shared_experts · intermediate_hidden`（`csrc/apis/mega_moe.hpp:52-53`：

```cpp
const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
```

它在前向里被拆成两个**普通 2D** 权重（`mega_moe.hpp:211-213` 的 `check_weights_layout_2d`）：

```
G1s = x @ W1s[:S]ᵀ      [T,H]@[S,H]ᵀ → [T,S]      # gate
G2s = x @ W1s[S:]ᵀ      [T,H]@[S,H]ᵀ → [T,S]      # up
a   = silu(G1s) * G2s   [T,S]                      # 不乘 top-k 权重（kernel:1068）
ys  = a @ W2sᵀ          [T,S]@[H,S]ᵀ → [T,H]
```

其中 `W1s` 是 `[2S, H]`、`W2s` 是 `[H, S]`（`csrc/apis/mega_moe.hpp` 的 host 断言
`shared_intermediate_hidden % 128 == 0`，`:90`）。前向两条 GEMM 走稠密 `fp8_gemm_nt`（第 168 篇）。

### 2.2 反向的数学：与路由专家逐项同形

给定上游 `dYs = ∂L/∂ys`（`[T,H]`），链式法则：

```
da   = dYs @ W2s                 [T,H]@[H,S]      → [T,S]
dW2s = dYsᵀ @ a                  [T,H]ᵀ@[T,S]     → [H,S]
dG1s = da * silu'(G1s) * G2s
dG2s = da * silu(G1s)
dGs  = [dG1s | dG2s]             [T,2S]
dW1s = xᵀ @ dGs                  [T,H]ᵀ@[T,2S]    → [H,2S]
dxs  = dGs @ W1s                 [T,2S]@[2S,H]    → [T,H]
```

把这四个式子与第 14 篇路由专家反向的四个大 GEMM 并排看，**形状角色完全一致**，只差「组数」：

| 段 | 路由专家（E 组） | 共享专家（1 组） | kernel |
|---|---|---|---|
| L2 wgrad | `dW2[e]=A_eᵀ dY_e`（M=I,N=H,K=token） | `dW2s=dYsᵀ a`（M=H,N=S,K=T） | `k_grouped_*_nt_contiguous` |
| L2 dgrad | `dA_e=dY_e W2_eᵀ`（M=token,N=I,K=H） | `da=dYs W2s`（M=T,N=S,K=H） | `m_grouped_*_nt_contiguous` |
| L1 wgrad | `dW1[e]=x_eᵀ dG_e`（M=H,N=2I,K=token） | `dW1s=xᵀ dGs`（M=H,N=2S,K=T） | `k_grouped_*_nt_contiguous` |
| L1 dgrad | `dX_e=dG_e W1_eᵀ`（M=token,N=H,K=2I） | `dxs=dGs W1s`（M=T,N=H,K=2S） | `m_grouped_*_nt_contiguous` |

**结论：共享专家的反向不是新算子，而是「组数 E=1」的同一对 grouped kernel。** 这正是 Mega MoE
之所以能用一套反向路径同时覆盖两条通路的根。

### 2.3 「组数 E=1」在 API 上确实合法

这不是比喻——host 入口本身接受任意组数：

* wgrad `k_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:373`）：
  `num_groups = get_shape<3>(d)`（`:387`），`check_k_grouped_args`（`:55-76`）只要求
  `grouped_layout.numel() == num_groups`（`:63`）且每个 `k % k_alignment == 0`（`:69`）——
  **组数 1 完全合法**（`host_ks = [align(T,128)]`）。注意它强制 `recipe == (1,1,128)`（`:383`）
  且 **累积张量 `c` 必填**（`:403`），这正是第 145/150 篇讲的「SM90 wgrad 强制累积」。
* dgrad `m_grouped_fp8_fp4_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:181`）：
  `num_groups = check_grouped_ab_fp8_fp4(b.first, ...)`（`:205`），
  contiguous 布局下 `grouped_layout.numel() == m`（`:217-218`）；**D 必 bf16**（`:209`）。
  `B` 的形状是 `[G, N, K]`，单组就是 `[1, S, H]` / `[1, H, 2S]`（转置后）——正是 `W2sᵀ`/`W1sᵀ`。

### 2.4 H100 实测：共享专家反向正确性（Part A）

`harness/exp_moe_train_step.py` 的 Part A 在**三组小形状**上把上面四个 kernel 接起来，双层对拍：

* `vs 反量化参考`：把喂进 kernel 的 FP8 反量化后在 fp32 重算同一公式——**隔离 kernel + 布局**误差；
* `vs fp32 autograd`：与 `torch.autograd.grad` 参考比较——**含 FP8 量化**的总误差。

| T | H | I | Es | S | dW2s | dW1s | da | dxs |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 256 | 1 | 256 | 2.954e-4 | 2.959e-4 | 3.422e-4 | 3.404e-4 |
| 128 | 1024 | 512 | 2 | 1024 | 3.156e-4 | 3.160e-4 | 3.169e-4 | 3.216e-4 |
| 256 | 1024 | 512 | 1 | 512 | 3.203e-4 | 3.213e-4 | 3.130e-4 | 3.267e-4 |

（表内是 `vs 反量化参考` 的相对误差。）

* **kernel + 布局**四路都在 **~3e-4**（= bf16 输出舍入地板），与第 14 篇路由侧 `vs deq` 的
  2.6e-4 同量级——证明「共享=单组 grouped GEMM」的接线（含 `W2sᵀ`/`W1sᵀ` 转置、单组
  `grouped_layout`、per-token/per-block SF）**全部正确**。
* **含量化**（vs fp32 autograd）：`dW2s` 3.2–3.4e-2、`dW1s` 4.9–5.1e-2、`da` 3.7e-2、
  `dxs` 5.1e-2，**cos ≥ 0.9987**——与第 14 篇路由侧（2.5e-2~5.1e-2、cos > 0.999）同量级，
  误差地板仍由 **FP8 格式**决定（per-128 权重块 + per-token 激活）。

至此**路由（第 14 篇）+ 共享（本篇）**两条反向通路都用真实 kernel 对拍完成，MoE 反向图闭合。

---

## 3. 一个训练步的分段成本账（Part B）

### 3.1 记账口径

`harness/exp_moe_train_step.py` 的 Part B 在 **DeepSeek 量级**（`H=7168, I=2048, E=8, topk=2,
Es=1`，`T=2048` 与 `T=4096` 两档）把每一段**单独计时**（`_bench`，30 次取平均，`torch.cuda.synchronize`
后计时）。分四类标签：

* **FWD / BWD**：前向段 / 反向段；
* **GEMM**：FLOPs > 0 的段（`m_grouped`/`k_grouped`/dense `fp8_gemm_nt`）；
* **non-GEMM**：SwiGLU 元素级、combine/scatter 的 `index_add`、以及**激活量化**。

> **重要口径注记**：这里的「激活量化」用的是 DeepGEMM 的 **torch 参考实现**
> `deep_gemm.utils.per_token_cast_to_fp8`（多趟 elementwise，launch-bound），**不是**生产路径。
> 真实 Mega MoE 把 UE8M0 量化**融进 L1 epilogue 的寄存器里**（`kernel:1160-1212`，第 163 篇的官方
> TileLang SwiGLU 算子 122us 才接近设备量级）。第 168 篇也提示过这 859us 的参考量化是测试侧开销。
> 因此下面同时给「含参考量化」与「扣掉参考量化」两个总数。

### 3.2 T=2048（每 rank 4096 路由行）

| 段 | 时间 (us) | TFLOPS |
|---|---|---|
| FWD 路由 L1 `m_grouped` | 659 | 365 |
| FWD 路由 L2 `m_grouped` | 416 | 289 |
| FWD 共享 L1 dense fp8 | 402 | 299 |
| FWD 共享 L2 dense fp8 | 215 | 280 |
| FWD 激活量化 ×4（参考） | 2922 | — |
| FWD SwiGLU（元素级） | 613 | — |
| FWD combine（index_add） | 861 | — |
| BWD 路由 dgrad L2 `m_grouped` | 411 | 292 |
| BWD 路由 wgrad L2 `k_grouped` | 1771 | 68 |
| BWD 路由 dgrad L1 `m_grouped` | 653 | 368 |
| BWD 路由 wgrad L1 `k_grouped` | 3392 | 71 |
| BWD 共享 dgrad L2 `m_grouped` | 206 | 293 |
| BWD 共享 wgrad L2 `k_grouped` | 410 | 146 |
| BWD 共享 dgrad L1 `m_grouped` | 409 | 294 |
| BWD 共享 wgrad L1 `k_grouped` | 654 | 184 |
| BWD 反向激活量化 ×3（参考） | 2876 | — |
| BWD SwiGLU（元素级） | 1305 | — |
| BWD scatter-add dX | 1117 | — |

汇总：**前向 6089us（31.6%）、反向 13204us（68.4%）**；**GEMM 9599us（49.8%）、非 GEMM
9694us（50.2%）**；参考量化 5798us（30.1%）；**一个训练步合计 19293us**，
**扣掉 torch 参考量化后 13495us**。

### 3.3 T=4096（每 rank 8192 路由行）

汇总：**前向 11247us（35.8%）、反向 20148us（64.2%）**；**GEMM 13308us（42.4%）、非 GEMM
18087us（57.6%）**；参考量化 10965us（34.9%）；**合计 31395us**，
**扣掉参考量化后 20429us**。

### 3.4 几条可读的结论

1. **反向 ≈ 前向的 1.8–2.2×**（6089→13204 / 11247→20148）。前向 4 条 GEMM（路由 L1/L2 +
   共享 L1/L2）只花 ~1.7ms（T=2048），反向 8 条 GEMM 花 ~7.9ms。这是训练比推理贵的直接来源。
2. **wgrad 是反向最贵的一段**：路由 `wgrad L1+L2` = 3392+1771 = **5163us**，占反向的 **39%**，
   而其 TFLOPS 只有 **68–71**（T=2048）/ **127–133**（T=4096）——远低于同形状 dgrad 的
   289–368。这与第 7/15/135 篇的归因**逐字吻合**：SM90 `k_grouped` 强制累积（`c` 必填
   `gemm.hpp:403`）+ fp32 权重梯度输出，使 wgrad **访存受限**；T 翻倍（K 从 512→1024/组）
   让算术强度翻倍，TFLOPS 也就翻倍（68→133）。
3. **共享专家反向很便宜**：4 条 `k/m_grouped`（单组）= 206+410+409+654 = **1679us**，仅占反向
   **~13%**。它的 `wgrad`（410/654us，146/184 TFLOPS）比路由 wgrad 高效得多——因为共享
   wgrad 是**单个大 K**（K=T=2048/4096，一个组），没有「每组 K 只有 token/E 那么小」的问题。
   这正是第 150 篇「big-K 折叠」的自然对照。
4. **激活量化在参考实现下 ~30%**，但在生产路径里是**融进 epilogue** 的（第 168 篇的注记）。
   这条差异本身是**支持融合**的量化依据：一个训练步里激活被量化 **7 次**（前向 A1/A2 ×2 通路 +
   反向 dY/dG/dGs），若每次都走独立 kernel，量化会吃掉 ~1/3 的步时。
5. **GEMM 与「非 GEMM」几乎对半**（扣掉参考量化后，GEMM 占比更高）。这解释了 Mega MoE 的融合
   动机：把 SwiftGLU/量化/combine 这些「非 GEMM」也塞进 kernel 的 epilogue/comm warp，才能让
   GEMM 之外的 ~一半时间不再是纯开销。

---

## 4. 与前向融合主 kernel 的关系（仅分析）

上面这个「分段账」在 Mega MoE 的单 kernel 里被压成三段 warp 流水：

* **dispatch warp（128 线程）** 负责 EP 通信（第 11/120 篇）；
* **non-epilogue warp（128 线程）** 负责 UMMA 的 L1/L2（`scheduler/mega_moe.cuh`）；
* **epilogue warp（256 线程）** 负责 SwiGLU + UE8M0 量化 + TMA 上屏（`kernel:1066-1215`）+
  L2 的 NVLink 直写 + combine 归约（`kernel:1346-1519`，第 10/11 篇）。

也就是说，本篇在 H100 上**逐段串行**量出的「前向 + 反向」预算，正是 Mega MoE 想用
**通信-计算重叠 + epilogue 融合**去压缩的对象——尤其第 4 条结论里的「7 次激活量化」和第 5 条
里的「非 GEMM 一半」，在融合 kernel 里都变成**寄存器内、与 UMMA 重叠**的旁路。

约束（重申）：Mega MoE 主 kernel 的 UMMA / tcgen05 / UTCCP / NVLink 直写是 **SM100 专属，
仅分析**；本篇的所有数字都来自 H100 上可跑的真实 SM90 FP8 kernel 与 torch elementwise。

---

## 5. 小结

* **共享专家反向** = 「组数 E=1」的同一对 grouped GEMM（`k_grouped` wgrad + `m_grouped`
  dgrad，`gemm.hpp:373`/`:181`），四路梯度 vs 反量化参考 ~3e-4、vs autograd 2.5e-2~5.1e-2／
  cos ≥ 0.9987。至此**路由 + 共享**两条反向通路都用真实 kernel 对拍完成。
* **一个 MoE 训练步的分段账**（H100，真实 SM90 FP8 kernel）：
  * T=2048：合计 **19.29ms**（前向 31.6% / 反向 68.4%；GEMM 49.8%）；扣参考量化 13.50ms；
  * T=4096：合计 **31.40ms**（前向 35.8% / 反向 64.2%；GEMM 42.4%）；扣参考量化 20.43ms；
  * **wgrad 是最贵的一段**（占反向 39%、TFLOPS 仅 68–133），与第 15/135 篇的访存受限归因一致；
  * **共享专家反向仅占 ~13%**（单组大 K，效率高于路由 wgrad）；
  * **激活量化在 torch 参考实现下占 ~30%**，但在生产路径里融进 epilogue——这是支持融合的量化依据。
* 主 kernel 的融合/重叠部分是 **SM100 专属、仅分析**（第 10/11/120/167 篇）。
