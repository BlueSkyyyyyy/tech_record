# 从 delta rule 到分块：KDA 的数学地图

> 这是 KDA 专题的第 1 篇。目标只有一个：把 **KDA 到底在算什么**、以及 FlashKDA 为什么把它切成
> 「16 个 token 一块」讲清楚。后面几篇精读 CUDA kernel 时，所有符号都从这里来。

## 0. 先记住一个画面

把注意力状态 $S$ 想成一块**白板**。每来一个 token，做两件事：

1. **先擦**：按每个通道各自的比例，把白板上旧的记忆擦淡一点（擦多少由 $g_t$ 决定）；
2. **再写**：拿这个 token 的 key 当笔、value 当墨，把「这次要记的东西」写上去。

普通线性注意力（GLA）只会「擦 + 写」；**DeltaNet 多了一步「先看看白板上已经写了什么」**——
如果旧内容已经能解释这个 value，就只补写「差的那部分」（delta）。KDA 则把这两件事**同时**做了：
既保留 delta 的「只补差值」，又保留 GLA 的「逐通道擦除」。

所以一句话：

> **KDA = 带「逐通道遗忘」的 delta rule。**

> **类比的边界**：白板不会互相干扰，但 GPU 上的 $S$ 是 $128\times128$ 的矩阵，一个 token 的更新
> 是一次**秩一（rank-1）修正**；而且真实 kernel 里 $S$ 只有 bf16 精度、还常驻在寄存器/SMEM 里，
> 「擦」和「写」的顺序稍有不同就会改变数值结果。这些细节在后文都会逐条交代。

读完这一篇，你应该能回答：

1. DeltaNet、GLA、KDA 三者的状态更新差在哪一行公式？
2. 为什么 FlashKDA 要把序列切成 16 个 token 一块？块内那个 $16\times16$ 的求逆是干什么的？
3. 为什么前向要拆成 K1、K2 两个 kernel，而不是一个大 kernel？

## 1. 三种线性注意力，差在一行公式

线性注意力把「softmax 注意力」改写成「状态递推」：不再保存整个 KV cache，而是维护一个
$D\times D$ 的状态矩阵 $S_t$，每来一个 token 更新一次，输出直接由 $S_t$ 读出。

设逐头的 query / key / value 为 $q_t,k_t,v_t\in\mathbb{R}^D$（这里 $D=128$），
**gated delta rule** 的更新是：

$$
S_t = S_{t-1}\big(\operatorname{diag}(\alpha_t) - \beta_t\,k_t k_t^\top\big) + \beta_t\,v_t k_t^\top,
\qquad o_t = q_t^\top S_t .
$$

三个符号各管一件事：

| 符号 | 含义 | 直觉 |
|---|---|---|
| $\alpha_t\in(0,1]^D$ | 逐通道遗忘门 | 白板每个格子擦掉多少 |
| $\beta_t\in(0,1)$ | 写入门 | 这一笔写多重（delta 强度） |
| $k_tk_t^\top$ | 秩一投影 | 「往 key 的方向写」 |

把括号展开，就看得很清楚：

$$
S_t=\underbrace{S_{t-1}\operatorname{diag}(\alpha_t)}_{\text{先擦}}
\;-\;\underbrace{\beta_t\big(S_{t-1}\operatorname{diag}(\alpha_t)k_t\big)k_t^\top}_{\text{减掉旧内容在 }k_t\text{ 上的投影}}
\;+\;\underbrace{\beta_t v_tk_t^\top}_{\text{再写入新内容}} .
$$

中间那一项就是「delta」：先看旧状态在 $k_t$ 方向上已经有多少，再从新写入里扣掉，避免重复记。

三种模型的区别，就在 $\alpha$ 和 $\beta$ 谁在场、$\alpha$ 是标量还是向量：

| 模型 | 遗忘门 $\alpha$ | 写入门 $\beta$ |
|---|---|---|
| **DeltaNet** | 没有（$\alpha\equiv1$） | 有 |
| **GLA** | 有，但**每个 head 一个标量** | 没有（直接写 $v_tk_t^\top$） |
| **KDA** | 有，且**每个通道一个** $2^{g_t}$ | 有 |

KDA 的表达能力更强，代价是门控从「一个标量」变成「128 维向量」——这也是后面前向 kernel 里
门控计算要单独占一个阶段的原因。

> 逐 token 的参考实现见 `src/kda_ref.py::kda_token_recurrent`；它和后文的 16-token 分块实现
> 在 fp64 下逐元素相等（§6），这是我们验证「分块公式没推错」的最硬证据。

## 2. 输入、输出，以及门是怎么造出来的

FlashKDA 的前向接口（`flash_kda/__init__.py`）收这些东西：

| 名称 | 形状 | dtype | 含义 |
|---|---|---|---|
| `q,k,v` | `[B,T,H,K]` | bf16 | query / key / value（$K=V=128$） |
| `g` | `[B,T,H,K]` | bf16 | 门控 logits（**激活前**） |
| `beta` | `[B,T,H]` | bf16 | 写入门 logits（**激活前**，kernel 内 sigmoid） |
| `A_log` | `[H]` | fp32 | 每 head 的 log 门参数 |
| `dt_bias` | `[H,K]` | fp32 | 门偏置 |
| `lower_bound` | 标量 | float | 门下界（期望落在 $[-5,0]$） |

门在 kernel 里分四步造出来（对应 `tests/torch_ref.py:158–169`）：

1. **L2 归一化**：对每行 $q,k$ 做 $\ell_2$ 归一化。kernel 用 warp shuffle 树归约，
   且**故意与逐元素累加同序**，保证和 torch 参考逐位一致（`l2_normalize_kernel_match`）。
2. **加偏置**：$g\leftarrow g+\mathrm{dt\_bias}$（fp32 算）。
3. **缩放 + sigmoid**：令 $a=\exp(A\_log)$，
   $$
   g\leftarrow \text{lower\_bound}\cdot\log_2 e\cdot\sigma(a\cdot g).
   $$
   sigmoid 用 PTX 的 `tanh.approx.f32` 实现（$\sigma(x)=\tfrac12\tanh(x/2)+\tfrac12$），
   快且精度够用。
4. **写入门**：$\beta\leftarrow\sigma(\beta)$，同样用 `tanh.approx`。

这样 $g_t\le0$，$\alpha_t=2^{g_t}\in(0,1]$，正好是「只擦不增」的衰减。注意这里用的是
**以 2 为底**：kernel 里直接 `ex2.approx`，省掉一次换底乘法。

## 3. 为什么要分块：把「串行递推」换成「矩阵乘」

逐 token 更新有个致命问题：**完全串行**。第 $t$ 个 token 必须等第 $t-1$ 个算完，GPU 上一万个
线程也只能排队。序列越长，越像单线程跑。

分块（chunk）的思路是：把 16 个 token 切成一块，**块内**这 16 步递推可以代数地展开成一个
$16\times16$ 的小矩阵问题，从而变成矩阵乘 + 一次小求逆；**块间**才真正串行。

$$\underbrace{S_0\to S_1\to\cdots\to S_{15}}_{\text{块内：展开成矩阵运算}}\quad
\underbrace{S_{16}\to\cdots}_{\text{块间：只有状态串行}}$$

于是并行度从「1 个 token」变成「1 个块」，而块内的 16 步被压成几次 GEMM 和一次
$16\times16$ 求逆。FlashKDA 取 `CHUNK=16`，正好是 Tensor Core 的一个 $m16n16k16$ 指令宽度。

## 4. 分块前向的六步公式

下面这块是本专题的「地图」，后面 K1/K2 的每个阶段都能在这里找到对应。
记一块内 $q,k,v,g\in\mathbb{R}^{C\times D}$、$\beta\in\mathbb{R}^{C}$（$C=16$），
块入口状态为 $S_{\text{in}}\in\mathbb{R}^{D\times D}$（kernel 里存的是转置 $S_{\text{in}}^\top$，
布局为 $[V,K]$，见 §6）。按 `tests/torch_ref.py:208–239`：

**① 累积门与衰减因子**

$$
g^{\text{cum}}_i=\sum_{j\le i}g_j,\qquad
g^{\text{total}}=g^{\text{cum}}_{C-1},\qquad
E_i=2^{g^{\text{cum}}_i},\qquad E^{\text{tot}}=2^{g^{\text{total}}}.
$$

$E_i$ 是「从块首到第 $i$ 个 token 的累计衰减」，一个 $C\times D$ 的表。

**② 四种变形后的 q/k**

$$
k^{\text{dec}}_i=k_i\odot E_i,\quad
q^{\text{dec}}_i=q_i\odot E_i\cdot\text{scale},\quad
k^{\text{inv}}_i=k_i\odot E_i^{-1},\quad
k^{\text{rest}}_i=k^{\text{inv}}_i\odot E^{\text{tot}}.
$$

为什么要四种？因为「衰减」在块内有两种用法：算块内注意力时要把 key 往前衰减（$k^{\text{dec}}$），
算「还原到块首」时要把 query 往回补（$k^{\text{inv}}$），而状态更新要的是「整块衰减后的 key」
（$k^{\text{rest}}$）。这四个量都是 $k$ 乘一个对角矩阵，kernel 里只是逐元素乘。

**③ 块内注意力矩阵**

$$
L=\operatorname{tril}\!\big(k^{\text{dec}}{k^{\text{inv}}}^\top\big)\odot\beta\ (\text{严格下三角}),\qquad
M_{qk}=\operatorname{tril}\!\big(q^{\text{dec}}{k^{\text{inv}}}^\top\big).
$$

直觉：$L_{ij}$ 是「第 $i$ 个 token 对第 $j$ 个 token 的 delta 修正系数」，$M_{qk}$ 是块内
「query 看 key」的注意力分数。两者都只需要下三角，因为因果性。

**④ 块内求逆** $P=(I+L)^{-1}$

这一步是 KDA 分块的关键，也是 K1 里最贵的部分。为什么要求逆？因为块内 16 步递推是
$S_i=S_{i-1}(I-\beta_ik_ik_i^\top)+\cdots$ 的连乘，展开后是一个下三角线性系统，
解它就等于求 $(I+L)^{-1}$。

FlashKDA 的求逆不是 Neumann 级数，而是**精确前代**：两个 $8\times8$ 的 fp32 前代，
再用一次 bf16 HMMA 合并 off-diagonal 块（`inv_fwd_subst_16`）。好处是 $|L|\to1$ 时也稳。

**⑤ 修正 value 并算输出**

$$
\tilde v_i=\big(v_i-k^{\text{dec}}_iS_{\text{in}}^\top\big)\beta_i,\qquad
U=P\,\tilde v,
$$
$$
o_i=q^{\text{dec}}_iS_{\text{in}}^\top+(M_{qk}U)_i .
$$

第一项是「这个 token 从历史状态里读到的内容」，第二项是「块内前面的 token 通过 delta 传过来的内容」。
$U$ 就是「本块真正写进状态的增量」。

**⑥ 状态更新**

$$
\Delta S={k^{\text{rest}}}^\top U,\qquad
S_{\text{out}}=\big(\Delta S+S_{\text{in}}^\top\odot E^{\text{tot}}\big)^\top
=E^{\text{tot}}^\top\!\odot S_{\text{in}}+\Delta S^\top .
$$

旧状态先整块衰减一次（$E^{\text{tot}}$ 一次性结算块内所有 token 的累积衰减），再加上本块的
delta 增量。这样块与块之间只需要传一个 $D\times D$ 的状态。

> 这六步与逐 token 递推在实数域**严格等价**，fp32 下的差异只来自舍入顺序；
> 实测见 §6。

## 5. 为什么拆成 K1 / K2 两个 kernel

公式有了，怎么映射到 GPU？FlashKDA 的选择是拆成两个 kernel：

| | K1（prepare） | K2（recurrence） |
|---|---|---|
| 并行维度 | **token 并行**：grid = $N\times H\times\text{num\_chunks}$ | **head 并行**：grid = $N\times H$ |
| 每个 CTA 干什么 | 一个块内：归一化、门、$L$/$M_{qk}$、求逆 | 一个 head 的整条序列：逐块递推 |
| 写什么 | workspace（$k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},g^{\text{total}},P,M_{qk}$） | 输出 $o$、终态 $S$ |
| 特征 | 短、无 pipeline、并行度极高 | 长、warp specialization、TMA 流水 |

为什么不合成一个？因为 K2 的并行度只有 $N\times H$（比如 batch=1、8 个 head 就只有 8 个 CTA），
如果 K1 的活也塞进来，大量 SM 会空转。拆开后 K1 可以吃满所有 SM，K2 专心跑串行递推。
实测至少快 **15%**（见 `docs/07`），而且两段可以各自调优。

代价是多写多读一份 workspace（$k^{\text{dec}}$、$q^{\text{dec}}$、$k^{\text{rest}}$ 各 4KB/块），
这笔访存账在 `docs/03` 和 `docs/07` 里都有量化。

## 6. 源码地图：先把每个文件钉死

后面精读时不再迷路。所有行号都在 `~/github/FlashKDA` 上核对过。

| 文件 | 职责 | 关键入口 |
|---|---|---|
| `csrc/flash_kda.cpp` | host API：校验、workspace 大小、形状规整、分派 | `get_workspace_size`（L5）、`fwd`（L28） |
| `csrc/fwd.h` | launch 模板声明（$D$ / 状态 / varlen 作编译期参数） | `launch_fwd`（L6） |
| `csrc/smxx/fwd_launch.cu` | 建 TMA descriptor、切 workspace 指针、两次 `<<<>>>` | `launch_fwd`（L7） |
| `csrc/smxx/fwd_kernel1.cuh` | **K1 prepare**：门/归一化/衰减/$L,M_{qk}$/求逆 → workspace | `_flash_kda_fwd_prepare`（L120） |
| `csrc/smxx/fwd_kernel2.cuh` | **K2 recurrence**：逐块递推 + 输出 + 状态更新 | `_flash_kda_fwd_recurrence`（L133） |
| `csrc/smxx/utils.cuh` | PTX 近似函数、workspace 尺寸、warp 角色、pipeline、求逆例程 | 见下 |

三层是分开的：`flash_kda.cpp` 只做 tensor 检查和 reshape，不碰 cute/TMA；descriptor 全在
`fwd_launch.cu` 里建；kernel 模板只收 descriptor 常量。

**K1 的九段流水**（`fwd_kernel1.cuh:120`，单次 TMA load、无 pipeline）：

1. 定位 tile（L168–199）：非 varlen 直接整除；varlen 用 `tile_prefix` 二分。
2. TMA 载入（L200–255）：`threadIdx.x==0` 一次发 5 个 copy；beta 做 `& ~7` 对齐（1D TMA 最小粒度 8B）。
3. 归一化（L265–304）：每行 16 线程、每线程 8 元素，4 步 `__shfl_xor_sync` 树归约（L290–293，`delta` 取 8、4、2、1），`rsqrtf(x+1e-6)`（L295–296），回写。
4. 门激活 + cumsum（L306–338）：128 线程按列并行，天然串行，不需要跨行同步。
5. decay_apply（L360–475）：算出四种 q/k 变体，用 union 复用 smem。
6. 两个 GEMM（L481–487）：warp 0 算 $L$（fp32 累加），warp 1 算 $M_{qk}$（bf16）。
7. tril 掩码 + 乘 $\beta$（L493–508）。
8. 求逆（L510–511）：`inv_fwd_subst_fused_1warp`（`utils.cuh:214`）。
9. 写 workspace（L515–583）：6 个 TMA store，`tma_store_wait<0>`（L584）。

**K2 的角色分工**（`fwd_kernel2.cuh:133`，一个 CTA 串完整个序列）：

- 4 个 MMA warp + 1 个 LOAD warp + 1 个 STORE warp，共 192 线程（`fwd_launch.cu:186`）。
- load 用 3-stage TMA pipeline，store 用 2-stage async pipeline。
- 六个 Phase：双 GEMM → cast/载入 v → $u=(v-u)\beta$ 与 $Pu$ → $M_{qk}U+\text{out}$ →
  写回 smem → 状态更新。
- `MOVM_T` 让 MMA 的输出片段直接当下一步的 B 操作数，省掉一次 smem 往返。

**workspace 布局**（`flash_kda.cpp:5` 与 `fwd_launch.cu:62–71` 必须一致）：

$$
\text{per\_tile}=\underbrace{3\cdot C\cdot D\cdot2}_{k^{\text{dec}},q^{\text{dec}},k^{\text{rest}}}
+\underbrace{D\cdot4}_{g^{\text{total}}}
+\underbrace{2\cdot C^2\cdot2}_{P,\,M_{qk}} .
$$

六段按 `kKDecayed → kQDecayed → kKRestored → kGTotal → kINV → Mqk` **分开排**（SoA，不是结构体数组），
最后留 varlen 的 `tile_prefix`。这样每个字段能独立配 TMA descriptor，且都满足 128B 对齐
（L16–18 的 `static_assert`）。

## 7. 第一步验证：逐 token 与分块必须逐位一致

在信任任何 kernel 之前，先验证公式。`src/kda_ref.py` 里有**两份独立实现**：

- `kda_token_recurrent`：逐 token 递推（§1 的公式），当 golden；
- `kda_chunked_fwd`：16-token 分块（§4 的六步）。

两者计算顺序完全不同，互拍能直接证明分块代数没推错。把状态布局统一为 $S\in\mathbb{R}^{K\times V}$、
输出 $o_t=\text{scale}\cdot q_tS_t$ 后：

- **fp64**：$(T,H)\in\{(1,1),(16,1),(20,2),(48,3)\}$、有无初态、$g$ 尺度 $\{0,1,4\}$ 的所有组合下，
  两种实现的 $o$ 与终态 **逐元素完全相等**（`max_abs=0`）。
- **fp32**：同样组合下 `max_abs` 约 $10^{-9}\sim10^{-7}$，即误差只来自舍入顺序。

再与 FLA 的 Triton `chunk_kda` 对拍（同一输入、$D=128$、$T=20/64$、$H=2$、非零初态）：
bf16 下 $o$ 的 `max_rel` 约 $4.7\times10^{-3}\sim5.9\times10^{-3}$，终态的
`max_rel` 约 $3.1\times10^{-3}\sim6.0\times10^{-3}$——和 bf16 输入本身的量化误差同量级。
原始输出见 `harness/kda_ref_selftest.out.txt` 与 `harness/check_kda_ref.out.txt`。

### 一个必须知道的坑：状态是 $S^\top$，不是 $S$

FlashKDA 的 `initial_state` / `final_state` 用的是 **$S^\top$ 布局**（`[N,H,V,K]`，
`torch_ref.py` 里 `state_slice` 存的就是 $S^\top$），和 K2 内部的 `s_acc`（$[V,K]$）一致。

FLA 的 `chunk_kda` 则默认返回 $[N,H,K,V]$。两者互为转置：直接比 `max_abs=0.511`，
转置后再比只剩 `0.00398`（`harness/probe_state_layout.py`）。

还有一个更隐蔽的坑：FLA 的 `transpose_state_layout=True`（新版叫 `state_v_first`）不只是改
**返回**布局，还会把传入的 `initial_state` 也按 V-first 重新解释。最初误用该开关后，
除 $t=0$（唯一直接吃初态的 token）外一切正常，$t=0$ 的误差却高达 $0.1$——看起来像
「随机小误差」，实际是布局错了。默认 `False` 时两边布局一致，对拍正常。

## 8. 三句话小结

- KDA 的更新是「**先按通道擦、再按 key 方向补写差值**」，即带逐通道遗忘的 delta rule。
- 分块把块内 16 步串行递推压成「四种 q/k 变形 + 两个 $16\times16$ GEMM + 一次块内求逆 + 一次状态更新」。
- 前向拆 K1（token 并行 prepare）与 K2（head 并行 recurrence），是为了让求逆这类高并行活吃满 SM，
  而把串行递推留给少数 CTA。

下一篇进入 K1 的代码：看这六步公式如何在一个 CTA 里落地，尤其是那个 16×16 求逆到底怎么写的。
