# KDA 数学与分块（chunk）公式

> 本文是「Kimi Delta Attention 前向 / 反向算子实现解析」的第 1 篇。先把 KDA 的数学定义、
> 门控构造、以及 FlashKDA 采用的 **CHUNK=16 分块公式**写清楚，作为后续精读 CUDA kernel 的地图。
> 公式与 FlashKDA 的 PyTorch 参照实现 `~/github/FlashKDA/tests/torch_ref.py` 逐项对应。

## 1. KDA 是什么：带门控的 delta rule

线性注意力把注意力写成「状态 + 递推」的形式。**DeltaNet / 门控 Delta rule** 的状态是一个
$D\times D$ 矩阵 $S_t$（这里 $K=V=D=128$），逐 token 更新：

$$
S_t = S_{t-1}\big(\operatorname{diag}(\alpha_t) - \beta_t\,k_t k_t^\top\big) + \beta_t\,v_t k_t^\top,
\qquad o_t = q_t^\top S_t,
$$

其中 $k_t,v_t,q_t\in\mathbb{R}^D$ 是逐头的 key/value/query，$\beta_t\in(0,1)$ 是写入门（delta 强度），
$\alpha_t\in(0,1]^D$ 是**逐通道**的遗忘/衰减门。

**KDA（Kimi Delta Attention）** 与 GLA / DeltaNet 的关键差别，就在这个门如何取：
KDA 使用 **per-channel（逐通道）的 log-decay** $g_t\in\mathbb{R}^D$（而不是每个 head 一个标量），
$\alpha_t=\exp(g_t)$ 在 $D$ 个通道上各不相同，表达能力更强。三者的定位可以这样对照：

| 模型 | 状态更新 | 遗忘门 $\alpha$ | 写入门 $\beta$ |
|---|---|---|---|
| **DeltaNet** | $S_t=S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$ | 无（$\alpha\equiv1$） | 有（delta 强度） |
| **GLA** | $S_t=S_{t-1}\operatorname{diag}(\alpha_t)+\text{(write)}$ | **每 head 一个标量**，数据相关 | 无（直接写入） |
| **KDA** | $S_t=S_{t-1}\operatorname{diag}(\alpha_t)+\beta_t(v_t-\cdots)k_t^\top$ | **每通道**，数据相关 | 有（delta 强度） |

也就是说：DeltaNet 有 $\beta$ 没有 $\alpha$，GLA 有 $\alpha$ 没有 $\beta$，KDA 两者兼有且
$\alpha$ 是逐通道的。把式 (1) 展开成 $S_t=S_{t-1}\operatorname{diag}(\alpha_t)
-\beta_t(S_{t-1}\operatorname{diag}(\alpha_t)k_t)k_t^\top+\beta_t v_tk_t^\top$ 即「先按当前 token 的门衰减旧状态，
再做 delta 写回」——这正是 §3 分块公式里 $k^{\text{dec}}$ 与 $S_{\text{in}}\odot E^{\text{tot}}$ 两处的来历。

> 逐 token 的等价递推实现见 `src/kda_ref.py::kda_token_recurrent`（$S$ 采用 $[K,V]$ 布局，
> 输出 $o_t=\text{scale}\cdot q_t S_t$），分块实现见同文件 `kda_chunked_fwd`。

## 2. 输入、输出与门控构造

FlashKDA 前向接口（`flash_kda/__init__.py`）的输入：

| 名称 | 形状 | dtype | 含义 |
|---|---|---|---|
| `q,k,v` | `[B,T,H,K]` | bf16 | query / key / value（$K=V=128$） |
| `g` | `[B,T,H,K]` | bf16 | 门控 logits（**激活前**） |
| `beta` | `[B,T,H]` | bf16 | 写入门 logits（**激活前**，kernel 内做 sigmoid） |
| `A_log` | `[H]` | fp32 | log 门参数 |
| `dt_bias` | `[H,K]` | fp32 | 门偏置 |
| `lower_bound` | 标量 | float | 门下界（期望在 $[-5,0]$） |

门控在 kernel 里这样构造（对应 `torch_ref.py:158–169`）：

1. **L2 归一化**：对每行 $q,k$ 做 $\ell_2$ 归一化（kernel 用 warp shuffle 树归约 + FMA，故意
   与逐元素累加顺序一致，见 `l2_normalize_kernel_match`）。
2. **加门偏置**：$g \leftarrow g + \mathrm{dt\_bias}$（fp32）。
3. **缩放 + sigmoid**：令 $a=\exp(A\_log)$，
   $$
   g \leftarrow \text{lower\_bound}\cdot\log_2 e \;\cdot\; \sigma\big(a\cdot g\big),
   $$
   sigmoid 用 PTX `tanh.approx.f32` 实现（$\sigma(x)=\tfrac12\tanh(x/2)+\tfrac12$），既快又够准。
4. $\beta \leftarrow \sigma(\beta)$（同样用 `tanh.approx`）。

于是 $g_t\le 0$，$\exp(g_t)\in(0,1]$ 就是逐通道衰减。

## 3. 分块（chunk）前向公式

把序列按 $\mathrm{CHUNK}=16$ 切块。对某一块（长度 $C$，通常 $C=16$），记该块的
$q,k,v,g\in\mathbb{R}^{C\times D}$、$\beta\in\mathbb{R}^{C}$，块起始的递推状态为
$S_{\text{in}}\in\mathbb{R}^{D\times D}$（布局为 $[V,K]$，即 `torch_ref` 里的 `state_slice`）。按 `torch_ref.py:208–239`：

**① 累积门 / 衰减因子**

$$
g^{\text{cum}}_i=\sum_{j\le i} g_j,\qquad
g^{\text{total}}=g^{\text{cum}}_{C-1},\qquad
E_i = 2^{\,g^{\text{cum}}_i},\quad E^{\text{tot}}=2^{\,g^{\text{total}}} .
$$
（代码里 `fp32_ex2_ftz` 即 `ex2.approx.ftz.f32`；base-2 免去换底 FMA。）

**② 衰减后的 q/k 与「还原」用的 k**

$$
k^{\text{dec}}_i = k_i\odot E_i,\quad
q^{\text{dec}}_i = q_i\odot E_i\cdot\text{scale},\quad
k^{\text{inv}}_i = k_i\odot E_i^{-1},\quad
k^{\text{rest}}_i = k^{\text{inv}}_i\odot E^{\text{tot}} .
$$

**③ 块内注意力矩阵**

$$
L = \operatorname{tril}\!\big(k^{\text{dec}}\,{k^{\text{inv}}}^\top\big)\odot \beta \ (\text{严格下三角}),\qquad
M_{qk} = \operatorname{tril}\!\big(q^{\text{dec}}\,{k^{\text{inv}}}^\top\big).
$$

**④ 块内求逆** $P=(I+L)^{-1}$
（$L$ 严格下三角且已乘 $\beta$）。FlashKDA 对 $16\times16$ 用「两个 $8\times8$ 的 fp32 前代 + 一次
bf16-HMMA 的 off-diagonal 合并」精确求逆（`inv_fwd_subst_16` / `fwd_kernel1.cuh` 的
`inv` 段），比 Neumann 级数在 $|L|\to1$ 时更稳。

**⑤ 修正后的 value 与输出**

$$
\tilde v_i = \big(v_i - k^{\text{dec}}_i S_{\text{in}}^\top\big)\,\beta_i,\qquad
U = P\,\tilde v,
$$
$$
o_i = q^{\text{dec}}_i S_{\text{in}}^\top + (M_{qk}U)_i .
$$

**⑥ 状态更新**

$$
\Delta S = {k^{\text{rest}}}^\top U,\qquad
S_{\text{out}} = \big(\Delta S + S_{\text{in}}^\top \odot E^{\text{tot}}\big)^\top
= E^{\text{tot}}^\top\!\odot S_{\text{in}} + \Delta S^\top .
$$

直觉：$E^{\text{tot}}$ 把整块累积衰减作用到旧状态（块内所有 token 的衰减一次性结算），
$\Delta S$ 是本块「delta 写回」的增量。

> 说明：以上分块公式与逐 token 递推（式 1）在 fp32 下等价，差异只在数值与计算顺序。
> 反向的推导、以及 chunked backward 如何复用 $L$ / $P$ / $M_{qk}$，见后续篇目（`05-kda-bwd.md`）。

## 4. 为什么拆成两个 kernel（K1 / K2）

FlashKDA 把前向拆成两个 kernel（见 `docs/20260420-flashkda-v1-deep-dive.md` §2）：

- **K1（token 并行，grid = $N\times H\times\text{num\_chunks}$）**：门激活 → L2 归一化 → 衰减施加
  → 构造 $L$ / $M_{qk}$ → 块内求逆；把 $k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},g^{\text{total}},
  P,M_{qk}$ 写进 workspace。
- **K2（head 并行，grid = $N\times H$）**：逐块跑 delta-rule 递推 → 输出投影 → 状态累加。

单 kernel 版本里，K1 的 token 并行度会被 K2 很低的并行度拖住，大量 SM 空转；拆开后至少快 **15%**，
而且两段可各自调优。这里是后文精读 K1/K2 的动机。

## 5. FlashKDA 源码结构脑图

这一节把仓库里每个文件「负责什么、边界在哪」先钉死，后面精读 K1/K2 时不再迷路。
所有行号均为核对过的真实行号（`~/github/FlashKDA`）。

### 5.1 文件与职责

| 文件 | 职责 | 关键入口 |
|---|---|---|
| `csrc/flash_kda.cpp` | host 侧 API：校验、workspace 大小、形状规整、按状态/varlen 分派 | `get_workspace_size`（L5）、`fwd`（L28） |
| `csrc/fwd.h` | launch 模板声明（$D$/状态/varlen 作为编译期参数） | `launch_fwd`（L6） |
| `csrc/smxx/fwd_launch.cu` | 建 TMA descriptor、切 workspace 指针、真正 `<<<>>>` 两次 | `launch_fwd`（L7） |
| `csrc/smxx/fwd_kernel1.cuh` | **K1 prepare**：门激活/归一化/衰减/构 $L,M_{qk}$/求逆 → 写 workspace | `_flash_kda_fwd_prepare`（L120） |
| `csrc/smxx/fwd_kernel2.cuh` | **K2 recurrence**：逐 chunk delta 递推 + 输出 + 状态更新 | `_flash_kda_fwd_recurrence`（L133） |
| `csrc/smxx/utils.cuh` | PTX 近似函数、workspace 尺寸、warp 角色、pipeline 工厂、求逆/状态转换例程 | 见 §5.4 |

「host → descriptor → kernel」三层是分离的：`flash_kda.cpp` 只做 tensor 检查与 reshape，不碰
cute/TMA；所有 descriptor 构造都在 `fwd_launch.cu`，kernel 模板只收 descriptor 常量。

### 5.2 K1（prepare）阶段流

`_flash_kda_fwd_prepare`（`fwd_kernel1.cuh:120`）是**单次 TMA load、无 pipeline** 的短 kernel，
一个 CTA 处理一个 `(head, tile)`：

1. **定位 tile**（L168–199）：`grid = total_tiles × H`。非 varlen 直接整除（L187–195）；
   varlen 用预生成的 `tile_prefix` 做 $O(\log N)$ 二分（L175–186）。多出的 CTA 在 L199 提前 `return`。
2. **TMA 载入**（L200–255）：`threadIdx.x==0` 一次发起 q/k/beta/g_bf16/dt_bias 五个 copy（L231–254），
   `kTmaTransactionBytes` 在 L158–161 累加。注意 beta 做了 `& ~7` 对齐（L223），因为 1D TMA 的最小粒度是 8。
3. **归一化**（L265–304）：$D=128$ 行内 16 线程、每线程 8 元素，先逐元素平方和，
   再 5 步 `__shfl_xor_sync` 树归约（L290–293），`rsqrtf(x+1e-6)`（L295–296），回写。
4. **门激活 + cumsum**（L306–338）：128 线程按列并行，`g = a_log_exp*g + dt_bias` →
   `gate_scale*sigmoid`（L321–323）→ 逐行累加得 prefix（L327–328），末值写 `g_total`（L330）。
   另 128 线程把尾部越界行的 k 置零（L331–337）。**没有任何跨行同步**——因为按列并行天然串行。
5. **decay_apply**（L360–475）：把 q/k 乘 $2^{g^{cum}}$ 等四种变体写回（L449–471）。
   用 union 复用 smem（下面 §5.3）。
6. **$L$ / $M_{qk}$ 两个 GEMM**（L481–487）：warp 0 算 fp32 累加的 $L$，warp 1 算 bf16 的 $M_{qk}$。
7. **tril 掩码 + beta**（L493–508）：$L$ 严格下三角乘 $\beta$（L499–503），$M_{qk}$ 抹上三角（L504–506）。
8. **求逆**（L510–511）：`inv_fwd_subst_fused_1warp`（`utils.cuh:214`）。
9. **写 workspace**（L515–583）：`threadIdx.x==0` 连发 6 个 TMA store，`tma_store_wait<0>`（L584）。

### 5.3 K2（recurrence）阶段流与 warp 分工

`_flash_kda_fwd_recurrence`（`fwd_kernel2.cuh:133`），`grid = N × H`，**一个 CTA 串完整个序列**，
所以是 head 并行、token 串行。

- **warp 角色**（L188–197）：`kComputeThreads=128`（4 个 MMA warp），第 4 个 warp 是 `LOAD_QKG`，
  第 5 个是 `STORE`；线程总数 `32*2+128=192`（`fwd_launch.cu:186`）。
- **双 pipeline**（L199–213）：load 用 `PipelineTmaAsync<3>`（3 stages），store 用 `PipelineAsync<2>`（2 stages），
  pipeline 工厂见 `utils.cuh:88`/`utils.cuh:120`。
- **初始状态**（L239–318）：bf16 直接 TMA 进 `state_acc`（L241–266）；fp32 先进 union 缓冲再
  `smem_cvt_fp32_to_bf16`（L267–304）；无状态则清零（L305–317）。
- **LOAD warp**（L323–422）：按 stage 轮转，一次发起 v/beta/k_decayed/q_decayed/k_restored/g_total/INV/Mqk
  共 8 个 TMA（L346–420）。
- **MMA warp 的六个 Phase**（L426–743）：
  1. 双 GEMM `k@s`、`q@s`（L533–568）；
  2. cast 输出到 bf16 并载入 v/INV/beta（L570–587）；
  3. $u=(v-u)\beta$，再 `u = INV @ u`，用 `MOVM_T` 把 u 从 C 片段直接当 B 操作数（L589–623）；
  4. `Mqk @ U + out`（L625–650）；
  5. 写 out 回 smem（L652–657）；
  6. 状态更新 `s_acc = s_acc*g_total + k_restored^T @ U`（L659–731）。
  之后 `compute_barrier.arrive_and_wait()` + 提交 pipeline（L733–741）。
- **STORE warp**（L746–802）：整 tile 走 TMA store（L767–779），尾 tile 手动逐元素写（L757–766）；
  最后按 `HasStateOut`/`StateFP32` 分支存 state（L786–835）。

### 5.4 `utils.cuh` 的可复用小件

- 近似函数：`ex2_approx_ftz_f32`（L38）、`tanh_approx_f32`（L44）、`sigmoid_tanh_approx_f32`（L50）、
  `bf16_to_f32`（L55）。这些是 fp32 精度的关键——gate 路径全走它们。
- `WorkspaceSizes`（L64–77）：与 `flash_kda.cpp:get_workspace_size` 的逐字段字节数一一对应
  （`kKDecayed=kQDecayed=kKRestored=4096`、`kGTotal=512`、`kINV=kMqk=512`）。
- `WarpRole`（L79–84）与 pipeline 工厂 `make_load_pipeline`（L88）/`make_store_pipeline`（L120）。
- 单 warp 16×16 GEMM 封装：`mma_m16n16_bf16bf16bf16_1warp`（L146，$M_{qk}$ 用）、
  `mma_m16n16_bf16bf16fp32_1warp`（L168，$L$ 用 fp32 累加）。
- `inv_fwd_subst_fused_1warp`（L214）：**K1 最烧脑的一段**——8×8 fp32 前代（L229–252）+
  两次 bf16 HMMA 合并 off-diagonal（L330–341）+ 元素加（L343–355）。
- 状态 layout 转换 `smem_cvt_fp32_to_bf16`（L379）/`smem_cvt_bf16_to_fp32`（L410），按 8×8 atom 处理。

### 5.5 workspace 布局（对上 host 计算）

`get_workspace_size`（`flash_kda.cpp:5`）与 `launch_fwd` 的指针切分（`fwd_launch.cu:62–71`）必须一致：

$$
\text{per\_tile} = \underbrace{3\cdot CHUNK\cdot D\cdot2}_{k^{dec},q^{dec},k^{rest}}
+ \underbrace{D\cdot4}_{g^{total}}
+ \underbrace{2\cdot CHUNK^2\cdot2}_{INV,\,M_{qk}},
$$

按 `kKDecayed → kQDecayed → kKRestored → kGTotal → kINV → Mqk` 六段分开排布（不是结构体数组），
最后留 `tile_prefix`（varlen 用，L22–23、L71）。**这个「SoA 而非 AoS」的排布让每个字段能独立配 TMA
descriptor，且都能 128B 对齐**（L16–18 的 `static_assert`）。

## 6. 数值自检：逐 token 递推 vs 分块（P1-4）

`src/kda_ref.py` 里放了两个**独立**的纯 PyTorch 前向：`kda_token_recurrent`（式 (1) 的逐 token
gated delta rule，作 golden）与 `kda_chunked_fwd`（§3 的 CHUNK=16 分块公式）。二者用完全不同的
计算顺序（逐 token 串行 vs 块内矩阵求逆），因此互拍能验证分块代数的正确性。把状态布局统一为
$S\in\mathbb{R}^{K\times V}$、输出 $o_t=\text{scale}\cdot q_tS_t$ 后：

- **fp64**：在所有试探的 $(T,H)\in\{(1,1),(16,1),(20,2),(48,3)\}$、有无初态、$g$ 尺度
  $\{0,1,4\}$ 组合下，两种实现的 $o$ 与终态 $S$ **逐元素完全相等**（max_abs $=0$）——分块公式与
  逐 token 递推在实数域严格等价。
- **fp32**：同样的组合下 max_abs $\sim10^{-9}\!\sim\!10^{-7}$、max_rel $\sim10^{-8}\!\sim\!10^{-7}$
  （机器精度量级），即误差只来自两种计算顺序的浮点舍入。

进一步与 FLA 的 Triton `chunk_kda`（`harness/check_kda_ref.py`）对拍：同一输入（$D=128$，
$T=20/64$，$H=2$，非零初态，raw logits 走 `use_gate_in_kernel` + `use_beta_sigmoid_in_kernel`），
bf16 计算下 $o$ 的 max_rel $\approx4.7\times10^{-3}\!\sim\!5.9\times10^{-3}$、终态 $S$ 的 max_rel
$\approx3.1\times10^{-3}\!\sim\!6.0\times10^{-3}$，与 bf16 输入本身的量化误差同量级。原始输出见
`harness/kda_ref_selftest.out.txt` 与 `harness/check_kda_ref.out.txt`。

**一个踩过的坑（layout）**：FLA `chunk_kda` 的 `transpose_state_layout=True`（新版改名
`state_v_first`）不仅改变**返回**终态的布局，也会把传入的 `initial_state` 当作 V-first 重新解释。
最初误用该开关后，除 $t=0$（唯一直接吃初态的 token）外一切正常、$t=0$ 误差却高达 $0.1$——误导性地
像是「随机小误差」。默认 `transpose_state_layout=False` 时终态就是 $[N,H,K,V]$，与本文 $S$ 同布局，
对拍一致。**已在 P4-2 钉死**：FlashKDA 的 `initial_state`/`final_state` 用 $S^\top$ 布局（$[N,H,V,K]$，
`torch_ref.py` 里 `state_slice` 存的就是 $S^\top$），与 K2 内部 `s_acc`（$[V,K]$）及包装文档串
（`flash_kda/__init__.py:21`）一致；默认 `transpose_state_layout=False` 时 FLA 终态是 $[N,H,K,V]$、与本文
$S$ 同布局。两者互为转置——直接把 kernel `final_state` 与 golden $S$ 比 `max_abs=0.511`，转置后再比只剩
`0.00398`（`harness/probe_state_layout.py`，见 `docs/04` §3）。

## 7. 下一步

- 精读 K1（`docs/02-kda-fwd-k1.md`）：TMA 载入、L2 norm、cumsum、decay、$L/M_{qk}$、求逆。
- 精读 K2（`docs/03-kda-fwd-k2.md`）：warp specialization、八个 Phase、状态更新。
- 数值对拍（`docs/04`）与反向（`docs/05`）。
