# KDA 前向数值对拍：kernel vs fp64 golden

> 本文是「Kimi Delta Attention 前向 / 反向算子实现解析」的第 4 篇，对应路线图 **P4**。
> 上一节（`docs/03`）把 K2 的八个 Phase 与状态递推拆到底层 bf16 操作，本篇回答一个更实证的问题：
> **编译好的 FlashKDA 前向 kernel 到底有多准？** 我们以一个纯 PyTorch、fp64 的逐 token
> `gated delta rule` 作为 golden，在多种序列长度、head 数、batch、随机种子与门控区间下做端到端对拍，
> 给出真实的 max abs / max rel 误差，并把误差归因到 bf16 输入舍入与 bf16 状态存储。

## 1. 对拍对象与 golden（P4-1）

**被检对象**：`~/github/FlashKDA` 编译出的 `flash_kda.fwd`（CUDA，SM90，bf16，`CHUNK=16`，`D=128`），
经由 `flash_kda/__init__.py` 的 Python 包装调用（内部自动分配 workspace）。输入 `q,k,v,g` 均为 bf16，
`beta` 为激活前 logits（kernel 内做 sigmoid），`A_log`/`dt_bias` 为 fp32。

**golden**：`src/kda_ref.py::kda_token_recurrent`，在 **fp64** 下逐 token 递推：

$$
S_t = \operatorname{diag}(2^{g_t})\,S_{t-1} + k_t \otimes \big(\beta_t\,(v_t - (k_t\odot 2^{g_t})\,S_{t-1})\big),
\qquad o_t = \text{scale}\cdot q_t S_t,
$$

其中状态 $S\in\mathbb{R}^{K\times V}$，`q,k` 先做 fp64 的逐行 $\ell_2$ 归一化，门控按 kernel 的定义构造
（`src/kda_ref.py:32` 的 `kda_gate`）：

$$
g \leftarrow \text{lower\_bound}\cdot\log_2 e\;\cdot\;\sigma\big(e^{\text{A\_log}}\cdot(g+\text{dt\_bias})\big)\le 0 .
$$

选择 fp64 golden 而不是 fp32，是为了让「参考」本身的浮点误差（$\sim10^{-7}$）远小于 kernel 的
bf16 误差（$\sim10^{-3}$），使测得的误差就是 kernel 的误差。golden 与 kernel 送入**同一份** fp32 随机
张量：kernel 侧取它的 bf16 舍入值，golden 侧取它的 fp64 值——因此 bf16 输入舍入也被计入端到端误差。

## 2. 口径

本文所有误差用三个量：`max_abs` $=\max|\text{pred}-\text{gold}|$，
`max_rel` $=\max|\text{pred}-\text{gold}| \,/\, \max|\text{gold}|$（**以 golden 的最大幅值为分母**，
即峰值相对误差，而非逐元素相对误差，避免分母近零放大），`mean_abs` 为平均绝对误差。
脚本 `harness/check_fwd_numerics.py`，原始输出 `harness/check_fwd_numerics.out.txt`。

## 3. 关键发现：状态布局是 $S^\top$（$[V,K]$），不是 $[K,V]$

第一版对拍时出现一个典型现象：`o` 的 `mean_abs` 只有 $\sim2\times10^{-4}$，但 `max_abs` 却高达
$1.7\times10^{-1}$、`max_rel` 超过 1。`harness/probe_state_layout.py` 把三种情形并列复现
（原始输出 `harness/probe_state_layout.out.txt`）：

```
h0=zero:                  o max_abs=2.28e-04   per-token: 2.3e-4 4.5e-5 3.1e-5 ...  (全程 bf16 量级)
h0=rand (布局正确，已转置): o max_abs=5.47e-04   per-token: 5.5e-4 2.2e-4 5.9e-5 ...
h0=rand (布局错误，未转置): o max_abs=1.62e-01   per-token: 1.6e-1 5.0e-2 1.8e-2 5.1e-3 ... 1.1e-3
```

即：零初态时误差就是 bf16 量级；随机初态若**不转置**，误差在 t=0 高达 $1.6\times10^{-1}$，
但在**头 16 个 token 内随门控衰减**迅速收敛到 bf16 量级。

这种「只在初态影响处出错、且随门控衰减收敛」的形态，指向**初态布局不一致**而非 kernel 算错。
进一步直接比状态矩阵：把 kernel 的 `final_state` 与 golden 的 $S$ 直接比 `max_abs=0.511`，
转置后比只有 `0.00398`——即 kernel 存的是 $S^\top$。

结论：**FlashKDA 的 `initial_state`/`final_state` 用 $S^\top$（$[N,H,V,K]$）布局**，
与 K2 内部 `s_acc`（$[V,K]$，见 `docs/03` §8）以及包装函数文档串
（`flash_kda/__init__.py:21`「Shape `[B, H, V, K]`」）一致；而 `src/kda_ref.py` 用 $[K,V]$。
因此对拍时要把初态转置成 $S^\top$ 传入、并把 golden 的终态转置成 $S^\top$ 再比对。
这也解释了为什么 P0-3 的 `probe_refs.py` 直接拿方阵 `h0` 传入仍能对上 FLA——那里两边
（FlashKDA 与 FLA `transpose_state_layout=True`）都在用 $V$-first，属于口径统一后的巧合一致；
真正要紧的是先钉死布局再谈精度。

> 经验：状态是方阵 $D\times D$ 时，布局错误**不会**在形状检查里暴露，只会表现为「初态处误差大、
> 有衰减时又自动收敛」。对拍必须先用零初态与转置比对把布局钉死，再谈精度。

## 4. 结果：FlashKDA vs fp64 golden（P4-2）

固定 `lower_bound=-5.0`、`scale=1/√128`，随机 `A_log`/`dt_bias`，多 shape / 多种子 / 多门控区间。
`o` 与状态（按 $S^\top$）误差如下（`harness/check_fwd_numerics.out.txt`）：

| case | o.max_abs | o.max_rel | S.max_abs | S.max_rel |
|---|---|---|---|---|
| T=16 | 8.37e-04 | 6.22e-03 | 5.22e-03 | 6.75e-03 |
| T=17 | 8.37e-04 | 6.22e-03 | 5.11e-03 | 7.31e-03 |
| T=64 | 8.37e-04 | 6.22e-03 | 2.96e-03 | 4.98e-03 |
| T=128 | 8.37e-04 | 6.22e-03 | 6.03e-03 | 8.74e-03 |
| T=512 | 8.37e-04 | 6.22e-03 | 5.29e-03 | 7.34e-03 |
| T=1024 | 8.37e-04 | 6.22e-03 | 5.48e-03 | 8.62e-03 |
| H=1 | 5.47e-04 | 5.32e-03 | 2.82e-03 | 9.28e-03 |
| H=2 | 5.47e-04 | 4.13e-03 | 3.78e-03 | 9.70e-03 |
| H=8 | 7.85e-04 | 5.92e-03 | 5.62e-03 | 7.76e-03 |
| B=3 | 1.00e-03 | 7.14e-03 | 5.59e-03 | 9.99e-03 |
| seed=1 | 7.52e-04 | 3.56e-03 | 4.59e-03 | 9.44e-03 |
| seed=2 | 9.77e-04 | 5.66e-03 | 3.94e-03 | 9.44e-03 |
| seed=3 | 1.12e-03 | 6.18e-03 | 7.11e-03 | 1.14e-02 |
| g=0（无衰减） | 8.40e-04 | 6.93e-03 | 1.43e-02 | 8.07e-03 |
| g=N(0,1) | 9.51e-04 | 7.59e-03 | 5.56e-03 | 7.33e-03 |
| g=N(0,4) | 7.51e-04 | 5.05e-03 | 4.40e-03 | 7.76e-03 |
| T=1023 | 7.72e-04 | 4.78e-03 | 4.76e-03 | 7.55e-03 |

几点观察：

- **输出 `o` 的峰值相对误差稳定在 $4\times10^{-3}\sim7.6\times10^{-3}$**，全部落在 bf16 有效精度
  （$\sim2^{-8}=3.9\times10^{-3}$）附近，与机器精度的预期一致；
- **状态 `S` 误差略大**（$5\times10^{-3}\sim1.4\times10^{-2}$），因为状态沿 $T$ 累积、且 K2 用 bf16
  存储 `s_acc`（见 `docs/03` §18：把状态 fp32→bf16 存储的边际输出变化本就只有 $\sim10^{-4}$，
  但终态本身直接受 bf16 存储影响）；
- **`o.max_abs` 在不同 `T` 下完全相同（`8.367e-04`）**：因为 seed 相同、峰值误差出现在**早期 token**，
  与序列总长无关；这也说明误差不随 $T$ 漂移（无累积爆掉），递推是数值稳定的；
- **`g=0`（无衰减）时状态误差升到 `1.43e-02`**：无衰减时状态是「累加」而非「遗忘」，bf16 舍入在
  $T$ 步上累积更充分，这与 P3-3 §18 的受控消融结论（`g=0` 病态区间 bf16 存储边际升到 $\sim2\times10^{-3}$）
  方向一致；
- **多 batch（B=3）、多 head（H=1/2/8）、不同种子**误差都在同一量级，无异常放大，说明 kernel 的
  varlen/head-major 调度没有引入额外误差。

## 5. 误差归因

把误差拆成三层来源（前两层在 `docs/02`/`docs/03` 已有受控实测，此处与端到端结果互洽）：

1. **bf16 输入舍入**（主导）：`q,k,v,g` 以 bf16 送入，归一化后的 `q,k` 再被舍入到 bf16
   （`torch_ref.py:158–159`，kernel 亦然），单次 $\sim3.9\times10^{-3}$，是端到端 $o$ 误差的主项。
2. **bf16 状态存储**：K2 的 `s_acc` 以 bf16 保存（`docs/03` §18），边际贡献 $\sim10^{-4}$（有衰减）；
   它主要影响终态 `S` 而非输出。
3. **门控近似**：sigmoid 用 `tanh.approx`、衰减用 `ex2.approx`，端到端相对误差 $\le2\times10^{-5}$
   （`docs/02` §11c），比 bf16 舍入小两个量级，可忽略。

因此端到端误差本质上由**数据本身的 bf16 表示**决定，kernel 的算法（分块、求逆、`MOVM_T` 等）没有
引入可测量的额外损失。

## 6. 三方对比：golden / FlashKDA / FLA（P4-3）

上一节只有「kernel vs fp64 golden」两方。本节引入第三个独立实现——FLA 的 Triton
`fla.ops.kda.chunk_kda`（`chunk_size=64`）——在**同一批输入**上做三方对拍，脚本
`harness/check_three_way.py`、原始输出 `harness/check_three_way.out.txt`。

三个实现的口径：三者共享同一份 fp32 随机张量；golden 取 fp64 值（`src/kda_ref.py::kda_token_recurrent`），
FlashKDA 与 FLA 各取自己的 bf16 舍入值。状态统一在 $S^\top$（$[V,K]$）布局下比较
（FlashKDA 原生即 $S^\top$；FLA 默认 $[N,H,K,V]$，转置后比较；golden 用 $[K,V]$，同样转置）。
`max_rel` 的分母为各自 pair 中「参照方」的最大幅值。

### 6.1 误差表

`o` 与终态 $S^\top$ 的 `max_abs / max_rel`（前缀 `fl`=FlashKDA，`fla`=FLA）：

| case | o fl−gold | o fla−gold | o fl−fla | S fl−gold | S fla−gold | S fl−fla |
|---|---|---|---|---|---|---|
| T=64 | 8.37e-04/6.22e-03 | 8.37e-04/6.22e-03 | 4.88e-04/3.65e-03 | 2.96e-03/4.98e-03 | 2.93e-03/4.93e-03 | 3.56e-03/6.00e-03 |
| T=128 | 8.37e-04/6.22e-03 | 8.37e-04/6.22e-03 | 4.88e-04/3.65e-03 | 6.03e-03/8.74e-03 | 7.06e-03/1.02e-02 | 5.73e-03/8.22e-03 |
| T=256 | 8.37e-04/6.22e-03 | 8.37e-04/6.22e-03 | 4.88e-04/3.65e-03 | 4.22e-03/8.81e-03 | 3.72e-03/7.76e-03 | 4.47e-03/9.34e-03 |
| T=512 | 8.37e-04/6.22e-03 | 8.37e-04/6.22e-03 | 4.88e-04/3.65e-03 | 5.29e-03/7.34e-03 | 4.01e-03/5.56e-03 | 4.91e-03/6.84e-03 |
| T=1024 | 8.37e-04/6.22e-03 | 8.37e-04/6.22e-03 | 4.88e-04/3.65e-03 | 5.48e-03/8.62e-03 | 3.01e-03/4.74e-03 | 3.95e-03/6.23e-03 |
| H=1 | 5.47e-04/5.32e-03 | 5.05e-04/4.91e-03 | 4.88e-04/4.74e-03 | 2.82e-03/9.28e-03 | 2.04e-03/6.71e-03 | 3.31e-03/1.10e-02 |
| H=2 | 5.47e-04/4.13e-03 | 5.05e-04/3.81e-03 | 4.88e-04/3.68e-03 | 3.78e-03/9.70e-03 | 2.79e-03/7.15e-03 | 3.12e-03/8.00e-03 |
| H=8 | 7.85e-04/5.92e-03 | 8.06e-04/6.08e-03 | 4.88e-04/3.68e-03 | 5.62e-03/7.76e-03 | 4.43e-03/6.11e-03 | 5.49e-03/7.60e-03 |
| B=3 | 9.99e-04/7.14e-03 | 5.59e-04/4.00e-03 | 9.77e-04/6.99e-03 | 5.59e-03/9.99e-03 | 2.79e-03/4.98e-03 | 6.35e-03/1.14e-02 |
| seed=1 | 7.52e-04/3.56e-03 | 6.52e-04/3.09e-03 | 9.77e-04/4.63e-03 | 4.59e-03/9.44e-03 | 3.76e-03/7.74e-03 | 3.84e-03/7.88e-03 |
| seed=2 | 9.77e-04/5.66e-03 | 5.66e-04/3.28e-03 | 9.77e-04/5.65e-03 | 3.94e-03/9.44e-03 | 2.46e-03/5.89e-03 | 3.75e-03/8.96e-03 |
| seed=3 | 1.12e-03/6.18e-03 | 5.99e-04/3.32e-03 | 9.77e-04/5.41e-03 | 7.11e-03/1.14e-02 | 5.23e-03/8.41e-03 | 5.82e-03/9.37e-03 |
| g=0（无衰减） | 8.40e-04/6.93e-03 | 7.05e-04/5.82e-03 | 9.77e-04/8.10e-03 | 1.43e-02/8.07e-03 | 8.34e-03/4.72e-03 | 1.46e-02/8.27e-03 |
| g=N(0,1) | 9.51e-04/7.59e-03 | 5.00e-04/3.99e-03 | 4.88e-04/3.91e-03 | 5.56e-03/7.33e-03 | 3.40e-03/4.48e-03 | 4.47e-03/5.88e-03 |
| g=N(0,4) | 7.51e-04/5.05e-03 | 6.60e-04/4.44e-03 | 9.77e-04/6.58e-03 | 4.40e-03/7.76e-03 | 2.72e-03/4.79e-03 | 3.05e-03/5.38e-03 |
| T=1023 | 7.72e-04/4.78e-03 | 7.31e-04/4.53e-03 | 9.77e-04/6.06e-03 | 4.76e-03/7.55e-03 | 3.95e-03/6.25e-03 | 4.88e-03/7.73e-03 |

### 6.2 观察与「为什么」

- **三方全部落在 bf16 有效精度（$\sim2^{-8}=3.9\times10^{-3}$）同一量级**：`o` 的三方 `max_rel`
  都在 $3\times10^{-3}\sim8\times10^{-3}$，状态在 $5\times10^{-3}\sim1.1\times10^{-2}$。这三个实现分别用
  逐 token fp64、SM90 HMMA 分块（CHUNK=16）、Triton 分块（chunk=64），算法路径完全不同，却给出同量级
  误差——说明误差**由 bf16 数据表示决定**，而非某一实现的算法/调度缺陷。
- **`flash vs fla` 的 `o` 误差小于任一者 `vs gold`**（T 扫描下稳定为 `4.88e-04/3.65e-03`，
  而两者各自 `vs gold` 都是 `8.37e-04/6.22e-03`）。原因是**共模抵消**：两个 bf16 实现送入的是**同一份
  bf16 量化后的 `q,k,v,g`**，它们都忠实逼近「bf16 输入版本的精确解」；而 fp64 golden 逼近的是
  「fp32/fp64 输入版本的精确解」。于是 `flash↔fla` 之差把**共有的 bf16 输入舍入误差**减掉了，只剩
  两者算法/累加序的差异。反过来说，`vs gold` 的峰值误差里，bf16 输入舍入是主项（与 §5 归因一致）。
- **T 扫描下 `o.fl−gold` 恒为 `8.367e-04`**：seed 固定、峰值误差出现在早期 token，与序列长度无关；
  这与 §4 的观察一致，也说明 FLA 与 FlashKDA 都用了相同的输入舍入，故该峰值两列完全重合。
- **状态误差普遍大于输出**，且 `flash−fla` 与各自 `vs gold` 同量级：输出是「当前 token 对状态的读」，
  误差被单步加权；终态则是 $T$ 步递推的累积，且两个实现的状态存储/规约顺序不同（FlashKDA 逐块 bf16
  `s_acc`，FLA chunk=64），故状态上的共模抵消不像输出那样彻底。
- **`g=0`（无衰减）是唯一三方误差都显著抬升的区间**：`S` 的 `fl−gold=1.43e-02`、`fla−gold=8.34e-03`、
  `fl−fla=1.46e-02`。无衰减时状态是纯累加，bf16 舍入在 $T$ 步上无处遗忘而充分累积；此时连
  `flash−fla` 都不再比 `vs gold` 小，因为两个实现的状态舍入噪声**相互独立**、不再共模。
  这与 §4、以及 P3-3 §18 的受控消融方向完全一致。
- **多 head / 多 batch / 多种子下三方误差无异常放大**，说明 FlashKDA 的 head-major、varlen 调度
  没有引入超出 bf16 底噪的额外误差。

> 结论：以 fp64 逐 token 递推为基准，FlashKDA 与独立的 FLA 实现**互相印证**——同一输入下两者之差
> 不超过各自相对 golden 的误差，且差值主要来自算法路径（分块大小、累加序）而非某一方算错。

## 7. varlen（packed `cu_seqlens`）前向的数值锚定（P4-4）

前几节（P4-2/P4-3）的数值锚定**只跑过 batched 路径**（$B$ 条等长 $T$ 的序列）。FlashKDA 还有另一条
host 分派路径：**varlen / packed**——`cu_seqlens` 给出拼接后各序列边界、`B=1`、序列长度不等。此前对
它唯一的验证是**字节账**（`harness/check_workspace_layout.py` 的 Part C：`tile_prefix` 的 int32 数组
落在预测偏移、canary 未被越界写），**从没验证过数值**：packing 后每条序列的输出/终态是否真的等于它
单独跑一遍？本节用 `harness/check_varlen_forward.py`（原始输出 `harness/check_varlen_forward.out.txt`）
把它钉死。

### 7.1 varlen 路径与 batched 路径的差别只在「tile → (序列, 块) 映射」

- batched：host 已知每序列等长，`total_tiles = N·ceil(T_seq/16)`（精确，`flash_kda.cpp:180`），
  kernel 直接整除定位（`fwd_kernel1.cuh` 的 `IsVarlen=false` 分支）。
- varlen：长度不等、host 无法预知每序列几块，于是**预生成 `tile_prefix`**（各序列累计 tile 数的前缀和），
  kernel 用 $O(\log N)$ 二分把「第几个 tile」映射回「哪条序列的第几块」（`fwd_kernel1.cuh` 的
  `IsVarlen=true` 分支）。host 只按**上界** `ceil(T_total/16)+N` 分配与 launch（`flash_kda.cpp:178`），
  多出的 CTA 在 kernel 头部 `return`。

关键点：**两条路径的「逐 tile 算术」完全相同**——varlen 只改「CTA 从哪些 token 取数、写到哪」，不改
每个 tile 内做的 L2 归一化、cumsum、decay、$L/M_{qk}$ 求逆、GEMM 与状态更新。所以只要 tile→序列的
映射没错，packed 结果应当与「逐序列单独跑」**逐位相同**。这正是本节最锋利的判据。

### 7.2 Part A：packed vs 逐序列 batched —— **逐位相同（`max_abs=0`）**

对每组 `lens` 构造 packed 输入 `[1,T_total,H,D]` + `cu_seqlens` + 每序列 `initial_state [N,H,D,D]`，
跑一次 varlen；再对每条序列切片，用 `B=1` 的普通 batched 路径（同一份输入、同一个 `h0`、同 `scale`）
跑一次。逐序列比输出切片与 `final_state` 行：

| case | `T_total` | `N` | `o` max_abs | `S` max_abs |
|---|---|---|---|---|
| `lens=[30,16,5,50]` | 101 | 4 | **0.000e+00** | **0.000e+00** |
| `lens=[1,16,17,64,3]`（含 $L{=}1<$CHUNK） | 101 | 5 | **0.000e+00** | **0.000e+00** |
| `lens=[64,64,64]`（均为 16 倍数） | 192 | 3 | **0.000e+00** | **0.000e+00** |
| `lens=[250,7]`（$T$ 非 16 倍数） | 257 | 2 | **0.000e+00** | **0.000e+00** |
| `lens=[33,47,19,100]` | 199 | 4 | **0.000e+00** | **0.000e+00** |
| `g=0`（无衰减） | 101 | 4 | **0.000e+00** | **0.000e+00** |
| `g=N(0,4)`（强衰减） | 101 | 4 | **0.000e+00** | **0.000e+00** |

七组（含单 token 序列、非整除 $T$、极端门控）全部**逐位相同**。这同时排除了三类易错：

1. **tile→序列映射错**（二分写错）：会让某条序列读到邻居的 token → O(1) 差；
2. **序列间状态泄漏**（一条序列的 `s_acc` 灌进下一条）：在 packed 里表现为「后一条序列输出/终态偏移」，
   单跑时不会；
3. **尾块记账错**（非整除 $T$、$L<$CHUNK）：尾块 mask / `tile_prefix` 边界错误 → 只在特定 `lens` 现形。

七组同时逐位，等价于「varlen 调度是 batched 的**无损重排**」。

### 7.3 Part B：packed vs fp64 golden —— 与 batched 同处 bf16 底噪

同七组 packed（`HasStateIn/Out` 均开、bf16），逐序列与 fp64 token-recurrent golden 比 `max_rel`
（分母取 golden 峰值）：

| case | `o` max_rel | `S^T` max_rel |
|---|---|---|
| `lens=[30,16,5,50]` | 6.06e-03 | 1.02e-02 |
| `lens=[1,16,17,64,3]` | 7.98e-03 | 8.54e-03 |
| `lens=[64,64,64]` | 6.32e-03 | 9.34e-03 |
| `lens=[250,7]` | 5.59e-03 | 8.53e-03 |
| `lens=[33,47,19,100]` | 7.17e-03 | 1.18e-02 |
| `g=0` | 5.81e-03 | 5.61e-03 |
| `g=N(0,4)` | 6.38e-03 | 1.25e-02 |

全部落在 §4 的 batched 带宽内（`o` $4\text{–}7.6\times10^{-3}$、`S` $5\times10^{-3}\text{–}1.4\times10^{-2}$）。
即 **packing 不引入任何超出 bf16 底噪的额外误差**——与 Part A 的「逐位无损重排」互为印证。

### 7.4 Part C：每序列的 `initial_state` 是按序列索引的（不是共享）

varlen 的 `initial_state`/`final_state` 形状是 `[N,H,D,D]`（`flash_kda.cpp:162-174` 强制）。若 kernel
误用第一条序列的 $h_0$ 当共享初态，packed 结果仍会「自洽」、只有与「逐序列单独跑」比才露馅。用
「distinct-$h_0$ vs shared-$h_0$」的差做判据，对每条序列比 **packed 差** 与 **孤立差**：

| 序列（长度） | 孤立 `|o(h0_i)−o(h0_0)|` | packed 同值 | 逐位相同 |
|---|---|---|---|
| 0（30） | 0.000e+00 | 0.000e+00 | 是 |
| 1（16） | 1.782e-01 | 1.782e-01 | 是 |
| 2（5） | 2.236e-01 | 2.236e-01 | 是 |
| 3（50） | 2.432e-01 | 2.432e-01 | 是 |

`seq0` 两口径都 $h_0$ 未变故差 0；其余序列 packed 差与孤立差**逐位相同**，说明 packed 里序列 $i$ 真的
吃的是 $h_0[i]$。终态侧：`seq0` 终态在两种口径下逐位相同（它的初态没变），而 `seq1` 终态因初态不同
移动了 **5.104e-02**（O($h_0$) 量级）——再次证明初态按序列生效。

### 7.5 Part E：varlen 模板的四种状态组合

`fwd_kda` 对 `HasStateIn/HasStateOut` 各有一版编译好的 varlen 模板（`flash_kda.cpp:209-213` 的
`DISPATCH_STATE(true)`）。四组合都相对「逐序列 batched」**逐位相同**：

```
no state         o vs per-seq batched max_abs=0.000e+00  (no S out)
state-in only    o vs per-seq batched max_abs=0.000e+00  (no S out)
state-out only   o vs per-seq batched max_abs=0.000e+00  S max_abs=0.000e+00
both (bf16)      o vs per-seq batched max_abs=0.000e+00  S max_abs=0.000e+00
```

### 7.6 边界（诚实）

- **`B==1` 硬约束**（`flash_kda.cpp:150`）：`cu_seqlens` 给的是拼接后的序列边界，故 varlen 只支持一条
  拼接轴；多 batch 的多序列要走 batched 或逐条 varlen。
- 只测 **`D=128`、bf16 操作数、bf16 状态**；`StateFP32=true` 的 varlen 模板（`flash_kda.cpp:203-207`）
  已由 **§8（P4-5）** 单独锚定。
- 本节只验**数值**，不测 varlen 的**性能**（`tile_prefix` 二分 vs 整除的 host/device 成本）；性能专属
  证据见 `docs/07`。

## 8. varlen 前向的 `StateFP32` 模板：它只是一个 fp32 I/O 外壳（P4-5）

§7 只锚了 **bf16** 状态，并明确把 `StateFP32=true` 的 varlen 模板留作**待验证**。本节用
`harness/check_varlen_forward_fp32state.py`（原始输出 `harness/check_varlen_forward_fp32state.out.txt`）
把它钉死，并顺带回答一个容易想反的问题：**`StateFP32` 会不会带来更精确的状态？答案是不会。**

### 8.1 结构事实：状态递推始终在 bf16 里跑，fp32 只是 I/O 壳

读 K2 kernel 可以直接判定这一点，无须实验（结构描述亦见 `docs/03` §3/§19）：

- `state_acc` 是一块 **`BF16`** 的 smem 数组（`fwd_kernel2.cuh:82`：
  `cute::ArrayEngine<BF16, cosize_v<StateSmemLayout>> state_acc`），整段递推
  `s_acc ← s_acc·g_total + k_rest^T U`（`docs/03`）都在它上面做，**没有 fp32 状态副本**。
- `StateFP32=true` 只在两端改行为：
  - **载入**：TMA 把 fp32 `initial_state` 搬进 staging 缓冲 `state_fp32_buf`，然后
    `smem_cvt_fp32_to_bf16` 逐元素 `BF16(fp32_view(r,c))` 落进 `state_acc`
    （`fwd_kernel2.cuh:300-303`；转换在 `utils.cuh:378-407`）——即**立即 round 到 bf16**。
  - **落盘**：先 `smem_cvt_bf16_to_fp32` 把 bf16 `state_acc` 升回 fp32 staging
    （`fwd_kernel2.cuh:811-814`；`utils.cuh:409-437` 的 `bf16_to_f32` 是**精确**升位），再 TMA 写 fp32。

所以 `StateFP32` 与 `bf16` 两条模板的**内部算术逐字相同**，只有输入/输出的 dtype 不同；bf16→fp32 升位
无损、fp32→bf16 降位是 round-to-nearest。一句话：`StateFP32` 让调用方**不必手动 `h0.to(bf16)`**，
而状态精度仍受 bf16 支配。

### 8.2 Part A：壳等价——与 bf16 模板**逐位相同**（`max_abs=0`）

七组 `lens`（含 $L{=}1<$CHUNK、非整除 $T{=}257/199$、$g{=}0$、$g{=}4$）用**同一个 bf16 可表示的
`h0`** 分别跑 fp32-state 与 bf16-state 两个 varlen launch，再逐序列与 `B=1` 的 fp32-state batched 比：

| case | `o`(f32 壳 vs bf16 壳) | `S`(fp32 vs upcast bf16) | packed vs `B=1` |
|---|---|---|---|
| 全部七组 | **0.000e+00** | **0.000e+00** | `o`/`S` **0.000e+00** |

- 两壳的 `o` **逐位相同**、fp32 落盘恰是 bf16 状态终值的**精确升位**（差 0）——与 §8.1 的结构
  预测完全一致。
- fp32-state 的 varlen 依旧与「逐序列 `B=1`」**逐位相同**——即 P4-4 Part A 的「varlen 调度是
  batched 的无损重排」在 `StateFP32` 模板上同样成立，再次排除 tile→序列二分错/状态泄漏/尾块记账错。

### 8.3 Part B：与 fp64 golden 比——fp32 壳与 bf16 壳误差**逐值相同**（无精度增益）

用**真正随机的 fp32 `h0`** 跑两个壳，逐序列与 fp64 token-recurrent golden 比 `max_rel`：

| case | fp32-state `o` / `S^T` | bf16-state `o` / `S^T` |
|---|---|---|
| `lens=[30,16,5,50]` | 6.06e-03 / 1.02e-02 | 6.06e-03 / 1.02e-02 |
| `lens=[1,16,17,64,3]` | 7.98e-03 / 8.54e-03 | 7.98e-03 / 8.54e-03 |
| `lens=[64,64,64]` | 6.32e-03 / 9.34e-03 | 6.32e-03 / 9.34e-03 |
| `lens=[250,7]` | 5.59e-03 / 8.53e-03 | 5.59e-03 / 8.53e-03 |
| `lens=[33,47,19,100]` | 7.17e-03 / 1.18e-02 | 7.17e-03 / 1.18e-02 |
| `g=0` | 5.81e-03 / 5.61e-03 | 5.81e-03 / 5.61e-03 |
| `g=N(0,4)` | 6.38e-03 / 1.25e-02 | 6.38e-03 / 1.25e-02 |

两列的误差在打印精度内**完全相同**，且落在 §4/§7 的 bf16 底噪带内。这不是巧合：由 §8.1，两个壳的
内部轨迹逐位一致，误差自然一致——**`StateFP32` 不改变任何数值精度**。

### 8.4 Part C：真 fp32 `h0` 的余下尾数被内部 round 掉

取一组**真实 fp32**（非 bf16 可表示）的 `h0`：`|h0_fp32 − round_bf16(h0_fp32)| = 1.31e-02`。把这份
真 fp32 `h0` 喂 fp32-state 壳，与「先把 `h0` round 成 bf16、再喂 bf16 壳」比：

```
o(fp32-state, genuine h0)  vs  o(bf16 shell, bf16(h0)) = 0.000e+00
S(fp32-state, genuine h0)  vs  S(bf16 shell, bf16(h0)) = 0.000e+00
```

**逐位相同**。即 fp32 壳对真 fp32 `h0` 做的，恰好等价于调用方自己先 `.to(bf16)`——多出来的尾数在
载入时就被 `BF16(...)` 丢掉（§8.1 的 `fwd_kernel2.cuh:300-303`）。

### 8.5 Part D：fp32 状态组合扫描

`flash_kda.cpp:192-207` 对「无状态 / 仅入 / 仅出 / 双端」各派发一版 `StateFP32` 模板，四组都相对
「逐序列 fp32-state batched」**逐位相同**：

```
no state               o max_abs=0.000e+00  (no S out)
state-in only (fp32)   o max_abs=0.000e+00  (no S out)
state-out only (fp32)  o max_abs=0.000e+00  S max_abs=0.000e+00
both (fp32)            o max_abs=0.000e+00  S max_abs=0.000e+00
```

（`out-only`+fp32 这一档即使 `initial_state_ptr=nullptr` 也正常——`fwd_launch.cu:119-143` 为
`StateFP32` 无条件构造 fp32 状态 TMA 图，但 `HasStateIn=false` 时它从不被发射。）

### 8.6 结论与设计输入

- §7.6 的「`StateFP32` 的 varlen 模板未单独测」**关闭**：七个 `lens` × 四种状态组合，fp32-state
  packed 与逐序列 batched **逐位相同**、与 fp64 golden 落 bf16 底噪。
- **设计输入**：`StateFP32` **不是精度旋钮**，只是把状态接口从 bf16 换成 fp32——内部递推恒为 bf16。
  想把状态精度提到 fp32，须改 kernel（`state_acc` 换 fp32 累加），传 fp32 `initial_state`/`final_state`
  是不够的。反过来，如果上游已持有 fp32 状态、又不介意内部 bf16，用 `StateFP32` 可省掉调用方的一次
  `.to(bf16)` cast。
- **边界（诚实）**：只测 `D=128`、bf16 操作数、varlen（`B==1`）；batched 的 `StateFP32` 模板
  （`IsVarlen=false`）同源、未单独列（其算术与 varlen 仅差 tile→序列定位，P4-4/本节已分别覆盖两端）。
  本节只验数值，不测 fp32 状态的性能/带宽。

## 9. 第三个参照：FlashKDA 自带的 **kernel-逐位** torch 参考（P4-6）

前面所有前向数值锚定都只用两方：**fp64 逐 token golden**（`src/kda_ref.py`）与 **FLA Triton**
（`chunk_kda`）。两者都是 *独立于 FlashKDA kernel* 的实现；而 FlashKDA 仓库里还自带一个
**故意逐位复刻 kernel** 的纯 torch 参考 `tests/torch_ref.py`——它严格镜像 kernel 的每一步低层选择：

- L2 归一化的 warp-shuffle 树 + FMA（`l2_normalize_kernel_match`，`torch_ref.py:62–74`）；
- 门控 sigmoid 用 `tanh.approx.f32`（`sigmoid_ext`，`torch_ref.py:7–34`）；
- 衰减 `exp2` 带 FTZ（`fp32_ex2_ftz`，`torch_ref.py:51–57`）；
- `(I+L)^{-1}` 用 8×8 fp32 前代 + 16×16 bf16-HMMA 合并（`inv_fwd_subst_16`，`torch_ref.py:81–122`）。

FlashKDA 自己的 `tests/test_fwd.py:260–261` 断言 `torch.equal(out_kernel, out_ref)`——即这个参考是
**精确**的、不是 bf16 级的近似。本节首次把它当成专题的**第三个 oracle** 用起来
（`harness/check_fwd_torchref.py`，原始输出 `harness/check_fwd_torchref.out.txt`），回答三件事。

### 9.1 Part A：kernel 与它自带的参考**逐位相同**（本轮独立复现该事实）

五组 case（含 $B{=}2,T{=}130$ 非整除 16、$g{=}0$、$g{=}N(0,4)$）：`o` 与 $S^\top$ 都 `torch.equal==True`、
`max_abs=0.000e+00`。这把我们后续结论的分母钉死——**该参考读到的就是 kernel 的代数**，任何与它的差异
都可归给「bf16 量化 / 累加序」，而不是「参考近似不准」。

### 9.2 Part B：我们的分块代数 vs 该 kernel-忠实参考——落 bf16 底噪

把 `src/kda_ref.py::kda_chunked_fwd`（P1-2 的分块公式）喂上与 kernel-忠实参考**完全相同**的
激活后操作数（同一份 `l2_normalize_kernel_match` 的 bf16 `q,k`、同一份 `tanh.approx` 的 `g/β`），
再比 `o`/$S^\top$（`max_rel` 分母取参照幅值）：

| case | `o` max_rel（我方 fp32） | $S^\top$ max_rel（我方 fp32） | `o` max_rel（我方 fp64） | $S^\top$ max_rel（我方 fp64） |
|---|---|---|---|---|
| B1T256H4 | 1.033e-02 | 7.799e-03 | 1.070e-02 | 7.973e-03 |
| B2T130H2（非整除） | 7.890e-03 | 8.352e-03 | 8.694e-03 | 8.254e-03 |
| B1T64H1 | 4.570e-03 | 1.076e-02 | 4.266e-03 | 1.059e-02 |
| g=0 | 7.413e-03 | 7.316e-03 | 7.380e-03 | 7.277e-03 |
| g=N(0,4) | 5.725e-03 | 5.578e-03 | 5.977e-03 | 5.680e-03 |

两点关键：

- 全部落在 §4/§5/§6 的 **bf16 底噪带**（$4\times10^{-3}\sim1.1\times10^{-2}$）内——一个**完全独立**于
  token-recurrent golden 的分块实现，与 kernel 的代数只差 bf16 量级。
- **我方 fp32 与 fp64 两列几乎重合**，说明残差**不来自我们的累加**，而来自参考自己的逐操作数
  bf16 量化（`torch_ref.py` 把 `kd/qd/ki/kr` 及 `INV` 都降到 bf16 再算）。这正是 §5 归因「误差由
  bf16 数据表示决定、非算法缺陷」的又一次独立印证。

### 9.3 Part C：alg0 的精确等式——我方 fp64 分块 == 参考 chunk 循环的 fp64 转录（逐位）

为把「公式本身对不对」与「bf16 量化」彻底分开，把 `torch_ref.py:200–244` 的分块循环**在 fp64 里
重抄一遍**（去掉所有 bf16 cast、用精确 `torch.sigmoid`），并与 `kda_chunked_fwd` 跑在**同一份 fp64
操作数**上：五组 case 的 `o`/$S^\top$ 全部 **`max_abs=0.000e+00`（逐位相同）**。两个独立转写的分块
代数在机器精度上重合，确认 P1-2 的 `L/Mqk/INV/U/S_new` 五条式子无一抄错。

同时给出「精确解 vs bf16 参考」的总误差（=bf16 量化 + 激活近似的合计）：`o` max_rel 为
1.10e-02 / 8.74e-03 / 4.39e-03 / 6.91e-03 / 6.29e-03，与 Part B 同带——再次说明端到端误差由 bf16
输入/状态/操作数量化主导，算法顺序只占零头。

### 9.4 结论

- FlashKDA 自带参考是**逐位的**（Part A 独立复现），可作为「kernel 代数」的严格 oracle。
- 我们的分块公式（P1-2）与该 oracle 在**同操作数**下只差 bf16 底噪（Part B），而它自身的 fp64 转写
  与对方 chunk 循环**逐位相同**（Part C）——分块公式获得迄今最强的一次独立确认。
- 这也给 `docs/03` §18.5 的 caveat（「纯 PyTorch 消融，未逐位复现 kernel 累加序」）补上了对照物：
  需要「kernel 累加序」时可直接用这份 bit-exact 参考，而不必再自建。

## 10. varlen × kernel-逐位 oracle 的交叉锚定（P4-7）

§7（P4-4）把 packed 前向锚到的是**另一个 kernel 运行**（逐序列 batched）与 fp64 golden；§9（P4-6）
引入了 FlashKDA 自带的 **kernel-逐位** torch 参考 `tests/torch_ref.py`，但只用它跑过 **batched**
（等长 $B$）。而 `torch_ref` 自己有一条**专用 `cu_seqlens` 分支**（`torch_ref.py:141-142` 断言
`B==1`、`torch_ref.py:178` 取 `N=len(cu_seqlens)-1`、`:185-193` 逐序列 `[bos,eos)` 循环 + 逐序列
`work_state[seq_idx]`、`:202-206` 按 `actual_len` 补零尾块、`:241` 只写有效 token）——这条分支
**此前从未被本专题用到过**，packed kernel 也**从未与一个 bit-exact oracle 直接比过**。

本节用 `harness/check_varlen_torchref.py`（原始输出 `harness/check_varlen_torchref.out.txt`）把两个
最近线索（varlen 与第三 oracle）在**同一点**合起来，给出一个「逐位三角」。

### 10.1 Part A：packed kernel vs `torch_ref(cu_seqlens)` —— **逐位相同**

同一份 packed 操作数 `[1,T_total,H,D]`、同一份**逐序列不同的** `initial_state [N,H,D,D]`，一侧跑
`flash_kda_C.fwd(..., cu_seqlens=...)`、另一侧跑 `torch_ref(..., cu_seqlens=...)`，比 `o` 与
`final_state`（$S^\top$）：

| case | `T_total` | `N` | `o` `torch.equal` | `S^T` `torch.equal` | `o` max_abs | `S` max_abs |
|---|---|---|---|---|---|---|
| `lens=[30,16,5,50]` | 101 | 4 | **True** | **True** | 0.000e+00 | 0.000e+00 |
| `lens=[1,16,17,64,3]`（$L{=}1<$CHUNK） | 101 | 5 | **True** | **True** | 0.000e+00 | 0.000e+00 |
| `lens=[64,64,64]`（均 16 倍数） | 192 | 3 | **True** | **True** | 0.000e+00 | 0.000e+00 |
| `lens=[250,7]`（$T$ 非 16 倍数） | 257 | 2 | **True** | **True** | 0.000e+00 | 0.000e+00 |
| `lens=[33,47,19,100]` | 199 | 4 | **True** | **True** | 0.000e+00 | 0.000e+00 |
| `g=0`（无衰减） | 101 | 4 | **True** | **True** | 0.000e+00 | 0.000e+00 |
| `g=N(0,4)`（强衰减） | 101 | 4 | **True** | **True** | 0.000e+00 | 0.000e+00 |

七组（含单 token 序列、非整除 $T$、极端门控）全部**逐位相同**。注意这**比 §7.2 更强**：§7.2 比的
是「packed kernel vs 逐序列 kernel」，两方共享同一份 kernel 代码；本节比的是「packed kernel vs
一个**独立转录**的 kernel 代数」。它一次性同时验证了：

- packed 路径的 tile→(序列,块) 映射在**跨序列边界**处正确（否则 `o` 会在某序列内 O(1) 偏移）；
- **b=1 的 cu_seqlens 分支**（`torch_ref.py:185-193`）与 kernel 的 varlen 分派**代数一致**，
  且**逐序列 `initial_state` 索引**（distinct-$h_0$）在两方都按 `seq_idx` 生效；
- 参考的尾块补零（`:196-206`）与 kernel 的 pad-tile mask 对非整除 $L$ 等价。

### 10.2 Part B：`torch_ref` packed vs `torch_ref` 逐序列 —— **逐位相同**

为把「packing 是否引入序列间状态泄漏」从 kernel 身上剥离开、单独钉在 oracle 自己身上，再拿同一份
序列分别跑 `torch_ref(cu_seqlens)` 与「逐序列 `B=1`、无 `cu_seqlens`」两条路径：七组的 `o` 切片与
`final_state` 行**全部 `max_abs=0`**（`S^T` 亦然）。即 oracle 的 `work_state[seq_idx]` 累加在 packed
循环里**没有跨序列串扰**——这是 §7.2 对 kernel 所做隔离测试的 oracle 侧对偶。

### 10.3 结论

- Part A + Part B 给出一个**逐位三角**：`packed-kernel == ref-packed == ref-per-seq`。packed 路径
  现在同时被（i）另一个 kernel 运行（§7）、（ii）一个独立转录的 kernel 代数（本节）两方逐位钉死。
- 这也**首次把 `torch_ref` 的 `cu_seqlens` 分支纳入专题**（§9 只用了等长 batched）——若将来要把
  kernel-逐位 oracle 用于 varlen 相关的反向/重算锚定，此分支已被证实可用且与 kernel 代数一致。
- 不改变任何部署结论。**边界**：`B==1`（与 kernel varlen 同硬约束）；只 `D=128`/bf16 操作数与状态；
  未跑 fp32 状态壳（§8 已单独覆盖）。

## 11. 小结与下一步

- FlashKDA 前向 kernel 在全部测试 shape / seed / 门控区间下，与 fp64 golden 的 `o` 峰值相对误差
  为 $4\times10^{-3}\sim7.6\times10^{-3}$、终态为 $5\times10^{-3}\sim1.4\times10^{-2}$，符合 bf16 精度预期。
- **varlen/packed 路径（P4-4）**：七组（含 $L{=}1<$CHUNK、非整除 $T$、$g{=}0$/$g{=}4$）× 四种状态
  组合，packed 与「逐序列 batched」**逐位相同（`max_abs=0`）**、与 fp64 golden 同处 bf16 底噪，
  且每序列 `initial_state` 按序列索引生效——varlen 调度是 batched 的无损重排。
- 三方对比（golden / FlashKDA / FLA）显示三者同处 bf16 底噪；`flash↔fla` 的 `o` 误差因共模抵消
  更小（$3.65\times10^{-3}$），佐证误差主项是 bf16 输入舍入。
- 钉死了一个易踩的布局约定：FlashKDA 状态是 $S^\top$（$[N,H,V,K]$）。
- 脚本/证据：`harness/check_fwd_numerics.py`、`harness/check_fwd_numerics.out.txt`；
  三方对比 `harness/check_three_way.py`、`harness/check_three_way.out.txt`；
  受控消融 `harness/check_state_bf16.py`（P3-3）；varlen 锚定 `harness/check_varlen_forward.py`、
  `harness/check_varlen_forward.out.txt`（P4-4）；`StateFP32` 壳 `harness/check_varlen_forward_fp32state.py`（P4-5）；
  第三 oracle（FlashKDA 自带 kernel-逐位 torch 参考）`harness/check_fwd_torchref.py`、
  `harness/check_fwd_torchref.out.txt`（P4-6）；varlen × 该 oracle 的逐位三角
  `harness/check_varlen_torchref.py`、`harness/check_varlen_torchref.out.txt`（P4-7）。

**下一步（P5）**：进入反向。用 FLA `chunk_kda` 的 `torch.autograd` 拿参考梯度，先在
`src/kda_bwd_ref.py` 里实现逐 token 反向（adjoint）并与 autograd 对拍（P5-1/P5-2），再做分块化推导。
