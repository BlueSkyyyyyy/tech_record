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

## 6. 小结与下一步

- FlashKDA 前向 kernel 在全部测试 shape / seed / 门控区间下，与 fp64 golden 的 `o` 峰值相对误差
  为 $4\times10^{-3}\sim7.6\times10^{-3}$、终态为 $5\times10^{-3}\sim1.4\times10^{-2}$，符合 bf16 精度预期。
- 钉死了一个易踩的布局约定：FlashKDA 状态是 $S^\top$（$[N,H,V,K]$）。
- 脚本/证据：`harness/check_fwd_numerics.py`、`harness/check_fwd_numerics.out.txt`；
  受控消融 `harness/check_state_bf16.py`（P3-3）。

**下一步（P4-3）**：在 `harness/` 里加入第三方参照 FLA `chunk_kda`，与本文的 golden、FlashKDA
做**三方对比**并给误差表（`harness/check_kda_ref.py` 已有 FLA↔golden 的两方对拍可复用）。
