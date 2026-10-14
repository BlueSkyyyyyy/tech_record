# 给通信让 SM 的「机会成本」：DeepEP 自动 SM 模型超配的那 48 个 SM 值多少计算

> 本文对应路线图 **P183-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 193 篇。
>
> 前面三篇把「给通信让 SM」这条设计轴量成了一张**双向汇率表**：
>
> - **第 190 篇（P180-1）** 量「计算侧」——一条 grouped GEMM 自己的 SM 预算曲线：给通信让出一个 SM，
>   计算腿多花多少 μs；
> - **第 191 篇（P181-1）** 量「通信侧（单链路）」——一条 NVLink P2P 在 ~8 个 SM 就饱和于 ~372 GB/s；
> - **第 192 篇（P182-1）** 量「通信侧（多对多）」——一个 rank 的 NVLink **总出口**同样在 **8–16 SM**
>   饱和于 ~370–376，**与 peer 数无关**；并发现 DeepEP 的真实模型
>   `get_theoretical_num_sms()`（`deep_ep/buffers/ep.py:431-541`）在 `prefer_overlap_with_compute=False`
>   时，会把单节点 `direct EP8` 顶到 `max(·, 64)`（`:531`）——**是膝点的 4×**。
>
> 但三篇都只把账**分开**量了，从没把两侧合起来回答一个最实际的问题：
>
> **模型多给的那 48 个 SM（64 vs 16）到底值多少计算时间？**
>
> 或者说：正确性上没有争议的「给通信留几个 SM」这个旋钮，**错拧到 64 会付出多大代价**。
>
> 本篇的答案（H100 真机，`src` 无，纯 Python + 复用第 190 篇的真实 SM90 FP8 grouped kernel）：
>
> - **模型给的 SM 数线性正比于「每 rank NVLink 总带宽」**——因为它四个候选里**起决定作用的那条**
>   是 combine 的 NVLink 写出腿 `bounded_gbs/bounded_traffic × (sm_read + nvlink_traffic)/sm_write_gbs`
>   （`ep.py:528`），分子分母都在 `nvlink_gbs` 上。把 `nvlink_gbs` 从模型默认 **430.3** 换成真机实测
>   **372（单 P2P）/ 325.7（满负荷 all-to-all）**，`direct EP8` 从 16 → 16 → **12**。
> - **生产路径（`prefer_overlap=True`）的 16（或 12）恰好落在真机膝点内**——模型这条路径**不离谱**；
> - **真正越界的是 `max(·,64)`**：那 48–52 个 SM 换不回任何通信带宽（第 192 篇已证膝点外白给），
>   却直接从计算腿扣走。用第 190 篇的真机曲线定价：**前向 L1 多花 ~244 μs（6.8× 膝点损失）**、
>   **反向 wgrad 多花 ~71 μs**。
> - 结论：**`max(·,64)` 是一个「零收益、纯损失」的旋钮**；正确的 SM 预算就是**链路膝点**
>   （约 8–16），这也从另一个角度印证了 Mega MoE 主 kernel 把通信与计算放进**同一块 SM 的不同
>   warp**、而不做 SM 分区的选择（`heuristics/mega_moe.hpp:66-70` 钉死全设备 SM）。
>
> **可跑性 / 边界**：DeepEP 的 **dispatch/combine 设备端**依赖 NCCL GIN 设备 API（本机 NCCL 2.29.7
> < 需求 ≥2.31，见 ROADMAP「阻塞」）→ 端到端**仅分析**。本篇量的是它**脚下的 host 模型**与
> **计算腿的真实 SM 曲线**，都不碰设备 API：Part A 是真实 SM90 FP8 grouped kernel（复用第 190 篇的
> `exp_sm_budget_curve.py`），Part B 用 P63-1 的 `sys.modules` stub + `ast` 抽取跑**未修改的真实**
> `get_theoretical_num_sms` 方法体。**H100（sm90）真机**，8×H100 80GB HBM3；本机是**共享节点**，
> 逐迭代取最小值以排除其他任务的争用（详见 §4 注）。
> 目标卡代号敏感：全文用「目标卡」指代。
>
> 参考坐标：`~/github/DeepEP@93eb6eb`（V2.5）`deep_ep/buffers/ep.py`、`deep_ep/utils/envs.py`；
> `~/github/DeepGEMM`（`057ca59`）`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`、
> `csrc/runtime/runtime.hpp`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`、
> `csrc/jit_kernels/heuristics/mega_moe.hpp`。
> 原始输出 `harness/exp_comm_sm_opportunity_cost.out.txt`
> （`harness/exp_comm_sm_opportunity_cost.{sh,py}`；H100 80GB HBM3, sm90）。

---

## 1. 账本：机会成本 = 通信收益（0）− 计算损失（>0）

先把这个「账」的结构讲清楚，它决定了本篇要量什么。

设 MoE 一个 step 里：
- **通信腿**要搬固定字节量 `V`（dispatch/combine 的 NVLink 流量），用时 `T_comm(n) = V / BW(n)`，
  其中 `BW(n)` 是「n 个 SM 参与时一个 rank 的 NVLink 出口带宽」；
- **计算腿**（routed grouped GEMM）用时 `T_comp(132 - n)`——通信占用 n 个 SM，计算只剩 `132 - n`；
- Mega MoE 把两者**重叠**（同 kernel 内 comm warp + MMA warp），故 step 时间 ≈ `max(T_comm, T_comp)`。

于是「改 n」的**净收益**是：
```
Δ = [T_comm(n0) - T_comm(n1)] - [T_comp(132-n1) - T_comp(132-n0)]
     \_____ 通信收益 _____/   \________ 计算损失 ________/
```
第 192 篇已经证明 **`BW(n)` 在 ~8–16 个 SM 后饱和**（一个 rank 的 NVLink 出口是共享资源），于是
对 `n > 16`：**通信收益恒为 0**（第一项 = 0），而计算损失（第二项）随 n **单调增加**。所以：

> **一旦 n 超过链路膝点，每多给一个 SM，净收益恒为负**——它既不加速通信，又纯扣计算。

这就把问题简化成：**模型给到 64 时，n > 膝点的那 48 个 SM 值多少计算时间？** 这正是 Part A 要量的。

---

## 2. 源码事实：模型的 SM 数为什么「∝ 每 rank NVLink 总带宽」

`get_theoretical_num_sms()`（`deep_ep/buffers/ep.py:431-541`）是一段解析式 roofline。它先按
「每 token 的单位流量」累加四条腿（`sm_read / sm_write / rdma_traffic / nvlink_traffic`），挑出
**瓶颈链路** `(bounded_traffic, bounded_gbs)`，再取四个候选最大值（`:522-529`）：

```
num_sms = max(
    bounded_gbs/bounded_traffic * sm_read                         / sm_read_gbs,   # :524 dispatch 读
    bounded_gbs/bounded_traffic * sm_write                        / sm_write_gbs,  # :525 dispatch 写
    bounded_gbs/bounded_traffic * (sm_write - nvlink_traffic)     / sm_read_gbs,   # :527 combine 本地读
    bounded_gbs/bounded_traffic * (sm_read  + nvlink_traffic)     / sm_write_gbs,  # :528 combine 写回
)
num_sms = align(max(4, ceil(num_sms * 1.3)), 4)                                   # :530
num_sms = num_sms if prefer_overlap_with_compute else max(num_sms, 64)            # :531
num_sms = min(num_sms, num_device_sms // 2 * 2)                                   # :532
```

对单节点 `direct EP8`（`num_scaleout_ranks==1`、`num_rdma_ranks==1`、`num_nvlink_ranks==8`、
`num_ranks==8`）走 `else` 支（`:493-503`），代入 `E=256, k=8` 可算得瓶颈是 **NVLink**，
且**起决定作用的是候选 #4（`:528`，combine 把结果经 NVLink 写回远端 combine 缓冲）**：

| 候选 | 值（默认常量） |
|---|---:|
| #1 dispatch 读 | 0.52 |
| #2 dispatch 写 | 10.93 |
| #3 combine 本地读 | 0.34 |
| **#4 combine 写回** | **11.64 ← max** |

`11.64 × 1.3 = 15.13 → ceil 16 → align4 = 16`，正是模型对 `direct EP8` 的输出。

**关键观察**：候选 #4 里 `bounded_gbs/bounded_traffic`（NVLink 场景 = `nvlink_gbs/nvlink_traffic`）
与 `nvlink_traffic` 相消后，`num_sms` 里剩下的 `nvlink_gbs` 在**分子**：

```
num_sms ≈ nvlink_gbs / sm_write_gbs × (sm_read + nvlink_traffic)/nvlink_traffic × 1.3
```

即 **模型的 SM 数线性正比于「每 rank NVLink 总带宽」`nvlink_gbs`**（分母是写带宽常量 45）。
这就是为什么第 191/192 篇把 `nvlink_gbs` 换成真机实测值时，SM 数会按比例移动——Part B 要复现这一步。

> 模型里两个每 SM 带宽常量 `get_sm_read_gbs()=180` / `get_sm_write_gbs()=45`
> （`deep_ep/utils/envs.py:194-197/201-204`，带 `TODO: use architecture-specific values`）是**写死**的；
> `get_nvlink_gbs()`（`envs.py:208-234`）则是 `nvidia-smi nvlink -s` 各链路求和 ×0.9
> （`:231`），在目标卡 H100 上 = **430.3 GB/s**。

---

## 3. Part B：把 `nvlink_gbs` 换成真机实测，看模型给几个 SM

用 P63-1 的 `ast` 抽取跑**未修改的真实** `get_theoretical_num_sms` 方法体（`ep.py` sha256
`a8f72c39…`、`envs.py` sha256 `23843e16…`），把 `nvlink_gbs` 依次设为
**430.3（模型默认）/ 372.0（第 191 篇单 P2P 实测）/ 325.7（第 192 篇满负荷 all-to-all 实测）**：

| 配置 | nv=430.3 | nv=372.0 | nv=325.7 |
|---|---:|---:|---:|
| `direct EP8`（单节点，无 RDMA） | **16** | 16 | **12** |
| `hybrid EP16`（2 节点 ×LSA8） | 16 | 16 | 16 |
| `hybrid EP32`（4 节点 ×LSA8） | 8 | 8 | 8 |
| `hybrid EP64`（8 节点 ×LSA8） | 8 | 8 | 8 |
| `hybrid EP128`（16 节点 ×LSA8） | 4 | 4 | 4 |
| `hybrid EP16`，RDMA 慢（瓶颈 RDMA） | 4 | 4 | 4 |
| `direct EP4`（1 节点 4 rank） | 20 | 16 | 16 |
| `direct EP1`（无流量） | 132 | 132 | 132 |

而 **`prefer_overlap=False` 分支**（`direct EP8, not preferring overlap`）在 `:531` 的 `max(·,64)` 下
恒为 **64**。

读法：

1. **`direct EP8` 从 16 → 16 → 12**：`nv=325.7` 时（满负荷实测）模型给 12——**恰好落在真机膝点
   （8–16）内**。模型这条路径（默认常量 16）**并不离谱**：第 192 篇就修正过第 191 篇略带歧义的
   说法——16 不是「多开 2×」，而是膝点上沿。
2. **只有 NVLink-bounded 的配置随 `nvlink_gbs` 动**（`direct EP8` 16→12、`direct EP4` 20→16）；
   RDMA-bounded（`hybrid EP16 RDMA 慢` = 4）与无流量（`EP1` = 132）不动。这正印证 §2 的
   「`num_sms ∝ nvlink_gbs` 只在 NVLink 是瓶颈时成立」。
3. **真正的越界只有 `max(·,64)`**：48–52 个 SM 对通信零收益（膝点外白给），却从计算腿扣走。

---

## 4. Part A：计算腿的真实 SM 曲线（132 − reserve）

用第 190 篇已验证的真实 SM90 FP8 grouped kernel（`deep_gemm.set_num_sms` 把 `num_sms` 编进
`kNumSMs` 模板参数，`scheduler/gemm.cuh:20/34/38`；`sm90_fp8_gemm_1d2d.hpp:125` 的
`.grid_dim = num_sms` 即 persistent grid 宽度），量两条 MoE 形状在 `reserve ∈ {0,8,12,16,22,48,64}`
（即 `num_sms = 132 − reserve`）下的纯 device 时长：

**Part A-1 前向 L1**（`m_grouped`，`A=[8192,H=7168]`、`W1=[2I=4096,H=7168]`，481 GFLOP，**计算受限**）

| reserve | num_sms | t (μs) | TFLOPS | t/t132 |
|---:|---:|---:|---:|---:|
| 0 | 132 | **360.3** | 1335 | 1.000 |
| 8 | 124 | 378.9 | 1269 | 1.052 |
| 12 | 120 | 378.2 | 1272 | 1.050 |
| **16** | **116** | **410.5** | 1172 | 1.139 |
| 22 | 110 | 406.5 | 1183 | 1.128 |
| 48 | 84 | 459.6 | 1047 | 1.275 |
| **64** | **68** | **654.4** | 735 | **1.816** |

**Part A-2 反向 wgrad**（`k_grouped`，`D[e]=A_e^T B_e`、`A=[7168,k_g=1024]`、`B=[2048,k_g=1024]`，
240 GFLOP，**访存受限**）

| reserve | num_sms | t (μs) | TFLOPS | t/t132 |
|---:|---:|---:|---:|---:|
| 0 | 132 | **523.8** | 459 | 1.000 |
| 8 | 124 | 553.3 | 435 | 1.056 |
| 12 | 120 | 547.3 | 439 | 1.045 |
| **16** | **116** | **530.8** | 453 | 1.013 |
| 22 | 110 | 537.6 | 447 | 1.026 |
| 48 | 84 | 547.1 | 440 | 1.044 |
| **64** | **68** | **601.4** | 400 | 1.148 |

> **共享节点注**：本机 8 卡被并行自驱任务共享，单点曾量到 2× 的争用假象（如 L1 `num_sms=132`
> 一度 777.6 μs）。实验改为**逐迭代用 CUDA event 计时取最小值**（`best_time`），复现稳定在
> ±3% 内；下表数字均为稳定复现值。这与第 190 篇的曲线同源（132 点 360.3 vs 其 357.7、
> 84 点 459.6 vs 其 491.9，量级一致，差异来自共享争用与 `best_time` 口径）。

两条形状的形态**完全复现第 190 篇的结论**：

- **L1（计算受限）**：`132→120`（试给 12 SM）几乎无损（+5%），但 `→116`（给 16）已经 +13.9%；
  到 `→68`（给 64）**+81.6%**——计算受限的腿上，扣 SM 的代价**陡**。
- **wgrad（访存受限）**：`132→116` 只 +1.3%（甚至在噪声内，116 的 530.8 比 124/120 都快）；
  到 `→68` 也仅 +14.8%——访存受限的腿上，SM **几乎白送**（HBM 在 ~84 SM 就饱和）。

---

## 5. Part C：机会成本定价

把 Part A 的曲线代进 §1 的账本（`n_knee=16`，`n_model=64`）：

| 形状 | t(132) | t(132−16) | t(132−64) | **膝点损失**（16）| **超配损失**（64）| **浪费 = 超配−膝点** |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 360.3 | 410.5 | 654.4 | +50.1 μs | +294.0 μs | **+243.9 μs（6.8×）** |
| wgrad | 523.8 | 530.8 | 601.4 | +7.0 μs | +77.6 μs | **+70.6 μs（11.1×）** |

注意这里的「浪费」= **超出膝点那 48 个 SM 的净负收益**（因为超过膝点后通信收益 = 0，见 §1）。
再对「真机膝点到底取多少」做敏感性（`n_knee ∈ {8, 16, 22, 64}`）：

| n_knee | L1 浪费（到 64） | wgrad 浪费（到 64） |
|---:|---:|---:|
| 8 | +275.4 μs | +48.1 μs |
| **16** | **+243.9 μs** | **+70.6 μs** |
| 22 | +247.8 μs | +63.9 μs |
| 64 | +0 μs | +0 μs |

- **只要膝点 < 64，超配到 64 就恒为纯损失**，量级 **L1 ~250 μs / wgrad ~50–70 μs**（单个 step、
> 单条腿）。对照第 190 篇的整 step 预算（L1 357.7 + L2 191.5 ≈ 549 μs 计算侧），这 250 μs 是
> **L1 单腿的 68%**——不是零头。
- **wgrad 的浪费带噪声**（8→48、16→71、22→64 μs 非单调）：因为它是**访存受限**的，
  `116` 点甚至比 `124/120` 快（HBM 饱和后的抖动）。但**方向一致为负**，且这正是第 190 篇
  「访存受限腿的 SM 白送」的另一面——白送的 SM 被通信拿走后，反正也拿不回来（对 wgrad 影响小，
  对 L1 影响大）。
- 若用 §3 满负荷实测的 **`n_knee=12`**（`nv=325.7`），超配浪费还要再大几个百分点
  （`132−12=120` 的 L1 是 378.2，比 116 的 410.5 还快——因为 `116` 点本身在 L1 曲线上有个小台阶）。

---

## 6. 与第 190/191/192 篇拼起来：完整的「SM 汇率」与一个零收益旋钮

把四篇拼成一句话的操作结论：

| 侧 | 量 | 结论 | 出处 |
|---|---|---|---|
| 计算（L1，计算受限） | 让 1 SM 的代价 | 平台区 ~1.4–3 μs，拐点区陡增 | 第 190 篇 |
| 计算（wgrad，访存受限） | 让 1 SM 的代价 | 平台区 ≈0 μs（白送） | 第 190 篇 |
| 通信（单链路 P2P） | 打满一条 NVLink 要几个 SM | ~8 个 | 第 191 篇 |
| 通信（多对多 a2a） | 一个 rank 总出口要几个 SM | **8–16 个，与 peer 数无关** | 第 192 篇 |
| **机会成本（本篇）** | **超配到 64 亏多少** | **L1 ~244 μs / wgrad ~71 μs（纯损失）** | 第 193 篇 |

**核心结论**：DeepEP 模型的**默认路径**（`prefer_overlap=True`）给 `direct EP8` 的 **16**
（换真机满负荷带宽后 **12**）**落在真机膝点内、是对的**；模型唯一的错是
`prefer_overlap=False` 时的 **`max(·,64)`**（`ep.py:531`）——它是一个 **「零收益、纯损失」的旋钮**：
多给的 48–52 个 SM 既买不到通信带宽（膝点外白给），又直接从计算腿扣走 **L1 ~244 μs**。
模型的封顶是 `device_sms//2`（`:532`），**完全不建模链路膝点**，这是它可能顶到 64 的根因。

这也从另一面解释了 **Mega MoE 主 kernel 为什么不做 SM 分区**：既然通信腿 ~8–16 个 SM 就封顶、
超过就是纯亏，把通信和计算放进**同一块 SM 的不同 warp**（第 2/9/10 篇的 warp specialization +
`setmaxnreg`）比「切一整块 SM 给通信、超过膝点就白给」更省。DeepGEMM 的 host 侧正是这么钉的——
`heuristics/mega_moe.hpp:66-70` 的 `get_num_sms_for_mega_moe()` 直接断言
`num_sms == jit->device.get_num_sms()`（**必须用整个 GPU**，因为要按 locality domain 分布任务）。

---

## 7. 小结与边界

- **模型的 SM 数 ∝ 每 rank NVLink 总带宽**：单节点 `direct EP8` 的起决定候选是 combine 写回腿
  （`ep.py:528`），分子分母都在 `nvlink_gbs` 上 → 换真机实测 430.3→372→325.7，SM 从 16→16→**12**
  （落在真机膝点 8–16 内）。
- **默认路径不离谱，越界的是 `max(·,64)`**（`:531`）：48–52 个 SM 对通信零收益。
- **机会成本（H100 真机，Part A 曲线定价）**：超配 64 相比膝点 16，前向 L1 **+244 μs**、
  反向 wgrad **+71 μs**（单个 step、单条腿的纯损失）；L1 这 244 μs 是其计算侧预算（~550 μs）的 44%。
- **访存受限腿对扣 SM 更钝**：wgrad 只 +71 μs（且非单调），L1 +244 μs——与第 190 篇「访存受限腿
  的 SM 白送」互为表里。
- **边界**：DeepEP dispatch/combine **设备端**（NCCL GIN）本机跑不了（`docs/86` 两道闸门、
  ROADMAP「阻塞」）——本篇量的是它**脚下的 host 模型**与**计算腿的真实 SM 曲线**，不碰设备 API；
  8 卡共享节点上逐迭代取最小值排除争用；计算腿用真实 SM90 FP8 grouped kernel 作**形状代理**
  （目标卡上是 UMMA，第 35 篇「仅分析」）。

---

## 附：复现

```
bash code/mega-moe/harness/exp_comm_sm_opportunity_cost.sh \
  > code/mega-moe/harness/exp_comm_sm_opportunity_cost.out.txt
```
- `harness/exp_comm_sm_opportunity_cost.py`：
  - **Part A** 真实 SM90 FP8 grouped kernel（复用 P180-1 的 `exp_sm_budget_curve.py` 的
    `build_m_grouped`/`build_k_grouped`/`run`），逐迭代 CUDA event 取最小值；
  - **Part B** 复用 P63-1 的 `exp_deepep_sm_qp.py`（`ast` 抽取真实 `get_theoretical_num_sms`），
    扫 `nvlink_gbs ∈ {430.3, 372.0, 325.7}`；
  - **Part C** 用 Part A 曲线给超配定价 + 膝点敏感性。判定 `PASS=7 / FAIL=0`。
