# MoE 反向的通信量模型与 H100 实测：反向为什么没有 mega-kernel

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P94 增量（第 104 篇）。
> 第 1 篇 §2 给了**前向** EP 通信量模型（dispatch ~$T\!\cdot\!d\!\cdot\!H\!\cdot\!1$B、
> combine ~$T\!\cdot\!k\!\cdot\!H\!\cdot\!2$B）；第 4/12/13/14 篇拆了反向的**计算侧**
> （四项 grouped GEMM），第 95 篇拆了反向的**语义侧**（dispatch/combine 互为转置的伴随恒等式）；
> 第 7/48/51/58/61/63/71 篇拆了通信的**设备 kernel**。但**反向的通信量到底是多少、和前向比谁大、
> 要不要重叠**——从来没有单独的模型与实测。本篇补上这最后一块。
>
> **结论先行**：反向的通信不是「更小」，而是**约 1.5× 前向**（字节量比 1.49×，H100 实测时间比
> 1.46–1.54×）。原因是**布局不对称**：前向 dispatch 在传输层按目标 rank 去重（dedup，$T\!\cdot\!d$ 行、FP8 1B），
> 而 combine/两个反向相位都在 **expanded** 空间搬（$T\!\cdot\!k$ 行、BF16 2B）。反向恰好是
> **两个 expanded 的伴随原语**，于是 BF16 的 $T\!\cdot\!k$ 行走两趟。
> 反向之所以**没有**融合成一个 mega-kernel，不是因为通信小，而是因为：① 它天然分解为
> 两份通信原语 + 四个相互独立的 grouped GEMM；② 官方直接用 `async_with_compute_stream=True`
> 在**通信库层**做重叠（`README.md:208/217/245/252`），而不是像前向那样在**kernel 内**做 warp 级重叠。
>
> **可跑边界**：端到端 `EPBuffer.dispatch/combine` 依赖 NCCL ≥2.32.3 设备 API（本机 2.29.7，见「阻塞」）
> → 仅分析；本篇用 **8×H100 真实 ragged all-to-all**（torch.distributed，含非均匀 split）量原语裸开销，
> 完全可跑。实验 `harness/exp_moe_backward_comm.py`（+`.sh`/`.out.txt`），结论数字全部来自真机。
> 源码坐标：`~/github/DeepEP`（commit `93eb6eb`，v2.5.0）。目标卡代号敏感：全文用「目标卡」。

---

## 1. 反向的两个通信原语：dispatch 与 combine 互为转置

第 95 篇已经证明：把 dispatch 看成线性算子 $D:\mathbb R^{T\times H}\to\mathbb R^{N\times H}$
（复制 + 置换），combine 的「按 token 求和」看成 $C:\mathbb R^{N\times H}\to\mathbb R^{T\times H}$，
则 $C=D^{\mathsf T}$、$D=C^{\mathsf T}$。**两个原语互为转置**，于是反向传播不需要任何新的通信 kernel：

| 前向 | 反向（w.r.t. 输入） | 反向调用的原语 |
|---|---|---|
| `recv_x = D(x)` | `grad_x = Dᵀ(grad_recv_x)` | **combine** |
| `combined_x = C(y)` | `grad_y = Cᵀ(grad_combined_x)` | **dispatch** |

这在 DeepEP 官方的**训练 helper** 里被写得明明白白（`README.md`）：

- `dispatch_backward`（`README.md:204-221`）内部调 `_buffer.combine`，注释
  「The backward pass of MoE dispatch is actually a combine.」（`README.md:208`）；
- `combine_backward`（`README.md:242-262`）内部调 `_buffer.dispatch`，注释
  「The backward pass of MoE combine is actually a dispatch.」（`README.md:245`）。

两个 helper 都带 `async_with_compute_stream=True`（`README.md:217`、`:252`），并且
`combine_backward`/`dispatch_forward` 还带 `do_expand=True`、`do_zero_padding=True`
（`README.md:197-198`、`:253-254`）。**这两点决定了下面的通信量模型和重叠结论。**

| 反向相位 | 物理含义 | 调用的原语 | 输入（梯度） |
|---|---|---|---|
| combine_bwd | `grad_y = Cᵀ(grad_combined_x)` | dispatch（expanded） | 输出层的 `dY`（$T\!\times\!H$，在源 rank） |
| dispatch_bwd | `grad_x = Dᵀ(grad_recv_x)` | combine | 专家反传的 `dX_e`（在专家 rank，按 token 归约回源 rank） |

---

## 2. 通信量模型：为什么反向反而更大

### 2.1 前向：dispatch 去重、combine 放大

设每 rank 本地 $T$ 个 token、hidden $H$、每 token 选 $k$ 个专家、专家总数 $E=EP\cdot e$（每
rank $e$ 个专家）。关键差异在**两个方向的复制因子不一样**：

- **dispatch（前向，FP8 1B）**：传输层按**目标 rank 去重**——一个 token 的多个 top-k 若落在同一
  rank，只在线上传一份。设备代码是 `dispatch.cuh:104` 的 `ptx::deduplicate(dst_rank_idx, lane_idx)`
  （`deep_ep/include/deep_ep/impls/ep/dispatch.cuh`），参考实现是
  `refs.py:67` 的 `mask_to_send = (...).any(dim=1)`（`deep_ep/utils/refs.py`）。于是每 token 只发
  $d$ 行（$d$ = 该 token 的 top-k 专家落在几个不同 rank 上，$d\le k$）：

  $$B_{\text{disp}} \;\approx\; T\cdot d\cdot H\cdot 1\text{ B}.$$

- **combine（前向，BF16 2B）**：专家输出是「每 (token, topk) 一行」，$k$ 份各自不同、无法去重，
  都在 **expanded** 空间送回源 rank 再做加权归约（`refs.py:177` 的 `combine` /
  `refs.py:201` 的 `grouped_reduce`）：

  $$B_{\text{comb}} \;\approx\; T\cdot k\cdot H\cdot 2\text{ B}.$$

### 2.2 反向：两个 expanded 原语，$T\!\cdot\!k$ 行走两趟

反向的两个相位（§1 表）都是 **expanded**（`do_expand=True`）：

- `combine_bwd = dispatch(dY)`：把源 rank 的 `dY` 复制到 $k$ 个专家槽 → $T\!\cdot\!k$ 行、BF16；
- `dispatch_bwd = combine(dX_e)`：把专家侧的梯度按 token 归约回源 rank → $T\!\cdot\!k$ 行、BF16。

于是

$$B_{\text{bwd}} \;\approx\; 2\cdot T\cdot k\cdot H\cdot 2\text{ B},\qquad
\frac{B_{\text{bwd}}}{B_{\text{fwd}}}
=\frac{4k}{d+2k}.$$

代入 DeepSeek-V3 量级（$k=8$、$EP=8$、$E=64$，随机路由下 $d\approx5.4$）：
$4k/(d+2k)=32/21.4\approx\mathbf{1.49\times}$。**反向的通信字节量比前向大约 1.5 倍**——因为它
把最贵的 BF16 expanded 原语走了两遍，而前向里最便宜的 FP8 dedup 原语只走一遍。

> 这与「反向通信规模小」的直觉相反。直觉来自「wgrad 留守专家本地 rank、无需 all-reduce」——
> 那说的是**权重梯度**（`dW`）不跨 rank；但**激活梯度**（`dX`/`dY`）必须跨 rank 来回，
> 而且因为 expanded 布局，它的量比前向只多不少。

---

## 3. H100 8 卡实测：真实 ragged all-to-all

实验用 torch.distributed(NCCL) 的 `all_to_all_single` 精确复刻四相位的搬运量与
**非均匀 split**（routing 用 `refs.dispatch` 语义的非 expand 布局构造）。配置：
$H=7168$、$EP=8$、$E=64$、$k=8$、$T\in\{1024,4096,16384\}$，30 次迭代取均值。
每相位的字节数按**本 rank 发出的行数**计（`exp_moe_backward_comm.py`）。

```
T=  4096  E=64 topk=8 | d=5.418  dedup 行数 Nd=  22193 (5.42×T)  expand 行数 Ne=  32768 (8.00×T)
    [1] dispatch      FP8     159.1 MB     1393.5 us    114.2 GB/s   (dedup T·d)
    [2] combine       BF16    469.8 MB     9957.1 us     47.2 GB/s  (a2a 4072.9 us=115.3GB/s + reduce 5931.6 us, expand T·k)
    [3] combine_bwd   BF16    469.8 MB     3901.6 us    120.4 GB/s   (= dispatch 伴随, expand)
    [4] dispatch_bwd  BF16    469.8 MB     9959.5 us     47.2 GB/s  (a2a 4064.3 us + reduce 5931.6 us = combine 伴随)
    前向 a2a    5466.4 us | 反向 a2a    7965.8 us | 时间比=1.46× | 字节量比 B_bwd/B_fwd=1.49×
```

| 量 | 结论（H100 真机） |
|---|---|
| ragged all-to-all 单向带宽 | **114–124 GB/s**（$d\approx5.4$、每 rank 向 ~5.4 个不等长目的发） |
| 反向/前向 字节量比 | **1.49×**（与模型 §2.2 一致） |
| 反向/前向 a2a 时间比 | **1.46–1.54×**（T=1024:1.54, 4096:1.46, 16384:1.50） |
| combine 的本地按 token 求和 | 与 a2a **同量级**（数据量 = expanded 行数×H），不可忽略 |

三点解读：

1. **时间比 ≈ 字节量比**：说明四相位都落在这台机器 ragged all-to-all 的带宽平台上，
   没有哪个相位被固定时延主导（$T\ge1024$，消息已够大）。模型的 1.49× 被真机坐实。
2. **ragged 相对 equal-split 有明显惩罚**：第 1 篇 `exp_ep_comm.{out.txt}` 里 8 等分
   `all_to_all_single`（每 rank 等量发往 8 个 peer）有效带宽在 300–390 GB/s 区间；这里因为
   $d\approx5.4$ 且各目的长度不等，单向带宽掉到 ~120 GB/s——**约 2–3× 的拆分惩罚**，
   这也是 DeepEP 要按目标 rank 去重、把「每 rank 的行数」凑匀的原因之一。
3. **本地求和不是零头**：combine / dispatch_bwd 的 `[2]/[4]` 里，a2a 只占 ~4.1 ms、本地
   `index_add` 归约占 ~5.9 ms。**它的数据量与通信量同级**（都是 expanded 行数×H）。真实
   combine kernel 把这个归约融进通信 kernel 的片上归约（`combine_reduce_epilogue.cuh:87`
   的 `ptx::deduplicate`），所以不会像 PyTorch `index_add` 这样慢；但「归约本身有一份和通信
   同级的显存流量」这个事实，正是 Mega MoE 要把 combine 也融进去的动机。

> 说明：`reduce` 一列是 **PyTorch `index_add` 朴素路径**的耗时（原子密集，非代表性实现），
> 这里只用来说明「本地归约有与通信同级的**数据量**」，不代表真实 combine kernel 的耗时。

---

## 4. 通信 vs 计算：谁才是反向的瓶颈

第 14 篇（`docs/14`）量过 SM90 四个反向 grouped GEMM 的吞吐（$T=2048$、$H=7168$、$I=2048$、
$E=8$、topk=2）：w2g 1171 us、w1g 2268 us、d2g 268 us、d1g 443 us，**四者合计约 4.15 ms**。
把本篇的通信量放到同一把尺子上（$T=4096$、$k=8$、$EP=8$）：

| 项 | H100 实测 | 备注 |
|---|---|---|
| 反向通信 a2a（两相位） | **~8.0 ms** | combine_bwd + dispatch_bwd（均为 expanded BF16） |
| 反向本地归约（PyTorch 朴素） | ~5.9 ms | 真实 kernel 融进通信，不计入 |
| 反向四个 grouped GEMM（第 14 篇） | ~4.15 ms | wgrad ~100 TFLOP/s（K=token 小）、dgrad ~450–540 TFLOP/s |

**结论**：在 DeepSeek-V3 量级（$k=8$、$EP=8$）下，反向的**通信与计算是同量级的**
（~8 ms vs ~4–6 ms），谁都不是可以忽略的零头。所以真正该问的不是「反向通信小不小」，
而是「**反向的重叠为什么不在 kernel 内做**」。

---

## 5. 为什么反向没有 mega-kernel

第 14 篇给过一句话「反向无 mega-kernel」。结合本篇的模型与实测，可以给出更准确的三条理由：

1. **计算侧天然稀疏可分解**：反向是**四个相互独立**的 grouped GEMM（wgrad 沿 K=token 分组、
   dgrad 沿 M=token 分组，见第 12/13/14 篇），不像前向那样是「dispatch→L1→SwiGLU→L2→combine」
   一条必须在同一 kernel 内流水线化的**长依赖链**。把四个独立 GEMM 融进一个 kernel，只会
   增加调度复杂度而不减数据流。
2. **反向的通信不是「两端」而是「两张网」**：前向是 dispatch 进、combine 出；反向是
   combine_bwd 进、dispatch_bwd 出，**两个方向的原语刚好对调**（§1）。要在 kernel 内把
   「专家 GEMM 的 warp」与「两张网的搬运 warp」都排好，收益远不如前向（前向只有一条 dispatch
   需要与 L1 重叠）。
3. **重叠交给通信库的 host 层**：官方 helper 全部 `async_with_compute_stream=True`
   （`README.md:217/252`）——把通信 launch 到独立 stream，让 **host 侧**决定何时与计算 overlap
   （配合 `defer_epilogue=True`，见 `README.md:285`）。前向用**设备端 warp 级重叠**（Mega MoE kernel），
   反向用**库层 stream 重叠**，两者目标相同、实现层次不同。

**库层重叠的收益有多大？** 本机 8 卡实测（T=4096，把 combine-BF16 的 a2a 与一次等量 dgrad GEMM
叠到两条 CUDA stream）：

```
    通信单独    3893.9 us | 计算单独    3870.0 us | 两流并发    7484.3 us
    串行基线    7763.8 us -> 重叠省    279.5 us，重叠效率   7.2% (相对 min)
```

**只有 ~7% 重叠**——因为 torch/NCCL 的 all-to-all 会用满 SM，与 GEMM 争抢计算资源，几乎不能并行。
这正是 DeepEP 为什么要把通信 kernel 限制在**少数 SM**上：`EPBuffer.get_theoretical_num_sms`
（`deep_ep/buffers/ep.py:518-532`）按流量算出需要的 SM 数，并在
`prefer_overlap_with_compute` 时**不**把 `num_sms` 放大（`:531`）。**留出 SM 给计算，才有真重叠**
（对应第 73 篇的 SM/QP 自动选择）。前向的 Mega MoE 用的是同一思路的**设备端版本**：
dispatch warp 与 MMA/epilogue warp 各占一部分 SM/warp 资源（第 3/9 篇）。

> **仅分析**：真实 `EPBuffer` 的重叠收益受 NCCL GIN 设备通信 API 与 `prefer_overlap_with_compute`
> 的 SM 分配共同决定，本机 NCCL 2.29.7 不满足设备 API 需求（「阻塞」），故这里只给 torch/NCCL 的
> all-SM 上限（7%），真实收益需在 ≥2.32.3 环境复现。

---

## 6. 结论与边界

1. **反向通信不比前向小**：模型 $B_{\text{bwd}}/B_{\text{fwd}}=4k/(d+2k)\approx1.49$（$k=8,d\approx5.4$），
   H100 8 卡 true-ragged a2a 实测时间比 **1.46–1.54×**——字节量与时间同时坐实。原因是布局不对称：
   前向 dispatch 去重（FP8 $T\!\cdot\!d$），combine/两个反向相位都在 expanded 空间（BF16 $T\!\cdot\!k$），
   而反向把 expanded 原语走了两遍。
2. **ragged split 有 2–3× 惩罚**：$d\approx5.4$ 的不等长拆分把单向带宽从 ~300+ GB/s 压到 ~120 GB/s；
   `refs.py:67`/`dispatch.cuh:104` 的按目标 rank 去重是压低这个惩罚的关键。
3. **本地归约与通信同级**：combine「按 token 求和」的数据量与通信量都是 expanded 行数×H；
   真实 kernel 把它融进片上归约。
4. **反向无 mega-kernel 不是因为通信小**，而是因为计算天然可分解为四个独立 grouped GEMM、
   通信是两张方向相反的网；重叠改由通信库的 `async_with_compute_stream` 在 host 层做，且
   必须**限制通信 SM 数**才有真重叠（`ep.py:518-532` 的 `num_sms`）。
5. **边界**：本篇量的是 **NVLink 单域**（8×H100，无 scaleout/RDMA）的裸 a2a 与 torch/NCCL 重叠上限；
   真实的 DeepEP 两级域（hybrid，第 61 篇）、`defer_epilogue`、SF 打包等会改变绝对数字，但
   「反向 ≈ 1.5× 前向」这一布局层面的比值与实现无关。

---

## 参考

- 前向通信量模型：第 1 篇 §2.1（`docs/01-moe-ep-basics.md:167-205`）。
- 反向数学与四项梯度：第 4 篇（`docs/04-moe-backward.md`）。
- 反向 kernel 接入与吞吐：第 14 篇（`docs/14-moe-backward-kernels.md`）。
- dispatch/combine 互为转置 + 路由权重梯度：第 95 篇（`docs/95-ep-backward-adjoint.md`）。
- DeepEP 训练 helper：`README.md:171-262`；`dispatch.cuh:104`；`refs.py:10-243`；
  `deep_ep/buffers/ep.py:518-562`。
- DeepEP 仓库：https://github.com/deepseek-ai/DeepEP （本地 `~/github/DeepEP`，commit `93eb6eb`）。
- 实验：`harness/exp_moe_backward_comm.py` / `.sh` / `.out.txt`（H100 8 卡，真机数字）。
