# KDA 专属 fused pre-GDR：把 `docs/14` 的契约实现成 Triton 核族

> 本子专题此前把 FLA KDA 移植进 Megatron-LM 的数值/接口逐层锚定（M0–M12，见
> `docs/00`–`docs/14`）。`docs/14` 给出结论：GDN 的 `fused_streamed_pre_gated_delta_rule`
> 不能复用于 KDA（四处会静默算错的硬差异），并给出一个 KDA 专属融合入口需要满足的
> **代数/布局契约**，用纯 PyTorch 参考 `kda_pre_gdr_fused_ref` 锚定、drop-in 验证。
>
> 本篇（M13-1）兑现 `docs/14` §7 的「下一步」：**按该契约把 KDA 专属 fused pre-GDR
> 真正实现成一个 Triton 核族**，并给出（a）逐张量数值锚定、（b）整层 drop-in、
> （c）隔离计时（wall vs device）。实现见 `src/kda_pregdr_fused.py`，实验见
> `harness/check_port_pregdr_kernel.py`（`harness/check_port_pregdr_kernel.out.txt`，
> 空闲卡单进程）。**未改 Megatron-LM 一行**；本文所有数字来自该 out.txt。

## 1. 契约回顾：三个 scope

KDA 的 pre-GDR（`kda.py:519` 的 `pre_gated_delta_rule`）在核心 `chunk_kda` 之前做三件事，
`docs/14` §5 已把它固化成契约：

1. **conv scope**：对 `qkvfg` 的前 `2qk+v` 个通道做 **depthwise causal conv1d（核宽 `W`，
   左 padding `W-1`）+ SiLU**；**不碰** `raw_g`/`gate`。
2. **qk scope**：把 `qkv` 的 `query_key`（`2·Hk·KD`）reshape 成 `[B,T,2Hk,KD]` 做
   **q/k L2-norm**（fp32 内部累加，同 FLA `l2norm`），再 split；GVA repeat 因子为 1
   （Megatron KDA 硬约束 `Hv==Hk`）。
3. **g/beta scope**：`raw_g` reshape 成 `[B,T,Hk,KD]`；`beta` 走 `sigmoid`（fp32）；
   **不做** decay 激活（`use_gate_in_kernel=True`，`A_log`/`dt_bias` 作为 kernel 输入
   交给 `chunk_kda`）。

输出 `(q, k, v, gate, beta, raw_g, A_log, dt_bias)`，其中 `gate`/`raw_g` 尽量以 `qkvfg`
的 view 返回（streamed）。`src/kda_pregdr_fused.py` 用 **三个 Triton kernel** 一一对应。

## 2. Triton 核族设计

三个 kernel 都在 `src/kda_pregdr_fused.py`，host 编排在 `_forward_raw`（`:127`）。

### 2.1 conv scope：`_kda_conv_silu_kernel`（`:46`）

depthwise conv 每个输出通道只依赖自己那一路输入，故按 `(channel-block, time-block, batch)`
三维网格并行。逐 tap 累加，**完全照抄 FLA 的 `causal_conv1d_fwd_kernel`**
（`fla/modules/conv/triton/kernels.py:36`）的写法：fp32 累加、`i_w` 从 `-W+1` 到 `0`、
taps 索引 `i_w+W-1`，最后 `silu(y)=y·sigmoid(y)`、写回原 dtype。之所以照抄而非「等价重写」，
是为了让输出与 eager 路径**逐位可比**（见 §3）：同一批 tap 用同一次序求和，
`b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1))[None, :], 1)`（`:77`）与 FLA 逐字一致。

### 2.2 qk scope：`_kda_l2norm_kernel`（`:94`）

按 `BT=64` 行一块、每行在 `KD` 维上做 fp32 累加求 `rstd = 1/sqrt(Σx²+eps)`，
再 `x*rstd` 写回原 dtype——这正是 FLA `l2norm_fwd_kernel` 的块化形式
（`fla/modules/l2norm.py:94`）。**这个块化是刻意做的**：初版「一行一 program」在
`B·T·2Hk` 很大时设备时间明显吃亏（见 §4.3 的坑）。

### 2.3 g/beta scope：`_kda_beta_sigmoid_kernel`（`:110`）

`beta` 的 `sigmoid` 是纯逐元素 fp32，写回 fp32（eager 也是 `.float().sigmoid()`）。
`raw_g` 只做 reshape（零成本 view），`A_log`/`dt_bias` 直接透传。

### 2.4 host 编排与 streamed 账

`_forward_raw`（`:127`）先 `torch.split(qkvfg.transpose(0,1), (C, qk, vd))` 把 `qkv`、
`raw_g`、`gate` 切成 view，conv 只作用在 `qkv` 上（`qkv.contiguous()` 一次，与 eager 路径
`conv_input = qkv.contiguous()` 同款），conv 输出落到新缓冲 `conv`；l2norm 读 `query_key`
写 `qk_norm`；`value`/`gate`/`raw_g` 都是 view。**新分配只有 `conv` 和 `qk_norm` 两块**，
与 eager 自己的新分配（l2norm 核写出的 q/k）同构——**`gate` 不再单独物化**，这正是
`docs/14` §6 Part D 点名的唯一「实打实拷贝」。

## 3. 数值锚定（真实数字）

### 3.1 Part A：Triton 核族 vs 真实层 eager `pre_gated_delta_rule`（逐张量 `max_abs`）

| dtype | T=64 B=2 | T=128 B=2 | T=250 B=1 | T=256 B=1 | T=1024 B=2 |
|---|---|---|---|---|---|
| fp32 worst | 1.19e-07 | 1.19e-07 | 5.96e-08 | 1.19e-07 | 1.19e-07 |
| bf16 worst | **0** | 2.44e-04 | 9.77e-04 | **0** | 1.95e-03 |

逐项：fp32 下 `v/gate/beta/g` **全部逐位 0**，只有 `q/k` 剩 ~1e-7（l2norm `rstd` 的
fp32 求和序）；bf16 下 `v/gate/beta/g` 也全 0，`q/k` 偶尔差 ≤1 ulp bf16
（1 ulp bf16 在 1.0 处是 2⁻⁸≈3.9e-3，故 1.95e-3 = 0.5 ulp）。**结论：conv/beta/reshape
逐位、l2norm 到 fp32 舍入/亚 ulp**——与 `docs/14` §6 Part B 用 FLA 原语时的结论一致，
说明「自研 conv」在 fp32 累加 + 同序 tap 下与 FLA causal_conv1d 等价到 fp32 舍入。

### 3.2 Part B：与纯 torch 原语参考（conv/l2 = torch）对拍

| dtype | T=64 B=2 worst | T=250 B=1 worst |
|---|---|---|
| fp32 | 1.19e-07 | 1.79e-07 |
| bf16 | 3.91e-03 | 3.91e-03 |

bf16 下与 torch `F.conv1d` 差 3.91e-03（=1 ulp），而 vs eager（FLA conv）逐位 0——
即**残差来自 conv 后端舍入**，与 `docs/14` §6 Part B 的归因完全一致（自研核走 FLA 的序，
所以比 torch 原语更贴 eager）。

### 3.3 Part C：整层 drop-in（同权重，只换 `pre_gated_delta_rule`）

| dtype | out max_abs | hs.grad max_abs | param.grad max_abs | param.grad rel |
|---|---|---|---|---|
| fp32 | 1.23e-04 | 3.04e-04 | 1.46e-02（max\|g\|=70.2） | 2.08e-04 |
| bf16 | **0** | 7.81e-03 | 2.50e-01（max\|g\|=70.5） | 3.55e-03 |

与 M12-1（`docs/14` §6 Part C）用纯 torch 参考的整层 drop-in 同阶（fp32 ~1e-4、bf16 ~3e-3）。
bf16 前向输出**逐位相同**，说明 Triton 前向对 `chunk_kda` 是数值透明的。

> **诚实边界**：Triton 核族是 **forward-only**；Part C 的梯度经
> `_FusedPreGDRAutograd`（`:190`）的 **参考（torch 原语）backward** 回传，故其梯度值反映
> 的是代数契约的梯度透明性（同 M12-1），而**不是** Triton conv/l2norm 的 kernel backward。
> 原生 Triton 反向是另一件事（见 §6）。

## 4. 隔离计时：wall vs device

对 `kda.pre_gated_delta_rule`（eager：FLA `causal_conv1d` + `l2norm` + torch split）与
Triton 核族各做「孤立 pre-GDR」计时（warmup=25）：
`wall` = CUDA event 对包 `N=100` 次（含 host），`dev` = `torch.profiler` CUPTI 各 kernel
`self_device_time` 之和 / `N`（纯设备，µs）。

### 4.1 fp32

| shape | wall eager | wall triton | wall × | dev eager | dev triton | dev × |
|---|---|---|---|---|---|---|
| T=64 B=2 | 429.7 | 155.3 | **2.77×** | 13.45 | 18.27 | 0.74× |
| T=256 B=2 | 455.2 | 191.1 | **2.38×** | 15.85 | 19.61 | 0.81× |
| T=1024 B=2 | 459.1 | 190.8 | **2.41×** | 25.56 | 25.39 | 1.01× |
| T=1024 B=8 | 466.2 | 193.8 | **2.41×** | 71.31 | 59.64 | **1.20×** |

### 4.2 bf16

| shape | wall eager | wall triton | wall × | dev eager | dev triton | dev × |
|---|---|---|---|---|---|---|
| T=64 B=2 | 462.1 | 194.2 | **2.38×** | 13.94 | 22.86 | 0.61× |
| T=256 B=2 | 459.7 | 193.6 | **2.38×** | 16.12 | 23.72 | 0.68× |
| T=1024 B=2 | 458.3 | 192.6 | **2.38×** | 23.70 | 26.59 | 0.89× |
| T=1024 B=8 | 460.3 | 195.2 | **2.36×** | 57.68 | 49.91 | **1.16×** |

**关键读数**：eager 的 `wall` 与 T 几乎无关、恒 **~450 µs**——这正是 P7-6 在 FLA 前向
上发现的「每调用 Python 包装税」（`causal_conv1d`/`l2norm` 的 Triton autotune wrapper +
split），**不是 GPU 时间**；它的 `dev` 只有 13–71 µs。Triton 核族把墙钟砍到 **~190 µs
（2.4–2.8×）**，靠的是**少发 Python launch**（3 个核 vs eager 的多段 torch/FLA 调用），
与「融合省 launch + 省中间激活」的卖点吻合。**在纯设备时间上**，小 shape 我们略慢
（0.6–0.8×，conv 网格小、3 次 launch 的固定开销占主导），大 shape 反转领先
（`B8T2048` 达 1.16–1.20×）。

### 4.3 一个真实的坑：l2norm 一行一 program

初版 l2norm 是「一行一 program」（每 program 只处理 `KD=32` 个 float），在**所有** shape
上 `dev` 都是 0.60–0.74×（被小 program 的调度/占用拖累）；改成 §2.2 的 `[BT,BD]` 块化后，
大 shape `dev` 从 ~0.74× 抬到 **1.16–1.20×**。教训与主系列 P7-3/P7-9 一致：**格状小 program
的 device 时间不可忽略，孤立计时必须看 CUPTI 而非只看墙钟**。

## 5. 边界

- **前向 only**：核族没有原生反向；Part C 的梯度走参考 backward（§3.3）。
- **布局/并行**：单卡 `tp=cp=1`、SBHD、`use_qk_l2norm=True`、GVA 因子 1
  （Megatron KDA 硬约束）。packed/THD、CP、`conv_bias=False` 未覆盖。
- **`deterministic_mode`**：eager 在该模式下走 `F.conv1d` 分支（`kda.py:558`）；本核族
  只重建了 FLA causal-conv 分支的语义（两者在 fp32 舍入内一致，见 §3.2），未做
  cu_seqlens/varlen 的 conv。
- **未接进 Megatron**：`kda.py:399-402` 的两道 guard 未动；本核族只是「可运行参照」，
  证明契约可实现、且数值/性能账成立。

## 6. 下一步

- 若要**真正关闭**该缺口：把本核族接到 `kda.py:399` 分支（去掉 guard），并把
  `_FusedPreGDRAutograd` 的参考 backward 换成一个 **原生 Triton backward**
  （`causal_conv1d_bwd` + `l2norm_bwd` + sigmoid bwd），才能让「融合」在反向也省 launch。
- 本子专题此前所有 M0–M12 目标均已完成；M13-1 补上了「实现参照」这一可选工程项。
