# Kimi-K3 的 AttnRes（Attention Residuals）在 FLA 中的实测

> **结论先说**
>
> - Kimi-K3 已经把 PreNorm 的 `h = h + sublayer(h)` 换成了**注意力残差 AttnRes**：每个
>   sublayer 的输入是所有 depth source（已完成 block 的残差和 + 当前 block 的 partial sum）
>   的 softmax 加权和。K3 的 `attn_res_block_size=12`（层）意味着一个 sublayer 要看
>   **L = 2…9** 个 source，93 层共 **186 次** AttnRes 调用/迭代（185 次在层内 + 1 次输出头）。
> - 所以这是**纯带宽算子**：`fused_attnres` 在 K3 形状（D=7168、bf16、T=4096）单次
>   **fwd 237 µs / bwd 770 µs（CUPTI）**，合计约 **1.0 ms**；fwd ≈ 2.5 TB/s、bwd ≈ 2.7 TB/s
>   （H100 峰值 3.35 TB/s，利用率 74–82%）。
> - 按实测逐 L 聚合 K3 的 186 次调用，**T=4096 一个训练迭代的 AttnRes 开销约 130 ms**
>   （单次 fwd 237 / bwd 770 µs 是 L=9 的值，**单次约 0.1–1 ms**；130 ms 是 186 次串行之和，
>   fwd 28.8 + bwd 101.0），搬运 **344.6 GB**；对比本目录先前实测的 KDA（527 ms）+ MLA（97 ms），
>   相当于在注意力主干之上再多 **~21%**。相对普通 PreNorm 残差（32.8 GB）是 **~10.5×** 的残差流量。
> - FLA 的 Triton 实现把「逐 source 的 RMSNorm/打分、softmax、加权求和、输出 RMSNorm」
>   全塞进**一个 kernel**，反向拆成两个**无 atomic** 的 kernel；对比 FLA 自带 naive
>   PyTorch 实现（同为 T=4096/L=9），fwd 快 **9.0×**、bwd 快 **7.7×**，峰值显存从 4.1 GB 降到 0.78 GB。
> - 另有 **Gluon 后端**：L ≤ 8 时反向比 Triton 快 17–44%，但 **L=9（K3 的最大 L）会崩**
>   （bwd 慢 3.7×），H100 上不建议直接用；K3 生产走 Triton。
>
> 数据：[`attnres_kimi_k3.csv`](attnres_kimi_k3.csv)、
> [`attnres_kimi_k3.log`](attnres_kimi_k3.log)（sweep）、
> [`attnres_check.log`](attnres_check.log)（正确性）、
> [`attnres_schedule.log`](attnres_schedule.log)（逐 L 聚合）、
> [`attnres_extras_4096.log`](attnres_extras_4096.log)、[`attnres_extras_16384.log`](attnres_extras_16384.log)。
> 脚本：[`bench_attnres_kimi_k3.py`](bench_attnres_kimi_k3.py)，可复跑。

## 1. AttnRes 是什么，K3 的 shape 与调用次数

AttnRes（Attention Residuals, Kimi Team, arXiv:2603.15031）把「残差流」看成可注意的
depth 维度：每个 sublayer 有一个可学习的 pseudo-query `q_l ∈ R^D`，对每个 depth source
`V_j` 做 RMSNorm 后打分、softmax、加权求和：

```
h_l = Σ_j α_j · V_j,     α = softmax_j( ⟨RMSNorm(V_j), q_l⟩ )
V = [b_0, …, b_{k-1}, partial]
```

其中 `b_i` 是已完成 block 的残差和，`partial` 是当前 block 内的累计和（`partial += sublayer_out`）。
这等价于把「深度」当成一个长度最多 9 的序列做一次 attention —— **没有矩阵乘，只有逐元素
+ 行归约 + softmax**，因此性能只由 HBM 流量决定。

从 `~/models/Kimi-K3/config.json` 的 `text_config`：

| 项 | 值 | 说明 |
|---|---|---|
| `hidden_size` | **7168** | 所有 source 的 D |
| `num_hidden_layers` | **93** | 69 KDA + 24 MLA |
| `attn_res_block_size` | **12** | 每 12 层（=24 个 sublayer）收一个 block |
| 每层 AttnRes 调用 | **2** | attention 前 + MLP 前各一次 |
| 每次调用的 source 数 L | **2…9** | 已完成的 block 数 + 当前 partial |

调用语义和 Megatron-LM 分支 `to-main/attnres-6840`
（`megatron/core/transformer/attention_residual.py`）一致：FLA 调用时
`output_rms_weight` 传下一个 RMSNorm 的权重（把 prenorm 融进 kernel）、`scale=1.0`、
`checkpoint_level=1`；pseudo-query 与 key-norm 权重保持 fp32。

L 的分布（脚本 `k3_sublayer_schedule()` 复刻 `modeling_kimi_linear._forward_attn_residual`
的 push 逻辑算出）：

| L | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 调用次数 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | **18** |

合计 **186** 次 = 93 层 × 2 − 1（第 0 层 attention 前没有 block，且单 source 的聚合被
`_aggregate` 短路成恒等）+ 1（最终输出头 `_apply_output_attn_res` / `final_attn_res`，L=9）。
注意 L=1 从不真正下发给 kernel：第一次聚合就已经是「embedding + attention 输出」两个 source。

### 1.1 精确形式（与 HF `modeling_kimi_linear._apply_attn_res` / Megatron `AttentionResidual` 逐行对应）

记号：`l` 是 sublayer 下标，`V_j ∈ R^{N×D}` 是第 j 个 depth source，`q_l, w_l ∈ R^D` 是该 sublayer
的 pseudo-query 与 key-norm 权重（`l` 省略）：

```
源列表   V = [b_0, b_1, …, b_{k-1}, partial]        # k 个已完成 block + 当前 partial
         b_0 = embedding
         b_i = Σ_{l ∈ block i} y_l                  # 块内各 sublayer 输出之和（不是累积流）
         partial = Σ_{l ∈ 当前 block, 已执行} y_l

打分     r_j = sqrt( (1/D) Σ_d V_{j,d}^2 + ε )       # RMS，无均值、无 bias
         s_j = Σ_d w_d q_d V_{j,d} / r_j = <w ⊙ q, V_j / r_j>
         α   = softmax_j(s)                          # 沿 depth 维，j = 0…k

聚合     h_in = Σ_j α_j V_j                          # 凸组合（不是求和）

sublayer y_l  = F_l( RMSNorm_{γ_l}(h_in) )            # γ_l 是下一层的 prenorm 权重；
                                                     # 开了 norm fusion 时这一步在 AttnRes kernel 里完成

输出头   h_out = Σ_j α'_j V'_j,  V' = [b_0…b_7, 最后一个 block 的 partial]   # L=9，再过最终 RMSNorm
```

关键点：

- **`b_i` 是块内输出之和，不是累积残差流**。HF 里 `prefix_sum` 在 block 边界被置 `None`，
  之后只累加本 block 的 `attn_out / mlp_out`；到下一个边界才把它 append 成 `b_i`。所以
  `V` 里装的是「embedding + 各 block 的净贡献 + 当前 partial」，聚合是 softmax 加权平均。
- **只有 `w ⊙ q` 进入 logits**：`s_j = (w⊙q)·V_j/r_j`。`w` 和 `q` 的分解在数学上是冗余的
  （任意 `δw = w⊙ε, δq = −q⊙ε` 都不改变输出），FLA kernel 也确实是先把 `q`、`rms_weight`
  逐元素乘起来再用。
- **初始化**：`q = 0`、`w = 1`（Megatron `pseudo_query` zero-init，`key_norm_weight` ones）。
  `q=0 ⇒ s≡0 ⇒ α` 均匀，等于对 depth 做平均；因为 `h_in` 紧接着就被 RMSNorm，均匀权重的
  `1/(k+1)` 缩放会被 norm 消掉，所以起点接近 PreNorm。零初始化同时是论文要求的训练稳定性条件。
- **常数**：K3 `ε = rms_norm_eps = 1e-5`，`scale = 1.0`（FLA 的 `scale` 只乘 logits），
  聚合全程 fp32，输出 cast 回 bf16。

反向（FLA 两段式 kernel 对应的解析式，`g = ∂L/∂h_in`）：

```
dp_j = α_j · g·(V_j − h_in)                    # softmax 的 Jacobian 投影
t_j  = (w⊙q) − s_j · V_j / D
dV_j = α_j · g + dp_j · t_j / r_j              # 每个 source 一份
dq   = Σ_j dp_j · (w ⊙ V_j / r_j)
dw   = Σ_j dp_j · (q ⊙ V_j / r_j)              # dq/dw 只有 [D]，沿 N 归约
```

其中 `t_j` 来自 RMSNorm 的 Jacobian（`∂s_j/∂V_j = t_j / r_j`）；`dq/dw` 的 `Σ_j` 在 batch 上
还要再过一遍 `N`，这正是 Triton 反向需要 fp32 `dqw` scratch、而 Gluon 用 32-token 分组把它
缩小 32× 的原因。

## 2. 测试口径

- 容器 `kernel_lab`；**空闲 GPU**（`CUDA_VISIBLE_DEVICES=7`，跑前/跑后 0 MiB、0% util）；
  H100 80GB SXM，132 SM；torch `2.12.0a0+…nv26.04`、triton 3.6.0。
- **FLA 是 checkout 里的 0.6.0**（`main @ cf9f520f`，2026-11-03），不是容器 site-packages
  里的 0.5.2 —— 脚本用 `FLA_SRC` 把 checkout 插到 `sys.path[0]`。**不设 `FLA_SRC` 会静默测到
  旧实现**（0.5.2 的 attnres 后端不同）。
- 该 checkout 的 `fla/__init__.py` 会 import `fla.models`，需要 transformers ≥5 的
  `PreTrainedConfig`；容器是 4.57，脚本在 import 前把该符号 alias 到 `PretrainedConfig`
  （AttnRes 本身不碰 transformers）。另外容器里 tilelang 的 TVM 依赖 `libz3.so.4.15`，
  需要把 Megatron-Bridge venv 的 `z3/lib` 加进 `LD_LIBRARY_PATH` 才能 import 整个 `fla.ops`。
- `dev_ms` = **CUPTI 纯 device kernel 时间**（`torch.profiler`，warmup=5、repeat=20），不含 launch；
  `wall_ms` = CUDA event 逐次计时（含 launch / host 派发）。
- **反向独立计时**：图在计时区外建一次，计时区只跑
  `torch.autograd.grad(..., retain_graph=True)`；`step` = 新建图 + fwd + bwd。
- bytes 模型（`bytes_fwd` / `bytes_bwd`）不只算 source：还计入 FLA 反向的 **fp32 `dqw`
  scratch（`[N,D]`，写+读）**、融合输出 norm 时的 **`dow_partial`（同样 `[N,D]` fp32）**，
  以及 `checkpoint_level=0` 保存的 `o_pre`。不算这些会把 bwd 的 GB/s 高估 10–30%。
- **autotune 口径**：FLA 的 autotune key 是 `['L2','D','HAS_ONORM','SAVE_OPRE']`，
  **不含 N**，所以同一进程里第一个 shape 会决定这个 L2 bucket 的 config。本文数字来自
  每个 mode 独立进程；Triton 两次独立 sweep 的偏差 <2%（见 log），Gluon 则见 §4.3 的抖动说明。

## 3. 正确性（bf16 vs fp64 golden）

`--mode check`：D=7168、T=1024、随机 pseudo-query/权重，比较 fwd 输出与所有 VJP
（dV、dq、dw、dow），golden 用 fp64 手写同一公式（含输出 RMSNorm）。

| backend | L | onorm | cos(o) | rel(o) | cos(dV) | cos(dq) | cos(dw) | cos(dow) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| triton | 2 | 1 | 0.9999986 | 1.66e-3 | 0.9999986 | 1.0000000 | 1.0000000 | 1.0000000 |
| triton | 9 | 1 | 0.9999986 | 1.66e-3 | 0.9999986 | 1.0000000 | 1.0000000 | 1.0000000 |
| triton | 9 | 0 | 0.9999999 | 5.18e-4 | 0.9999986 | 1.0000000 | 1.0000000 | — |

Gluon 的 8 行结果与 Triton **逐位相同**（同一份 fp32 数学，只是调度不同），不再单列。
cos≈0.9999986、rel≈1.7e-3 是 bf16 输入 + fp32 累加的预期舍入，不是 bug。

## 4. 实测性能

### 4.1 K3 生产形状 T=4096（Triton，bf16，onorm 融合，ckpt=1）

CUPTI µs；`fwd+bwd` 与单独测的 `step` 相差 <2.5%，可互换使用
（L=2/3/5/8/9 取 sweep，L=4/6/7 取 schedule，两者在同一 shape 上相差 ≤1.5 µs）。

| L | fwd µs | bwd µs | fwd+bwd µs | fwd GB/s | bwd GB/s |
|--:|--:|--:|--:|--:|--:|
| 2 | 74 | 318 | 392 | 2368 | 2766 |
| 3 | 118 | 417 | 535 | 1993 | 2533 |
| 4 | 119 | 438 | 557 | 2468 | 2815 |
| 5 | 158 | 537 | 695 | 2235 | 2625 |
| 6 | 160 | 563 | 723 | 2570 | 2816 |
| 7 | 199 | 656 | 855 | 2362 | 2685 |
| 8 | 200 | 692 | 892 | 2643 | 2799 |
| 9 | 237 | 770 | 1007 | 2475 | 2745 |

两个直接可读出的结论：

1. **时间按 `ceil(L/2)` 呈台阶**（L=3≈L=4、L=5≈L=6、L=7≈L=8）：autotune 给这个 D 选的是
   **每 tile 2 个 source**（`BL=2`），第 9 个 source 单独占一个（只填一半）的 tile。
   L=8→9 只多 1 个 source（字节 +12.5%），但 fwd 实测 +18%（半个空 tile 的浪费），
   bwd 只有 +11%（source 流量占比更大、摊薄了固定开销），合计 +13%。
2. **bwd/fwd ≈ 4.2（L=2）→ 3.3（L=9）**：小 L 时 `dqw`/`dow_partial` 两个 `[N,D]` fp32 scratch
   占了大头，大 L 时 source 流量才是主导。K3 早期层（L=2）恰好落在 scratch 主导的区间。

#### 4.1.1 相对 H100 带宽峰值

把上表按 H100 HBM3 峰值 3.35 TB/s 折算成利用率：

| L | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| fwd %峰值 | 71% | 59% | 74% | 67% | 77% | 71% | 79% | 74% |
| bwd %峰值 | 83% | 76% | 84% | 78% | 84% | 80% | 84% | 82% |

按 K3 的 L 分布加权（186 次）：**fwd 72%、bwd 81%，整体 79%**（roofline 口径见 §4.5）。两点：

- **前向是弱侧**（72% vs 81%）：D=7168 不是 2 的幂，kernel 用 `BD=next_pow2(D)=8192`，
  12.5% 的 lane 被 mask、`b_o` 还要 8192 个 fp32 累加器；L=3/5 又落在 `BL=2` 的半空 tile 上，
  掉到 59%/67%。
- **3.35 TB/s 是标称值**：H100 上 STREAM 类读流通常只到 2.9–3.0 TB/s，按 3.0 折算则
  bwd 2.75 TB/s ≈ 92%、fwd 2.48 TB/s ≈ 82% —— **反向已基本打满可用带宽，前向还留 10–15 个点**。

### 4.2 序列长度扩展（L=9，K3 的最大 L）

| T | fwd µs | bwd µs | fwd GB/s | bwd GB/s |
|--:|--:|--:|--:|--:|
| 1024 | 57 | 200 | 2578 | 2648 |
| 2048 | 121 | 389 | 2435 | 2717 |
| 4096 | 237 | 770 | 2475 | 2745 |
| 8192 | 451 | 1519 | 2603 | 2783 |
| 16384 | 879 | 3021 | 2673 | 2799 |

T=1024/2048 时 fwd 只有 2.58 / 2.44 TB/s（太小，填充/尾部效应），T≥8192 稳定在 2.6–2.8 TB/s。
（2048 是后续为 seq 对比补测的，`attnres_sweep_2048.csv`，同为空闲卡独立进程；其余同一次 sweep。）

### 4.3 后端对比：Triton vs Gluon vs naive

**T=4096，CUPTI µs（Gluon 同为 FLA 0.6.0 的 `FLA_ATTNRES_GLUON=1` 路径）：**

| L | Triton fwd/bwd | Gluon fwd/bwd | Gluon 峰值显存 | Triton 峰值显存 |
|--:|--:|--:|--:|--:|
| 2 | 74 / 318 | **63 / 179** | 175 MB | 392 MB |
| 3 | 118 / 417 | **95 / 246** | 231 MB | 448 MB |
| 5 | 158 / 537 | **134 / 324** | 343 MB | 560 MB |
| 8 | 200 / 692 | **190 / 575** | 511 MB | 728 MB |
| 9 | 237 / 770 | 230 / **2844** | 567 MB | 784 MB |

- **L ≤ 8：Gluon 全面更快**（反向快 17–44%，显存低 30–55%），符合它「按 32 token 聚合
  `dqw` partial、cp.async 流水」的设计。
- **L = 9：Gluon 反向崩到 2.8 ms（Triton 的 3.7×）**，T=16384 时更到 11.1 ms（3.7×）。
  与 `_prune_bwd_smem` 在 `L > 8` 时**只保留 `BT=2` 的配置**一致：D=7168 时 `BD=8192`，
  每 program 两个 token × 8192 个 fp32 累加器，寄存器压力过大（推测溢出到 local memory）。
- 独立进程重复测 3 次，L=9 的 bwd 稳定在 **2.84 ms（±1 µs）**；但另一次「连续压测之后」的
  sweep 在同一 shape 测到 6.0 ms。这类几毫秒的长 kernel 对 GPU 频率/占用很敏感，Gluon 的
  绝对数必须在**空闲卡、独立进程**里取。H100 上仍不建议用它跑 K3 —— 18/186 次调用正好是 L=9。

**naive（FLA 自带 PyTorch 参考实现）作为「不融合」基线**，T=4096/L=9（wall ms）：

| | fwd | bwd | fwd+bwd | 峰值显存 |
|---|--:|--:|--:|--:|
| FLA `fused_attnres`（wall） | 0.370 | 0.973 | 1.001 | 0.78 GB |
| FLA `naive_attnres`（wall） | 3.317 | 7.506 | 10.78 | 4.09 GB |
| 加速比 | **9.0×** | **7.7×** | **10.8×** | **5.2×** |

naive 的慢来自两件事：`torch.stack` 把 L 个 source 复制成 `[L,N,D]` **fp32** 张量（4× 流量），
以及每步都物化中间结果；fused 版本一个 kernel 读完 L 个 source 直接出结果。
naive 这一侧统一用 wall：它是几十个 eager kernel 的拼装，CUPTI 求和在同一形状上两次能差 2×
（profiler 事件缓冲/对齐问题），不适合当逐 kernel 口径；fused 只有 2–3 个 kernel，CUPTI 稳定。

### 4.4 融合与显存开关（T=4096，L=9）

| 项 | 数字 | 解读 |
|---|---|---|
| 输出 RMSNorm 融合 | 237 µs vs 282 µs（bf16 权重）/ 711 µs（fp32 权重） | 融合省掉一次 `[N,D]` 读+写（117 MB，实测 45 µs / 2.6 TB/s）；Megatron 把 norm 权重留 fp32，torch 的 `rms_norm` 会退化成非融合路径（+474 µs），融合收益更大 |
| `checkpoint_level=0` | bwd 767 µs、峰值 840 MB | 存 `o_pre`，省掉反向里 L 次 source 重读 |
| `checkpoint_level=1` | bwd 775 µs、峰值 784 MB | Megatron 默认；只慢 1.2%，省 56 MB |
| L=8 → L=9 | fwd 201→234、bwd 694→773 | 多一个 source（且跨到 L2=16 的指针表桶）；桶本身的影响在噪声内 |
| `BL` 台阶 | L=3≈L=4、L=5≈L=6、L=7≈L=8 | 每 tile 2 个 source（`BL=2, warps=4, stages=3`） |

`checkpoint_level` 的取舍比预期小：反向更可能是被 `BD=8192` 的大累加器/占用率卡住，
而不是被「重算读 source」卡住，多读的那点 source 被掩盖了。**显存紧张时放心用 1，几乎不掉性能。**

### 4.5 整个 K3 schedule 的每次迭代代价

`--mode schedule` 用逐 L 实测值 × K3 的 L 分布聚合（Triton，bf16，含 fwd+bwd）：

| T | fwd ms | bwd ms | 合计 ms | 流量 GB | 带宽 roofline ms | 效率 |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 7.20 | 26.36 | **33.6** | 86.2 | 25.7 | 76.5% |
| 4096 | 28.79 | 100.96 | **129.8** | 344.6 | 102.9 | 79.4% |
| 8192 | 55.40 | 199.26 | **254.7** | 689.2 | 205.7 | 80.9% |
| 16384 | 108.07 | 394.01 | **502.1** | 1378.4 | 411.5 | 82.0% |

拆开看（T=4096）：fwd 28.79 ms vs roofline 20.8 ms = **72%**，bwd 100.96 vs 82.0 ms = **81%**。
反向占了总流量的 80%（274.8 / 344.6 GB），所以整体效率被拉向反向一侧；这也是为什么 4.1.1 里
「前向只有 72%」没有把整体拖到 70% 附近。

量级别搞混：**单次 AttnRes 是 0.1–1 ms**（L=2 约 0.39 ms、L=9 约 1.0 ms），
上表的 34–502 ms 是 186 次调用在 K3 里**串行累加**的结果（每一次都要等上一层 sublayer 的输出）。

放到 K3 的每迭代账单里（T=4096，单卡口径）：

| 组成 | 每迭代时间 | 来源 |
|---|--:|---|
| KDA × 69 层（fwd+bwd） | 527 ms | 本目录 `results_kda_kimi_k3.md`（1.586+6.058 ms/层） |
| MLA × 24 层（fwd+bwd） | 97 ms | 本目录 `results_mla_kimi_k3.md`（0.878+3.177 ms/层） |
| **AttnRes × 186 次** | **130 ms** | 本文 |
| 普通 PreNorm 残差（参照） | ~9.8 ms（roofline） | 186 × 3 × N·D·2 B = 32.8 GB |

也就是说 AttnRes 在注意力主干之上再增加 **~21%**，相对普通残差则是 **~10.5×** 的流量。
block size=12 是 K3 在「残差表达力」和「每层多看几个 source」之间的折中：block 越小 L 越大，
流量近似线性涨。

## 5. FLA 里的优化手段（以 AttnRes 为线索）

### 5.1 一次流式、能融尽融

`fla/ops/attnres/fused.py` 的前向是「一个 program 处理一个 token」的 kernel：循环 L 个 source，
每个 source 只读一次，在线 softmax（running max / sum + 重标定累加器），最后一次性写出混合结果。
它把四件事融进同一个 kernel：

- 每个 source 的 **RMSNorm**（只算 `sum(v²)` 与 `sum(v·q·w)` 两个归约，不落盘）；
- **打分 + softmax**（log-space，在线）；
- **加权求和**；
- 可选的 **输出 RMSNorm**（即下一个 sublayer 的 prenorm，见 §5.4）。

对比 naive 实现要 `stack → fp32 → rms_norm → softmax → einsum` 五步、每步一次全量读写，
这就是 §4.3 里 7.7–10.8× 差距的来源。

### 5.2 不 materialize 输入：pointer table + L2 分桶

`_build_ptr_table()`（fused.py:303）把 `list[Tensor]` 补齐到 `max(8, next_pow2(L))` 个槽位，
只构造一张**地址表**；kernel 内用 `static_range` + `tl.where` 从表里选 base pointer。
好处：

- 宿主侧**不 cat / 不 stack**（否则 L=9、T=4096 要先写 528 MB bf16 / 1.05 GB fp32）；
- kernel 的 tuple 长度只有 8/16 两档（`L2`），配合 `@triton.jit(do_not_specialize=['L'])`
  （fused.py:37/134），**L=2…8 共用一个编译产物**，L=9…16 共用一个，避免每个 L 编译一次；
- 要求每个 source 16B 对齐（`assert t.data_ptr() % 16 == 0`），方便向量化。

代价是 L=9 会跨到 16 槽的桶，但槽位只用于指针选择，实测开销在噪声内（§4.4）。

### 5.3 反向：两段式、无 atomic、按需重算

反向拆成两个 kernel：

1. `attnres_bwd_kernel_dv`：仍是一个 token 一个 program，重算/读取 `o_pre` 与 `rstd/logit`，
   对每个 source 算 `dV`，同时把 `dq·w` 的逐 token 贡献写进 **fp32 `dqw` scratch `[N,D]`**；
2. `attnres_bwd_kernel_dqdw`：对 N 维做列归约，得到 `dq`、`dw`（以及融合 norm 时的 `dow`）。

这样**完全不用 atomic**（`dq/dw` 只有 `[D]`，用第二段归约而不是 `atomic_add`），代价是
`dqw`/`dow_partial` 两个 `[N,D]` fp32 缓冲的写+读 —— 在 L=9 时约占反向流量的 **22%**，
在 L=2 时约 **53%**（这也是 §4.1 里 bwd/fwd 从 3.3 涨到 4.2 的原因）。

另外两个省流量的点：

- **不保存 softmax 概率**：只存 `logit` 与 `lse`（`[L,N]`、`[N]` fp32），反向用
  `exp(logit*scale - lse)` 重算，省掉 `[L,N,D]` 级别的中间量；
- **`checkpoint_level`**：0 = 存 `o_pre`（前向多写 1 份、反向少读 L 份），
  1 = 不存、反向重算（默认）。实测两者只差 1.2%（§4.4）。

### 5.4 输出 RMSNorm 融合

AttnRes 的输出紧接着就是下一个 sublayer 的 prenorm。FLA 的 `output_rms_weight`
让 kernel 直接输出**归一化后的结果**，把这次 `[N,D]` 读+写彻底省掉。Megatron 的
`attn_res_norm_fusion` 就是走这条路（`AttentionResidual._aggregate` 里
`output_rms_weight=output_norm_weight`）。实测每次调用省 **45 µs**（bf16 norm 权重，
2.6 TB/s）到 **474 µs**（torch 在 fp32 权重下退化成非融合路径），186 次调用即
**8.4–88 ms/迭代** —— 是本文里性价比最高的一个开关。

### 5.5 autotune 与「按 L 分档」

`fla_cache_autotune`（`fla/ops/utils/cache.py:415`）在 Triton autotune 外包了一层：

- 可以按 GPU 型号从 `fla/configs/<GPU>/` 读预调好的 JSON（`FLA_CACHE_MODE=STRICT/FUZZY/FULL`），
  没有就现场 autotune；
- AttnRes 前向扫 `BL ∈ {1,2,4,8}`（每次处理几个 source）× `num_warps` × `num_stages`，共 24 个候选；
  K3 的 D=7168、T=4096、L=9 上实测选中 **`BL=2, num_warps=4, num_stages=3`**（从
  `attnres_fwd_kernel.cache` 读出），所以时间呈 `ceil(L/2)` 台阶；
- Gluon 版按 smem 预算裁剪候选（`_prune_fwd_smem` / `_prune_bwd_smem`），L>8 时只留 `BT=2`
  —— 这正是 §4.3 里 L=9 崩掉的来源。

注意它的 key **不含 N**，同一进程里先跑的 shape 决定后续 shape 的 config。生产上要么用
预置 config（`FLA_CACHE_MODE`），要么接受「第一个 shape 调好、后面复用」的近似。

### 5.6 Gluon 后端：为静态调度而写

`fla/ops/attnres/backends/gluon.py` 是同一算子的另一套实现，思路和 Triton 完全不同：

- **L 是 constexpr**，循环完全展开，**不需要 pointer table**；
- 每个 program 处理 `BT` 个 token（Triton 是 1 个），提高访存合并、摊薄权重加载；
- 用 **`cp.async` 双缓冲**把 source tile 搬进 shared memory，`evict_first` 提示流式读；
- 所有 L 个 tile 能放进 smem 就**常驻**（`_resident_tiles`），否则退化成 2 深 ring；
- 显式指定 `BlockedLayout`/`SliceLayout`/`SwizzledSharedLayout`、warp/lane 划分；
- 一次跨 warp 归约同时算出 `sum(v²)` 和 `sum(v·q·w)`（自定义 `combine_fn`）；
- 反向按 **32 个 token 一组**聚合 `dqw`/`dow_partial`（`GROUP=32`），把 fp32 scratch 缩小 32×
  —— 这也是它峰值显存低得多的原因。

在 H100 上它是「L≤8 更快、L=9 崩」，在 Blackwell 上才是它的主场（smem/TMA 更大方）。

### 5.7 FLA 更通用的几类手段

- **多后端 dispatch**（`fla/backends.py:163`）：Triton 参考实现 + TileLang / FlashKDA(CUTLASS) /
  Gluon / Ascend，注册时带 `verifier`、`priority`、环境变量开关；不可用就**静默回落到下一个**，
  且 dispatch 外层套 `torch.compiler.disable`，不破坏 `torch.compile`。
- **算法分块**：线性注意力（KDA/GLA/DeltaNet…）统一走 chunk 形式（chunk 内并行、chunk 间传 state），
  另有 `fused_recurrent` 的逐 token 版本；同一个算子在不同 T 下选不同 kernel。
- **精度策略**：bf16 读入、fp32 累加/统计，最后只 downcast 一次；softmax 在 log 空间；
  `FLA_USE_FAST_OPS=1` 时切 `fast_expf/fast_logf`（`fla/ops/utils/op.py`）。
- **内存策略**：能重算就不存（`checkpoint_level`、各算子的 `disable_recompute`）、
  用 scratch + 第二段归约代替 atomic、避免 host 侧 cat/stack。
- **工程集成**：`input_guard` 统一做 contiguity/dtype 检查（`fla/utils/decorators.py:97`）、
  `autocast_custom_fwd/bwd` 接 `torch.amp`（`fla/utils/hardware.py:189`）、
  自定义 `autograd.Function` 里 `mark_non_differentiable` 掉不需要梯度的概率输出。

### 5.8 K3 上还能怎么优化

1. **D=7168 不是 2 的幂**：kernel 用 `BD=next_pow2(D)=8192`，12.5% 的 lane 被 mask 掉，
   `b_o` 累加器也要 8192 个 fp32。这是 fwd 只能到 2.5 TB/s（而非 3.0+）的主因之一。
   若为 7168 做专门 tile（如 3584×2 或 7168 直接铺）可回收一部分。
2. **Triton 反向的 fp32 scratch**：`dqw`（+`dow_partial`）在 L 小的时候是主要流量。
   可以学 Gluon 做 token 分组聚合（32 token 一组），把 `[N,D]` 降到 `[N/32,D]`；
   K3 有 24 次 L=2、24 次 L=3 的调用，收益集中在这里。
3. **L=9 的 Gluon 路径**：放开 `BT=1` 或允许 `BD` 分块，避免 `BT=2` 的寄存器爆炸；
   否则 K3 的 18 次 L=9 调用会把 Gluon 的整体收益吃掉。
4. **block size 与流量的权衡**：block=12 层（24 sublayer）给出平均 L≈5.4；若改大到 24 层，
   L 减半、流量近似减半，但表达力下降 —— 这是模型侧的取舍，kernel 侧只能把常数做小。
5. **host 侧**：186 次调用/迭代，每次都要 `reshape/contiguous` L 个张量、建地址表、走 autograd
   Function 和 dispatch。wall 与 CUPTI 的差（T=4096/L=9：fwd 370 vs 237 µs）里有一部分是这笔
   host 税；训练时能与计算重叠，但小 T 下不可忽略。

## 6. 复跑

```bash
# 在 kernel_lab 容器内，选一张空闲卡
export FLA_SRC=~/proj/flash-linear-attention
export LD_LIBRARY_PATH=~/proj/Megatron-Bridge/.venv/lib/python3.12/site-packages/z3/lib:$LD_LIBRARY_PATH
cd ~/proj/tech_record/code/te-perf

CUDA_VISIBLE_DEVICES=7 python bench_attnres_kimi_k3.py --mode check
CUDA_VISIBLE_DEVICES=7 python bench_attnres_kimi_k3.py --mode sweep --csv attnres_kimi_k3.csv
CUDA_VISIBLE_DEVICES=7 python bench_attnres_kimi_k3.py --mode schedule --seqs 1024,4096,8192,16384
CUDA_VISIBLE_DEVICES=7 python bench_attnres_kimi_k3.py --mode extras --seqs 4096 --Ls 9
```

`--backends triton,gluon,naive` 可只跑其中几个；`--no-onorm` 测不融合输出 norm 的路径；
`--Ls 2,3,5,8,9 --seqs 1024,4096,8192,16384` 是本文默认网格。
