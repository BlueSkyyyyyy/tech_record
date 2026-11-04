# Kimi-K3 MLA（TE fused attention 前向 + 反向）H100 实测

> 结论先说：**TE 能测 Kimi-K3 的 MLA，前向和反向都可以**。核心注意力 shape 为
> `(b, s, 96, 192, 128)`（96 头，qk = nope128 + rope64 = 192，v = 128），TE 走
> `F16_arbitrary_seqlen` 后端，训练态 `is_training=True` 的 fwd / bwd 都返回可用 backend
> 并跑通；与 torch fp32 参考实现对比，fwd / dq / dk / dv 的 cos 均 ≈ **0.999997**。
>
> 数据：[`mla_kimi_k3.csv`](mla_kimi_k3.csv)（bf16 + fp16）、[`mla_kimi_k3.log`](mla_kimi_k3.log)（完整 stdout）；
> 脚本：[`bench_mla_kimi_k3.py`](bench_mla_kimi_k3.py)（复用 `bench_te.py` 的 CUPTI 计时口径，可复跑）。

## 1. Kimi-K3 的 MLA 到底是什么 shape

从 `~/models/Kimi-K3/config.json` + `modeling_kimi_linear.py::KimiMLAAttention`：

| 项 | 值 | 说明 |
|---|---|---|
| `num_attention_heads` | **96** | 每个 MLA 层 96 个 query 头 |
| `qk_nope_head_dim` | 128 | 内容部分 |
| `qk_rope_head_dim` | 64 | **NoPE**：K3 的 `rotary_emb = None`，这 64 维只保留、不旋转 |
| qk head dim | **192** | 128 + 64 |
| `v_head_dim` | **128** | 值维度 |
| `kv_lora_rank` / `q_lora_rank` | 512 / 1536 | 压缩 latent（属于投影 GEMM，不在 fused-attn 内） |
| MLA 层数 | **24** / 93 | 其余 69 层是 KDA（`full_attn_layers`） |

训练时（`KimiMLAAttention.forward`）`kv_b_proj` 把 latent 上投影成**每头**的
`k_nope(128) + v(128)`，`q_b_proj` 产生每头 `q(192)`，所以进入注意力 kernel 的就是
**每头 q/k = 192、v = 128、96 头** 的普通 causal softmax attention：

```
(batch, seqlen, num_heads, qk_head_dim, v_head_dim) = (b, s, 96, 192, 128)
```

这正是 `bench_te.py` 里 5 元组 MLA 形式（Kimi-K2.6 是 64 头版本），唯一变化是头数 64 → 96。
**NoPE 对 benchmark 没有影响**：TE fused_attn 本来就不做 rope，K3 的 64 维 rope 槽位是原样带过的。
softmax scale 也一致：K3 用 `192 ** -0.5`，TE 默认 `1/sqrt(q.size(-1))`。

## 2. TE 支持性（backend 探测 + 实跑）

用 `tex.get_fused_attn_backend` 探测 + 真实执行验证（H100 sm90、TE 2.14.0、torch 2.12）：

| 场景 | qk/v | 结果 |
|---|---|---|
| 训练 fwd+bwd，bf16 / fp16，s = 256…16384 | 192 / 128，96 头 | ✅ `F16_arbitrary_seqlen`，实跑通过 |
| 推理 fwd（`is_training=False`） | 192 / 128，96 头 | ✅ `F16_arbitrary_seqlen` |
| FP8 推理 fwd | 192 / 128，96 头 | ✅ `FP8` backend |
| **FP8 训练 fwd+bwd** | 192 / 128，96 头 | ❌ `No_Backend`（本栈 FP8 训练只支持 head_dim ≤ 128） |
| FP32 | — | ❌ TE fused_attn 不提供 FP32 路径 |

关键点：**96 头本身不构成限制**，限制仍然来自 head_dim（bwd ≤ 256）；`qk=192 / v=128`
恰好在支持范围内，和 K2.6 的 64 头结论一致。另外 TE 原生支持 `qk != v`，**不需要把 v 从
128 padding 到 192**——而 HF 的 FA2 路径（`KimiMLAAttention` 里 `F.pad(value_states, ...)`）
必须补齐到 192，白白多做 20% 的 PV FLOPs（PV 从 128 变 192）。

## 3. 计时口径

- 容器 `kimi26_train`；**空闲卡 GPU 0**（`CUDA_VISIBLE_DEVICES=0`，跑前/跑后均 0 MiB、0% util，
  其余 7 张卡也全空闲）；H100 80GB SXM，132 SM，SM clock 1980 MHz。
- `time(us)` = **CUPTI 纯 device kernel 时间**（`torch.profiler`，warmup=10、repeat=50 取均值），
  **不含 launch / 主机派发**；与 `bench_te.py` 完全同一套 `KernelTimer.device_time`。
- 训练态：`is_training=True`，`causal`、`no_bias`、`dropout=0`、`qkv_layout=bshd_bshd_bshd`。
- **反向独立计时**：bwd 只计 `fused_attn_bwd` 本身（吃 fwd 存下的 `aux_ctx=[M, rng_state]`，
  在 kernel 内重算 S），**不含**前向；因此「单层训练一步」≈ fwd + bwd 两行相加。
- FLOPs 沿用本目录口径：fwd `2·b·s·h·s·(qk+v)`，bwd `2× fwd`。这是**完整 S²** 口径，
  而 causal kernel 实际只算约一半 tile，所以表里的 `TFLOPS/%TC(S²)` 是 **S² 等效值**，
  真实 tensor-core 利用率约为其 **0.5 倍**（见 §4 注）。

## 4. 实测性能（bf16 主表，fp16 见附表）

### 4.1 前向（training=True）

| b | s | h | qk/v | time(us) | TFLOPS(S²) | %TC(S²) | GB/s | AI |
|--:|--:|--:|--|--:|--:|--:|--:|--:|
| 1 | 1024 | 96 | 192/128 | 86.5 | 744.7 | 75.3% | 1454.5 | 512 |
| 1 | 2048 | 96 | 192/128 | 252.4 | 1021.1 | 103.2% | 997.2 | 1024 |
| 1 | 4096 | 96 | 192/128 | **878.0** | **1174.1** | **118.7%** | 573.3 | 2048 |
| 1 | 8192 | 96 | 192/128 | 3490.0 | 1181.4 | 119.4% | 288.4 | 4096 |
| 1 | 16384 | 96 | 192/128 | 14181.2 | 1163.0 | 117.5% | 142.0 | 8192 |
| 2 | 4096 | 96 | 192/128 | 1862.7 | 1106.7 | 111.9% | 540.4 | 2048 |

### 4.2 反向（training=True，独立计时）

| b | s | h | qk/v | time(us) | TFLOPS(S²) | %TC(S²) | GB/s | AI | bwd/fwd |
|--:|--:|--:|--|--:|--:|--:|--:|--:|--:|
| 1 | 1024 | 96 | 192/128 | 364.0 | 354.0 | 35.8% | 691.4 | 512 | 4.21× |
| 1 | 2048 | 96 | 192/128 | 945.6 | 545.0 | 55.1% | 532.3 | 1024 | 3.75× |
| 1 | 4096 | 96 | 192/128 | **3177.2** | **648.9** | **65.6%** | 316.8 | 2048 | 3.62× |
| 1 | 8192 | 96 | 192/128 | 11520.7 | 715.8 | 72.3% | 174.8 | 4096 | 3.30× |
| 1 | 16384 | 96 | 192/128 | 45043.4 | 732.3 | 74.0% | 89.4 | 8192 | 3.18× |
| 2 | 4096 | 96 | 192/128 | 6482.9 | 636.0 | 64.3% | 310.6 | 2048 | 3.48× |

> 注：`TFLOPS(S²)` 的分母是 kernel 时间，分子是**完整 S²** 的 FLOPs。causal kernel 跳过
> 约一半上三角 tile，所以真实算力利用率 ≈ 表值 × 0.5：s=4096 时 fwd ≈ **59%**、bwd ≈ **33%**
> 的 H100 BF16 dense peak（989.4 TFLOPS）。表值 >100% 并不违反 roofline，正是「没算的那一半」
> 被计进了分子，和本目录既有的 K2.6 行（115.9%）同一口径。

### 4.3 与 Kimi-K2.6（64 头）的对照（同目录既有数据，b=1, s=4096, bf16）

| 模型 | 头数 | fwd(us) | fwd TFLOPS(S²) | bwd(us) | bwd TFLOPS(S²) |
|---|--:|--:|--:|--:|--:|
| Kimi-K2.6 MLA | 64 | 599.3 | 1146.8 | 2161.7 | 635.8 |
| **Kimi-K3 MLA** | **96** | **878.0** | **1174.1** | **3177.2** | **648.9** |
| 比值 | 1.5× | 1.46× | 1.02× | 1.47× | 1.02× |

头数从 64 涨到 96，时间基本**线性**（1.46× / 1.47×），单位 FLOP 吞吐持平（±2%），说明
TE 在 192/128 这个 MLA 几何上没有随头数出现新的效率拐点。

### 4.4 折算到 K3 单层 / 整模型的 24 个 MLA 层（bf16, b=1, s=4096）

| 阶段 | 每层(ms) | ×24 层(ms) |
|---|--:|--:|
| fwd | 0.878 | 21.07 |
| bwd | 3.177 | 76.25 |
| **fwd+bwd** | **4.055** | **97.32** |

即 s=4096、b=1 时，K3 全部 MLA 层的核心注意力约 **97 ms/step**（前向 21 ms + 反向 76 ms）。
注意这只是 **QKᵀ→softmax→PV 核心 kernel**；q_a/q_b、kv_a/kv_b、o_proj、输出 gate 都是另外的
GEMM，KDA 层也不在此列，所以不能当成整模型 step time。

### 4.5 fp16 附表（趋势与 bf16 一致，普遍慢 1–4%）

| b | s | fwd(us) | fwd TFLOPS(S²) | bwd(us) | bwd TFLOPS(S²) |
|--:|--:|--:|--:|--:|--:|
| 1 | 1024 | 87.0 | 740.5 | 355.8 | 362.1 |
| 1 | 2048 | 256.4 | 1005.0 | 957.7 | 538.2 |
| 1 | 4096 | 896.7 | 1149.6 | 3291.2 | 626.4 |
| 1 | 8192 | 3713.9 | 1110.2 | 12205.4 | 675.6 |
| 1 | 16384 | 14560.6 | 1132.7 | 46287.0 | 712.6 |
| 2 | 4096 | 1895.5 | 1087.6 | 6544.3 | 630.0 |

## 5. 怎么读这些数

1. **fwd 在 s≥2048 就饱和**：1021 → 1181 S²-TFLOPS（s=1k 只有 745，固定开销/尾块占比高），
   之后到 16k 基本持平（±1.5%）。真实利用率 ~52–60% TC。
2. **bwd 是短板**：绝对时间是 fwd 的 **3.2–4.2×**（理想 2×），S² 等效吞吐只有 fwd 的
   48–63%，s=4096 时真实利用率 ~33% vs fwd ~59%。这是 attention 反向的普遍规律：要重算 S、
   再做一遍 QKᵀ、累加 dq/dk/dv 并多读写一轮梯度；TE 在 Hopper 上默认用自家 fused bwd
   （`NVTE_FUSED_ATTN_USE_FAv2_BWD=0`），其优化程度不如前向 kernel。
   所以**优化 K3 MLA 训练，反向的收益空间明显大于前向**。
3. **不是带宽瓶颈**：s=4096 时 fwd 573 GB/s、bwd 317 GB/s，都远低于 3.35 TB/s；AI = 2048
   FLOP/byte，远在 roofline 拐点（295）右侧，是纯 compute-bound。
4. **小 seq 的固定开销不可忽略**：s=1024 时 fwd 只有 86.5 µs，bwd/fwd 比冲到 4.2×；
   短序列训练应优先看 kernel launch / 尾块效率，而不是峰值算力。
5. **头数线性扩展**：96 头 vs 64 头时间 1.46–1.47×，吞吐不变，说明 K3 把头数加到 96
   不会让 TE fused-attn 掉效率；成本就是线性增加。

## 6. 口径与 caveat（写博客/汇报时请带上）

- **只测核心注意力**：`QKᵀ → causal softmax → PV`。K3 的 MLA 投影（q_a/q_b、kv_a/kv_b、
  o_proj）和 full-rank 输出 gate 是独立 GEMM，不含在内。
- **S² 等效 FLOPs**：causal 实际只算约一半，表里的 TFLOPS/%TC 是「按完整 S² 折算」的
  等效值，真实 TC 利用率约 ×0.5；与本目录其他 attention 行口径一致，横向可比。
- **FP32 输出**：K3 训练时注意力输出**保持 FP32**（论文里为修 biased rounding），而 TE
  fused_attn 的输出 dtype 跟随输入（bf16/fp16）；kernel 内部 softmax/PV 仍是 FP32 累加，
  但 API 不暴露 FP32 输出张量。要做 bit-level 对齐需自行在 kernel 外处理。
- **NoPE**：K3 的 MLA 不做 rope，64 维 rope 槽位原样参与 QKᵀ；TE 不施加 rope，二者数学一致。
- **FP8 训练不可用**：本栈 `qk=192/v=128` 的 FP8 训练返回 `No_Backend`（只有 128/128 可行）；
  FP8 **推理** fwd 可用。K3 训练本身也是 bf16，故主表只测 bf16/fp16。
- **测量抖动**：同一空闲卡上重复跑，主要数据点波动 <2%；长时间连续跑（16k bwd 之后）
  时钟/功耗会让后续小 shape 慢 5–10%，所以本文所有数字都取自同一轮、按 seq 升序的表格，
  折算表直接复用表中 s=4096 的行，不额外重测。

## 7. 复现

```bash
# 进容器（TE 2.14 / torch 2.12 / H100）
docker exec -it kimi26_train bash
cd ~/proj/tech_record/code/te-perf

# 1) 正确性：TE fwd+bwd vs torch fp32 参考（s=512，cos 应 >0.999）
CUDA_VISIBLE_DEVICES=0 python bench_mla_kimi_k3.py --check

# 2) 完整性能（本文件的数）
CUDA_VISIBLE_DEVICES=0 python bench_mla_kimi_k3.py --warmup 10 --repeat 50 \
    --csv mla_kimi_k3.csv | tee mla_kimi_k3.log

# 3) 只跑某几个 seq / 只跑 bf16
CUDA_VISIBLE_DEVICES=0 python bench_mla_kimi_k3.py --seqs 1024,4096,16384 --dtypes bf16
```

`--seqs` 默认 `1024,2048,4096,8192,16384`，`--extra-batches` 默认 `2x4096`；脚本会打印
`CUDA_VISIBLE_DEVICES`，跑前请确认目标卡 `nvidia-smi` 为 0 MiB / 0%（本次 8 张卡全空闲，用 GPU 0）。
