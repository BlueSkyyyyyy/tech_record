# Kimi K3 KDA 算子性能实测（FLA Triton / TileLang vs FlashKDA，含数据类型支持）

> 面向 Kimi K3 的 **KDA（Kimi Delta Attention）** 层，在 H100 上实测 **前向 / 反向单独统计**，
> 并给出各 kernel 的**数据类型支持矩阵（含 fp8）**。
> 数据：[`kda_kimi_k3.csv`](kda_kimi_k3.csv)（bf16）、[`kda_dtypes.csv`](kda_dtypes.csv)（fp16/fp32）、
> [`kda_dtype_support.log`](kda_dtype_support.log)、[`kda_kimi_k3.log`](kda_kimi_k3.log)、
> [`kda_dtypes.log`](kda_dtypes.log)；脚本 `bench_kda_kimi_k3.py`（可复跑）。
>
> TE 无 KDA 算子，故放此目录单独测；`bench_te.py` 仍是 TE 算子口径。

## 1. Kimi K3 的 shape

官方 `text_config`：93 层 = **69 KDA + 24 full-MLA**；KDA `num_heads=96, head_dim(K=V)=128,
gate_lower_bound=-5.0`；MLA 96 头 / qk=192 / v=128。每个 KDA 层是 `[B,T,96,128]`（`beta`：`[B,T,96]`）。
本测 `B=1`。

## 2. 计时口径（反向是独立测的）

- 容器 `kernel_lab`（torch 2.12、**FLA 0.5.2**、FlashKDA、TE 2.14）；H100 80GB；**空闲 GPU**（util 0%）。
- `device_ms` = **CUPTI 纯 device kernel 时间**（`torch.profiler`，不含 launch）；warmup=5、repeat=20。
- **前向 `fwd`**：`no_grad` 下直接计时。
- **反向 `bwd`（独立）**：先**在计时区外**用 `requires_grad` 建一次图，随后计时区**只跑**
  `torch.autograd.grad(o, inputs, do, retain_graph=True)`——计时区内执行的全部是反向 kernel
  （含 FLA 在反向里做的前向重算），因此是真正的「反向单独」。
- **`f+b`**：新建图 + 前向 + 反向整体，仅作交叉参照。
- **为什么不用「减法」**：`f+b − fwd` 会**高估**反向 ~6–10%——因为 `f+b` 里的前向带 grad、
  要建图并保存激活，比 `no_grad` 前向更贵。实测同一空闲卡上 `bwd_iso / bwd_sub ≈ 0.90–0.94`，
  且**GPU 有干扰时减法与独立测都会飘**，所以必须用「空闲卡 + 独立计时」。
- 配置：kernel 内 **gate + q/k L2norm + beta sigmoid**，`safe_gate=True`、`lower_bound=-5.0`、
  `state_v_first=True`。`A_log∈[H]`、`dt_bias∈[H*K]`（fp32）。
- 后端选择：切 `FLA_TILELANG`/`FLA_FLASH_KDA`；**被拒后端静默回落 Triton**，同时给 `requested`/`effective`。

## 3. 数据类型支持矩阵（T=2048）

`cell = <effective>/<runtime>`。

| dtype | 请求 | 前向 | 反向 |
|---|---|---|---|
| bf16 | triton | triton/ok | triton/ok |
| bf16 | tilelang | tilelang/ok | tilelang/ok |
| bf16 | **flashkda** | **flash_kda/ok** | triton/ok（回落，无反向） |
| fp16 | triton | triton/ok | triton/ok |
| fp16 | tilelang | tilelang/ok | tilelang/ok |
| fp16 | flashkda | triton/ok（回落） | triton/ok（回落） |
| fp32 | triton | triton/ok | triton/ok |
| fp32 | tilelang | tilelang/ok | **tilelang/FAIL（编译错误）** |
| fp32 | flashkda | triton/ok（回落） | triton/ok（回落） |
| **fp8 e4m3 / e5m2** | triton / tilelang / flashkda | **全部 FAIL** | **全部 FAIL** |

- Triton（fallback）：`bf16/fp16/fp32` 前向+反向全支持。
- TileLang：`bf16/fp16` 全支持；`fp32` 仅前向（反向编译失败）。
- FlashKDA：**仅 bf16 推理前向**；其余静默回落 Triton。
- **fp8（e4m3/e5m2）三后端全不支持**：卡在 FLA `fused_beta_sigmoid_fwd_kernel`
  （`tl.load(..., other=0)` 报 `cannot cast int32 to fp8e4nv`）。

## 4. bf16 前向 / 反向（B=1, H=96, K=V=128，CUPTI device ms，反向独立测）

| T | Triton fwd | **Triton bwd（独立）** | Triton f+b | FlashKDA fwd | fwd 加速 | bwd/fwd |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 0.438 | 1.554 | 2.157 | 0.209 | **2.10×** | 3.55 |
| 4096 | 1.586 | 6.058 | 8.230 | 0.774 | **2.05×** | 3.82 |
| 8192 | 3.139 | 12.069 | 16.347 | 1.529 | **2.05×** | 3.85 |
| 16384 | 6.276 | 24.230 | 32.733 | 3.034 | **2.07×** | 3.86 |
| 32768 | 12.598 | 48.347 | 65.411 | 6.044 | **2.08×** | 3.84 |
| 65536 | 25.248 | —（>32768 未测） | — | 12.095 | **2.09×** | — |

- **前向：FlashKDA 稳定快 ~2.05–2.10×**（纯 device，非 host 税），随 T 几乎不变。
- **反向（独立）：约为前向的 3.5–3.9×**（Triton bf16）。
- Triton 前向按 token 线性 ≈ **0.39 µs/token**（`B·H=96 < 132 SM` → 每 CTA 串行延迟受限）。
- **FlashKDA 无反向**（grad-enabled 即回落 Triton，故 bwd 记 n/a）。

## 5. fp16 / fp32 计时（Triton vs TileLang，CUPTI device ms，反向独立测）

| dtype | T | 后端 | fwd | **bwd（独立）** | f+b |
|---|--:|---|--:|--:|--:|
| fp16 | 1024 | triton | 0.440 | 1.556 | 2.158 |
| fp16 | 1024 | tilelang | 0.440 | **1.330** | **1.932** |
| fp16 | 4096 | triton | 1.594 | 6.063 | 8.238 |
| fp16 | 4096 | tilelang | 1.594 | **5.057** | **7.235** |
| fp16 | 8192 | triton | 3.149 | 12.068 | 16.357 |
| fp16 | 8192 | tilelang | 3.148 | **10.019** | **14.302** |
| fp32 | 1024 | triton | 0.728 | 2.875 | 3.805 |
| fp32 | 4096 | triton | 2.695 | 11.273 | 14.706 |
| fp32 | 8192 | triton | 5.343 | 22.582 | 29.369 |
| fp32 | 1024/4096/8192 | tilelang | ≈triton | —（不支持） | — |

- **fp16 前向 ≈ bf16**；**TileLang 反向比 Triton 反向快 ~1.15–1.2×**。
- **fp32 更慢**：Triton 前向 ~1.65–1.70× bf16、反向 ~1.85× bf16；**fp32 反向只有 Triton**。
- fp8 不可用（见 §3）。

## 6. 总结

1. **前向**：bf16 下 **FlashKDA ≈ 2.05–2.10× FLA Triton**；fp16≈bf16；fp32 慢 ~1.7×；**fp8 不支持**。
2. **反向（独立测）≈ 前向的 3.5–3.9×**（Triton bf16），只能走 FLA；
   **TileLang 反向比 Triton 快 ~1.2×**（bf16/fp16），但 **fp32 反向不支持**。
3. **数据类型**：Triton `bf16/fp16/fp32`；TileLang `bf16/fp16`（fp32 仅前向）；FlashKDA `bf16`（仅前向）；fp8 全不支持。
4. 训练 KDA：**TileLang（bf16/fp16）> Triton**；推理前向用 **FlashKDA（bf16）**。
5. TE 与 KDA 无关。正确性 FlashKDA vs Triton `o` max_rel ≈ 1.2e-2（bf16 底噪）。

## 7. 复跑

```bash
# kernel_lab 容器、确认 GPU 空闲（util=0%）后
cd code/te-perf
python bench_kda_kimi_k3.py --mode support --dtypes bfloat16,float16,float32,float8_e4m3fn,float8_e5m2
python bench_kda_kimi_k3.py --mode sweep --dtypes bfloat16 \
    --seqs 1024,4096,8192,16384,32768,65536 --backends triton,flashkda --csv kda_kimi_k3.csv
python bench_kda_kimi_k3.py --mode sweep --dtypes float16,float32 \
    --seqs 1024,4096,8192 --backends triton,tilelang --csv kda_dtypes.csv
```
