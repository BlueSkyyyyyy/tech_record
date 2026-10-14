# KDA 精度比对：FlashKDA vs FLA Triton（vs fp64 golden）

> 在 **Kimi K3 的 KDA shape**（B=1, H=96, K=V=128；safe gate, `lower_bound=-5`；kernel 内 gate/l2norm/
> beta sigmoid）上，把**同一份 bf16 输入**分别喂给 **FLA `chunk_kda`（Triton）** 与 **FlashKDA**，
> 和 **float64 逐 token 递推 golden** 比较，隔离「kernel 算术误差」与「bf16 输入量化误差（两者共有）」。
> 数据：[`kda_precision.log`](kda_precision.log)；脚本 `check_kda_precision.py`（可复跑）。
>
> 口径：golden = 在**同一份 bf16 输入**上跑 fp64 逐 token gated delta rule（含 safe gate）。
> `refrel = max_abs / max|golden|`；`relf = max 逐元素相对误差（限制在 |golden|>1% 峰值）`；
> `rms = 逐元素 RMS`。

## 结果（T × seed，批 1 头 96）

| T | seed | Triton bf16 max_abs | refrel | rms | FlashKDA bf16 max_abs | refrel | rms | Triton fp32 max_abs | refrel | FK−Tri max_abs |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 0 | 6.74e-4 | 0.88% | 2.1e-5 | 7.32e-4 | 0.95% | 3.0e-5 | 1.65e-4 | 0.21% | 7.32e-4 |
| 1024 | 0 | 5.94e-4 | 0.64% | 2.1e-5 | 8.54e-4 | 0.91% | 3.1e-5 | 1.71e-4 | 0.18% | 9.77e-4 |
| 4096 | 1 | 7.08e-4 | 0.64% | 2.2e-5 | 8.35e-4 | 0.76% | 3.1e-5 | 2.24e-4 | 0.20% | 9.77e-4 |
| 8192 | 1 | 8.43e-4 | 0.87% | 2.1e-5 | 1.04e-3 | 1.10% | 3.0e-5 | 2.26e-4 | 0.23% | 1.22e-3 |
| 16384 | 0 | 7.67e-4 | 0.80% | 2.1e-5 | 9.10e-4 | 0.96% | 2.9e-5 | 2.37e-4 | 0.25% | 1.22e-3 |
| 32768 | 1 | 8.40e-4 | 0.83% | 2.0e-5 | 9.20e-4 | 0.91% | 2.9e-5 | 2.44e-4 | 0.24% | 1.46e-3 |

（完整 12 行见日志；两 seed 结论一致。）

## 结论

1. **两者都是 bf16 底噪**：对 fp64 golden 的峰值误差 `refrel` 都在 **0.6%–1.1%**，`rms ≈ 2–3e-5`。
2. **FlashKDA 略逊于 Triton（bf16）**：`max_abs` 约为 Triton 的 **1.1–1.4×**、`rms` 约 **1.4×**
   （例：8192/seed1 1.04e-3 vs 8.43e-4）。同阶，不是量级差。
3. **两者互差 ≈ 各自 vs golden 的误差**（FK−Tri `max_abs ≈ 0.7–1.5e-3`），说明**没有共模相消/放大**，
   差异主要来自各自的归约/分块次序（FlashKDA CHUNK=16 vs Triton chunk=64）与 bf16 累加。
4. **Triton fp32 约准 3–4×**（`refrel ≈ 0.2%`），符合 tf32 `tl.dot` 的底噪；也能作为「参考实现的
   kernel 算术地板」。
5. **误差不随 T 累积**（`rms` 在 T=512…32768 恒为 ~2e-5），说明状态递推的舍入没有漂移。

## 备注

- FlashKDA 无反向，故本比对只覆盖前向 `o`。
- golden 用 bf16 输入 → fp64，**不含输入量化**；若把输入量化算进去，两者误差都会再抬到 ~1e-2 量级
  （与 `results_kda_kimi_k3.md` §3 的「FK vs Tri ≈ 1.2%」互洽）。

## 复跑

```bash
# kernel_lab、空闲 GPU
cd code/te-perf
python check_kda_precision.py --seqs 512,1024,4096,8192,16384,32768 --seeds 0,1
```
