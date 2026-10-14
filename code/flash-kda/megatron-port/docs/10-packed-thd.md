# 10 · Megatron 真层 packed / THD（varlen）端到端对拍（M10-2）

> 系列第 11 篇。M10-1 首次在容器里跑通真实 `KimiDeltaAttention` 模块，并把它的核心算子
> 锚定到 fp64 参考——但那条路径只覆盖 **SBHD 非 packed**（`packed_seq_params=None`，每个
> batch 行一条序列）。本篇（M10-2）把同样的两个粒度（核心算子 / 整模块）扩到 **packed THD**
> 路径：多序列沿 token 维拍平、用 `cu_seqlens` 切分、`batch==1`。
>
> 脚本：`harness/check_port_e2e_packed.py`；原始输出：`harness/check_port_e2e_packed.out.txt`。
> 行号按本机版核对：`Megatron-LM/megatron/core/ssm/gated_delta_net/kda.py`、
> `megatron/core/packed_seq_params.py`、`tests/unit_tests/ssm/gated_delta_net/test_kda.py`
> 与容器内 `fla==0.5.2`。

---

## 1. Megatron 的 packed（THD）路径长什么样

packed 输入用 `PackedSeqParams`（`packed_seq_params.py:15` 定义）描述：填
`qkv_format="thd"`、`cu_seqlens_q`/`cu_seqlens_kv`（含 `..._padded` 变体）与
`max_seqlen_*`。真实 KDA 层的 packed 分支在 `kda.py:272-289`：

```python
if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
    if batch != 1:
        raise ValueError("Packed KDA expects batch dimension to be 1.")   # kda.py:273-274
    cu_seqlens_q = self._resolve_cu_seqlens(
        packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q,
        seq_len_global, "cu_seqlens_q", cp_size=cp_size_runtime)          # kda.py:275-281
    cu_seqlens_kv = self._resolve_cu_seqlens(...)                          # kda.py:282-288
    self._validate_packed_cu_seqlens(cu_seqlens_q, cu_seqlens_kv)          # kda.py:289
```

随后这份 `cu_seqlens_q` 原样被喂给核心算子（`kda.py:447`：`cu_seqlens=cu_seqlens_q`）。
输入张量形状：SBHD 是 `[T, B, hidden]`，THD 把它按 batch 拍平成 `[T_total, 1, hidden]`
（`batch=1`、`seq_len_global = T_total`）。Megatron 自带测试给了标准构造
（`test_kda.py:217-223`）：

```python
unpacked = torch.randn((16, 2, hidden))                          # [T,B,H]
packed   = unpacked.transpose(0, 1).contiguous().reshape(-1, 1, hidden)  # [T*B,1,H]
packed_seq_params = make_test_packed_seq_params(cu_seqlens=[0, 16, 32])
```

即：**把每个 batch 行当成一条长度 `T` 的序列沿 token 维拼起来**，`cu_seqlens` 为
`[0, T, 2T, …]`。这也是本项目 Part D 的构造。

### 为什么 `cu_seqlens` 同时传给核心算子

FLA `chunk_kda` 收 `cu_seqlens` 后，**每条序列独立地从零状态开始递推**（`initial_state=None`、
`output_final_state=False`，`kda.py:441-442` 写死）。这正是 M8-1 Part B 锚定过的 varlen 语义：
不是把 `T_total` 当一条长序列，而是按 `cu_seqlens` 分段、段间状态隔离。因此 packed 路径的
正确性锚定必须用**逐序列循环**的参考——见 §3。

---

## 2. 对拍设置

除 M10-1 的两条 import 期 workaround（nvrx 版本号 patch + `EP_SUPPRESS_NCCL_CHECK=1`）外，
本脚本新增两处：

**(a) packed 参数与输入。** `make_packed_seq_params(cu_seqlens)` 照抄
`test_multi_latent_attention.py:45-59`：`cu_seqlens` 转 `IntTensor(…).cuda()`、
`max_seqlen_*` 取段长最大值、`qkv_format="thd"`。三组测试 pack（含非 64 整除段长）：

| pack `cu_seqlens` | 段长 | `T_total` |
|---|---|---|
| `[0, 64, 128]` | 64, 64 | 128 |
| `[0, 50, 130]` | 50, 80 | 130 |
| `[0, 64, 100, 250]` | 64, 36, 150 | 250 |

**(b) packed-aware fp64 参考。** 参考芯的 `gated_delta_rule` 现在接收 `cu_seqlens`，对它
做逐序列循环（`_ref_core`）：把 `q/k/v/g/beta` 按相邻 `cu_seqlens` 切片，对每段单独调
`kda_chunked_fwd(q[:, s:e], …)`（`chunk_size=64`、零初态），再沿 token 维拼回
`[1, T_total, H, *]`。门控激活 `kda_gate` 先在整段上算再切（逐元素、可交换），`beta`
Megatron 侧已是 sigmoid 后的值、两边都直接吃。**这条逐序列参考与 FLA 的 packed 语义一致**，
因此 Part B 的残差能干净地归到 tf32/bf16 数值（而不是状态串扰）。

---

## 3. Part A：packed 真层前向/反向 smoke

三组 pack × fp32/bf16：

```
     cu=[0, 64, 128]       T= 128  shape/dtype=True  hs.grad finite=True  nonfinite_params=0
     cu=[0, 50, 130]       T= 130  shape/dtype=True  hs.grad finite=True  nonfinite_params=0
     cu=[0, 64, 100, 250]  T= 250  shape/dtype=True  hs.grad finite=True  nonfinite_params=0
```

输出形状/dtype 与输入一致（`[T_total, 1, hidden]`）、`bias is None`、`hs.grad` 有限、全部
参数梯度有限；段长非 64 整除（`36`/`80`/`150`）也不炸。

---

## 4. Part B：packed 核心算子锚定（FLA vs fp64 参考）

在真层**当次**喂给 `chunk_kda` 的那份操作数（`cu_seqlens=[0,64,128]`、`q` 形状
`(1, 128, 4, 32)`）上，FLA `chunk_kda`（带 `cu_seqlens`）对比逐序列 fp64 参考：

`float32`（tf32）：

```
     fwd o: (1,128,4,32) vs (1,128,4,32)  max_abs=1.038e-04  rel=1.993e-03
     grad q rel=3.529e-03  grad k rel=3.611e-03  grad v rel=3.242e-03
     grad g rel=3.601e-03  grad beta rel=3.349e-03
```

`bfloat16`：

```
     fwd o: max_abs=2.441e-04  rel=3.676e-03
     grad q rel=4.255e-03  grad k rel=6.369e-03  grad v rel=5.917e-03
     grad g rel=9.390e-03  grad beta rel=4.898e-03
```

**关键结论**：packed 模式下 FLA 与「逐序列独立处理」的参考落在**同一份 tf32/bf16 底噪**
（fp32 ~2–3.6e-3、bf16 ~3.7–9.4e-3，与 M6/M7/M8/M10-1 同量级）。若 FLA 的 `cu_seqlens`
分段真把相邻序列的状态串了起来，误差会是 O(1)（状态泄漏），而不会稳在底噪——所以这条锚定
**同时验证了 packed 路径的段间状态隔离**。

---

## 5. Part C：packed 整模块端到端（FLA 芯 vs 参考芯）

两个权重逐位相同的模块，核心算子分别走 FLA 与 fp64 参考，都用 `packed_seq_params`，
比整模块输出 / `hidden_states.grad` / 全部参数梯度：

`float32`：

```
     cu=[0, 64, 128]       T= 128  out=4.788e-04  hs.grad=2.222e-08  param.grad=9.901e-07
     cu=[0, 50, 130]       T= 130  out=4.420e-04  hs.grad=2.447e-08  param.grad=1.350e-06
     cu=[0, 64, 100, 250]  T= 250  out=5.231e-04  hs.grad=1.323e-08  param.grad=1.946e-06
```

`bfloat16`：

```
     cu=[0, 64, 128]       T= 128  out=2.686e-03  hs.grad=1.341e-07  param.grad=5.245e-06
     cu=[0, 50, 130]       T= 130  out=2.075e-03  hs.grad=1.192e-07  param.grad=4.292e-06
     cu=[0, 64, 100, 250]  T= 250  out=2.930e-03  hs.grad=8.941e-08  param.grad=3.815e-06
```

量级与 M10-1 的 SBHD 行一致（fp32 `out` ~5e-4、bf16 ~2–3e-3；梯度 fp32 ~2e-8/2e-6、
bf16 ~1e-7/5e-6）。**结论**：把核心算子从 FLA 换成独立 fp64 参考，经过 packed 路径的
`in_proj/conv1d/a2a(恒等)/核心/RMSNorm/out_proj` 完整前反向之后，误差仍在底噪内——
packed 的 `cu_seqlens` 记账（切段、段间隔离、conv1d 的跨段处理）没有引入隐藏偏差。

---

## 6. Part D：packed == unpacked 自洽

用**同一个 FLA 芯**、两份逐位同权重的模块，把同一批序列内容喂成两种布局：

- unpacked：`[T, B, hidden]`（SBHD，B=2）；
- packed：`x.transpose(0,1).reshape(T*B,1,hidden)`，`cu_seqlens=[0,T,2T]`（THD）。

比整模块输出（packed 的输出按同序拍平后比）、`hidden_states.grad`、参数梯度：

```
  float32 :   out max_abs=3.156e-05  hs.grad max_abs=2.361e-09  param.grad max_abs=8.315e-08
  bfloat16:   out max_abs=0.000e+00  hs.grad max_abs=0.000e+00  param.grad max_abs=2.384e-07
```

bf16 下输出与输入梯度**逐位相同**（`max_abs=0`）——packed 与 unpacked 走的是同一批数学、
只是 token 维排布不同。fp32 的 `out`/`hs.grad` 有 ~3e-5/2e-9 的微差、param.grad ~8e-8，
来源是两种布局下 conv1d / GEMM 的**归约顺序不同**（非算法差）；bf16 因为两者都量化到同一
批数值反而逐位对齐。这与 Megatron 自己的测试（`test_kda.py:212-232`，只比前向、
`atol=rtol=5e-3`）互洽，且本项目把口径收紧到反向与参数梯度。

---

## 7. 结论与边界

- **真实 `KimiDeltaAttention` 的 packed / THD 路径在容器内跑通**（多序列拍平、
  `cu_seqlens` 切分、`batch=1`、含非 64 整除段长），前反向数值健全；
- **核心算子粒度**：FLA `chunk_kda`（带 `cu_seqlens`）与逐序列 fp64 参考只差 tf32/bf16
  底噪 → packed 段间状态隔离被验证，无隐藏偏差；
- **整模块粒度**：FLA 芯 vs 参考芯在 packed 下仍只差底噪，`cu_seqlens` 记账不放大误差；
- **自洽**：packed 与 unpacked 在 bf16 下逐位一致、fp32 下差异仅来自归约顺序。
- **边界**：仍是**单卡 tp=cp=1**；packed **没有 padding**（`cu_seqlens_q_padded` 未用，
  `cu_seqlens_q == cu_seqlens_kv`，未触发 `_validate_packed_cu_seqlens` 的对齐/相等分支）、
  未测 **cp>1 的 headwise/chunkwise CP** 下的 packed（需多 rank + zigzag 布局，见 M8-1/M8-2）、
  GVA（Megatron KDA 本就不支持，见 `docs/08`）与 `gdn_pre_gated_delta_rule_fusion` 缺口未覆盖。
- **复现**：`harness/check_port_e2e_packed.py`；RNG 由 `model_parallel_cuda_manual_seed`
  与 `torch.manual_seed` 固定；输出 `harness/check_port_e2e_packed.out.txt`。
