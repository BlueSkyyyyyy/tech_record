# 11 · 真层 packed 的 padding 语义（`cu_seqlens_q_padded`）对拍（M10-3）

> 系列第 12 篇。M10-2 锚定了真实 `KimiDeltaAttention` 的 packed / THD 路径，但那条路径
> **没有 padding**：`cu_seqlens_q == cu_seqlens_kv`、`cu_seqlens_q_padded` 留 `None`（`docs/10`
> §7 明确列为未覆盖边界）。本篇（M10-3）补上它：给 THD 批次带上 padding（`cu_seqlens_padded`
> 描述物理布局、`cu_seqlens` 描述真实序列边界），看真实层的核心算子/整模块在带 padding 时是否
> 仍然数值健全，并**单独考察 padding 会不会污染真实 token 的输出**。
>
> 脚本：`harness/check_port_e2e_padded.py`；原始输出：`harness/check_port_e2e_padded.out.txt`。
> 行号按本机版核对：`Megatron-LM/megatron/core/ssm/gated_delta_net/kda.py`、
> `megatron/core/ssm/gated_delta_net/common.py`、`megatron/core/packed_seq_params.py`，
> 容器内 `fla==0.5.2`。

---

## 1. padding 在 KDA 路径上是什么意思

`PackedSeqParams`（`packed_seq_params.py:15`）带两组边界字段（`packed_seq_params.py:27-29`）：

- `cu_seqlens_q`/`cu_seqlens_kv`：**真实**序列边界；
- `cu_seqlens_q_padded`/`cu_seqlens_kv_padded`：**物理布局**边界（序列被 padding 到对齐长度后
  的累计长度）。

真实层的 packed 分支（`kda.py:272-289`）并不直接吃 `cu_seqlens_q`，而是先解析：

```python
cu_seqlens_q = self._resolve_cu_seqlens(
    packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q,
    seq_len_global, "cu_seqlens_q", cp_size=cp_size_runtime)     # kda.py:275-281
...
self._validate_packed_cu_seqlens(cu_seqlens_q, cu_seqlens_kv)     # kda.py:289
```

`_resolve_cu_seqlens`（`common.py:482-506`）的逻辑是：**只要 `..._padded` 存在就用它**（`common.py:486-489`），
并要求 `cu_seqlens[-1] == total_sequence_length`（`common.py:491-497`）且每段长度能被 `cp_size` 整除
（`common.py:499-504`，本机 cp=1 恒真）。`_validate_packed_cu_seqlens`（`kda.py:643-660`）再要求
`cu_seqlens_q` 与 `cu_seqlens_kv` **逐元素相等**（padding 后两者同形）。

后果很关键：**核心算子（`kda.py:447`）、causal-conv1d（`kda.py:580/598-607`）拿到的都是
*phrasing* 后的物理边界**。也就是说，物理流里每条序列的 padding token 会被当成**真实 token**
参与 conv 与 KDA 递推——真实 token 排在各自 padded slot 的**前段**，pad 填在后段。这与
Megatron 打包 THD（`pad_between_seqs`）的布局一致。

本篇就构造这种布局（`[real | pad]` 逐序列拼接），验证两件事：

1. 带 padding 时，核心算子与 fp64 参考（按同一物理边界逐段处理）是否仍只差底噪；
2. padding 内容对**真实 token** 的输出有没有影响——尤其是**跨序列**（其它序列的 pad 会不会串进来）。

---

## 2. 对拍设置

沿用 M10-1/M10-2 的两条 import 期 workaround。两组用例（实际段长 → padded 段长）：

| `cu_seqlens`（真实） | `cu_seqlens_padded`（物理） | real / pad |
|---|---|---|
| `[0, 50, 110]` | `[0, 64, 128]` | 50+14、60+4 |
| `[0, 50, 100, 220]` | `[0, 64, 128, 256]` | 50+14、50+14、120+8 |

物理输入形状 `[T_padded, 1, hidden]`；真实 token 放在每个 slot 前段，pad 用独立随机数填充。
fp32/bf16 各跑一遍。

---

## 3. Part A：带 padding 的真层前向/反向 smoke

```
  float32: a=[0,50,110] p=[0,64,128]      T_pad=128  shape/dtype=True  hs.grad finite=True  bad=0
           a=[0,50,100,220] p=[0,64,128,256] T_pad=256  shape/dtype=True  hs.grad finite=True  bad=0
  bfloat16: 同上两组，hs.grad finite=True、非有限参数梯度数 0
```

输出形状/dtype 与输入一致（`[T_padded,1,hidden]`）、`bias is None`、输入与全部参数梯度有限。
带 padding、非 64 整除的真实段长都不炸。

---

## 4. Part B：带 padding 的核心算子锚定（FLA vs fp64 参考）

在真层**当次**喂给 `chunk_kda` 的那份操作数上，`chunk_kda`（带 **padded** `cu_seqlens=[0,64,128]`）
对比「按同一 padded 边界逐段处理」的 fp64 参考：

```
  float32 :  fwd o max_abs=1.325e-04 rel=2.558e-03
             grad q rel=3.275e-03  grad k rel=3.088e-03  grad v rel=3.302e-03
             grad g rel=3.599e-03  grad beta rel=2.666e-03
  bfloat16:  fwd o max_abs=2.441e-04 rel=3.876e-03
             grad q rel=4.545e-03  grad k rel=6.250e-03  grad v rel=7.194e-03
             grad g rel=5.495e-03  grad beta rel=2.574e-03
```

落 tf32/bf16 底噪（与 M6/M7/M8/M10-1/M10-2 同量级）。**关键**：这里参考也是按 padded 边界逐段
处理的——因为层的确把 padded 边界交给了核心，物理流里 pad token 就是段内 token。所以这条锚定
验证的是「核心在 padded 段上算得对」，不是「pad 被忽略」。

---

## 5. Part C：padding 污染的**归因**（逐操作数 flip）

一个自然的问题：pad token 会不会影响真实 token 的输出？把捕获的核心操作数里**所有 pad 位置的
`q/k/v/g/beta` 换成另一组随机值**，逐操作数翻转，看真实 token 的输出变化：

```
  float32 : flip fill of q    : real-token max=0.000e+00   (0/110)
            flip fill of k    : real-token max=0.000e+00   (0/110)
            flip fill of v    : real-token max=0.000e+00   (0/110)
            flip fill of g    : real-token max=1.293e-05 (at t=49)  (2/110)
            flip fill of beta : real-token max=0.000e+00   (0/110)
  bfloat16: 同上，只有 g 非零（1.526e-05 at t=48）
```

**唯一驱动是门控 `g`**：pad 的 `q/k/v/beta` 对真实 token 输出**逐位无影响**（`max_abs=0`），
只有 `g`（log-decay）会让**每个 padded slot 的最后一个（几个）真实 token**产生 ~1e-5 的扰动。
机制：FLA chunked KDA 在块内用累计衰减 `g_cum` 做数值稳定化，块尾（含 pad）参与该块的衰减
参考量；pad 的 `g` 变 → 块尾衰减变 → 块内真实 token 的 `exp` 参考量微变。这是**块尾的
causal-tail 舍入效应**，不是 `q/k/v` 那种代数串扰。

---

## 6. Part D：模块级 padding 隔离（跨序列 vs 自身 slot）

在真实层上（FLA 芯、只前向、同一权重），对每个 padded slot 单独把它的 **pad 尾**换成另一组
随机值（真实 token 一字不动），比较真实 token 位置的最大输出变化，分成「自身 slot」与「其它
序列 slot」：

```
  float32 :  own-slot fill flip   : out max_abs=9.119e-05
             other-slot fill flip : out max_abs=0.000e+00   ← 跨序列隔离（精确 0）
  bfloat16:  own-slot fill flip   : out max_abs=1.953e-03
             other-slot fill flip : out max_abs=0.000e+00
```

**两条结论**：

- **跨序列隔离是精确的（`max_abs=0`）**：改一条序列的 padding 永远不会影响**别的**序列的真实
  token——无论是 fp32 还是 bf16、跑 2 组用例 × 3 个 seed 都是零。这是打包训练最要紧的安全属性。
  （causal-conv1d 与 KDA 递推都在 padded 边界处重置，pad 只落在本 slot 内。）
- **自身 slot 的 pad 尾会让该序列最后几个真实 token 有 ~1e-4（fp32）/ ~2e-3（bf16）的扰动**
  ——即 §5 归因的 `g` causal-tail 效应经 RMSNorm/out_proj 放大后的量级；bf16 的 2e-3 约 1–2 ulp。

---

## 7. Part E / F：整模块锚定与 padded↔unpadded 等价

**Part E**（整模块，FLA 芯 vs fp64 参考芯，权重逐位相同，都用带 padding 的 `packed_seq_params`）：

```
  float32 :  worst out=5.066e-04  hs.grad=2.269e-08  param.grad=1.641e-06
  bfloat16:  worst out=2.930e-03  hs.grad=1.192e-07  param.grad=5.722e-06
```

与 M10-1（SBHD）/M10-2（packed 无 padding）的整模块行同量级（fp32 `out` ~5e-4、bf16 ~3e-3）。

**Part F**（同一 FLA 芯，带 padding 的真实 token 位置 vs 不带 padding 的布局）：

```
  float32 :  worst out=1.084e-04  hs.grad=5.876e-06
  bfloat16:  worst out=1.953e-03  hs.grad=6.013e-06
```

真实 token 的输出/输入梯度在两种布局下只差 `~1e-4`（fp32）/ `1–2 ulp`（bf16）——差异来自
**chunk 划分不同**（无 padding 时短序列按 32 分块，带 padding 时一个 slot 恰一个 64 块）与
§6 的 `g` causal-tail，属数值舍入而非语义差。

---

## 8. 结论与边界

- **真实 `KimiDeltaAttention` 的带 padding packed / THD 路径在容器内跑通**（`cu_seqlens_padded`
  走 `_resolve_cu_seqlens` 的 padding 分支、pad token 参与 conv 与递推、含非 64 整除段长），
  前反向数值健全；
- **核心/整模块锚定**：带 padding 时 FLA `chunk_kda` 与 fp64 参考仍只差 tf32/bf16 底噪；
- **跨序列隔离精确**：任何序列的 pad 对其它序列真实 token 的影响 `max_abs=0`（fp32/bf16、
  多 seed）；
- **自身 slot 的 pad 尾**会经门控 `g` 的块尾衰减参考量，让该序列最后几个真实 token 产生
  `~1e-4`（fp32）/ `~2e-3`（bf16，1–2 ulp）的扰动，`q/k/v/beta` 无影响；padded↔unpadded 真实
  token 差异同量级。这是**块尾舍入**，不是跨序列串扰，且低于 tf32/bf16 部署底噪。
- **边界**：仍是**单卡 tp=cp=1**；未测 **cp>1 的 headwise/chunkwise CP** 下的 padded packed
  （需多 rank + zigzag，见 M8-1/M8-2）、GVA（Megatron KDA 本就不支持，见 `docs/08`）与
  `gdn_pre_gated_delta_rule_fusion` 缺口未覆盖；padding 只测「每序列各自补齐」布局
  （`pad_between_seqs` 语义），「只在最后一段补齐」布局未单独测。
- **复现**：`harness/check_port_e2e_padded.py`；RNG 由 `model_parallel_cuda_manual_seed` 与
  `torch.manual_seed` 固定；输出 `harness/check_port_e2e_padded.out.txt`。
