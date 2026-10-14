# 把原生 fused pre-GDR 接到 `gdn_pre_gated_delta_rule_fusion` 路径（M14-1）

> `docs/14`（M12-1）证明了 GDN 的 streamed 融合核**不能**复用于 KDA，并给出了 KDA 专属
> fused pre-GDR 的代数/布局契约；`docs/15`–`docs/17`（M13-1/2/3）把这个契约兑现成一个
> Triton 核族（conv+SiLU / L2-norm / beta sigmoid 前向 + 原生 Triton 反向），并把它作为
> `pre_gated_delta_rule` 的 **drop-in** 逐层数值锚定。
>
> 剩下最后一件事（megatron-port 路线图里唯一未做的「实现工程」项）：让层自己的开关
> `gdn_pre_gated_delta_rule_fusion` 真的**分派到**这套原生核，而不是像今天这样直接
> `raise`。本篇用**运行时 shim** 完成这次接线——**不改 Megatron-LM 一行**（与历轮一致），
> 从而可以在动真实树之前把「接线后数值是否仍透明」端到端钉死。
>
> 实现：`src/kda_pregdr_fusion_shim.py`；数值/计时：
> `harness/check_port_pregdr_fusion_wire.py`（`harness/check_port_pregdr_fusion_wire.out.txt`，
> 空闲卡单进程）。行号按容器内 Megatron-LM（`~/proj/Megatron-LM`）核对。

## 1. 要接的是哪两处，以及为什么它们是「同一个判断」

KDA 在两个位置拒绝 `gdn_pre_gated_delta_rule_fusion`，且都抛同一句
`NotImplementedError`：

- **配置期**（`transformer_config.py:1936-1938`，`TransformerConfig.__post_init__` 内联）：
  当 `gdn_pre_gated_delta_rule_fusion and experimental_attention_variant == "kda"` 时拒绝；
- **前向期**（`kda.py:399-402`，`KimiDeltaAttention.forward`）：
  即便绕过配置、把模块属性手动置 True，前向仍会命中：
  ```python
  if self.gdn_pre_gated_delta_rule_fusion:
      raise NotImplementedError(
          "gdn_pre_gated_delta_rule_fusion is not implemented for KDA yet."
      )
  ```

模块属性来自 `_setup_variant_attrs` 的 `self.gdn_pre_gated_delta_rule_fusion =
self.config.gdn_pre_gated_delta_rule_fusion`（`kda.py:126`）。两道 guard 不是冗余防御，
而是**同一句结构判断的两次落点**：`docs/14` §4 已逐条给出为什么 GDN 融合核与 KDA 的
pre-GDR 在**投影段**（`kda.py:133-140`，`g` 是 per-channel、`beta` 走独立 `beta_proj`）、
**门控激活位置**（`kda.py:167-184`，KDA 只 sigmoid `beta`、把 decay 激活留给核心核）、
**门控参数形状**（`kda.py:141-142`，`dt_bias` per-channel、`A_log` per-head）四处不同，
直接复用会静默算错。

所以「接线」不是删掉 guard，而是**把 raise 换成分派**：KDA 分支改调 KDA 专属的
fused pre-GDR（前向 + 原生反向）。真实树里这会是一段
`if self.gdn_pre_gated_delta_rule_fusion: q,k,v,gate,beta,g,A_log,dt_bias =
self.pre_gated_delta_rule_fused(...) else: self.pre_gated_delta_rule(...)`；
本篇先用 monkeypatch 得到同样的语义。

## 2. shim 的设计（`src/kda_pregdr_fusion_shim.py`）

`install(kimi_cls, config_cls, fused_fn)` 打两个补丁，语义就是把两处 guard「换成 KDA 分支」：

1. **config guard**：把 `config_cls.__post_init__` 包一层——若
   `gdn_pre_gated_delta_rule_fusion=True` 且 `variant=="kda"`，则在执行原 `__post_init__`
   期间把该字段**临时屏蔽成 False**，跑完再还原。于是配置能带 `fusion=True` 构造出来，
   而 `__post_init__` 的其它校验一字未动。
2. **forward guard**：把 `kimi_cls.forward` 包一层——当实例 `gdn_pre_gated_delta_rule_fusion`
   为真时，**临时**把实例属性置 False（通过 `:399` 的检查）、把实例的
   `pre_gated_delta_rule` 临时换成 `types.MethodType(fused_fn, self)`，调用**原始 forward**，
   finally 里还原。这样走的仍是真实 forward 的其余全部逻辑（切段、CP、RMSNorm 输出门等），
   只有「pre-GDR 由谁算」被替换。

`fused_fn` 传的是 `src/kda_pregdr_bwd.py::kda_pre_gdr_fused_native`——即 M13-1 的 Triton
前向 + M13-2/3 的**原生 Triton 反向**（含 `conv1d.weight`/`conv1d.bias` 作为 autograd
输入）。shim 本身不 import Megatron，保持可单测。

## 3. Part A/B：guard 确实存在，且能被 shim 抬起

**Part A（shim OFF）**：直接构造带 `fusion=True` 的 KDA 配置 → 命中配置 guard；把已建模块
的属性手动置 True 再前向 → 命中前向 guard。fp32/bf16 两档都如此：

```
config guard: raised NotImplementedError('gdn_pre_gated_delta_rule_fusion is not imple')
forward guard: raised NotImplementedError('gdn_pre_gated_delta_rule_fusion is not imple')
```

**Part B（shim ON）**：`install(...)` 后，同一份 `fusion=True` 配置能构造
（`flag=True`）、模块属性为 True、前向**跑通**（`out shape=(64, 2, 128) finite=True`）。
即两处 guard 都被「换成分派」，而不是被删除。

## 4. Part C：接线后数值仍透明（同权重，逐层前反向）

用同一 seed 建两个模块并 `load_state_dict` 对齐：一个 `fusion=False`（eager
`pre_gated_delta_rule`），一个 `fusion=True`（shim 分派到原生 fused）。同一输入前向 +
`sum(o).backward()`，比前向输出、输入梯度、以及**全部 7 个参数**的梯度：

| dtype | out max_abs | hs.grad max_abs | param.grad max_abs | param.grad rel | compared |
|---|---|---|---|---|---|
| fp32 | 1.095e-04 | 2.298e-04 | 1.305e-02（max\|g\|=72.09） | **1.810e-04** | 7 |
| bf16 | 1.953e-03 | 3.906e-03 | 2.500e-01（max\|g\|=72.00） | **3.472e-03** | 7 |

量级与 M13-1 的整层 drop-in（fp32 `out 1.230e-04`、`param.grad rel 2.082e-04`）一致，
bf16 与 M10/M11 的整模块 bf16 锚定（~3e-3）同阶。fp32 的 `out ~1e-4` 来自
「FLA `l2norm` 原语 vs Triton `l2norm` 核」的归约序残差，经 delta-rule 递推 + RMSNorm
放大，远低于部署底噪。

`conv1d.weight.grad` 两侧**不是逐位相同**（脚本打印 `equal=False`）——这是预期的：eager 侧
走 FLA `causal_conv1d` 反向，fused 侧走 M13-2 的 Triton conv 反向，两条不同实现只在 fp32
舍入序上不同（其误差已包含在 `param.grad max_abs=1.305e-02` 里，rel 1.810e-04）。M13-3 记的
「两侧同值」指的是**打印的最大幅值**相同（`8.637e+00`），并非逐位——此处明确口径。

## 5. Part D：整层（前向+反向）计时

每 shape 用 event-pair `wall` + CUPTI `dev` 各 warmup 10 / iter 30：

| dtype | T×B | wall eager/fused（×） | dev eager/fused（×） |
|---|---|---|---|
| fp32 | 64×2 | 4942.7 / 4478.4（1.104） | 173.0 / 188.5（**0.918**） |
| fp32 | 256×2 | 5565.6 / 5033.0（1.106） | 239.5 / 250.9（**0.955**） |
| fp32 | 1024×2 | 5370.2 / 4838.7（1.110） | 433.6 / 440.8（**0.984**） |
| fp32 | 1024×8 | 5413.6 / 5243.5（1.032） | 808.9 / 781.7（**1.035**） |
| bf16 | 64×2 | 5386.2 / 4955.0（1.087） | 163.1 / 186.9（**0.872**） |
| bf16 | 256×2 | 5576.8 / 4941.1（1.129） | 211.8 / 233.7（**0.906**） |
| bf16 | 1024×2 | 5405.6 / 4841.1（1.117） | 369.9 / 383.3（**0.965**） |
| bf16 | 1024×8 | 5435.4 / 4885.5（1.113） | 597.7 / 599.0（**0.998**） |

两点：

- **整层 `wall` 稳定赢 ~1.03–1.13×**：整层的 host 税（`chunk_kda` 的 Triton launch +
  autograd/dispatch）本就有 ~5 ms 量级，fused pre-GDR 把三 scope 融成少量 kernel 后
  省下的 launch 被摊薄，但方向为正。
- **纯设备 `dev` 小 shape 略输、大 shape 翻正**（0.87→1.04），与 `docs/17` §5.2 的
  M13-3 隔离口径同形。整层 `dev` 里 `chunk_kda` 的反向占大头，pre-GDR 份额小，
  所以这里只作参考；pre-GDR 自身的 `dev` 口径见 `docs/15`/`docs/17`。

## 6. 结论与边界

- **接线完成（运行时）**：两处 `gdn_pre_gated_delta_rule_fusion` guard 都能被 shim 抬起，
  层带着 `fusion=True` 跑通、且 pre-GDR 由 KDA 专属原生核（前向 + 原生反向）服务。
- **数值透明**：同权重逐层前反向，7 个参数梯度 fp32 rel **1.81e-04** / bf16 rel **3.47e-03**，
  与 M13-1 整层 drop-in 同阶；`conv1d.weight.grad` 与 eager（FLA `causal_conv1d`）只在
  fp32 舍入序上不同，非逐位。
- **性能**：整层 `wall` 1.03–1.13×；设备 `dev` 小 shape 0.87–0.96×、大 shape ~1.0×。
- **未改 Megatron-LM 一行**：真实树里这一步应写成
  `kda.py:399` / `transformer_config.py:1936` 的分派分支（把 `raise` 换成 KDA
  `pre_gated_delta_rule_fused`），本 shim 就是那段分支的语义等价物与数值前哨。
- **边界**：SBHD、单卡 `tp=cp=1`、GVA 因子 1（Megatron KDA 硬约束）、`use_qk_l2norm=True`；
  shim 在 forward 里对 `pre_gated_delta_rule` 的 cp/varlen 参数**按契约忽略**（原生核族仅
  覆盖 cp=1）；未覆盖 packed THD / cp>1 的 fused pre-GDR；`conv_bias` 端到端（非零 `dbias`）
  已由 `docs/17` §4.2 Part E 在 stub 上单独验证，真实 KDA 层 `conv1d.bias=None`。
