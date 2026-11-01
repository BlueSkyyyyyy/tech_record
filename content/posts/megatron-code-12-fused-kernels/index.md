---
title: "Megatron 源码精读（十二）：fused 算子"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "cuda", "算子融合"]
categories: ["训练框架"]
weight: 12
series: ["megatron-code"]
---

上一篇[《优化器》]({{< relref "megatron-code-11-optimizer" >}})收束在「下一步看 fused 算子」。本篇回答一个问题：`megatron/core/fusions/` 里那些「fused」，到底把什么和什么融在了一起，又在什么条件下会悄悄退回未融合路径。

**一句话结论**：这个目录里的「融合」至少有四个来源——预编译 CUDA 扩展、Triton kernel、`@jit_fuser` 编译的逐点算子、只把通信合并掉的 cross entropy；共同点只有一个接口范式：用 `torch.autograd.Function` 接进计算图，不满足条件就退回。

---

## 1. 问题：融合省的是带宽，不是算力

attention 里的 `scale → mask → softmax` 是三个独立 kernel，中间 tensor 每次都要写回 global memory 再读出来。这类逐元素/归一化算子算力密度极低，**瓶颈在 memory bandwidth**，多一次往返就多一次纯浪费。

所以 fuse 只买两样东西：少启动两次 kernel；让中间结果留在寄存器或共享内存里。它不会让 FLOPs 变少——若某环节本来是 compute-bound，融合收益会迅速趋零。

代价是灵活性和自动求导。MCore 的补偿方式是：kernel 外面套一层 `torch.autograd.Function`，把「能不能用」写成一个显式判断，判断失败就走未融合路径。

## 2. 机制：四类「fused」，同一个降级范式

### 2.1 先分清你在读哪一种 fused

同名目录下的实现手段差别很大，混着读必然读错：

| 类别 | 代表 | 融合手段 |
| --- | --- | --- |
| 预编译 CUDA 扩展 | `fused_softmax.py`、`fused_layer_norm.py` | `*_cuda` / apex extension |
| Triton kernel | `fused_mrope.py`、`fused_mhc_kernels.py` | `@triton.jit`，运行时编译 |
| 编译器融合 | `fused_bias_swiglu.py`、`fused_bias_gelu.py` | `@jit_fuser`（默认 `torch.compile`） |
| 通信合并 | `fused_cross_entropy.py` | 两次 all-reduce 合成一次 |

一个判据是看 `import` 写在哪：CUDA 扩展的 `import xxx_cuda` 全在函数体内，只有真被调用时才 ImportError。

`@jit_fuser`（`megatron/core/jit.py`）默认就是 `torch.compile`，torch < 2.2 才用 `torch.jit.script`，也能被 `disable_jit_fuser()` 降成空装饰器。所以「加了 autograd.Function」不等于「有 CUDA kernel」——swiglu 那层包 Function 是为了 fp8 存储、CPU offload 和重算，不是为了 kernel。

### 2.2 降级谓词不一定叫 `is_kernel_available`

softmax 用一个返回布尔值的 `is_kernel_available`；`fused_mrope.py` 换了个写法，用 `get_fused_mrope_unavailable_reason`（`fused_mrope.py:149`）返回「为什么不能用」的字符串：没装 Triton、dtype 与算力不匹配、张量不在 CUDA 上，都能给出具体原因。

形式不同，作用一样：把「硬件/依赖/形状不满足」这类信息集中在一处，调用点只做一次判断。

## 3. 代码落点：三个符号

### 3.1 `FusedLayerNorm`：24 个值的白名单

`FusedLayerNorm`（`fused_layer_norm.py:30`）有两档 kernel。persistent 档（apex `FastLayerNormFN`）只支持一份写死的列表，共 24 个值（`fused_layer_norm.py:73-98`）：

1024 / 1536 / 2048 / 2304 / 3072 / 3840 / 4096 / 5120 / 6144 / 8192 / 10240 / 12288 / 12800 / 15360 / 16384 / 18432 / 20480 / 24576 / 25600 / 30720 / 32768 / 40960 / 49152 / 65536

它不是「1024 到 65536 的 2 的幂」，中间混着 1536、2304、3840、12800 这类非 2 幂值。

判定顺序：构造参数 `persist_layer_norm` 先被 `self.config.persist_layer_norm` 覆盖，然后才检查「hidden_size 在不在列表里、apex 的 persistent 模块能不能 import」，任一不满足就置 False（`fused_layer_norm.py:99-101`）。

没有 apex 不是降级，是构造期报错：

```python
if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:
    raise ValueError(f'Apex must be installed to use FusedLayerNorm.')
```

（`fused_layer_norm.py:103-105`）两档都依赖 apex：不在白名单只退到 apex 的普通 fused layer norm，**没有纯 PyTorch 分支**（源码里还留着 `# TODO: Add pytorch only layer norm`）。

### 3.2 `FusedScaleMaskSoftmax`：一个布尔函数决定走不走 CUDA

`forward` 只有一行判断（`fused_softmax.py:233-236`）：`is_kernel_available(mask, *input.size()) and softmax_offset is None`。所以「代码里有融合路径」不等于「跑的是融合 kernel」。

第一层是六个开关（`fused_softmax.py:253-260`，注释为笔者所加）：

```python
self.scaled_masked_softmax_fusion   # 用户开了融合
and self.input_in_float16           # fp16 或 bf16
and 16 < sk <= 4096                 # sk 严格大于 16
and sq % 4 == 0 and sk % 4 == 0
and attn_batches % 4 == 0           # b * np
```

`16 < sk` 是严格不等号（注释里的 `16 ~ 2048` 已过时）。第二层容易漏：还要拿 CUDA 侧算出的 `batch_per_block` 做整除检查，causal 看 `attn_batches % batch_per_block`，非 causal 看 `sq % batch_per_block`，都过才返回 True（`fused_softmax.py:261-269`）。

三个 Function 对应三种 mask 语义：`ScaledUpperTriangMaskedSoftmax`（`fused_softmax.py:11`）是 causal，自己在 kernel 里生成上三角 mask，因此要求 `sq == sk`，并把 4D 输入 `view` 成 3D 再进 kernel；`ScaledMaskedSoftmax`（`:60`）吃显式 additive mask；`ScaledSoftmax`（`:108`）无 mask。

三者 forward 都只 `save_for_backward(softmax_results, scale_t)`，backward 也只把 softmax 结果和 scale 喂回 CUDA。这不是偷懒：softmax 的 Jacobian 完全由输出决定，

$$
\frac{\partial L}{\partial x} = y \odot \left( \frac{\partial L}{\partial y} - \left\langle \frac{\partial L}{\partial y}, y \right\rangle \mathbf{1} \right)
$$

因而既不需要保存原始 logits，也不需要重算 mask。

### 3.3 `fused_vocab_parallel_cross_entropy`：融合的是通信

这里最容易混的是两个同名类。`megatron/core/tensor_parallel/cross_entropy.py` 的 `_VocabParallelCrossEntropy` 是通用版，带 `label_smoothing`，做**两次** SUM all-reduce（predicted logits 和 sum_exp 各一次）。

`fusions/fused_cross_entropy.py` 的版本没有 label smoothing，真正的改动是把两者 `torch.cat` 成一条再 all-reduce 一次（`fused_cross_entropy.py:42`、`:112-114`），外加把逐点部分挂上 `@jit_fuser`。

`fused_vocab_parallel_cross_entropy`（`fused_cross_entropy.py:136`）只是 `_VocabParallelCrossEntropy.apply` 的薄封装——两者不是并列的两种写法，而是「通用版」和「融合版」。

通信顺序：先对局部 `logits_max` 做 MAX all-reduce 得全局 max，再对 `[predicted_logits, sum_exp_logits]` 做一次 SUM all-reduce，最后本地算

$$
L = \log \sum_j e^{x_j - m} - (x_t - m)
$$

`m` 是跨 TP 的全局 max，所以各 rank 的局部求和能安全地先减后加；loss 不需要再 all-reduce 一次，因为分母已经是全局和。

## 4. 工程细节与坑

**view 必须洗掉。** persistent 档返回的是 view（带 `_base` 字段），会触发 `schedule.py` 里 `deallocate_output_tensor()` 的断言，所以要 `make_viewless_tensor` 洗一遍（`fused_layer_norm.py:143-149`）。

**gamma 可以存成偏移。** `zero_centered_gamma` 时 forward 用 `weight + 1`，参数初始化改成全零（`fused_layer_norm.py:124-133`）：把 gamma 存成围绕 0 的偏移，改善数值稳定性。

**apex 版本差异靠反射兜。** 两档都用 `inspect.getfullargspec` 探测 apex 是否支持 `memory_efficient` 参数，是跨版本兼容的土办法。

**fallback 不是一句 `F.softmax`。** `forward_torch_softmax`（`fused_softmax.py:299`）里有三个分支。有 `window_size` 就生成滑窗 causal mask；causal 且 `mask is None` 且 `sq > 1` 才现造默认 causal mask，`sq == 1` 故意保留 `mask=None`（decode / KV cache 路径）。

第三支是 `softmax_offset`：传了它就走 `SoftmaxOne`（attention sink 的 off-by-one softmax），这条路径永不融合。

**两条会直接报错的前置断言。** `forward` 断言输入是 4 维；构造函数断言 `self.scale is None or softmax_in_fp32`（`fused_softmax.py:217`），即只要带了 scale，softmax 就必须在 fp32 下算。

**CE 默认关闭，而且梯度 dtype 写死。** `cross_entropy_loss_fusion` 默认 False；打开后还要 `cross_entropy_fusion_impl == 'native'`（默认值）才落到这个文件，`'te'` 走 TE，`'linear'` 把 CE 融进输出层 GEMM。baseline 的 `calculate_gradients` 还把梯度硬转 `torch.bfloat16`（`fused_cross_entropy.py:82`）。

## 5. 小结

- 融合的收益上限由 memory-bound 程度决定；它省带宽和 launch，不省 FLOPs。
- 手写 kernel 分三条路线：预编译 CUDA 扩展（softmax）、apex（layer norm）、Triton（mrope 等）；其余多是 `@jit_fuser` 编译的逐点算子。
- `FusedLayerNorm` 的 persistent 白名单是 24 个具体值；不在名单只退到非 persistent 档，两档都要求装 apex。
- `is_kernel_available` 除六个开关还有一层 `batch_per_block` 整除检查；`softmax_offset` 一出现必定 fallback。
- `fused_cross_entropy` 的价值是把两次 TP all-reduce 合成一次，且默认关闭。

下一篇讲 **强化学习**：rollout 采样、GRPO 的组内归一化，怎么和既有训练循环拼起来，也就是《[强化学习]({{< relref "megatron-code-13-rl" >}})》。

---

**版本提示（2026-10）**：dev 给 `is_kernel_available` 补了一条 `self.window_size is None`（apex kernel 不支持滑窗），并把 CE 的 `jit_fuser` 拆成 gather / exp / scale 三段、去掉写死的 bf16 转换。

`fusions/` 新增 `fused_gated_norm.py`、`fused_row_copy.py`、`fused_mtp_prefix.py`、`fused_mtp_tv.py`、`fused_moe_residual_add.py`。「autograd.Function + 条件降级」的范式未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
