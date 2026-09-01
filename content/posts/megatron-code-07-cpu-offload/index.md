---
title: "Megatron 源码精读（七）：CPU offload"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "cpu-offload", "显存优化"]
categories: ["训练框架"]
weight: 7
series: ["megatron-code"]
---

上一篇[《与 Transformer Engine 的关系》]({{< relref "megatron-code-06-transformer-engine" >}})结尾留了一个钩子：`get_cpu_offload_context` 如何包装 TE 的 offload、双缓冲和 pin memory 怎么把 PCIe 带宽压榨到极致。本篇把这个命题展开成完整的三块，讲清 Megatron Core（下称 MCore）里**三种不同粒度的 CPU offload**：

1. **优化器混合更新**（`HybridDeviceOptimizer`）：把一部分参数的 optimizer step 挪到 CPU 上算；
2. **优化器状态分块下放**（`ChunkedOptimizerStateOffloader`）：state 放 CPU、step 仍在 GPU，按 chunk 换入换出；
3. **流水线细粒度激活下放**（`fine_grained_activation_offload.py`）：前向保存的激活张量下放到 CPU，反向再取回。

三者的粒度从「参数」到「参数的状态」再到「激活张量」逐级变细，但共享同一套硬骨头：**PCIe 带宽有限，所以要靠双缓冲、pin memory、独立 CUDA stream + event 来把传输和计算重叠起来**。先把这个「隐藏延迟」的核心思想立住，后面三块代码就好读了。

---

## 1. 问题背景：为什么会显存不够

显存压力来自两个方向：

- **前向**：不激活重计算（或重计算不足）的情况下，每一层都要把中间激活保存下来留给反向；层数多、batch 大的时候这部分是显存大头。
- **反向/更新**：优化器（Adam 一阶+二阶矩、Muon 的动量、FP32 master weight）的参数状态通常是**参数本身的 2~4 倍**，而它在整个训练里一直常驻 GPU。

MCore 的显存优化谱系里，序列并行（series parallel）削减激活、分布式优化器（DistOpt）分片掉 state 的冗余、重计算用计算换显存；而 **CPU offload 是用「更便宜的 PCIe 传输 + CPU 算力/内存」去换 GPU 显存**。代价是带宽：PCIe 单向几十 GB/s，而 HBM 是 TB/s 量级。所以整篇文章要回答的问题就一个——**怎么在带宽这么紧的情况下，把 offload 的延迟藏起来**。

---

## 2. 三块 offload 的入口与开关

先看这三个机制分别由哪些 flag 打开（`optimizer_config.py:345-386`）：

| 机制 | 开关 | 默认 |
|---|---|---|
| 优化器混合更新 | `--optimizer-cpu-offload` / `--optimizer-offload-fraction` | 关 / 0.0 |
| 优化器状态分块 | `--chunked-optimizer-state-offload` / `--optimizer-state-offload-chunk-size-mb` / `--optimizer-state-offload-fraction` | 关 / 0 / 1.0 |
| 激活下放 | `--cpu-offloading*`（经 `transformer_config` 转到 pipeline） | 关 |

需要特别留意的几条互斥/组合约束（`optimizer_config.py:438-487`）：

- `chunked_optimizer_state_offload` 与 `optimizer_cpu_offload` **互斥**（`optimizer_config.py:451`）——一个把 state 放 CPU 但 step 在 GPU，一个把 state **和 step** 都放 CPU，两者语义冲突；
- 分块 offload 目前只支持 **Adam + Muon**，且 Adam 要求 `use_distributed_optimizer`、Muon 要求 `LayerWiseDistributedOptimizer` + bf16（`optimizer_config.py:475-487`）；
- 分块 offload 不支持 optimizer CUDA graph（`optimizer_config.py:455`）。

Deprecated 的 `--offload-optimizer-states`（`optimizer_config.py:385`）现在只是 `chunked_optimizer_state_offload` 的别名拼写（`optimizer_config.py:418-429`），解析时发 `FutureWarning` 并把它翻译成新字段。

---

## 3. 优化器混合更新：`HybridDeviceOptimizer`

### 3.1 现象与设计目标

`HybridDeviceOptimizer`（`hybrid_optimizer.py:14`）字面意思是「混合设备优化器」：把整个模型的参数组切成**一部分在 GPU 上 step、一部分在 CPU 上 step**。用户通过 `offload_fraction`（0~1）控制多大比例的（GPU 参数）挪到 CPU 更新。它支持 bf16 混合精度，并且实现了 D2H（梯度 GPU→CPU）和 H2D（参数 CPU→GPU）的**重叠**来遮带宽延迟。

### 3.2 根因：step 前要把两大状态搬来搬去

想想一个朴素的「CPU 上做 Adam 更新」需要什么：

1. 反向算完，梯度在 GPU 上 → 要拷贝到 CPU；
2. CPU 上做 Adam，需要参数（至少要参数的一份 CPU 拷贝）和 state；
3. 更新完，权重要拷回 GPU 给下一轮前向。

如果这些都串行、且每次现分配现拷贝，那 PCIe 传输就成了纯纯的 latency 大头。所以 `HybridDeviceOptimizer` 做了三件抵消的事：**按 fraction 拆分参数组**、**pin memory 的 CPU 拷贝**、**专门的 d2h/h2d stream 去重叠**。

### 3.3 解法一：参数组的拆分

核心函数 `_get_sub_optimizer_param_groups`（`hybrid_optimizer.py:251-300`）干这件事：

```python
offload_threshold = gpu_params_total_numel * offload_fraction
offload_params_numel = 0
...
for param in group["params"]:
    cpu_copy = False
    if offload_params_numel < offload_threshold and param.is_cuda:
        param = param.detach().clone().cpu().pin_memory()   # 关键一行
        offload_params_numel += param.numel()
        cpu_copy = True
    ...
    if param.is_cuda:
        gpu_group["params"].append(param)
    else:
        cpu_group["params"].append(param)
```

- 阈值 `offload_threshold = gpu_params_total_numel * offload_fraction`（`hybrid_optimizer.py:258`）按**元素个数**算，不是按字节，因为这里参数 dtype 大体一致；
- 对每个「要被下放」的 GPU 参数，执行 `param.detach().clone().cpu().pin_memory()`（`hybrid_optimizer.py:274`）——`pin_memory` 是这次 offload 效率的基石：pinned（锁页）内存才能 `non_blocking=True` 地异步拷贝，否则 PCIe DMA 每次都要先做一次页锁定；
- 结果得到 `cpu_param_groups` / `gpu_param_groups`，以及两张双向映射 `gpu_params_map_cpu_copy`（GPU 参数 ↦ 它的 CPU 拷贝）和 `cpu_copys_map_gpu_param`（反向）。

拆分发生在 `_init_sub_optimizers`（`hybrid_optimizer.py:181`），它在构造时和 `load_state_dict` 之后都会跑一次（`hybrid_optimizer.py:437`），保证 checkpoint 恢复后重新生成 CPU/GPU 参数配对。

### 3.4 解法二：用原生 Torch optimizer 组合

拆完参数组后，`HybridDeviceOptimizer` 内部持有**一个 GPU 优化器 + 一列 CPU 优化器**（`hybrid_optimizer.py:203-214`）：

```python
if self.overlap_cpu_optimizer_d2h_h2d:
    self.cpu_optimizers = self.build_cpu_optimizer_list(self.cpu_optimizer_cls, self.cpu_param_groups)
elif len(self.cpu_param_groups) > 0:
    self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]
if len(self.gpu_param_groups) > 0:
    self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
```

注意 `build_cpu_optimizer_list`（`hybrid_optimizer.py:227-249`）在重叠加速模式下**把每个参数单独放进一个独立 CPU optimizer**。为什么？因为要重叠 D2H/H2D，就需要「多个小的 CPU step 单元」能彼此错峰：某个 optimizer 的梯度还在 D2H，另一个 optimizer 已经在 step、再一个已经在 H2D。若所有 CPU 参数塞进一个 optimizer，step 就成了串行的一坨。

`sub_optimizers` 这个 property（`hybrid_optimizer.py:468-475`）把「这列 CPU optimizer + GPU optimizer」统一暴露给后续钩子。

### 3.5 解法三：step 的编排与梯度/参数搬运

`step()`（`hybrid_optimizer.py:150-179`）是核心编排，分四步（按注释）：

1. `_sync_hdo_param_groups_to_sub_optimizers()`：把外层 HDO 的 param_groups（lr、wd 等可能被 scheduler 改过）同步进子 optimizer（`hybrid_optimizer.py:334-355`）；
2. `_set_sub_optimizer_grads()`：在 d2h stream 上把 GPU 梯度搬到 CPU（`hybrid_optimizer.py:83-115`）；
3. 依次 `gpu_optimizer.step()` 和每个 `cpu_optimizer.step()`；
4. `_sync_sub_optimizers_state_to_hdo()`：把子 optimizer 的 state 汇回 HDO。

梯度搬运 `_set_sub_optimizer_grads`（`hybrid_optimizer.py:83-115`）的精髓在 `non_blocking` 拷贝 + 事件同步：

```python
for optimizer in self.cpu_optimizers:
    for param in _param_generator(optimizer):
        gpu_param = self.cpu_copys_map_gpu_param[param]
        grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
        ...
        self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)
    self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()
```

每个 CPU optimizer 搬完自己的梯度后，在 d2h stream 上**记录一个 event**。等真正要 step 这个 optimizer 时，才 `d2h_event.synchronize()`（`hybrid_optimizer.py:170-174`）——也就是**延迟等待到最晚一刻**，让前面的 optimizer 的 step 和后面的 D2H 拷贝重叠起来。

参数搬回 GPU 靠 `register_step_post_hook`（`hybrid_optimizer.py:117-148`）：每个 CPU optimizer step 完立刻在 h2d stream 上把更新后的参数 `copy_` 回 GPU（`hybrid_optimizer.py:120-125`），然后 `record_event().wait(current_stream)` 保证主 stream 用到前已就绪。这个「post hook 里立刻回拷」就是 README 里反复强调的 `--overlap-cpu-optimizer-d2h-h2d` 的落地。

### 3.6 fp32 master weight 的处理

`param_update_in_fp32`（`hybrid_optimizer.py:51`）开启时，非 fp32 参数会额外 clone 一份 fp32 拷贝 `param.detach().clone().float()`（`hybrid_optimizer.py:277-279`），这张 `param_to_fp32_param` 映射在 sync state 时回写 `fp32_param.data.copy_(v["master_param"])`（`hybrid_optimizer.py:369-377`）。

还有个容易踩的坑：`HybridDeviceOptimizer` 的 `zero_grad`（`hybrid_optimizer.py:443-454`）额外清了 `decoupled_grad`——这是配合 TE 的 decoupled weight decay 用的独立梯度字段，普通 `torch.optim` 不管这个，漏掉会导致残差梯度残留。

---

## 4. 优化器状态分块下放：`ChunkedOptimizerStateOffloader`

### 4.1 现象与设计目标

和上一节不同，`ChunkedOptimizerStateOffloader`（`chunked_optimizer_state_offload.py:57`）**把 master weight 和 state 放 CPU，但 step 仍然在 GPU 上跑**。外部 optimizer（DistOpt 等）完全不感知 offload 的存在——它看到的参数、state 都是普通 CUDA tensor。这是它最大的卖点（文件头 docstring `chunked_optimizer_state_offload.py:3-10`）。

它解决的问题：Adam/Muon 的 state 可能是参数的几倍大，但 **step 时并不需要一次性把所有 state 都放 GPU**——可以按「chunk」把一小组参数的 state 换入 GPU、step 完换出。这样 GPU 上常驻的只有 `chunk_size` 大小的 state 窗口 + 完整的 master weight 窗口。

### 4.2 核心数据结构

两个关键 dataclass：

- `OptimizerStateChunk`（`chunked_optimizer_state_offload.py:31`）：一对「一起换入换出的参数」`params: tuple[Tensor, ...]`；
- `_StateStagingSlot`（`chunked_optimizer_state_offload.py:49`）：一个可复用的 GPU state 窗口，`buffers` 按 `(device, dtype)` 存 buffer。

还有两个常量：

- `_MASTER_PARAM_KEY = "master_param"`（`chunked_optimizer_state_offload.py:24`）；
- `_NON_OFFLOADABLE_STATE_KEYS = frozenset({ _MASTER_PARAM_KEY, "step", "found_inf" })`（`chunked_optimizer_state_offload.py:25`）——`master_param`（单独走 master 窗口）、`step`（整型 step 计数器）、`found_inf`（梯度溢出标志）这三类**不下放**，要么 CPU 本来就该在 CPU、要么是标量。

### 4.3 选参、估大小、切 chunk

构造时先对「哪些参数下放」做一次规划：

- `_select_params_for_offload`（`chunked_optimizer_state_offload.py:205-226`）：按 `offload_fraction` 算目标字节数 `target_bytes`，然后**优先选「有 master 拷贝」的 bundle**（`candidates = sorted(self._params, key=lambda param: not self._param_has_master(param))`，`chunked_optimizer_state_offload.py:218`）。理由注释写得很清楚：部分字节预算下，先摘掉「state+master」都能腾的 bundle，比只腾 state 的 bundle 划算；
- `_estimated_state_bytes`（`chunked_optimizer_state_offload.py:228-229`）：`param.numel() * (各 state dtype 的 itemsize 和)`，`_state_bytes_per_param` 在 `__init__` 里算好（`chunked_optimizer_state_offload.py:111`）；
- `_build_chunks`（`chunked_optimizer_state_offload.py:231-270`）：贪心装箱，累计字节超过 `chunk_size_bytes` 就切一个新 chunk。**参数是原子单位**——单个参数的 state 超过 chunk 目标时，它成为一个「oversized chunk」，只发 warning 不切分（`chunked_optimizer_state_offload.py:244-269`），因为 Muon 这类矩阵优化器没法对半个参数的 state 构造一个合法参数。

### 4.4 两槽 staging window：传输与计算重叠的关键

`__init__` 里这行注释是理解整个设计的钥匙（`chunked_optimizer_state_offload.py:171-173`）：

```python
# Two reusable state windows allow H2D(N+1), step(N), and D2H(N-1) to overlap
# without letting host run-ahead allocate one CUDA tensor set per chunk.
self._state_staging_slots = (_StateStagingSlot(), _StateStagingSlot())
```

`self._state_staging_slots` 只有**两个** `_StateStagingSlot`，配合 `_next_state_staging_slot` 轮转（`_state_staging_views`，`chunked_optimizer_state_offload.py:712-756`）：

- chunk N 在 GPU 上 step 的同时，
- chunk N+1 正在 H2D（从 CPU 换入下一个槽），
- chunk N-1 刚 step 完正在 D2H（换出上一个槽）。

两个槽的轮转保证同时只有「当前 + 下一个」两份 state 窗口在 GPU，峰值内存被压在 `2 * chunk_size` 附近（README 里也确认了这个 bound）。没有这个双缓冲的话，host 端要么每 chunk 现分配一组 CUDA tensor（allocator 抖动、峰值不可控），要么串行等拷贝。

`_state_staging_views` 的具体做法：给当前 chunk 里每个 `(param, key)` 的 CPU state 在下一个槽的大 buffer 里 `narrow().view()` 出一个 view（`chunked_optimizer_state_offload.py:747-755`），并做 256 字节对齐（`alignment_numel = max(1, 256 // element_size)`，`chunked_optimizer_state_offload.py:727`）。

### 4.5 step 的流水线

`step()`（`chunked_optimizer_state_offload.py:956-1013`）把上一节的双缓冲跑起来：

1. `prefetch_for_step()`（`chunked_optimizer_state_offload.py:785-792`）：异步 H2D 第一个 chunk；
2. 先 step 常驻（resident）参数 `self._resident_params`（`chunked_optimizer_state_offload.py:974-980`）——它们不需要换入换出，顺手在第一个 chunk prefetch 时提供计算，遮住传输；
3. 循环每个 chunk：`_wait_state_prefetch`（等本 chunk H2D 完成）→ **在 step 前发出下一个 chunk 的 prefetch**（`chunked_optimizer_state_offload.py:989-991`）→ `_step_subset`（只对当前 chunk 的参数 step）→ `_schedule_state_d2h`（把本 chunk state 换回 CPU）；
4. 懒初始化补偿：第一次 step 外部 optimizer 可能「惰性」分配 moment，`_has_unregistered_cuda_state` 检测到（`chunked_optimizer_state_offload.py:352-359`）就 `d2h_stream.synchronize()` 把这些一次性分配排空（`chunked_optimizer_state_offload.py:996-1003`）。

关键点：`_step_subset`（`chunked_optimizer_state_offload.py:927-954`）临时把 `optimizer.param_groups` 换成「只含当前 chunk 参数」的子集，step 完再换回 `original_groups`。它同时校验「param_groups 的元数据更新必须与子集无关」——因为同一个 group 的元数据（lr 衰减、step 等）会被多个 chunk 分别 update，若结果不一致会抛 `RuntimeError`（`chunked_optimizer_state_offload.py:944-950`）。

### 4.6 master weight 的全窗口换入换出

master weight 不走 chunk staging，而是**一次全窗口**换入换出（docstring `chunked_optimizer_state_offload.py:65-69`）：

- `_schedule_master_d2h`（`chunked_optimizer_state_offload.py:617-648`）：把 selected master 从 GPU 拷到 pinned CPU，**拷贝一入队就把 `param.data` / `state["master_param"]` 改成 CPU tensor**（canonical binding，`chunked_optimizer_state_offload.py:645-648`），源 tensor 靠 `record_stream` 保活；
- `_schedule_master_h2d`（`chunked_optimizer_state_offload.py:650-691`）：step 前把 master 换回 GPU，并记录 `_master_h2d_event`；
- `_wait_master_h2d`（`chunked_optimizer_state_offload.py:693-698`）：主 stream 等这个 event。

`assert_master_weights_resident`（`chunked_optimizer_state_offload.py:606-615`）是给「外部读 master」的守卫——只要 master 当前是 CPU 绑定或 H2D 未完成，就拒绝外部读取，防止读到半写状态。

`offload_for_forward`（`chunked_optimizer_state_offload.py:808-830`）是训练主循环「optimizer→forward 边界」处调用的：把刚 step 完仍常驻的 state/master 换回 CPU，且 `offload_master=False` 时只下放 state、保留 master 给 MXFP8 参数 staging 读（`chunked_optimizer_state_offload.py:816-819`）。

### 4.7 为什么依赖 distributed checkpoint

README 反复强调 chunked offload **要求 distributed checkpoint**（README.md:86-96）：因为它的 sharded-state 构造能让 offloader「一次只初始化一个 chunk 的状态」，`initialize_state_for_loading`（`chunked_optimizer_state_offload.py:412-464`）正是这么办的——每个 chunk 初始化完立刻 `_schedule_state_d2h` + `synchronize` + `adopt`（`chunked_optimizer_state_offload.py:449-457`），峰值的临时 CUDA state 被压在 chunk 内。旧的 torch checkpoint 路径会把完整 state 一次性重建到 CUDA，所以不支持。

而 `load_state_dict_without_device_cast`（`chunked_optimizer_state_offload.py:466-582`）这个名字已经说明一切：PyTorch 原生的 `load_state_dict` 会把每个 state tensor cast 到参数所在 device，等于把刚下放的 state 又整套搬回 CUDA。这里手动复刻了 id 重映射和 param_groups 恢复，但**跳过 device cast**，保住 CPU pinned 的 canonical state（`chunked_optimizer_state_offload.py:466-475` 的 docstring 讲得很白）。

---

## 5. 细粒度激活下放：`fine_grained_activation_offload.py`

### 5.1 现象与设计目标

前两块都在搞优化器，这一块专攻**前向激活**。思路是：前向时用 PyTorch 的 `saved_tensors_hooks` 把「本来要留在 GPU 上等反向」的激活张量，在层组（layer group）粒度**整体换到 CPU**；反向用到时再从 CPU 取回。粒度是「层组」而不是单层，因为按组批量 D2H/H2D 能摊薄 launch 开销、更好 overlap。

### 5.2 抓手：`saved_tensors_hooks`

整块的核心钩子是 `PipelineOffloadManager.__init__` 里的这行（`fine_grained_activation_offload.py:463-465`）：

```python
self._saved_tensors_hooks = saved_tensors_hooks(
    self.on_save_for_backward, self.on_get_saved_tensor
)
```

`saved_tensors_hooks` 是 PyTorch autograd 提供的 hook：前者在 autograd 保存 tensor 时被调（可返回一个轻量「tag」替代真 tensor 存进图），后者在反向取回时被调（用 tag 把真 tensor 找回来）。于是：

- `on_save_for_backward`（`fine_grained_activation_offload.py:807-814`）`return self.cur_forward_chunk().tensor_push(tensor)` —— 把 tensor 交给当前 forward chunk，返回一个 tag；
- `on_get_saved_tensor`（`fine_grained_activation_offload.py:816-822`）`return self.cur_backward_chunk().tensor_pop(saved_state)` —— 用 tag 取回 tensor（若已被 offload 则 reload）。

`__enter__` / `__exit__`（`fine_grained_activation_offload.py:779-805`）用 context manager 包住要 offload 的 scope，进入时 `saved_tensors_hooks.__enter__()`、并把 TE 的 `CPUOffloadEnabled` 置 True，退出时反向。

### 5.3 tensor 的 tag 与 push/pop

`tensor_push`（`fine_grained_activation_offload.py:960-970`）给 tensor 发一个 `(group_index, tensor_count)` 的 tag，塞进当前 group 的 `offload_groups[group_index-1]`。`tensor_pop`（`fine_grained_activation_offload.py:972-984`）根据 tag 把 tensor 从 group 里 pop 出来；如果存的是个 tuple（说明已被 offload），就 `reload` 回 GPU。

`tensor_need_offloading_checker`（`fine_grained_activation_offload.py:986-1002`）是一串过滤：非 CUDA、是 `Parameter`（权重不能这样 offload）、是 FakeTensor/FunctionalTensor、TE 标记了 `_TE_do_not_offload`、元素数小于 `min_offloaded_tensor_size` 的都不下放——小 tensor offload 不划算，反而多一次 PCIe 往返。

### 5.4 offload/reload 与 pin memory

`ChunkOffloadHandler.offload`（`fine_grained_activation_offload.py:831-847`）：

```python
if use_cpu_pool:
    cpu_backup = self.cpu_tensor_pool.allocate(src_tensor.shape, dtype=src_tensor.dtype)
else:
    cpu_backup = torch.empty(..., device="cpu", pin_memory=pin_memory)
cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
```

`reload`（`fine_grained_activation_offload.py:849-861`）是对称的：`torch.empty(..., device=dev)` 建 GPU tensor，再 `copy_(cpu_backup, non_blocking=...)`。注意 `non_blocking` 默认取 `cpu_backup.is_pinned()`——**pinned memory 是异步拷贝的前提**，这一条和 `HybridDeviceOptimizer` 里 `.cpu().pin_memory()` 是同一件事。

CPU 侧有个 `OffloadTensorPool`（`fine_grained_activation_offload.py:115`）：按 `(shape, dtype)` 分池、用 `deque` 做 O(1) 的分配/归还（`fine_grained_activation_offload.py:172-267`），避免每 iteration 现分配 pinned CPU 内存。pinned 内存分配本身较贵（要锁页），池化复用是关键优化。

### 5.5 组粒度的事件同步与「margin」

`bulk_offload_group`（`fine_grained_activation_offload.py:1004-1028`）在 d2h stream 上把一个 group 的所有 tensor 批量 offload，并在结尾 `record_offload_event`。`bulk_reload_group`（`fine_grained_activation_offload.py:1038-1058`）在 h2d stream 上做对称操作，reload 前先 `wait_offload_event`（保证 offload 完成才能读 CPU 数据），reload 后 `record_reload_event`。

反向消费前 `on_group_commit_backward`（`fine_grained_activation_offload.py:1150-1170`）会 `wait_reload_event` 确保数据就位。

`post_warmup_callback`（`fine_grained_activation_offload.py:566-650`）里有个叫 `_offload_margin` 的巧思：**最后一个同名的 group 不下放**（`fine_grained_activation_offload.py:583-595`），因为「马上要被反向用到」的 tensor 再下放只会白白阻塞主 stream。另外它还会按 `delta_offload_bytes_across_pp_ranks` 在各 PP rank 间做负载均衡，并按 `activation_offload_fraction` 关掉末尾一部分 group（`fine_grained_activation_offload.py:597-627`）。

### 5.6 用 autograd.Function 把 offload 缝进计算图

offload/reload 的调度点靠三个 `torch.autograd.Function` 插进图里：

- `FineGrainedOffloadingGroupStartFunction`（`fine_grained_activation_offload.py:1301-1323`）：forward 里 `on_group_start_forward`（起始组、递增 group index），backward 里 `on_group_start_backward`（触发 reload）；
- `FineGrainedOffloadingGroupCommitFunction`（`fine_grained_activation_offload.py:1219-1249`）：forward 里 `on_group_commit_forward`（批量 offload），backward 里 `on_group_commit_backward`（同步 reload）；
- `FineGrainedOffloadingBackwardRecordFunction`（`fine_grained_activation_offload.py:1334-1355`）：CUDA graph replay 下，连接 TE graph stream 与 reload stream 的 event 同步。

公共入口 `fine_grained_offloading_group_start` / `_group_offload`（`fine_grained_activation_offload.py:1252-1331`）是模型层代码实际调用的函数，它们把「没有 chunk 可管理」时退化为直接返回原 tensor，保证关掉 offload 时零开销。

CUDA graph 是这块最复杂的分支：capture/replay 阶段 tensor 形状已知，用 `_cached_chunks_forward/backward` 缓存 chunk；replay 时 TE 的 graph 在独立 stream 上跑，需要 `cuda_graph_stream`/`cuda_graph_event` + `delay_offload`（`fine_grained_activation_offload.py:1230-1237`）把 D2H 延迟到 replay 返回后、CPU 调度可重叠的间隙里执行。

### 5.7 warmup 模式：先跑一遍摸清形状

注意 `is_warmup` 的分支（`fine_grained_activation_offload.py:1073-1074`、`1181-1184`）：warmup 阶段（第一个 iteration）会把所有 tensor 都记录到 `offload_groups`、统计字节数，`post_warmup_callback` 之后才真正开始「选择性 offload」。这是 CUDA graph 的经典套路——第一次跑 capture 并摸清所有 tensor 的形状、数量、名字，之后 replay 才能确定性地复用池、缓存 chunk。

---

## 6. 小结

- **三种粒度，一条主线**：`HybridDeviceOptimizer`（参数级 CPU step）、`ChunkedOptimizerStateOffloader`（state 级分块换入换出）、细粒度激活下放（激活张量级），都是「用 PCIe 换显存」，主线是**遮带宽延迟**。
- **三个遮延迟的通用手段**：`pin_memory` + `non_blocking` 异步拷贝；独立 d2h/h2d CUDA stream + event 延迟同步；双缓冲/池化让「拷贝(N±1)」和「计算(N)」重叠。
- **优化器混合更新**：`_get_sub_optimizer_param_groups`（`hybrid_optimizer.py:251`）按 `offload_fraction` 拆参数组；重叠模式下每个参数一个独立 CPU optimizer（`build_cpu_optimizer_list`，`hybrid_optimizer.py:227`）；post hook 里立刻 H2D 回拷。
- **状态分块 offload**：step 仍在 GPU；`_state_staging_slots` 两个槽（`chunked_optimizer_state_offload.py:173`）实现 H2D(N+1)/step(N)/D2H(N-1) 重叠；master 走全窗口换入换出；强依赖 distributed checkpoint。
- **激活下放**：抓手是 `saved_tensors_hooks`（`fine_grained_activation_offload.py:463`）；层组粒度批量 D2H/H2D；`OffloadTensorPool` 复用 pinned CPU 内存；warmup 摸形状 + CUDA graph 缓存 chunk。

下一篇《数据并行与 DistributedDataParallel》回到并行切分的主线，讲 Megatron 的 DP 实现、梯度 all-reduce 的时机与 `distrib_optimizer` 如何在 DP 之上继续分片优化器 state（正好接上本篇第 4 节没展开的 DistOpt 内部）。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`、`megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py`、`megatron/core/pipeline_parallel/fine_grained_activation_offload.py`、`megatron/core/optimizer/optimizer_config.py`。）
