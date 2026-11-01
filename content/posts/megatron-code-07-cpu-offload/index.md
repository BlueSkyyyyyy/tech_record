---
title: "Megatron 源码精读（七）：CPU offload"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "cpu-offload", "显存优化"]
categories: ["训练框架"]
weight: 7
series: ["megatron-code"]
---

上一篇[《与 Transformer Engine 的关系》]({{< relref "megatron-code-06-transformer-engine" >}})结尾提到 `get_cpu_offload_context`，那是 TE 的**层粒度**下放：整层的激活和权重按 layer index 搬到 CPU，TE ≥ 2.10 才支持复用 pinned buffer。

本篇把它放回全景，讲 MCore 自研的三档粒度，每一档的粒度都比上一档细。

**一句话结论：三档共用同一套遮延迟手法——pinned 内存、独立 stream、event 延后同步、双缓冲；省不省得出显存，取决于你愿不愿用 PCIe 带宽和 CPU 算力去换。**

---

## 1. 问题：两处显存大头，一条窄路

显存压力来自两头：前向保存给反向的中间激活，和常驻的优化器 state（fp32 master、Adam 两阶矩、Muon 动量，通常是参数的 2~4 倍）。

HBM 是 TB/s 量级，PCIe 单向只有几十 GB/s。所以 offload 的全部工程难度不在「搬」，而在**别让 GPU 等搬运**。记住这条，下面三节的每个设计都能对号入座。

---

## 2. 三条路，各有各的开关和坑

| 机制 | 开关 | 命令行默认 |
|---|---|---|
| 参数级混合更新 | `--optimizer-cpu-offload` + `--optimizer-offload-fraction` | 关 / 1.0 |
| state 级分块下放 | `--chunked-optimizer-state-offload` + `--optimizer-state-offload-chunk-size-mb` | 关 / 0 |
| 激活级细粒度下放 | `--fine-grained-activation-offloading` + `--offload-modules` | 关 / 空 |

第一个坑就在默认值上：`OptimizerConfig` 里 `optimizer_offload_fraction` 的字段默认是 0.0，但 `arguments.py` 给命令行参数的默认值是 1.0。走命令行时只加 `--optimizer-cpu-offload`，**等于把 100% 的参数都丢到 CPU**。

几条硬约束（`optimizer_config.py` 与 `arguments.py`）：

- `optimizer_cpu_offload` 必须配 `--use-precision-aware-optimizer`（混合优化器复用了那条代码路径），并要求 `decoupled_weight_decay` 为真（默认即是 AdamW 模式）；开 fp8 param gather 时还要求 `--fp8-recipe delayed`；
- 两个 offload 模式**互斥**；分块模式不支持 Megatron-FSDP，也不支持 full-iteration CUDA graph；
- 分块模式只支持 Adam+DistOpt（不能配 LayerWise）或 bf16 Muon+LayerWise（且要关掉 `--use-layer-wise-param-layout`）；
- 保存/加载优化器 state 时分块模式强制 `--ckpt-format torch_dist`，且**不能 `--async-save`**——后台写线程会攥住可复用的 pinned CPU buffer。

---

## 3. 参数级：`HybridDeviceOptimizer`

### 3.1 按元素数拆参数组

`_get_sub_optimizer_param_groups` 逐个参数判断：累计下放的元素数没到 `gpu 参数总数 × offload_fraction` 就下放。

下放的动作是 `param.detach().clone().cpu().pin_memory()`，并维护两张映射：GPU 参数 ↦ 它的 CPU 副本、CPU 副本 ↦ 原参数。`pin_memory()` 是关键——只有锁页内存才能 `non_blocking=True` 地异步 DMA。

这里有个不对称：梯度的 CPU 缓冲按 `pin_memory=self.pin_cpu_grads` 分配，参数这一行却是**硬编码**调 `pin_memory()`，`--no-pin-cpu-params` 只落到 `self.pin_cpu_params` 属性上，没人读它。

### 3.2 为什么每个参数一个 CPU optimizer

重叠模式下，`build_cpu_optimizer_list` 把每个参数单独塞进一个 CPU optimizer。理由不是洁癖：要重叠 D2H/H2D，就需要多个能错峰的小 step 单元——A 在搬梯度、B 在 step、C 在回拷。全部塞进一个 optimizer，step 就退化成一整块串行。

### 3.3 step 的编排：把等待推到最后一刻

`step()` 四步：同步 param_groups（lr/wd 可能被 scheduler 改过）→ 在 d2h stream 上搬梯度 → GPU optimizer 与各 CPU optimizer 依次 step → 把子 optimizer 的 state 汇回外层。

延迟同步藏在两处。搬梯度时，每个 CPU optimizer 搬完就在 d2h stream 上记一个 event；真正要 step 它时才 `event.synchronize()`。于是第一个 optimizer 不必等最后一个的拷贝结束。

回拷则挂在 `register_step_post_hook` 上：每个 CPU optimizer 一 step 完，立刻在 h2d stream 上把更新后的参数 `copy_` 回 GPU，再 `record_event().wait(current_stream)`。注意 h2d stream 先 `wait_stream(current_stream)`：回拷必须排在主 stream 已有工作之后，不能覆盖还在被读的旧参数。

### 3.4 fp32 master 与 `decoupled_grad`

`param_update_in_fp32` 打开时，非 fp32 参数再 clone 一份 fp32 副本，step 后通过 `state["master_param"]` 把值写回。

另一个容易漏的点在 `zero_grad`：它额外清了 `decoupled_grad`。这是 precision-aware 路径下主参数的独立梯度字段——搬梯度时按 `getattr(param, "decoupled_grad", param.grad)` 二选一读取，而原生 `torch.optim` 只认 `.grad`，漏清会留残差。

---

## 4. state 级：`ChunkedOptimizerStateOffloader`

### 4.1 外部优化器完全无感

和上一节相反：这里 **state 和 master 常驻 CPU，step 仍在 GPU 上跑**。外部优化器（DistOpt 等）看到的永远是普通 CUDA tensor，不需要任何 offload 专用 API——这是它最大的卖点，也是它能包住各种优化器的原因。

### 4.2 两块 staging 槽，峰值约 2 倍 chunk

state 按 chunk 换入换出，`_state_staging_slots` 只有两个可复用的槽：chunk N 在算、N+1 在 H2D、N-1 在 D2H。若每 chunk 都新分配 CUDA tensor，峰值就不可控了。

所以 `--optimizer-state-offload-chunk-size-mb` 约束的是**每个槽**：常驻峰值约 2 倍该值。README 还点明一个例外——Muon+Adam 混合链里，LayerWise 与兄弟 DistOpt 各有一组槽，叠加起来可能摸到 3 倍。

参数是**原子单位**：单个参数的 state 超过 chunk 目标时不会被切开，只发 warning。原因有两个：矩阵优化器（如 Muon）没法对半个参数构造合法 state；也不能为 DistOpt shard 的切片凭空造一个优化器参数。

部分下放（fraction < 1）时还会挑肥拣瘦：优先选「带独立 master」的参数，因为同样的字节预算能一次腾掉 state 和 master 两份驻留。

`_step_subset` 另有一条容易被忽略的校验：param_groups 的元数据更新必须与参数子集无关。同一个 group 的 lr、step 计数会被每个 chunk 各更一次，结果不一致就抛 `RuntimeError`；CUDA tensor 型字段只比 shape/dtype/device，以免每 chunk 一次 host 同步。

### 4.3 master 走全窗口

master weight 不参与分块，而是一次全窗口换入换出。`_schedule_master_d2h` 在拷贝入队的一瞬间就把 `param.data` / `state["master_param"]` 改绑成 CPU tensor（CPU 才是 canonical），源张量靠 `record_stream` 保活；`_schedule_master_h2d` 在 step 前换回并记 event。

正因如此，`assert_master_weights_resident` 会拒绝任何「master 还在 CPU 或 H2D 未完成」时的外部读取，防止读到半写状态。

### 4.4 为什么非分布式 checkpoint 不可

分块加载靠 `sharded_state_dict` 的分片构造，才能一次只初始化一个 chunk 的 state，初始化完立刻 D2H 排空。旧的 torch checkpoint 会把完整 state 一次性重建到 CUDA，峰值直接失控。

`load_state_dict_without_device_cast` 解决的是另一半问题：原生 `load_state_dict` 会把每个 state cast 到参数所在 device，等于刚下放的 state 又被整套搬回 GPU。这里手动复刻了 id 重映射与 param_groups 恢复，但**跳过 device cast**。

---

## 5. 激活级：`saved_tensors_hooks` + 层组

### 5.1 用一个 tag 顶替真张量

抓手是 PyTorch 的 `saved_tensors_hooks`：`on_save_for_backward` 不返回张量，而是把张量交给当前 forward chunk、返回一个 `(group_index, tensor_count)` 的 tag；`on_get_saved_tensor` 拿 tag 去 pop，若取出的是 tuple（已被下放）就 reload 回 GPU。

autograd 图里从此只留 tag，真张量在 CPU。

### 5.2 哪些张量不下放

`tensor_need_offloading_checker` 是一串过滤：非 CUDA、`Parameter`（权重不该走这条路）、FakeTensor/FunctionalTensor、TE 标了 `_TE_do_not_offload`、以及元素数小于 `min_offloaded_tensor_size` 的。

最后一条默认值是 1M 元素：小张量下放不划算，一次 PCIe 往返的固定开销可能比省下的显存还贵。

### 5.3 池化、组粒度、事件

CPU 侧用 `OffloadTensorPool` 按 `(shape, dtype)` 分池复用 pinned 内存，避免每 iteration 重新锁页。下放/取回都以「层组」为单位批量进行，`bulk_offload_group` 在 d2h stream 上搬完立刻记 event，reload 前先等这个 event。

分组的实际作用在同步粒度：一个组只记一对 offload/reload event，而不是每个张量一对。反向时等到组级 event，就一次性拿到整组激活。

### 5.4 先 warmup 摸底，再决定下放多少

第一个 iteration 是 warmup，行为很特别：**所有组一律下放**（`should_bulk_offload` 在 warmup 阶段直接返回 True），同时把每组的张量数、字节数记下来。跑完才由 `post_warmup_callback` 决定往后真正下放哪些。

它做三个取舍：把**每个同名的最后一组**标记为不下放（马上要被反向用，下了只会白等）；按 `delta_offload_bytes_across_pp_ranks` 在 PP rank 间挪负载；按 `activation_offload_fraction` 从后往前关掉一部分组（靠前的组早释放，收益更大）。

CUDA graph 让它更绕一层：replay 时 TE 的 graph 跑在独立 stream 上，`delay_offload` 把 D2H 先压进队列，等 replay 返回后再 flush，正好塞进 CPU 调度与通信的空隙。`--delay-offload-until-cuda-graph` 就是为这个场景准备的。

---

## 6. 小结

- **三档粒度，一条主线**：参数级换 step 设备、state 级换 state 驻留、激活级换激活驻留，全靠遮住 PCIe 延迟。
- **四个通用手段**：pinned 内存 + `non_blocking`、独立 d2h/h2d stream、event 延后同步、双缓冲/池化。
- **默认值有陷阱**：命令行的 `--optimizer-offload-fraction` 默认 1.0，只开开关就是全量下放。
- **分块模式的边界**：峰值约 2 倍 chunk（混合链可到 3 倍），master 全窗口，绑 `torch_dist` 且不能异步存。
- **激活下放的关键数字**：默认只搬 ≥1M 元素的张量，最后一组同名 group 不下放。

下一篇[《ZeRO-1 / FSDP 实现》]({{< relref "megatron-code-08-zero-fsdp" >}})回到并行切分主线：`use_distributed_optimizer` 到底切了哪一份数据。

---

**版本提示（2026-10）**：上游给分块 offload 补了两条约束——master 权重按 offload chunk size 受界（#8232）、chained optimizer 的 state staging 约束（#8233）；RL 侧支持用 NCCL UBM/UBR 缓冲 offload 梯度（#6925）。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
