---
title: "Megatron 源码精读（十）：checkpoint 处理"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "checkpoint", "分布式"]
categories: ["训练框架"]
weight: 10
series: ["megatron-code"]
---

上一篇[《数据集处理》]({{< relref "megatron-code-09-dataset" >}})把数据喂进训练循环；本篇讲另一头：一个被 TP/PP/EP 切成几十上百个 shard 的模型，怎么落到磁盘、又怎么在换了并行布局之后读回来。

一句话结论：**MCore 把 checkpoint 拆成三个正交维度——目录布局编码并行 rank，`CheckpointType` 与 `ckpt_format` 决定序列化抽象，加载侧再用「嗅探格式 + 结构比对 + 条件恢复 RNG」兜底。**

## 1. 为什么不能每 rank 存一份 `state_dict`

一个直接的问题是「找文件」：恢复时 rank 0 得猜历史 check 用了 TP 还是 PP、EP，命名规则随并行配置变，猜错就找不到。

另一个更隐蔽：DP 组内每个 rank 各存一份模型权重是等价的浪费，而分布式优化器的 state 与模型参数压根不是同一套坐标系，靠「文件名带 rank」无法保证恢复时能对齐。

## 2. 布局：`iter_%07d/mp_rank_TP_PP[_EP]/model_optim_rng.pt`

`get_checkpoint_name`（`checkpointing.py:203`）先把目录定成 `iter_{iteration:07d}`，`release=True` 时换成 `release`。

然后逐段拼文件名：先 `mp_rank_{tp:02d}`；有 PP 时追加 `_{pp:03d}`；有 EP 时再追加 `_{ep:03d}`；最后接 `model_optim_rng.pt`。**后缀是条件追加的，缺哪段就没有那段下划线**——`mp_rank_00` 和 `mp_rank_00_000` 是两个不同的名字，别按固定宽度去解析。

`return_base_dir=True` 会提前返回 `.../iter_NNNNNNN`（`:218-220`）。分布式 checkpoint 要的就是这个目录；只有 legacy 才需要下钻到具体文件。

`find_checkpoint_rank_0`（`:273`）在恢复时并不知道历史的并行配置，于是按四种命名依次探测：无 PP 无 EP → 无 EP 有 PP → 有 PP 无 EP → 有 PP 有 EP，都不中再试「分布式 checkpoint 的 base dir」。它只探 rank 0 的文件，够判断布局了。

## 3. 格式：`CheckpointType` 是五档，不是三档

`CheckpointType`（`:468`）有 `LEGACY` / `LOCAL` / `GLOBAL` / `TORCH_DCP` / `FSDP_DTENSOR` 五个成员，但保存时只会产出前三个。后两个是**加载侧**的产物：`_get_checkpoint_format` 嗅探到 `.metadata` 目录、且开了 Megatron-FSDP，才判成 `FSDP_DTENSOR`。

看起来是 `use_dist_ckpt` 决定 GLOBAL 还是 LEGACY，其实它是**推出来的**——`update_use_dist_ckpt` 把它设成 `ckpt_format != 'torch'`（`utils/common_utils.py:766`），而 `--use-dist-ckpt` 这个 flag 已经废弃、只剩一条 warning（`arguments.py:1966-1970`）。所以真正要选的只有 `ckpt_format`，它有 `torch` / `torch_dist`（默认）/ `torch_dcp` / `fsdp_dtensor` 四个值；非 GLOBAL 档位一律落回 `'torch'`（`:666`）。

| 档位 | 触发条件 | 落盘方式 |
| --- | --- | --- |
| LEGACY | `--ckpt-format torch` | `torch.save` 一个 `.pt` |
| GLOBAL + torch_dist | 默认 | `dist_checkpointing.save` |
| GLOBAL + torch_dcp | `--ckpt-format torch_dcp` | `torch.distributed.checkpoint.save` |
| GLOBAL + fsdp_dtensor | 同上 + Megatron-FSDP | 先 `preprocess_fsdp_dtensor_state_dict` |
| LOCAL | `--non-persistent-ckpt-type local` | resiliency-ext 的 `MCoreTensorAwareStateDict` |

`save_checkpoint`（`:562`）的分叉就三段：`torch_dist` 走 `dist_checkpointing.save`（`:857`，配 `TorchDistSaveShardedStrategy`）；`torch_dcp` 与 `fsdp_dtensor` 走 `torch.distributed.checkpoint.save`（`:922`）；剩下的进 else，LOCAL 交给 resiliency-ext，LEGACY 才 `torch.save`。

## 4. 两种序列化抽象：惰性元数据 vs 真实张量

`generate_state_dict`（`:1284`）里一句 if 决定了整份 checkpoint 的形态：`torch_dist` 下调 `sharded_state_dict()`，返回带 sharding 元数据的惰性引用；其余格式调 `state_dict_for_save_checkpoint()`，返回真实张量（`:1309-1313`）。

这层差别解释了一件容易困惑的事：为什么 LEGACY 的优化器 state 能单独落一个文件，而 `torch_dist` 不需要——后者把模型与优化器放进同一张元数据图里统一切分，谁写哪块由元数据决定。

## 5. `distrib_optim.pt` 只在 LEGACY 下单独写

四个条件同时成立才写：`use_distributed_optimizer`、没开 `no_save_optim`、optimizer 不为 None、且 `ckpt_type == LEGACY`（`:709-720`）。文件名由 `get_distributed_optimizer_checkpoint_name` 拼出来，就是主文件同目录下的 `distrib_optim.pt`（`:269-270`）。

为什么要单独存？分布式优化器的 state 是「每个 DP rank 只持一段参数切片」，坐标原点与 `mp_rank` 那套不同。`torch_dist` 有 sharding 元数据可以统一描述，legacy 没有，只好单独一份文件、按 DP rank 对齐。

同一个位置还有个新分支：LayerWise 分布式优化器在 `ckpt_format == 'torch'` 时写 `layer_wise_optimizer_{dp_rank}.pt`，一个 DP rank 一个文件。

## 6. 加载路径：三个守护，各只覆盖一条分支

**格式嗅探**（`_get_checkpoint_format`，`:1606`）：先看 `check_is_distributed_checkpoint` 判 `torch_dist`；再看目录里有没有 `mp_rank_0*` 文件判 `torch`；再看有没有 `.metadata` 判 `torch_dcp`（开了 Megatron-FSDP 则改判 `fsdp_dtensor`）。三者都不中直接 `NotImplementedError`。`auto_detect_ckpt_format` 的作用是让 `load_checkpoint` 在格式不是 `torch_dist` 时也走这条探测（`:2046`），嗅探本身每次都会做。

嗅探发生在 `_load_base_checkpoint` 里，而 `load_checkpoint` 会调它两次：第一次 `rank0=True` 只把元数据读回 rank 0 用来定格式（`:2047`），第二次 `rank0=False` 才真正往各 rank 的 `sharded_state_dict` 里填张量（`:2284`）。所以日志里出现两次 “loading checkpoint” 不是重复加载。

**结构比对**：`check_checkpoint_args`（`:143`）用内部 `_compare` 逐字段比 `num_layers` / `hidden_size` / `num_attention_heads` / `add_position_embedding` 等，不一致直接 assert 失败。

**RNG 条件恢复**：真正执行恢复的点只有一个（`:2435`），条件是「非 release、非 finetune、没开 `no_load_rng`、且 `ignore_rng_state` 为假」。而 `ignore_rng_state` 只在 `torch_dist` 分支里被置真（`:2092-2113`）：`(TP, PP)` 与 checkpoint 不一致，或 checkpoint 里记了 `no_save_rng`。理由很实在——并行布局变了，同 seed 的随机路径也变了，恢复只会制造假的可复现性。

两处不对称都值得记住。`check_checkpoint_args` 全仓库只有一个调用点，在 legacy 装载分支里（`:2328`），且要求 `state_dict` 里有 `args`、不是 finetune；`torch_dist` 不调它，靠 sharding 元数据和 `load_state_dict` 的 key 匹配报错。`ignore_rng_state` 反过来只在 `torch_dist` 里被设置，所以 legacy 换 TP 恢复时不做这个比对、照样恢复 RNG。把三个守护说成「三条路径都有」，两层都错。

## 7. 工程细节与坑

- **LEGACY 的写盘 rank 是两个条件的并集**：`dp_rank == 0 or expt_dp_rank == 0`（`:767-773`）。源码注释交代了原因：dense 与 expert 的并行布局不一致时（例如 TP > EP×ETP），单靠 `dp_rank == 0` 覆盖不全所有 `(tp_rank, ep_rank)` 分片，只有并集能保证每片恰好一个 rank 写。
- **`release/` 不按迭代存**：发布版固定落 `release` 目录，`find_checkpoint_rank_0` 也随之走 release 分支。
- **跟踪文件只存一行**：`latest_checkpointed_iteration.txt` 记录最近一次成功保存的 iteration，`read_metadata` 读它定恢复点，因此恢复不需要扫目录。

## 8. 小结

- **布局 = 三层目录**：`iter_NNNNNNN/mp_rank_TP_PP[_EP]/model_optim_rng.pt`，后缀条件追加，恢复靠 `find_checkpoint_rank_0` 试四种命名。
- **格式 = 五档枚举但三条保存分叉**：`TORCH_DCP` / `FSDP_DTENSOR` 只在加载侧被判定，保存只产出 LEGACY / LOCAL / GLOBAL。
- **两种序列化抽象**：`torch_dist` 用 `sharded_state_dict()` 的惰性元数据，其余用真实张量。
- **`distrib_optim.pt` 只服务 LEGACY**：分布式优化器 state 与模型参数不同坐标系，有元数据时不需要它。
- **加载三道守护不对称**：格式嗅探总在跑；`check_checkpoint_args` 只在 legacy 分支被调用；RNG 的 TP/PP 比对只在 `torch_dist` 分支做。

下一篇讲**优化器**：分布式优化器如何在 DP 维切分 state、`sharded_state_dict` 怎么产生那些惰性张量——正好和本篇「优化器 state 单独落盘」接上，也就是《[优化器]({{< relref "megatron-code-11-optimizer" >}})》。

---
**版本提示（2026-10）**：新增 `docs/user-guide/features/checkpoint-merge.md`，配套 `tools/checkpoint/weighted_merge.py`——纯元数据驱动的多 checkpoint 加权合并，不构造模型，可在无 GPU 机器上跑。`dist_ckpt` 支持 FP8/FP4 param-gather 的「仅推理」流式反量化（`strategies/torch.py` 的 `stream_ckpt_dequant`，#7405）。`torch_dcp` / `fsdp_dtensor` 的加载会先读 checkpoint 元数据里的 TP size 再重定向 sharding，因此也支持跨 TP 恢复；`checkpointing.py` 本身从约 2600 行涨到 3686 行。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
