---
title: "Megatron 源码精读（十五）：多模态（LLaVA）实现"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "多模态", "训练框架"]
categories: ["训练框架"]
weight: 15
series: ["megatron-code"]
---

一张图加一段文本，怎么变成一条统一的 token 序列喂进语言模型？变长的图像 embedding 塞进定长文本序列时，position_ids、labels、loss_mask 各自要怎么挪？

一句话结论：**MCore 的 LLaVA 不是新模型，而是「两个现成模型 + 一个投影层 + 一次 scatter」；全部精巧之处集中在 `_preprocess_data` 的那一次索引写回。**

上一篇讲 MoE 的专家并行，本篇换到另一条支线：把视觉编码器当成一个普通子网络接进语言模型。

---

## 1. 三个组件，靠两个字符串分派

结论：**结构是教科书式的「视觉编码器 → 投影层 → 语言模型」，但每个组件具体实例化成什么，由 `language_model_type` 和 `vision_model_type` 两个字符串决定。**

| 组件 | 实现 | 分派依据 |
|---|---|---|
| 语言模型 | `GPTModel` / `HybridModel` | `nemotron5-hybrid`、`nemotron6-moe` 前缀走 Hybrid |
| 视觉编码器 | `CLIPViTModel` / `RADIOViTModel` / HF 模型 | `vision_model_type` |
| 投影层 | `MultimodalProjector` | `vision_projection_type` |

视觉侧的分派很直白（`llava_model.py:278` 起）：`clip`/`siglip`/`internvit` 走 `CLIPViTModel`；`radio`/`radio-g`/`cradio-g` 走 `RADIOViTModel`（三者图像上限、class token 数、embedder bias 各不相同）；`hf://` 前缀直接交给 `build_hf_model`，其余类型抛 `ValueError`。

语言侧只看前缀：`nemotron5-hybrid` / `nemotron6-moe` 建 `HybridModel`，其余一律 `GPTModel`。

还有一条容易漏掉的约束：开 SP 或 CP 时，语言模型必须用 TE 的 `DotProductAttention`（hybrid 两个类型豁免），CP 还额外要求 TE ≥ 1.10 且 CP 组大小与配置一致（`llava_model.py:189` 附近）。

## 2. 视觉编码器：三个子类型共用一套 ViT

结论：**`CLIPViTModel` 就是标准 ViT，`clip`/`siglip`/`internvit` 的差别只有三处：`ln_pre` 有没有、conv 有没有 bias、有没有 class token。**

结构三段：`conv1` 做 patch 卷积（kernel=stride=patch_dim，`clip_vit_model.py:115`）；可选 class token 拼在序列最前，再加可学习 `position_embeddings`（`:126`）；`self.decoder = TransformerBlock(...)` 复用语言模型那套 layer spec（`:147`）。

差异在于：`clip` 建 `ln_pre`、conv 无 bias、padding 0；`siglip` 只建 `ln_post`、conv 有 bias、padding `valid`，并且**不支持 class token**（`class_token_len` 必须为 0，`clip_vit_model.py:62` 直接 assert）；`internvit` 有 conv bias、无 `ln_pre`。

ViT 内的序列长度就是 `num_patches + class_token_len`（`:85`）。而「一张图占几个 token」由 `get_num_image_embeddings` 单独算，因为还要叠加 `pixel_shuffle`（每 4 个 patch 并成 1 个 token，通道 ×4）和 tile tag（依 tokenizer 加 5 或 6 个位置）。结果存进 `self.img_seq_len`（`llava_model.py:405`），后面铺 embedding 全靠它。

## 3. 投影层：两种类型，且必须挂在语言模型的 TP 组上

结论：**投影层只做维度对齐，但它的并行组必须和语言模型一致，否则 hidden 维对不上。**

两种类型（`multimodal_projector.py:47` 起）：`mlp` 建一个完整 `MLP`（带激活）；`affine` 只建一个 `linear_fc1`（`skip_bias_add=True`，bias 在 forward 里加回去）。

TP 组直接取 `self.pg_collection.tp`（`llava_model.py:386`）：编码器、投影层、语言模型共用同一套 process group，而不是各建各的——因为投影层的下游就是语言模型的 embedding 接口。输出还刻意过了一次 `make_viewless_tensor`，代码注释写明原因：encoder 产出 view 张量时，pipeline schedule 的 `deallocate_output_tensor` 会报错。

## 4. 核心：`_preprocess_data` 的一次 scatter

结论：**图像 token 在文本里只占「一格」负 index 占位，前向时被展开成 `img_seq_len` 个 embedding；所有位置/label/mask 的调整都是为了这次展开。**

docstring 给的例子就是完整契约（`llava_model.py:518`）：

```
input_ids = [0, 1, -200, 2, 3]     labels = [1, -200, 2, 3, 4]
-->
final_embeddings = [0, 1, image_embeddings, 2, 3]   # 长度 img_seq_len
final_labels     = [1, -100, 2, 3, 4]
final_loss_mask  = [1, 0, 0, 1, 1]
```

`-100` 出现在 label 的第 1 格而不是第 2 格，因为 labels 是 input_ids **左移一位**的结果：第 2 格是图像 token，预测它的那一格是第 1 格，所以被抹掉。`final_loss_mask` 里连图像的前一格（文本 token）也是 0——不让文本去预测首个图像 token。

位置重排靠一次 cumsum：图像占位在累加里的步长是 `num_image_tiles * img_seq_len`，文本 token 的步长是 1，前缀和减 1 就是新位置。

$$
\text{pos} = \mathrm{cumsum}(\text{step}) - 1
$$

拿到 `pos` 后先 `final_embedding[b, text_position_ids] = language_embeddings[b, non_image_indices]` 写文本，再用 `images_mask` 布尔索引写图像（`llava_model.py:665`）。

这里藏着一个坑：图像 embedding 的写入顺序是 `image_embeddings.permute(1, 0, 2).reshape(-1, h)`，即按 `[tile, tile 内 token]` 行优先展开；它必须与 `images_mask` 中 True 的行优先顺序严格一致，否则 tile 会错位。

`num_image_tiles` 也在这里消费：先按每个样本的图像 token 数把 tile 数组切段求和，再算 `seq_lens = tiles * img_seq_len - num_images + text_seq_len`（`:587`）。

### 4.1 pipeline 下只做一半

三段分工写得很明确：首段 `pre_process=True` 只更新 embedding；中间段两个 flag 都是 False，直接返回 `None`；末段 `post_process=True` 只更新 labels/loss_mask。所以「替换图像 token」只在第一个 stage 发生。

### 4.2 SP/CP 再切分：shard_factor 有三种组合

结论：**混合序列长度不天然整除，必须先 pad 再切；CP 有两套完全不同的分发路径。**

`shard_factor` 分三种：CP+SP 为 `tp*cp*2`，只 CP 为 `cp*2`，只 SP 为 `tp`（`llava_model.py:793` 起）。整除不了就补零，labels 补 `-100`、loss_mask 补 0。

CP 下：SBHD 布局走通用的 `get_batch_on_this_cp_rank`（zigzag）；THD 布局走 TE 的 `thd_get_partitioned_indices` 取索引再 `index_select`，这条要求 TE ≥ 1.10。

两个顺序细节容易被忽略：CP 分发要求序列在 **dim=1**，所以 `_preprocess_data` 末尾只在 `cp_size == 1` 时把结果转成 `[s, b, h]`；SP 的 `scatter_to_sequence_parallel_region` 又是在 CP 切完之后沿 dim 0 再切一刀。

## 5. forward 的流水线

1. 视觉编码 + 可选 `pixel_shuffle`（`llava_model.py:1012` 起，`pixel_shuffle` 函数在 `:1342`）。
2. 投影：`vision_projection(image_embeddings)`，输出 `[img_seq_len, num_tiles, h_language]`。
3. 可选 tile tagging：给每个 tile 前拼 `<tile_N>` 标签（NVLM 做法，`:906`），当前实现断言只支持单图。
4. 文本 embedding：先把 image token index 换成 0 再 embed（`:1236`），避免负 index 进词表。
5. `_preprocess_data` 拼接 → SP/CP 切分 → `language_model(decoder_input=...)`；注意 `position_ids=None`，位置已经隐含在拼接后的顺序里。

## 6. 词表侧：`<image>` → -200 不在 tokenizer 里

这是一个常见误解。`MegatronMultimodalTokenizer` 只是 HuggingFace tokenizer 的包装，`convert_tokens_to_ids` 直接透传（`multimodal_tokenizer.py:309`），不会产生负数 id；它的真正职责是 prompt 格式化——`_apply_image_tag`（`:190`）把 `<image>` 包成 `<img><image></img>`，前后两个 tag 是词表里的普通 token，占位的只有 `<image>` 一格。

`<image>` → `-200` 只发生在 `MegatronNullMultimodalTokenizer`（预分词数据路径）：它的 `convert_tokens_to_ids` 把 `<image>` 换成 `image_token_id`，默认值才是 `-200`。

模型侧并不关心 id 怎么来的，它只认构造函数传进来的 `image_token_index`（默认 `-200`）；换数据管线时这个值要跟着改。

## 7. 小结

- **结构三层不变**：视觉编码器 + 投影层 + 语言模型，两个字符串决定实例化哪个类。
- **ViT 差异只有三处**：`ln_pre`、conv bias、class token；`siglip` 强制无 class token。
- **投影层挂语言模型的 TP 组**，输出过 `make_viewless_tensor` 以适配 pipeline 的激活释放。
- **`_preprocess_data` 的契约**：负 index 占位、cumsum 重排位置、label 左移后抹掉预测图像的那一格、图像前一格 loss_mask 置 0。
- **SP/CP 前必须 pad**，且 CP 要求序列在 dim=1，因此转置只在 `cp_size == 1` 时做。

---

**版本提示（2026-10）**：上游大幅重构了多模态——新增通用 `vit_model.py`（`ViTModel`/`QwenVLViTModel`/`KimiViTModel`，运行时不依赖 HF）与 `encoder_registry.py`（按 `vision_model_type` 集中管理 patch_dim、图像尺寸、归一化与整套 TransformerConfig），`llava_model.py` 接近重写（2100+ 行）；SP/CP 逻辑抽到 `multimodal/context_parallel.py`，并把 CP 接进了 MIMO（#7758）。「编码器 + 投影层 + LM」的结构未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
