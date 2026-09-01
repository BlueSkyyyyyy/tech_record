---
title: "Megatron 源码精读（十五）：多模态（LLaVA）实现"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "多模态", "训练框架"]
categories: ["训练框架"]
weight: 15
---

这是「Megatron-LM 源码精读」系列的第十五篇。前面 14 篇把语言模型的单模态训练管线（并行拓扑、重计算、优化器、MoE、蒸馏等）讲完了，本篇转向多模态（Vision-Language Model, VLM），聚焦 Megatron 内置的 LLaVA 实现（分析基准 commit `f713506ce`）：**一张图 + 一段文本，如何变成一条统一的 token 序列喂进语言模型**。核心代码在 `megatron/core/models/multimodal/llava_model.py`（1378 行）、`megatron/core/models/vision/`、`megatron/core/tokenizers/vision/`。

---

## 1. 总览：三个组件拼成一个模型

LLaVA 这类 VLM 的经典结构是「视觉编码器 → 投影层 → 语言模型」。Megatron 的 `LLaVAModel`（`llava_model.py:57-521`）正是这个结构，`__init__` 里组装了三块（`llava_model.py:222-403`）：

| 组件 | 类 | 来源 |
|---|---|---|
| 语言模型 | `GPTModel` 或 `HybridModel` | `llava_model.py:244-261` |
| 视觉编码器 | `CLIPViTModel` / `RADIOViTModel` / HF 模型 | `llava_model.py:289-365` |
| 投影层 | `MultimodalProjector` | `llava_model.py:381-387` |

还有一块可选的音频分支：`sound_model` + `sound_projection`（`llava_model.py:168-169`），本篇以图像为主、音频在末尾带一句。

语言模型根据 `language_model_type` 决定：`nemotron5-hybrid` / `nemotron6-moe` 走 `HybridModel`（`llava_model.py:223-242`），其余走普通的 `GPTModel`（`llava_model.py:244-261`）。视觉编码器根据 `vision_model_type` 分派：`clip/siglip/internvit` 走 `CLIPViTModel`（`llava_model.py:278-300`），`radio*` 走 `RADIOViTModel`（`llava_model.py:301-359`），`hf://` 前缀则透传到 HuggingFace 加载（`llava_model.py:360-365`）。

## 2. 视觉编码器：CLIP ViT 的 Megatron 化

`CLIPViTModel`（`clip_vit_model.py:26-202`）是标准的 ViT，分三步：

1. **patch 卷积** `conv1`（`clip_vit_model.py:115-122`）：`Conv2d(3, hidden, kernel=patch_dim, stride=patch_dim)` 把图像切成 patch 并升维。
2. **class token + 位置编码**（`clip_vit_model.py:124-139`，forward 里 `clip_vit_model.py:177-192`）：可选的 `class_token` 拼在序列最前，再加上可学习的 `position_embeddings`。
3. **Transformer 堆栈** `self.decoder = TransformerBlock(...)`（`clip_vit_model.py:147-154`）：复用了语言模型同一套 layer spec 机制，`pre_process=True, post_process=False`。

forward（`clip_vit_model.py:164-202`）就是 `conv → reshape/permute → 拼 class token → 加位置编码 → LN → Transformer → LN_post`。值得注意的是 `siglip` 子树没有 class token（`clip_vit_model.py:61-63`），`internvit` 用 `conv_bias`。

ViT 内 seq_len 的计算公式（`clip_vit_model.py:85`）：

```
seq_length = (img_h/patch_dim) * (img_w/patch_dim) + (class_token_len if add_class_token else 0)
```

## 3. 投影层：把视觉维度对齐到语言维度

`MultimodalProjector`（`multimodal_projector.py:15-91`）就是把 vision encoder 的 `hidden_size` 映射到 language model 的 `hidden_size`。支持两种类型（`multimodal_projector.py:47-65`）：

- `mlp`：一个完整 `MLP`（含激活）
- `affine`：单个 `linear_fc1` 线性层

投影层的关键约束是**走 TP**——`tp_group=self.pg_collection.tp`（`llava_model.py:386`），因为语言模型是 TP 切分权重，投影输出的并行度要与之对齐。forward 里对输出做 `make_viewless_tensor`（`multimodal_projector.py:87-89`）是为了配合 pipeline schedule 的 `deallocate_output_tensor`。

## 4. 核心：`_preprocess_data` —— 把图像 token 拼进文本序列

这是 LLaVA 实现最精妙的部分（`llava_model.py:494-759`）。docstring 用一个小例子说清了输入输出约定（`llava_model.py:518-523`）：

```
input_ids = [0, 1, -200, 2, 3]     # -200 是 <image> token 占位
labels    = [1, -200, 2, 3, 4]
-- 处理后 --
final_embeddings = [0, 1, image_embeddings, 2, 3]   # image_embeddings 长 img_seq_len
final_labels     = [1, -100, 2, 3, 4]               # 图像位置 label 用 -100 忽略
final_loss_mask  = [1, 0, 0, 1, 1]                  # 图像位置不参与 loss
```

`IMAGE_TOKEN = "<image>"`，默认 `DEFAULT_IMAGE_TOKEN_INDEX = -200`（`llava_model.py:47-53`）。一个关键概念是 `image_token_index` 只是**占位符**：文本被 tokenize 后，`<image>` 位置用一个负数 index 标记，forward 时把它替换成真正的 `img_seq_len` 个视觉 embedding。

处理流程（`_preprocess_data` 内部）：

1. 找出 `input_ids == image_token_index` 的位置（`llava_model.py:575`）。
2. 按 `num_image_tiles` 统计每样本的图像块数，算出拼图后的 `seq_len`（`llava_model.py:579-595`）。
3. 用 `cumsum` 生成新的 `position_ids`，把文本 token 的位置往后挪 `img_seq_len` 格（`llava_model.py:600-607`）。
4. 构造 `images_mask`，标出哪些位置放图像 embedding（`llava_model.py:624-637`）。
5. 组装 `final_embedding`（pre_process 时）：先填文本 embedding，再把 `image_embeddings` 按 `images_mask` 铺进去（`llava_model.py:640-695`）。
6. 组装 `final_labels`/`final_loss_mask`（post_process 时）：图像位置置 `-100`/`0`（`llava_model.py:697-737`）。

一个细节：label 是**左移一位**的（`llava_model.py:610-622, 716-719`），且图像 token 前紧邻的文本 token 不预测首个图像 token，所以 loss mask 要被清掉（`llava_model.py:724-737`）。

### 4.1 流水线并行下的分工

`_preprocess_data` 对 pipeline 很敏感（`llava_model.py:528-556`）：

- 首个 chunk（`pre_process=True`）：只更新 embedding
- 中间 chunk（都 False）：什么都不做
- 末个 chunk（`post_process=True`）：只更新 label/loss mask

所以「替换图像 token」只在**第一个 stage** 发生，后面的 stage 拿到的是已经拼好的 embedding 序列。

### 4.2 序列/上下文并行的再切分

`_process_embedding_token_parallel`（`llava_model.py:761-904`）在 SP/CP 开启时，把拼好的 `[s,b,h]` 序列按 `tp_size`（SP）或 `cp_size*2`（CP）切成 shard，先 pad 到可整除（`llava_model.py:802-853`），再 `scatter_to_sequence_parallel_region`（`llava_model.py:899-902`）。这里的坑是：**文本 + 视觉 token 混合后的总长不天然对齐 shard_factor**，所以要手动 pad（`llava_model.py:804-849`）。

## 5. `forward` 主流程：一条龙

`forward`（`llava_model.py:942-1290`）串起了全部：

1. **视觉编码**（`llava_model.py:1012-1192`）：`vision_model(images)` 得 `[num_tiles, img_seq_len, h_vision]`。若开 `pixel_shuffle` 则先做像素重排（`llava_model.py:1119-1128, 1152-1168`，实现见 `pixel_shuffle` 函数 `llava_model.py:1342-1378`，源自 InternVL，把 patch 数降到 1/4、维度 ×4）。
2. **投影**（`llava_model.py:1176-1178`）：`vision_projection(image_embeddings)` → `[img_seq_len, num_tiles, h_language]`。
3. **tile tagging**（可选，`llava_model.py:906-940`）：给每个 tile 前拼 `<tile_N>` 标签 token（NVLM 的做法）。
4. **文本编码**（`llava_model.py:1233-1246`）：`language_model.embedding(input_ids_text)`，其中 image token index 先被替换成 0（`llava_model.py:1236`）。
5. **拼接**（`llava_model.py:1256-1270`）：调 `_preprocess_data` 把图像 embedding 插进文本序列。
6. **SP/CP 切分**（`llava_model.py:1272-1277`）：必要时分发到各 rank。
7. **语言模型前向**（`llava_model.py:1279-1288`）：`language_model(decoder_input=combined_embeddings, labels=new_labels, ...)`，拿 logits 或 loss。

> 注意视觉分支和文本分支的并行切分是**分开**的：视觉模型走自己的 `pg_collection`（通常只在第一个 PP stage 存在，`add_encoder` 控制，见 docstring `llava_model.py:80-83`），文本模型走语言模型的并行配置。两者通过投影层衔接。

## 6. 图像 token 数量：`get_num_image_embeddings`

`get_num_image_embeddings`（`clip_vit_model.py:205-260`）回答了「一张图占几个 token」：

```
num_patches = (img_h/patch_dim) * (img_w/patch_dim)
per_tile = num_patches + (class_token_len if keep_class_token else 0)
若 pixel_shuffle: per_tile *= 0.25
若 tile_tags: per_tile += (5 或 6，依 tokenizer 而定)
```

它在 `LLaVAModel.__init__` 里被调用，结果存进 `self.img_seq_len`（`llava_model.py:405-416`），后续 `_preprocess_data` 用它来铺图像 embedding。

## 7. 数据/词表侧：`<image>` 特殊 token

多模态不只是在模型层，词表层也要处理。`megatron/core/tokenizers/vision/` 下的 `MegatronMultimodalTokenizer`（`libraries/multimodal_tokenizer.py:55`）是核心：

- 维护 `<image>` / `<video>` / `<so_embedding>` 等特殊 token（对应 `llava_model.py:51-53`）。
- `tokenize`（`multimodal_tokenizer.py:205`）里 `_apply_image_tag`（`:190`）把对话里的图像标记替换成 image token。
- `convert_tokens_to_ids`（`:309`）把 `<image>` 映射到那个负数 index（`-200`），正是 `LLaVAModel` 里 `image_token_index` 的来源。

## 8. 小结

Megatron 的多模态实现没有发明新机制，而是**把 VLM 拆成「视觉编码器」和「语言模型」两个子网络，复用同一套 Transformer layer spec / 并行拓扑 / pipeline 基础设施**，再用一个投影层把它们缝起来。精华集中在 `_preprocess_data`——它用「负 index 占位 + 位置重排」的方式，把变长的图像 embedding 干净地嵌进定长的文本序列，同时正确维护了 label 左移和 loss mask。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`；涉及文件：`megatron/core/models/multimodal/llava_model.py`、`megatron/core/models/vision/clip_vit_model.py`、`megatron/core/models/vision/multimodal_projector.py`、`megatron/core/tokenizers/vision/libraries/multimodal_tokenizer.py`。）