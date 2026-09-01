# Megatron 的多模态模型实现

> 基于本地仓库 `/home/xieminglin/proj/Megatron-LM`（commit f713506cea2e7705dd2ebb00c5c58a046ff974fe）精读。
> 核心结论：Megatron-Core 的多模态主线是 **LLaVA 风格的"encoder 侧外挂"架构**——vision/audio encoder 只活在 PP 第一个 stage，输出经 projector 对齐到 LM hidden size 后，在 embedding 层面把 `input_ids` 里的占位 token（-200）原位替换为图像 embedding，之后就是一个普通的 GPT 前向。真正精妙的部分全在这个"替换"和它在 TP/SP/CP/PP 下的分片上。

## 1. 支持的多模态模型清单

| 模型 | 位置 | 定位 |
|---|---|---|
| **LLaVA** | `megatron/core/models/multimodal/llava_model.py`（1378 行，主线） | ViT + projector + GPT 的经典 VLM，同时支持图像 tile、视频（temporal tubelet）、音频（sound token）。仓库自我标注 "work in progress"（llava_model.py:155） |
| **MIMO** | `megatron/core/models/mimo/`（`model/base.py` MimoModel + `submodules/{vision,audio}.py` + README） | "Multimodal In/Out Model"：理解+生成统一框架。模态子模块负责 encode（模态→embedding）和 decode（embedding→模态），MimoModel 按 special token 位置对齐 embedding（`align_embeddings_by_token_positions`）。experimental |
| **Bagel** | `megatron/core/models/bagel/` | 生成-理解统一模型（对标 ByteDance BAGEL）。核心是 **MoT（Mixture of Transformers）**：`transformer_mot_layer.py` 里理解/生成两条权重流共享注意力；配 `bagel_rope.py`、FlexAttention、扩散头（`examples/mimo_bagel/diffusion/` 里 VAE + diffusion wrapper，loss 是 MSE，见 `bagel_mimo.py` 的 `_native_bagel_squared_error`） |
| **Audio（NeMo）** | `megatron/core/models/audio/` | NeMo 语音模型（FastConformer/Parakeet 系）的接入：audio feature config、audio projector、`.nemo` checkpoint 加载、packed audio embedding。在 LLaVA 里以 `sound_model`/`sound_token_index=-300` 出现 |
| **HF 封装** | `megatron/core/models/huggingface/` | `module.py` 的 `HuggingFaceModule` 基类 + `build_hf_model()` 按 `hf://` URI 分发：`clip_model.py` 包 SigLIP（vision encoder）、`qwen_model.py` 包 Qwen2ForCausalLM（**整个 LM 用 HF 权重跑**）、`fastconformer_model.py` 包 Parakeet（语音） |

示例目录：`examples/multimodal/`（LLaVA+Mistral+CLIP 全流程，energon 数据）、`examples/multimodal_dev/`（新一代 model-agnostic 入口，含 Qwen3.5-VL + MRoPE，走 FSDP+EP）、`examples/mimo/` 和 `examples/mimo_bagel/`（MIMO/Bagel 训练入口）。

## 2. LLaVA 主流程精读（`megatron/core/models/multimodal/llava_model.py`）

### 2.1 模型组装图与维度对齐

```
 images [num_tiles, 3, H, W]
   │
   ▼  vision_model (CLIPViTModel / RADIOViTModel / HF SigLIP)   ── 只在 PP first stage (add_encoder)
 [num_tiles, img_seq_len(+class_token), h_vision]
   │  drop class token (llava_model.py:1145-1148)
   │  pixel_shuffle 可选：4 倍通道合并、token 数 ÷4 (llava_model.py:1342-1378)
   ▼  permute → [img_seq_len, num_tiles, h_vision]              (llava_model.py:1171-1173)
 vision_projection (MultimodalProjector: mlp / affine)          (llava_model.py:1176-1178)
   ▼  [img_seq_len, num_tiles, h_language]
 input_ids（含 -200 占位）──embedding──▶ language_embeddings [b, text_len, h_language]  (llava_model.py:1240-1246)
   │
   ▼  _preprocess_data：按 -200 位置把图像 embedding "摊铺"进文本 embedding (llava_model.py:494-759)
 combined_embeddings [combined_seq_len, b, h_language]  ──▶  _process_embedding_token_parallel（SP/CP 分片）
   │
   ▼  language_model = GPTModel / HybridModel (decoder_input=combined_embeddings, input_ids=None)  (llava_model.py:1279-1288)
 logits / loss
```

组装发生在 `__init__`：

- **decoder**：默认 `GPTModel`（llava_model.py:244-261），`language_model_type` 以 `nemotron5-hybrid`/`nemotron6-moe` 开头时用 `HybridModel`（llava_model.py:226-242）。注意 `scatter_embedding_sequence_parallel=False`——embedding 的 SP 分片由 LLaVA 自己控制。
- **encoder**：按 `vision_transformer_config.vision_model_type` 分派（llava_model.py:278-371）：
  - `clip`/`siglip`/`internvit` → `CLIPViTModel`（siglip 无 class token，llava_model.py:281-288）
  - `radio`/`radio-g`/`cradio-g` → `RADIOViTModel`，class_token_len 默认 8/5/8，FP8 时强制 16（mxfp8 时 32）以满足对齐（llava_model.py:325-333）
  - `hf://...` → `build_hf_model`（llava_model.py:360-365）
- **projector**：输入维度 = `vision_config.hidden_size`（pixel_shuffle 时 ×4，llava_model.py:377-378），输出到 LM hidden size（llava_model.py:381-387）。
- **图像序列长度**：`get_num_image_embeddings`（clip_vit_model.py:205-261）统一计算 `img_seq_len = patches + class_token_len`，pixel_shuffle 后 ×0.25，tile_tags 再 +5/+6。

### 2.2 image embedding 注入机制：`_preprocess_data`（llava_model.py:494-759）

这是全篇核心。输入约定（docstring，llava_model.py:518-523）：

```
input_ids = [0, 1, -200, 2, 3]      # -200 = DEFAULT_IMAGE_TOKEN_INDEX (llava_model.py:49)
labels    = [1, -200, 2, 3, 4]
→ final_embeddings = [emb(0), emb(1), image_embeddings(576个), emb(2), emb(3)]
  final_labels     = [1, -100, 2, 3, 4]
  final_loss_mask  = [1, 0, ..., 0, 1, 1]
```

逐段拆解：

1. **定位 image token**（llava_model.py:574-587）：`image_token_mask = input_ids == image_token_index`；每个样本的新序列长 = `文本长 - 占位token数 + tile数 × img_seq_len`（llava_model.py:587）；PP 下为固定 shape 还需 pad 到 `_language_max_sequence_length`（llava_model.py:590-595）。
2. **算新位置**（llava_model.py:597-607）：把每个 -200 视作 `num_tiles × img_seq_len - 1` 个额外位置做 cumsum，得到每个**文本 token**在合并序列里的新下标 `text_position_ids`。图像占的位置就是"文本位置之外的空位"。
3. **labels 左移对齐**（llava_model.py:613-622）：预测错位，所以 label 用的下标整体 -1，并丢弃 <0 的。
4. **images_mask**（llava_model.py:625-637）：先全 True，再扣掉文本位置和 padding 位置，剩下的就是图像 embedding 该填的位置。
5. **填 embedding**（llava_model.py:641-667）：`final_embedding` 先填文本（`final_embedding[batch_indices, text_position_ids] = ...`），再 `final_embedding[images_mask] = image_embeddings...reshape(-1, h)`。注意 llava_model.py:659-663 的 **FSDP 坑**：纯文本 batch 时 FSDP 会 hang，workaround 是拿 dummy image 过一遍 vision model，再把输出以 `0 *` 的形式加进去（保证参数参与建图但数值为零）。
6. **音频同理**（llava_model.py:669-695）：sound token（-300）位置用 `sound_embeddings` 替换，支持变长 clip（`sound_embeddings_len`）。
7. **labels/loss_mask**（llava_model.py:699-737）：`final_labels` 全量初始化为 `IGNORE_INDEX=-100`（llava_model.py:47, 700-702），只填文本位；`final_loss_mask` 初始化全 0，只填文本位的 loss_mask。

### 2.3 loss mask：image token 不算 loss 的三重证据

1. `final_loss_mask[images_mask] = 0`（llava_model.py:722）——图像位置显式置 0。
2. **图像前最后一个文本 token 的 loss 也被 mask**（llava_model.py:724-737）——因为它要预测的第一个"图像 token"根本不是文本，预测了也没意义。这是很细的一个正确性处理。
3. `final_labels` 中图像位置保持 -100（llava_model.py:700-702），CE 的 `ignore_index` 再兜一层。

另外数据侧 `pretrain_vlm.py:289-294` 在 `_preprocess_data_for_llava` 里给 image token 占位符前置时就把 loss_mask 对应位置补 0。

### 2.4 多图 / 视频 / 动态分辨率

`forward`（llava_model.py:942-1290）里图像路径分三种：

- **普通多 tile**（llava_model.py:1134-1148）：`images [num_tiles, ...]` 一次过 ViT，`num_image_tiles` 记录每张图几 tile；`_apply_tile_tagging`（llava_model.py:906-940）实现 NVLM 的 tile tag：在每个 tile 的 embedding 前拼 5 个 `<tile_N>` 文本 token 的 embedding（借 LM 的 embedding 表，llava_model.py:933）。
- **packed dynamic resolution**（llava_model.py:1093-1133）：RADIO 输出 `[1, sum(patches_i + ct_len), h]`，按 `imgs_sizes` 切回每图、剥 class token、可选逐图 pixel_shuffle，再把每图当作 1 个"tile"喂给 `_preprocess_data`（`img_seq_len` 折叠为 1，llava_model.py:564）。
- **时序视频**（llava_model.py:1013-1087）：`temporal_patch_dim > 1` 时按 tubelet 分组，CP 下先把帧/tubelet 切到各 CP rank（`split_to_context_parallel_ranks_dynamic_res`），过完 ViT 再 `gather_from_context_parallel_ranks_dynamic_res` 拼回全集，保证文本侧 merge 时每个 rank 看到相同的图像集合。

## 3. vision encoder

### 3.1 CLIPViTModel（`megatron/core/models/vision/clip_vit_model.py`）

- 结构教科书式：`conv1`（kernel=stride=patch_dim 的 Conv2d 做 patchify，clip_vit_model.py:115-122）→ 加 class token（clip_vit_model.py:181-187）→ learned position embedding（clip_vit_model.py:126-128, 190）→ ln_pre → `TransformerBlock`（post_process=False，无最终 LN，clip_vit_model.py:147-154）→ 可选 ln_post。
- 子类型差异（clip_vit_model.py:91-113）：clip 有 ln_pre 无 conv bias；siglip 无 class token、有 ln_post、conv 带 bias + "valid" padding；internvit conv 带 bias。
- **并行策略**：spec 里 attention 用 `TELayerNormColumnParallelLinear`/`TERowParallelLinear`（vit_layer_specs.py:37-59），即 **ViT 也吃 TP**；但 `pretrain_vlm.py:163-169` 强制 vision tower `context_parallel_size=1`、关 SP、关 TP comm overlap；`pretrain_vlm.py:182-183` 强制 encoder 和 projector `pipeline_model_parallel_size=1`——即 **ViT+projector 整体只放在 PP stage 0，TP 内分片、DP 内复制**。
- ViT attention 用 `AttnMaskType.no_mask`（vit_layer_specs.py:47），与 decoder 的 causal 形成对照；当 CP/SP 需要 padding 时，`pretrain_vlm.py:131-150` 会把 decoder 的 mask 类型升级成 `padding_causal`。

### 3.2 RADIOViTModel（`megatron/core/models/vision/radio.py`）

NVIDIA RADIO 系列，支持 dynamic resolution（packed THD 输入 + 逐图位置编码插值，radio.py:381-400）、CPE（cropped position embedding，可强制 eval 模式保持预训练行为）、temporal tubelet（`temporal_patch_dim>1` 时 `_apply_temporal_grouping`，radio.py:349-363）、可选独立 video embedder（radio.py:365-373）。dynamic-res 下 class token 是逐图插入并同步修 `cu_seqlens`（radio.py:401-420）。

### 3.3 MultimodalProjector（`megatron/core/models/vision/multimodal_projector.py`）

- `projector_type="mlp"`：复用标准 `MLP` 模块，`input_size=vision hidden`（multimodal_projector.py:47-50）；
- `"affine"`：单个 `linear_fc1` 直接 `input_size → config.hidden_size`（LM hidden）（multimodal_projector.py:51-63）；
- 输出包 `make_viewless_tensor`（multimodal_projector.py:87-89）防止 schedules.py 的 `deallocate_output_tensor` 报错——这是 pipeline 并发的细节坑。

## 4. 数据与训练

### 4.1 数据集

`megatron/core/datasets/multimodal_dataset.py` 里只有 `MockMultimodalDataset`（继承 MockGPTDataset，追加 `sample["image"] = zeros(3, H, W)`，multimodal_dataset.py:55-57）+ `MultimodalDatasetConfig`（多 `image_h/image_w/preprocess_func`，multimodal_dataset.py:11-32）。**真实数据在 `examples/multimodal/`**：energon/webdataset 格式（VQASample，image=jpg、context/answers=json），LLaVA-Pretrain → wds → energon prepare 的流程见 `examples/multimodal/README.md`。

样本约定（`pretrain_vlm.py:267-300` `_preprocess_data_for_llava`）：tokens 前 prepend 一个 -200 占位，labels/loss_mask/position_ids 同步前补（loss_mask 补 0）。

### 4.2 pretrain_vlm.py vs pretrain_gpt.py

- 同一套 `pretrain()` 驱动，`forward_step` 复用 `pretrain_gpt.loss_func`（pretrain_vlm.py:29, 414）；
- **seq_length 语义被重定义**（pretrain_vlm.py:79-101）：`seq_length/encoder_seq_length` 被改成 `num_image_embeddings`（ViT 的序列长），`decoder_seq_length = dataloader_seq_length + num_image_embeddings + mp_padding`，`max_position_embeddings` 相应抬高；
- `get_batch`（pretrain_vlm.py:303-381）：CP/SP 时按 `tokens== -200` 算出图像将引入的额外长度，做 padding 并构造 `PackedSeqParams`（THD 格式时把 [B,S] 拉平成 [T,1]，pretrain_vlm.py:368-377）；
- **限制**：`assert not (CP>1 and PP>1)`（pretrain_vlm.py:63-66）；只支持 `ckpt_format=torch`（pretrain_vlm.py:60-62）；`ModelType.encoder_or_decoder`；
- PP 划分：embedding ranks = 首尾 stage（`llava_embedding_ranks`，pretrain_vlm.py:451-462），position embedding 只在 stage 0（pretrain_vlm.py:465-470）。

### 4.3 冻结策略

`LLaVAModel.freeze(freeze_language_model, freeze_vision_model, freeze_vision_projection, freeze_sound_model, freeze_sound_projection)`（llava_model.py:459-492）纯 `requires_grad=False`。入口参数 `--freeze-LM` / `--freeze-ViT`（pretrain_vlm.py:420-425；examples 版在 `examples/multimodal/multimodal_args.py:10-11`），调用点 pretrain_vlm.py:218-222——注意 **projector 永远训练**（`freeze_vision_projection=False` 写死），这正对应 LLaVA 两阶段里"只训 projector"（freeze LM+ViT）和"全量 SFT"两档。示例 `examples/multimodal/pretrain_mistral_clip.sh:115-116` 预训练阶段就是 `--freeze-LM --freeze-ViT`。

## 5. 高级模型：Bagel / MIMO

- **MIMO**（`megatron/core/models/mimo/`，README 写得很清楚）：统一骨架 = 1 个 LM + N 个 `ModalitySubmodules`（各带 encoders/decoders/input_projections/output_projections，`submodules/base.py`；默认实现 `VisionModalitySubmodules`、`AudioModalitySubmodules`）。理解路径：`Input → Encoder → Projection → align_embeddings_by_token_positions（按 special token 占位替换）→ LM`；生成路径：`LM hidden states（取 special generation token 位置）→ Output Projection → Decoder → 模态输出`。即 LLaVA 机制的"双向 + 多模态"泛化。还支持模块与数据并行网格的 colocated 布局（`comm/colocated_communicator.py`、`partition/`）。
- **Bagel**（`megatron/core/models/bagel/` + `examples/mimo_bagel/`）：理解-生成统一模型，LLM 侧用 **MoT（Mixture of Transformers）**——`MoTTransformerLayer` 里理解流和生成流各有独立 FFN/注意力权重但共享注意力计算（`attention_mot.py`、`mot_streams.py`），配 `BagelRotaryEmbedding` 与 FlexAttention，packed 序列用 `MoTPackedSeqParams`。生成侧是扩散：VAE encoder 把图压到 latent，LM 的生成流输出预测 noise，loss 是 MSE（`bagel_mimo.py` 的 `_native_bagel_squared_error`），`examples/mimo_bagel/diffusion/` 提供 `DiffusionModalitySubmodules`（把 diffusion 包装成 MIMO 的一个 modality）和 `hf_bagel_vae.py`。LLM 骨架可以是 HF 的 Qwen2（`hf_bagel_llm.py`）或 MCore（`mcore_bagel_llm.py`），入口 `examples/mimo_bagel/train.py`（`model_provider_bagel`，可选 `Qwen2MoTDecoderLayer`）。细节代码量很大，此处指路。

## 6. 多模态下的并行特殊处理（`megatron/core/models/multimodal/context_parallel.py`）

- **padding 计算** `get_padding`（context_parallel.py:16-66）：CP+SP 要求总长被 `tp×cp×2` 整除；只 CP 是 `cp×2`；只 SP 是 `tp`；连 FP8 都有对齐要求（16，mxfp8 为 32）。若 decoder 开 TP comm overlap，则直接用用户给的 `decoder_seq_len` 反推 padding（context_parallel.py:44-51）。
- **THD packed params** `get_packed_seq_params`（context_parallel.py:69-118）：有效长 = 文本 + 图像 - padding，padded 长 = 文本 + 图像；CP>1 且有 padding 或 packing 时切 THD 格式。
- **图像沿 batch 维切 CP** `split_to_context_parallel_ranks`（context_parallel.py:121-146）：ViT 不做序列 CP，改为把 **图片按张数**分到各 CP rank（不够补零），算完再 all-gather 回来（`gather_from_context_parallel_ranks`，autograd.Function 前向 all-gather / 反向 reduce-scatter，context_parallel.py:188-212）。
- **dynamic-res 变长切分** `split_to_context_parallel_ranks_dynamic_res`（context_parallel.py:313-536）：按 `cu_seqlens` 保证每 rank 拿整数张图；视频时还有 tubelet 边界感知的切点计算 `_compute_tubelet_aware_split_points`（context_parallel.py:233-287，保证不把一个 tubelet 切开）；图比 rank 少时塞 dummy 图（context_parallel.py:389-414）；gather 侧用 all_gather shape + all_to_all 处理变长（context_parallel.py:215-230）。
- **合并序列的 SP/CP 分片**在 `_process_embedding_token_parallel`（llava_model.py:761-904）：先 pad 到 shard_factor（llava_model.py:802-849），CP 下 sbhd 走 `get_batch_on_this_cp_rank`、THD 走 `tex.thd_get_partitioned_indices`（llava_model.py:869-888），SP 最后 `scatter_to_sequence_parallel_region`（llava_model.py:899-902）。labels pad -100、loss_mask pad 0，保证 padding 不污染 loss。

## 7. 精妙细节 / 坑

1. **"图像前 token 不算 loss"**（llava_model.py:724-737）：不只 mask 图像位本身，还 mask 图像前最后一个文本位——因为它要预测的对象是图像内容，文本 LM 根本预测不了。容易漏。
2. **FSDP 纯文本 batch 会 hang**（llava_model.py:657-663）：workaround 是跑 dummy image 再过 `0 * embedding` 加进结果，保证 vision 参数参与 autograd 图。
3. **image token id 不进配置**：`transformer_config.py` 里没有 image_token 字段；`pretrain_vlm.py` 硬编码 `DEFAULT_IMAGE_TOKEN_INDEX=-200`（llava_model.py:49），而 `examples/multimodal/model.py:222` 用 `tokenizer.convert_tokens_to_ids("<image>")` 动态取——两套入口约定不同，混用 checkpoint/数据时要对齐。
4. **ViT 的 TP 和 decoder 的 TP 是同一组**，但 ViT 强制 CP=1、关 SP（pretrain_vlm.py:163-169）：图像靠"按张数切 CP rank"来均摊，不走序列维 CP；这导致 ViT 计算在各 CP rank 间天然不均衡（图多张少时靠 dummy 补齐）。
5. **`scatter_embedding_sequence_parallel=False`**（llava_model.py:257）：LLaVA 不用 GPTModel 内置的 embedding SP scatter，因为合并序列在 embedding 之后才生成，必须自己 pad→CP 切→SP scatter（llava_model.py:899-902）。顺序错了就维度对不上。
6. **PP 的中间 stage 完全跳过 preprocess**（llava_model.py:555-556）：`not pre_process and not post_process` 时 `_preprocess_data` 直接返回 None；首 stage 只做 embedding 替换，末 stage 只做 labels/loss_mask 重排——图文 merge 被拆到 pipeline 两端。
7. **class_token_len 与 FP8 联动**（llava_model.py:332-333）：RADIO 开 FP8 时 class token 数被强制改成 16/32，纯粹为了让 token 数满足 FP8 GEMM 对齐——改 class token 数来凑对齐，而不是 pad 序列。
8. **projector 输出必须 `make_viewless_tensor`**（multimodal_projector.py:87-89）：否则 pipeline 调度器的 `deallocate_output_tensor` 会报错；`permute` 后也必须 `.contiguous()`（llava_model.py:1170-1173 注释），否则 PP 通信挂。
9. **CP+PP 不支持**（pretrain_vlm.py:63-66）、推理时 image token 只算一次（KV cache 里记 `image_tokens_count` 当 offset，llava_model.py:1187-1190）——增量 decode 直接跳过 vision tower。

（bagel/mimo 的内部行号未逐行精读，第 5 节为概述；如需深入可另开一篇。）
