---
title: "vLLM 研读（六）Attention 后端：FlashAttention、FlashInfer 与 MLA 怎么选"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 7
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "Attention", "FlashAttention", "FlashInfer", "MLA", "KV Cache"]
categories: ["推理框架"]
---

同一份 DeepSeek-V3 权重，在 H100 上默认走 `FLASH_ATTN_MLA`，换到 B200 上默认走 `FLASHINFER_MLA`。而如果你在 A100 上手选 `FLASHMLA`，启动会直接失败，报 `compute capability not supported`；给 `CUTLASS_MLA` 配一个 `--block-size 64`，报的是 `block_size not supported`。

这些错误都来自同一个地方：vLLM 启动时那行日志——`Using FLASH_ATTN attention backend out of potential backends: [...]`（`vllm/platforms/cuda.py:513`）。

先记住一个画面：**attention 层在构造时向平台要一个类；平台拿一张按优先级排好的名单，逐个问"你能不能吃下这个配置"；第一个说能的就是赢家。** 这个类随后要交出四样东西：kernel 实现、metadata builder、一组能力声明，以及它决定的 KV cache 形状。

这一篇回答三个问题：

1. "attention backend" 在代码里到底包含什么，为什么它不只是"选一个 kernel"？
2. 自动选择和手动指定各走哪条路，`validate_configuration()` 到底查哪些项？
3. MLA 为什么把"选后端"变成了"筛硬件"，以及 prefill/decode 为什么经常不是同一个后端？

## 1. 一个 backend 是四件东西

`AttentionBackend` 是抽象基类（`vllm/v1/attention/backend.py:59`）。它的三个抽象方法定义了最小契约（`backend.py:82-94`）：

- `get_name()`：日志和配置里出现的名字；
- `get_impl_cls()`：真正调 kernel 的类；
- `get_builder_cls()`：把 `CommonAttentionMetadata` 翻译成这个 kernel 认识的 metadata。

第三件常被忽略。metadata builder 是 `AttentionMetadataBuilder`（`backend.py:605`），每层每步都要跑一次，负责把 block table、slot mapping、seq_lens 加工成 kernel 要的参数；它同时声明 CUDA Graph 支持等级（`backend.py:608`，枚举定义在 `backend.py:570`）。builder 写得慢，kernel 再快也会被 host 侧开销吃掉——这就是为什么 MLA 的 builder 要预分配 scheduler metadata 缓冲区。

第四件是能力声明。它看起来像注释，其实是选择器的输入：`supports_head_size`、`supports_kv_cache_dtype`、`supports_block_size`、`supports_compute_capability`、`supports_sliding_window`（`backend.py:194-196`）、`is_sparse`（`backend.py:187-188`）。默认实现都偏保守，由各后端覆盖。比如 FlashAttention 的头维上限跟着 FA 版本走：≤256 一律支持，>256 要 FA4（`vllm/v1/attention/backends/flash_attn.py:425-435`）。

还有一件更隐蔽的事：backend 能改 KV cache 的布局。`customize_spec()`（`backend.py:146-157`）允许后端在 spec 建好之后再调整，`supported_kv_cache_layouts()`（`backend.py:377-380`）声明自己只能吃哪种排布。FlashInfer 在 SM100 上只接受 LBHNC/BLHNC（`vllm/v1/attention/backends/flashinfer.py:731-737`），FlexAttention 只接受 LBNHC（`vllm/v1/attention/backends/flex_attention.py:130-133`）。换后端可能意味着 KV cache 要重排，这是"backend 不只是 kernel"最硬的证据。

边界还体现在写 cache 上：FlashAttention 的 `forward_includes_kv_cache_update = False`（`flash_attn.py:369`），写 cache 是单独的 `do_kv_cache_update`；另一些后端把它融进 forward，默认值是 True（`backend.py:70`）。**一个 backend 划走的是"attention 这一层的全部职责"，不只是那个矩阵乘。**

## 2. 选择：优先级名单 + 逐个 validate

后端名单是一个枚举，每个成员的值是类路径（`vllm/v1/attention/backends/registry.py:34-44`）。`register_backend()` 能在运行时覆盖或新增（`registry.py:175-205`），所以第三方后端不用改 vLLM 就能挂进来，`--attention-backend` 传的也只是枚举名。

真正的顺序在 `_get_backend_priorities()`（`vllm/platforms/cuda.py:84`）。非 MLA、SM100（B200/B300）、且不是 non-causal 场景时，名单是 FlashInfer 打头（`cuda.py:160-168`）：

```text
FLASHINFER -> FLASH_ATTN -> TRITON_ATTN -> FLEX_ATTENTION -> TURBOQUANT
```

其余平台则是 FlashAttention 打头（`cuda.py:169-181`）。MLA 走单独的分支（`cuda.py:96-155`）：SM90 的主干是 `FLASH_ATTN_MLA -> FLASHMLA -> FLASHINFER_MLA -> TRITON_MLA`，SM100 是 `FLASHINFER_MLA -> TOKENSPEED_MLA -> CUTLASS_MLA -> FLASH_ATTN_MLA -> FLASHMLA -> TRITON_MLA`，SM120 只剩 `TRITON_MLA` 和稀疏的 SM120 后端。

`validate_configuration()`（`backend.py:284`）是一张检查清单，逐项 append 原因：

| 检查项 | 代码位置 | 典型失败 |
|---|---|---|
| head_size | `backend.py:307` | MLA 只接受 320/512/576（`mla_attention.py:1625-1627`） |
| dtype / kv_cache_dtype | `backend.py:309-312` | FLASH_ATTN_MLA 不支持 fp8 KV |
| block_size | `backend.py:313` | CUTLASS_MLA 要求 128 |
| MLA / sparse / sink | `backend.py:319-330` | 普通后端选了 MLA 模型 |
| compute capability | `backend.py:333` | FlashMLA 只认 SM90/SM100 |
| sliding window / DCP / PCP | `backend.py:337-352` | 后端没实现该并行模式 |

自动选择就是"按优先级逐个 validate，第一个通过者"（`cuda.py:402-419`），最后按 priority 取最小值（`cuda.py:483-487`）。手动指定则只 validate 那一个，失败直接抛（`cuda.py:446-450`）：

```text
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

这是官方文档里的原样例子（`docs/design/attention_backends.md:120-122`），对应的事实是 `FlashMLABackend.supports_compute_capability` 只认 major 9/10（`vllm/v1/attention/backends/mla/flashmla.py:73-74`）——在 A100（SM80）上手选它，就会得到这句话。

block size 是最容易踩的耦合项。FA 要求 16 的倍数（`flash_attn.py:357-367`），FlashMLA 固定 64（`flashmla.py:57-58`），CUTLASS_MLA 固定 128（`vllm/v1/attention/backends/mla/cutlass_mla.py:49-50`），且 `supports_block_size` 用的是整除判断（`backend.py:127-143`）。所以钉 `--block-size 64` 会让 CUTLASS_MLA 直接出局，钉 32 会让 FlashMLA 出局。vLLM 甚至会为此发一条 warning：`--block-size ... precluded higher-priority backend(s)`（`cuda.py:491-506`）。**"选后端"经常等于"选 block size"。**

## 3. prefill 和 decode 为什么是两个 kernel

判断标准是算术强度。设单层、单请求、上下文长 $L$、$h_q$ 个 query head、$h_{kv}$ 个 KV head、头维 $d$、每元素 $s$ 字节：

$$
\text{AI}_{\text{decode}} \approx \frac{4h_qdL}{2h_{kv}dLs} = \frac{2h_q}{h_{kv}s}
$$

$h_q/h_{kv}=16$、bf16（$s=2$）时只有 **16 FLOP/B**，而且与 $L$ 无关。H100 的 bf16 "脊点"约 $989/3.35 \approx 295$ FLOP/B——decode 差了一个数量级，**瓶颈在把 KV 从 HBM 读进来，不在算**。

prefill 换一个形状（$S$ 个 token 一起算）：

$$
\text{AI}_{\text{prefill}} \approx \frac{4S^2d}{2Sds} = \frac{2S}{s}
$$

$S=2048$ 时约 2048 FLOP/B，远在脊点之上；$S=128$ 时只有 128，又掉回 memory-bound。**prefill 与 decode 的分界不是相位，是算术强度**——这也解释了为什么小 prefill 会被后端塞进 decode 路径：FlashMLA 的 `reorder_batch_threshold=128`、FA_MLA 是 512（`flashmla.py:112`、`vllm/v1/attention/backends/mla/flashattn_mla.py:129`）。

decode 还有第二层问题：并行度。batch=1 时只有 $h_q$ 个头能铺满 SM，KV 那么长却没人读。flash-decoding 把 KV 维切成 `num_splits` 段，各段算局部 softmax，再用 LSE 合并。FA 的 `max_num_splits` 默认交给启发式，只有 CUDA Graph 场景才钉死（`flash_attn.py:855-869`），因为 split 的中间 buffer 是 `[num_splits, num_heads, num_tokens, head_size]`（同段注释），**并行度是拿显存换的**。FA4 把这个参数一路传到 dense kernel（`flash_attn.py:1413-1443`）。

FlashInfer 把这件事写在了类型里：prefill 和 decode 是两个 wrapper（`flashinfer.py:744-752`），decode 内部还分 XQA 与 trtllm-gen 两条路（`flashinfer.py:758-763`）。MLA 更彻底，干脆是两套后端：文档写明 "MLA uses separate backends for prefill and decode phases"（`docs/design/attention_backends.md:210`），prefill 由 `-ac.mla_prefill_backend` 单独选（`vllm/v1/attention/backends/mla/prefill/selector.py:46`），SM100 上则按 MLA 维度在 TRTLLM_RAGGED / FLASH_ATTN / FLASHINFER / TOKENSPEED 之间排（`vllm/platforms/interface.py:468-484`）。

## 4. 主流后端对比

| 后端 | 内核来源 | 计算能力 | block size | KV dtype | CUDA Graph | 典型场景 |
|---|---|---|---|---|---|---|
| `FLASH_ATTN` | FA2/FA3/FA4 | ≥8.0 | 16 的倍数 | fp16/bf16/fp8 | FA3/4 ALWAYS | 通用默认，Hopper 首选 |
| `FLASHINFER` | FlashInfer + XQA/trtllm-gen | 8.0–12.1 | 16/32/64（SM100 可到 1024） | 含 nvfp4 | 按子路径 | Blackwell 默认，低延迟 decode |
| `TRITON_ATTN` | Triton 手写 | 任意 | 16 的倍数 | 含 int4/int8 per-token-head | ALWAYS | 兜底、可复现、改 kernel 快 |
| `FLEX_ATTENTION` | torch flex_attention | 任意 | 16 的倍数 | 仅 fp16/bf16 | ALWAYS | 自定义 mask、non-causal、多模态 |
| `FLASHMLA` | FlashMLA | SM90/SM100 | 固定 64 | fp8 | UNIFORM_BATCH | DeepSeek decode，H100/B200 |
| `FLASH_ATTN_MLA` | FA3 MLA | SM90 | 16 的倍数 | **不含 fp8** | UNIFORM_BATCH | H100 上 BF16 KV 的 MLA |
| `CUTLASS_MLA` | CUTLASS | SM100 | 固定 128 | fp8 | 单 token decode | B200 上 MLA decode |
| `TRITON_MLA` | Triton | 任意 | 16 的倍数 | fp8 | UNIFORM_BATCH | MLA 的可移植兜底 |

几处值得单独说：

- `FLASH_ATTN` 是"一个后端、三代内核"。版本选择在 `get_flash_attn_version()`（`vllm/v1/attention/backends/fa_utils.py:99-107`）：SM90 优先 FA3，SM100 优先 FA4，其余回落 FA2。头维、KV dtype、sink、mm_prefix 都会反过来影响这个选择（`flash_attn.py:461-512`）；fp8 KV 也不是无条件成立，只放行 FA3/SM90、FA4/SM100，或 SM90 上头维 512 的 FA4（`flash_attn.py:483-499`）。
- `FLASHINFER` 的 block size 是条件声明的：只有 SM100、GQA、且 trtllm-gen 可用时才 advertise 128–1024 的大 page（`flashinfer.py:587-606`），否则只有 16/32/64。头维只支持 64/128/256/512（`flashinfer.py:693-695`）。
- `TRITON_ATTN` 的计算能力检查永远返回 True（`vllm/v1/attention/backends/triton_attn.py:405-406`），所以它总是名单里的最后一道保险；它和 FA 都声明了 batch invariance（`triton_attn.py:367-368`），是逐位复现时的常用选择。
- `FLEX_ATTENTION` 不是性能选项。它把 mask 写成 `mask_mod` 交给 `torch.compile` 后的 `flex_attention`（`flex_attention.py:55-58`），换来任意 mask 的表达力，代价是 KV dtype 只支持 fp16/bf16（`flex_attention.py:96-100`）。
- Mamba / 线性注意力不在这个枚举里，走 `MambaAttentionBackendEnum`（`registry.py:254`）和另一个选择器 `get_mamba_attn_backend`（`vllm/v1/attention/selector.py:231-235`）。它的"cache"是固定形状的 recurrent state，不是 per-token KV（详见第 6 节）。

## 5. MLA：压缩、absorb 与一笔 KV 账

DeepSeek 的 MLA 不缓存 per-head 的 K/V，而是缓存一份 512 维 latent $c_{KV}$ 加一份所有头共享的 64 维 RoPE key $k_{pe}$。落到 spec 上就是 `num_kv_heads=1`、`head_size=576`、`head_size_v=0`（`vllm/model_executor/layers/attention/mla_attention.py:1373-1391`，下称 `mla_attention.py`；`vllm/v1/kv_cache_interface.py:723-736`）——**一个 token 只有一行**。

两种算法并存，代码里的注释写得很清楚：

- **不 absorb**（"compute friendly"，`mla_attention.py:65-89`）：先把 $c_{KV}$ 上投影成 per-head 的 $k_{nope}$、$v$，再做标准 MHA（QK 头维 $128+64$，V 头维 128）。计算友好，但每次都要对全上下文做 `kv_b` 投影，prefill 的 $S_{kv}$ 一大，中间张量就顶不住，所以要分块（`mla_attention.py:120` 起的 Chunked Prefill 一节）。
- **absorb**（"data-movement friendly"，`mla_attention.py:93-116`）：把 $W_{UK}$ 吸收进 Q，先算 $q_{l,nope} = q_{nope} W_{UK}$，attention 退化成 MQA：Q 是 $N \times 512$，K 和 V 都是同一份 512 维 latent，输出再乘 $W_{UV}$ 升回 128 维。

decode 走 absorb（`mla_attention.py:1135` 调 `forward_mqa`，抽象接口在同文件 3398 附近）。理由就是第 3 节的账：decode 是 memory-bound，MQA 让 128 个 query head 共享一份 KV，读的量从"per-head K/V"降到"一份 latent"；代价是 head 维从 192 变成 576，算得更多、读得更少——在 memory-bound 区间里这是划算的。

账本。假设 DeepSeek-V3 的配置：61 层、128 个 head、`qk_nope=128`、`qk_rope=64`、`v=128`、`kv_lora=512`；单副本、不含 TP 切分、忽略 block 内 padding：

| 布局 | 每 token 每层 | 每 token（61 层） | 128K 上下文 |
|---|---|---|---|
| MLA bf16 $(512+64)\times2$ | 1152 B | 68.6 KiB | 8.58 GiB |
| MLA fp8 `fp8_ds_mla` | 656 B | 39.1 KiB | 4.88 GiB |
| GQA-8 bf16 $2\times8\times128\times2$ | 4096 B | 244 KiB | 30.5 GiB |
| MHA bf16 $2\times128\times128\times2$ | 65,536 B | 3.81 MiB | 488 GiB |

656 不是 576：`get_fp8_ds_mla_row_bytes()` 返回 656（`backend.py:1053-1054`），拆开是 512 字节 fp8 latent + 16 字节 scale + 64 个 bf16 的 RoPE key（128 字节）。

结论：**MLA bf16 的 KV 是 GQA-8 的 $1/3.6$、MHA 的 $1/57$；换 fp8 之后是 MHA 的 $1/100$。** 这才是 DeepSeek 敢在 128K 上下文上做单机部署的底层原因——不是 attention 算得快，是 KV 存得下。

代价是"更挑硬件"。latent attention 不是标准 attention，kernel 必须专门写，于是每个 MLA 后端都带着明确的硬件与 block 约束：FlashMLA 只认 SM90/SM100、block 固定 64（`flashmla.py:57-58`、`73-74`）；CUTLASS_MLA 只认 SM100、block 固定 128（`cutlass_mla.py:49-50`、`65-66`）；FLASH_ATTN_MLA 只认 SM90，且 KV dtype 不含 fp8（`flashattn_mla.py:42-46`、`69-70`）；TRITON_MLA 是唯一不挑计算能力的兜底（`triton_mla.py:114-115`、`140-141`）。

把这三条约束叠到优先级上，就能复现开头那幕：H100 上默认 `FLASH_ATTN_MLA`，一开 fp8 KV 就被 validate 踢到 `FLASHMLA`；B200 上默认 `FLASHINFER_MLA`（只吃 block 32/64，`flashinfer_mla.py:203-205`），一旦有人钉了 64，排在后面的 `CUTLASS_MLA` 就先出局。**这不是"选快的"，是"选能跑的"。**

## 6. 混合模型：接口层发生了什么

**sliding window。** 后端要先声明 `supports_sliding_window`（`backend.py:194-196`），KV spec 换成 `SlidingWindowSpec`，按 `sliding_window - 1` 个保留 token 算每请求的块上限（`vllm/v1/kv_cache_interface.py:907-913`）。同一模型里 full 层和 SWA 层可以进不同的 KV group（`vllm/v1/core/kv_cache_planning.py:1128-1155`），于是可以用 `backend_per_kind` 按 kind 分别指定后端（`selector.py:182-192`；`vllm/config/attention.py:48-58`）。这是接口层最实用的一把钥匙：**一个模型可以同时用两个 attention 后端**。

**Mamba / 线性注意力。** 它不在 `AttentionBackendEnum` 里，而是 `MambaAttentionBackendEnum`（`registry.py:254-302`），`is_ssm()` 返回 True（`vllm/v1/attention/backends/mamba2_attn.py:104-106`）。KV cache 也不是 per-token：`MambaSpec` 存的是固定形状的 state（`kv_cache_interface.py:1092-1112`），`max_memory_usage_bytes` 按 state 块数算，跟序列长度无关（同文件 1133-1139）。所以混合模型里 Mamba 层的显存是常数，随上下文涨的只有 attention 层。

**稀疏。** 接口上只是两件事：`is_sparse()` 必须和 `use_sparse` 匹配（`backend.py:326-330`），以及 spec 仍然是 MLA 的 latent 布局，只是不再全读。稀疏 MLA 是另一串枚举（`registry.py:97-99`），默认选择还会看 KV dtype 和 head 数（`cuda.py:97-120`）。

## 7. 工程建议与常见坑

什么时候手动指定：

1. **要逐位可复现**：`TRITON_ATTN` + `VLLM_BATCH_INVARIANT=1`（`triton_attn.py:367-368`），接受它比 FA 慢。
2. **要用 fp8 KV**：先查目标后端在不在 `supported_kv_cache_dtypes` 里。`FLASH_ATTN_MLA` 不在（`flashattn_mla.py:42-46`），`FLEX_ATTENTION` 也不在（`flex_attention.py:96-100`）。
3. **要自定义 mask / non-causal**：选 `FLEX_ATTENTION`，或者用 composite 后端把 Triton 和 FA 拼起来（`docs/design/attention_backends.md:59-89`）。
4. **要钉 block size**：先看它会踢掉谁（第 2 节）。

`--attention-backend` 和 `--attention-config.backend` 的区别比名字小：前者就是后者的 CLI 别名（`vllm/engine/arg_utils.py:1118-1120`），同时给会报 `mutually exclusive`（`arg_utils.py:2740-2746`）。区别不在"选了哪个后端"——`--attention-backend` 最终就是写进 `AttentionConfig.backend`（`arg_utils.py:2747-2749`），差别在于 structured config 还能顺手带上 `backend_per_kind` 和 `mla_prefill_backend`。想动 MLA prefill 必须额外给 `-ac.mla_prefill_backend`，这一点 `--attention-backend` 给不了。

H100 与 B200 的取舍，只讲机制不给 benchmark：H100 上 FA3 是默认（`fa_utils.py:99-107`），MLA 有 FlashMLA 和 FA_MLA 两条原生路，选谁取决于 KV dtype；B200 上默认 FlashInfer（`cuda.py:160-168`），因为 trtllm-gen 的 decode 路径在 SM100 上最完整，FA 则必须用 FA4。**两代卡的 decode 都远在脊点之下，所以差异不在算力，而在"能不能把 KV 带宽读满"**——split-K 的并行度、page 大小、TMA/异步拷贝，这三件事比峰值算力更决定胜负。

### 常见坑

1. **import 失败会被吞掉。** `get_valid_backends` 把 `ImportError/OSError` 记成一条 reason（`cuda.py:409-413`），没装 FlashInfer 不会报错，只会静默降级——而且这条 reason 是 debug 级日志才看得到。
2. **手动指定也要过 validate。** 它不是"强制"，只是"跳过优先级"。
3. **block size 的连锁反应。** 见第 2 节，钉一个值可能同时改变后端、page 大小和 prefix cache 的粒度。
4. **CUDA Graph 等级不一样。** FA3/4 是 `ALWAYS`，FA2 是 `UNIFORM_BATCH`（`flash_attn.py:619-623`）；MLA 后端多是 `UNIFORM_BATCH`，CUTLASS_MLA 只支持单 token decode（`cutlass_mla.py:33-36`）。混着 prefill 和 decode 的 batch 可能掉到 piecewise 甚至 eager，profile 时别把它算成 kernel 的锅。
5. **MLA 的 prefill 后端不受 `-ac.backend` 控制**（`docs/design/attention_backends.md:210`）。只改 decode，会得到"看起来换了、其实没换"的 profile。
6. **`TORCH_SDPA` 是空字符串，只给 ViT 用**（`registry.py:70`），别在 serve 时手选。

## 小结

- 一个 backend = kernel 实现 + metadata builder + 能力声明 + KV cache spec/layout；只把它当 kernel 会漏掉一半的失败模式。
- 自动选择是"按优先级逐个 `validate_configuration()`，第一个通过者"（`cuda.py:402-419`）；手动选择只跳过优先级，不跳过校验。
- prefill 与 decode 的分界是算术强度：decode 约 16 FLOP/B，prefill 约 $2S/s$；前者必然 memory-bound，所以 decode 要用 split-K、MQA/absorb 把读的量压下来。
- MLA 把每 token 每层的 KV 从 MHA 的 65,536 B 压到 1152 B（bf16）或 656 B（fp8），128K 上下文从 488 GiB 降到 8.58 GiB / 4.88 GiB。
- MLA 的代价是后端强绑定硬件：FlashMLA 只认 SM90/SM100 且 block 64，CUTLASS_MLA 只认 SM100 且 block 128，FA_MLA 不支持 fp8 KV。
- 混合模型在接口层的抓手是 `backend_per_kind`：full / sliding-window / MLA 可以各用各的后端；Mamba 的 state 不随上下文增长。
- Blackwell 默认 FlashInfer、Hopper 默认 FlashAttention，是 priority 列表写死的，不是 benchmark 结论；换卡前先看这条列表。

下一篇进编译：[torch.compile：切分、融合与编译预算]({{< relref "vllm-07-torch-compile" >}})，看 attention 这些 kernel 之外的图怎么被切开又拼回去。

参考：

- vLLM 官方文档：[Attention Backend Feature Support](https://docs.vllm.ai/en/latest/design/attention_backends/)
- 仓库内历史文档：[docs/design/paged_attention.md](https://docs.vllm.ai/en/latest/design/paged_attention/)（开头已声明它不再描述今天的代码）
