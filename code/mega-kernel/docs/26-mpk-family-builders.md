# MPK 生产路径的模型家族 builder 走读：MLA 吸收、FP8 UE8M0、MoE 与投机 draft

> 本文补齐 `docs/25`（P111-1）**§9 明留的尾巴**：
>
> > **未逐行展开每个家族的 `_build_*`**：GLM/Inkling/DeepSeek-V3 的 MoE/MLA 管线只给算子名与
> > 文件头架构说明，未逐行走读其权重折叠/吸收细节（那些在各自 `builder.py` 的 `_build_moe_mlp`/MLA 段）。
>
> `docs/25` 以 **Qwen3** 为规范样本讲透了「谁把 95 个手工 `*_layer()` 拼成一个模型」。本文把这层
> **往下钻到家族特异实现**：以 **DeepSeek-V3**（MLA + FP8 MoE + MTP，全专题最大的一份 builder，
> 2255 行）为主，对照 **GLM-4.6 / Inkling / GPT-OSS**，并补上不属于目标主干的**投机 draft builder**
> （`dflash`/`eagle3`）。headline——**模型家族的结构差异全部在 builder 层被手工"折算"成统一的
> 标准 task 管线**：运行层只看到 `linear` / `rmsnorm` / `paged_attention` / `moe_w13` / `silu_mul` /
> `moe_w2` / `moe_mul_sum_add` / `allreduce` 这些**标准化算子**，MLA 权重吸收、FP8 UE8M0 重定标、
> 共享专家折叠、滑窗/全局双 attention、深度短卷积、投机 draft …… 都是在这层用普通 PyTorch 张量
> 变换"折平"的。这正对应 `docs/22` §8 的结论：**前端只划 C3 的范围，不拿 C1/C2/C4/C5**。
>
> 本文是**纯源码走读（资料分析）**，不构建、不运行 mirage，不引用任何性能数。快照
> `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/25` 同一快照）。

## 0. 定位：家族 builder 在编译栈里的坐标

`docs/25` 已经把「注册 → `MPK.build()` 取类 → `build_from_dict` → `build_layers`」这条主干走通。
本文补的是主干里**每个家族自己实现的那段**：`models/<家族>/builder.py`。各家族的注册与规模：

| 家族 | 注册名 | `builder.py` | 行数 | 注册/入口行 |
|---|---|---|---|---|
| Qwen3 | `qwen3` | `models/qwen3/builder.py` | 696 | `docs/25` §3 已走 |
| **DeepSeek-V3** | `deepseek-v3`/`DeepSeek-V3`/`deepseek-ai/DeepSeek-V3` | `models/deepseek_v3/builder.py` | **2255** | `:63` |
| GLM-4.6 | `Glm4Moe`/`zai-org/GLM-4.6`/`glm4_moe`/`glm-4.6` | `models/glm4_moe/builder.py` | 454 | `:66` |
| GPT-OSS | `gpt_oss`/`GptOss`/`openai/gpt-oss-20b` | `models/gpt_oss/builder.py` | 374 | `:47` |
| Inkling | `Inkling`/`thinkingmachines/Inkling`/`inkling` | `models/inkling/builder.py` | 705 | `:70` |
| DFlash（草稿） | 不注册 | `models/dflash/builder.py` | 364 | —（`DFlashBuilder`） |
| Eagle3（草稿） | 不注册 | `models/eagle3/builder.py` | 461 | —（`Eagle3Builder`） |

注册装饰器是 `@register_model_builder(...)`（`model_registry.py:5-29`，`docs/25` §1）。**只有"目标
主干"家族注册**——`dflash`/`eagle3` 是投机解码的**草稿/验证子图**，由目标 builder 或 `demo.py` 用
构造器把它们**追加**到目标图上（见 §5），因此不出现在 `MPK.build()` 的名字表里。

一个贯穿全篇的观察：**家族差异的"体量"与 builder 代码量不成正比**。GLM/Inkling/GPT-OSS 都是
「GQA/paged attention + 标准 MoE」家族，几百行就够；DeepSeek-V3 因为**MLA 权重吸收 + FP8 分块量化
+ 双 dispatch + MTP**，需要 2255 行——**每多一项"推理期权重变换"，builder 就多一段手工张量代数。**

## 1. DeepSeek-V3：MLA 权重吸收 + FP8 UE8M0 + MoE + MTP

模块 docstring 把全貌写死了（`deepseek_v3/builder.py:1-11`）：61 层，层 0–2 是 Dense MLP、层 3–60 是
MoE（256 路由专家 top-8 + 1 共享专家），MLA 128 Q 头、吸收后 1 KV 头、head_dim=576（512 latent + 64
rope），可选 MTP 预测层。关键一句在 `:9-10`：

> **Weight absorption: at load time, `kv_b_proj` is absorbed into `q_b_proj` so that runtime only
> needs compressed KV cache `[c_latent(512), k_pe(64)]` = 576 dims.**

### 1.1 架构常量与两条构建入口

架构常量集中在 `:24-42`：`HIDDEN_SIZE=7168`、`NUM_LAYERS=61`、`NUM_Q_HEADS=128`、
`Q_LORA_RANK=1536`、`KV_LORA_RANK=512`、`QK_NOPE_HEAD_DIM=128`、`QK_ROPE_HEAD_DIM=64`、
`V_HEAD_DIM=128`、`QK_HEAD_DIM_TOTAL=576`、`V_HEAD_DIM_TOTAL=512`、`NUM_EXPERTS=256`、
`NUM_EXPERTS_PER_TOK=8`、`NUM_SHARED_EXPERTS=1`、`FIRST_MOE_LAYER=3`。

- **`build_from_model` 直接 `NotImplementedError`**（`:97-101`）——「DeepSeek-V3 太大，不能直接走
  HuggingFace 加载，必须用 `build_from_config()` 配预转换（吸收后）的权重」。这与 Qwen3 能直接
  `build_from_model` 不同，是家族差异在**入口**上的第一处。
- `build_from_config`（`:103-116`）接 `model_config.k_cache`（`[num_layers, num_pages, page_size,
  576]`）与 `position_embeddings`，再下沉到 `build_from_dict`。**注意 KV cache 是逐层的
  `ckv_kpe_cache[layer_idx]`**（`:698-700`），即 MLA 的压缩 KV 直接由调用方预分配。
- `__init__`（`:65-95`）里所有维度都先按 `world_size` 切：`num_local_q_heads = 128//world_size`
  （`:83`）、`intermediate_size = 18432//world_size`（`:88`）、`moe_intermediate_size =
  2048//world_size`（`:89`）——**TP 切分在 builder 建张量时就用除法落进形状**（对上 `docs/25` §5
  「builder 内 `//world_size` 手写切分」）。另有一处 `_fuse_residual = True`（`:92`）注释
  「Fuse residual into linear kernels (with_residual). Always on.」。

### 1.2 FP8 UE8M0 重定标：`_requantize_fp8_for_ue8m0`

这是家族特异权重代数里最硬的一块（`:475-542`）。注释把动机讲清（`:477-483`）：

> SM100 block-scaled UMMA 用 **UE8M0**（8-bit 只存指数的 scale）。checkpoint 的 float32 scale
> **不是 2 的幂**，直接转 UE8M0 每块最多引入 **2× 误差**。修法与 SGLang/vLLM 同：**dequant →
> 用 2 的幂重新量化**。

四步（`:493-542`）：(1) 用 float32 `scale_inv` 把 FP8 权重 dequant 回 float32；(2) 每 128 元素块取
`amax`，`new_scale = 2^ceil(log2(amax/448))`（448 是 E4M3 最大值）得到**精确 2 的幂**；(3) 用新 scale
重新量化回 FP8；(4) 把 4 个连续 UE8M0 字节**打包成 uint32**、按列主序存成
`[packed_k, aligned_M]`（`torch.empty_strided((M, packed_k), (1, aligned_M))`，`:537-540`）。

`_attach_fp8_weight`（`:549-561`）是统一入口：若 `state_dict` 有 `<key>_scale_inv` 就走 requantize
并 attach `(weight_fp8, packed_ue8m0)`；否则当 BF16（dequant 后）权重直接 attach。

> 这一块的产物**不是数值等价变换**——它是**为了喂给 SM100 UMMA 而改的表示**。所以 DeepSeek-V3
> 的图里 FP8 权重与 scale 都是"builder 造出来的张量"，不是 checkpoint 原样（对上 `docs/25` §3.3
> `new_intermediate_tensors` 的 `cuda_tensor` 一路）。

### 1.3 `_build_mla_attention_layer`：七步把 MLA 折成标准算子

`:563-792`。逐层 prefix 是 `model.layers.{i}.self_attn.`（`:565-566`）：

1. **`q_a_proj`（FP8）**：`_fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out, ...)`
   （`:568-573`）。
2. **`q_a_layernorm`（BF16）**：就地写回 `q_a_out`（`:575-581`）。
3. **`q_b_proj`（已吸收，BF16——吸收后 scale 被删）**（`:583-615`）。这里有一段 **dual-dispatch**
   注释（`:584-594`）：decode/MTP 只产 fused `[H*576]` 的 `q_nope_pe`；prefill 时**两种都要产**
   ——decode kernel 吃 `q_nope_pe`，prefill kernel 吃拆开的 `q_nope[H*512]`+`q_pe[H*64]`；
   **两个 attention kernel 都注册，运行时按 `Q_LEN` 早退**（见第 6 步）。拆开的 `q_b_nope`/
   `q_b_pe` 权重是 `demo.py` 的 Phase-2 吸收阶段在 load time 造好的（`:592-594`）。
4. **`kv_a_proj` 拆成 c_latent（FP8）+ k_pe（BF16 padded）**（`:617-676`）。关键工程细节：
   `k_pe` 输出只有 64 < `MMA_M=128`，所以**先 requantize、再把权重 pad 到 `[128, H]`**
   （`:636-653`）——因为输出 64 行在 SM100 上凑不满一个 MMA tile。scale 也要按
   `latent_ratio = 512/(512+64)` 切出前 512 行的分块 scale（`:626-631`）。
5. **`kv_a_layernorm` 只加在 c_latent 上**（`:678-684`）——注释（`:291-293`）说「用两个 linear 而不是
   一个 576 维输出，就是为了只对 c_latent 做 norm」。
6. **MLA attention（KV gather + decode + reduce）**（`:686-774`）。双 dispatch：prefill 走
   `mla_kv_gather_split_layer` + `mla_prefill_layer`（`:703-728`，`PF_BM=64`）；decode 路**总是注册**
   `mla_kv_gather_layer`（`:733-741`），再按 `world_size` 选 `mla_mtp_decode_tp2/tp4/tp8_layer`
   （`:743-766`）或单卡 `mla_mtp_decode_layer`（`:767-774`），各配一个 `*_reduce_layer`。**MLA 的
   split-K 把 partial `o`/`lse` 写进 `mla_partial_o`/`mla_partial_lse`**（形状在 `:340-375`）。
7. **O projection：V un-absorption 融进 o_proj**（`:776-792`）。注释（`:776-778`）：
   `o_proj_fused: [7168, H*kv_lora_rank]` 直接吃 `attn_out [N, H*512]`，**V 的反吸收在转换期融进
   o_proj**；残差通过 `with_residual` FP8 kernel 融进去（`residual=self.x`，`:792`）。输出写到
   **逐层的 `attn_proj_out`**（`:783-788`）以避免 `self.x` ↔ `self.attn_proj_out` 在下层的
   `with_residual` 调用里别名（`:781-782`）。

一句话：**MLA 在 builder 里被折成「一堆标准 linear + 两个（prefill/decode）paged/gather attention
kernel + 一次 fused o_proj」**——运行层不认 MLA，只认 `linear`/`paged_attention`/`mla_*` task。

### 1.4 `_build_dense_mlp` 与 `_build_moe_mlp`

**Dense（层 0–2）**（`:794-830`）：`gate_up_proj` FP8 linear → `silu_mul_layer` → `down_proj`
（带 `residual=self.x`）。这里有一个**交织网格**细节（`:805-816`）：`silu_mul` 期望 gate 在前半、
up 在后半，而交织拆分的粒度按**未切分前的** gate_up 维度算，故
`silu_mul_grid = interleave_split // world_size`（`:811-813`）。

**MoE（层 3–60）**（`:832-1160`）是本文最有份量的一段：

- **路由张量**（`:842-860`）：`moe_topk_weights`（fp32）、`moe_routing_indices`（int32
  `[NUM_EXPERTS, mbt]`，即"每个专家收哪些 token"）、`moe_mask`（int32 `[NUM_EXPERTS+1]`，
  末尾一个哨兵）。注意 `moe_routing_indices` 的形状是**专家维在前**——这是 group GEMM 的散列布局。
- **Router gate linear**（`:862-879`）：输出 256 维，`router_grid = min(grid_for_...(256), 256//8)`
  （`:869-872`，注释「每 block ≥8 个 BF16 元素以满足 TMA 描述符 16B 对齐」）。
- **`tensor_init_layer`** 先把 `moe_output` 清零（`:881-896`）；`io_category` 取
  `nvshmem_tensor if self._use_nvshmem else cuda_tensor`（`:883-889`）——**TP>1 时 MoE 输出必须落在
  对称显存**（对上 `docs/17` §3）。
- **`moe_topk_sigmoid_routing_layer`**（`:898-911`）：DeepSeek V3 用 sigmoid 打分 + 加
  `gate.e_score_correction_bias`（`:900-904`）；`block_dim=(256,1,1)` 注释「topk kernel 需要 8 warps」。
- **FP8 group GEMM**（`:913-984`）：`use_fp8_experts = w13_scale_key in state_dict`（`:915-916`）。
  组 GEMM 期望**逐行** weight_scale（不是分块 `scale_inv`），所以把 `[E, out/128, K/128]` 的
  `scale_inv` 用 `repeat_interleave(128, dim=1)` 展开成 `[E, out, K/128]`（`:920-931`）。**GEM 的 N
  拆分 `w13_m_split`** 由 `moe_fp8_m_split`（`:55-60`）算出（`grid_dim=(NUM_EXPERTS, w13_m_split, 1)`，
  `:961-974`）——模块顶部注释（`:44-51`）说这个 helper 让每个 active-expert CTA 的 N 切片保持
  `MMA_M=128` 的整数倍且≥1 个 tile，**TP=2 MTP=2 压力下 Y=2 约快 20%**（6.6 vs 8.3 ms/tok）。
  非 FP8 路走 `moe_w13_linear_layer`（`:975-984`）。
- **`moe_silu_mul_layer`**（`:986-997`）→ **W2 group GEMM**（`:999-1062`，同样 FP8/BF16 两路 +
  `w2_m_split`）。
- **共享专家**（`:1064-1159`）：1 个专家、**在所有 token 上跑**，其输出加到残差上。它的
  `gate_proj`/`up_proj` 用 `shuffle_tensors`（从 `models/utils.py` 导入，`:1081`）交织融合，split
  取 `min(linear_grid//2, scale_dim_0)` 并保证整除（`:1082-1098`）；`down_proj` 输出是
  **row-parallel 的部分和**，TP>1 时靠 `moe_output` 的 allreduce 求和，**所以共享专家的残差要在
  每个 rank 上都加**（`:1145-1150`，`add_residual_once=False`）。
- **最终 `moe_mul_sum_add_layer`**（`:1151-1159`）：`moe_output = Σ_k w_k·down_k + shared_residual`。

### 1.5 TP 与 MoE 残差

`build_layers`（`:2049-2128`）逐层串：`input_layernorm → MLA（残差融进 o_proj）→ self.x =
attn_proj_out → post_attention_layernorm → MLP`。MoE 层**不做 fused residual**（`:2094-2128`）：
TP>1 时先 `allreduce_layer` 再 `elementwise_add` 加跳连（`:2095-2114`），单卡直接 add
（`:2115-2128`）。**残差靠张量身份 `self.x = ...` 串联**（对上 `docs/25` §3.2）。

`build_from_dict`（`:2130-2255`）：embed → `_new_intermediate_tensors` + `_precompute_rope_embeddings`
→ `build_layers` → final norm → lm_head（vocab pad 到 129280，`:2137/2176-2202`）→ argmax
（`argmax_partial_layer` + `argmax_reduce_layer`，`:2242-2252`）→ `_build_mtp_layer`（`:2254`）。

### 1.6 MTP（多 token 预测）：把"第二个解码器"编进同一张图

`_build_mtp_layer`（`:1649-...`）的 docstring（`:1650-1661`）说明它是 vLLM
`DeepSeekMultiTokenPredictorLayer` 的翻版：`embed(draft) → enorm`、`hnorm(prev_hidden)`、
`eh_proj(cat)`（拆成 W1@e + W2@h，`:1690-1704`，**用 `.contiguous()` 切出两个张量后必须持有引用，
否则常驻 kernel 的裸指针会悬空**，`:1691-1693`）、整层 decoder（MLA + MLP）、共享 lm_head →
argmax。**草稿步在编译期静态展开**（`:1659`），MTP 层权重按 `step_idx % num_mtp_layers` 复用
（`:1660`）。

`_build_mtp_decoder_layer`（`:1183-1261`）复用主层的 `_build_mla_attention_layer_with_prefix` /
`_build_moe_mlp_with_prefix`（`:1206/1231`），但**在进入前先 `self.x = self.mtp_x`、退出后恢复**
（`:1204-1212/1228/1261`），让 `with_residual` 融到正确的隐藏态上。`_cached_attach`/`_cached_new_tensor`
（`:1162-1182`）用缓存避免 MTP 草稿循环里重复声明同名 C++ 变量（`:75-76`）。

**模型家族在 MTP 上的体现**：MTP 不是"额外的算子"，而是**把主层的 `_build_*` 用另一个 weight prefix
再跑一遍，挂到同一张图上**——所以整条解码 + 投机草稿仍是**一个常驻 kernel**（对上 `docs/01` §5.8
「每 batch size 一张静态图」的工程兑现）。

## 2. GLM-4.6：fused-QKV-with-bias + 部分 RoPE + 共享专家折叠成 expert 160

模块 docstring（`glm4_moe/builder.py:1-24`）给出全貌：92 层、hidden 5120、96 Q 头 / 8 KV 头、
head_dim 128；attention **带 bias**、逐头 q/k-norm（eps 1e-5）、**部分 RoPE**（`partial_rotary_factor
0.5 → rotary_dim 64/128`，theta 1e6）；MLP 层 0–2 dense、层 3–91 MoE（**160 路由专家 top-8 + 1 共享
专家**）。

**家族特异折叠一览**：

- **fused QKV + bias**（`:193-218`）：`q/k/v` 三个权重用 `shuffle_tensors([wq,wk,wv], NUM_KV_HEADS,
  0)` 按 KV 头交织成一个 `qkv_proj`；bias 同样交织后**tile 成 `[mbt, FUSED]` 当 residual 输入**
  骑在 `linear_with_residual_layer` 的残差位上（`:206-218`）。注意 builder 会 `state_dict.pop`
  掉已融合的原权重/bias（`:204-205/213-214`）。
- **部分 RoPE + qk-norm 的 paged attention**（`:220-237`）：`paged_attention_layer(..., rotary_dim=64,
  qk_norm_eps=1e-5, grid_dim=(max_num_batched_requests, NUM_KV_HEADS, 1))`。
- **家族专用路由 kernel `glm_moe_router_layer`**（`:305-311`）：sigmoid+bias top-k（`n_group=1`），
  权重是**无偏 sigmoid 分数归一化 × `routed_scaling_factor=2.5`**（docstring `:13-14`），用
  `routed_scaling_factor=...`、`n_shared=...` 作参数。
- **共享专家折叠成 expert 160**（`:312-357`）：`w13`/`w2` 把共享专家的 `gate/up/down` **stack 到
  路由专家数组后面**（`:322-332`），共享专家"总是在 slot 8、权重 1.0"（docstring `:11` 注释
  `folded here as expert 160, always selected at slot 8 with weight 1.0`），于是**标准
  `moe_w13 / silu_mul / moe_w2 / mul_sum_add` 管线原样跑，不用改运行层**（`:339-357`）。
- `grid_for_rmsnorm_linear_layer` + 权重交织（`shuffle_tensors`）仍是前端"人肉把维度换算成网格"
  （对上 `docs/25` §4）。
- **v1 局限**（docstring `:16-17`）：`world_size==1`、BF16 权重；跳过 MTP nextn 层。
- **已知近似**（docstring `:19-24`）：hidden-state RMSNorm task 硬编码 eps 1e-6，而 GLM 用 1e-5
  （相对影响 ~5e-6，远低于 bf16 分辨率）；attention 的 q/k norm 用精确 1e-5。

## 3. Inkling：无 RoPE 相对位置 + 深度短卷积 + 滑窗/全局层

docstring（`inkling/builder.py:1-24`）：66 层、hidden 6144、64 Q 头、head_dim 128、**无 RoPE**；
attention 是 `q=q_norm(wq·x)`、`k=k_norm(k_sconv(wk·x))`、`v=v_sconv(wv·x)`，**相对位置 bias** 由
`r=wr·x`（d_rel=16/头）经 `rel_logits_proj` 给出；局部层 55 个（16 KV 头、滑窗 512）、全局层 11 个
（8 KV 头、无窗口、extent 1024）。**每层输出（attn 和 mlp）过一个深度短卷积（k=4, fp32）**再加外层
残差。MLP 层 0–1 dense、层 2–65 MoE（256 路由专家 top-6 + **2 个共享专家**）。

**家族特异折叠**：

- **`inkling_moe_router_layer`**（`:526-531`）：sigmoid+bias top-6，权重 =
  `softmax(logsigmoid(sel ++ shared)) × route_scale(8) × global_scale`（docstring `:13`），额外传
  `global_scale` 张量（`:523-525`）与 `route_scale`。
- **共享专家折叠成 expert 256/257**（`:533-568`）：`torch.cat([routed_w13, shared_w13], dim=0)`、
  `w2` 同理；共享专家"总在 slot 6/7"（docstring `:14-15`）。`moe_mul_sum_add_layer` 的 residual
  传 `self.zero_residual`——**真正的残差在外层 sconv 之后加**（`:570-575`）。
- **深度短卷积 `inkling_sconv_layer`**（`:594-614`）：`mlp_sconv.weight` 经 `_conv_w` 整形、卷积状态
  `conv_state` 清零，sconv 后再 `elementwise_add` 外层残差（`:608-614`）。attention 侧同样有 sconv
  段（`:290-458` 的 `_build_attention`）。
- **v1 局限**（docstring `:21-22`）：decode-only（`max_num_batched_tokens==1`）、`world_size==1`、
  BF16；跳过 `model.mtp.*`/audio/visual。

## 4. GPT-OSS：fused-QKV + sinks + 双 KV 流 + clamped SwiGLU

`gpt_oss/builder.py` 是"标准 GQA + MoE"家族里**工程折叠最密**的一份：

- **`plan_kv_cache` + KV groups**（`:17-47`）：GPT-OSS **两种 attention 各有一条 KV 流**——12 个滑窗
  层 + 12 个全 attention 层，**共享一个 12 槽页池**（docstring `:21-24`），由
  `kv_planner.plan_kv_groups` 在 `PersistentKernel` 构造前生成 `kv_groups` 与页表 meta tensor（对上
  `docs/25` §7 的 `kv_planner`）。
- **`_fused_qkv`**（`:142-156`）：q/k/v 权重与 bias 都用 `shuffle_tensors([...], num_kv_heads, 0)`
  按 KV 头交织。
- **`_experts`**（`:158-171`）：GPT-OSS 存 `gate_up` 为 `[E, hidden, 2I]` 且 gate/up **在最后一维
  交织**，`down` 为 `[E, I, hidden]`；builder 先 `permute(0,2,1)` 再 `cat([gu[:,0::2,:], gu[:,1::2,:]])`
  折成 MPK 期望的 `[E,2I,hidden]`（gate 在前半）。**这就是"家族存储布局 → MPK 布局"的手工转换。**
- **带 sinks 与窗口的 paged attention**（`:302-312`）：`paged_attention_layer(..., enable_qk_norm=False,
  window_size=(sliding_window if layer_types[i]=="sliding_attention" else 0), sinks=sinks,
  group_id=group_id)`——**`group_id` 指向该层所属的 KV 流槽位**（`:287-297`）。GPT-OSS 无 QK-norm，
  仍传两个 dummy（`:187-190`）。
- **o_proj 的双 addend 拆解**（`:314-330`）：注释（`:314-316`）——o_proj 既有 bias 又有两种加法，
  epilogue 只有一个槽位，于是**残差走 `with_residual`（因为必须恰好加一次、且在 TP allreduce 之前）、
  bias 另走一个 `elementwise_add`**。
- **`moe_topk_softmax_routing_layer`**（`:345-348`）+ **`moe_clamped_swiglu_layer`**
  （`:354-357`，`limit=..., alpha=1.702`）——GPT-OSS 的 SwiGLU 变体是
  `(clamp(up)+1)*clamp(gate)*sigmoid(gate*alpha)`（对上 `runtime_header.h:223-224` 的
  `TASK_CLAMPED_SWIGLU`）。
- `_grid_x`（`:15-24`）把输出列切成每 task 64 列且 ≥8（16B 对齐），MoE 的 grid 显式写
  `(10, _grid_x(2I,128), 1)` / `(8, _grid_x(h,64), 1)`（`:349-366`）。

## 5. 投机 draft builder：DFlash 与 Eagle3（不注册、追加到目标图）

这两份 builder **不是模型主干**，而是**由目标 builder / `demo.py` 实例化后，把一段草稿/验证子图
追加到目标 MPK 图上**（因此不进 `MPK.build()` 的名字表）。

- **`DFlashBuilder`**（`dflash/builder.py`，docstring `:1-40`）：构造参数里显式接
  `target_w_embed`/`target_w_lm_head`（**与目标共享词嵌入和 lm_head**，`:36-38`）；
  `build_step(target_hidden_dt, input_ids_dt, ctx_len, slot_start)` 每次解码返回草稿 token
  （`:43-48`）。docstring（`:14-20`）说明它是**单次非因果前向**（不是 Eagle3 的 K 步自回归），
  **拥有一个跨步累积上下文的 paged KV cache**。`build_yarn_cos_sin_cache`（`:162`）建 YaRN
  （mscale=1.4159）的 cos/sin。
- **`Eagle3Builder`**（`eagle3/builder.py`，docstring `:49-56`）：`Eagle3Builder(mpk, draft_state_dict,
  draft_config, ...)`，`build_draft_loop`（`:271`）编 **K 步自回归**的草稿 + verify。它还自带
  `_load_draft_state_dict`（`:37-47`，从 HF 或本地 `*.safetensors` 分片加载）。

**要点**：草稿/验证的 KV cache、RoPE cache、`eh_proj` 权重…… 全部是**追加到目标图上的额外张量与
task**；因为它们在**同一个 `PersistentKernel`** 上，投机解码的每一步也是**一跳常驻 kernel 内**的
事件依赖，而不是多次 host launch（对上 `docs/01` §5.8 与 `docs/09` 的端到端延迟）。

## 6. 家族差异 → 统一 task 管线的"折叠表"

把上面四份 builder 的家族特异部分压成一张表（**左侧是"家族原样"，右侧是 builder 折平后图上出现的
标准算子**）：

| 家族 | attention 家族形态 | builder 的折叠动作 | 折平后的图上算子 | 家族专用 `TaskType`（`docs/24`） |
|---|---|---|---|---|
| Qwen3 | GQA（`docs/25` 规范样本） | —（直接拼） | `linear`/`rmsnorm`/`rmsnorm_linear`/… | — |
| **DeepSeek-V3** | MLA（吸收后 1 KV 头） | **`kv_b_proj`→`q_b_proj` 吸收；V 反吸收融进 o_proj；FP8 UE8M0 重定标；k_pe pad 到 128** | `linear`/`paged`/`mla_kv_gather`/`mla_prefill`/`mla_mtp_decode`/`moe_w13_fp8`/`silu_mul`/`moe_w2_fp8`/`moe_mul_sum_add` | `TASK_MLA_*`/`TASK_MTP_*`（`runtime_header.h:153-189`） |
| GLM-4.6 | MHA + 部分 RoPE + qk-norm + bias | **fused QKV+bias**（bias 作 residual）；**共享专家折叠成 expert 160** | `linear_with_residual`/`paged_attention`/`glm_moe_router`/`moe_w13`/… | `TASK_GLM_MOE_ROUTER_SM100=401`（`:216-220`） |
| Inkling | 无 RoPE + 相对位置 + 短卷积 | **深度短卷积 sconv**；**共享专家折叠成 256/257**；外层残差后置 | `inkling_attention`/`inkling_moe_router`/`inkling_sconv`/`moe_w13`/… | `TASK_INKLING_{SCONV,MOE_ROUTER,ATTENTION}_SM100=351/352/353`（`:207-215`） |
| GPT-OSS | 滑窗+全 attention 双 KV 流 + sinks | **gate_up 交织布局重排**；**KV 双流共享页池**；o_proj 残差/bias 拆位 | `paged_attention`(window/sinks/group_id)/`moe_topk_softmax`/`moe_clamped_swiglu`/… | `TASK_CLAMPED_SWIGLU=451`（`:221-225`） |

**这张表就是"`docs/24` 家族 ID 块的前端生产者"**（`docs/25` §6 已点出 `docs/24` 的家族 ID 块
350–399 Inkling / 400–449 GLM / 450–499 GPT-OSS，本文给出每块的**具体前端函数**）：

- Inkling：`_build_attention`→`inkling_attention_layer`、`_build_moe_mlp`→`inkling_moe_router_layer`
  + `inkling_sconv_layer`；
- GLM：`_build_moe_mlp`→`glm_moe_router_layer`；
- GPT-OSS：`build_layers`→`moe_clamped_swiglu_layer`；
- DeepSeek：`_build_mla_attention_layer`→`mla_*`、`_build_mtp_layer`→`mtp_*`（这些在 SM100 通用块
  153–189，不是家族块，因为 MLA/MoE/MTP 是 DeepSeek 系被多家族复用的）。

**读法**：`docs/24` 从"运行层 task 枚举"往回看，本文从"前端 builder"往前看——**两侧在
`register_task` 大表处汇合**。凡是有家族专用 `TaskType` 的，本文都能给出它的前端生产者函数；
凡是没有的（如 GPT-OSS 的 qkv/路由复用通用 task），就说明该家族只用手工张量变换折平、不新增 task。

## 7. 与论文 / 判据对照

| 论文（快照行） | 实现 | 差异 |
|---|---|---|
| §4.1 算子分解 operator decomposition（行 462–468） | 家族 builder 的 `_build_*` **手工**把 MLA/MoE/MTP 拆成 task；`grid_for_*`/`_moe_fp8_m_split` 手工定网格 | 论文叙述"编译器自动分解"；`mpk` 生产前端是**逐家族手写**（`docs/14` §5 同一落差，本文给出家族侧证据） |
| §4.2 每 task 的 CUDA 实现生成（行 1126–1146） | 家族专用 `TaskType`（MLA/MTP/inkling/GLM/clamped-swiglu）各有一套 device 实现 | 一致（`docs/07`/`docs/24`），但**device 实现本身也按家族分叉** |
| §5.8 每 batch size 一张静态图（局限，行 ~1800） | MTP 草稿步**编译期静态展开**、MTP 层权重 modulo 复用（`:1659-1660`） | 一致：投机草稿也进同一张静态图 |

**判据映射**（同 `docs/22` §8 / `docs/25` §8）：家族 builder 层**只贡献 C3 的范围**——它决定"哪些
家族算子进同一张图"，但**不 launch（C1）、不常驻（C2）、不做设备内协调（C4）、不做 SM 级调度
（C5）**。一个容易误解的点：**家族专用 `TaskType`（如 `TASK_INKLING_SCONV_SM100`）看起来像"运行层
特化"，其实它们的"什么时候跑、依赖谁"完全由运行层的通用 event/worker/scheduler 机制管**——task
体不同，调度机制不分家族。

一句话：**家族的复杂度落在 C3（融合范围）上，运行机制对所有家族是同一套。**

## 8. 局限与坐标

- **纯源码走读，未构建/运行**：所有结论基于 `f9eb70c` 快照行号；不采信性能数。端到端行为见
  `docs/09`（含 P25-1 的 launch 顺序根因）。
- **只逐行展开 DeepSeek-V3 的 `_build_mla_attention_layer`/`_build_moe_mlp`/`_build_mtp_*`**；
  GLM/Inkling/GPT-OSS 给出家族特异折叠的关键行，**未逐行走读其全部 `_build_*`**（尤其 Inkling 的
  `_build_attention:290-458` 的相对位置 bias 段、GPT-OSS 的 KV 流规划全细节）。对应的 device 实现
  （`task_register.cc` / `include/mirage/persistent_kernel/tasks/`）见 `docs/24` §7 明留的同一尾巴。
- **`demo.py` 的权重预处理（Phase-2 吸收、gate_up 交织拆分）未走读**：DeepSeek-V3 多处注释指向
  `demo.py` 的 load-time 处理（`:592-594`、`:806-807`、`:1078`），本文只覆盖 builder 侧契约。
- **FP8/量化的数值正确性未验证**：`_requantize_fp8_for_ue8m0` 的"2× 误差"是源码注释，本文未独立复现。
- **行号已核对**：`deepseek_v3/builder.py`(2255)、`glm4_moe/builder.py`(454)、`gpt_oss/builder.py`(374)、
  `inkling/builder.py`(705)、`dflash/builder.py`(364)、`eagle3/builder.py`(461)、
  `include/mirage/persistent_kernel/runtime_header.h`(462)。

## 9. 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
  主文件：`python/mirage/mpk/models/deepseek_v3/builder.py`（`build_from_config:103-116`、
  `_requantize_fp8_for_ue8m0:475-542`、`_attach_fp8_weight:549-561`、
  `_build_mla_attention_layer:563-792`、`_build_dense_mlp:794-830`、`_build_moe_mlp:832-1160`、
  `_build_mtp_decoder_layer:1183-1261`、`_build_mtp_layer:1649-...`、`build_layers:2049-2128`、
  `build_from_dict:2130-2255`）；`python/mirage/mpk/models/glm4_moe/builder.py`
  （`_build_attention:181-246`、`_build_moe_mlp:274-357`）；`python/mirage/mpk/models/inkling/builder.py`
  （`_build_moe_mlp:491-575`、`_build_mlp:577-614`）；`python/mirage/mpk/models/gpt_oss/builder.py`
  （`plan_kv_cache:17-47`、`_experts:158-171`、`build_layers:262-367`）；
  `python/mirage/mpk/models/{dflash,eagle3}/builder.py`；
  `include/mirage/persistent_kernel/runtime_header.h`（家族 ID 块 `:207-225`、MLA/MTP `:153-189`）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462–468、
  §4.2 行 1126–1146、§5.8 局限）。
- 同专题相关篇：`docs/25`（本文补其 §9 尾巴）、`docs/24`（task 目录，本文给出家族专用 `TaskType`
  的前端生产者）、`docs/14`（前端层 API）、`docs/08`（依赖分析器消费这些登记）、`docs/17`（多卡，
  本文 §1.5 的 allreduce/nvshmem 对照）、`docs/07`（单 task device 实现）、`docs/23`（总纲）、
  `docs/02`（DeepGEMM MoE/MLA 生产对照，跨 `../mega-moe/` 专题）。
