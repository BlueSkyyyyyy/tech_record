# MPK 权重预处理层走读：从 HF checkpoint 到 builder 的 state_dict

> 本文补齐 `docs/26`（P112-1）**§8 明留的尾巴**：
>
> > **`demo.py` 的权重预处理（Phase-2 吸收、gate_up 交织拆分）未走读**：DeepSeek-V3 多处注释指向
> > `demo.py` 的 load-time 处理（`:592-594`、`:806-807`、`:1078`），本文只覆盖 builder 侧契约。
>
> `docs/25` 讲「谁把 95 个 `*_layer()` 拼成模型」，`docs/26` 讲「家族 builder 如何把结构差异折平成标准
> task」。两篇都**默认 `state_dict` 已经是 builder 想要的样子**——但真实 checkpoint（HuggingFace 原始
> FP8 分块量化权重）**并不是**。中间隔着一层谁都没走读的 **权重预处理层**：MLA 权重吸收、V 反吸收融进
> `o_proj`、gate/up 交织、MoE 专家折叠、FP8 重定标、TP 切分。
>
> headline——**这一层在 Python 里用普通 PyTorch 张量变换把「HF 权重布局」改写成「builder 契约布局」**，
> 且存在**两条独立实现**：离线 `convert.py main()`（写 per-rank safetensors）与 demo 内联
> `Phase 1 / Phase 2`（`--layers` 选择性加载时才跑）。运行层看到的仍是标准 `linear`/`silu_mul`/`moe_w13`，
> 但它们的**权重长什么样、在哪个 rank、什么精度**，全由这一层决定。
>
> 本文是**纯源码走读（资料分析）**，不构建、不运行 mirage，不引用任何性能数。快照
> `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/26` 同一快照）。

## 0. 定位：这一层在编译栈里的坐标

```
HF checkpoint (FP8 block-quant, HF 键名)
        │   ← 本文走读的「权重预处理层」
        ▼
builder 契约 state_dict (吸收后 / 交织后 / 折叠后 / 已切分)
        │   docs/26：家族 builder 把结构折平成标准 task
        ▼
bared task 图 → annotated_graph → runtime → 常驻 megakernel (docs/06/08)
```

**为什么必须有这一层**（不是可有可无的「加载脚本」）：

1. **MLA 权威吸收**是数学等价改写（`kv_b_proj` 折进 `q_b_proj`、V 折进 `o_proj`），builder 消费的是
   **改写后的权重**，HF 权重直接喂进去会多一次每步投影、且布局对不上（`docs/26` §1）。
2. **FP8 分块 scale** 与 task 期望的 `scale_inv` 键名/形状是约定，需要搬运与重定标。
3. **gate/up 交织**：`silu_mul` kernel 假设「每个输出块前半是 gate、后半是 up」，这是**布局契约**，
   不是数值问题（§4）。
4. **TP 切分**：多卡下每个 rank 只该持有该拿的切片；切错维度 = 数值错（§6）。

两条实现路径（关键事实，后面反复用）：

| 路径 | 代码 | 何时跑 | 输出 | FP8? |
|---|---|---|---|---|
| **离线转换** | `demo/deepseek_v3/models/convert.py::main` `:302-427` | 事先 `python convert.py --hf-ckpt-path … --model-parallel mp` | `model{rank}-mp{mp}.safetensors` | **反量化成 BF16**（`:337-347`） |
| **内联转换** | `demo/deepseek_v3/demo.py::Phase 1/2` `:438-691` | 仅当 `--layers` 选择性加载（`_need_conversion = bool(args.layers)` `:434`） | 内存里的 `state_dict` | **保留 FP8 + `scale_inv`**（`:531`） |

`demo.py` §加载：优先读离线产出的 `model{rank}-mp{world_size}.safetensors`（`:367-371`）；没有才走
`model.safetensors.index.json` 的**选择性加载**（`:374-401`，按 `--layers` 过滤 shard/键），此时
`_need_conversion=True`，触发 Phase 1/2。**所以两条路径一般只走其一**：整模型用离线转换、调试小层子集
用内联转换。

## 1. 输入契约：HF FP8 分块量化权重的键与 scale

预处理层的输入是 HF 键名 + FP8 权重 + 伴随 scale。三个支撑谓词（`convert.py`）：

- `is_fp8(t)` `:298-299`：`dtype in (float8_e4m3fn, float8_e4m3fnuz)`。
- `find_scale_for_weight(name, scales)` `:272-291`：给权重名找 scale，按 `name+"_scale_inv"` /
  `name+"_scale"` / 去掉 `.weight` 再拼后缀的顺序探测——DeepSeek-V3 存的是 `weight_scale_inv`。
- `dequantize_fp8(weight, scale, target_dtype)` `:75-123`：**三档 scale 形态**分别处理——per-tensor
  标量（`:105-107`）、per-output-channel `[out]`（`:108-110`）、block-wise `[ceil(out/128),ceil(in/128)]`
  （`:111-118`，`repeat_interleave(128)` 展开成逐元素再乘）。

`collect_all_tensors` `:240-269` 把 safetensors 里的键分成 `weights` 与 `scales`（`*_scale_inv` /
`*_scale` 结尾的进 scales）。

> 注意：**离线路径与内联路径对 FP8 的处理相反**。离线 `main()` 第 2 步统一 `dequantize_fp8(...,
> target_dtype=bfloat16)`（`:337-347`）把整模型变 BF16 再吸收/切分/存盘；内联路径**保留 FP8**，
> 只在吸收的中间步骤按需 `dequantize_fp8` 出 FP32（见 §2），因为 builder 的 FP8 GEMM 管线要用
> `scale_inv`（`:529-533` 注释明说）。

## 2. MLA 权重吸收：`absorb_kv_into_q` + V 反吸收进 `o_proj`

### 2.1 核心数学（`convert.py:150-233`）

`absorb_kv_into_q(q_b_proj_weight, kv_b_proj_weight, params)` 做三件事：

1. reshuffle 成 per-head：`q_b → [H, q_head_dim=192, q_lora_rank]`，拆 `q_nope[.., :128, ..]` /
   `q_rope[.., 128:, ..]`（`:204-206`）；`kv_b → [H, kv_head_dim=256, kv_lora_rank]`，取
   `k_nope[.., :128, ..]`（`:210-211`）。
2. 每头吸收：`q_absorbed_nope = k_nopeᵀ @ q_nope`，形状 `[H, kv_lora_rank=512, q_lora_rank]`
   （`bmm` `:217-220`）。
3. concat 回 rope：`[q_absorbed_nope | q_rope] → [H, 512+64=576, q_lora_rank]`，展平成
   `[H*576, q_lora_rank]`（`:225-231`）。

效果：注意力直接对**压缩 KV 潜向量 c_KV** 打分，不必每头物化完整 K——这正是 MLA 的「吸收」惯用法
（builder 侧的对应消费见 `docs/26` §1）。

### 2.2 V 反吸收融进 `o_proj`

`kv_b_proj` 的另一半（V 部分）不能吸收进 Q，而要**折进 `o_proj`**。demo 内联实现（`demo.py:505-524`）：

- 从 `kv_b` 取 V 部分 `W_UV = kv_b_reshaped[:, :v_dim, :]`，`[H, v_dim, kv_lora_rank]`（`:505-507`）；
- `o_proj` 反量化成 BF16（`:510-516`），reshape `[hidden, H, v_dim]`（`:518`）；
- `o_fused = einsum('dhn,hnk->dhk', o_reshaped, W_UV)`（`:519`），把 `o_proj` 从 `[hidden, H*128]`
  收成 `[hidden, H*512]`（`:520`）；
- 再**重新分块量化成 FP8** 写回 `state_dict[o_key]` + `o_s_key`（`:521-523`）。

### 2.3 吸收后拆 `q_b_nope` / `q_b_pe`（chunked-prefill 双形态）

吸收后的 `q_b` 还要按 per-head `[nope(512) | pe(64)]` 拆成两个独立张量（`demo.py:495-504` 与
`:615-627` 两处，逻辑相同）：

```
absorbed_r = absorbed.reshape(H, 576, -1)
q_b_nope = absorbed_r[:, :512, :].reshape(H*512, -1)
q_b_pe   = absorbed_r[:, 512:, :].reshape(H*64, -1)
```

原因（`:495-498` / `:615-620` 注释）：**chunked-prefill 的 MLA kernel 要 `Q_nope`/`Q_pe` 分开**，
而 **decode kernel 消费融合的 `[H*576]`**。两套形态都留下，builder 按 `max_num_batched_tokens`
运行期二选一（也即 `docs/26` §1 的 dual-dispatch 前端契约）。

### 2.4 一个「已实现未跑」的分支：Phase 2 的吸收块

`demo.py` 的 Phase 1（`:438-527`）与 Phase 2（`:529-691`）**都**包含吸收代码。但 Phase 1 在
`:525` `del state_dict[kv_key]`，而 Phase 2 的吸收块入口是
`if q_key in state_dict and kv_key in state_dict:`（`:552`）——`kv_key` 已被删，**Phase 2 的吸收块
（`:553-632`）在本流程下不会执行**。Phase 2 真正生效的是**紧接着的 gate/up 融合（`:634-661`）与
专家折叠（`:663-691`）**。这与本专题反复见到的「已实现未启用 / 同义实现有两份」是同一型
（`docs/16`/`docs/20`/`docs/22` §6）：**Phase 2 的注释（`:544`「Absorption already done in Phase 1」）
与代码里的重算块并存**，读代码时须以「谁先删键」判实际路径。

## 3. FP8 重定标：`_quantize_f32_to_checkpoint_fp8` 与 checkpoint 约定一致

吸收/V 融合都在 FP32 里做，结果要写回 checkpoint 的 **FP8 + 分块 `scale_inv`** 约定。demo 内的
`_quantize_f32_to_checkpoint_fp8`（`:452-466`）：

```
pad 到 128 的倍数 → reshape [bM,128,bK,128] → amax over (1,3) → scale_inv = amax/448
→ w_q = clamp(blocks/scale, ±448) → 展平裁回原形状 → to(float8_e4m3fn)
```

要点：**448** 是 E4M3 可表示上界的「安全值」，采用 **128×128 二维块 + `amax/448`** 的 `scale_inv`，
与 `dequantize_fp8` 的块状分支（§1）**互逆**——所以吸收重定标后的权重，仍能被同一套加载逻辑
（`_attach_fp8_weight` / `dequantize_fp8`）消费。这是「重定标不破坏契约」的关键。

## 4. gate/up 交织：为什么不是 `torch.cat`

`silu_mul` kernel 的布局假设（`demo.py:634-639` 注释原文）：每个输入块**前半是 gate、后半是 up**。
FP8 linear 把输出维按 `grid_for_rmsnorm_linear_layer` 切成 N 个 task，于是 silu_mul 用 N/2 个 task
读「前半块」。

必须**按 chunk 交织**而不是 `torch.cat`：

```
split = grid_for_rmsnorm_linear_layer(out_dim_total) // 2
gate_up = shuffle_tensors([g, u], split=split, dim=0)
```

`shuffle_tensors`（`python/mirage/mpk/models/utils.py:50-128`）按 `split` 把每个张量等分，再**按 tensor
顺序逐 chunk 交错**（`:118-126`）：`[g0,u0, g1,u1, ...]`。`torch.cat` 会得到 `[g 全, u 全]`，与 kernel
的「逐块前后半」不符 → 数值错。scale 也要按**同一 split** 交织（`:653-661`，scale 行数 = 权重行数/128）。

对照 **Qwen3 demo**（`demo/qwen3/demo.py`）用 **MPK 对象方法**做同一件事：
`mpk.shuffle_tensors(inputs=[w_gate_proj,w_up_proj], shuffled_dim=0, num_groups=rmsnorm_num_tasks//2, ...)`
（`:708-714`），QKV 同理（`:563-568`，`num_groups = num_key_value_heads // world_size`）。方法定义在
`persistent_kernel.py:811-816`（断言 `shuffled_dim==0`），底层走 `kn_graph.shuffle_tensors`。**注意
与 `models/utils.py` 的 `shuffle_tensors` 是两套同名不同签名**：前者是 `(tensors, split, dim)`、后者是
`(inputs, shuffled_dim, num_groups)`，语义等价（chunk 数 vs 组数）、载体不同（纯 torch vs 图算子）。

## 5. MoE 专家折叠：`experts.w13` / `experts.w2`

Phase 2 把「每专家三个投影」折成两个堆叠张量（`demo.py:663-691`）：

```
w13[e] = cat([gate_proj[e], up_proj[e]], dim=0)   # 融合 gate+up
w2[e]  = down_proj[e]
→ state_dict["...experts.w13.weight"] = stack(w13_list)   # [E, 2*inter, hidden]
  state_dict["...experts.w2.weight"]  = stack(w2_list)    # [E, hidden, inter]
```

scale 同样 `cat`/`stack`（`:681-691`）。折叠后与 builder 的 `moe_w13`/`moe_w2` task 契约对齐
（`docs/24`/`docs/26`）。**共享专家**不在这里折——它保持 `shared_experts.gate_proj`/`up_proj` 分开，
由 builder 在 build time 自己做交织（`:749-753` 注释明说），这也是 `docs/26` §2/§3 讲的「共享专家折叠
成 slot」。

## 6. 对齐 + TP 切分：两条路径的 shard 规则**不一致**

### 6.1 对齐

`:693-708`：对每个张量 `contiguous()` + 保证 `data_ptr() % 16 == 0`（TMA 需要 16 B 对齐，见「已知坑」）。
TP>1 时**先留 CPU**（`:693-697` 注释），切分后再把小的分片搬 GPU，避免「全量未切权重 × rank 数」OOM；
搬 GPU 在 `:800-807`。

### 6.2 内联路径的后转换 shard 规则（`demo.py:719-760`）

`_TP_SHARD_RULES` 按**吸收/融合后的键**匹配：`dim=0` 行并行（切输出）、`dim=1` 列并行（切输入）、
`None` 复制。要点：

- `embed_tokens`/`norm`/`lm_head` **复制**（`:720-722`，argmax/查表要全 vocab）；
- `q_a_proj`/`q_a_layernorm` 复制（`:723-724`，输出喂满宽 `q_b_proj`）；`q_b_proj`/`q_b_nope`/`q_b_pe`
  切 dim0（`:725-731`）；`o_proj` 切 dim1（`:734`）；
- dense `gate_up_proj` 切 dim0、`down_proj` 切 dim1（`:737-738`）；
- **MoE 专家 w13 切 dim1、w2 切 dim2**（`:741-748`，每 rank 持有全部专家、专家内张量并行，AllReduce
  收 partial）——这条与 builder 的 `moe_intermediate_size = FULL//world_size` 相互对应（`:746` 注释）；
- **w13 特殊处理**（`:774-787`）：w13 是 `cat([gate,up], dim=1)`，naive 切 dim1 会让一个 rank 全 gate、
  另一个全 up → **先拆 gate/up 两半、各自切、再 re-cat**。

规则表用 `_get_tp_shard_dim`（`:763-767`）正则匹配，未命中默认复制。

### 6.3 离线路径的 shard 规则（`convert.py:439-499`）

| 键 | 离线 `_SHARD_RULES` | 内联 `_TP_SHARD_RULES` |
|---|---|---|
| `embed_tokens.weight` | **dim0**（切） | **None**（复制） |
| `lm_head.weight` | **dim0** | **None** |
| `q_a_proj.weight` | **dim0** | **None** |
| `q_b_proj.weight` | dim0 | dim0 |
| `o_proj.weight` | dim1 | dim1 |
| 专家 `experts.\d+.gate/up/down` | **None**（复制） | **w13 dim1 / w2 dim2**（切） |
| 共享专家 gate/up | dim0 | dim0 |
| shared_head.head | dim0 | None |

**这是一处真实的不一致**：离线路径切 embedding/lm_head/q_a_proj、复制专家；内联路径相反。二者面向
**不同的下游消费形态**（离线面向「BF16 全量转换」主路径，内联面向「保留 FP8 的调试子集」），且内联
规则是**吸收后**键、离线规则是**吸收前**键（离线在吸收**之前**按原键切、再吸收；内联先吸收后按
新键切）。本文只**如实记录**两条规则不同，**不判定谁对**——真正确性以端到端输出为准，而 P25-1 的端到端
路径当前依赖 scheduler-first 构建（见 `docs/09` §11），故此处不做数值断言。

## 7. 与 builder 的契约：这一层的输出键

预处理层产出的 `state_dict` 键（DeepSeek-V3）与 builder 的消费点（`docs/26` §1）：

| 预处理产出键 | 来源 | builder 消费 |
|---|---|---|
| `self_attn.q_b_proj.weight`（吸收后 FP8） | §2.1/§3 | `_attach_fp8_weight` → `_fp8_linear` |
| `self_attn.q_b_nope.weight` / `q_b_pe.weight` | §2.3 | chunked-prefill MLA 的两个 dense Q |
| `self_attn.o_proj.weight`（V 已融合，可能 FP8） | §2.2 | attention 输出投影 |
| `mlp.gate_up_proj.weight` + `.weight_scale_inv`（交织） | §4 | `silu_mul` 前后半 |
| `mlp.experts.w13.weight` / `w2.weight` + scale | §5 | `moe_w13` / `moe_w2` |
| 其余（norm/router/共享专家…） | 直传 | 各标准 layer |

**对照 GLM-4.6 demo**（`demo/glm4_moe/demo.py`）：**没有内联预处理**——`load_state_dict`
(`:58`) 拿到的就是 HF 权重，直接 `Glm4MoeBuilder(...).build_from_dict(state_dict)`（`:248-249`）。
GLM 的 fused-QKV、共享专家折叠全在 **builder 内部 build time** 完成（`docs/26` §2）。所以「权重预处理」
在 GLM 系是**空层**、在 DeepSeek-V3 系是**厚层**——**差异来自 MLA 吸收与 FP8 分块量化这两件事只有
DeepSeek 系需要**。

## 8. 判据 C1–C5 映射

与 `docs/14`/`docs/25`/`docs/26` 同口径：权重预处理层**不贡献任何 C 判据**，它只是 **C3（融合范围）
的输入准备**——决定「进同一张图的 task 拿到的权重是什么形状/精度/在哪个 rank」。

| 判据 | 本层是否贡献 | 说明 |
|---|---|---|
| C1 单次 launch | 否 | 纯 host 侧张量变换，不 launch |
| C2 常驻 | 否 | 一次性的 load-time 工作 |
| C3 多算子范围 | 否（只准备输入） | 它把权重折平，但**融合边界**由 builder 定（`docs/26`） |
| C4 设备内协调 | 否 | 无 device 逻辑 |
| C5 SM 级调度 | 否 | 无 worker/scheduler |

一句话：**权重预处理层把「checkpoint 的物理布局」对齐到「builder 的逻辑契约」，是 megakernel
编译栈最上游、最不像 kernel 的一层，却是 MLA/FP8 模型能进图的前提。**

## 9. 局限与坐标

- **纯源码走读，未构建/未运行**：所有结论基于 `f9eb70c` 快照行号；不采信性能数、不做数值断言。
- **只逐行展开 DeepSeek-V3 的内联预处理**（`demo/deepseek_v3/demo.py` + `models/convert.py`）；GLM 的
  预处理是空层、Qwen3 的 `shuffle_tensors` 只给出两处使用点（`:563`/`:709`），**未逐行走读
  `kn_graph.shuffle_tensors` 的图算子内部**（属 `docs/14` 前端层）。
- **两条 shard 规则的不一致只记录、不判对**（§6.3）；离线路径的数值正确性未独立复现。
- **MTP 层的预处理未单独展开**：`absorb_layers.append(num_layers)`（`:449-450`/`:546-547`）把 MTP 层
  一并纳入吸收；MTP 权重的 modulo 复用细节见 `docs/26` §1（`:1659-1660`）。
- **行号已核对**：`demo/deepseek_v3/demo.py`(975)、`demo/deepseek_v3/models/convert.py`(548)、
  `python/mirage/mpk/models/utils.py`(156)、`python/mirage/mpk/persistent_kernel.py`(shuffle_tensors
  `:811-816`)、`demo/glm4_moe/demo.py`(286)。

## 10. 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
  主文件：`demo/deepseek_v3/demo.py`（`use_mirage` 分支 `:258`、加载 `:344-425`、
  `_need_conversion:434`、Phase 1 `:438-527`、`_quantize_f32_to_checkpoint_fp8:452-466`、
  `absorb_kv_into_q:486`、V 融合 `:505-524`、Phase 2 `:529-691`、gate/up 交织 `:634-661`、
  专家折叠 `:663-691`、对齐 `:693-708`、TP 规则 `:719-760`、切分 `:770-798`、搬 GPU `:800-807`、
  builder `:809-827`）；`demo/deepseek_v3/models/convert.py`
  （`dequantize_fp8:75-123`、`shard_tensor:126-143`、`absorb_kv_into_q:150-233`、
  `find_scale_for_weight:272-291`、离线 `main:302-427`、`_SHARD_RULES:439-499`、`_get_shard_dim:505-517`）；
  `python/mirage/mpk/models/utils.py`（`grid_for_rmsnorm_linear_layer:3-15`、`shuffle_tensors:50-128`）；
  `python/mirage/mpk/persistent_kernel.py`（`attach_input:746`、`shuffle_tensors:811-816`）；
  `demo/qwen3/demo.py`（QKV 交织 `:563-568`、gate/up 交织 `:708-714`）；
  `demo/glm4_moe/demo.py`（`load_state_dict:58`、`build_from_dict:248-249`）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 算子分解行 462–468、
  §4.2 每 task 生成行 1126–1146、§5.8 每 batch 一张静态图局限）。权重布局本身论文未提，属**实现细节**。
- 同专题相关篇：`docs/26`（本文补其 §8 尾巴）、`docs/25`（模型组装，本文补其上游）、`docs/24`（task
  目录，本文给出其权重输入形态）、`docs/14`（前端 API `shuffle_tensors`/`attach_input`）、`docs/07`
  （单 task device 实现，本文权重最终喂给它）、`docs/23`（总纲）。
- 交叉专题：DeepGEMM mega 家族的 MoE 前反向权重量化/布局见 `../mega-moe/`（本专题不重复）。
