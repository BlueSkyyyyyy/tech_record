# MPK 前端的模型组装与请求/元数据管理层走读：谁把 95 个 layer 拼成一个模型

> 本文补齐 `docs/14`（P100-1）**§7 明留的尾巴**：
>
> > **未走读模型 builder 的完整组装逻辑**：`models/{qwen3,glm4_moe,deepseek_v3,dflash,eagle3,gpt_oss,
> > inkling}/builder.py` 只以 Qwen3 的 `build_layers` 为例，未覆盖分布式/张量并行切分
> > （`multigpu.py`）与请求/元数据管理（`mpk.py` 的 `MPKMetadata`）。
>
> `docs/14` 打开的是「一个 `*_layer()` 如何登记成 task」（前端**层** API），`docs/24` 打开的是
> 「`register_task` 那张 name→`TaskType` 大表」（**能力上界**）。但两者之间还有个更上一层的问题：
> **是谁、按什么顺序、带着哪些切分/缓冲/多卡决策，把 95 个手工 `*_layer()` 调成一个完整模型的？**
> 答案就在 `python/mirage/mpk/`：一套「**手写作图器（model builder）+ 元数据/请求管理层**」。
> 这正是论文 §4.1「operator decomposition」在 `mpk` 生产路径里的最后一环——**模型架构知识被写成了
> Python 类**，而不是由编译器从高层算子自动搜出来（后者见 `docs/15`，生产前端不调）。
>
> 本文是**纯源码走读（资料分析）**，不构建、不运行 mirage，不引用任何性能数。快照
> `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/24` 同一快照）。

## 0. 定位：这一层在编译栈里的位置

把前几篇串起来看，一条从「模型」到「常驻 mega-kernel」的链路是：

```
 模型架构知识(手写)      层 API             task 大表          编译/运行
 ─────────────────   ───────────────   ─────────────────   ──────────────
 models/<fam>/        mpk.<*_layer>()    graph.cc 95 分支     annotated_graph /
 builder.py      ──►  (docs/14)      ──►  (docs/24)       ──► runtime / worker
   │                                        │                 (docs/06/08)
   │ MPK.metadata / MPKMetadata              │
   └── 请求/元数据/多卡/模式 ─────────────────┘
```

- **`builder.py`（本文主角）** 决定「**模型长什么样**」：层序、每层的 grid/block、权重交织/吸收、
  哪些中间张量落在常驻 kernel 的显存、哪些跨卡（`io_category`），以及**要不要插 AllReduce**。
- **`mpk.py` / `MPKMetadata`（本文主角）** 决定「**这一次运行的环境**」：serving 模式、batch/seq/page
  上限、meta tensors 指针、多卡 world/rank、profiling、sampling、spec decode。
- 二者都**只写 Python / 只登记任务**，真正的 C1–C5 判据（单次 launch / 常驻 / 设备内协调 / SM 级
  调度）在**运行层**才拿到（`docs/22` §8 的关键修正）。本层只决定 **C3 的范围**（哪些算子进同一张图）
  与 C1 的**前提**（所有算子先得被登记进同一张 `KNGraph`）。

一句话：**megakernel 融合的边界，是这一层用 Python 手写死的。**

## 1. 注册与发现：一个模型 = 一个手写 builder 类

### 1.1 `model_registry.py`：字符串 → builder 类

`model_registry.py:5-23` 是一个注册装饰器，把类登记到模块级字典 `_MODEL_BUILDERS`（`:3`）；同一个类
可注册**多个别名**（HF repo id、短名等）。`get_builder(name)`（`:25-29`）按键取类，取不到就抛错并列出
已登记的名字。

```python
# model_registry.py:5-23（节选）
def register_model_builder(*names):
    def decorator(cls):
        for n in names:
            _MODEL_BUILDERS[n] = cls
        return cls
    return decorator
```

各家族在 `builder.py` 顶部用装饰器声明别名，例如（行号已核对）：

| 家族 | 文件:行 | 注册别名（节选） |
|---|---|---|
| Qwen3 | `models/qwen3/builder.py:12` | `Qwen3`、`Qwen/Qwen3-8B`、`Qwen/Qwen3-1.7B` … |
| GLM-4.6 | `models/glm4_moe/builder.py:66` | `Glm4Moe`、`zai-org/GLM-4.6`、`glm4_moe`、`glm-4.6` |
| DeepSeek-V3 | `models/deepseek_v3/builder.py:63` | `deepseek-v3`、`DeepSeek-V3`、`deepseek-ai/DeepSeek-V3` |
| GPT-OSS | `models/gpt_oss/builder.py:47` | `gpt_oss`、`GptOss`、`openai/gpt-oss-20b` |
| Inkling | `models/inkling/builder.py:70` | `Inkling`、`thinkingmachines/Inkling`、`inkling` |

注意：`models/dflash/builder.py`、`models/eagle3/builder.py` **不注册**（grep 无 `register_model_builder`
命中）——它们是**草稿模型（draft）builder**，由目标模型的 builder 在需要时实例化（speculative decoding），
不是可独立 `get_builder()` 的顶层模型。

### 1.2 `models/__init__.py`：自动导入即注册

`models/__init__.py:1-17` 在包导入时用 `Path(__file__).parent.rglob("builder.py")` 遍历所有家族的
`builder.py` 并 `importlib.import_module`，**导入触发装饰器执行 = 完成注册**（任何 `ImportError` 直接
向上抛）。所以「加一个新模型」= 在 `models/<新家族>/` 放一个带 `@register_model_builder(...)` 的
`builder.py`，无需改注册表。

### 1.3 `MPK.build()`：从名字到实例

`mpk.py:471-480`：

```python
def build(self):
    model_builder_class = get_builder(self.model_name)
    self.model_builder = model_builder_class(self.persistent_kernel)
    if self.weight_from_model:
        self.model_builder.build_from_model(model_name=self.model_name)
        self.tokenizer = self.model_builder.tokenizer
    else:
        self.model_builder.build_from_config(self.model_config)
    self.is_built = True
```

两条入口（`weight_from_model` 开关）：
- `weight_from_model=True`：builder 自己去 HuggingFace 拉权重、建 tokenizer（`build_from_model`）。
- `weight_from_model=False`：调用方通过 `MirageModelConfig` 传入 `state_dict` / `k_cache` / `v_cache` /
  `position_embeddings`（`build_from_config`）——这是**离线/嵌入**用法。

`MirageModelConfig`（`models/graph_builder.py:7-41`）与抽象基类 `GraphBuilder`（`:44-51`，只强制实现
`build_from_model`）定义了这层的最小契约。两者把「模型配置」与「权重」分开：配置走 dataclass，
权重走 `state_dict`。

## 2. `MPKMetadata` / `MPK`：请求、元数据与运行模式

### 2.1 `MPKMetadata` 字段分组（`mpk.py:15-83`）

这个 dataclass 是「一次 MPK 会话的全部外部状态」，按用途可分成六组：

| 组 | 字段（行） | 作用 |
|---|---|---|
| **serving 参数** | `mode`、`total_num_requests`、`max_seq_length`、`max_num_batched_requests`、`max_num_batched_tokens`、`max_num_pages`、`page_size`、`max_sm_num`、`device`（`:19-28`） | 决定 kernel 的编译期容量常量与可用 SM 数 |
| **模型/权重** | `weight_from_model`、`model_name`、`model_path`、`model_config`（`:30-32,51`） | 选 builder 与权重来源 |
| **多卡** | `world_size`、`rank`、`num_workers`、`num_schedulers`（`:34-37`） | 见 §5 |
| **meta tensors** | `step`、`tokens`、`input_tokens`、`output_tokens`、`num_new_tokens`、`prompt_lengths`、`qo_indptr_buffer`、`paged_kv_*`（`:39-49`） | 传进设备侧运行时的缓冲区指针 |
| **profiling / pinned ring** | `profiling`、`profiler_tensor`、`trace_name`、`pinned_ring_capacity`、`pinned_*`（`:53-70`） | 性能剖析与在线锁页环 |
| **spec decode / sampling** | `spec_decode`、`spec_decode_config`、`do_sample`、`temperature`、`top_p`、`top_k`、`sampling_seed`、`sampling_topk_max`（`:72-82`） | 采样与投机解码配置（**编进图里**，一进程一套） |

`check_valid()`（`:84-90`）区分两条路径：`weight_from_model=True` 要求 `model_name`/`model_path` 至少一个；
否则要求 `model_config` 里 `state_dict`/`k_cache`/`v_cache` 都非空。

### 2.2 `MPK.__init__` 的初始化顺序（`mpk.py:134-296`）

1. `init_mpi()`（`:298-312`）：尝试 `from mpi4py import MPI` 拿 world/rank；`world_size>1` 才
   `dist.init_process_group(backend="nccl", ...)`。单机（无 mpi4py）时 world=1/rank=0。
2. `torch.set_default_dtype(torch.bfloat16)` + `torch.cuda.set_device(self.rank)`（`:150-151`）。
3. 处理 `step`/`input_tokens`：**若传入的是大 buffer 的切片/视图，重建整块 buffer**
   （`storage_offset()>0` 或 numel 小于 storage 时用 `set_(untyped_storage(), 0, (storage_size,))`，
   `:161-198`）。这是为 vLLM-MPK 集成里「把整块 buffer 交给 kernel、按 batch 动态用前缀」准备的。
4. `get_tensors()`（`:354-375`）：构造 profiler tensor、`compensate_meta_tensors()`（`:314-352`）为
   **任何 None 的 meta tensor 补一块占位 GPU 张量**（`step`/`tokens`/`input_tokens`/… /
   `paged_kv_*`；源码自注 `TODO: Ideally we should only allocate tensors we need.`）。
5. 若 `num_workers<=0 || num_schedulers<=0`，调 `get_configurations_from_gpu(self.rank)` 按目标卡自动
   算 worker/scheduler 数（`:230-234`）——**这正是 `docs/06` 里 128 worker/16 scheduler 的来源**。
6. 组装 `meta_tensors` 字典（`:237-263`），构造 `PersistentKernel(...)`（`:264-288`），把
   `meta_tensors` 的**指针数组**存下（`:289-292`）。

### 2.3 运行模式 → 编译宏（`persistent_kernel.py:320-334`）

`mode` 不是纯 Python 分支：它被翻成 `-DMODE_*` 编进 JIT kernel：

```
offline          → -DMODE_OFFLINE
online           → -DMODE_ONLINE
online_notoken   → -DMODE_ONLINE_NOTOKEN
onepass          → -DMODE_ONEPASS
online_multi_turn→ -DMODE_MULTI_TURN
online_pinned    → -DMODE_ONLINE_PINNED  + -DMPK_PINNED_RING_CAPACITY=<N>
```

即**同一份 builder、同一张图，按 serving 语义编出不同的常驻 kernel 变体**。容量常量也在这里编进
（`MPK_MAX_NUM_BATCHED_REQUESTS`/`TOKENS`/`MAX_NUM_PAGES`/`MAX_SEQ_LENGTH`/`NUM_KV_GROUPS`/…，
`:336-345`）。这就是论文「每 batch size 一张静态图」局限（`docs/01` §5.8）在源码里的落点：**上限是
编译期常量**。

### 2.4 生命周期（`mpk.py:471-511`）

```
build()                # 选 builder + 建图（= 调 95 个 layer）
generate_task_graph()  # kn_graph.generate_task_graph(num_gpus=world, my_gpu_id=rank)  → docs/08
compile()              # persistent_kernel.compile(eos_token_id=...)                    → docs/06
__call__()             # 首次调用自动 compile；然后 persistent_kernel(**kwargs) 单次 launch
```

`generate_task_graph`（`:482-489`）把 `world_size`/`rank` 交给图生成器——**依赖分析器要知道有几张卡、
我这张卡编哪个 rank 的图**（对上 `docs/17` 的「每 rank 各生成 tGraph」）。`__call__`（`:499-508`）在
`online_notoken` + `with_lm_head=False` 时直接返回 `model_builder.returned_hidden_state`（不算 lm_head
的「只跑主干」模式）。

## 3. 模型 builder 的组装逻辑：以 Qwen3 为规范样本

### 3.1 两条入口：`build_from_model` vs `build_from_config`

- `build_from_config`（`models/qwen3/builder.py:34-53`）：从 `MirageModelConfig` 取 hidden/inter/heads/
  layers/k_cache/v_cache/position_embeddings，算出 `fused_outdim_1 = (q+2*kv)*head_dim`、
  `fused_outdim_2 = 2*inter`，然后转 `build_from_dict`。
- `build_from_model`（`:55-118`）：`Qwen3ForCausalLM.from_pretrained`（或 `load_model` 加载
  `model{rank}-mp{world}.safetensors` 分片）、建 tokenizer、**现算 RoPE**（`rotary_emb(dummy_x,
  positions)`，`:75-76`）、**现开 KV cache**（`torch.empty((num_layers, max_num_pages, page_size,
  num_local_kv_heads, head_dim))`，`:93-114`）、最后也转 `build_from_dict`。

两入口都收敛到 `build_from_dict`（`:560-690`），差异只在「权重/配置从哪来」。

### 3.2 `build_from_dict`：主干组装顺序（`models/qwen3/builder.py:560-690`）

1. **lm_head 补 vocab padding**（`:564-575`）：把 `lm_head.weight` 用 0 行 pad 到
   `padded_vocab_size`（Qwen3 硬编码 **153600**，`builder.py:45,82`，源码注释 `TODO: A better way to decide?`
   ——为了对齐 task 图的整除约束）。
2. **attach 输入/位置嵌入**（`:583-591`）：`input_tokens→x`、`position_embeddings` 的前 4096 行切成
   `cos_position_embedding`/`sin_position_embedding` 两块。
3. **`new_intermediate_tensors()`**（`:593`，定义在 `:120-287`）——见 §3.3。
4. **`output_tokens→argmax_out`**（`:595`）。
5. **Embedding**（`:608-621`）：`embed_tokens` 权重 + `embed_layer(input, weight, output, grid=(1,1,1),
   block=(128,1,1), input_source=1)`；`x=y`。
6. **`build_layers(state_dict)`**（`:623`，定义 `:288-558`）——见 §3.4。
7. **最终 norm + lm_head**（`:625-644`）：`rmsnorm_layer` → `linear_layer`（输出 `argmax_in`，
   维度 = `padded_vocab_size`）。
8. **采样/argmax 两段式**（`:646-681`）：`do_sample` 走 `sampling_partial_layer`（grid=`num_workers`）
   + `sampling_reduce_layer`；否则走 `argmax_partial_layer` + `argmax_reduce_layer`（**部分归约在 worker
   上分块、再单 block 归约**——一次激活全部 worker，正好是常驻 kernel 擅长的形状）。

### 3.3 `new_intermediate_tensors`：中间张量「进不进常驻 kernel 显存 / 跨不跨卡」在此决定

这是本文**最关键的机制点**。它按 `mode` 走两条路（`models/qwen3/builder.py:120-287`）：

**(a) `online_notoken` → 预分配固定 tensor + `attach_input`**（`:126-183`）：直接把 `torch.zeros(...)`
喂给 `mpk.attach_input(name=...)`，并保留引用（如 `self.y_tensor`）。这是「张量由外部持有、kernel 只拿
裸指针」的模式（`docs/06` §运行层：`attach_input` 会把**裸 GPU 指针烘进生成代码**，见
`persistent_kernel.py:746-777` 的注释 *"if the tensor is freed, the pointer becomes dangling"* 与
`self._torch_tensor_refs`）。

**(b) 其它模式 → `new_tensor(dims, dtype, name, io_category=...)`**（`:185-286`）：只声明**形状/类型/
类别**，由编译器管理显存。关键在 `io_category`：

```
"cuda_tensor"    → attach_cuda_tensor（本卡显存）              persistent_kernel.py:795-796
"nvshmem_tensor" → attach_nvshmem_tensor（对称显存, 可跨卡访问）  :797-798
```

Qwen3 里 `io_category` 的取值规则直白（`models/qwen3/builder.py:210-251`）：**只含本地计算的张量用
`cuda_tensor`；会在 AllReduce 里被别的 rank 读写的张量（`attn_proj_out`/`all_reduce_buf`/
`attn_allreduce_out`/`mlp_out`/`mlp_final`）在 `world_size>1` 时用 `nvshmem_tensor`**。也就是说
**「哪些中间张量要进对称显存」是 builder 手写的**，编译器据此决定通信 task 怎么寻址（对上 `docs/17`
§3 的 NVSHMEM event 与对称显存 rank 切片）。

其余中间量（`embed_out`/`rmsnorm_out`/`attn_in`/`attn_out`/`mlp_mid`/`silu_mul_out`…）都是
`cuda_tensor`：**它们活着的时候只在同一张常驻 kernel 内被读写**——这正是 megakernel「中间量不落/少落
HBM 往返」的物理前提（`docs/04` §17 的损失三；注意这只减少*跨算子*往返，同一张量仍可能过 HBM/L2，
`docs/04` §29 L2 常驻一节的边界）。

### 3.4 `build_layers`：逐层手写的算子序列（`models/qwen3/builder.py:288-558`）

Qwen3 每层的**手写序列**（对照论文「operator decomposition」，这里是人工版）：

```
rmsnorm(input_layernorm)                       :354-360
linear(qkv)                                    :361-367
paged_attention(q_norm/k_norm/rope/kv-cache)   :399-410
linear_with_residual(o_proj, residual=x)       :425-432   （SM100 走 splitk_linear_layer :415-423）
[world>1] allreduce(attn_proj_out→attn_allreduce_out) :436-444
rmsnorm(post_attention_layernorm)              :505-511
linear(gate_up)                                :512-518
silu_mul                                       :520-525
linear_with_residual(down_proj, residual=x)    :540-547 （SM100 走 splitk :530-538）
[world>1] allreduce(mlp_out→mlp_final)         :550-558
```

几个值得记的手工决策：
- **`target_cc` 分支**（`:291-293`）：`use_splitk = (target_cc == 100)`——**split-K linear 只在 B200
  (SM100) 启用**，H100 走普通 `linear_with_residual_layer`。这是 builder 里的**硬件代际分支**。
- **grid/block 全手写**：例如 o_proj 的 `grid_dim=(hidden_size // 64, 1, 1)`（`:430`）、qkv 的
  `grid_for_rmsnorm_linear_layer(w_qkv.dim(0))`（`:365`）。**并行度切分是模型 builder 的职责**，不是
  编译器自动推的（这是个「每个模型要调 grid」的工程负担，也解释了 `models/utils.py` 里那些 ad-hoc
  helper 的必要性）。
- **残差更新靠张量身份**：`self.x = self.attn_proj_out`（`:434`）/ `self.x = self.mlp_out`（`:549`），
  AllReduce 后再 `self.x = ...`（`:444,558`）。层与层之间靠**同一批中间张量名复用**串成 DAG，依赖
  分析器再从这个 DAG + 中间张量身份重建事件（`docs/08` §2 的「最近写者 SSA」）。
- **AllReduce 只在 `world_size>1` 时插**（`:436`/`:550`）——单卡图里根本没有通信 task。

## 4. 权重预处理与 grid helper（`models/utils.py`）

### 4.1 权重交织：`shuffle_tensors` / `inplace_shuffle_tensors`

把 `[Q,K,V]` 按 head 交织成 `[Q_h0,K_h0,V_h0,Q_h1,…]` 的布局（`models/utils.py:50-128`），或原地写进一块
预分配的目标张量（`:130-156`）。用途（`models/qwen3/builder.py:300-347`）：当权重里**同时存在**
`qkv_proj.weight` 和分开的 `q/k/v_proj.weight` 时，把分开的三块交织进 `qkv_proj` 槽位（head 级交错），
以便后续 linear 一次算完 QKV；gate/up 同理（`:451-497`）。`inplace_shuffle_tensors` 用一块临时 GPU
张量再 `copy_`，注释说明是为避免回收。

### 4.2 grid helper：把「输出维度」映射成 task 数

- `grid_for_rmsnorm_linear_layer(size)`（`models/utils.py:3-15`）：优先取 96（整除）、否则 64；`size/96>400`
  时改 `size//256`。注释自陈是 ad-hoc，且提到「输出太大时 MPK ptx 与 cutlass 版会给不确定结果」。
- `grid_for_splitk_linear_layer(output_size, reduction_size)`（`:17-34`）：SM100 split-K 的 grid
  `(output//128, splits, 1)`，**splits 必须整除 reduction 且 `reduction//splits` 是 64 的倍数**（TMA 对齐
  + K tile）；注释记录了旧固定 `y=128*128//output_size` 在 Qwen3-32B o_proj（8192/3）上产生**错位 TMA
  描述符**的真实 bug。
- `max_factor_leq_n(m, n)`（`:38-48`）：取不超过 n 的最大因子（注释里 `embed_layer` 的 grid 曾用它，但
  现在 Qwen3 走固定 `(1,1,1)`，见 `models/qwen3/builder.py:617`）。

这些 helper 的存在本身就说明：**前端仍是「人肉把模型维度换算成 task 网格」**，与论文描述的自动
decomposition 有明确落差（`docs/14` §6 已记，`docs/22` §7 落差清单）。

## 5. 多卡 / 张量并行：切分在 builder，集合通信选择在 `multigpu.py`

### 5.1 切分（builder 内手写）

Qwen3 的 TP 切分直接写死：`num_local_q_heads = num_q_heads // world_size`、
`num_local_kv_heads = num_kv_heads // world_size`（`models/qwen3/builder.py:85-86`），MLP 中间维
`fused_outdim_2 // world_size`、`intermediate_size // world_size`（`:149,152,229,235`）；KV cache 额外按
`num_local_kv_heads` 开（`:96-99`）。**每张卡编的是自己那份图**，`generate_task_graph(num_gpus, my_gpu_id)`
再告诉图生成器 rank（`mpk.py:486`）。

### 5.2 集合通信选择（`multigpu.py` + `cuda_utils.py`）

`multigpu.py` 是一套**策略模式**的集合通信选择器：

- `GPUArchitecture`（`:17-22`）：`AMPERE=80 / HOPPER=90 / BLACKWELL=100`。
- `CollectiveCapabilities`（`:24-63`）：构造时查询 4 项硬件能力——`peer_access`（两两 peer 访问）、
  `vmm`（虚拟内存管理）、`multicast`、`posix_handle`，并取 `target_cc`。查询走 `cuda_utils.py` 里的
  `cuda.bindings.driver`（如 `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, ...)`，
  `cuda_utils.py:34-40`）。
- 策略基类 `CollectiveStrategy`（`:102-124`）约定 `register_tasks(mpk, tensors, grid_dim, block_dim, params)`
  ——**一个集合通信 = 若干 task 的注册**，这正好是「把通信编进 mega-kernel」的接缝（对上 `docs/17`）。
- 两种 AllReduce 实现：
  - `AllReduceStrategy_AllgatherReduce`（`:156-184`）：注册 `nvshmem_allgather_strided_put` +
    `reduction` 两个 task（论文 §6.5 策略 A）。
  - `AllReduceStrategy_NvshmemTile`（`:187-208`）：注册单个 `nvshmem_tile_allreduce` task，并调
    `allocate_nvshmem_teams(mpk, grid 的 block 总数)`。
- `auto_select_allreduce_implementation`（`:224-247`）：`target_cc >= 90` **且** VMM+multicast+peer
  全支持 → 选 `NvshmemTile`；否则回退 `AllgatherReduce`。这条 `cc>=90 + 三项能力` 判据，正好解释了
  `docs/17` §6「两种 AllReduce」为何并存。
- `allocate_nvshmem_teams`（`:74-96`）：设 `NVSHMEM_MAX_TEAMS = num + 6 + 32`（6 是 NVSHMEM 内建团队数、
  32 是余量），并强制 `NVSHMEM_MAX_CTAS=1`（避免 NVSHMEM 建重复 team）；注释自陈该 env **未文档化、
  需读 NVSHMEM 源码**。
- `auto_select_allgather/broadcast/reduce_scatter_*`（`:250-298`）**全是 `raise NotImplementedError`**
  ——**多卡集合通信选择目前只落地了 AllReduce 一条线**（与 `docs/17` §6 的观察一致）。

## 6. 其它模型家族：MoE / MLA / 投机解码（与 `docs/24` 的家族 ID 块对应）

`docs/24` §1 列出了 `enum TaskType` 里按模型家族切的 ID 块（`INKLING` 350-399、`GLM` 400-449、
`GPT_OSS` 450-499）。**这些家族专用 `TaskType` 的前端生产者正是这些 builder**：

| 家族 | builder 特色（文件头注释 + 关键算子） |
|---|---|
| **GLM-4.6** | 92 层、前 3 层 dense、其余 160 routed + 1 shared（**shared 折叠成 expert 160、slot 8 权重 1.0**，`models/glm4_moe/builder.py:9-18`）；router 用 `glm_moe_router_layer`（`:305`），管线 `moe_w13_linear_layer`→`moe_silu_mul_layer`→`moe_w2_linear_layer`→`moe_mul_sum_add_layer`（`:339-355`）。 |
| **Inkling** | 66 层、短卷积（depthwise k=4）+ 相对位置 bias + 滑窗/全局混合注意力（`:9-20`）；2 shared 折成 expert 256/257（`inkling/builder.py:14-17`）；router `inkling_moe_router_layer`（`:526`）。 |
| **GPT-OSS** | clamped-alpha SwiGLU MoE、**每个投影带 bias**（`gpt_oss/builder.py:47-50`）；router `moe_topk_softmax_routing_layer`（`:345`）；用 **`kv_planner`**（`KVCachePlan/KVSpec/plan_kv_groups`，`:4`）建 KV。 |
| **DeepSeek-V3** | 61 层 MLA + MoE（256 experts top-8 + shared）；**权重吸收**：加载时把 `kv_b_proj` 吸收进 `q_b_proj`，运行时只需压缩 KV `[c_latent 512, k_pe 64]`（`deepseek_v3/builder.py:3-10`）；**FP8 MoE group GEMM** 带 UE8M0 scale（`:44-62,936-949`）、`moe_topk_sigmoid_routing_layer`（`:905`）。 |
| **DFlash / Eagle3** | 投机解码**草稿** builder（不注册）：Eagle3 是 K 步自回归草稿；DFlash 是 Kimi-K2.6 的**单次非因果前向**草稿、自带分页 KV、与目标共享 embedding+lm_head（`dflash/builder.py:3-18`）。 |

工程含义：**每适配一个模型家族，就是手写一份 builder + 登记若干家族专用 `TaskType`**。这解释了
`docs/24` 里为什么会有 `INKLING`/`GLM`/`GPT_OSS` 这样的**模型家族 ID 块**——它们不是通用算子，而是
「这个模型的 router/注意力在 device 侧的专用实现」。也是 `docs/23` §5「少而重」的另一面：**可融合的
task 类型有 90+ 种，但生产里一个模型只挑其中几十条编进同一张图**。

## 7. KV 规划、权重加载与在线锁页运行时（同层附属件）

这三者都在 `python/mirage/mpk/`，同属「请求/元数据管理」的支撑：

- **`kv_planner.py`**（624 行）：**支持异构 KV 流**（不同大小/布局的 KV stream 共享一个物理页池）。
  文档串（`:1-20`）定义 `KVSpec`（一种 cache 元素/每 entry 字节/层集合/压缩比/滑窗）→
  `plan_kv_groups(...)`（`:433`）→ `KVCachePlan.allocate_pool(...)`（`:274`）→
  `group_specs()`/`build_meta_tensors()`（`:150,194`）喂给 `PersistentKernel(kv_groups=...)`。**这是
  `PersistentKernel.kv_groups`（`persistent_kernel.py:440,476-488`）与 `-DMPK_NUM_KV_GROUPS` 宏
  （`:341`）的来源**，也解释了 GPT-OSS builder 为何调 `plan_kv_groups`。滑窗、混合注意力、跨 stream
  页复用这些「模型级 KV 语义」在此被编译成运行时的页表（对上 `docs/17` 的多卡页管理）。
- **`base_dynamic_shard_loader.py`**（449 行）：HF safetensor **动态分片加载器**。`ShardType`
  （`:54-65`）有 `COL_PARALLEL=0`（沿 dim0 切）/`ROW_PARALLEL=1`（沿 dim1 切）/`EXPERT_PARALLEL=2`
  （按 EP 组保留本组 experts）/`NONE=100`（各卡复制）。调用方给一个 `mapping`（HF 权重名 → shard 规格，
  文档串 `:27-35` 给了 q_proj/o_proj/gate_proj 的例子），加载器下载分片、按规则切片、物化到 CUDA、
  挂到 meta 初始化的模型上。**这把 §5.1 手写的 TP/EP 切分从「代码里 //world_size」提升为「声明式
  分片规则」**，是多卡权重加载的通用路径。
- **`online_pinned_runtime.py`**（384 行）：`MODE_ONLINE_PINNED` 的 **CPU 侧驱动**。两个**锁页（pinned）
  的 2 的幂环形缓冲**让 CPU/GPU 无 DMA 直接访问：请求环（CPU→GPU，写 `{rid, prompt_len,
  initial_step}` 后置 ready）与完成环（GPU→CPU，写 `{rid, buffer_row, final_step}` 后置 ready），每槽
  独立 pinned inbox（`:1-25`）。`submit`/`wait_for_request`/`drain_completions`/`read_tokens_at_row`
  等（`:95-267`）构成在线 serving 的 host 侧泵。这是「常驻 kernel 不退出、host 持续投喂请求」的
  具体落地（对上 `docs/06` §4 的 `prepare_next_batch` 连续服务）。

## 8. 与论文对照 + 判据映射

| 论文 | 这一层 | 落差 |
|---|---|---|
| §4.1 operator decomposition（快照行 462–468，每算子 32–47 task） | **手工 builder 的 `build_layers`**：人写层序 + grid | 论文是编译器/超优化器自动分解；`mpk` 生产前端是**人手写**（`docs/14` §6 已记，自动路径在 `docs/15`） |
| 每 batch size 一张静态图（局限，`docs/01` §5.8） | `mode` + 容量常量编进 `-DMODE_*`/`-DMPK_MAX_*`（`persistent_kernel.py:320-345`） | 一致：batch/seq/page 上限是编译期常量 |
| 自动选择通信实现 | `multigpu.py` 的 `auto_select_allreduce_implementation` | 只落地 AllReduce，其余三种集合通信 `NotImplementedError` |
| 对称显存 rank 切片（§6.5） | `io_category="nvshmem_tensor"` 由 builder 手选 | 一致（`docs/17` §3 已证） |

**判据映射（C1–C5，`docs/00` §3）**：本层**不直接拿任何一条判据**——它只 **（a）划定 C3 的范围**（把哪些
算子登记进同一张图，为 C1「单次 launch」提供前提），**（b）为 C4/C5 准备输入**（把 AllReduce 注册成
in-graph task、把中间量标成 `cuda`/`nvshmem`）。真正的 C1/C2/C4/C5 在运行层拿到（`docs/22` §8）。
换言之：**builder 决定「能融在一起的东西」，运行层决定「融得成不成 megakernel」。**

## 9. 局限与坐标

- **纯源码走读，未构建/运行**：所有结论基于 `f9eb70c` 快照行号；不采信性能数。端到端行为见 `docs/09`
  （含 P25-1 的 launch 顺序根因）与 `docs/26`（若续做多算子延迟）。
- **未逐行展开每个家族的 `_build_*`**：GLM/Inkling/DeepSeek-V3 的 MoE/MLA 管线只给算子名与文件头
  架构说明，未逐行走读其权重折叠/吸收细节（那些在各自 `builder.py` 的 `_build_moe_mlp`/MLA 段）。
- **未覆盖 `annotated_graph.cc` 如何消费这些张量**：本文停在「builder 登记了什么」；依赖分析器如何从
  中间张量身份重建事件在 `docs/08`。
- **`MirageModelConfig` 的调用方（如 `demo/`/`tests/`）未走读**：本文只看库侧契约。
- **行号已核对**：`python/mirage/mpk/mpk.py`(516)、`model_registry.py`(29)、`multigpu.py`(298)、
  `models/graph_builder.py`(51)、`models/__init__.py`(17)、`models/utils.py`(156)、
  `models/qwen3/builder.py`(696)、`persistent_kernel.py` 相关行。

## 10. 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
  主文件：`python/mirage/mpk/mpk.py`（`MPKMetadata:15-83`、`MPK:132-516`）、
  `python/mirage/mpk/model_registry.py:5-29`、`python/mirage/mpk/models/__init__.py:1-17`、
  `python/mirage/mpk/models/graph_builder.py:7-51`、
  `python/mirage/mpk/models/qwen3/builder.py`（`build_from_dict:560-690`、`build_layers:288-558`、
  `new_intermediate_tensors:120-287`）、`python/mirage/mpk/models/utils.py:3-156`、
  `python/mirage/mpk/multigpu.py:17-298`、`python/mirage/mpk/kv_planner.py:31-563`、
  `python/mirage/mpk/base_dynamic_shard_loader.py:54-221`、
  `python/mirage/mpk/online_pinned_runtime.py:36-384`、
  `python/mirage/mpk/persistent_kernel.py:320-345,414-513,746-840`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462–468、
  §4.2 行 1126–1146、§5.8 局限）。
- 同专题相关篇：`docs/14`（前端层 API，本文补其 §7 尾巴）、`docs/24`（task 目录，本文给出家族专用
  `TaskType` 的前端生产者）、`docs/08`（依赖分析器消费这些登记）、`docs/06`（运行层，`meta_tensors`/
  `nvshmem_teams` 的消费方）、`docs/17`（多卡运行层，本文 §5/§7 的编译层对照）、`docs/23`（总纲）、
  `docs/02`（DeepGEMM MoE 生产对照）。
