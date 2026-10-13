# MPK Eagle3 权重/中间张量装配层走读：`_prepare_weights` / `_attach_weights` / `_allocate_intermediates`

> 本文补齐 `docs/46`（P132-1）**§9「参考坐标」里列出、正文却一直没打开的** `Eagle3Builder` 前三个方法
> —— `_prepare_weights:119`、`_attach_weights:190`、`_allocate_intermediates:220`，也即「下一步候选」
> 第二条 `Eagle3Builder` 尚未走的 `_prepare_weights`/`_attach_weights`。`docs/46` 讲的是
> `build_draft_loop:271-449`（**构图**：一张图里有哪些 task、K=1 与 K>1 怎么分叉），本文讲的是它**之前**
> 那三步——**权重怎么从 HF state_dict 摆成常驻 kernel 要的布局、怎么挂进 `mpk` 图、中间张量怎么开**。
>
> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/46` 同快照）。
> **纯源码走读（资料分析）**，不构建、不运行；Eagle3 device 头只被 blackwell `task_header.cuh` include，
> SM100 专属，本机（目标卡 H100/sm90）不可执行。

## 0. 三条 headline

- **(A) `_prepare_weights` 是「布局契约层」，不是数值计算层**：它把 HF/Eagle3 checkpoint 里的
  `q/k/v` 三支权重、`gate/up` 两支权重，用同一个 `shuffle_tensors`（`models/utils.py:50-128`）
  **按 KV 头 / 按 rmsnorm-linear 任务块交错**成 fused 布局，再把 `d2t`、`fc`、`lm_head`、paged draft
  KV cache、dummy qk-norm 准备好。它们**只改内存排布**（`shuffle_tensors` 是纯 `torch.empty` + 切片写入，
  无数值运算），正确性全靠「下游 task 期望的布局」与「这里摆出的布局」对齐。
- **(B) 三处「硬编码 padding / dummy」是 TMA 与 paged-attention 的接口约束**：
  `lm_head` 从 32000 行**补零到 32256**（96 的整数倍，让 lm_head linear 的 336 行/任务满足 16 B TMA
  输出对齐）；`paged_attention_layer` 强制要求 `q_norm`/`k_norm` 非空（`:1345-1352`），故 builder 塞一个
  `(head_dim,)` 的**全零 dummy** 并传 `enable_qk_norm=False`；paged draft KV cache 是**自持的独立 buffer**
  （与 target 的 paged cache 分开）。这三处不是「顺手写死」，而是「上游 task 接口的最小满足」。
- **(C) `_attach`/`_new` 用一个 `_attach_cache` 同时当「去重 + 张量注册表」**：两者都按 `name` 缓存，
  且 `_attach` 把 torch tensor 塞进 `_kept_tensors`（防 GC——MPK 常驻 kernel 存的是**裸 GPU 指针**，
  `attach_input` 注释 `:772-774` 明说）。因此 `_attach_cache` 既是幂等层，也是 demo 侧拿 DTensor 的
  唯一入口（demo `:882` 用 `eagle3._attach_cache["eagle3_all_draft_ids"]` 取 scatter 的 dst）。

---

## 1. 定位与坐标

`Eagle3Builder` 的公开入口只有 `build_draft_loop:271`，它在构图前**顺序**调三件事（`:296-298`）：

```
self._prepare_weights()        # :119  从 self.sd（HF state_dict）摆出 fused 权重
self._attach_weights()         # :190  把摆好的权重 attach 进 mpk 图，得到 w_* DTensor
self._allocate_intermediates() # :220  开本 iter 复用的全部中间张量（mbt, ...）
```

三者与 `build_draft_loop` 的分工：

| 层 | 方法 | 产出 | 谁消费 |
|---|---|---|---|
| 布局 | `_prepare_weights:119` | `_qkv_w/_gateup_w/_d2t/_fc_w/_lm_head_w/_k_cache_buf/_v_cache_buf/_dummy_norm_buf` | `_attach_weights` |
| 注册 | `_attach_weights:190` | `w_*`/`d2t`/`k_cache`/`v_cache`/`dummy_norm`（`DTensor`） | `build_draft_loop` 的 `*_layer()` |
| 中间量 | `_allocate_intermediates:220` | `embed_out`…`all_draft_ids`（`DTensor`） | `build_draft_loop` 的 `*_layer()` |

demo 侧调用顺序（`demo/qwen3/demo_30B_A3B_eagle3.py`）：`load_eagle3_draft:802` →
`Eagle3Builder(...):803-813` → `build_draft_loop(...):854-860`。注意 `__init__:70` 断言
`mpk.world_size == 1`（Eagle3 builder v1 **只支持单卡**），`:71-72` 的 `use_aux_norm=True` 直接
`NotImplementedError`（默认 ckpt 无 aux norms）。

---

## 2. `_prepare_weights`（`builder.py:119-188`）逐行

### 2.1 Fused QKV：按 KV 头交错（`:123-131`）

```python
q = sd["midlayer.self_attn.q_proj.weight"]   # (num_q*head_dim, 2H)
k = sd["midlayer.self_attn.k_proj.weight"]   # (num_kv*head_dim, 2H)
v = sd["midlayer.self_attn.v_proj.weight"]   # (num_kv*head_dim, 2H)
self._qkv_w = shuffle_tensors([q, k, v], self.num_kv_heads, 0)
```

- **输入维是 `2H` 而非 `H`**：`:127-129` 断言三支权重的 `shape[1] == 2 * hidden_size`。原因是 Eagle3
  的 attention 输入是 `concat([embed_normed, hidden_normed])`（`docs/46` §2.2；builder `:344-348` 的
  `concat_layer`），Eagle3 论文里 Q/K/V 同时对「当前 token 的 embedding」与「上一层 draft hidden」做投影，
  故输入拼成 `2H`。这是 Eagle3 draft 层与普通 Llama 层最显眼的结构差异。
- **`shuffle_tensors([q,k,v], num_kv_heads, 0)` 的布局**（`models/utils.py:50-128`）：把每个张量沿 dim0
  均分 `split=num_kv_heads` 份，再按张量顺序交错。对 QKV 就是
  `[Q_group0, K_head0, V_head0, Q_group1, K_head1, V_head1, …]`，其中 `Q_groupi` 是
  `num_q/num_kv × head_dim` 行（GQA 的一组 q 头）、`K_headi/V_headi` 各 `head_dim` 行。
  docstring `:54-55` 的示例正是这个形态。函数体 `:118-128` 是纯 `torch.empty(out_shape)` + 逐 head
  切片赋值，**无任何算术** —— 唯一正确性来源是「下游 `paged_attention_layer` 读 fused QKV 时按同一
  head 分组解释」（前端 `:1407` 的 `new_input(input, (-1, 1, -1), …)` 把 dim1 当 fused 维 token 切分）。
- 输出行数 = `sum(t.shape[0])`（`:109` `out_shape[dim] = per_head_size * split`），
  即 `(num_q+2*num_kv)*head_dim = self.fused_outdim_qkv`（`__init__:90-91`），与 `attn_in` 中间量
  (`:242-243`) 对齐。

### 2.2 Fused gate+up（`:133-139`）

```python
gate = sd["midlayer.mlp.gate_proj.weight"]   # (interm, H)
up   = sd["midlayer.mlp.up_proj.weight"]     # (interm, H)
rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(gate.shape[0] + up.shape[0])
self._gateup_w = shuffle_tensors([gate, up], rmsnorm_num_tasks // 2, 0)
```

- `gate/up` 各 `(interm, H)`，fused 后 `(2*interm, H) = self.fused_outdim_gateup`（`__init__:92`）。
- 交错组数取 `grid_for_rmsnorm_linear_layer(2*interm) // 2`：`grid_for_rmsnorm_linear_layer`
  （`models/utils.py:3-15`）返回的是 rmsnorm-linear 融合 kernel 的**任务块数**（`96` 或 `64`；`size/96>400`
  时退化成 `size//256`，`assert size%256==0`，注释 `:4-5` 自陈「96/64 covers all Qwen3 model? 请更新」）。
  用它的**一半**当交错组数，使每组正好「一个 gate 块 + 一个 up 块」，与下游 `silu_mul_layer` 的
  grid=`gateup_num_tasks//2`（builder `:397-406`）一一对应——即 silu_mul 每个任务拿到一组相邻的
  `[gate_chunk, up_chunk]`。**布局契约由「交错组数」与「下游 grid 的一半」两个数同时决定**，改任一都要改另一。
- 注意这里**调用的是 `shuffle_tensors(tensors, split, dim)`，参数含义与同名 `mpk.shuffle_tensors`
  不同**：`models/utils.py` 版是「把 N 支张量交集成一支」（纯 torch），`PersistentKernel.shuffle_tensors`
  （`persistent_kernel.py:811`）是「把 N 个 DTensor 在**图层面**融合成一支」。builder 这里用的是 torch 版
  （import 自 `models/utils.py:7-10`）。

### 2.3 `d2t` 表（`:141-147`）

```python
d2t = sd["d2t"]
if d2t.dtype != torch.int64:
    d2t = d2t.to(torch.int64)
self._d2t = d2t.contiguous()
```

- 约定（源码注释 `:142` 与 device 头 `eagle3_ops.cuh:85-89`）：**`target_id = hot_id + d2t[hot_id]`**
  （sglang convention，`d2t` 是**有符号 int64**）。draft 头只在较小的 hot 词表（`draft_vocab_size`）上
  做 argmax，靠 `d2t` 映射回 target 全词表 id。
- 强制转 `int64` 并 `contiguous`，因为 device 侧 `eagle3_d2t_remap_kernel`（`eagle3_ops.cuh:102-122`）
  按 `long long*` 读（`:107-111`），且要求连续布局。

### 2.4 `fc` / `eh_proj`（`:149-153`）

```python
fc = sd["fc.weight"]                     # (H, 3H)
assert fc.shape == (self.hidden_size, 3 * self.hidden_size)
self._fc_w = fc.contiguous()
```

- `(3H → H)` 的 `eh_proj`，**只在 step 0 用**（builder `:318-330`：step 0 把三个 aux hidden `concat` 成
  `3H` 再过一个 linear 得到 `hidden_in`；step>0 直接用上一步的 `draft_hidden`）。所以它虽然挂在 draft 头里，
  却是「每 iter 只走一次」的融合前馈。

### 2.5 `lm_head` 补零到 32256（`:155-174`）

```python
lm = sd["lm_head.weight"]                # (draft_vocab_size, H)
self._padded_draft_vocab = 32256
assert self._padded_draft_vocab % 96 == 0, "…must be 96-aligned"
pad_rows = self._padded_draft_vocab - self.draft_vocab_size
self._lm_head_w = torch.cat([lm, zeros((pad_rows, H))], 0).contiguous()
```

- 注释 `:156-157`：**补零是为了让「每任务输出 16 B 对齐」**，从而 `cuTensorMapEncodeTiled` 接受 OUTPUT
  的 TMA 描述符。`draft_vocab_size`（`__init__:89`）对 Eagle3 是 **32000**；`32000 % 96 = 32000 - 333*96
  = 32`，不整除，故补到最近的 `96` 倍数 **32256 = 96 × 336**（注释 `:157`）。
- 为什么是「96」：lm_head 的 linear grid 由 `grid_for_rmsnorm_linear_layer(self.w_lm_head.dim(0))`
  给出（builder `:419-421`）。`grid_for_rmsnorm_linear_layer(32256)`：`32256/96 = 336 < 400` 且
  `32256 % 96 == 0` → 返回 **96**。于是 96 个任务、每任务 336 行；336 行 bf16 = 672 B = 42 × 16 B，
  恰是 TMA box 的 16 B 倍数。所以「96 对齐」同时服务两件事：**grid 整除** 与 **输出 tile 16 B 对齐**。
- 补进来的行是**全零**（`:167-168`）——它们会在 argmax 里被当成 logits=0 参与竞争（见 §5）。

### 2.6 paged draft KV cache + dummy qk-norm（`:176-188`）

```python
self._k_cache_buf = torch.zeros((max_num_pages, page_size, num_kv_heads, head_dim), bf16)
self._v_cache_buf = torch.zeros_like(self._k_cache_buf)
self._dummy_norm_buf = torch.zeros((head_dim,), bf16)
```

- KV cache 布局 `(max_num_pages, page_size, num_kv_heads, head_dim)`（`:176-177` 注释），与前端
  `paged_attention_layer:1327-1328` 的断言 `k_cache.num_dims == 4` 一致。`max_num_pages`/`page_size`
  来自 `mpk`（`__init__:78-79`）。这是 **draft 自己的** KV cache，**与 target 的 paged cache 分开**
  （`:214` 注释）。
- dummy qk-norm：`paged_attention_layer:1345-1347` **硬断言** `q_norm is not None and k_norm is not None`
  并提示「pass a dummy + `enable_qk_norm=False` when norm is disabled」。builder 就照做：塞 `(head_dim,)`
  全零、attach 成同一张量给 `q_norm`/`k_norm`（`_attach_weights:217-218` 只 attach 一次
  `self.dummy_norm`），调用时传 `enable_qk_norm=False`（`build_draft_loop:367`/`:380`）。Eagle3 draft 的
  `midlayer` 没有 q/k norm（`use_aux_norm=False` 是默认），但 paged attention task 的输入契约要求槽位存在。

---

## 3. `_attach_weights`（`builder.py:190-218`）逐行

### 3.1 `_attach` / `_new` 的缓存语义（`:102-117`）

```python
def _attach(self, tensor, name):
    if name in self._attach_cache: return self._attach_cache[name]
    self._kept_tensors.append(tensor)                 # 防 GC
    dt = self.mpk.attach_input(torch_tensor=tensor, name=name)
    self._attach_cache[name] = dt
    return dt

def _new(self, dims, dtype, name, io_category="cuda_tensor"):
    if name in self._attach_cache: return self._attach_cache[name]
    dt = self.mpk.new_tensor(dims=dims, dtype=dtype, name=name, io_category=io_category)
    self._attach_cache[name] = dt
    return dt
```

- `_attach` 走 `mpk.attach_input`（`persistent_kernel.py:746`）：把**已有 torch tensor** 注册进图，
  **保持其 GPU 裸指针**（`:772-774` 注释：「常驻 kernel 存的是 raw data pointer，tensor 被释放指针就悬空」，
  故 `_kept_tensors.append` + `mpk` 内部 `_torch_tensor_refs` 双重保命）。
- `_new` 走 `mpk.new_tensor`（`:782`）：**新建**一个 MPK 管理的 CUDA 张量（`io_category="cuda_tensor"`）。
- 二者共用 `_attach_cache`：**同名的第二次调用返回同一 DTensor**（幂等）。所以 `build_draft_loop` 里
  同一权重被 `w_qkv` 等属性在每 step 复用（`for step in range(K)`）**不会重复注册**。
- **`_attach_cache` 也是外部取句柄的入口**：demo `:882` 用
  `eagle3._attach_cache["eagle3_all_draft_ids"]` 拿 scatter 的 `dst`；`all_draft_ids` 正是
  `_allocate_intermediates` 里用 `_new(..., "eagle3_all_draft_ids")` 注册的（`builder.py:268-269`）。

### 3.2 attach 清单（`:192-218`）

按顺序：`input_layernorm`（`eagle3_input_layernorm`）、`hidden_norm`、`post_attention_layernorm`、
`_qkv_w`（`eagle3_qkv_proj`）、`o_proj`、`_gateup_w`、`down_proj`、`_fc_w`、`norm`（final）、
`_lm_head_w`、`_d2t`、`k_cache`/`v_cache`、`dummy_norm`。

- **norm 权重是从 `sd` 现取 + `.contiguous()`**（`:192-211`），而 `_qkv_w`/`_gateup_w`/`_fc_w`/`_lm_head_w`/
  `_d2t`/`k_cache`/`v_cache`/`dummy_norm` 是 `_prepare_weights` 刚造出来的（用 `self._xxx`）。
- 命名前缀统一 `eagle3_`，与 target 模型的张量名隔离——常驻 kernel 里 target 与 draft 的 norm/投影
  同名不同物，靠 name 区分（`attach_input:775-776` 会把 `.` 换成 `_` 生成安全标识符）。
- 注意 `:213` `self.d2t = self._attach(self._d2t, "eagle3_d2t")`：`d2t` 既是权重也是 `build_draft_loop`
  `:436-441` `eagle3_d2t_remap_layer` 的 `d2t_table`。

---

## 4. `_allocate_intermediates`（`builder.py:220-269`）逐行

### 4.1 全部 `(mbt, …)`、跨 step 复用

- `mbt = self.mbt = mpk.max_num_batched_tokens`（`__init__:77`）。所有中间张量第一维都是 `mbt`：
  `embed_out/embed_normed/hidden_normed/hidden_in/draft_hidden/attn_proj_out/post_ln_out/norm_out`
  都是 `(mbt, H)`；`aux_concat_out` `(mbt, 3H)`；`qkv_in_2H` `(mbt, 2H)`；`attn_in`
  `(mbt, fused_outdim_qkv)`；`attn_out` `(mbt, num_q*head_dim)`；`mlp_mid` `(mbt, fused_outdim_gateup)`；
  `silu_mul_out` `(mbt, I)`。
- 这些张量**在 `for step in range(K)` 里被反复覆写**（`docs/46` §2.2）——`build_draft_loop` 每 step 重新
  `*_layer()`，但 `output=` 都指向同一批 buffer。正确性靠 MPK 的 event 机制保证「上一步写完、下一步才读」
  （`docs/46` §8 明记「buffer 生命周期/alias 只读到静态注册层」；本文同样只记静态事实）。
- `draft_hidden` 与 `hidden_in` 的接力：`hidden_in`（`:234-235`，step 0 的 `eh_proj` 输出）只作 step 0 的
  `step_hidden`；`draft_hidden`（`:238-239`，也是 draft block 的输出）作 step>0 的 `step_hidden`
  （`build_draft_loop:330-332`）。注释 `:236-237` 明确「named so step>0 can read it as input」。

### 4.2 logits / argmax 三段

```python
self.logits_hot      = _new((mbt, self._padded_draft_vocab), bf16, "eagle3_logits_hot")
self.argmax_part_value = _new((mbt, mpk.num_workers), bf16, "eagle3_argmax_part_value")
self.argmax_part_index = _new((mbt, mpk.num_workers), int64, "eagle3_argmax_part_index")
self.hot_token         = _new((mbt, 1), int64, "eagle3_hot_token")
self.target_token      = _new((mbt, 1), int64, "eagle3_target_token")
```

- `logits_hot` 第二维用 **`self._padded_draft_vocab`（32256）**（`:256-258` 注释「Use padded draft vocab
  to satisfy 16B TMA alignment of OUTPUT」），与 §2.5 的补零 lm_head 对齐。
- `argmax_part_value/index` 第二维 = `mpk.num_workers`，因为 `argmax_partial_layer` 的 `grid_dim[0]`
  就是 `num_workers`（builder `:425-429`），而前端算 `argmax_partial_output_size = input.dim(1)//num_tasks`
  （`:2616`）、输出形状断言 `(batch, num_tasks)`（`:2609-2610`）。即 **两段式 argmax 把 32256 个 logits
  按 worker 数一分为多**（每个 worker 一个 partial），再由 `argmax_reduce_layer`（`:2632`）合成
  `hot_token`——这正是 `docs/04` §8/§10「task 数/事件扇入扇出」在 Eagle3 上的来源。
- `target_token` 是 1 列 `int64`，既是 `eagle3_d2t_remap_layer` 的输出（`:436-441`），又是下一步
  `embed_layer` 的输入（`build_draft_loop:309` `draft_in_token = … if step==0 else self.target_token`），
  还是 `mtp_token_scatter_layer` 的 `src`（`:443-447`）。**一支 buffer 串起「d2t 映射 → 下一步 embed →
  scatter 进 all_draft_ids」三条边**，是 cross-step 串行接力的载体。

### 4.3 `all_draft_ids`（`:267-269`）

```python
self.all_draft_ids = _new((mbt, self.num_draft_steps), int64, "eagle3_all_draft_ids")
```

- 形状 `(mbt, K)`，与 demo 的 `eagle3_drafts_prev_buf` `(mbt_e, K)`（`demo:826-829`）一致；
  `mtp_token_scatter_layer(..., num_slots=K, slot_idx=step)`（`:443-447`）把 step 的 `target_token`
  写进第 `step` 槽。返回时 `build_draft_loop` 把它交给 verify 管线（`docs/46` §1）。

---

## 5. 跨层契约闭合：pad → argmax → d2t sentinel

这一节把 §2.5 的补零与下游接起来，是本文最有价值的一条「隐藏契约」：

1. `_prepare_weights` 把 `lm_head` 补零到 32256（`:155-174`）→ `logits_hot` 也开成 `(mbt, 32256)`
   （`:256-258`）。补进来的 256 行 logits 恒为 **0**。
2. `build_draft_loop` 调 `argmax_partial_layer(..., vocab_size` **未传**（`:425-429`，`vocab_size=None`）。
   前端注释 `:2614-2615` 说 `vocab_size` 是「padded 时告诉 kernel 真实词表、≥它的位置永不获胜」的**可选项**
   —— builder **没传**，即 argmax **不知道** 32000 以后是 padding。
3. 于是当真实 logits 全为负时，**补零行（logit=0）会赢得 argmax**，`hot_token ∈ [32000, 32256)`。
4. 但下游 `eagle3_d2t_remap_layer(..., draft_vocab_real=32000)`（`:436-441`，参 `:3191-3193`）会兜底：
   device 侧 `eagle3_d2t_remap_kernel`（`eagle3_ops.cuh:116-120`）判断
   `hot_id >= DRAFT_VOCAB_REAL` 时**写 sentinel 0**（注释：「padded-row argmax → sentinel; verify will
   reject」）。`target_token=0` 之后被 verify 判为「不匹配 target argmax」，自然被拒、只提交 bonus token。

**结论**：这是一条**跨三个函数的契约链**——`_prepare_weights` 负责制造 padding、`argmax` 层允许 padding
获胜、`d2t_remap` 层把 padding 获胜翻译成 sentinel。三者任一改动（例如给 argmax 传 `vocab_size`、或改
`_padded_draft_vocab`）都要同时核对另外两处。`docs/44`（`eagle3_d2t_remap` 的 device 身体）与 `docs/46`
（builder 的构图）各看到链条的一段，本文把「补零」这一段接上。

---

## 6. 判据映射

| 判据 | 本段是否贡献 |
|---|---|
| C1 单次 launch | 间接（这三步只**注册**张量与权重的布局，launch 由 `PersistentKernel` 统一发） |
| C2 常驻 | 间接（权重/中间量以裸指针常驻，`attach_input` 的 GC 保命注释即此） |
| C3 多算子范围 | **是**——本文正是「一张图里注册了哪些权重量、哪些中间张量、什么形状/布局」的清单（§2–§4） |
| C4 设备内协调 | 间接（`(mbt,…)` 跨 step 复用靠 event 依赖，`docs/06`/`docs/46`） |
| C5 SM 级调度 | 否（builder 层不涉及 worker/scheduler 划分；`argmax_part` 用到 `num_workers` 只是借它的**数值**当 grid） |

结论：**本文贡献 C3 的「权重/中间量清单与布局契约」**，与 `docs/46`（贡献 C3 的「范围与分支」）、
`docs/44`（贡献 C1+C3 的「commit/映射 device 身体」）互补。

---

## 7. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实；Eagle3 路径 SM100 专属，本机不可执行。
- **`shuffle_tensors` 的布局只按函数体/ docstring 推断**，未取 device 端「attention 读 fused QKV」的
  实际索引逐元素对照（那需要 `attention_sm100`/`paged_attention_sm100` 的 QKV 解包逻辑，本文未展开）。
- **`draft_vocab_size`/`intermediate_size` 等具体数值取自源码注释与断言**（`draft_vocab_size` 由 ckpt
  config 给，注释与 `lm_head` 断言指向 32000；`fused_outdim_*` 由 config 算），**未加载真实 ckpt 验证**。
- **`_allocate_intermediates` 的 buffer 生命周期/alias 未做数据流分析**（同 `docs/46` §8 的口径）：
  跨 step 覆写是否与图中其它写者冲突，只记静态注册、不判对错。
- **§5 的契约链是静态推断**：「argmax 会选中补零行」需「该行 256 个真实 logits 全为负」这一前提；本文
  只说明代码**允许**这一路径发生、且下游**有** sentinel 兜底，未构造实例证明它真发生。
- **未与 GQA/norm 之外的 family builder 对照**（如 DeepSeek-V3/GLM 的 `_prepare_weights` 是否有同构写法）；
  `docs/27`（权重预处理层）覆盖的是 HF→state_dict 的**上一层**，本文只到 builder 内部。

---

## 8. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/46` 同快照）。
- builder：`python/mirage/mpk/models/eagle3/builder.py`
  （`_resolve_draft_path:15`；`_load_draft_state_dict:23`；`class Eagle3Builder:34`；`__init__:58`；
  `world_size==1` 断言 `:70`；`mbt:77`；`_attach:102`；`_new:111`；`_prepare_weights:119`
  （QKV `:123-131`、gate/up `:133-139`、d2t `:141-147`、fc `:149-153`、lm_head `:155-174`、
  KV cache `:176-183`、dummy norm `:185-188`）；`_attach_weights:190`（清单 `:192-218`）；
  `_allocate_intermediates:220`（logits/argmax `:256-264`、`all_draft_ids:267-269`）；
  `build_draft_loop:271`（三调用 `:296-298`；`paged_attention_layer` K=1 `:356-368`、K>1 `:369-383`；
  argmax `:425-434`；d2t remap `:436-441`；scatter `:443-447`））。
- 工具：`python/mirage/mpk/models/utils.py`（`grid_for_rmsnorm_linear_layer:3-15`；
  `shuffle_tensors:50-128`，交错循环 `:118-128`；`inplace_shuffle_tensors:130-156`）。
- 前端层：`python/mirage/mpk/persistent_kernel.py`
  （`attach_input:746`（裸指针/GC 注释 `:772-774`）；`new_tensor:782`；
  `PersistentKernel.shuffle_tensors:811`；`rmsnorm_linear_layer:883`；`paged_attention_layer:1303`
  （q_norm/k_norm 断言 `:1345-1352`；params `:1354-1402`=含 `q_len_override/tail_offset`）；
  `argmax_partial_layer:2603`（`num_tasks=grid_dim[0]`、`output_size:2616`、`vocab_size` 注释 `:2614-2618`）；
  `argmax_reduce_layer:2632`；`mtp_token_scatter_layer:2895`；`eagle3_d2t_remap_layer:3184`）。
- device：`include/mirage/persistent_kernel/tasks/speculative_decoding/eagle3_ops.cuh`
  （`eagle3_d2t_remap_kernel:102-122`，含 `target_id = hot_id + d2t[hot_id]` 约定 `:85-88`、
  padding guard `:91-95`/`:116-120`）。
- demo：`demo/qwen3/demo_30B_A3B_eagle3.py`（`Eagle3Builder(...)`:803-813；
  `build_draft_loop(...)`:854-860；`_attach_cache["eagle3_all_draft_ids"]`:882；
  `eagle3_drafts_prev_buf (mbt,K)`:826-829）。
- 相关本专题文档：`docs/46`（Eagle3 draft loop 构图）、`docs/44`（`eagle3_commit`/`d2t_remap` device 身体）、
  `docs/43`（MTP 验证链）、`docs/27`（权重预处理层：HF→state_dict）。
