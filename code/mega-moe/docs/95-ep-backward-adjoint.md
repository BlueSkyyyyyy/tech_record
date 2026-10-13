# MoE dispatch / combine 的反向语义：伴随恒等式与路由权重梯度（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P85 增量（第 95 篇）。
> 第 4 篇推了 MoE 反向的**数学**（linear1/linear2 的 dgrad/wgrad、SwiGLU 反向、router/topk 反向）、
> 第 12/13/14 篇拆了**计算侧**的 wgrad/dgrad grouped GEMM、第 43 篇拆了 EPLB 的**权重预取/梯度归约**、
> 第 7/51/58/61/71 篇拆了**通信侧**的 dispatch/combine 设备 kernel。但把前向通信接回反向的那一层
> ——**「dispatch 的反向是 combine、combine 的反向是 dispatch」到底为什么成立、路由权重 `w` 的梯度怎么走**
> ——一直只在第 7 篇 (§8) 和 README 里被一句话带过。本篇把它单独拆透。
>
> **关键点**：这一层是**纯线性代数 + host/Python 语义**，与 SM100 / NCCL 设备 API 无关。端到端
> `deep_ep.EPBuffer.dispatch/combine` 依赖 C++/NCCL 设备通信 API（本机 NCCL 2.29.7 不匹配，见「阻塞」）
> → **仅分析**；但反向恒等式可在 H100 上真机验证：单进程纯张量验四条恒等式 + 路由权重梯度对拍 autograd，
> 8×H100 真机 NCCL 用**真实 `refs.dispatch`** 跨 rank 验伴随恒等式。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`，v2.5.0）：
> - 训练/推理示例：`README.md:158-289`（四个 helper `dispatch_forward/dispatch_backward/combine_forward/combine_backward`）
> - host API：`deep_ep/buffers/ep.py`（`dispatch:562-757`、`combine:770-843`）
> - 参考实现：`deep_ep/utils/refs.py`（`dispatch:10-123`、`combine:177-243`、`ordered_accumulate:156-174`）
> - 验收判据：`tests/ep/test_ep.py:514`（`combined_topk_weights == topk_weights`）
> - 设备侧：`impls/ep/combine_utils.cuh:58-119`（`combine_reduce` 求 slot 和）、
>   `impls/ep/combine_reduce_epilogue.cuh:22-139`（`combined_topk_weights` 的**选择**语义）

---

## 1. 一句话总览：EP 的「反向」不是新 kernel，而是把两个原语互为转置

MoE 的通信侧只有两个原语（第 7 篇 §8）：

- **dispatch**：把本地 token 按 `topk_idx` 送到「拥有目标专家」的 rank；在源 rank 视角它把
  一个 token **复制/散射**到若干接收行（row space）。
- **combine**：把各 rank 上的专家输出按原路送回 token 的来源 rank，并把同一个 token 的各行
  **求和/归约**回一个 token。

把 dispatch 看成线性算子 `D : ℝ^{T×H} → ℝ^{N×H}`（`N` = 接收行总数）、combine 的 token 求和看成
`C : ℝ^{N×H} → ℝ^{T×H}`，则：

- `C` 恰好是 `D` 的**转置/伴随**（`Dᵀ`）：`D` 复制、`Dᵀ` 求和；
- 反过来 `D` 是 `C` 的伴随：`D` 复制、`C` 求和。**两个原语互为转置**。

于是**反向传播不需要任何新的通信 kernel**：

| 前向 | 反向（w.r.t. 输入） | 反向用什么原语 |
|---|---|---|
| `recv_x = D(x)` | `grad_x = Dᵀ(grad_recv_x)` | **combine**（按 token 求和） |
| `recv_w = G(w)`（权重 gather） | `grad_w = Gᵀ(grad_recv_w)` | **combine**（权重 reduce） |
| `combined_x = C(y)` | `grad_y = Cᵀ(grad_combined_x)` | **dispatch**（按行复制） |

这正是 README 里两个 helper 的全部内容，见 §3。

## 2. 两类线性算子的精确定义（对齐 `refs.py`）

### 2.1 dispatch = 复制 + 置换

`refs.dispatch`（`refs.py:10-123`）对每个目的 rank `d` 做：

- `refs.py:67`：`mask_to_send = ((expert_start<=topk_idx) & (topk_idx<expert_end)).any(dim=1)` ——
  **只要该 token 的 top-k 里有任一专家落在 rank `d`，就整行发给它**（非 expand 布局：一个 token
  对一个目的 rank 只发一行，不是每个 k 槽一行）。
- `refs.py:76-77`：把该行 `topk_idx` 中不属于 rank `d` 的槽置 `-1`。
- `refs.py:85-90`：`send_src_token_idx = 源 rank*num_max + 本地 token 索引`，接收侧按此**升序**排列
  （docstring `:16` 明说）。
- 因此对固定的 `(token, 目的 rank)`：`recv_x[row] = x[token]`（复制），
  `recv_w[row] = w[token, :]`（**整行**权重，`refs.py:75`）。行到 token 的映射由一个「源 token 索引」
  数组给出——这就是 `EPHandle.recv_src_metadata` 的语义（第 83 篇）。

`refs.py:50-53` 校验 `topk_weights` 形状与 `topk_idx` 一致；`refs.py:112-117` 在接收侧再次把越界槽置 `-1`。

### 2.2 combine = 求和 + 权重选择

`refs.combine`（`refs.py:177-243`）做的是**分组求和**：`grouped_reduce`（`:201-234`）把同一个 token
的若干行按「组 id」**严格升序累加**到该组最右槽，`ordered_accumulate`（`:156-174`）再沿 top-k 求和并
加上 `bias`。等价地，`combine` 把行空间按 token 分组、组内**求和**：`combined_x[t] = Σ_{row∈t} y[row]`。

设备侧 `combine_reduce`（`combine_utils.cuh:58-119`）正是这个语义：`enable_hadd_bypass` 路径直接把
slot0+slot1 用 `bf16x2` 相加（`:73-108`），通用路径对 `kNumValidTopk` 个 slot 求和（`:110-166`）。

**注意**：combine 对 `x` 是求和，但**不做权重乘法**——README `:279` 明说「专家计算在 combine 之前就用
这些 router 权重乘好了；combine 只把送进来的专家输出求和」。这与第 4/14 篇「wgrad/dgrad 在计算侧」是
一致的：权重乘法属于 grouped GEMM 的 epilogue。

## 3. README 的两个 helper 就是伴随恒等式

`README.md:204-220`：

```
def dispatch_backward(grad_recv_x, grad_recv_topk_weights, handle, bias=None):
    """The backward pass of MoE dispatch is actually a combine."""
    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x, handle=handle, bias=bias,
        topk_weights=grad_recv_topk_weights, num_sms=_num_comm_sms,
        async_with_compute_stream=True)
    return combined_grad_x, combined_grad_topk_weights, event
```

`README.md:242-258`：

```
def combine_backward(grad_combined_x, handle):
    """The backward pass of MoE combine is actually a dispatch."""
    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x, handle=handle, num_sms=_num_comm_sms,
        async_with_compute_stream=True, do_expand=True,
        do_zero_padding=True, use_tma_aligned_col_sf=True)
    return grad_x, event
```

逐字对应 §1 的表：

- **`dispatch_backward`**：`grad_x = Dᵀ(grad_recv_x)`（combine 把行按 token 求和），
  `grad_w = Dᵀ(grad_recv_w)`——后者就是 `combine(..., topk_weights=grad_recv_topk_weights)` 里
  combine 顺带返回的 `combined_grad_topk_weights`（因为 `w` 也是被 gather 的操作数）。
- **`combine_backward`**：`grad_y = Cᵀ(grad_combined_x)`（dispatch 把 token 梯度复制回每一行）。
  README `:253-255` 特意开 `do_expand=True, do_zero_padding=True`，让复制出的行空间与专家 GEMM 的
  展开布局对齐。

**`combine` 的 `topk_weights` 参数于是有两种用途**（`ep.py:788-790`）：非 expand 是 `[T,K]`、expand 是
1D `[N]`；它既给出「反向里要 reduce 的权重梯度」，又是 `combined_topk_weights` 的来源。

## 4. 一个隐蔽的非平凡点：非 expand 的复制让权重不能朴素求和

非 expand dispatch 按**目的 rank**复制（`refs.py:67`），所以一个 token 若在 `d` 个 rank 上都有专家，
它的整行 `w` 会被复制 `d` 份。若把收到的行按 token **朴素求和**，得到的是 `d·w` 而不是 `w`。

H100 实测把它量了出来（Part B 的 B3）：T=128、E=32、world=8、K=4 时**平均 degree = 3.328**，
`Σ recv_w == degree·w` 逐元素成立。这解释了为什么 DeepEP 的 `combined_topk_weights` 要等于**原
`topk_weights`**（`test_ep.py:514` 的 `torch.equal` 断言）——它**不能**是朴素求和。

设备侧的实现正是**选择而非求和**：`combine_reduce_epilogue.cuh:128-139` 用
`master_lane_idx = get_master_lane_idx(match(stored_dst_rank_idx))` 选出「主 rank」，只从**那一个**
buffer 读 `get_topk_weights_ptr()` 的值写进 `combined_topk_weights`——因为所有副本携带的是同一行 `w`，
选一份就等于还原。**这是一个只在「复制语义」下才成立的巧妙设计**：求和会错 `d×`，取最大也不对（权重
可能为 0），只有「任选一个非空副本」才精确。

## 5. 训练图里的路由权重梯度（权重乘法融合点）

真正的 MoE 前向把权重乘法放在专家计算里（README `:279`）：`z[t] = Σ_k w[t,k]·y[t,k]`（`y` 是各槽的
专家输出）。它的反向（第 4 篇 (7)/(8) 的展开）为：

- `∂L/∂y[t,k,h] = w[t,k]·g[t,h]` —— **按槽复制**（`k` 维广播 `w`），结构上就是 `D`（dispatch）；
- `∂L/∂w[t,k]   = <y[t,k], g[t]>` —— **每槽一个内积**（把 `h` 收缩掉）。

H100 实测（Part A 的 A5a/A5b）用 fp64 + autograd 对拍：两条公式与小节 `≤1e-12` 逐元素吻合。
把二者与 §1 对照：`∂L/∂y` 是「按行复制」= dispatch，`∂L/∂w` 是「沿 `k` 归约」= combine 的权重 reduce。
**所以整条 MoE 反向的通信部分 = 把前向的 dispatch/combine 原样再跑一遍、只是喂进去的是梯度**——
没有新的通信 kernel，这也是官方能在 DeepGEMM 侧只实现前向 mega-kernel、反向用普通 grouped GEMM +
复用前向路由（第 4/14 篇）的原因。

## 6. `EPHandle`：前向存路由账本，反向只换数据

反向要复现前向的复制/置换，必须记住行到 token 的映射。这由 `EPHandle` 承担（第 83 篇详拆），反向相关
的字段：`recv_src_metadata`（行→源 token 全局索引）、`topk_idx`（路由，带版本守卫 `ep.py:103-111`）、
两套 `psum_num_recv_tokens_per_*`（专家分组前缀和）。`ep.py:613-615` 的 dispatch docstring 明说：缓存
handle 时可以只换 `topk_weights` 复用整套 expand 布局——**前向路由在反向不必重算**。README `:281`
补充训练约定：保存前向 handle 供 `combine_backward/dispatch_backward` 用；`topk_idx` 在所有 handle
用尽前不得改动。

## 7. H100 实测

`harness/exp_ep_backward_adjoint.{sh,py,out.txt}`（DeepEP `refs.py` sha256 `c5f551b4…`，总判定 ALL PASS）：

**Part A（单进程纯张量，R=3, T=8, H=6, E=12, K=3）**

| 检查 | 恒等式 | 实测 |
|---|---|---|
| A1 | `<D(x), g> == <x, Dᵀ(g)>` | LHS 0.765600 / RHS 0.765601 ✓ |
| A2 | `<G(w), gw> == <w, Gᵀ(gw)>` | LHS −0.466352 / RHS −0.466353 ✓ |
| A3 | `<C(y), v> == <y, D(v)>` | LHS −5.027745 / RHS −5.027744 ✓ |
| A4 | 朴素求和 combine == 真实 `refs.combine`（单 rank 全专家） | 逐位 ✓ |
| A5a | `∂L/∂y[t,k,h] == w[t,k]·g[t,h]` | autograd 对拍 ≤1e-12 ✓ |
| A5b | `∂L/∂w[t,k] == <y[t,k], g[t]>` | autograd 对拍 ≤1e-12 ✓ |
| A6 | `dispatch_backward == combine` | `grad_x=Dᵀ(grad_recv_x)`、`grad_w=Dᵀ(grad_recv_w)` ✓ |

**Part B（8×H100 真机 NCCL，真实 `refs.dispatch`，T=128, H=64, E=32, K=4；标量内积用 `all_reduce` 聚合）**

| 检查 | 恒等式 | 实测 |
|---|---|---|
| B1 | 跨 rank `<D(x), g> == <x, Dᵀ(g)>` | LHS 100.3833 / RHS 100.3833 ✓ |
| B2 | 跨 rank `<G(w), gw> == <w, Gᵀ(gw)>` | LHS −18.4716 / RHS −18.4716 ✓ |
| B3 | `Σ recv_w == degree·w`（复制语义） | 平均 degree 3.328，逐元素 ✓ |

Part B 用真实 `refs.dispatch`（内部走 `torch.distributed.all_to_all_single`，`refs.py:94-110`）跑了真正的
跨 rank all-to-all，再在**全局行空间**上算内积、用 `all_reduce` 求和——所以验证的是真通信路径的伴随性，
而非单进程模拟。

## 8. 与既有篇章的关系与边界

- 第 4 篇：MoE 反向的**数学**（含 router/topk 反向）；本篇是它在**通信原语**上的对应物。
- 第 7 篇 §8：一句话点出「combine 的前向就是 dispatch 的反向」；本篇把它补成可验证的伴随恒等式 + 权重梯度。
- 第 12/13/14 篇：反向的**计算侧**（wgrad/dgrad grouped GEMM）；本篇是**通信侧**。
- 第 43 篇：EPLB 冗余专家的权重预取/梯度归约（另一条反向通信）；本篇是普通路由的 dispatch/combine 反向。
- 第 83 篇：`EPHandle` 作为「dispatch→GEMM→combine→反向」的路由账本；本篇讲账本**怎么被反向消费**。
- 第 65/85 篇：`refs.py` / `test_ep.py` 的前向语义规范；本篇在其上加反向恒等式与权重梯度。

**边界**：端到端 `EPBuffer.dispatch/combine` 反向 helper 的真实设备路径依赖 NCCL ≥2.31（本机 2.29.7，
见「阻塞」）→ 仅分析；本篇的伴随恒等式与权重梯度是纯语义/纯张量事实，H100 已真机验证。
