# 多卡 EP 端到端 MoE：前向/反向与单卡稠密参考的梯度核验（8×H100 实测）

> 本文对应路线图 **P131-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 141 篇。
>
> 专题到第 140 篇，**Mega MoE 的每一段都被拆过**：前向的 dispatch→L1→SwiGLU→L2→combine
> （第 2/3/9/10/11 篇）、反向的四个 grouped GEMM（第 12/13/14 篇）、前向的 H100 数值复刻
> （第 99 篇）、反向的 H100 数值复刻（第 111 篇）、dispatch/combine 的**伴随恒等式**（第 95 篇）。
> 但有一个**结构性的空档**一直没人补：
>
> - 第 99 篇的「前向复刻」和 第 111 篇的「反向复刻」都是**单卡**的——作者**明确把 dispatch/combine
>   的通信折算成单卡 gather/scatter**（第 99 篇 §2 原文「跳过通信」），所以它们的误差分解里**不含
>   通信**、也**没有多 rank**；
> - 第 95 篇虽然跨 8×H100 跑了真实 `refs.dispatch`，但它验的是**两个原语互为转置**这条**线性代数
>   恒等式**，并**没有把 EP 接进一条完整的 MoE 前向/反向图**，更没有量「参数梯度」。
>
> 于是「EP 真的把 token 散到别的 rank、算完再收回来，**前向输出**与 **dX/dWg/dW1/dW2 四组梯度**
> 是否与单卡稠密 MoE 严格一致」这件事，一直没被端到端量过。本篇补上：在 **8×H100** 上用真实 NCCL
> `all_to_all_single` 实现 expand 版 dispatch/combine，把整条 MoE 前向 + autograd 反向与单卡稠密
> 参考对拍，并给 fp64/fp32/bf16 三档、真实通信量与耗时。
>
> 源码坐标（`~/github/DeepEP@93eb6eb`、v2.5.0，行号已核对）：
> - `deep_ep/utils/refs.py:10-123` —— dispatch 参考：`:67` 按目的 rank 的 `mask`、`:76-77` 越界 slot 置 `-1`、
>   `:90` 源 token 全局索引 `rank*num_max+local`、`:93-94` 换长度、`:105` 换数据（两次 `all_to_all_single`）
> - `deep_ep/utils/refs.py:177-243` —— combine 参考：`:201-234` `grouped_reduce` 分组求和、`:242` `ordered_accumulate` 再加 bias
> - `deep_ep/utils/refs.py:156-174` —— `ordered_accumulate`（沿 top-k 维按序求和）
> - `deep_ep/buffers/ep.py:562-575` dispatch 签名；`:577` `do_expand` 开关；`:619` 「one slot per expert per token」docstring
> - `README.md:204-220/242-258` —— `dispatch_backward`=combine、`combine_backward`=dispatch 两个 helper
>
> 可跑性：全篇是**纯 PyTorch 张量 + NCCL `all_to_all_single`**（架构无关，不碰 UMMA/TMA/tcgen05），
> H100 真机可跑；Mega MoE 融合 kernel 本体是 SM100 专属（`sm100_fp8_fp4_mega_moe.cuh`）→ **仅分析**。
> 原始输出：`harness/exp_ep_e2e_moe.out.txt`（`harness/exp_ep_e2e_moe.{py,sh}`）。

## 1. 一句话：把「通信」真正放回 MoE 的图里

前面的复刻都走「单卡 gather/scatter」的捷径，等价于假设 `world=1`。本篇把 `world` 拉回 8：

```
每 rank 本地 x[T,H] ──router(softmax+topk)──> (idx[T,k], w[T,k])
    │
    ├─ expand dispatch：每个 (token,k-slot) 一行，按 dst_rank=idx//per_rank 分组
    │       └─ all_to_all_single（换长度 + 换 x / expert / w / src 四组）
    │
    ├─ 收方：按本地专家 argsort 成 m-grouped contiguous → L1 → SwiGLU → L2（× w）
    │
    └─ combine：all_to_all_single 送回源 rank（split 对调）→ 按 src token index_add 求和
```

整个 `out_ep` 与它的 autograd 反向，和「把所有专家/所有 token 放一张卡、直接稠密 matmul」的参考逐张量对比。

**为什么这能证「反向无 mega-kernel」**（专题反复出现的结论，第 4/14/95/104 篇）：MoE 反向的**通信部分**
就是把前向的 dispatch/combine **原样再跑一遍、只是喂进去的是梯度**——因为 `all_to_all` 的雅可比就是
它自己的转置（把 `input_split_sizes` 与 `output_split_sizes` 对调）。本篇的 `_A2AFn` 把这个事实
**写成代码**：`forward` 用一套 split、`backward` 用对调的 split 再跑一次同一个原语（§4）。

## 2. dispatch：从「参考实现」到本篇的 expand 版

DeepEP 的 `refs.dispatch`（`refs.py:10-123`）是**非 expand**（non-expand）语义：docstring `:16` 明说
「Not expanded. Sorted by rank and then by token」。它对**每个目的 rank** `d` 取
`mask_to_send = ((d*per<=topk_idx)&(topk_idx<(d+1)*per)).any(dim=1)`（`:67`）——**一个 token 只要在
rank `d` 上有任一专家，就把整行发过去**，并把不属于 `d` 的槽置 `-1`（`:76-77`），接收侧再按
`:115-117` 把本地范围外的槽置 `-1`。这正是第 95 篇量到「复制 degree ≈ 3.3、`Σ recv_w = degree·w`」
的来源。

**训练真正用的是 expand 布局**（`do_expand=True`，`ep.py:577/619`：docstring `:619` 写明
「one slot per expert per token」）——**一个 (token, k-slot) 一行**，没有「整行复制」的 degree 问题。
`combine_backward` 在 `README.md:253-255` 特意开 `do_expand=True, do_zero_padding=True`，就是让
反向复制出的行空间与专家 GEMM 的展开布局对齐。**本篇采用 expand**，因此：
- 行空间 = `Σ_t k`（而不是非 expand 的 `Σ_t (含该 token 的目的 rank 数)`）；
- 每行只需一个**本地专家号**（`expert - d*per_rank`），不需要在收方再 mask `-1`；
- combine 是干净的**按 token 求和**，没有 degree 缩放。

实现（`ep_forward`）：先把 `dest = idx // per_rank`，对每个 `d` 收集 `(x_row, local_e, w, src)`；
`send_counts[d]` 统计行数；两次 `all_to_all_single`——一次换长度（`refs.py:93-94` 的同款手法），
一次换数据（`refs.py:105` 的同款手法，只是把 4 组张量各自 a2a）。

## 3. 收方：把到货行变成 m-grouped，跑 L1/L2

行到达后**按源 rank 分组**（`all_to_all_single` 的输出顺序）。要做专家 GEMM，先
`order = argsort(recv_e, stable=True)` 把它们按本地专家号排开（这就是 **m-grouped contiguous** 的
构建，第 11 篇讲的布局在此处「长出来」），再用 `bincount` 算每专家块偏移：

```
xe = rx[order]                         # [N,H]
counts = bincount(re, minlength=per_rank); offs = cumsum(counts)
for e in range(per_rank):
    ye_list.append(emlp(xe[a:b], W1_local[e], W2_local[e]) * we[a:b, None])
y = cat(ye_list, 0)[inv]               # inv 反置换，回到「按源 rank 分组」的到货顺序
```

`emlp` 就是标准 SwiGLU MLP：`gate = x@W1[:I]ᵀ`、`up = x@W1[I:]ᵀ`、`h = silu(gate)*up`、`out = h@W2ᵀ`。
**权重 `w` 在这里乘进激活**——与第 99 篇 §1 的结论一致（Mega MoE 把 top-k 权重乘在 L1 输出、L2 输入
量化**之前**，于是 combine 退化成纯求和）。combine 送回的每一行按 `recv_src`（全局 token 号）
`index_add` 回 `out_global`，再切出本 rank 的本地 token。

## 4. 反向：`all_to_all` 的雅可比 = 对调 split 的同一个原语

`torch.distributed.all_to_all_single` 是 in-place 风格、**不挂 autograd**（实测 `out.requires_grad=False`；
`torch.distributed.nn.functional` 版本传入预分配 `output` 时同样不挂）。本篇因此手写一个
`torch.autograd.Function`：

```python
class _A2AFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, recv_counts, send_counts):
        out = t.new_empty((sum(recv_counts), *t.shape[1:]))
        dist.all_to_all_single(out, t.contiguous(),
                               output_split_sizes=recv_counts, input_split_sizes=send_counts)
        ctx.rc, ctx.sc = recv_counts, send_counts
        return out
    @staticmethod
    def backward(ctx, g):
        out = g.new_empty((sum(ctx.sc), *g.shape[1:]))
        dist.all_to_all_single(out, g.contiguous(),
                               output_split_sizes=ctx.sc, input_split_sizes=ctx.rc)
        return out, None, None
```

**forward 把「发给 rank d 的 `send_counts[d]` 行」变成「从 rank d 收到 `recv_counts[d]` 行」；backward
把梯度按 `send_counts`/`recv_counts` 对调再搬回去**。这不是「专门为反向写一个通信 kernel」，而是
**同一个原语**——dispatch 的反向就是 combine、combine 的反向就是 dispatch（`README.md:204-220/242-258`、
第 95 篇的伴随表）。于是整条 MoE 的反向通信**零新增 kernel**，这正是官方能只做前向 mega-kernel
（第 4/14/95/104 篇）的根据。

## 5. H100 实测：四组梯度 vs 单卡稠密

配置：`world=8, E=64`（每 rank 8 专家）、`topk=2/4`；单卡参考把**所有专家、所有 token** 放一张卡做
稠密 matmul（同一套权重、同一个 router、同一个确定性 target），EP 与稠密各自只对本 rank 的 token
算 `mse` loss，梯度沿 rank 求和后对比。**判定量是相对误差 `rel = ‖Δ‖/‖ref‖`**。

```
--- [fp64]        T=64  H=128  I=256  E=64 topk=2 dtype=float64 matmul=tf32 ---
  forward  : out_ep vs dense  rel=4.312e-08   max|Δ|=2.384e-07
  backward : dX rel=3.946e-08  dWg rel=2.705e-08  dW1 rel=3.398e-08  dW2 rel=4.443e-08
--- [small]       T=64  H=128  I=256  E=64 topk=2 dtype=float32 matmul=tf32 ---
  forward  : out_ep vs dense  rel=2.815e-04   max|Δ|=1.278e-03
  backward : dX rel=1.716e-04  dWg rel=1.821e-04  dW1 rel=3.052e-04  dW2 rel=3.351e-04
--- [small-strict] T=64 H=128 I=256 E=64 topk=2 dtype=float32 matmul=fp32 ---
  forward  : out_ep vs dense  rel=3.220e-07   max|Δ|=8.531e-07
  backward : dX rel=1.657e-07  dWg rel=1.266e-07  dW1 rel=4.099e-07  dW2 rel=4.158e-07
--- [mid]         T=128 H=512 I=1024 E=64 topk=4 dtype=float32 matmul=tf32 ---
  forward  : out_ep vs dense  rel=1.871e-04   max|Δ|=1.181e-03
  backward : dX rel=2.394e-04  dWg rel=8.124e-05  dW1 rel=2.183e-04  dW2 rel=2.449e-04
--- [bf16]        T=128 H=512 I=1024 E=64 topk=4 dtype=bfloat16 matmul=tf32 ---
  forward  : out_ep vs dense  rel=2.334e-03   max|Δ|=1.562e-02
  backward : dX rel=6.095e-03  dWg rel=4.822e-03  dW1 rel=4.432e-03  dW2 rel=4.756e-03
```

三档读法：

1. **fp64（4.3e-8）**：EP ≡ 稠密到 **4e-8**。这个数不是 fp64 的机器精度（~1e-16），而是本篇 router
   里 `softmax(logits.float())` 的 **fp32 地板**——softmax 在 fp32 里做、其余在 fp64，故两者共享同一个
   fp32 归一化。算法本身是**精确等价**的。
2. **fp32-strict（3.2e-7，`allow_tf32=False`）**：EP ≡ 稠密到 **~3e-7**，即单精度浮点规约顺序差异。
   四组梯度同样 ~1.7e-7~4.2e-7。**这是本篇最干净的一个结论**：把通信放回图里，EP 与稠密在
   单精度下逐张量一致。
3. **tf32（2.8e-4，容器默认）**：NGC 容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，`matmul.allow_tf32`
   默认 **True**（`torch.backends.cuda.matmul.fp32_precision == 'tf32'`）——所有 fp32 matmul 实际走 **TF32**
   （10 位尾数、eps≈5e-4），而 EP 与稠密的 matmul **形状/分批不同**，TF32 舍入路径不同，于是两者差 2.8e-4。
   这个数**不是 EP 的错**，而是「fp32 名字下的 TF32」两个实现的正常分歧——顺手记下：
   **在 NGC 容器里做 fp32 数值对拍，必须显式 `allow_tf32=False`**，否则会把 TF32 误差当成算法误差。
4. **bf16（2.3e-3 前向 / 4~6e-3 反向）**：bf16 输出的舍入地板，与第 99/111 篇的 bf16 地板同量级。

**loss 也逐位对得上**（如 fp64 `ep=1.294152e+00` vs `dense=1.294152e+00`；bf16 `1.333452e+00` vs
`1.333400e+00`，差在 bf16 末位）。

## 6. 真实通信量：dispatch/combine 的「K 倍放大」在真机上

平衡负载下（每个 (token,k-slot) 均匀落到各 rank），expand dispatch 等价于把 `[world, T_local, H]`
沿 rank 维做一次 all-to-all，与第 1 篇 `exp_ep_comm.py` 同口径。`H=7168, bf16, topk=2`：

```
--- dispatch(expand, bf16, 平衡负载) 有效带宽 vs batch（单向；combine 同量）---
    本地    512 tok -> 发出     58.7 MB/rank     562.3 us    104.4 GB/s
    本地   2048 tok -> 发出    234.9 MB/rank    2061.0 us    114.0 GB/s
    本地   8192 tok -> 发出    939.5 MB/rank    7713.8 us    121.8 GB/s
    本地  32768 tok -> 发出   3758.1 MB/rank   30580.9 us    122.9 GB/s
```

**读法**：`T_local` 个 token、`topk=2` ⇒ dispatch 单向搬运 `8·T_local·2·H·2B`（combine 同量）；
这一量与 token 数**线性**，且相对「token 本体」放大了 `topk=2` 倍——这就是第 1 篇讲的
**「EP 通信量相对 token 是 K 倍放大」**在真实 MoE 图里的体现。

**一个诚实的边界**：本次测量在**共享 8 卡节点**上完成，同节点还有其它自驱任务在跑
（实测 `nvidia-smi` GPU0 利用率 100%），故上表带宽是**受争用的下界**（~104–123 GB/s）；同一台机器
在没有争用时，第 1 篇对**同样字节数**（58.7 / 234.9 / 939.5 / 3758.1 MB/rank）量到的是
**303 / 342 / 371 / 392 GB/s**（达 NVLink 硬件上限 ~478 GB/s 的 ~82%）。两表对照正好说明
**通信带宽对「同时还有谁在用这张卡」极其敏感**——这也从侧面支持「通信必须与计算**在同一张卡上**
精细地分 SM、分时」的设计（第 63 篇「半卡上限」、第 65 篇重叠的软硬件两条件）。

## 7. 与既有篇章的关系与边界

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 95 篇 | 跨 8×H100 验 dispatch/combine **互为转置**（线性代数内积） | 把 EP **接进完整 MoE 图**，量四组**参数梯度** |
| 第 99 篇 | 前向数值复刻 + 误差分解（**单卡、跳过通信**） | 前向**含真实多 rank 通信** |
| 第 111 篇 | 反向数值复刻（**单卡、跳过通信**） | 反向**含真实多 rank 通信** |
| 第 1 篇 | all-to-all **裸带宽/时延** | 把带宽代回**真实的数据集** MoE shape |
| 第 4/14/104 篇 | 「反向无 mega-kernel」的论证 | 用 `_A2AFn` 把「反向=对调 split 的原语」**跑成数字** |

**能跑/能核**：EP dispatch→L1→SwiGLU→L2→combine 的前向输出与 dX/dWg/dW1/dW2，vs 单卡稠密参考，
fp64 4e-8 / 真 fp32 3e-7 / bf16 2~6e-3；真实 8×H100 NCCL 通信量与带宽曲线；`all_to_all` 雅可比=
对调 split 的同一原语。

**不能核、仅分析**：Mega MoE 融合 kernel 本体（`sm100_fp8_fp4_mega_moe.cuh`：UMMA/UTCCP/TMEM/
dispatch-combine 的 NVLink 直写与 readiness 水位线）、DeepEP 的真实设备路径
（`EPBuffer.dispatch/combine` 走 NCCL GIN，本机 NCCL 2.29.7 不匹配，见路线图「阻塞」）——
**SM100 专属 / 运行期受阻，仅分析**。本篇用 `all_to_all_single` 复刻其**语义**（行空间、返回顺序、
梯度伴随），并**不**复刻其 SM 内重叠与 NVLink 直写。

**下一轮可做（opt-in）**：把本篇的多 rank EP 图换成**真实 DeepGEMM SM90 grouped FP8 kernel**
（第 99 篇用的 `m_grouped_fp8_gemm_nt_contiguous`）接在 dispatch 之后，量「EP + FP8 量化」叠加后的
端到端误差；或按第 65 篇，把 dispatch 发在通信流、L1 发在计算流，量**真机重叠收益**（在独占卡的
安静环境下）。
