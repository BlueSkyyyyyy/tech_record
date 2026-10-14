# KDA 专属 fused pre-GDR 的**原生 Triton 反向**（M13-2）

> `docs/15`（M13-1）把 KDA 专属 fused pre-GDR 实现成三 kernel 的前向 Triton 核族
> （conv+SiLU / L2-norm / beta sigmoid），但它的 autograd 包装**只有前向**：反向用一个
> 可微的 torch 原语链（`_FusedPreGDRAutograd.backward`，`src/kda_pregdr_fused.py:207-224`）
> 重放一遍 eager 代数。本篇（M13-2）把它换成**原生 Triton 反向**，并顺带修掉 M13-1
> 一个此前**没被发现**的缺口。
>
> 实现见 `src/kda_pregdr_bwd.py`，实验见 `harness/check_port_pregdr_bwd_kernel.py`
> （`harness/check_port_pregdr_bwd_kernel.out.txt`，空闲卡单进程）。**未改 Megatron-LM
> 一行**；本文所有数字来自该 out.txt。

## 1. 为什么要做：M13-1 的参考反向漏了 conv 参数

M13-1 的 `backward`（`src/kda_pregdr_fused.py:207-224`）在函数内部**重新**跑一遍
`_ref_torch`，用 **detached clone** 求导：

```
q_ = qkvfg.detach().requires_grad_(True)
b_ = beta.detach().requires_grad_(True)
w_ = module.conv1d.weight.detach().requires_grad_(True)   # 注意是 clone
...
loss.backward()
return (None, q_.grad, b_.grad, None, None)               # 只回传 q/beta
```

`w_`/`bias_` 的梯度算出来了，但**没有回传**（`src/kda_pregdr_fused.py:224` 只 return
`q_.grad, b_.grad`）。而 M13-1 的整层 drop-in 对拍（`harness/check_port_pregdr_kernel.py:247`）
是：

```
if ga is None or gb is None:
    continue
```

于是 `conv1d.weight`/`conv1d.bias` 因为 fused 侧是 `None` 被**静默跳过**——M13-1 的
Part C 看着全对，实际上**根本没比较卷积参数**。这不是数值问题，是**测试覆盖的漏洞**：
它会把「fused 反向完全不产 conv 梯度」这种错误当成通过。

M13-2 的两件事因此绑在一起：写原生反向，并让 conv 参数成为 autograd `Function` 的
**显式输入**（梯度自然流回真实参数），从而**关掉这个跳过**。

## 2. 三个反向 kernel

`docs/15` §1 的三个 scope 各有一个反向。`src/kda_pregdr_bwd.py` 新增：

### 2.1 SiLU 反向（`_kda_silu_bwd_kernel:60`）

前向 conv scope 在因果卷积后接 `silu(a)=a·σ(a)`（`src/kda_pregdr_fused.py:83-84`）。
反向需要**激活前的 `a`**（记为 `pre`）：

```
σ = sigmoid(a);  silu'(a) = σ·(1 + a·(1-σ))
da = dconv · silu'(a)
```

`pre` 由前向**同一 kernel、关掉激活**再跑一遍得到（`_forward_raw_save:164` 分配、
`:175-176` 第二趟），同一累加序故 `pre` 与前向逐位一致，没有额外求导误差。

### 2.2 深度可分离因果卷积反向（`_kda_conv_bwd_kernel:79`）

前向（channel-last，taps `w=0..W-1`）：`a[t,c] = Σ_w x[t+w-(W-1), c]·wgt[c,w]`。反向：

```
dX[m,c]  = Σ_w da[m+(W-1)-w, c] · wgt[c,w]
dW[c,w]  = Σ_{b,t} x[b, t+w-(W-1), c] · da[b,t,c]
dbias[c] = Σ_{b,t} da[b,t,c]
```

`dX` 是 fp32 输出（按 shift 直接求和，无原子）；`dW`/`dbias` 是跨 `(b,t)` 归约，用
`tl.atomic_add` 累积（`_kda_conv_bwd_kernel:110-117`）。**这是本模块唯一用原子加的地方**，
实测对拍 1e-6（Part A），说明累积序无碍。

### 2.3 L2-norm 反向（`_kda_l2norm_bwd_kernel:124`）

前向 `y = x·rstd`，`rstd = 1/√(Σx²+ε)`。反向：

```
dx = rstd·dy − rstd³ · x · (dy·x)
```

逐行（最后一维 `KD`）在 fp32 内做，和前向 `_kda_l2norm_kernel`（`src/kda_pregdr_fused.py:94`）
同一分块 `[BT,KD]`。

### 2.4 sigmoid 反向（`_kda_sigmoid_bwd_kernel:143`）

beta scope：`dx = y·(1−y)·dy`，直接吃前向落下的 `beta_sig`（不再重算 sigmoid）。

## 3. 组装（`_backward_raw:204`）

反向按前向的逆序把梯度拼回 `qkvfg`/`beta`/conv 参数：

1. 六项上游梯度 `(dq, dk, dv, dgate, dbeta_sig, draw_g)`；`gate`/`raw_g` 是 `qkvfg` 的
   view，其梯度**直接散射回**对应通道（`_backward_raw:254-257`），不过任何 kernel。
2. `cat(dq,dk)`（沿 `[Hk,Hk]` split 轴）→ L2-norm 反向 → `dqk_in`；与 `dv` 拼成
   **齐整的 `[B,T,C]`** `dconv`（前向 conv output 覆盖全部 `C=2qk+vd` 通道，
   `_backward_raw:222-231`）。
3. SiLU 反向（用保存的 `pre`）→ `da`；conv 反向 → `dX`/`dW`/`dbias`。
4. sigmoid 反向 → `dbeta`。
5. `dX` 写回 `qkvfg[..., :C]`、`draw_g`/`dgate` 写回后两段，转回 SBHD；
   `dW`/`dbias` 转成 conv 参数形状 `[C,1,W]`/`[C]`（`_backward_raw:254-262`）。

## 4. autograd 重接（`_FusedPreGDRNativeAutograd:265`）

与 M13-1 的关键区别：`conv1d.weight`/`conv1d.bias` 作为 **Function 的输入**传入
（`kda_pre_gdr_fused_native:292`）：

```
out = _FusedPreGDRNativeAutograd.apply(module, qkvfg, beta,
                                       module.conv1d.weight, module.conv1d.bias,
                                       batch, seq_len)
```

于是 `backward` 返回 `(None, dqkvfg, dbeta, dweight, dbias, None, None)`，conv 参数梯度
**流回真实 Parameter**；无 bias 的层（本子专题的真实层即 `conv_bias=False`）由两处守卫兼容：
`_forward_raw_save` 给 kernel 传 dummy 指针（`conv_b_eff`）、`backward` 在该位置回传 `None`
（`has_conv_bias`）。至此 M13-1 §1 的「跳过」被彻底关掉。

## 5. 数值锚定

### 5.1 Part A：孤立反向 kernel vs 解析参考（`harness:57`）

用**同一批保存的中间量**（`x`/`pre`/`query_key`）驱动 Triton 反向，参考侧用解析公式
（torch）算出 `dqk`/`dconv`/`da`，只对 `F.conv1d` 段落做一次 autograd 取 conv 的输入/权重
梯度——**这样比的是 kernel 实现，而不是前向 conv 后端**。

| T×B | fp32 dweight | fp32 dx | fp32 dbeta | bf16 dweight | bf16 dx | bf16 dbeta |
|---|---|---|---|---|---|---|
| 64×2 | 9.5e-7 | 1.2e-6 | 6.0e-8 | 1.2e-6 | 1.5e-2 | 1.9e-3 |
| 128×2 | 1.9e-6 | 1.2e-6 | 1.2e-7 | 1.4e-6 | 1.4e-2 | 1.9e-3 |
| 250×1 | 1.6e-6 | 1.2e-6 | 6.0e-8 | 1.4e-6 | 1.5e-2 | 1.9e-3 |
| 256×1 | 1.9e-6 | 9.5e-7 | 6.0e-8 | 1.7e-6 | 1.2e-2 | 2.0e-3 |
| 1024×2 | 5.7e-6 | 1.4e-6 | 6.0e-8 | 4.8e-6 | 1.5e-2 | 2.0e-3 |

（本层 `conv_bias=False`，故 `dbias` 列恒为 `nan`——参考侧没有 bias。）

**怎么读**：`dweight`/`dbias` 是**以 fp32 返回**的，故两种 dtype 下都落 **1e-6**
（= 解析参考的 fp32 底噪），**直接证明卷积反向的代数与实现正确**。`dx`/`dbeta` 的 bf16 列
（~1.5e-2 / ~1.9e-3）**不是反向误差**：`dqkvfg`/`dbeta` 在出口被 cast 回 `qkvfg.dtype`/`beta.dtype`
（`_backward_raw:261-263`），而 `|dx|≈2.6`、`|dbeta|≈0.25`，bf16 的一个 ulp 就是
`2^-8≈4e-3`/`1e-3` 量级——那正是**输出量化的 ulp**（`~0.5 ulp`），与 §5.1 fp32 列逐位同源。
更强的旁证：同一 kernel 直接输出 fp32 的 `dqk(l2)` 在 bf16 输入下仍 **4.8e-7**。

### 5.2 Part B：端到端反向 vs `torch.autograd`（`harness:120`）

原生反向（经 autograd Function）对 `torch.autograd` 穿**纯 torch 原语前向**：

| T×B | fp32 dqkvfg | fp32 dbeta | fp32 dweight | bf16 dqkvfg | bf16 dbeta | bf16 dweight |
|---|---|---|---|---|---|---|
| 64×2 | 8.3e-7 | 6.0e-8 | 1.9e-6 | 3.1e-2 | 0 | 3.1e-2 |
| 250×1 | 9.5e-7 | 6.0e-8 | 1.4e-6 | 3.1e-2 | 0 | 3.9e-2 |
| 128×2 | 9.5e-7 | 6.0e-8 | 1.9e-6 | 3.1e-2 | 0 | 3.5e-2 |

fp32 全落 **1e-6**。bf16 的 ~3e-2 是**前向量化**：native forward 的 conv/qk 输出存 bf16，
而参考前向是 torch conv（`docs/15` Part B 已记 fp32 1.8e-7、bf16 3.9e-3 的**前向**差
本来就是 conv 后端舍入）；端到端反向把这同一批前向差放大到梯度上，量级与 `|dqkvfg|·ulp` 一致。
`dbeta` 逐位 0（它只吃 sigmoid，不经过 conv）。

### 5.3 Part C：整层 drop-in（**含 conv 参数**）（`harness:167`）

同权重、只换 `pre_gated_delta_rule` 为原生 fused（前向 Triton + 反向 Triton），比
forward / 输入梯度 / **全部**参数梯度：

| dtype | out max_abs | hs.grad max_abs | param.grad max_abs（rel） | 比较参数数 / 跳过数 | `conv1d.weight.grad`（eager vs native） |
|---|---|---|---|---|---|
| fp32 | 0.0 | 2.8e-5 | 1.25e-3（1.95e-5） | **7 / 0** | 8.637 vs **8.637** |
| bf16 | 0.0 | 7.8e-3 | 2.50e-1（3.9e-3） | **7 / 0** | 8.625 vs **8.625** |

对比 M13-1（`docs/15` Part C）：那时 `param.grad rel 2.08e-4` 是**跳过 conv 参数**后的数；
本轮 **7 个参数全部比较、0 跳过**，且 `conv1d.weight.grad` 两侧同值（fp32 8.637、bf16 8.625）
——**M13-1 的漏洞关闭**，整层梯度（含卷积权重）由原生反向承担、量级与 M10/M11/M12 的
整层 bf16 锚定同阶。

### 5.4 Part D：隔离计时（前向+反向）（`harness:228`）

| T×B | fp32 wall eager/native（×） | fp32 dev（×） | bf16 wall（×） | bf16 dev（×） |
|---|---|---|---|---|
| 64×2 | 1463/840（1.74） | 69.2/86.8（0.80） | 1712/1224（1.40） | 86.7/125.4（0.69） |
| 256×2 | 1555/1124（1.38） | 99.2/113.9（0.87） | 1694/1185（1.43） | 116.4/152.9（0.76） |
| 1024×2 | 1674/1081（1.55） | 117.9/129.9（0.91） | 1818/1320（1.38） | 128.0/168.3（0.76） |
| 1024×8 | 1642/1216（1.35） | 254.3/258.4（0.98） | 1787/1321（1.35） | 226.9/287.1（0.79） |

**诚实结论**：(a) `wall` 上原生核族稳定赢 **1.35–1.74×**——eager `pre_gated_delta_rule`
的 host 税恒定 ~1.5 ms/次（与 `docs/15` Part D 一致），原生核族 launch 数更少；
(b) 纯设备 `dev` 上原生**小 shape 略慢（0.69–0.98×）**——这是**原型代价**不是算法问题：
原生反向是 **4 个独立 kernel**（silu / conv / l2norm / sigmoid）+ 前向为存 `pre` 多跑一遍
conv；eager 侧用 FLA 的融合 `causal_conv1d`/`l2norm` 及其高度优化反向。把 conv 反向合并进
一个 kernel、或省掉 `pre` 的第二趟，是可做的工程化，**不影响本篇的数值/覆盖结论**。

## 6. 结论与边界

- **原生 Triton 反向落地**：三个反向 kernel + 组装，孤立对拍（Part A）与端到端（Part B）
  fp32 全落 **1e-6**；`dweight` 在 bf16 下仍 1e-6，证反向代数正确。
- **修掉 M13-1 的测试漏洞**：conv 参数作为 Function 输入，整层 drop-in 现在**比较全部 7 个
  参数、0 跳过**，`conv1d.weight.grad` 两侧同值。
- **未改 Megatron-LM 一行**；入口 `kda_pre_gdr_fused_native` 与 M13-1 的
  `kda_pre_gdr_fused_autograd` 同签名、可互换。
- **边界**：`BT=64`/fp32-internal；`conv_bias` 支持但本层为 `False`（`dbias` 未实测非零）；
  反向为多 kernel 原型（`dev` 小 shape 略慢于 FLA 融合反向）；仍未接 `kda.py:399` 的
  `gdn_pre_gated_delta_rule_fusion` guard（与 M12/M13 同一边界：不修改 Megatron 树）。
