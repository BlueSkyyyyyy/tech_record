# fused pre-GDR 原生反向的**融合与剖析**（M13-3）

> `docs/16`（M13-2）给了 KDA 专属 fused pre-GDR 一个原生 Triton 反向，但它的 Part D 诚实记下
> 一个反常：原生核族 `wall` 稳定赢 eager（host 税），纯设备 `dev` 却**小 shape 反输**
> （fp32 0.80–0.98×、bf16 0.69–0.79×）。本篇先**逐 kernel 剖析**把这个「dev 回归」定位清楚，
> 再据此做一次**有依据的融合**，然后重测。实现见 `src/kda_pregdr_bwd.py`；剖析见
> `harness/profile_port_pregdr_bwd.py`（`harness/profile_port_pregdr_bwd.out.txt`）；数值/计时见
> `harness/check_port_pregdr_bwd_kernel.py`（`harness/check_port_pregdr_bwd_kernel.out.txt`，空闲卡）。
> **未改 Megatron-LM 一行**。

## 1. 先剖析，别猜：dev 回归不在 Triton kernel 里

M13-2 的反向是「4 个 Triton kernel + 一段 torch glue」。把它放进 `torch.profiler`
（`ProfilerActivity.CUDA`，看 `self_device_time_total`）就能看到每个 kernel 各占多少。
以 `float32 / T=1024 B=8` 为例（`profile_port_pregdr_bwd.out.txt`）：

| kernel | 类型 | us/iter | 次数 |
|---|---|---|---|
| `_kda_conv_bwd_kernel` | Triton | 30.9 | 1 |
| `at::native::elementwise_kernel` | **torch glue** | 31.7 | **5** |
| `_kda_silu_bwd_kernel` | Triton | 10.7 | 1 |
| `_kda_l2norm_bwd_kernel` | Triton | 7.8 | 1 |
| `CatArrayBatchedCopy_vectorized` | **torch glue** | 7.8 | 1 |
| `FillFunctor` | **torch glue** | 2.1 | 2 |
| `_kda_sigmoid_bwd_kernel` | Triton | 1.7 | 1 |

**关键读数**：四个 Triton 反向 kernel 合计 ≈ 51 µs，而 torch 的一次性 kernel
（elementwise ×5 + cat + 两次 zero-fill）合计 ≈ 41.6 µs——**约 45% 的反向设备时间花在
「把张量搬来搬去」上，而不是算子本身**。那 5 次 `elementwise_kernel` 正是 M13-2
`_backward_raw` 里的：

1. `dconv[..., :2qk] = dqk_in.reshape(...)`（写 qk 段），
2. `dconv[..., 2qk:] = fd(dv).reshape(...)`（写 v 段），
3. `dqkvfg_t[..., :C] = dx`，
4. `dqkvfg_t[..., C:C+qk] = fd(draw_g)...`，
5. `dqkvfg_t[..., C+qk:] = fd(dgate)...`；

`CatArrayBatchedCopy` 是 `torch.cat([dq,dk])`；两次 `FillFunctor` 是
`torch.zeros(C,W)`/`torch.zeros(C)`（给 conv 反向的原子累加器）。

**为什么 M13-2 会写成这样**：它的反向是「按前向逆序逐段翻译」——L2-norm 反向输出一个
`[B,T,2Hk,KD]` 的 `dqk_in`，要与 `dv`「拼成齐整的 `[B,T,C]`」再喂 SiLU 反向，于是自然
产生了 `cat` + 两次 slice-copy + `dconv` + `da` 四层 fp32 中间量，外加 `fd()`
（`.detach().float().contiguous()`）对每个上游梯度做一次 dtype 归一。这些在
「一段一段验证」时合理，合起来就是 §1 表里的 glue。**这不是算法代价，是数据搬运代价。**

**顺带一个 dtype 观测**：bf16 下还多出 `bfloat16_copy_kernel`×2 与 `direct_copy`×1
（`fd()` 把 bf16 上游梯度升到 fp32）。所以越是用 bf16，这层 glue 越贵——这也是 bf16 的
`dev` 回归（0.69×）比 fp32（0.80×）更重的原因。

## 2. 融合：让 producer **直接写 `da`**

M13-3 去掉整条「拼 `dconv` → 再 SiLU」的中转，让两个 producer 直接把 **`da`（对激活前
conv 输出的梯度）** 写到位（`src/kda_pregdr_bwd.py:167-219`）：

- **qk 通道**（`_kda_l2norm_silu_qk_bwd_kernel:168`）：一个 kernel 同时做 L2-norm 反向
  **和** SiLU 反向。它按行 `r=(b,t,hh)` 索引 `query_key`（`x`）、并在**行内**选择
  `dq`（`hh<Hk`）还是 `dk`（`hh≥Hk`）——于是既不需要 `cat(dq,dk)`，也不需要
  `dqk_in`/`dconv` 中间量；算出 `dxk` 后立刻乘 `silu'(pre)` 写进 `da`：

  ```
  rstd = 1/√(Σx²+ε);  dot = dy·x
  dxk  = rstd·dy − rstd³·x·dot        # L2-norm 反向
  sig  = σ(pre);  da = dxk·(sig·(1 + pre·(1−sig)))   # ∘ SiLU'
  ```
- **v 通道**（`_kda_silu_v_bwd_kernel:203`）：只需 `da = dv·silu'(pre)`。

`da` 现在是**唯一**的 `[B,T,C]` fp32 中间量，且由两个 producer 原地写满。
conv 反向（`_kda_conv_bwd_kernel:79`）读 `da` 不变；区别只在它现在**把 `dx` 直接写进
最终 `dqkvfg_t[..., :C]`**：kernel 多了三个输出步长参数 `OST/OB/OOFF:106`（token/batch
stride + 通道偏移），于是 `dqkvfg_t[..., :C] = dx` 那次 slice-copy 也消失了。

`fd()` 也基本退场：producer 与 conv 反向都在 **load 时 `.to(tl.float32)`**，可以吃原始
（bf16/fp32）上游梯度；`draw_g`/`dgate` 的散射改成 `dqkvfg_t[...].copy_(...)`（copy 时
顺带 cast），省掉「先把 bf16 升 fp32、再拷贝一次」。

## 3. 一个必须记住的坑：autograd 会递 **broadcast（stride 0）梯度**

去掉 `fd()` 后，producer 按**连续**方式线性索引 `dq/dk/dv`。第一次跑整层 `Part D` 立刻
`illegal memory access`——但用 `_mk_grads`（`torch.randn` dense）的剖析脚本却一路正常。
用 hook 打印上游梯度才看清（`harness/_dbg_repro.py`）：

```
grad shape (2, 64, 4, 32) stride (0, 0, 0, 0) contig False
```

`sum(o.float().sum())` 对输出求导时，autograd 给的是**全 1 的广播视图（stride 全 0）**，
不是 dense 张量。M13-2 用 `fd(...).contiguous()` 顺手把它实体化了，所以从未暴露。修法是在
fused 分支入口对三个被线性索引的梯度做 `dq, dk, dv = dq.contiguous(), ...`
（`src/kda_pregdr_bwd.py:293`）——**dense 时是零成本 no-op，只在广播时才真正 copy**。

**判据**：任何「直接线性索引上游梯度」的 kernel 入口，都要假设梯度可能是**广播/strided**
视图；`contiguous()` 是廉价保险，不要因为「平时是 dense」而省掉。

## 4. 数值锚定

### 4.1 Part F：融合前后在同一批中间量上逐项比（`harness:337`）

同一份 `saves`、同一份上游梯度，分别走 `fused=True`/`fused=False` 两条反向，比四个输出：

| T×B | fp32 dqkvfg | fp32 dbeta | fp32 dweight | fp32 dbias | bf16 dqkvfg |
|---|---|---|---|---|---|
| 64×2 | 4.8e-7 | 0 | 7.2e-7 | 3.8e-6 | 3.9e-3 |
| 250×1 | 4.8e-7 | 0 | 9.5e-7 | 5.7e-6 | 3.9e-3 |
| 128×2 | 7.2e-7 | 0 | 1.2e-6 | 3.8e-6 | 9.8e-4 |

fp32 全部落 **≤1e-6**（`dbias` 略大，是跨 CTA 原子加**累积序**的 fp32 底噪，`docs/16` §2.2
已记）；bf16 的 ~1e-3–4e-3 是**出口 cast 回 `qkvfg.dtype` 的一次量化**（`dqkvfg` 的 bf16 ulp）。
即：**融合改变的是数据的搬运路径，不是代数**。

### 4.2 Part E：`conv_bias=True` 端到端（`harness:294`）

`docs/16` 记下一个边界：真实 Megatron KDA 层建 conv1d 时 `bias=None`，所以整层对拍里
`dbias` 恒为 `nan`——**卷积偏置的反向从未在整层被驱动非零**。Part E 用 stub 模块
（`_Stub`，`bias=True` + 非零初始化）把这条路径跑全，对 `torch.autograd` 穿可微参考前向：

| T×B | fp32 dqkvfg | fp32 dbias | **bias.grad（native / ref）** | bf16 dqkvfg | bf16 dbias |
|---|---|---|---|---|---|
| 64×2 | 4.8e-7 | 2.4e-6 | **17.46 / 17.46** | 1.6e-2 | 1.3e-1 |
| 250×1 | 3.6e-7 | 3.8e-6 | **33.03 / 33.03** | 1.6e-2 | 1.3e-1 |
| 128×2 | 4.8e-7 | 3.8e-6 | **25.97 / 25.97** | 1.6e-2 | 1.3e-1 |

fp32 `dbias` 落 **2.4–3.8e-6**、`bias.grad` 打印值两侧**逐位相同**（bf16 的 1e-1 是前向
conv 后端量化的放大，与 Part B 同源）。即：**`conv_bias` 反向在非零偏置下也被验证**，
`docs/16` 的 `dbias=nan` 边界关闭。

### 4.3 回归没退：Part A/B/C 与 M13-2 一致

默认走 fused 后，`Part A`（孤立反向 vs 解析参考）fp32 `dweight/dx/dbeta` 仍 **~1e-6**、
bf16 `dweight` 仍 **1e-6**；`Part C`（整层 drop-in）`params compared=7 skipped=0`、
`conv1d.weight.grad` 两侧同值、`param.grad rel` **1.95e-5(fp32) / 3.9e-3(bf16)**——
与 `docs/16` §5 同阶。

## 5. 收益：dev 回归基本关闭

### 5.1 隔离反向的设备时间（逐 kernel 剖析，`profile_port_pregdr_bwd.out.txt`）

| T×B | fp32 unfused | fp32 fused | × | bf16 unfused | bf16 fused | × |
|---|---|---|---|---|---|---|
| 64×2 | 24.6 | 21.0 | **0.85** | 31.9 | 24.6 | **0.77** |
| 256×2 | 26.1 | 21.7 | **0.83** | 33.2 | 25.1 | **0.76** |
| 1024×2 | 37.7 | 24.5 | **0.65** | 45.7 | 29.1 | **0.64** |
| 1024×8 | 92.6 | 66.9 | **0.72** | 106.2 | 78.0 | **0.73** |

反向设备时间砍到 **0.64–0.85×**。剩下的大头是**不可避免的**：`conv_bwd`（本核最大单项）
+ 2 次 `copy_`（把 `draw_g`/`dgate` 散射回输出 `qkvfg` 的后两段）+ 2 次 zero-fill
（conv 反向的原子累加器）+ 出口 cast。

### 5.2 整链（前向+反向）Part D（`harness:256`）

| T×B | fp32 wall（×） | **fp32 dev（×）** | bf16 wall（×） | **bf16 dev（×）** |
|---|---|---|---|---|
| 64×2 | 1357/753（1.80） | 69.1/79.2（**0.87**） | 1690/1024（1.65） | 87.3/108.3（**0.81**） |
| 256×2 | 1549/990（1.57） | 98.6/105.7（**0.93**） | 1671/998（1.67） | 116.4/134.8（**0.86**） |
| 1024×2 | 1553/926（1.68） | 114.9/112.9（**1.02**） | 1680/1015（1.66） | 127.9/140.2（**0.91**） |
| 1024×8 | 1622/1163（1.40） | 256.8/224.6（**1.14**） | 1698/1007（1.69） | 227.7/236.1（**0.96**） |

对比 `docs/16` §5.4 的 M13-2 基线（fp32 dev 0.80/0.87/0.91/0.98、bf16 0.69/0.76/0.76/0.79）：
**小 shape 的 dev 缺口从 0.69–0.80× 抬到 0.81–0.87×，大 shape 已翻正**
（fp32 1024×2 = 1.02×、1024×8 = 1.14×）。`wall` 仍稳定赢 **1.4–1.8×**（host launch 税）。

## 6. 结论与边界

- **诊断**：M13-2 的 dev 回归**不在 Triton kernel**，而是「拼 `dconv` / 散射 / `fd()` cast」
  的 torch glue（≈45% 反向设备时间）。判据：先把每个 kernel 的 `self_device_time` 拆开，
  再决定优化对象——不要凭「多 kernel」就断言 kernel 慢。
- **优化**：让 producer 直接写 `da`（qk 段一个 kernel 同时做 L2-norm ∘ SiLU 反向，v 段一个），
  conv 反向直写 `dqkvfg_t[..., :C]`——反向设备时间 **0.64–0.85×**、整链 dev 缺口基本关闭
  （大 shape 翻正）。
- **数值**：融合前后 fp32 ≤1e-6、逐项同源（Part F）；Part A/B/C 与 M13-2 同阶。
- **补测**：`conv_bias=True` 非零端到端（Part E），`dbias` 落 1e-6 量级、`bias.grad` 两侧同值
  ——关闭 `docs/16` 的 `dbias=nan` 边界。
- **坑**：autograd 会递 broadcast（stride 0）梯度；线性索引上游梯度的 kernel 需
  `.contiguous()` 兜底（dense 时免费）。
- **边界**：`BT=64`/fp32-internal；反向仍是 **4 kernel**（producer qk / producer v / conv /
  sigmoid）+ 前向存 `pre` 的第二趟 conv；**未做 ncu**（本轮性能口径为 CUPTI self-dev + 自有
  `wall`/`dev`，逐 kernel 归因已足够定位）；仍未接 `kda.py:399` 的
  `gdn_pre_gated_delta_rule_fusion` guard（与 M12/M13 同一边界：不修改 Megatron 树）。
