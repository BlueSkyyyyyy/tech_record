# 多卡 EP MoE 接上真实 SM90 FP8 grouped kernel：EP 通信 × FP8 量化的端到端误差与预算（8×H100 实测）

> 本文对应路线图 **P151-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 161 篇。
>
> 专题到第 160 篇，Mega MoE 的**每一段**都被拆过：前向 dispatch→L1→SwiGLU→L2→combine
> （第 2/3/9/10/11 篇）、反向的四个 grouped GEMM（第 12/13/14 篇）、前向/反向的 H100 数值复刻
> （第 99/111 篇）、多卡 EP 端到端前向/反向的**纯 torch** 复刻（第 141 篇）、反向通信量
> （第 104 篇）。但第 141 篇结尾（`docs/141-ep-e2e-moe.md:209-213`）自己点出了缺口：
>
> > **下一轮可做（opt-in）**：把本篇的多 rank EP 图换成**真实 DeepGEMM SM90 grouped FP8 kernel**
> > 接在 dispatch 之后，量「EP + FP8 量化」叠加后的端到端误差。
>
> 本篇补上：**EP 的通信（NCCL all-to-all）+ 真实 SM90 FP8 grouped kernel 的量化**，第一次叠在
> **同一条端到端 MoE 图**里，前向和反向都做，并给出 DeepSeek 量级的通信 + FP8 kernel 预算。
>
> **可跑边界**：EP 通信用 `torch.distributed` 的 `all_to_all_single`（架构无关），计算腿是真实
> DeepGEMM SM90 FP8 kernel（`m_grouped_*`/`k_grouped_*`，sm90 有实现），**H100 真机可跑**；
> Mega MoE 融合 kernel 本体（UMMA/tcgen05/TMEM/NVLink 直写）是 **SM100 专属，仅分析**。
> 目标卡代号敏感，全文用「目标卡」指代。
> 参考坐标：`~/github/DeepGEMM@057ca59`（Public release 26/09/30）、`~/github/DeepEP@93eb6eb`（v2.5.0）。
> 实测见 `code/mega-moe/harness/exp_ep_fp8_e2e_moe.{sh,out.txt}`、代码 `harness/exp_ep_fp8_e2e_moe.py`。

---

## 1. 缺口的精确形状

| 篇目 | EP 通信 | 真实 FP8 kernel | 前向 | 反向 |
|---|---|---|---|---|
| 第 99 篇（前向复刻） | ✗（单卡 gather/scatter） | ✓ | ✓ | ✗ |
| 第 14 篇（反向 kernel） | ✗（单卡） | ✓ | — | ✓ |
| 第 141 篇（EP e2e，纯 torch） | ✓ | ✗（torch matmul） | ✓ | ✓ |
| **本篇** | **✓** | **✓** | **✓** | **✓** |

第 141 篇证明了「EP 把 token 散到别的 rank、算完再收回来，前向输出与四组梯度与单卡稠密 MoE
严格一致」——但它用的是高精度 torch matmul，误差分解里**不含** FP8 量化。第 99 篇把 FP8 量化
放进前向，但把通信折算成单卡 gather/scatter。两者之间那道缝就是本篇。

## 2. 实验图：EP dispatch/combine + 真实 FP8 grouped kernel

一张图（8 rank，每 rank 持 `per_rank = E/8` 个本地专家）：

```
x[T_local,H] ──(per-token 量化)──► FP8 L1: m_grouped  ──► gate/up (bf16)
   │                                A=[收到的 token] (M 轴分组)
   │                                B=W1_local (N=2I,K=H, per-block FP8)
   ▼
 dispatch(NCCL a2a) ──► 按本地专家分组 ──► SwiGLU(逐元素) ──► act
                                                │
                                                ▼
                                      FP8 L2: m_grouped  ──► ye (bf16)
                                      A=act (M 轴分组), B=W2_local (N=H,K=I)
                                                │
                    combine(NCCL a2a, 按 top-k 权重 w 求和) ──► out[T_local,H]
```

反向是它的**转置**（第 4/95/141 篇）：

| 段 | 数学 | 真实 kernel |
|---|---|---|
| combine 的伴随 | `dY = w · (dispatch of dOut)` | NCCL a2a（交换 split）|
| linear2 wgrad | `dW2[e] = act_e^T dY_e`（K=token 分组） | `k_grouped_fp8_gemm_nt_contiguous` |
| linear2 dgrad | `dact_e = dY_e @ W2_e^T`（M=token 分组） | `m_grouped_fp8_gemm_nt_contiguous` |
| SwiGLU 反向 | `dgate=dact·silu'(gate)·up`、`dup=dact·silu(gate)` | 逐元素 torch |
| linear1 wgrad | `dW1[e] = x_e^T dG_e` | `k_grouped_*` |
| linear1 dgrad | `dx_e = dG_e @ W1_e^T` | `m_grouped_*` |
| dispatch 的伴随 | `dx = combine of dx_e` | NCCL a2a |

**源码契约（逐条核对）**：
- 前向 / dgrad 是 **M 轴分组**：`m_grouped_fp8_fp4_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:181`），
  **输出 D 必 bf16**（`:209`），`grouped_layout` 是 int（`:210`）；
- wgrad 是 **K 轴分组**：`k_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:373`），
  在 SM90 上**必须** `recipe==(1,1,128)`（`:383`）且 **C 必填**（`:403`，即强制 fp32 累积）；
- SM90 只有两种互斥输出形态：1D1D **必 fp32 且必须累积**（`sm90_fp8_gemm_1d1d.hpp:84`），
  1D2D **必 `not c` 且 D 必 bf16**（`sm90_fp8_gemm_1d2d.hpp:81`）。这正是第 15 篇「wgrad 慢」
  根因（fp32 强制累积）的复现入口。

**为什么向后做 forward 的权重要转置**：前向 `D = X @ W^T`，`W1_local[e]` 是 `[H,2I]`，喂
`m_grouped` 的 `B` 必须是 `[N,K]=[2I,H]`，所以 `W1_local.transpose(1,2)`；反向 dgrad 的 `B`
恰相反（`W2_local` 本身就是 `[I,H]=[N,K]`）。前向后向互为转置，操作数的主序也随之对偶。

## 3. 前向误差：三层分解

用单卡稠密 fp64 参考（同一 routing）做三层对拍。第 2 层「fp8 模拟」= 把 per-token 量化后的
x（gran 128）、per-block 量化后的权重在 fp64 里重算、并把 L1 输出 round 成 bf16——它**只保留
FP8 量化的效果**，不含 kernel/布局/通信。

`world=8, T_local=128, H=512, I=1024, E=64, topk=4`（真机，见 `.out.txt`）：

| 比较 | rel | cos | 隔离的是 |
|---|---|---|---|
| EP FP8 vs fp64 | 6.396e-02 | 0.997948 | **总计** |
| EP FP8 vs fp8 模拟 | 2.273e-02 | 0.999740 | **kernel + 布局 + 通信** |
| fp8 模拟 vs fp64 | 5.946e-02 | — | **纯 FP8 量化噪声** |

**读法**：总误差（6.4e-2）**几乎全部来自 FP8 量化**（5.9e-2），与第 99 篇的 6.3e-2 同量级；
而「kernel + 布局 + 8 rank 通信」这一层只剩 2.3e-2，且它主要是 **L2 输出 bf16 舍入**（D 必 bf16，
`gemm.hpp:209`）——**通信本身是精确搬运、零误差**（第 141 篇同理：dispatch/combine 只是复制+置换，
其雅可比是转置）。换句话说：**EP 的通信不引入数值误差，FP8 才是误差地板**。
`world=8, T_local=64, H=256, I=512, E=32, topk=2` 的第二个形状给出同样的数
（6.303e-2 / 2.299e-2 / 5.950e-2）。

## 4. 反向：四路梯度 vs fp64

`dOut` 取一个固定随机方向，反向往回传，梯度沿 rank 聚合后与单卡稠密 fp64 autograd 对拍：

| 梯度 | rel | cos |
|---|---|---|
| dW1 | 6.440e-02 | 0.997964 |
| dW2 | 6.264e-02 | 0.998080 |
| dX | 6.511e-02 | 0.997881 |

**和前向同量级**（≈6.4e-2、cos≈0.998）——因为反向的 dgrad/wgrad 操作数同样被 FP8 量化，误差在
同一条 FP8 地板上。`dW1/dW2` 对「fp8 模拟操作数参考」的误差是 4.8e-2 / 3.2e-2（低于对 fp64 的
6.4e-2，因为 fp8 模拟参考消掉了权重/激活的输入量化，剩下的主要是**梯度张量自身的独立量化**）。
这与第 14 篇单卡 FP8 反向的结论（`vs deq` ~2e-4、`vs autograd` 2.5e-2~5.1e-2）一致，
只是此处叠加了多卡 EP。

## 5. 每个 grouped GEMM 的隔离：「vs deq」

第 14 篇的核心手法是把每个 kernel 与「喂进它的 FP8 操作数**反量化后**在 fp32 重算的 GEMM」比，
隔离出 kernel + 布局误差。本篇把同款检查下沉到 8 个 rank 各自的本地分组上，取**全局最大**：

| 量 | 全局最大 rel |
|---|---|
| dW1（k_grouped） | 3.018e-04 |
| dW2（k_grouped） | 3.001e-04 |
| dact（m_grouped） | 3.210e-04 |
| dx（m_grouped） | 3.497e-04 |

**每个 grouped kernel 与它的「反量化语义参考」都在 3e-4 内**（就是 FP8 输入的舍入级），
说明 **EP 的行空间、分组、布局桥接全部正确**，前面那 6.4e-2 是量化误差而不是 bug。
第二个形状给出 2.9e-4~3.7e-4，一致。

## 6. DeepSeek 量级的预算：通信 vs FP8 kernel

`T_local=2048, topk=2, H=7168, I=2048, E=64`（每 rank 8 专家）。每 rank 收到 **4240 行**
（平均 530/本地专家，按 `mk_alignment=128` padding 后 `m=4736`）。

**只计 grouped GEMM kernel 本体**（操作数预量化，排除 Python 侧 torch 量化开销）：

| 段 | 用时 | TFLOPS |
|---|---|---|
| fwd L1（`m_grouped`，m=4736,N=4096,K=7168） | 707.8 us | 393 |
| fwd L2（`m_grouped`，N=7168,K=2048） | 278.2 us | 500 |
| dgrad L2（`m_grouped`） | 279.2 us | 498 |
| dgrad L1（`m_grouped`） | 699.3 us | 398 |
| **wgrad L2（`k_grouped`，fp32 强制累积）** | 1615.0 us | **86** |
| **wgrad L1（`k_grouped`）** | 3234.1 us | **86** |

- **前向 kernel 合计 985.9 us**，**反向 kernel 合计 5827.6 us**（**反向 ≈ 5.9× 前向**）。
- 反向的钱在 **wgrad 两个 kernel**（占反向 kernel 的 83%），正是第 15 篇「fp32 强制累积 ⇒
  访存受限」根因的**多卡复现**：wgrad 86 TFLOPS vs dgrad/fwd 400-500 TFLOPS。

**通信**（本实验按 fp32 搬运，是**保守上界**；真机 dispatch 为 FP8/1B、combine 为 BF16/2B）：

| | 字节/rank | 用时 |
|---|---|---|
| dispatch a2a（ragged split） | 121.6 MB | 4945.7 us |
| combine a2a（ragged split） | 121.6 MB | 4946.3 us |

- **EP kernel + 通信**：前向 985.9 + 9892.0 = **10877.9 us**；反向 5827.6 + 9892.0 = **15719.6 us**。
- 通信是**受争用下界**：本机 8 卡共享（1 张卡被另一自驱任务占满），ragged a2a 只有
  **24.6 GB/s**；同机**平衡负载** a2a 锚（`world×T×H` 连续张量、无 split）为
  **117.1 / 113.5 / 122.2 GB/s**（T=512/2048/8192）——与第 104 篇「ragged split 有 2-3× 惩罚」
  同向但更重（此处 ~4.7×）。这从多卡侧再次支持 Mega MoE 的设计动机：**通信必须在卡内精细地
  分 SM、分时、与计算重叠**（第 63/142/143/144 篇），而不是「发两条流」。
- 真机字节口径：`nrecv·H·1B`（dispatch FP8）= 30.4 MB/rank、`nrecv·H·2B`（combine BF16）
  = 60.8 MB/rank —— 比本实验的 fp32 121.6 MB 各小 4×/2×。

## 7. 与既有篇章的关系与边界

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 141 篇（EP e2e，纯 torch） | 多 rank EP 图 + 四组梯度，**无 FP8** | 计算腿换成**真实 FP8 kernel**，量出 FP8 地板 |
| 第 14 篇（反向 kernel 接线） | 真实 FP8 grouped kernel 接进反向，**单卡** | 同一套 kernel 放回 **8 rank EP 行空间** |
| 第 99 篇（前向 FP8 复刻） | FP8 前向数值，**通信折算成单卡** | 前向接**真实 NCCL a2a** |
| 第 104 篇（反向通信量） | 反向 ≈ 1.5× 前向、ragged 2-3× 惩罚 | 把 EP 通信与 FP8 kernel 放进**同一条图**量预算 |
| 第 15/135~151 篇 | wgrad fp32 强制累积的根因 | 在多卡 EP 图里复现「wgrad 86 vs dgrad 500 TFLOPS」|

**能跑/能核**（`harness/exp_ep_fp8_e2e_moe.{sh,py}`，全 H100 sm90 真机）：EP 前向三层误差分解、
四路反向梯度、6 个 grouped kernel 的 `vs deq`、DeepSeek 量级 kernel + NCCL a2a 预算。

**仅分析**：Mega MoE 融合 kernel 本体（`sm100_fp8_fp4_mega_moe.cuh` 的 UMMA/tcgen05/UTCCP/TMEM、
dispatch/combine 的 NVLink 直写与 readiness 水位线、warp 级 mbarrier 重叠）——**SM100 专属**；
本篇用真实 SM90 grouped kernel + NCCL a2a 复刻其**语义与量级**，**不**复刻其 SM 内重叠与 NVLink 直写。

**边界/注记**：① 本实验用 per-token（gran 128）激活量化 + per-block（128×128）权重量化，与
SM90 默认 recipe `(1,128,128)` 一致，不是目标卡 SM100 的 packed UE8M0/UTCCP 粒度；② 通信按
fp32 搬运是保守上界，真机是 FP8/BF16；③ 8 卡共享节点，带宽是**受争用下界**（已给平衡锚对照）。

## 8. 一句话结论

把第 141 篇的多卡 EP 图接上真实 SM90 FP8 grouped kernel 后：**EP 通信不引入数值误差，FP8 量化
才是误差地板**——前向/反向总误差都 ≈6.4e-2、cos≈0.998，其中「kernel+布局+通信」层仅 2.3e-2
（=L2 bf16 输出舍入），每个 grouped kernel 与反量化语义参考都在 3e-4 内。预算上**通信与计算
同量级甚至更贵**（FP8 kernel 前向 986 us / 反向 5828 us 对 ragged a2a 9892 us），且反向的钱在
**wgrad 的 fp32 强制累积**（86 vs 500 TFLOPS）——这正是 Mega MoE 把通信与计算塞进单 kernel、
warp 级重叠的根本原因。

## 参考

- 本专题：第 141 篇（EP e2e 纯 torch）、第 14 篇（单卡 FP8 反向）、第 99 篇（前向 FP8 复刻）、
  第 104 篇（反向通信量）、第 15 篇（wgrad 强制累积）、第 142/143/144 篇（前向预算与重叠）。
- 源码：`csrc/apis/gemm.hpp:181/209/210/373/383/403`；`sm90_fp8_gemm_1d1d.hpp:84`；
  `sm90_fp8_gemm_1d2d.hpp:81`；DeepEP `deep_ep/utils/refs.py:10-123/177-243`、`README.md:204-220`。
- 上游仓库：https://github.com/deepseek-ai/DeepGEMM 、 https://github.com/deepseek-ai/DeepEP 。
- 实验：`harness/exp_ep_fp8_e2e_moe.{sh,py,out.txt}`（8×H100，真机数字）。
