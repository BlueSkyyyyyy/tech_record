---
title: "LLM 模型结构变迁（七）：条件记忆——Engram、MTP 与 DSpark"
date: 2026-11-02
draft: false
weight: 8
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "Engram", "记忆", "MTP", "投机解码", "稀疏"]
categories: ["LLM算法"]
---

前面六篇的稀疏化都发生在**计算**上：少看位置（稀疏注意力）、少激活专家（MoE）、少存 KV（线性注意力）。这一篇讲一条 2026 年才成型的新轴——**稀疏的「查表」**。

DeepSeek-V4.1 有 748B 总参数，其中 **196.6B 是 Engram**：每个 token 只读其中 48 行、约 12.1 KiB。这 26% 的参数**永远不参与矩阵乘**，只被索引。

这是本专题里最反直觉的一次搬家：MoE 扩容按 FLOPs 计费，Engram 扩容按**字节常驻**计费。

## 0. 先记住一个画面

语言模型里混着两类完全不同的任务：

- **组合推理**：把已知事实拼起来，推出新结论。要深、要动态、要算。
- **知识检索**：看到一个熟悉的名字，想起它是谁。局部、静态、高度套路化。

第二类正是 N-gram 模型擅长的。但标准 Transformer 没有「查表」这个原语，于是被迫**用计算模拟检索**：一个常见的多 token 实体，要消耗好几层 attention + FFN 才能拼出来。

Engram 论文（arXiv:2601.07372）给了个很直观的证据：要让最后一层表示说出 "Diana, Princess of Wales"，第 1-2 层只知道 "Wales"，第 3 层是 "Country in Europe"，第 4-5 层才到 "Princess of Wales（unspecific）"，第 6 层才对上人。

**前 6 层里有 4 层在做一张静态查找表一次 O(1) 就能做完的事。**

> **类比的边界**：查表不是「外挂知识库」（RAG）。RAG 在输入侧拼文本，模型仍要用算力读；Engram 在**参数侧**做条件查找，查到的向量直接进残差流。它也不是「更大的 embedding」——embedding 按 token id 查，Engram 按 **n-gram 哈希**查，而且每个 (阶, 头) 有独立的表。

## 1. Engram 的四个步骤

### 1.1 压缩词表

先做一个满射 $P: V \to V'$，把归一化后文本等价的 token 折叠成同一个 id：NFKC → NFD → StripAccents → Lowercase → 空白折叠 → Strip。实测 128k 词表压缩 **23%**（129280 → 99092）。

代码里有个细节：中间用一个私用区字符 `\ue000` 当 sentinel，防止「恰好是一个空格」的 token 被 Strip 成空串后跟无关 token 合并（`annotated/engram.py:72-84`）。

### 1.2 多头哈希：n-gram → 桶

对每个阶 $n \in \{2,3,4\}$ 和每个头 $k \in 1..8$：

$$
z_{t,n,k} = \varphi_{n,k}(x'_{t-n+1}, \ldots, x'_t), \qquad e_{t,n,k} = E_{n,k}[z_{t,n,k}]
$$

$$
e_t = \mathrm{concat}_{n=2..4}\ \mathrm{concat}_{k=1..8}\ e_{t,n,k}
$$

$\varphi$ 是**滚动 XOR 乘法哈希**——论文正文没写，只有代码能看出来：

```python
products = tokens * multipliers          # [B,L,n_layers,max_ngram]
rolling  = products[..., 0]
for i in 1..max_ngram-1:
    rolling = rolling ^ products[..., i]         # 累积到 (i+1)-gram
    hashes.append(rolling.unsqueeze(-1) % primes[:, i-1])
return cat(hashes) + offsets
```

三个实现细节：

- **前缀一致性**：2-gram 的哈希是 3-gram 哈希的中间量（`rolling` 一路 XOR 下去），一次扫描顺带算出所有阶，代价是 $O(N)$ 次整数运算；
- **每个 (阶, 头) 一段独立素数模数**：从 15,999,999 往下取不重复的素数，保证各桶区间不重叠，最后用 offsets 拼成一张大表；
- **乘子**：每 (层, 回看距离) 一个随机乘子，强制奇数并限幅，防止 `token_id * multiplier` 溢出 int64。

### 1.3 上下文感知门控

检索到的 $e_t$ 与上下文无关，可能是噪声（哈希碰撞、多义）。所以用当前 hidden state 当 Query、$e_t$ 过 $W_K/W_V$ 当 Key/Value：

$$
k_t = W_K e_t,\quad v_t = W_V e_t,\quad
\alpha_t = \sigma\Big(\frac{\mathrm{RMSNorm}(h_t)^\top \mathrm{RMSNorm}(k_t)}{\sqrt{d}}\Big),\quad
\tilde v_t = \alpha_t v_t
$$

代码和论文式 4 有个数值差异：sigmoid 之前先做一次 **signed-sqrt**（`model.py:438-441`），注释写 "matching the training kernel"——训练 kernel 用它压动态范围，推理必须逐位对齐，否则复现对不上。

另外门控是**按 (token, hc copy) 各自归一化**的，不是跨 copy 联合归一化。

### 1.4 注入：加，且门控

$$
H^{(\ell)} \leftarrow H^{(\ell)} + Y
$$

而且是在该层 **attention 之前**（`model.py:1427-1431`）。加法给残差通路，门控 $\alpha_t \in (0,1)$ 决定「这份记忆这次要不要信」。

多分支（mHC，$M=4$）下：表 + $W_V$ 全 branch 共享，$W_K$ 每个 branch 一份。这样 4 个 $W_K$ 和 1 个 $W_V$ 能融成一个稠密 FP8 matmul——代码里是 `wkv` 输出 `dim * (hc_mult + 1)`，再 split。

V4.1 还砍掉了论文 Eq.5 的短因果卷积（kernel 4、dilation = 最大 n-gram 阶）：性能收益不抵推理栈复杂度。

## 2. 账本：198 GB 换 12 KiB

一层 Engram 的表有多大？

$$
\underbrace{3}_{\text{阶}} \times \underbrace{8}_{\text{头}} \times 16.0\text{M} \text{ 桶} = 384\text{M 行/层}
$$

每行两个 256 维向量（$W_K$ 和 $W_V$ 的查表值），所以：

$$
384{,}006{,}168 \times 256 + 384{,}016{,}682 \times 256 = 196{,}613{,}849{,}600 \approx 196.6\text{B}
$$

FP8 存权重 + block=128 的 scale，常驻约 **198.1 GB**。

而每个 token 只查 $3 \times 8 \times 2 = 48$ 行（2 层 × 24 张表），每行 256 字节 + 2 字节 scale：

$$
48 \times 258 = 12{,}384\text{ B} \approx 12.1\text{ KiB}
$$

三个对比，把「记忆」的定位说清楚：

| 对比对象 | 数值 | 倍数 |
|---|---|---|
| 全部 HBM KV（V4.1，890 B/token） | 12.4 KiB | **13.9×** |
| decode 权重流（16B 激活 × 2B） | 32 GB | 0.00004× |
| MoE 稀疏度（8B/748B） | 1.07% | — |
| Engram 稀疏度（48 行 / 384M 行） | $6.25 \times 10^{-8}$ | **比 MoE 稀 1.7×10⁵ 倍** |

所以：**「记忆」在带宽上一点不便宜**（查表流量是 KV 的 14 倍），它便宜在**容量**——198 GB 的表可以放在 HBM 里（V4.1 是 8 卡 × 大显存），也可以放 CPU/CXL 内存，靠后台预取。

真正的成本是三个工程问题：

1. **198 GB 放哪**。单卡放不下，必须跨卡切表 + all-to-all 或 NVLink 访问；
2. **预取窗口**。第 1 层只留一个 block 的预取窗口，batch 越大越盖不住延迟；
3. **哈希碰撞**。表越大碰撞越少，但 384M 行仍然远小于 n-gram 空间，碰撞是常态——门控就是用来压碰撞噪声的。

## 3. U 型 scaling law：MoE 和记忆怎么分配

Engram 论文最重要的不是结构，是给出了一条**分配律**：给定总算力，多少给 MoE（条件计算）、多少给 Engram（条件记忆）？

定义记忆占比：

$$
\rho = \frac{P_{\text{Engram}}}{P_{\text{MoE}} + P_{\text{Engram}}}
$$

论文在 10B 档扫出一条 **U 型曲线**：最优 $\rho$ 在 **75%-80%**；$\rho \approx 40\%$ 时才与纯 MoE 打平；两端都变差。

数字：10B 档 $\rho$ 从 0 到最优，val loss 从 1.7248 降到 1.7109（$\Delta = 0.0139$）。

V4.1 的实测 $\rho = 543.58/(543.58+196.61) = 0.734$；Engram-27B 是 $5.79/22.9 = 0.749$。**都压在 75%-80% 的边缘**——说明这个比例不是拍脑袋，是真按 scaling law 配的。

> 这条 law 的意义：它把「记忆」从「一个 trick」变成了「一个可以和 MoE 一起做预算规划的维度」。这是 2026 年结构设计最重要的方法论变化。

## 4. 多 token：另一条轴

MTP / DSpark 扩的不是容量，也不是宽度，而是**每个目标模型前向能兑现几个 token**。

**MTP（V3）**：训练时加一个预测 $t+2$ 的辅助头（$D=1$），densify 训练信号；推理时改造成投机解码，第二个 token 接受率 **85%-90%**，TPS **1.8×**，而且推理时可以整段丢掉。

**DSpark（V4.1，arXiv:2607.05147）**：半自回归草稿 + 置信度调度校验。配置是 3 个 block、窗口 128、block 5、markov_rank 256；损失是 $0.1 \mathcal{L}_{\text{ce}} + 0.9 \mathcal{L}_{\text{tv}} + 1.0 \mathcal{L}_{\text{conf}}$。

两个数字值得记：

- 平均接受长度 $\tau$：GSM8K 上 **6.11**，Eagle3 5.14、DFlash 5.40——并行草稿器在很多格上**打过**了全自回归的草稿器，推翻了「自回归草稿质量必然更高」的常识；
- 生产环境 per-user 生成速度：Flash **+60%~85%**、Pro **+57%~78%**（等吞吐口径）。

DSpark 的增益来源不是「更长的块」，而是两个细节：**Markov head 修 cross-mode collision** + **置信度调度砍掉尾部校验**。

## 5. 三个坑

**坑一：`assert vocab_size == 99092` 只校验大小，不校验映射。** 换 tokenizer 后 assert 照样通过，196B 参数**静默失效**——哈希桶全错位，模型输出退化但不报错。

**坑二：门控公式和论文不一致。** 照论文 Eq.4 复现，数值对不上；必须加上 signed-sqrt。

**坑三：Engram 不是「免费的知识」。** 12.1 KiB/token 的查表流量是 KV 的 14 倍；在长上下文、大 batch 的场景下，查表带宽会成为新的瓶颈。**MoE 把瓶颈从算力搬到通信，Engram 把它从通信搬到「随机读」**——这是两种完全不同的优化对象。

## 6. 小结

| 轴 | 扩容方式 | 计费科目 | 代表 |
|---|---|---|---|
| MoE | 稀疏激活矩阵 | FLOPs + 通信 | 896 专家 top-16 |
| Engram | 稀疏查表 | 字节常驻 + 随机读带宽 | 196B / 12 KiB per token |
| MTP/DSpark | 多 token 兑现 | 草稿器算力 | $\tau$ 6.11 |

三者的共同点：**都在用「稀疏」换「容量」**，但稀疏的对象不同——激活、寻址、时间。2027 年如果还有新的稀疏轴，大概率在这三个之外。

下一篇是总结：五个收敛、三个分叉，以及我对 2027 的三个赌注。

---

**系列导航**：

1. [导读：2026 年的结构地图]({{< relref "llm-arch-00-index" >}})
2. [先把账本立起来]({{< relref "llm-arch-01-budget" >}})
3. [注意力四代]({{< relref "llm-arch-02-attention" >}})
4. [线性注意力与 3:1 混合]({{< relref "llm-arch-03-linear" >}})
5. [MoE 十年]({{< relref "llm-arch-04-moe" >}})
6. [残差流的加宽：HC 与 mHC]({{< relref "llm-arch-05-residual" >}})
7. [位置的退场]({{< relref "llm-arch-06-position" >}})
8. 本篇：条件记忆
9. [五个收敛、三个分叉与三个赌注]({{< relref "llm-arch-08-trends" >}})
10. [残差连接与 AttnRes]({{< relref "llm-arch-09-attnres" >}})
