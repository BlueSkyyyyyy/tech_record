# LLM 模型结构变迁 · 配套代码

《LLM 模型结构变迁》专题的预算计算器。给一个结构配置，算出三个数字：

1. **参数量**（总参 / 激活参）；
2. **每 token FLOPs**（prefill 与 decode 分开）；
3. **每 token KV 字节数**（容量与每步实际读取量分开）。

纯标准库，CPU 可跑，不需要装 torch。

```bash
python3 budget.py                 # 默认 128K 上下文，打印总表
python3 budget.py --ctx 1048576   # 改上下文长度（1M）
python3 budget.py --test          # 与 9 个已发布模型对拍
python3 budget.py --detail        # 逐模型拆解（embed / attn / ffn / extra）
python3 budget.py --json          # 机器可读输出
```

## 为什么要这个脚本

结构论文里的参数量、FLOPs、KV 数字经常口径不一：有的算 embedding、有的不算；有的 KV 含索引器、有的不含；有的按 BF16、有的按 FP8/FP4。**把口径写进代码，才能横向比较。**

脚本的价值不在「算得准」，在**可证伪**：每个模型的复算值都和论文/报告公布值对拍，误差超过阈值就报 FAIL。

## 对拍结果

```
[PASS] Llama-3.1-405B      复算   405.8B  vs 公布   405.0B  ( 0.2%)
[PASS] Qwen3-32B           复算    32.8B  vs 公布    32.8B  ( 0.1%)
[PASS] Qwen3-30B-A3B       复算    30.5B  vs 公布    30.5B  ( 0.1%)
[PASS] DeepSeek-V3         复算   670.9B  vs 公布   671.0B  ( 0.0%)
[PASS] DeepSeek-V4-Flash   复算   283.8B  vs 公布   284.0B  ( 0.1%)
[PASS] DeepSeek-V4.1-Flash 复算   551.4B  vs 公布   552.0B  ( 0.1%)
[PASS] Kimi-K2.6           复算     1.03T  vs 公布     1.04T  ( 1.3%)
[PASS] Kimi-K3             复算     2.77T  vs 公布     2.78T  ( 0.3%)
[PASS] gpt-oss-120b        复算   116.8B  vs 公布   117.0B  ( 0.2%)
[PASS] 结构断言全部通过（V4.1 KV 复算 890 B/token = 报告 890 B/token）
```

误差来自没算的 bias、norm、router 与对齐。

## 三个公式

**参数量**（MLA + MoE 为例）：

$$
P_{\text{total}} = Vd + L_{\text{dense}} P_{\text{ffn}} + L_{\text{moe}}\big(n_r P_{\text{expert}} + n_s P_{\text{shared}}\big) + L P_{\text{attn}} + P_{\text{extra}}
$$

MoE 的激活参数只算 top-k 个专家；`P_extra`（如 Engram）计入总参但不计入激活。

**FLOPs**（每 token）：

$$
\text{FLOPs} = 2 P_{\text{active}} + L \cdot 2 n_h (d_{qk} + d_v) \cdot n_{\text{entries}} + \text{indexer}
$$

- 稠密注意力：$n_{\text{entries}} = N$；
- 稀疏注意力：$n_{\text{entries}} = \min(N, k + w)$；
- 线性层：$n_{\text{entries}} = 0$，但多一项状态读写 $2 n_h d_k d_v$。

**KV**：

$$
\text{KV}_{\text{bytes/token}} = \sum_{\text{source}} \frac{n_{\text{src}} \cdot (b_{\text{entry}} + b_{\text{indexer}})}{m}
$$

其中 $m$ 是序列压缩比（每个条目代表多少 token），$n_{\text{src}}$ 是这一档有几份独立 KV。

## 两个容易混淆的概念

脚本把「容量」和「带宽」分开算：

- `kv_bytes_per_token()`：**常驻容量**，乘以上下文长度就是显存占用；
- `kv_read_per_token(ctx)`：**decode 每步实际从 HBM 读的 KV**，稀疏注意力只读 top-k 个条目。

稠密注意力两者相等；稀疏注意力把两者解耦——容量仍 $O(N)$，带宽只 $O(k)$。这个区别决定了长上下文 decode 的瓶颈是 KV 还是权重。

## 一个例子：V4.1 的 890 B/token

```
python3 budget.py --detail
```

```
== DeepSeek-V4.1-Flash ==
  总参数       747.4B   (embed 1.3B, attn 5.1B, ffn 545.0B, hc 39.4M, extra 196.0B)
  激活参数     16.0B
  KV/token     890 B
```

890 的拆解（报告只给聚合值，脚本按 config 复算）：

| 项 | 每条目字节 | source 数 | 压缩比 | 贡献 |
|---|---|---|---|---|
| 压缩 KV latent（512 维 FP4 + 32 scale） | 288 | 3（encoder） | 2 | 432 |
| 压缩 KV latent | 288 | 1（decoder） | 1 | 288 |
| 索引器 K（128 维 FP4 + 4 个 E8M0 scale） | 68 | 3（encoder） | 2 | 102 |
| 索引器 K | 68 | 1（decoder） | 1 | 68 |
| **合计** | | | | **890** |

## 已知边界

- 权重按 BF16 计，未建模 FP8/FP4 权重的收益（KV 精度单独建模）；
- 未建模 attention sink、RoPE 的 64 维 mix 精度（V4-Flash 用等效 9 bit/维近似）；
- 线性层的状态大小按 $n_h d_k d_v$ 估，未含卷积/门控的低秩参数细节；
- Engram 只计参数与查表字节，未建模预取与哈希碰撞；
- 对拍容差按模型给（1%-3%），K2.6/K3 因专家/线性层细节误差稍大。
