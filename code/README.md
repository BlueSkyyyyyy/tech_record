# 配套代码

本目录存放博客文章的可运行代码。目录名与 `content/posts/` 下的文章目录名一一对应。

| 文章 | 代码 | 说明 |
|---|---|---|
| [开篇](../content/posts/hello-world/) | — | 无代码 |
| [Flash Attention 精读系列](../content/posts/flash-attention-01-theory/) | [flash-attention/ref_impl.py](flash-attention/ref_impl.py) | 纯 PyTorch 参考实现：online softmax 递推、分块前向（含 LSE）、recompute 反向，含自测（`python ref_impl.py`） |
| [Megatron-LM 源码精读系列](../content/posts/megatron-code-01-structure/) | [megatron-code/](megatron-code/README.md) | 源码分析笔记（analysis-notes/），基于 commit f713506ce |
| [H100 上 TE 算子的性能与 Roofline 对比](../content/posts/te-perf-roofline/) | [te-perf/](te-perf/) | `bench_te.py`：TE 原生 kernel（rmsnorm / bwd / bwd_add / fused_attn）微基准 + launch overhead |
