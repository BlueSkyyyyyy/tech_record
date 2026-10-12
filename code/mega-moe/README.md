# mega-moe — DeepSeek Mega MoE 前向 / 反向算子实现解析

以 **DeepGEMM**（`deepseek-ai/DeepGEMM`）与 **DeepEP**（`deepseek-ai/DeepEP`）为参考，
拆解 DeepSeek 的 **Mega MoE**：把 **EP dispatch → linear1 → SwiGLU → linear2 → EP combine**
融合进**单个 mega-kernel**、并让 **NVLink 通信与 Tensor Core 计算重叠**的设计；
以及 MoE 的**反向**（weight-gradient / input-gradient 的 grouped GEMM）实现。

- 参考实现：`~/github/DeepGEMM`（Mega MoE、grouped GEMM、backward 的 wgrad API）；
  `~/github/DeepEP`（dispatch/combine 通信原语、symmetric memory）。
- 环境：`kernel_lab` 容器（H100 sm90）。**注意：Mega MoE 主实现是 SM100（Blackwell）**，
  本机跑不了端到端；本专题以**源码精读 + 线上资料 + 可行的 grouped-GEMM/backward 数值实验**为主。
- 博客专题：`content/posts/mega-moe-*`，正文由 `code/mega-moe/docs/*.md` 经 `doc_include` **实时内联**。

## 目录

```
mega-moe/
  ROADMAP.md                 # 任务/进度/下一步（autopilot 每轮先读）
  docs/                      # 分析文档（内联到博客）
  harness/                   # 参考实现探测 / 数值实验 / 基准
  src/                       # 我们自己的参考实现与实验 kernel
  scripts/                   # lab / autopilot
```

## 参考坐标

| 资源 | 位置 | 用途 |
|---|---|---|
| Mega MoE API | `~/github/DeepGEMM/csrc/apis/mega_moe.hpp` | 接口、symmetric buffer、weight transform |
| Mega MoE 实现 | `~/github/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | 主体 kernel（1528 行） |
| Mega Gate / mHC | `~/github/DeepGEMM/csrc/apis/mega_gate.hpp`、`mega_mhc.hpp` | 相关 mega-kernel |
| 调度/布局 | `~/github/DeepGEMM/deep_gemm/include/deep_gemm/{scheduler,layout}/mega_moe.cuh` | ring buffer、task 调度 |
| 测试/示例 | `~/github/DeepGEMM/tests/test_mega_moe.py` | 多进程启动、基准 |
| EP 通信 | `~/github/DeepEP/csrc/kernels/ep/` | dispatch/combine |
| 反向 wgrad | `k_grouped_fp8_gemm_tn_contiguous`（README §Grouped GEMMs） | MoE weight backward |
