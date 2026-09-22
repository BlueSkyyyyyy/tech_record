# fa-bwd — FlashAttention 反向工程项目的自驱流程

## 触发场景

用户说「继续 FlashAttention 反向」「fa-bwd」。本仓库长期自驱任务之一。

## 前置条件

- 读 `agent_guide.md`（目录约定/坑，尤其：**不要写目标卡代号**、`fa_include` 内联渲染机制）。
- 读控制面板 `code/flash-attention/fa-bwd/ROADMAP.md`（每次先读，从「下一步」做起，做完更新）。
- 环境：复用 `kernel_lab` 容器（`code/flash-attention/fa-bwd/scripts/lab.sh` 指向 kernel-opt 的脚本）。

## 关键事实

- 参考实现已装好：**flash_attn 2.7.4** 与 **TE 2.14**；`harness/probe_refs.py` 验证反向与 fp32 ref 吻合。
- dump/对拍/基准：`harness/fa_bwd_bench.py`；I/O 落在 **`/home/xieminglin/proj/output/fa-bwd/<slug>/`**（CPU npy + meta.json）。
- ours 实现：`src/{fp16,bf16,fp8}/`，各有 **单文件** 与 **两文件**（`_kernels.cuh` + `_main.cu`）。
- 生产形状（GQA/MQA/MLA）在 `harness/REQUESTED_SHAPES`；MLA `head_dim=512` FA/TE 都不支持，只能靠 ours。
- 博客专题由 `code/flash-attention/fa-bwd/docs/*.md` **实时内联**渲染（`{{< fa_include "..." >}}`），
  改 docs 即等于改博客，无需另建文章。

## 步骤（每轮一个增量）

1. 读 ROADMAP「下一步」，选一个 P/O 项。
2. 改 `src/<dtype>/`，`scripts/run.sh` 编译运行；`scripts/ncu.sh` 剖析（bound 在哪）。
3. 对拍：读 dump 的 `q,k,v,do` + `ref_*.npy`，报告 max abs/rel；必要时 `fa_bwd_bench.py dump/bench`。
4. 对标：与 FA2/TE（同 shape、CUPTI 纯 device 时间）比，给出 TFLOPS 与差距。
5. 更新 `docs/`（会同步到博客）、`ROADMAP.md`；commit（`fa-bwd:` 前缀）→ push。
6. 不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。

## 验收标准

- [ ] 代码在 `kernel_lab` 编译运行、结果正确（fp16~1e-3 / bf16~1e-2 / fp8 另定）
- [ ] 有实测数字与 ncu 证据；dump/对拍/性能表更新
- [ ] 博客专题页面（内联 docs）构建无 ERROR、线上 200
- [ ] 无目标卡代号等敏感信息

## 已知坑

- 目标卡代号敏感：一律写「目标卡」。
- `{{< relref >}}` 必须放在 markdown 链接里；裸用只输出 URL 文本。
- fp8 转换不能用 `float(fp8)`（返回位模式），要用 `__nv_cvt_*`。
- `ldmatrix` 的 smem 布局要 padding（`+8` 元素）消 bank conflict。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。
