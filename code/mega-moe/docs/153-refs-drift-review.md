# 引用行号巡回复核：113 条 near_miss 逐条判定，修正 12 处（含 1 处写错的符号名）

> 本文对应路线图 **P143-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的**第 153 篇**。
> 目标：把前面三次「引用体检」留下的 **near_miss（轻微漂移）清单**做一次**定点复核**——
> 不是重跑 linter 就交差，而是**逐条判定「这条到底错没错、错在哪、要不要改」**，把可机械修正的
> 直接改掉，并给出一个**可复现的复核工具** `fix_refs.py`。
>
> **可跑性**：整条链路（`check_refs.py` → `triage_refs.py` → `fix_refs.py`）都是**纯文件解析、
> 架构无关、不加载 GPU**——在 H100(sm90) 上跑只为与专题其它实验同环境。参考坐标：
> DeepGEMM `057ca59`、DeepEP `93eb6eb`。
>
> 结果：**硬错误（无法解析 / 越界 / 空行 / 歧义）保持 0**；`near_miss` **113 → 102**；严格命中
> **705 → 716**；共修正 **12 处**（8 处区间加宽 + 3 处真错行号 + **1 处写错的符号名**）。
> 剩下 102 条经逐条核对**全部是 linter 粗粒规则的假阳性**（分类依据见 §5）。原始输出见
> `harness/exp_refs_drift_review.out.txt`。

---

## 1. 为什么单开一篇：三次体检都只到「分类」，没到「处置」

这个专题的每一个数字都带源码行号，而这些行号会随两库的上游提交**漂移**。为此前面已经建了三层体检：

- **P96**（`docs/106`）：第一次源码索引审计，发现「把 `third-party/` 也扫进来」导致大量同名歧义；
- **P97**（`docs/107`，工具 `harness/check_refs.py`）：修掉三个静默误判（排除 vendored 拷贝、
  用「被引行是否为空」消歧、占位省略不算错），给出**硬错误**（`unresolved/oob/empty/ambiguous`）
  与**内容抽查**（严格命中 / 宽松命中 / `near_miss` 漂移）两级指标，退出码可进 CI；
- **P123**（`docs/133`，工具 `harness/triage_refs.py`）：把 `near_miss` 粗分成
  `owner_body` / `far_other` / `absent` / `adjacent_code` / `same_line_other` / `adjacent_decl`，
  并挑出 **7 条 `candidate off-by-one`** 供人工复核。

问题在于：`triage_refs.py` 只是**分类 + 挑候选**，它不回答复核时真正的两个问题——
**这个标识符到底在不在（被解析到的）文件里？** 以及 **在文件里的话，应该改到哪一行？**
于是「7 条 candidate off-by-one」就悬在那里，没人真正处置；而 `far_other` / `absent`
里混着的**真错误**（例如「写错的符号名」）也永远不会被单独挑出来。本篇补的就是这最后一步。

---

## 2. 复核方法：`fix_refs.py` 的六类判定

新工具 `harness/fix_refs.py`（纯文件解析、复用 `check_refs` / `triage_refs` 的索引与消歧）对
`check_refs.py` 报出的每一条 `near_miss`，对行内每个反引号标识符给出**确定性判定**：

| verdict | 含义 | 处置 |
|---|---|---|
| `in_range` | 标识符就在被引区间内（linter 的分裂区间/子串判定漏了） | 无需改 |
| `widened` | 标识符在 `l1-1` / `l2+1`，把区间**扩到包含它**即可 | **可机械改** |
| `wrong_range` | 标识符在文件里但离被引区间很远 | 人工判断（多见假阳性） |
| `wrong_file` | 标识符不在被引文件、但在**别的文件**里 | 人工判断（含分辨率歧义） |
| `unknown_symbol` | 全语料都找不到该标识符 | 人工判断（含**写错的符号名**） |
| `cross_talk` | 同一行多条引用、该标识符更贴近另一条 | 假阳性，无需改 |

`--apply` 只把 `widened` 类机械改写回文档——扩区间**只可能让引用更精确（变成超集）**，
不会引入新的事实错误，因此是安全的自动化边界；`wrong_range` / `wrong_file` / `unknown_symbol`
一律只报告、不自动改。

关键点：`wrong_file` 与 `unknown_symbol` 是**本篇新增的、triage 没有的两个信号**——它们正是
「写错的符号名」的落点。

---

## 3. 结果总览

| 指标 | 复核前 | 复核后 |
|---|---|---|
| 总引用条数 | 2296 | 2303 |
| 无法解析 / 越界 / 空行 / 歧义 | 0 / 0 / 0 / 0 | **0 / 0 / 0 / 0** |
| 内容抽查条数（同行有标识符） | 958 | 959 |
| 严格命中（标识符在被引行） | 705 (73.67%) | **716 (74.66%)** |
| 宽松命中（符号在文件内） | 818 | 818 |
| `near_miss`（在文件、不在被引行） | **113** | **102** |
| 覆盖的源文件数 | 255 | 255 |

严格命中增加 11、`near_miss` 相应减少 11；召回（宽松命中）与硬错误不变，说明所有修正都是
「把引用改到标识符真正所在的行」，没有靠删引用或放宽判据来美化指标。两列总引用条数差 7
是**本篇正文自身**新增的被检查引用（复核前的基线取自加入本篇之前的语料）。

---

## 4. 修正的 12 处

### 4.1 区间加宽（8 处，`--apply` 机械完成）

这 8 处的共同模式是「**作者引了代码块的起始行（如 `if (env)` 守卫行），而标识符落在下一行**」。
把区间扩到包含标识符后，严格命中即成立：

```
07-deepep-dispatch-combine.md:37    symmetric.hpp:79        -> symmetric.hpp:78-79
117-deepep-env-surface.md:181       buffers/ep.hpp:163      -> buffers/ep.hpp:163-164
117-deepep-env-surface.md:181       buffers/bucket.hpp:51   -> buffers/bucket.hpp:51-52
123-deepep-python-caching.md:22     find_pkgs.py:58         -> find_pkgs.py:58-59
123-deepep-python-caching.md:194    envs.py:207             -> envs.py:207-208
13-moe-backward-dgrad.md:293        gemm.hpp:218            -> gemm.hpp:217-218
134-deepep-bucket-test-oracle.md:202 allocator.py:11        -> allocator.py:10-11
74-dense-gemm-test-oracle.md:285    test_bf16.py:49         -> test_bf16.py:49-50
```

例如 `buffers/ep.hpp:163` 引的是 `if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {` 守卫行，
而 `tensors_to_record` 成员赋值在 164 行；`allocator.py:11` 引的是 `class BufferAllocator`
的 docstring，类定义其实在 10 行。

### 4.2 真错行号（3 处，人工核对后手改）

<!-- refcheck:ignore-begin -->
这三处不是「引守卫行」的书写约定，而是**指向了完全无关的行**——用 `fix_refs.py` 的
`wrong_range` verdict（标识符在文件里但离被引区间很远）挑出，再回源码确认后改：

```text
93-deepep-bucket-session.md:216    event.py:73      -> deep_ep/utils/event.py:70
98-deepep-jit-config.md:173        config.hpp:44    -> deep_jit/runtime/config.hpp:48
25-sf-layout-transform.md:99       layout.hpp:35    -> csrc/utils/layout.hpp:101
```

- `deep_ep/utils/event.py:70` 才是那条 `assert self.hook_after_wait is None`（73 行是 `def __call__`）；
- `third-party/deep_jit/include/deep_jit/runtime/config.hpp:48` 才是 `get_python_path`（44 行是
  `lexically_normal()` 那行）；
- `csrc/utils/layout.hpp:101` 才是 `check_sf_layout`（35 行是另一个函数）。

### 4.3 写错的符号名（1 处）

**`docs/126-deepgemm-deepep-exception-layer.md:235`** 把一个**不存在的**符号名当成了结构体名：

> 因为它在 host 侧调用 `DG_UNIFIED_ASSERT` 的地方——例如
> `layout/mega_moe.cuh:119-125` 的 `MegaMoEWorkspace` 构造函数（`CUTLASS_HOST_DEVICE`…）

而 `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` 里**根本没有**那个名字——真正承载这些
`DG_UNIFIED_ASSERT` 的结构体叫 **`Workspace`**（`:93` 定义、`:105-126` 构造函数、`:119-125` 就是
那 7 条断言）。这是 `unknown_symbol` / `wrong_file` 信号挑出来的**唯一一处真错误**。
→ 符号名改为 `Workspace`（行号 `:119-125` 本就正确，不动）。
<!-- refcheck:ignore-end -->

---

## 5. 剩下 102 条为什么是假阳性

复核后仍有 102 条 `near_miss`。逐条核对（`fix_refs.py` 报告 + 回源码）后，**全部是 linter
粗粒规则的假阳性**，无一需要改。按 `fix_refs.py` 的 verdict 分布（120 个「行×标识符」对）：

- **`wrong_range`（92 对）**：标识符在文件里、但离被引区间远。绝大多数是**同一行里另一条引用的
  标识符**，或**行文顺带提到的符号**。已核对为正确的例子：
  - `10-mega-moe-mma-epilogue.md:117` 引 `mma/sm100.cuh:30`（`make_smem_desc` 的形参在 15/48…），<!-- refcheck:ignore -->
    而 30 行 `desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);` **正是**正文那句
    「`start_address_ = uint_ptr >> 4`」——引用正确；
  - `30-sm90-paged-mqa-logits-schedule.md:264` 引 `sm90_fp8_mqa_logits.hpp:192`，192 行
    `DG_HOST_ASSERT(next_n == 1 or next_n == 2);` **正是**正文「`next_n∈{1,2}` 是硬约束」；
  - `87-mega-moe-sync-primitives.md:125` 引 `barrier.cuh:46-47/83-84`，那两段**正是**传给
    `wait_until` 的 `print_timeout` 回调（标识符 `print_timeout` 作为模板形参在 14/16/20 行）。
- **`wrong_file`（20 对）**：标识符不在**被解析到**的文件里。两类：
  1. **行文顺带提到的无关符号**（`k_grouped`、`BLOCK_M`、`int64`、`AssertionError`、`RuntimeError`…）；
  2. **同名文件分辨率歧义**——典型是 `52-kernel-launch-pdl.md:86` 引 `runtime/runtime.hpp:31` 说
     `default_launch_options`，它其实在 **DeepJIT** 的 `third-party/deep_jit/include/deep_jit/
     runtime/runtime.hpp:31`（`:42` 初始化）；linter 把它解析成了 DeepGEMM 自己的
     `csrc/runtime/runtime.hpp`（那里当然没有）。行号是对的，只是路径段 `runtime/runtime.hpp`
     有歧义——这是 linter 已知的、记录在案的边界，不是文档错误。
- **`cross_talk`（6 对）**：一行多条引用，标识符按字符距离更贴近另一条。纯方法学假阳性。
- **`unknown_symbol`（2 对）**：`105-fp4-mega-moe-weights.md:147` 的 `ALIGN16B` 是行文解释
  TMA 常量名时顺带写的（引用的是 `runtime_utils.hpp:63,125` 的默认参数，正确）；
  `130-mega-mhc-backward.md:49` 的 `impls:501` 是一个**跨文件行号简写**（不是符号名），
  被 `IDENT` 正则误当成标识符。

一句话：`near_miss` 里有**大量**「同一行有多个反引号标识符」造成的串扰——这正是 P123 引入
`triage_refs.py` 的原因。本篇的增量是把它再往前推一格：**能自动改的改掉、解释不了的归类存档**，
让这个清单从「113 条未知」收敛到「102 条已判定假阳性」。

---

## 6. 可复现

```bash
# 在 kernel_lab 容器里（复用专题环境）
bash code/mega-moe/harness/exp_refs_drift_review.sh \
  > code/mega-moe/harness/exp_refs_drift_review.out.txt
```

三件套顺序：`check_refs.py`（硬错误 + 严格/宽松/near_miss）→ `triage_refs.py`（六类粗分
+ candidate off-by-one）→ `fix_refs.py`（六类**处置**判定 + 可机械修改清单）。`fix_refs.py`
默认 dry-run，`--apply` 才写回文档；`--only widened` 只看某一类。复核时**必须先跑
`check_refs.py`**（`fix_refs.py` 复用它算出的 `near_miss`）。

---

## 7. 结论与边界

- 引用体检从「发现坏引用」（P96/P97）到「分类 + 挑候选」（P123）再到本篇的「**逐条处置**」形成闭环：
  `near_miss` 113→102、严格命中 705→716、硬错误恒 0。
- 真正的错误只有 **4 处**（3 处错行号 + 1 处写错的符号名），其余 8 处是「引守卫行」的书写约定，
  可用区间加宽机械修正；剩下 102 条是 linter 粗粒规则的假阳性。
- **边界**：`near_miss` 仍然是「同一行多标识符」的串扰重灾区——只要作者在一行里同时用反引号点名
  多个符号，linter 就会把它们互相错配。`fix_refs.py` 的判定能把其中「真错误」的三类
  （`wrong_file` / `unknown_symbol` / `wrong_range` 中的远距离项）单独列出，但**最终仍需人读**——
  本轮的 4 处真错误都是这么挑出来再回源码确认的。
