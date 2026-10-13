# DeepGEMM / DeepEP 引用体检（二）：把体检做成可复用 linter，并收口 P96 的 defer 项

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P97-1 增量（第 107 篇），是第 106 篇
> （P96-1）的**第二遍**。第 106 篇第一次给全专题约 1900 处 `文件:行号` 引用做了可复现的体检，留下
> 两类未竟项：**10 条同名歧义 + 92 条「轻微漂移」供后续复核**，以及一句结论——「未来若要严谨，
> 应统一写成全相对路径」。但第二遍一开始就发现：**P96 的体检脚本自身有三个会静默误判的方法学问题**，
> 不修它们，复核就没有可信的输入。
>
> 本篇做三件事：① 把体检重写成**可复用的 linter** `harness/check_refs.py`（修掉三个方法学问题、
> 支持 fenced/ignore、给出退出码可直接进 CI）；② 用新 linter 把「硬错误」收敛到 **0**
> （无法解析 / 行号越界 / 被引行全空 / 同名歧义未消解 全为 0），并**闭环修掉 2 处真实 off-by-one**；
> ③ 量化一个非平凡现象——**审计文档的自指涉**：第 106 篇按设计引用了一批「坏引用」当作修正名单，
> 下一轮体检会把这 5 条当成新错误。
>
> **纯文件解析，不加载 GPU，任何机器都能跑**；在 `kernel_lab(H100)` 里跑只为与专题其它实验同环境。
> 源码坐标：`~/github/DeepGEMM`（commit `057ca59`）、`~/github/DeepEP`（commit `93eb6eb`）。
> 实验：`harness/check_refs.py` + `harness/check_refs.sh` → `harness/check_refs.out.txt`。

---

## 1. 为什么要有第二遍：P96 留下了什么

第 106 篇的产出是一次**快照**：`总 1904 条引用、可解析 99.68%、行号越界 0`，同时明确留下：

- **10 条同名歧义**（`runtime.hpp:44`、`gemm.hpp:806`、`python_api.cpp:22` …）——脚本无法确定指向哪个同名文件；
- **92 条「轻微漂移」**（符号在本文件、却不在被引的那一行）——文档说「多为 off-by-one / 同行多引用串扰的启发式假阳性」。

这两类项的共同点是：**判据不可信**。例如第 106 篇说 `runtime.hpp:44` 落在
`csrc/jit_kernels/heuristics/runtime.hpp:44`（一个空行），结论是「同名歧义，待观察」；但专题里
`runtime.hpp:44` 真正想指的是 `csrc/runtime/runtime.hpp:44`（cuBLASLt handle 那行）。P96 之所以判错，
不是文档错，是**脚本消歧的排序键不对**。

## 2. P96 体检脚本的三个「会静默误判」的方法学问题

复现 P96 的 `exp_deepgemm_source_audit.py`（今天重跑）得到：`total 1933 / unresolved 9 / oob 3 /
empty 24 / near_miss 95`。逐条追因，三个问题都出在**索引与消歧**，而非文档本身：

### 2.1 索引把 vendored `third-party/` 也扫了进来 → 同名爆炸

P96 的 `build_file_index()`（`:72-85`）从仓库根 `os.walk`，只排除了 `.git/build/__pycache__`，
于是 DeepGEMM 自带的 **`third-party/cutlass`（含 `deep_gemm/include/cute|cutlass` 一份拷贝）** 与
**两份 `third-party/deep_jit`（DeepGEMM、DeepEP 各一）** 全部进了索引。后果是几个最常被引用的名字
候选数暴涨：`runtime.hpp` **4 个**、`python_api.cpp` **4 个**、`math.cuh/math.hpp/math.py` 各 2~3 个、
`cute/*` 一系列 **2 个（完全相同的拷贝）**。消歧一旦落到错的那份，就产生「行号越界 / 空行 / 漂移」的假警报。

### 2.2 消歧的排序键没利用「被引行是否为空」

作者核对过行号，**引用极少指向空行**。所以「某候选的被引行全空、另一候选非空」几乎是判定
选错文件的铁证。P96 的 `score()`（`:185-196`）把 `(行号在界, 严格命中, 宽松命中, 路径后缀)` 当字典序，
其中「行号在界」对空行也算命中，于是铁证被浪费。

### 2.3 占位省略号被当成「错误」

文档里 `impls/...cuh:56`、`sm90_...hpp:30` 是作者**故意省略**（避免写全长的 1528 行文件名），
P96 把它归入「无法解析」（6 条），并解释为「人为占位」。这不算错，但混在错误里干扰读数。

## 3. 新 linter 的设计：三层消歧 + 内容去重

`harness/check_refs.py` 相对 P96 的脚本有四处关键改进（行号对 `check_refs.py`）：

1. **候选按「内容哈希」去重**（`dedup_by_content`，`:179-188`）。`third-party/cutlass` 与
   `deep_gemm/include/cutlass` 是同一份拷贝、两份 `third-party/deep_jit` 也是同一份——**内容相同的候选
   折叠成一个**，歧义立刻消失，同时又不必整体排除 `third-party/`（因为 DeepJIT 的 `deep_jit/...` 确实
   是被第 33/47 篇精读过的真源码，`:44` 的 `EXCLUDE_DIRS` 只排构建产物）。
2. **消歧排序键 = `(被引行非空, 文档主仓库, 文档目录线索, 全语料引用偏好, 路径短)`**（`score`，`:236-240`）：
   - `lines_nonempty`（`:212-217`）——被引行非空优先；
   - `repo_score`（`:232-234`）——**先判定「这篇文档属于哪个仓库」**（`doc_repo_hist`，`:156-169`，
     用文档里所有**带目录**的引用投票），再优先同仓库候选。这一条修掉了 P96 的「`csrc/python_api.cpp`
     在 DeepEP 的文档里被解析到 DeepGEMM 那份」这类跨仓库错配；
   - `doc_dir_score`（`:219-229`）——同一篇文档里若出现过**该 basename 的带目录写法**，取该目录。
3. **占位省略号单列 `elided`**（`:50` 的正则先匹配、`:302` 处 `"..." in cited` 归为此类），不计入失败。
4. **fenced code block / ignore 标记**（`strip_doc`，`:109-126`）：``` 之间的示例默认不检查；
   支持 `refcheck:ignore` 行与 `<!-- refcheck:ignore-begin --> ... -end -->` 区块。

退出码：`unresolved + oob + empty + ambiguous > 0` 时返回 1（`main`，`:370-382`），可直接挂 CI。

## 4. H100 实测结果（新 linter）

`bash harness/check_refs.sh > harness/check_refs.out.txt`（`exit=0`）：

```
总引用条数            : 1699
  · 占位省略（...）    : 6（非错误）
  · 无法解析到文件     : 0
  · 行号越界/文件缺失  : 0
  · 被引行全为空       : 0
  · 同名歧义未消解     : 0
内容抽查（同行有标识符）: 741
  · 严格命中           : 530 (71.52%)
  · 宽松命中（符号在文件内）: 617 (83.27%)
  · 轻微漂移（在文件、不在被引行）: 87
覆盖的源文件数        : 236
```

对照 P96 的快照：

| 指标 | P96（docs/106 记录） | P96 脚本今日重跑 | 新 linter |
|---|---|---|---|
| 总引用 | 1904 | 1933 | 1699（排除 2 篇审计文档） |
| 无法解析 | 6（占位） | 9 | **0**（6 条占位归 `elided`） |
| 行号越界 | 0 | 3（全是审计文档的示例） | **0** |
| 被引行全空 | 10 | 24（12 条来自审计文档） | **0** |
| 同名歧义未消解 | （未单列） | — | **0** |
| 严格命中 | 554（74.76%） | 555 | 530（71.52%） |
| 宽松命中 | 646（87.18%） | 650 | 617（83.27%） |
| 覆盖源文件 | 198 | 199 | 236 |

两点读法：

- **总数从 1904/1933 降到 1699**，是因为把两篇**审计文档**（`docs/106`、`docs/107`）从健康检查里排除
  （见 §5）；被排除的正是「按设计引用坏引用」的那些条目。
- **严格命中率 71.52% 略低于 P96 的 74.76%**，不是变差：新 linter 把 `cute/cutlass` 双份拷贝按内容
  合并后，`resolve` 到的那一份可能不是文档作者当时看的那一份（内容相同、但行号未必一致），
  少数「同行多引用串扰」的样例因此从「严格」掉进「宽松」。**宽松命中 83.27% 才是真实一致性下限**，
  且**四类硬错误全为 0**——这才是本篇要的结论。

## 5. 非平凡现象：审计文档的自指涉

把两篇审计文档也纳入检查，新 linter 立刻报出 **oob 3 + empty 2**（`total 1718`）——**全部来自
`docs/106` 自己的示例名单**（`:183-185` 的「原引用」列、`:210-211` 的漂移例）：

```
oob:   106:183 m_grouped_gemm.py:92  -> legacy/m_grouped_gemm.py (len 84)
       106:184 mlopart.py:135        -> utils/mlopart.py (len 134)
       106:185 mega_moe.cuh:1440     -> scheduler/mega_moe.cuh (len 481)
empty: 106:210 heuristics/common.hpp:16
       106:211 gemm.hpp:806
```

这正是 P96 §5.1 亲手修好的三条旧错、以及 §5.2 列的待观察项。**审计文档按设计就在引用「已知的坏引用」
（作为修正记录），任何不做区分的体检都会把它们当成新错误**——体检因此**不幂等**，第二轮会「重新发现」
上一轮已修的东西。本篇的处置：把两篇审计文档列入 `EXCLUDE_DOCS`（`check_refs.py:47-48`），并在
linter 里保留 fenced/`refcheck:ignore` 两种细粒度豁免（`docs/106` 的名单里，fenced 块那部分已被自动跳过，
只剩散在表格/正文里的 5 条）。这是「元层级文档」与「内容文档」必须分开检查的一个具体理由。

## 6. 闭环修掉的 2 处真实 off-by-one

新 linter 的 `empty` 类别收敛出 2 个**非审计文档**的真错误（被引行是全空行＝行号写错），已修：

| # | 位置 | 原引用 | 证据 | 修正 |
|---|---|---|---|---|
| 1 | `docs/34-locality-domain-probe.md:47` | `csrc/runtime/locality_domain.hpp:13` | `:13` 是空行；「CUDA 13.4 兼容层」注释在 `:16`（与同句 `mlopart.hpp:13` 的注释对应） | `:16` |
| 2 | `docs/98-deepep-jit-config.md:131` | `csrc/buffers/bucket.hpp:78` | `:78` 是空行；文中所说的 `num_storage_bytes % kNumAllocationAlignmentBytes == 0` 契约断言在 `:77` | `:77` |

修完 `unresolved/oob/empty/ambiguous` 四类全为 0（§4 表）。

## 7. 87 条「轻微漂移」的诚实处理

新 linter 无法自动判定的还有 87 条 `near_miss`（符号在本文件、不在被引行）。逐条看过代表后，
绝大多数是**引用约定**而非错误，例如：

- **引用「块首行」**：`05-mega-gate-mhc.md` 引 `mega_mhc.hpp:81-82` 讲 `y_fp8`，而 `y_fp8` 声明在
  `:83`（区间取到函数签名行，符号在下一行）——这是「引用函数/结构体所在区间」的常见写法；
- **同行多引用串扰**：一行里同时有 `check_k_grouped_args` 与 `DG_HOST_UNREACHABLE` 两处引用，
  脚本把后者的标识符拿去核对前者所在的区间，必然「不在被引行」（P96 §5.3 已指出）；
- **缩写/别名**：文档写 `k_grouped_nt`，源码里是 `k_grouped_fp8_gemm_nt_contiguous`（宽松命中）。

因此本篇**不逐条改写**这 87 条（改写反而有引入新错的风险），而是把它们作为**已知、可复核**的一类
保留在 `check_refs.out.txt`，并说明判读口径：**真正要清零的是四类硬错误**（已 0），`near_miss` 是
「引用风格」的观测量。

## 8. 用法与边界

```
# 报告 + JSON 汇总 + 退出码（CI 用）
python harness/check_refs.py
# 只出 JSON
python harness/check_refs.py --quiet
```

- **可复现**：linter 的内建索引只读两库源码 + 本专题 `docs/`，行号核对是纯文本比对，无 GPU、无网络。
- **边界**：linter 是**启发式**——`ambiguous` 判据是「前两名得分完全相同」（保守，宁缺勿滥）；它
  不能判断「引用的行号虽非空、但语义指错了行」（那需要人读，正是 `near_miss` 的用途）。
- **与 P96 的关系**：`exp_deepgemm_source_audit.py` 仍是「算子 / API 全索引 + 首次体检快照」；
  本篇的 `check_refs.py` 是它的**可复用化与消歧修正版**。两者可并存：前者回答「哪些算子被哪篇覆盖」，
  后者回答「引用行号还对不对」。

**参考链接**：DeepGEMM <https://github.com/deepseek-ai/DeepGEMM>、DeepEP
<https://github.com/deepseek-ai/DeepEP>（本机核对 commit `057ca59` / `93eb6eb`）。
