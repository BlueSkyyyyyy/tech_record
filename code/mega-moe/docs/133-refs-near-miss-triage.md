# 引用行号的 near_miss 定点复核：分裂区间合并 + 五类分工 + 7 处修正

> 本文对应路线图 **P123-1**。第 106 篇（P96-1）给全专题约 1900 处 `file:line` 做了一次体检，
> 第 107 篇（P97-1）把它做成可复用的 linter `harness/check_refs.{py,sh}`，并留下两类 defer：
> 「10 条同名歧义 + 87 条轻微漂移（near_miss）」与「应统一写全相对路径」。歧义已由第 107 篇的
> 内容哈希去重消掉；**但 87 条 near_miss 一直没被逐条落账**——它们被含糊描述为「多为引用块/区间
> 首行约定与同行多引用串扰的启发式假阳性」。
>
> 本篇把这个 defer 收口：① 先说清 near_miss 这顶帽子**本来就会罩住大量假阳性**（它的判据太粗）；
> ② 顺手修掉 linter 里一个**真会漏判的方法学缺口**（分裂区间）；③ 把剩下的 near_miss 用脚本做
> **逐条五分类**，量化每类各多少；④ 对其中**确属引用写错**的 7 处当场改正并复跑验证。
> 纯文件解析、架构无关，任意机器可跑。

## 1. near_miss 为什么这么吵：判据本身太粗

`check_refs.py` 判 near_miss 的规则（`harness/check_refs.py:324-334`）是：

- 取这一行 Markdown 里**所有**反引号标识符（`IDENT`，`check_refs.py:51`）；
- 若它们**都**不在被引行区间内、但**至少一个**在同文件的别处出现，就算一条 near_miss。

这条规则有两个结构性噪音源：

1. **一行多引用**。作者常在一句话里并排写两三个引用，如
   `...（`dispatch.cuh:103`-`:105` 的 `ptx::deduplicate`）`。linter 把该行所有标识符都拿去
   对每一条引用做检查，于是「属于另一条引用的标识符」必然误报。
2. **行文提到 owner 名**。如「按组补齐 K（K 对齐硬要求，见 `check_k_grouped_args`，`gemm.hpp:69`）」——
   `check_k_grouped_args` 是**函数名**（定义在 `gemm.hpp:55`），而引用指的是它体内第 69 行的断言。
   引用**没有错**，只是「引体、点名 owner」。

所以 near_miss **不是「错误数」，而是「待人工判读数」**。要把它变成可交付的结论，必须逐条分类。

## 2. 先修一个真缺口：分裂区间（split range）被整段丢掉

作者写区间有两种风格：`a.hpp:100-120`（普通区间，linter 认）和 `a.hpp:100`-`:120`
（**分裂区间**：两个反引号 token，第二段只写 `:120`）。`CITE` 正则（`check_refs.py:50`）
只匹配带文件名的 `file:line`，于是第二种风格里**终点 `:120` 被整个丢弃**，区间退化成单行
`file:100`。后果有两层：

- 命中率偏低（本该在区间内的标识符若不在第 100 行，就被判 near_miss）；
- **偶尔把一个正确引用误报成 near_miss**。典型：`docs/01-moe-ep-basics.md:172` 写
  `dispatch.cuh:103`-`:105` 的 `ptx::deduplicate`，而 `ptx::deduplicate(...)` 的调用在**第 104 行**
  （区间中段），linter 只看到 `:103` 就报了 near_miss。

修法（`check_refs.py:56-71`）：新增 `CONT_RANGE` 先把 `file:l1`-`:l2` / `file:l1`-`l2` 合并成
`file:l1-l2`，再交给 `CITE`；`strip_doc` 在逐行扫描前先过一遍 `normalize_citations`
（`check_refs.py:135`），并统计合并条数 `MERGED_RANGES`（`:62/69`）。当前语料共合并 **13 条**：

| 文档:行 | 合并后区间 |
|---|---|
| `01-moe-ep-basics.md:172` | `dispatch.cuh:103-105` |
| `02-mega-moe-architecture.md:194/203/218/238/321` | `test_mega_moe.py:130-133`、`sm100_fp8_fp4_mega_moe.cuh:1017-1020`/`145-149`、`smxx_layout.hpp:137-138`、`heuristics/mega_moe.hpp:220-221` |
| `07-deepep-dispatch-combine.md:58/69/78/90/104/122/169` | `context.cpp:226-232`/`174-183`、`token.cuh:10-30`、`workspace.cuh:9-23`、`dispatch.cuh:78-257`/`258-390`、`combine_utils.cuh:67-70` |

其中 `01:172` 与 `07:169` 合并后标识符落回区间内，near_miss 直接 -2。

## 3. 五分类：把 near_miss 变成可判读的台账

新脚本 `harness/triage_refs.{py,sh}` 复用 linter 的解析，对每条 near_miss 里的**每个标识符**，
算出它在源文件里的**最近出现行**到被引区间的距离，并据此分类：

- `in_range`：标识符其实**就在（合并后的）区间内**（linter 边界/子串判定漏掉）；
- `same_line_other`：该行有多条引用，标识符按字符距离更贴近**另一条**引用（一行多引用串扰）；
- `adjacent_decl`：最近出现在 `l1-1`/`l2+1`，且那一行是**装饰器/注释/模板头/签名头**
  （`@lru_cache`、`// 注释`、`template <...>`、`static void f(`）——「引体、名在头」的约定；
- `owner_body`：标识符是**定义在被引行之上的函数/结构体名**（行文在体内引用时顺带点名 owner）；
- `far_other`：最近出现处离区间 > 5 行（行文顺带提到的无关符号）；
- `absent`：标识符在该文件里根本没有（多半本就属于另一个文件）。

按当前 99 条 near_miss、**标识符级**统计：

| 类别 | 数量 | 判读 |
|---|---|---|
| `owner_body` | 59 | 良性：引体点名 owner |
| `far_other` | 25 | 良性：行文提到的无关符号（同 `same_line_other` 性质） |
| `absent` | 22 | 良性：符号属于别的文件/是行文词 |
| `adjacent_code` | 6 | **待判**：相邻行是代码（非注释/签名） |
| `same_line_other` | 2 | 良性：一行多引用串扰 |
| `adjacent_decl` | 1 | 良性：装饰器/模板头 |

也就是说，**约 93% 的 near_miss 是「引用没错、只是判据太粗」**。真正需要动作的只有
`adjacent_code`/`adjacent_decl` 里「被引行确实少写了一行」的那几处。

（脚本里 `is_decl_head`（`triage_refs.py:68`）用「行首是 `@ / // / /* / * / # / --`、或含
`template<`、或以 `(`/`,`/`\` 收尾」来识别头行；`nearest_occurrence`（`:75`）按词边界找出现行。）

## 4. 闭环修正：7 处「引用少写了定义/签名那一行」

把 `adjacent_code`/`adjacent_decl` 逐条对照源文件后，确认 7 处是**引用区间起点晚了一行**
（符号/签名在上一行或区间外一行），当场改正：

| 文档:行 | 原引用 | 改为 | 真实锚点 |
|---|---|---|---|
| `07-deepep-dispatch-combine.md:83` | `token.cuh:84` | `token.cuh:84-85` | `template <bool kWithMBarrier>`(84) + `struct BufferLayout`(85) |
| `07-deepep-dispatch-combine.md:170` | `combine_utils.cuh:8` | `combine_utils.cuh:8-9` | `template <...>`(8) + `use_rank_layout()`(9) |
| `05-mega-gate-mhc.md:648` | `mega_mhc.hpp:81-82` | `mega_mhc.hpp:81-83` | `y_fp8` 的说明在第 83 行 |
| `45-muon-syrk-symm.md:31` | `smxx_cublaslt.hpp:223-234` | `smxx_cublaslt.hpp:222-235` | 函数签名在第 222、闭括号在 235 |
| `58-deepep-combine-reduce.md:136` | `ptx.cuh:594-597` | `ptx.cuh:593-597` | `accumulate(...)` 定义在第 593 行 |
| `78-sf-layout-test-oracle.md:311` | `generators.py:277` | `generators.py:276-277` | `def enumerate_sf_layout()` 在第 276 行 |
| `84-deepgemm-python-api-surface.md:150` | `gemm.hpp:845` | `gemm.hpp:844-845` | `py::arg("alpha")` 在第 844 行 |

## 5. 效果：near_miss 与命中率

| 阶段 | 总引用 | 严格命中 | 严格% | near_miss |
|---|---|---|---|---|
| P97 linter + 今日语料（基线） | 2074 | 640 | 72.64% | 108 |
| + 分裂区间合并（§2） | 2074 | 642 | 72.87% | 106 |
| + 7 处引用修正（§4） | 2074 | 649 | 73.67% | **99** |

（P97 文档记录的 1703/87 是当时语料；此后新增了 129~132 等篇，故基线已涨到 2074/108。
所有 oob / unresolved / empty / ambiguous 四类**硬错误仍恒为 0**，`check_refs.py` 的 CI 退出码为 0。）

## 6. 剩下的 6 条 `adjacent_code` 逐条判读

修正后仍剩 6 条相邻代码行，逐条核对后**都是书写约定、非错误**，保留不改：

- `117-deepep-env-surface.md:181` 的 `tensors_to_record`：引用点在 `EP_AVOID_RECORD_STREAM`
  分支（`ep.hpp:163`/`bucket.hpp:51`），字段赋值在下一行（`164`/`52`）；引用指「该分支」，正确。
- `123-deepep-python-caching.md:22` 的 `find_nccl_root` / `:194` 的 `get_nvlink_gbs`：
  引用点在 `@functools.lru_cache()` 装饰器行（`find_pkgs.py:58`/`envs.py:207`），函数名在下一行；
  引用「装饰器」是本文的论点，正确。
- `13-moe-backward-dgrad.md:293` 的 `grouped_layout`：符号在表格首列的散文里，引用点是断言
  `gemm.hpp:218` 的 `m == m__`；引用指「哪一行拒绝」，正确。
- `74-dense-gemm-test-oracle.md:285` 的 `BF16StochasticRounding`：引用点是该测试的
  `if get_arch_major() == 10 ...`（`test_bf16.py:49`），符号用在下一行（`50`）；引用指「该分支」，正确。

## 7. 复现

```
bash harness/check_refs.sh   > harness/check_refs.out.txt     # linter 主报告 + JSON
bash harness/triage_refs.sh  > harness/triage_refs.out.txt    # near_miss 五分类
```

两份 `.out.txt` 均入库。`triage_refs.py` 支持 `--far N`（远近阈值，默认 5）与 `--quiet`。

按第 107 篇的先例，**本篇自身也加入 `check_refs.py` 的 `EXCLUDE_DOCS`**（`check_refs.py:47-49`）：
它在 §4 的「原引用」列**按设计引用了改正前的坏引用**，纳入体检会造成「自指涉」不幂等
（第 107 篇已量化过同类效应）。故上表数字是**不含本篇**的语料。

## 8. 与前面各篇的关系

- 第 106 篇：建立「有多少引用对得上」的**数字**与全算子索引，修 2 处硬错误；
- 第 107 篇：把体检**可复用化**（内容哈希去重、多层消歧、CI 退出码），修 2 处 off-by-one，
  留下 87 条 near_miss defer；
- **本篇**：修掉 linter 的**分裂区间缺口**，把 near_miss 从「一顶含糊的帽子」变成**五分类台账**，
  闭环修 7 处，并证明剩余 93% 是判据噪音而非错误。至此引用体检这条线**收口**：硬错误恒 0、
  near_miss 全部归类可判读。

## 9. 引用

- 路线图 **P123-1**（`code/mega-moe/ROADMAP.md`）。
- linter：`code/mega-moe/harness/check_refs.py`（`CONT_RANGE:56`、`normalize_citations:65`、
  `strip_doc:135`）、`harness/check_refs.sh`、`harness/check_refs.out.txt`。
- 分类器：`code/mega-moe/harness/triage_refs.py`（`citations_on:54`、`is_decl_head:68`、
  `nearest_occurrence:75`、`triage:86`）、`harness/triage_refs.sh`、`harness/triage_refs.out.txt`。
- 源仓库：DeepGEMM `057ca59`（#462）、DeepEP `93eb6eb`（v2.5.0）。
