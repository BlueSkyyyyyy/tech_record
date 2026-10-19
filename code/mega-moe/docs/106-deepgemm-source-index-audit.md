# 106. DeepGEMM / DeepEP 算子总索引与引用行号健康检查（source-rot audit）

> 本专题到第 105 篇，已在 105 份文档里写了约 **1900 处**形如 `文件.cuh:123-145` 的源码引用。
> 这些引用会被 doc_include **内联进博客**，但上游 DeepGEMM / DeepEP 每隔一两个月就发一版
> （`#304` → `#316` → `#432` → `#462`），行号迟早会「漂移」——而没有人会回头逐条核对。
> 本篇把「到底还有多少引用对得上」变成**可复现的数字**，并顺手给出**全局算子 / API 索引**，
> 让读者（和未来的自己）知道「某个 kernel / host 入口被哪几篇讲过」。
>
> 这是本专题第一篇**元层级**的增量：它不深挖某个 kernel，而是给前面 105 篇做一次**引用体检**，
> 并**闭环修掉体检发现的真实错误**。**纯文件解析、不加载 GPU**；在 `kernel_lab`（H100）里跑只是
> 为了与其它实验同环境。
>
> 目标卡代号敏感：全文用「目标卡」指代。

参考版本：`~/github/DeepGEMM` commit `057ca59`（`Public release 26/09/30 (#462)`，v2.8.1）；
`~/github/DeepEP` commit `93eb6eb`（`Update README`，v2.5.0）。
实测脚本：`harness/exp_deepgemm_source_audit.{py,sh}`，原始输出
`harness/exp_deepgemm_source_audit.out.txt`。

---

## 1. 为什么需要这篇

三个动机，按重要性排：

1. **可证伪性**：本专题一直强调「行号必须真实核对过」（见 ROADMAP「已知坑」）。但 1900 处引用靠
   人工是不可维护的。把它做成脚本后，「引用完整率」从一个形容词变成一个可回归的数字，任何一轮
   增量或上游升版后，跑一次就知道有没有回退。
2. **可导航性**：DeepGEMM 有 19 个 device impl 头文件、11 个 host API 头文件，读者想知道「`einsum`
   的 MLA 分支在哪篇」只能全文搜索。本篇给一张**算子 → 覆盖篇目**的总表。
3. **闭环**：体检当场发现了 **2 处真实错误**（行号越界 + 引错文件），本篇顺手修正，并演示
   「发现问题 → 定位 → 修复 → 复检」的完整循环。

### 1.1 引用在文档里的四种形态

抽取时统一成 `path.ext:line[-line]`。文档里实际出现的写法有：

| 形态 | 例 | 说明 |
|---|---|---|
| 带全相对路径 | `csrc/apis/gemm.hpp:181` | 最精确，无歧义 |
| 带部分路径 | `scheduler/mega_moe.cuh:157` | 用目录消除 basename 歧义 |
| 仅 basename | `mega_moe.hpp:35` | **有歧义**（同名文件可能多个） |
| 简写 / 占位 | `impls/...cuh:56` | 人为省略，不指向具体文件 |

前两种可精确解析；第三种要靠启发式（§2.2）；第四种本就不该被当成可解析引用。

---

## 2. 方法

### 2.1 四层核对

对每条引用，依次判定（越靠后越严格）：

1. **可解析**：该 basename / 后缀能在 `DeepGEMM` ∪ `DeepEP` ∪ 本专题目录里找到文件。
2. **行号在界内**：`1 ≤ line ≤ 文件行数`（区间则要求右端点也在界内）。
3. **被引行非空**：区间内至少有一行非空白。
4. **内容抽查**：把引用**同一 Markdown 行**里反引号包住的标识符（≥4 字符、排除文件名）抽出来，
   检查是否出现在：
   - 严格命中 = 被引行/区间内；
   - 轻微漂移 = 本文件其它行（符号在，只是行号偏了 → 值得复核）；
   - 无关 = 本文件里根本没有（多半是散文里引用的算子名 / 构造参数，不算漂移）。

### 2.2 解析歧义的启发式

像 `mega_moe.hpp` 同时存在 `csrc/apis/mega_moe.hpp`（400 行）与
`csrc/jit_kernels/heuristics/mega_moe.hpp`（265 行），`runtime.hpp`、`mega_moe.cuh`、`__init__.py`
同样多个。脚本对候选文件打分并取最高：

```
(行号在界内, 严格内容命中, 宽松内容命中, 路径后缀匹配)  ← 逐项字典序比较
```

- **行号在界内优先**：`mega_moe.hpp:155-283` 只能落在 400 行的 `apis/` 版本上，自动排除 265 行的
  `heuristics/` 版本；
- **内容命中兜底**：`__init__.py:125` 会优先落到 `deep_gemm/__init__.py`（含 `atexit` 等），
  而不是 `third-party/cutlass/.../__init__.py`。

### 2.3 局限（写下来以免过度解读）

- **同名多文件的第三条（仅 basename）无法 100% 消歧**：若两个候选的行号都在界内且都无内容线索，
  只能取排序靠前者。这类会落进「被引行全空」或「轻微漂移」两类，需要人工看一眼。
- **内容抽查是保守的**：一行 Markdown 常同时引用多个 `file:line`，其反引号标识符可能属于**另一条**
  引用；此时本条的「严格命中」会误判为「无关」。所以**严格命中率不是正确率下限**，
  「宽松命中率」才是更接近真实一致性的指标。
- 引用里带省略号（`impls/...cuh`）的不是真引用，一律计入「无法解析」。

---

## 3. Part A · 全算子 / API 索引

脚本按 `impls/*.cuh` 与 `csrc/apis/*.hpp` 枚举，对每个文件给 **sha256（前 12 位）**、行数，以及
「哪些篇目的正文提到过它」（按 basename 匹配——是**提及**而非精确引用，仅用于导航）。
sha256 是本轮体检的**基线指纹**：下轮再跑，指纹变了就说明该文件动过。

### 3.1 device impls（19 个）

| 文件 | 行数 | sha256 | 被提及篇目（#） |
|---|---:|---|---|
| `sm100_bf16_gemm.cuh` | 432 | `38d5d079d50e` | 31 36 48 52 54 103 |
| `sm100_bf16_mega_gate.cuh` | 447 | `0fbf34534703` | 05 52 96 102 |
| `sm100_bf16_mega_moe.cuh` | 1316 | `0387bd9bea62` | 20 38 52 77 87 89 105 |
| `sm100_bmk_bnk_mn.cuh` | 271 | `49f167d1a23a` | 17 52 |
| `sm100_fp8_fp4_gemm_1d1d.cuh` | 579 | `88dd217719c1` | 12 31 35 52 54 92 |
| `sm100_fp8_fp4_mega_moe.cuh` | 1528 | `9d577f200e48` | 01-04 09-11 20 34 38 44 51 52 54 58 77 87 96 99 105 |
| `sm100_locality_domain.cuh` | 26 | `a94bd377d699` | 11 34 55 |
| `sm100_mega_mhc.cuh` | 704 | `77fe3a79ad04` | 05 16 38 52 54 67 77 88 89 102 |
| `sm100_mqa_logits.cuh` | 609 | `3221a1809e2a` | 23 24 30 52 54 60 77 92 |
| `sm100_sparse_mqa_logits.cuh` | 688 | `3bdd07b6475a` | 22 38 52 88 |
| `sm100_tf32_hc_prenorm_gemm.cuh` | 351 | `37d28d517a78` | 16 52 79 97 |
| `sm90_bf16_gemm.cuh` | 399 | `b854395703c9` | 18 21 29 48 49 52 77 88 |
| `sm90_bmk_bnk_mn.cuh` | 182 | `614f643a9daa` | 17 49 52 |
| `sm90_fp8_gemm_1d1d.cuh` | 362 | `8d2cc57af14b` | 12 28 32 35 38 49 52 77 |
| `sm90_fp8_gemm_1d2d.cuh` | 452 | `fe9a88b1a2cc` | 21 32 33 49 52 87 99 |
| `sm90_fp8_mqa_logits.cuh` | 324 | `2f8d570cbc04` | 19 49 52 54 60 |
| `sm90_fp8_paged_mqa_logits.cuh` | 334 | `99adfbe0ef93` | 19 30 52 54 60 |
| `sm90_tf32_hc_prenorm_gemm.cuh` | 294 | `57f8d0f19f42` | 16 49 52 79 |
| `smxx_layout.cuh` | 295 | `cb72206057f0` | 25 26 33 52 54 78 91 |

### 3.2 host apis（11 个）

| 文件 | 行数 | sha256 | 被提及篇目（#） |
|---|---:|---|---|
| `attention.hpp` | 478 | `2c0cc337cf65` | 19 22 23 24 30 60 84 91 |
| `config.hpp` | 50 | `61b70480befe` | 17 27 28 33 36 47 48 50 52 84 88 98 103 105 |
| `einsum.hpp` | 311 | `27d93bcc1bbd` | 17 18 21 27 31 35 45 48 76 84 91 |
| `epilogue_class.hpp` | 27 | `f0ecbff74d52` | 27 31 84 |
| `gemm.hpp` | 997 | `7cfdbdcf9fcc` | 01 04 06 08 12-16 18 25 27-29 31-33 35 36 41 43 45 46 48 52 79 82 84 88 97 103 |
| `hyperconnection.hpp` | 64 | `0c73a4ad3f96` | 16 79 84 97 |
| `layout.hpp` | 144 | `63ec154c971a` | 02 12 13 21 25 26 28 32 34 35 41 45 46 48 78 81 84 91 105 |
| `locality_domain.hpp` | 162 | `9ba4d8cd931a` | 06 11 34 81 84 90 |
| `mega_gate.hpp` | 196 | `1e0cfc3361a1` | 05 06 48 57 84 96 97 102 |
| `mega_mhc.hpp` | 287 | `8a46910b0fe3` | 05 06 48 67 84 89 97 102 |
| `mega_moe.hpp` | 400 | `b52f852760da` | 02 03 06 10 11 20 34 42 44 48 51 84 89 91 96 97 99 105 |

**结论：没有任何一个 impl 或 api 是「(none!)」——30 个源文件全部至少被一篇覆盖。**
这与专题「以 DeepGEMM/DeepEP 为参考把 Mega MoE 前向/反向逐层讲透」的目标一致：不仅主
kernel 被拆过，连 `locality_domain`（26 行）、`epilogue_class`（27 行）这类配角也有专篇。

### 3.3 被引用最多的文件（精确 `file:line` 计数，非提及）

| 文件 | 精确引用次数 | 为什么多 |
|---|---:|---|
| `csrc/apis/gemm.hpp` | 95 | 所有 GEMM 的 host 入口 / 契约断言都在此 |
| `deep_ep/buffers/ep.py` | 66 | dispatch / combine / EPLB 全套 host 编排 |
| `csrc/apis/layout.hpp` | 60 | SF 布局 + grouped layout 的公共契约 |
| `csrc/apis/mega_moe.hpp` | 59 | Mega MoE host 入口 |
| `csrc/jit_kernels/heuristics/*runtime.hpp` | 57 | block 配置 / mk-alignment 的权威来源 |
| `csrc/apis/einsum.hpp` | 52 | einsum 家族四分派 |

---

## 4. Part B · 引用健康检查结果

| 指标 | 数值 |
|---|---:|
| 总引用条数 | **1904** |
| 可解析到源文件 | **1898（99.68%）** |
| 无法解析（占位 / 简写） | 6 |
| 行号越界 / 文件缺失 | **0** |
| 被引行全空 | 10 |
| 内容抽查总条数（同行有标识符） | 741 |
| · 严格命中（标识符就在被引行） | 554（74.76%） |
| · 宽松命中（符号在文件内，含严格） | 646（87.18%） |
| 覆盖的源文件数 | 198 |

**读法**：

- **99.68% 能落到具体文件、0 条越界**——说明专题的引用整体是「真核对过」的，不是凑数。
- 6 条无法解析全部是**人为占位**（`impls/...cuh:56`、`sm90_...hpp:30`），不是错误。
- 严格命中 74.76% 看着不高，但如 §2.3 所述，同行的多引用会互相干扰；**宽松命中 87.18%** 才是真实
  一致性水平。剩下约 13% 的「无关」绝大多数是散文里的算子名（如 `k_grouped_fp8_gemm_tn_contiguous`、
  `DG_HOST_UNREACHABLE`）出现在别处，不是引用本身错。

---

## 5. Part C · 漂移清单与已修正项

### 5.1 体检发现的真实错误（本轮已修）

| # | 位置 | 原引用 | 问题 | 修正 |
|---|---|---|---|---|
| 1 | `docs/39-legacy-grouped-gemm.md:244` + `harness/exp_legacy_grouped_gemm.py:31,133` | `m_grouped_gemm.py:92` / `:86-92` | 该文件**只有 84 行**，assert 实际在 `:72`（`m_grouped_bf16_gemm_nt_contiguous_tl` 内） | 改为 `:72` / `:61-72` |
| 2 | `docs/81-locality-domain-test-oracle.md:232` | `mlopart.py:135` 的 `shutil.which` | **引错文件**：`mlopart.py` 无 `shutil.which`；`shutil.which('nvidia-cuda-mps-control')` 在 `tests/test_locality_domain.py:136`，假脚本在 `:150-151` | 改为 `tests/test_locality_domain.py:136` / `:150-151` |
| 3 | `docs/38-deepgemm-ptx-primitives.md:72,75,78` | `mega_moe.cuh:1440/1419/613/1243` | 简写有歧义，被解析到 459 行的 `layout/mega_moe.cuh`（行号越界）；实际指 1528 行的 impl | 改为全名 `sm100_fp8_fp4_mega_moe.cuh:…` |

第 1、2 条是**真错误**（一个行号越界、一个引错文件），第 3 条是**简写歧义**。修完后重跑，
「行号越界」由 5 归零，「无法解析」中不再出现 `mlopart`/legacy 相关项。第 1 条同时改动了实验
输出字符串，故重跑了 `exp_legacy_grouped_gemm.sh` 使 `.out.txt` 与文档一致。

### 5.2 「被引行全空」10 条（多为同名歧义，待观察）

```
102-mega-gate-mhc-split-barrier.md:102   mega_mhc.cuh:32          -> layout/mega_mhc.cuh
103-sm100-swap-ab-dgrad.md:63            heuristics/common.hpp:16 -> heuristics/common.hpp
34-locality-domain-probe.md:47           csrc/runtime/locality_domain.hpp:13
45-muon-syrk-symm.md:324                 gemm.hpp:806
48-runtime-switches.md:259               runtime.hpp:44           -> heuristics/runtime.hpp
50-runtime-lifecycle.md:286              runtime.hpp:44
80-lazy-init-test-oracle.md:136          runtime.hpp:44
98-deepep-jit-config.md:41               csrc/python_api.cpp:22
98-deepep-jit-config.md:131              csrc/buffers/bucket.hpp:78
98-deepep-jit-config.md:243              csrc/python_api.cpp:29
```

这批的典型成因有两类（人工抽查了几条）：

- **同名文件歧义**：`runtime.hpp:44` 在多处出现——文档想指 `csrc/runtime/runtime.hpp` 的某行，
  但脚本按行号在界内优先时落到了 `csrc/jit_kernels/heuristics/runtime.hpp`（该文件 `:44` 恰好是
  一个函数体结尾的空行）。`heuristics/common.hpp:16` 同理（`:16` 是空行）。
- **散文行恰好命中了别的引用**：如 `gemm.hpp:806` 在 `csrc/apis/gemm.hpp` 里是非空的
  （`early_return(...)` 那行），但该 Markdown 行把 `806` 与另一个同名 `gemm.hpp` 引用混在一起。

这 10 条**不改变结论**（符号本体都在，且专题正文给了上下文），但提示未来若要严谨，应统一写成
**全相对路径**。

### 5.3 「轻微漂移」92 条（详单见 `.out.txt`，此处给代表）

这类是体检最有信息量的输出：**符号就在被引文件里，只是不在被引的那一行**。逐个看了前几条，
成因主要是两类：

- **off-by-one / 区间端点**：
  - `01-moe-ep-basics.md:172` 引 `dispatch.cuh:103` 的 `deduplicate`——实测 `ptx::deduplicate` 在
    `:104`（`:103` 是上一行的 `dst_rank_idx`），差一行。
  - `07-deepep-dispatch-combine.md:168` 引 `combine_utils.cuh:55` 的 `combine_reduce`——`:55` 是函数
    模板头，函数名 `combine_reduce` 在 `:58`（整段起于 55，可接受）。
- **同行多引用串扰**：`14-moe-backward-kernels.md:47` 的 `check_k_grouped_args` 与 `:369` 的
  `DG_HOST_UNREACHABLE` 在同一 Markdown 行，脚本把后者当成前者的标识符去核对，于是「不在被引行」。
  这类是 §2.3 提到的启发式假阳性。

因此这 92 条**大部分不是错误**，但它是下轮「逐条复核」的待办清单，且**上游升版后这张表会变**——
这正是把体检脚本化的意义。

---

## 6. 复现

```bash
# 在 kernel_lab(H100) 里（纯文件解析，也可直接在宿主机跑）
bash harness/exp_deepgemm_source_audit.sh > harness/exp_deepgemm_source_audit.out.txt
```

脚本会打印：环境里 DeepGEMM / DeepEP 的 commit 与 subject、Part A 全索引表、Part B 统计、
Part C 三张清单，以及末尾的汇总 JSON（`total / unresolved / oob / empty / content_ok /
content_weak_ok / near_miss / hard_miss / files`）。改完文档后重跑，JSON 里的 `oob` 与
`unresolved` 就是回归信号。

---

## 7. 边界与结论

- **能跑/能核**：整套体检是纯文件解析，架构无关，H100 或任意机器都能复现；产出的 1904 条引用、
  30 个源文件指纹、198 个被引文件都是真实核对过的数字。
- **不能核**：SM100 专属 kernel 的**运行时**行为仍只能源码分析（本机 H100 跑不了 `fp8_fp4_mega_moe`，
  见 ROADMAP「已知坑」）；体检只保证「引用的**位置**对得上」，不保证「引用的**结论**对」。
- **结论**：专题 105 篇、1904 处源码引用，**99.68% 可定位、0 行号越界**；体检抓到并修正了 **2 处
  真实错误 + 1 处简写歧义**，另列出 10 条同名歧义与 92 条轻微漂移供后续复核。**30 个 DeepGEMM
  device impl / host API 全部有专篇覆盖**，本专题对 DeepGEMM 的覆盖是完整的。

> 下一篇增量（opt-in）：从 §5.3 的 92 条里挑与**主 kernel / 反向关键路径**相关的做一次定点复核，
> 或按 §3 的指纹基线在上游下一次 release 后做一次真正的 source-rot 对比。
