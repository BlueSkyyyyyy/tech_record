# DeepGEMM 的 NCU PM-sampling 离线绘图工具：`scripts/quick_plot_pm.py`

> 本文对应路线图 **P118-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 128 篇。
> 第 78 篇（P78-1）扫「官方 profiling 基础设施」时只一句话带过 `scripts/quick_plot_pm.py`，
> 第 88 篇（P88-1）拆了 `run_ncu_mega_moe.sh` 的 NCU 编排，但**把报告变成一眼能读的时间序列图**
> 的这一步一直没被单独读过。本篇补上：这份 448 行、只依赖 `matplotlib`/`numpy`/标准库的脚本，
> 如何把 `ncu --import … --page raw --csv` 的输出整理成一张「按物理域分组的 PM-sampling 面板」。
>
> **可跑性**：`quick_plot_pm.py` 是**纯 host / Python，与 GPU 架构完全无关**；它唯一的真机依赖是
> `ncu --query-metrics`。本文在 H100(sm90) 上直接 `importlib` 加载**真实脚本**逐层验证
> （**73 PASS / 0 FAIL**），并用 `ncu 2026.1.1.0 --chip gh100` 跑真实指标查询——顺带量出
> 「这套 curated 指标面向**目标卡**的 realtime 域，在 H100 上有 11 条解析不到」。

---

## 1. PM sampling 是什么，为什么值得单独成篇

「Mega MoE 前向 / 反向」的性能证据在前面的篇章里主要来自两类工具：

| 工具 | 观测对象 | 采集者 | 粒度 | 对应篇章 |
| --- | --- | --- | --- | --- |
| `common/probe.cuh` | **内核内** per-warp `%clock64` 轨迹 | kernel 自己（JIT 打开 `DG_PROBE_ENABLE`） | 每条记录 8B，warp 级时间线 | 第 55 篇 |
| **NCU PM sampling** | **整卡各单元**的实时利用率/吞吐 | Nsight Compute 硬件采样器 | 固定时间间隔一个采样点 | 本篇 + 第 78/88 篇 |

两者互补：`probe.cuh` 回答「这一条 warp 在这件事上花了多久」，PM sampling 回答
「在整段 kernel 执行期间，SM / L1 / L2 / DRAM / NVLink 各自忙到什么程度、有没有阶段性起伏」。
对 Mega MoE 这种**通信与计算重叠**的 mega-kernel，后者尤其关键——它能看出
「前段 dispatch 拉数据时 DRAM/NVLink 在忙、后段 MMA 时 Tensor pipe 在忙」这种**时间上的分工**，
而单次 kernel 汇总的 SpeedOfLight 指标会把整段时间平均掉。

PM sampling 的报告（`.ncu-rep`）本身无法直接读，必须先导出成表再画图。
`quick_plot_pm.py` 干的就是这最后一步：**一份 curated 指标清单 + 一条 CSV 解析管线 + 一张分组面板**。

它和 `run_ncu_mega_moe.sh`（第 88 篇）的分工是：

```
run_ncu_mega_moe.sh  ──►  产出 myreport.ncu-rep（多进程 NCU 采集，含 PM sampling section）
quick_plot_pm.py     ──►  myreport.ncu-rep  ──►  myreport.pm_sampling.png
```

---

## 2. 数据模型：五个小结构（`quick_plot_pm.py:21-74`）

整个脚本围绕五个不可变结构展开，职责切得很干净：

| 结构 | 行号 | 字段 | 作用 |
| --- | --- | --- | --- |
| `MetricSpec` | `:21-27` | `name, metric, kind, category, aliases=()` | **人工 curated 条目**：展示名 + 一个候选 ncu metric 名 + 该用哪个后缀 + 归到哪一类 |
| `ResolvedMetricSpec` | `:30-34` | `name, metric, kind, category` | 在**具体 chip** 上「命中」后的条目（多一个 `aliases` 都没有，`metric` 已确定） |
| `MetricSeries` | `:38-42` | `name, metric, category, unit, values` | 从报告里取到的**时间序列**（`values` 是 tuple，`unit` 来自报告的 units 行） |
| `CATEGORY_ORDER` | `:47-54` | `["Overview","SM","L1","L2","DRAM","Interconnect"]` | 面板的**分组顺序**（同时也是分组标题的排序键） |
| `KIND_SUFFIXES` | `:57-74` | 9 种 `kind → [ncu 后缀候选]` | 「同一物理量在 NCU 里有多种聚合后缀」的**回退表** |

关键设计：`MetricSpec.metric` 存的是**未加后缀的基础名**，真正去查的是
`f"{spec.metric}{suffix}"`，`suffix` 从 `KIND_SUFFIXES[spec.kind]` 里按序尝试。
`KIND_SUFFIXES` 里有两种 kind 带了**两个候选**：

```python
"pct": [".pct", ".avg.pct_of_peak_sustained_elapsed"],
```

这是因为 NCU 的「占峰值百分比」在不同 metric 上一个是 `.pct`（已经算好的百分比）、
另一个只有原始计数需要先按峰值归一。脚本不预先判断，**运行时先试 `.pct`、不成就试
`.avg.pct_of_peak_sustained_elapsed`**（`collect_metric_series:233-240`）。

`MetricSpec.aliases` 唯一被使用的例子是 C2C 吞吐（`:118`）：

```python
MetricSpec("C2C Throughput", "TriageCompute.ctc__throughput", "pct_peak",
           "Interconnect", aliases=("TriageCompute.ctx__throughput",)),
```

> C2C（chip-to-chip）在不同世代硬件上分别叫 `ctc__` 与 `ctx__`，所以留了一个别名。

---

## 3. 45 条 curated 指标：三种「metric 名形态」

`CURATED_METRICS`（`:77-125`）共 **45 条**，按类别分布（真机计数）：

| category | Overview | SM | L1 | L2 | DRAM | Interconnect |
| --- | --- | --- | --- | --- | --- | --- |
| 条数 | 5 | 12 | 8 | 8 | 4 | 8 |

它们对应 NCU PM sampling 的「物理单元」视角：Overview（发射/占用）、SM（IPC、各条管线）、
L1、L2、DRAM、互连（NVLink / PCIe / C2C）。这正是分析 Mega MoE 时的六条主线索。

curated 名字有三种形态，理解它们才能读懂 `_metric_candidates`：

1. **带 `.TriageCompute.` 中缀的全名**，如
   `"TPC.TriageCompute.sm__inst_executed_realtime"`（`:88`）。
   `.TriageCompute.` 是 NCU 2026 把 metric 按「物理域前缀 + 分组域」重命名后的产物；
   同一物理量在**目标卡**上是 `TPC.TriageCompute.sm__inst_executed_realtime`，
   而在别的 chip 上可能就叫 `sm__inst_executed_realtime`。
2. **裸名**，如 `"dram__throughput"`（`:107`）、`"nvltx__bytes"`（`:120`）——
   这些名字在各代之间没变。
3. **带 `TriageCompute.` 前缀但没有物理域**，如 `"TriageCompute.ctc__throughput"`（`:118`）——
   只带分组域。

脚本对「带 `.TriageCompute.` 中缀」的形态做**双候选**处理（`_metric_candidates:156-161`）：

```python
def _metric_candidates(metric):
    candidates = [metric]
    marker = ".TriageCompute."
    if marker in metric:
        candidates.append(metric.split(marker, 1)[1])
    return candidates
```

即：**先试完整名，再试去掉 `X.TriageCompute.` 前缀后的裸名**。
`split(..., 1)` 只切第一个中缀（H100 真机验证：`"A.TriageCompute.B.TriageCompute.C"`
切出 `["A.TriageCompute.B.TriageCompute.C", "B.TriageCompute.C"]`）。

于是 `resolve_metric_specs`（`:164-181`）对每条 curated 展开成候选序列
（`metric` + 所有 `aliases`，每个再各自双候选），**取第一个在 `available` 集合里出现的**。
真机结果很能说明问题：

| curated 条目 | 在 H100 上命中的实际 metric | 说明 |
| --- | --- | --- |
| `Blocks Launched` | `FE_B.TriageCompute.gr__ctas_launched_realtime` | 保留**全名**（H100 也认 `TriageCompute` 域） |
| `SM Active Cycles` | `TPC.TriageCompute.sm__cycles_active` | 保留全名 |
| `DRAM Throughput` | `dram__throughput` | **去掉** `FBSP.TriageCompute.` 前缀（H100 只认裸名） |
| `NVLink Transmitted Throughput` | `nvltx__bytes` | 裸名 |

也就是说 `_metric_candidates` 的双候选是**为了同一份 curated 清单能同时适配多代 chip**，
而不是某一代专属。

---

## 4. host 侧解析管线：从 `ncu` 到 `MetricSeries`

### 4.1 三条命令封装

```python
def _run_csv_command(command, timeout):           # :126-132
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 and not result.stdout:
        return None
    reader = csv.reader(io.StringIO(result.stdout))
    return list(reader)
```

注意那个**看似矛盾的返回条件**：`returncode != 0` **且** `stdout` 为空才 `None`。
因为 `ncu --import` 在导出的表里若某条 metric 没有采样点，会**在 stderr 打印告警但仍返回非零**，
此时 stdout 里的表头/单位行仍有用——脚本选择「只要还有 stdout 就继续解析」。
真机验证：命令直接返回 `None`（模拟无 stdout）→ `_probe_metric_series` 返回 `None`。

### 4.2 两步确定体系

`_query_available_metrics(chip)`（`:134-153`）跑 `ncu --query-metrics --chip <chip>`，
逐行取第一个 token，**只保留含 `__` 的**（过滤掉分隔线/说明），返回一个 `set`。
`--chip` 非法时抛 `RuntimeError`（真机用假 chip 名验证过）。

`resolve_metric_specs(chip)`（`:164-181`）用这个集合把 45 条 curated 一分为二：
`resolved`（能在该 chip 上解析到具体 metric 名）与 `missing`（解析不到）。
**这一步完全不需要报告文件**，只查 chip 能力——所以它可以在 H100 上真跑（见 §6）。

### 4.3 数值解析：一个正则都不用的「三种格式」

`_parse_metric_values(raw)`（`:184-198`）把 `--print-metric-instances values` 打印的原始串
拆成浮点元组，它要对付 NCU 表格第 12 列出现的三种形态：

| 原始串 | 形态 | 结果 |
| --- | --- | --- |
| `"1234.5"` / `"1,234.5"` | 裸标量（千分位逗号） | `(1234.5,)` |
| `"(1; 2; 3)"` | 面板元组（分号分隔） | `(1.0, 2.0, 3.0)` |
| `"avg (10; 20; 30)"` | 聚合前缀 + 元组 | `(10.0, 20.0, 30.0)` |
| `"no data"` / `""` | 空 | `()` |
| `"abc"` | 解析不动 | `()`（`except ValueError` 兜底） |

实现是纯字符串技巧：先判 `raw` 是否被一对括号包住；否则若有 `" ("` 就从那里切开、去掉尾括号；
再统一 `replace(",", "")` 去千分位、按 `;` 分列。**一个 `re` 都没用**。
真机验证还发现一个「宽容」点：半截元组 `"foo (7; 8)"` 也会被解析成 `(7.0, 8.0)`——
因为它含 `" ("` 且以 `)` 收尾，`rstrip(")")` 后正好是 `"7; 8"`。

### 4.4 表列约定 `idx == 11`（最脆弱的一环）

`_probe_metric_series(report, metric_name)`（`:201-224`）真正去 `--import` 报告取一个 metric：

```python
rows = _run_csv_command(["ncu", "--import", report, "--page", "raw", "--csv",
                         "--metrics", metric_name, "--print-metric-instances", "values"], timeout=60)
if not rows or len(rows) < 3 or len(rows[0]) <= 11:
    return None
header, units, row = rows[0], rows[1], rows[2]
unit = units[11] if len(units) > 11 else ""
raw  = row[11]  if len(row) > 11 else ""
values = _parse_metric_values(raw)
return header[11], unit, values
```

三处硬约定：

1. **表结构固定为三行**：`rows[0]` 表头、`rows[1]` 单位、`rows[2]` 数据。行数不足 3 直接 `None`。
2. **目标列写死为第 12 列（`idx == 11`, 0-based）**。少了这一列（`len(rows[0]) <= 11`）就放弃。
3. 取的是**该列的第一条数据行**（`rows[2]`），而不是遍历所有实例——因为 PM sampling 的
   时间序列本身就在一个「面板元组」里（`"(t0; t1; …)"`），一个 metric 一行就够。

真机用**合成三行表**逐条验证：正常三行 → 返回 `(full_name, unit, values)`；
只有两行 → `None`；表头只有 11 列 → `None`。

### 4.5 回退与闸门：`collect_metric_series`（`:227-248`）

```python
for spec in resolved_specs:
    series = None
    for suffix in KIND_SUFFIXES[spec.kind]:          # ① 后缀回退
        probe = _probe_metric_series(report, f"{spec.metric}{suffix}")
        if probe is None:
            continue
        full_metric, unit, values = probe
        if len(values) > 1:                          # ② 闸门：单点不算时间序列
            series = MetricSeries(spec.name, full_metric, spec.category, unit, values)
            break
    ...
```

两级策略：

- **后缀回退**（`KIND_SUFFIXES` 的多候选）：第一个后缀取不到就试下一个。
- **`len(values) > 1` 闸门**：只有**两个及以上采样点**才算「时间序列」；
  单点数据对「看随时间的起伏」毫无意义，直接跳过（于是会继续尝试下一个后缀）。

真机合成用例把这个顺序钉死了：对 `kind="pct"` 的 spec，先调 `M.pct` 返回**单值**
（`(5.0,)`，被闸门拒），再调 `M.avg.pct_of_peak_sustained_elapsed` 返回**三点**
（被接受）——最终 `collected[0].metric` 是**第二个后缀**，而 `name` 仍是人看的 `display name`。

---

## 5. `plot_pm`：把时间序列画成「按物理域分组的面板」（`:274-417`）

渲染本身不是本节重点，但有两个值得记录的设计：

**① 共享 x 轴的竖向叠图 + 类别色带。** `plot_pm` 用 `matplotlib.gridspec.GridSpec`
把每条 metric 画成一个「窄条」（`row_h=0.55` 英寸、`plot_w=14` 英寸、`label_w=3.6`），
所有条共享同一 x 轴（`Sample Index`），因此能直接上下对齐看「同一采样时刻各单元在干什么」。
每条用 `fill_between + plot` 画成半透明面积图，颜色由 `category` 决定
（Overview 灰蓝、SM 蓝、L1 青、L2 橙、DRAM 红、Interconnect 紫，`:288-295`）。
同一 `category` 的第一条会在左上角打一个分组标题（`:388-395`），
排序键是 `category_rank`（`:298`）——**面板顺序严格按 `CATEGORY_ORDER`，不是按 curated 顺序**。

**② y 轴对百分比特判。** 若 `unit == "%"` 就固定 `ylim=(0,100)`，否则 `(0, max*1.15)`（`:338-343`）；
y 刻度只取上下两端（`:347-348`），右侧用 `_format_value` 标出**峰值**（`:351-359`）。
`_format_value`（`:251-266`）按 1e12/1e9/1e6/1e3 分档成 `T/G/M/K`，
且 `value == 0` 特判返回 `"0"`（否则会显示成 `0.00`）。

输出路径写死在报告名旁（`:404-408`）：`report.replace(".ncu-rep", ".pm_sampling.png")`，
`--interactive` 时改用 `QtAgg` 弹窗、否则 `Agg` 存 PNG。

> 真机验证：用 4 条合成序列喂给 `plot_pm(..., save=True)`，在临时目录生成 55 KB 的 PNG；
> 空 `metrics` 时打印 `"No curated metrics had time-series data in the report."` 并**直接返回**
> （不尝试画空图）。

---

## 6. H100 真机：这套 curated 清单到底面向哪一代

用 `ncu 2026.1.1.0` 对 `--chip gh100`（H100）跑真实查询：

- `_query_available_metrics("gh100")` 返回 **3728** 个含 `__` 的 metric；
- `resolve_metric_specs("gh100")`：**resolved = 34 / missing = 11**（45 条 curated）。

H100 上**解析不到**的 11 条（逐条确认「完整名 + 去前缀名 + aliases」都查无此 metric）：

| 缺失条目 | 候选 metric 名 |
| --- | --- |
| Average / Total Blocks Active | `tpc__ctas_active_realtime` |
| SM FMA / FMA Heavy / FMA Light Pipe Throughput | `sm__pipe_fma{,heavy,lite}_cycles_active_realtime` |
| **SM TMEM Pipe Throughput** | `sm__mem_tensor_cycles_active_realtime` |
| SysL2 Throughput to Peer / System Memory | `syslts__t_sector_throughput_aperture_{peer,sysmem}` |
| DRAM Read / Write Throughput | `dram__read_throughput` / `dram__write_throughput` |
| C2C Throughput | `ctc__throughput` / `ctx__throughput` |

其中最能说明「面向目标卡」的一条是 **SM TMEM Pipe Throughput**：TMEM 是
**目标卡（SM100）专属**的张量内存，Mega MoE 的 UMMA 累加器就住在那里（第 10/35/36 篇），
H100 根本没有这个单元，自然没有对应实时域指标。其余缺失项也多是新世代才拆细的 realtime 域
（FMA 管线细分、Read/Write 分开的 DRAM 吞吐、C2C）——这正是 P78/P88 里「面向目标卡 profiling」
的落点：**这份 curated 清单是为目标卡的 PM sampling 采样器设计的**。

而在 H100 上仍能解析到 34 条，覆盖了 Overview / SM（IPC、ALU、Tensor）/ L1 / L2 / DRAM
总量 / NVLink 收发 / PCIe——也就是说**这份工具在 H100 上依然可用**，
只是看不到 TMEM 等目标卡专属单元。别名机制也真机验证有效：`C2C Throughput` 在 H100 上
两条候选都不存在 → 正确落入 `missing`。

---

## 7. 边界与源码漂移

1. **注释引用的 `scripts/ncu-metrics.txt` 在仓库里不存在。** `:66` 的注释写
   「Curated from scripts/ncu-metrics.txt」，但真机确认该文件缺失——curated 清单是**内联**的，
   那句注释是历史遗留的漂移。
2. **`idx == 11` 依赖 NCU CSV 的列布局**。一旦 Nsight Compute 改表结构，解析会静默变成 `None`
   （而非报错），表现为「图里少了几条」而不是崩——排查时要注意。
3. **`main` 的默认 `--chip` 是目标卡代号**（`:423`），默认 report 名是
   `mega-moe-kk.3.ncu-rep`（`:422`，与第 88 篇的 Mega MoE NCU 工作流一致）。
   按本专题的约定，目标卡代号一律不写进文档，故此处只指出「默认值是目标卡代号，H100 上要显式传 `--chip gh100`」。
4. **`_query_available_metrics` 超时 30s、`_run_csv_command` 超时 60s**（`:139/:215`）：
   对超大报告导入慢的场景是硬上限，超时抛异常而非降级。
5. **`_parse_metric_values` 的 `except ValueError` 把「解析失败」与「空数据」都归成 `()`**——
   两者在下游都表现为「这条 metric 被跳过」，所以一个格式异常不会中断整张图。

---

## 8. 小结

`scripts/quick_plot_pm.py` 是 DeepGEMM profiling 闭环的**最后一公里**：

- **输入契约**：`ncu --import <rep> --page raw --csv --metrics <m> --print-metric-instances values`
  的三行表，目标列固定 `idx == 11`；
- **多代适配**：45 条 curated × `_metric_candidates` 双候选 × `aliases` × `KIND_SUFFIXES` 回退，
  让**同一份清单**在目标卡和 H100 上都能解析（H100 解析 34/45）；
- **成图语义**：按 `CATEGORY_ORDER` 分组、共享 x 轴、`unit=="%"` 固定 0–100——
  正好服务「Mega MoE 通信/计算重叠在时间上如何分工」的读图需求；
- **边界**：面向目标卡的 realtime 域（TMEM/FMA 细分/C2C 在 H100 缺失），以及一处
  `ncu-metrics.txt` 的注释漂移。

与第 55 篇的内核内 `%clock64` 探针合起来，本专题把 DeepGEMM 的**两套性能观测手段**
（warp 级时间线 vs 整卡单元时间序列）都拆到了行号级。

---

## 附：源码与真机坐标

- 源码：`~/github/DeepGEMM@057ca59`（`v2.1.1.post3-24-g057ca59`，PR #462）
  `scripts/quick_plot_pm.py`（448 行）：
  `:1-9` 文件 docstring / `:21-42` 三个 dataclass / `:47-54` `CATEGORY_ORDER` /
  `:57-74` `KIND_SUFFIXES` / `:77-125` `CURATED_METRICS`（45 条）/
  `:126-132` `_run_csv_command` / `:134-153` `_query_available_metrics` /
  `:156-161` `_metric_candidates` / `:164-181` `resolve_metric_specs` /
  `:184-198` `_parse_metric_values` / `:201-224` `_probe_metric_series` /
  `:227-248` `collect_metric_series` / `:251-266` `_format_value` /
  `:268-271` `_format_with_unit` / `:274-417` `plot_pm` / `:420-448` `main`。
- 真机：H100(sm90)、`ncu 2026.1.1.0`、python 3.12.3、matplotlib 3.10.8、numpy 2.1.0。
- 实验：`harness/exp_quick_plot_pm.{sh,py,out.txt}`（**73 PASS / 0 FAIL**）。
- 线上资料：NVIDIA Nsight Compute 文档
  <https://docs.nvidia.com/nsight-compute/NsightCompute/index.html>（PM sampling / `--page raw` 导出）；
  DeepGEMM 仓库 <https://github.com/deepseek-ai/DeepGEMM>。
