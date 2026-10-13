#!/usr/bin/env python3
"""P118-1 实验：DeepGEMM 的 NCU PM-sampling 离线绘图工具 `scripts/quick_plot_pm.py`。

纯 host/Python，架构无关。按路径加载**真实脚本**（importlib），把它拆成四个可验证的层次：
  A. 数据结构与常量（MetricSpec / CATEGORY_ORDER / KIND_SUFFIXES / CURATED_METRICS）
  B. metric 名解析：`_metric_candidates` / `_parse_metric_values` / `_format_value`
  C. host 侧解析管线：`_probe_metric_series`（表列约定 idx=11）、`collect_metric_series`
     （KIND_SUFFIXES 回退序 + `len(values) > 1` 闸门）
  D. 真机 `ncu --query-metrics --chip gh100`：`resolve_metric_specs` 到底在 H100 上解析出多少、
     漏了哪些（SM100/TMEM/NVLink realtime 域专属），以及 `plot_pm` 无 report 时的合成渲染。
另记录一处源码漂移：注释引用的 `scripts/ncu-metrics.txt` 在仓库里并不存在。

运行：bash harness/exp_quick_plot_pm.sh > harness/exp_quick_plot_pm.out.txt
"""
import importlib.util
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

DG = Path(os.environ.get("DEEPGEMM_ROOT", os.path.expanduser("~/github/DeepGEMM")))
SCRIPT = DG / "scripts" / "quick_plot_pm.py"

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def load_module():
    spec = importlib.util.spec_from_file_location("quick_plot_pm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    print(f"DeepGEMM root : {DG}")
    print(f"script        : {SCRIPT}  (exists={SCRIPT.exists()})")
    print(f"python        : {sys.version.split()[0]}")
    import matplotlib
    import numpy
    print(f"matplotlib    : {matplotlib.__version__}   numpy: {numpy.__version__}")

    mod = load_module()

    # ---------------------------------------------------------------- Part A
    section("Part A 数据结构与常量")
    spec = mod.MetricSpec("X", "m", "sum", "SM")
    check("MetricSpec 字段", (spec.name, spec.metric, spec.kind, spec.category) == ("X", "m", "sum", "SM"))
    check("aliases 默认空元组", spec.aliases == () and isinstance(spec.aliases, tuple))
    check("frozen dataclass 不可改", _raises(lambda: setattr(spec, "name", "Y")))
    rs = mod.ResolvedMetricSpec("X", "m.sum", "sum", "SM")
    check("ResolvedMetricSpec 无 aliases 字段", not hasattr(rs, "aliases"))
    ms = mod.MetricSeries("X", "m.sum", "SM", "%", (1.0, 2.0))
    check("MetricSeries.values 是 tuple", isinstance(ms.values, tuple))

    check("CATEGORY_ORDER 顺序", mod.CATEGORY_ORDER == ["Overview", "SM", "L1", "L2", "DRAM", "Interconnect"])
    check("KIND_SUFFIXES 共 9 种 kind", len(mod.KIND_SUFFIXES) == 9)
    check("pct 有两个回退后缀", mod.KIND_SUFFIXES["pct"] == [".pct", ".avg.pct_of_peak_sustained_elapsed"])
    check("pct_peak 只有一个后缀", mod.KIND_SUFFIXES["pct_peak"] == [".avg.pct_of_peak_sustained_elapsed"])

    curated = mod.CURATED_METRICS
    print(f"  CURATED_METRICS 条数 = {len(curated)}")
    check("curated 全部是 MetricSpec", all(isinstance(s, mod.MetricSpec) for s in curated))
    bad_kind = [s.name for s in curated if s.kind not in mod.KIND_SUFFIXES]
    check("curated 的 kind 都在 KIND_SUFFIXES 里", bad_kind == [], f"bad={bad_kind}")
    bad_cat = [s.name for s in curated if s.category not in mod.CATEGORY_ORDER]
    check("curated 的 category 都在 CATEGORY_ORDER 里", bad_cat == [], f"bad={bad_cat}")
    aliased = [s.name for s in curated if s.aliases]
    print(f"  带 alias 的 curated: {aliased}")
    check("只有 C2C 一条带 alias", aliased == ["C2C Throughput"])
    cats = {}
    for s in curated:
        cats[s.category] = cats.get(s.category, 0) + 1
    print(f"  按 category 计数: {cats}")
    check("六个 category 都有条目", set(cats) == set(mod.CATEGORY_ORDER))

    # ---------------------------------------------------------------- Part B
    section("Part B metric 名解析与数值格式化")
    mc = mod._metric_candidates
    check("无 TriageCompute 标记 -> 原样", mc("lts__throughput") == ["lts__throughput"])
    check("带 .TriageCompute. -> 全名 + 去前缀名",
          mc("LTS.TriageCompute.lts__throughput") == ["LTS.TriageCompute.lts__throughput", "lts__throughput"])
    check("只切第一个 .TriageCompute.", mc("A.TriageCompute.B.TriageCompute.C") ==
          ["A.TriageCompute.B.TriageCompute.C", "B.TriageCompute.C"])

    pv = mod._parse_metric_values
    check("空串 -> ()", pv("") == ())
    check("'no data' -> ()", pv("no data") == ())
    check("裸标量 -> 单元素", pv("1234.5") == (1234.5,))
    check("千分位裸标量", pv("1,234.5") == (1234.5,))
    check("面板元组 (...)", pv("(1; 2; 3)") == (1.0, 2.0, 3.0))
    check("聚合前缀 x (...)", pv("avg (10; 20; 30)") == (10.0, 20.0, 30.0))
    check("元组内千分位", pv("Avg (1,000; 2,000)") == (1000.0, 2000.0))
    check("尾随空格/分号容错", pv("(1; 2; )") == (1.0, 2.0))
    check("非法文本 -> ()", pv("abc") == ())
    check("半截元组仍可解析", pv("foo (7; 8)") == (7.0, 8.0))

    fv = mod._format_value
    check("0 -> '0'", fv(0) == "0")
    check("0.25 -> '0.25'", fv(0.25) == "0.25")
    check("1.5 -> '1.5'", fv(1.5) == "1.5")
    check("1500 -> '1.50 K'", fv(1500) == "1.50 K")
    check("2.5e6 -> '2.50 M'", fv(2.5e6) == "2.50 M")
    check("3.2e9 -> '3.20 G'", fv(3.2e9) == "3.20 G")
    check("4e12 -> '4.00 T'", fv(4e12) == "4.00 T")
    check("负数保留符号", fv(-1500) == "-1.50 K")
    fwu = mod._format_with_unit
    check("带单位拼接", fwu(0.5, "%") == "0.50 %")
    check("空单位不加尾巴", fwu(1234.5, "") == "1.23 K")

    # ---------------------------------------------------------------- Part C
    section("Part C host 侧解析管线（表列约定 idx=11 + 回退序 + 闸门）")
    # 合成 ncu --csv 输出：header/units/row 三行，第 12 列(idx=11) 是我们要的 metric 实例值
    def synth_rows(metric_full, unit, raw, ncols=13):
        header = ["ID", "Process Name", "Process ID", "Host Name", "Kernel Name", "Kernel Time",
                  "Context", "Stream", "Block Size", "Grid Size", "Device", metric_full, "extra"][:ncols]
        units = ["", "", "", "", "", "", "", "", "", "", "unit-ish", unit, ""][:ncols]
        row = ["0", "python", "1", "host", "k", "1", "ctx", "7", "256", "128", "0", raw, ""][:ncols]
        return [header, units, row]

    calls = []

    def fake_run(rows):
        return lambda command, timeout: calls.append(list(command)) or rows

    saved = mod._run_csv_command
    try:
        mod._run_csv_command = fake_run(synth_rows("sm__cycles_active.avg", "%", "(10; 20; 30)"))
        r = mod._probe_metric_series("x.ncu-rep", "sm__cycles_active.avg")
        check("正常三行 -> (full_name, unit, values)", r == ("sm__cycles_active.avg", "%", (10.0, 20.0, 30.0)), f"got={r}")
        check("_probe_metric_series 走 ncu --import/--csv/--page raw",
              "--import" in calls[-1] and "--csv" in calls[-1] and "raw" in calls[-1]
              and "--print-metric-instances" in calls[-1])

        mod._run_csv_command = fake_run([["h"] * 13, ["u"] * 13])
        check("不足三行 -> None", mod._probe_metric_series("x", "m") is None)
        mod._run_csv_command = fake_run([["h"] * 11, ["u"] * 11, ["r"] * 11])
        check("表列 <= 11 -> None（idx=11 读不到）", mod._probe_metric_series("x", "m") is None)
        mod._run_csv_command = fake_run(None)
        check("命令返回 None -> None", mod._probe_metric_series("x", "m") is None)
    finally:
        mod._run_csv_command = saved

    # collect_metric_series：KIND_SUFFIXES 回退 + len(values)>1 闸门
    spec_pct = mod.ResolvedMetricSpec("P", "M", "pct", "SM")
    spec_sum = mod.ResolvedMetricSpec("S", "N", "sum", "L2")
    seen = []

    def fake_probe(report, metric_name):
        seen.append(metric_name)
        if metric_name == "M.pct":                       # 单值 -> 命中但被 len>1 闸门拒绝
            return ("M.pct", "%", (5.0,))
        if metric_name == "M.avg.pct_of_peak_sustained_elapsed":  # 第二个回退后缀 -> 接受
            return ("M.avg.pct_of_peak_sustained_elapsed", "%", (1.0, 2.0, 3.0))
        return None                                       # N.sum 恒 None -> skipped

    saved = mod._probe_metric_series
    try:
        mod._probe_metric_series = fake_probe
        collected, skipped = mod.collect_metric_series("x", [spec_pct, spec_sum])
    finally:
        mod._probe_metric_series = saved
    print(f"  probe 调用序: {seen}")
    check("回退序正确：先 .pct 再 .avg.pct...", seen == ["M.pct", "M.avg.pct_of_peak_sustained_elapsed", "N.sum"])
    check("收下 1 条、跳过 1 条", len(collected) == 1 and len(skipped) == 1)
    check("收下的是第二个后缀、name 用 display name",
          collected[0].name == "P" and collected[0].metric == "M.avg.pct_of_peak_sustained_elapsed")
    check("单值系列被 len>1 闸门挡在门外", all(s.metric != "M.pct" for s in collected))

    # ---------------------------------------------------------------- Part D
    section("Part D 真机 ncu --query-metrics（H100 / gh100）")
    available = mod._query_available_metrics("gh100")
    print(f"  gh100 可用 metric 数 = {len(available)}")
    check("gh100 查询到大量 metric", len(available) > 1000, f"n={len(available)}")
    for probe in ("sm__cycles_active", "dram__throughput", "lts__throughput", "sm__inst_executed_realtime"):
        check(f"gh100 含 {probe}", probe in available)

    raised = _raises(lambda: mod._query_available_metrics("__definitely_not_a_chip__"))
    check("非法 chip -> RuntimeError", raised, f"raised={raised}")

    resolved, missing = mod.resolve_metric_specs("gh100")
    print(f"  resolve_metric_specs('gh100'): resolved={len(resolved)} / missing={len(missing)} / total={len(curated)}")
    check("every resolved.metric ∈ available", all(r.metric in available for r in resolved))
    check("resolved + missing == curated", len(resolved) + len(missing) == len(curated))
    resolved_names = {r.name for r in resolved}
    missing_names = [s.name for s in missing]
    print("  H100 上缺失的 curated 指标：")
    for name in missing_names:
        print(f"    - {name}")
    check("缺失清单非空（Curated 面向 SM100 realtime 域）", len(missing) > 0)
    # 逐项核对：缺的确实没有任何候选在 gh100 里
    for s in missing:
        cands = []
        for m in (s.metric, *s.aliases):
            cands.extend(mod._metric_candidates(m))
        check(f"'{s.name}' 确无候选可用", not any(c in available for c in cands),
              f"cands={cands}")
    # TMEM 指标只在目标卡实时域
    tmem = next((s for s in curated if s.metric.endswith("sm__mem_tensor_cycles_active_realtime")), None)
    check("有一处 TMEM 实时域指标", tmem is not None)
    if tmem is not None:
        check("TMEM 指标在 H100 上缺失", tmem.name in missing_names)

    # ---------------------------------------------------------------- Part E
    section("Part E plot_pm：无 report 的合成渲染 + 空输入早退")
    series = [
        mod.MetricSeries("SM Throughput", "a.avg", "SM", "%", (10.0, 40.0, 80.0, 60.0)),
        mod.MetricSeries("L2 Throughput", "b.avg", "L2", "%", (5.0, 30.0, 90.0, 70.0)),
        mod.MetricSeries("Blocks Launched", "c.sum", "Overview", "", (1.0, 2.0, 3.0, 4.0)),
        mod.MetricSeries("DRAM Throughput", "d.avg", "DRAM", "%", (1.0, 2.0, 3.0, 4.0)),
    ]
    matplotlib.use("Agg")
    with tempfile.TemporaryDirectory() as td:
        report = str(Path(td) / "mega-moe-kk.3.ncu-rep")
        Path(report).write_bytes(b"")  # 占位，plot_pm 只用文件名
        mod.plot_pm(report, series, save=True)
        png = Path(report.replace(".ncu-rep", ".pm_sampling.png"))
        check("生成 PNG", png.exists() and png.stat().st_size > 1000,
              f"size={png.stat().st_size if png.exists() else 'NA'}")
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.plot_pm("x.ncu-rep", [], save=False)
    check("空 metrics 早退并提示", "No curated metrics" in buf.getvalue(), buf.getvalue().strip())

    # ---------------------------------------------------------------- Part F
    section("Part F 源码漂移与默认值核对")
    metrics_txt = DG / "scripts" / "ncu-metrics.txt"
    print(f"  注释引用的 scripts/ncu-metrics.txt 存在? {metrics_txt.exists()}")
    check("源码注释引用的 ncu-metrics.txt 在仓库缺失（漂移）", not metrics_txt.exists())
    # main() 的默认 report / 默认 chip（chip 是目标卡代号，按要求不打印字面量）
    import inspect
    src = inspect.getsource(mod.main)
    check("main 默认 report 名是 mega-moe-kk.3.ncu-rep", "mega-moe-kk.3.ncu-rep" in src)
    check("main 默认 chip 是目标卡代号（非 H100）", "gh100" not in src and "default=" in src)

    print()
    print("=" * 78)
    print(f"结果：{PASS} PASS / {FAIL} FAIL")
    print("=" * 78)
    return 1 if FAIL else 0


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    sys.exit(main())
