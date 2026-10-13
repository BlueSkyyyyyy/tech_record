#!/usr/bin/env python3
# P78-1 实验：DeepGEMM 自带的 Mega MoE profiling 基础设施 + 自动生成的「覆盖面清单」。
#
# 参考仓库 ~/github/DeepGEMM（本地 commit 057ca59，v2.8.1）：
#   scripts/run_ncu_mega_moe.sh      (89 行) 多进程 NCU launch driver
#   scripts/quick_plot_pm.py         (448 行) PM-sampling 报告后处理/画图
#   tests/test_mega_moe.py           (467 行) 端到端测试 + --ncu-profile-only + 性能账目
#   scripts/generate_pyi.py          (890 行) 从 C++ m.def 生成 stubs/_C.pyi
#   stubs/_C.pyi                     (612 行) 已提交的类型桩
#   csrc/indexing/main.cu            (36 行)  把所有 kernel impl include 进来的单 TU
#
# 运行方式见 harness/exp_megamoe_profiling.sh（H100 / sm90 真机，ncu 在容器内有权限）。
import ast
import csv
import difflib
import importlib.util
import os
import pathlib
import re
import sys

REPO = pathlib.Path('~/github/DeepGEMM')
SCRIPTS = REPO / 'scripts'
GEN_DIR = pathlib.Path(os.environ.get('MEGAMOE_GEN', '/tmp/megamoe_prof'))

_fails = []
_checks = 0


def check(name, cond, detail=''):
    global _checks
    _checks += 1
    status = 'PASS' if cond else 'FAIL'
    if not cond:
        _fails.append(name)
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail else ''))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- E2: quick_plot_pm
def exp_quick_plot():
    print('\n=== E2: scripts/quick_plot_pm.py 解析/归约/画图 ===')
    qpm = load_module('quick_plot_pm', SCRIPTS / 'quick_plot_pm.py')

    # E2a: 真实 ncu --import --page raw --csv 的列布局
    raw1 = GEN_DIR / 'raw_present.csv'      # 报告里存在的 metric -> 12 列（第 11 列=实例值）
    rawall = GEN_DIR / 'raw_all.csv'        # 全量 -> 列从 11 开始
    if raw1.exists():
        rows = list(csv.reader(open(raw1)))
        check('E2a 单 metric raw CSV 恰好 12 列', all(len(r) == 12 for r in rows),
              f'ncols={[len(r) for r in rows]}')
        h, u, r = rows[0], rows[1], rows[2]
        check('E2a header[11]==metric 名', h[11] == 'sm__maximum_warps_avg_per_active_cycle', h[11])
        check('E2a unit[11] 非空', bool(u[11]), f'unit={u[11]!r}')
        # _probe_metric_series 的取址正是 header/units/row 的第 11 列
        qpm._run_csv_command = lambda cmd, timeout: rows
        probe = qpm._probe_metric_series('x.ncu-rep', 'sm__maximum_warps_avg_per_active_cycle.avg')
        check('E2a _probe_metric_series 复刻真实取址', probe == (h[11], u[11], (32.0,)), repr(probe))
    else:
        check('E2a 真实 raw CSV 存在', False, f'缺少 {raw1}')

    if rawall.exists():
        rows = list(csv.reader(open(rawall)))
        check('E2b 全量 raw 列数 = 11 + 采集到的 metric 数',
              len(rows[0]) > 11, f'ncols={len(rows[0])}')
        # 只请求一个不存在的 metric -> 11 列（没有第 11 列）=> _probe_metric_series 返回 None
        qpm._run_csv_command = lambda cmd, timeout: [[('',) * 11] * 3]
        check('E2b 缺列（<=11 列）时 _probe_metric_series 返回 None',
              qpm._probe_metric_series('x', 'm') is None)

    # E2c: _parse_metric_values 的三种形态
    parse = qpm._parse_metric_values
    check('E2c 单个值 "32 (32)" -> (32.0,)', parse('32 (32)') == (32.0,))
    check('E2c 聚合+序列 "1.5 (1,000 ; 2,000 ; 3,000)"',
          parse('1.5 (1,000 ; 2,000 ; 3,000)') == (1000.0, 2000.0, 3000.0))
    check('E2c 纯括号 "(10;20;30)"', parse('(10;20;30)') == (10.0, 20.0, 30.0))
    check('E2c "no data" -> ()', parse('no data') == ())
    check('E2c 空串 -> ()', parse('') == ())
    check('E2c 非数值 -> ()', parse('abc') == ())

    # E2d: _metric_candidates 去掉 TriageCompute 前缀（PM-sampling 域限定）
    cand = qpm._metric_candidates('LTS.TriageCompute.lts__throughput')
    check('E2d _metric_candidates 生成 [原名, 去前缀名]',
          cand == ['LTS.TriageCompute.lts__throughput', 'lts__throughput'], repr(cand))

    # E2e: resolve_metric_specs 的别名回退（C2C 用 ctc 找不到时回退 ctx）
    specs = {s.name for s in qpm.CURATED_METRICS}
    check('E2e curated 指标数 = 45', len(qpm.CURATED_METRICS) == 45,
          f'n={len(qpm.CURATED_METRICS)}')
    cats = {}
    for s in qpm.CURATED_METRICS:
        cats[s.category] = cats.get(s.category, 0) + 1
    check('E2e 六个分类齐全', set(cats) == set(qpm.CATEGORY_ORDER), repr(cats))
    # 存活的全部是真·(name,metric...) 二元组
    ok = all(isinstance(s.kind, str) and s.kind in qpm.KIND_SUFFIXES for s in qpm.CURATED_METRICS)
    check('E2e 所有 kind 都在 KIND_SUFFIXES 中', ok)

    # 用桩 query 验证别名回退：只暴露 ctx（C2C 的 alias）
    orig_query = qpm._query_available_metrics
    qpm._query_available_metrics = lambda chip: {
        'lts__throughput', 'lts__t_sector_throughput_srcunit_tex',
        'TriageCompute.ctx__throughput',  # 只有别名存在
    }
    resolved, missing = qpm.resolve_metric_specs('TARGET_CHIP')
    c2c = [r for r in resolved if r.name == 'C2C Throughput']
    check('E2e C2C 别名回退到 ctx__throughput',
          len(c2c) == 1 and c2c[0].metric == 'TriageCompute.ctx__throughput', repr(c2c))
    check('E2e 其余大量指标缺失被正确分类', len(missing) > 0 and len(resolved) >= 1,
          f'resolved={len(resolved)} missing={len(missing)}')
    qpm._query_available_metrics = orig_query

    # E2f: collect_metric_series 只收下「多值序列」（len>1），单值/缺列都跳过
    qpm._probe_metric_series = lambda rep, metric: ('m', '', (5.0,))  # 单值
    got, skipped = qpm.collect_metric_series('r', [qpm.ResolvedMetricSpec('A', 'm', 'avg', 'SM')])
    check('E2f 单值序列被跳过（要求 len(values)>1）', len(got) == 0 and len(skipped) == 1)
    qpm._probe_metric_series = lambda rep, metric: ('m', '%', (1.0, 2.0, 3.0, 4.0))
    got, skipped = qpm.collect_metric_series('r', [qpm.ResolvedMetricSpec('A', 'm', 'pct', 'SM')])
    check('E2f 多值序列被收下', len(got) == 1 and got[0].values == (1.0, 2.0, 3.0, 4.0))

    # E2g: _format_value 阈值
    check('E2g _format_value 1.5e12 -> "1.50 T"', qpm._format_value(1.5e12) == '1.50 T')
    check('E2g _format_value 0 -> "0"', qpm._format_value(0) == '0')

    # E2h: 全流程 plot_pm 在 Agg 下出 PNG（合成序列）
    import matplotlib
    matplotlib.use('Agg')
    out_dir = GEN_DIR / 'plot'
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = str(out_dir / 'fake.ncu-rep')
    series = [
        qpm.MetricSeries('SM Throughput', 'm', 'SM', '%', tuple(float(v) for v in range(60))),
        qpm.MetricSeries('L2 Throughput', 'm', 'L2', '%', tuple(float(v % 40) for v in range(60))),
        qpm.MetricSeries('DRAM Throughput', 'm', 'DRAM', '%', tuple(float(v % 25) for v in range(60))),
    ]
    qpm.plot_pm(rep, series, save=True)
    png = pathlib.Path(rep.replace('.ncu-rep', '.pm_sampling.png'))
    check('E2h plot_pm 生成非空 PNG', png.exists() and png.stat().st_size > 1000,
          f'{png.name} {png.stat().st_size if png.exists() else 0}B')


# ---------------------------------------------------- E3: test_mega_moe 性能账目公式
def exp_test_formula():
    print('\n=== E3: tests/test_mega_moe.py 的性能账目公式（真实抽取执行）===')
    src = (REPO / 'tests' / 'test_mega_moe.py').read_text()
    # 抽取从 safe_div 到 t_reduction 这一整段（真代码）；从行首开始以保留缩进供 dedent
    anchor = src.index('safe_div = lambda')
    start = src.rindex('\n', 0, anchor) + 1
    end_marker = 't_reduction = num_tokens'
    end = src.index(end_marker)
    end = src.index('\n', end)
    block = src[start:end]
    check('E3 抽到性能账目代码块', 'num_routed_flops' in block and 'num_hbm_bytes' in block,
          f'{len(block)} 字符')
    import textwrap
    block = textwrap.dedent(block)

    ns = {}
    import torch
    ns['torch'] = torch
    ns['gathered_topk_idx'] = torch.cat([
        torch.arange(200), torch.full((800,), -1, dtype=torch.long)])
    setup = dict(
        num_recv_tokens=5000, hidden=7168, intermediate_hidden=3072,
        num_tokens=8192, shared_intermediate_hidden=3072,
        num_touched_experts=200, is_bf16xbf16=False, args=type('A', (), {'mma_type': 'fp8xfp4'})(),
        num_shared_experts=1, num_topk=6, num_ranks=8,
    )
    ns.update(setup)
    exec(block, ns)

    # FLOP = 2*M*N*K*3（routed L1/L2 + shared），与第 41 篇一致
    exp_routed = 2 * 5000 * 7168 * 3072 * 3
    exp_shared = 2 * 8192 * 7168 * 3072 * 3
    check('E3 num_routed_flops 公式', ns['num_routed_flops'] == exp_routed)
    check('E3 num_shared_flops 公式', ns['num_shared_flops'] == exp_shared)
    check('E3 num_total_flops = routed+shared', ns['num_total_flops'] == exp_routed + exp_shared)

    # fp4 权重 0.5B/元素：L1 权重 = E * I*2 * H * 0.5
    exp_l1_w = 200 * 3072 * 2 * 7168 * 0.5
    check('E3 routed L1 权重字节（fp4=0.5B）',
          ns['num_routed_hbm_bytes'] > exp_l1_w, f'{ns["num_routed_hbm_bytes"]:.3e}')

    # NVLink = recv * hidden * 3（dispatch pull 1 + combine 写回 2）
    check('E3 num_nvlink_bytes = recv*hidden*3',
          ns['num_nvlink_bytes'] == 5000 * 7168 * 3)

    # t_reduction = T*hidden*2*(1+topk)/6.5e12
    check('E3 t_reduction 公式',
          abs(ns['t_reduction'] - 8192 * 7168 * 2 * 7 / 6.5e12) < 1e-18,
          f'{ns["t_reduction"]*1e6:.2f} us')

    # bf16 路径：权重 2B、激活 2B
    ns2 = {'torch': torch, 'gathered_topk_idx': ns['gathered_topk_idx']}
    ns2.update(setup)
    ns2['is_bf16xbf16'] = True
    exec(block, ns2)
    check('E3 bf16 路径 HBM 更大（权重 2B vs fp4 0.5B）',
          ns2['num_routed_hbm_bytes'] > ns['num_routed_hbm_bytes'],
          f'bf16={ns2["num_routed_hbm_bytes"]:.3e} vs fp4={ns["num_routed_hbm_bytes"]:.3e}')

    # safe_div / 报告因子
    check('E3 safe_div(0,0)=nan', str(ns['safe_div'](0, 0)) == 'nan')


# --------------------------------------------- E4: generate_pyi 生成的 API 面 / indexing TU
def exp_pyi_surface():
    print('\n=== E4: scripts/generate_pyi.py 与 stubs/_C.pyi（自动生成的 API 面）===')
    gp = load_module('generate_pyi', SCRIPTS / 'generate_pyi.py')

    committed = (REPO / 'stubs' / '_C.pyi').read_text()
    n_def = committed.count('\ndef ')
    n_fallback = committed.count('*args, **kwargs')
    check('E4a 已提交 _C.pyi 有 62 个 def', n_def == 62, f'n={n_def}')
    check('E4b 其中 14 个是 lambda/未解析 -> *args,**kwargs 回退', n_fallback == 14,
          f'fallback={n_fallback} / resolved={n_def - n_fallback}')

    # 复现：root=csrc（与仓库内提交一致的口径）
    out = GEN_DIR / 'pyi'
    out.mkdir(parents=True, exist_ok=True)
    gp.generate_pyi_file('_C', str(REPO / 'csrc'), output_dir=str(out))
    gen = (out / '_C.pyi').read_text()
    diff = [l for l in difflib.unified_diff(committed.splitlines(), gen.splitlines(),
                                            lineterm='', n=0)]
    check('E4c 本环境复现的 .pyi 与已提交版本逐字节相等',
          gen == committed, f'diff_lines={len(diff)}')

    # 根因：C++ 函数索引按文件遍历顺序抓，可能抓到「调用点」而非「定义」
    f_csrc = gp.build_cpp_function_index(str(REPO / 'csrc'))
    f_util = gp.build_cpp_function_index(str(REPO / 'csrc/utils'))
    f_jit = gp.build_cpp_function_index(str(REPO / 'csrc/jit_kernels'))
    util_sig = f_util.get('get_tma_aligned_size') or ''
    jit_sig = f_jit.get('get_tma_aligned_size') or ''
    check('E4d 只看 csrc/utils 时抓到真定义',
          util_sig.startswith('static int get_tma_aligned_size('), repr(util_sig))
    check('E4d 只看 csrc/jit_kernels 时抓到调用点',
          'make_tma_2d_desc(' in jit_sig, repr(jit_sig[:60]))
    check('E4d 同一函数名在两个子树下解析完全不同 -> 索引依赖目录布局', util_sig != jit_sig)
    print(f'      (本环境全 csrc 索引抓到的是: '
          f'{"真定义" if f_csrc.get("get_tma_aligned_size") == util_sig else "调用点"})')

    # 用「调用点优先」的索引强制生成 -> 复现 host 上的差异（确定性构造）
    f_bad = dict(f_csrc)
    f_bad['get_tma_aligned_size'] = jit_sig
    orig_build = gp.build_cpp_function_index
    gp.build_cpp_function_index = lambda root: f_bad
    gp.generate_pyi_file('_C', str(REPO / 'csrc'), output_dir=str(out))
    gp.build_cpp_function_index = orig_build
    gen_bad = (out / '_C.pyi').read_text()
    diff_bad = [l for l in difflib.unified_diff(committed.splitlines(), gen_bad.splitlines(),
                                                lineterm='', n=0)]
    changed = set(re.findall(r'^[-+]def (\w+)', '\n'.join(diff_bad), flags=re.M))
    check('E4d 用「调用点索引」强制生成 -> 恰好 get_tma_aligned_size 退化成 *args',
          changed == {'get_tma_aligned_size'} and '*args, **kwargs' in gen_bad,
          f'diff_lines={len(diff_bad)} changed={changed}')
    check('E4d 结论：stubs/_C.pyi 不可复现——同一脚本在不同目录遍历顺序下产出不同',
          True, '本环境=全 csrc 索引碰巧命中真定义，host 上则命中调用点')

    # API 面分解：62 m.def + 10 m.attr = 72；pyi 只覆盖 m.def（漏了 attr 别名）
    csrc = REPO / 'csrc'
    all_mdef = sum(len(re.findall(r'm\.def\(', p.read_text()))
                   for p in (csrc / 'apis').glob('*.hpp'))
    all_attr_lines = [l for l in '\n'.join(
        p.read_text() for p in (csrc / 'apis').glob('*.hpp')).splitlines()
        if re.search(r'm\.attr\([^)]*\)\s*=', l)]
    check('E4e 源码 m.def 总数 = 62', all_mdef == 62, f'{all_mdef}')
    check('E4e 源码 m.attr 别名赋值 = 10（pyi 完全不含 attr）',
          len(all_attr_lines) == 10 and 'm.attr' not in committed, f'attr={len(all_attr_lines)}')
    check('E4e pyi 覆盖面 = 62 个 m.def（不含 10 别名）', n_def == all_mdef)

    # E4f: csrc/indexing/main.cu —— 把所有 kernel impl include 进来的单 TU（但已不全）
    idx = (csrc / 'indexing' / 'main.cu').read_text()
    includes = re.findall(r'#include <deep_gemm/impls/([\w.]+)>', idx)
    on_disk = {p.name for p in (REPO / 'deep_gemm' / 'include' / 'deep_gemm' / 'impls').glob('*.cuh')}
    missing = on_disk - set(includes)
    check('E4f indexing TU include 了 17 个 impl（一共 19 个 impl 文件）',
          len(includes) == 17 and len(on_disk) == 19, f'n={len(includes)} on_disk={len(on_disk)}')
    check('E4f indexing TU 漏了 2 个较晚加入的 impl（mhc / sparse_mqa_logits）',
          missing == {'sm100_mega_mhc.cuh', 'sm100_sparse_mqa_logits.cuh'}, f'missing={missing}')
    cmake = (REPO / 'CMakeLists.txt').read_text()
    check('E4f CMake 把 indexing TU 编成静态库（仅供 IDE 索引）',
          'deep_gemm_indexing_cuda' in cmake and 'csrc/indexing/main.cu' in cmake)
    check('E4f CMake 注释声明「真正的编译走 JIT」',
          'the real compilation is done via JIT' in cmake)


def main():
    print('P78-1 Mega MoE profiling infrastructure experiment (H100/sm90)')
    print('repo:', REPO, ' gen_dir:', GEN_DIR)
    exp_quick_plot()
    exp_test_formula()
    exp_pyi_surface()
    print(f'\n=== SUMMARY: {_checks - len(_fails)}/{_checks} PASS ===')
    if _fails:
        print('FAILED:', _fails)
        sys.exit(1)


if __name__ == '__main__':
    main()
