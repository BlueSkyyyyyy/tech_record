"""P115-1 实验：DeepGEMM 的 Python 类型桩生成器 `scripts/generate_pyi.py`。

目标源码（`~/github/DeepGEMM@057ca59`，`deep_gemm.__version__ == 2.8.1`）：
  * `scripts/generate_pyi.py`（890 行，纯 stdlib，`re` + `pathlib`）
      - `build_cpp_function_index:5-94`（正则扫 `csrc/**` 的 `.cpp/.hpp/...` 建「函数名→签名」索引）
      - `BracketTracker:97-148`（**四种括号的嵌套计数**，`<>` 仅当不在 `()[]{}` 内才算模板括号）
      - `extract_m_def_statements:151-222`（逐行扫 `m.def(...)`，括号配平取多行完整语句）
      - `parse_m_def_statement:225-342`（拆分 `m.def` 顶层实参：字符串名 / `&fn` / `py::arg` 默认值）
      - `extract_cpp_signature_from_content:345-386`（**死代码**：定义但全仓库无调用点）
      - `parse_mdef_and_attach_cpp_signatures:389-414`（把索引里的 C++ 签名挂回每条 m.def）
      - `parse_cpp_signature:417-462`（拆返回类型 / 参数表，函数名取 `(` 前最后一个 token）
      - `parse_parameter_declaration:494-578`（**从右向左、顶层识别**拆「类型 + 形参名」）
      - `cpp_type_to_python_type:608-696`（C++ 类型 → Python 注解；模板分支 ORDER MATTERS）
      - `split_template_args:699-720` / `cpp_default_to_python_default:723-778`
      - `generate_pyi_function:781-825` / `generate_pyi_file_content:828-869` / `generate_pyi_file:872-890`
  * 调用点 `setup.py:19/120/136-146`（`generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')`
    再拷进 wheel 的 `deep_gemm/_C.pyi`）。

本脚本**纯 host / Python、架构无关**，用 importlib 按路径加载**真实** `generate_pyi.py`，逐条验证其行为：
  A. 端到端确定性与 gold：跑两遍逐位一致；若存在 `stubs/_C.pyi`（`.gitignore:21` 忽略的构建产物）则对拍；
  B. `extract_m_def_statements`：真机 62 条；整行注释 `// m.def` 跳过、**行内尾注释仍被采集**；
     `m.attr` 别名**不计入**；
  C. `parse_m_def_statement`：函数指针 vs lambda、`&ns::foo` 归一化、`py::arg` 默认值索引；
  D. `build_cpp_function_index`：真机 290 条；首定义胜（`if base_name not in func_index`）；
  E. `BracketTracker`：`>`/`>>`/`->`/括号内 `<`/移位 `<<` 的语义边界；
  F. `cpp_type_to_python_type`：映射表 + 嵌套 `optional<tuple<...>>` + 两条「过宽正则」反例；
  G. `cpp_default_to_python_default`：字面量 / `nullopt` / `make_tuple` / `tuple<>({...})` / 未识别；
  H. `parse_cpp_signature` + `parse_parameter_declaration`：真机 `fp8_fp4_mega_moe` 的 16 参拆解；
  I. **别名缺口**：`m.attr` 造的 10 个公开别名（`fp8_gemm_nt` 等）在 `.pyi` 里**没有**；
  J. `default_args` 是「`py::arg` 序位」而消费端按「C++ 形参序位」取——当前巧合对齐，构造错位反例；
  K. 死代码 `extract_cpp_signature_from_content`。

运行：bash harness/exp_generate_pyi.sh
"""
import hashlib
import importlib.util
import os
import re
import tempfile
from pathlib import Path

FAILS = []


def check(label, cond, extra=''):
    tag = '[PASS]' if cond else '[FAIL]'
    if not cond:
        FAILS.append(label)
    print(f'  {tag} {label} {extra}')


def head(section):
    print(f'\n== {section} ==')


DG_ROOT = Path(os.environ.get('DEEPGEMM_ROOT', Path.home() / 'github' / 'DeepGEMM'))


def load_generator():
    path = DG_ROOT / 'scripts' / 'generate_pyi.py'
    spec = importlib.util.spec_from_file_location('generate_pyi', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import warnings
    warnings.simplefilter('ignore')

    g = load_generator()
    print(f'DeepGEMM root : {DG_ROOT}')
    print(f'generator     : {DG_ROOT / "scripts" / "generate_pyi.py"}')
    print(f'csrc root     : {DG_ROOT / "csrc"}')
    print(f'__version__   : '
          f'{re.search(r"__version__ = .([0-9.]+).", (DG_ROOT / "deep_gemm" / "__init__.py").read_text()).group(1)}')

    # ---------------------------------------------------------------- A
    head('A. 端到端：确定性与 gold（.pyi 逐位）')
    with tempfile.TemporaryDirectory() as td:
        o1 = Path(td, 'r1')
        o2 = Path(td, 'r2')
        g.generate_pyi_file(name='_C', root=str(DG_ROOT / 'csrc'), output_dir=str(o1))
        g.generate_pyi_file(name='_C', root=str(DG_ROOT / 'csrc'), output_dir=str(o2))
        p1, p2 = o1 / '_C.pyi', o2 / '_C.pyi'
        s1 = sha256_of(p1)
        s2 = sha256_of(p2)
        content = p1.read_text()
        n_def = len(re.findall(r'^def ', content, re.M))
        n_lines = len(content.splitlines())
        print(f'  generated lines={n_lines} funcs={n_def} sha={s1[:16]}')
        check('A1 两次生成逐位一致（确定性）', s1 == s2, f'{s1[:12]}=={s2[:12]}')
        check('A2 函数条数 == csrc 里 m.def 数（62）', n_def == 62, f'got {n_def}')
        check('A3 import 行 = "from typing import Any, Optional"',
              content.splitlines()[2] == 'from typing import Any, Optional',
              f'got {content.splitlines()[2]!r}')
        check('A4 只 import torch、不 import numpy',
              'import torch' in content and 'import numpy' not in content)
        gold = DG_ROOT / 'stubs' / '_C.pyi'
        if gold.exists():
            same = sha256_of(gold) == s1
            check('A5 与仓库既有 stubs/_C.pyi（构建产物）逐位一致', same,
                  f'gold sha={sha256_of(gold)[:12]}')
        else:
            print('  (skip A5: stubs/_C.pyi 不存在，跳过既有产物对拍)')

    # ---------------------------------------------------------------- B
    head('B. extract_m_def_statements（扫 m.def）')
    res = g.extract_m_def_statements(str(DG_ROOT / 'csrc'))
    stmts = [s for it in res for s in it['m_def_statements']]
    files = {Path(it['file']).name: len(it['m_def_statements']) for it in res}
    check('B1 真机共 62 条 m.def', len(stmts) == 62, f'got {len(stmts)}')
    check('B2 分布在 9 个 apis/*.hpp', len(res) == 9, f'{files}')
    # 注释行里的 m.def 不会被当成语句：构造同目录临时 cpp
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / 'x.cpp').write_text(
            'void f() {\n'
            '  // m.def("commented", &g);\n'
            '  m.def("real", &g);\n'
            '}\n')
        r2 = g.extract_m_def_statements(td)
        got = [s for it in r2 for s in it['m_def_statements']]
        check('B3 整行注释 `// m.def` 被跳过，只取真实调用',
              len(got) == 1 and '"real"' in got[0], f'got {got}')
    # 行内尾注释里的 m.def 仍会被采集（只跳过整行注释）
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / 'x.cpp').write_text(
            'void f() {\n'
            '  int x = 0;  // m.def("inline_fake", &g);\n'
            '  m.def("real", &g);\n'
            '}\n')
        r3 = g.extract_m_def_statements(td)
        got3 = [s for it in r3 for s in it['m_def_statements']]
        check('B4 **边界**：行内尾注释 `...; // m.def(` 仍被当成语句采集',
              len(got3) == 2, f'got {len(got3)}: {got3}')
    # m.attr 不计入
    src_all = ''.join(f.read_text(errors='ignore')
                      for f in (DG_ROOT / 'csrc').rglob('*.hpp'))
    n_attr = len(re.findall(r'm\.attr\s*\(', src_all))
    check('B5 `m.attr` 别名（20 处调用 / 10 个别名）不被采集', n_attr == 20, f'm.attr( 出现 {n_attr} 次')

    # ---------------------------------------------------------------- C
    head('C. parse_m_def_statement（拆顶层实参）')
    by_name = {}
    for s in stmts:
        p = g.parse_m_def_statement(s)
        by_name[p['python_function_name']] = p
    fp = by_name['fp8_fp4_gemm_nt']
    check('C1 fp8_fp4_gemm_nt：非 lambda + cpp 名归一化',
          (not fp['is_lambda']) and fp['cpp_function_name'] == 'fp8_fp4_gemm_nt')
    check('C2 fp8_fp4_gemm_nt：num_args == 11（py::arg 计 11 个）',
          fp['num_args'] == 11, f"got {fp['num_args']}")
    check('C3 fp8_fp4_gemm_nt：默认值索引 {3..10}',
          sorted(fp['default_args']) == [3, 4, 5, 6, 7, 8, 9, 10],
          f"got {sorted(fp['default_args'])}")
    check('C4 compiled_dims 默认 = "\\"nk\\""', fp['default_args'][7] == '"nk"')
    check('C5 init 是 lambda（`[&](...)`）→ is_lambda=True',
          by_name['init']['is_lambda'] is True)
    check('C6 mega_moe 无 py::arg → num_args=0、无默认值',
          by_name['fp8_fp4_mega_moe']['num_args'] == 0 and
          by_name['fp8_fp4_mega_moe']['default_args'] == {})
    # &ns::foo 归一化
    p_ns = g.parse_m_def_statement('m.def("x", &deep_gemm::gemm::foo, py::arg("a"))')
    check('C7 `&deep_gemm::gemm::foo` → cpp 名只留末段 `foo`',
          p_ns['cpp_function_name'] == 'foo', f"got {p_ns['cpp_function_name']}")
    # lambda
    p_lam = g.parse_m_def_statement('m.def("y", [&](int a) { return a; })')
    check('C8 lambda 起始 `[` → is_lambda=True 且 cpp 名 None',
          p_lam['is_lambda'] and p_lam.get('cpp_function_name') is None)
    # 缺参数
    try:
        g.parse_m_def_statement('m.def("z")')
        check('C9 参数不足 2 个应抛 ValueError', False)
    except ValueError as e:
        check('C9 参数不足 2 个应抛 ValueError', 'insufficient' in str(e), str(e))

    # ---------------------------------------------------------------- D
    head('D. build_cpp_function_index（建 C++ 签名索引）')
    idx = g.build_cpp_function_index(str(DG_ROOT / 'csrc'))
    check('D1 真机索引 290 条', len(idx) == 290, f'got {len(idx)}')
    check('D2 命中 fp8_fp4_mega_moe 且返回 `static void`',
          idx.get('fp8_fp4_mega_moe', '').startswith('static void fp8_fp4_mega_moe('))
    check('D3 命中 batched_syrk（单行声明）',
          idx.get('batched_syrk') == 'static void batched_syrk(const torch::Tensor& a, const torch::Tensor& d)',
          repr(idx.get('batched_syrk')))
    # 首定义胜
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / 'a.hpp').write_text('int foo(int a);\nint foo(double b);\n')
        i2 = g.build_cpp_function_index(td)
        check('D4 同名多声明只保留首条（`if base_name not in func_index`）',
              i2.get('foo') == 'int foo(int a)', repr(i2.get('foo')))

    # ---------------------------------------------------------------- E
    head('E. BracketTracker（四括号嵌套语义）')

    def state(s):
        t = g.BracketTracker()
        for c in s:
            if c in '()[]{}<>':
                t.update(c)
        return t.is_top_level(), (t.paren, t.bracket, t.brace, t.angle)

    check('E1 `std::vector<std::pair<int, int>>` 顶层（`>>` 两次闭环）',
          state('std::vector<std::pair<int, int>>')[0] is True)
    check('E2 `->` 里的 `>` 不计数（angle=0）', state('a->b') == (True, (0, 0, 0, 0)))
    check('E3 括号内 `<`（比较）不误当模板：`foo(a<b)` 顶层',
          state('foo(a<b)') == (True, (0, 0, 0, 0)))
    check('E4 比较 `a > b` 的 `>` 不计数', state('a > b') == (True, (0, 0, 0, 0)))
    check('E5 **边界**：顶层移位 `a << b` 被误算成 2 层模板（非顶层）',
          state('a << b') == (False, (0, 0, 0, 2)), f'got {state("a << b")}')

    # ---------------------------------------------------------------- F
    head('F. cpp_type_to_python_type（类型映射）')
    cases = {
        'void': 'None',
        'torch::Tensor': 'torch.Tensor',
        'const torch::Tensor&': 'torch.Tensor',
        'int': 'int', 'const int&': 'int', 'int64_t': 'int', 'size_t': 'int',
        'bool': 'bool', 'float': 'float', 'double': 'float',
        'std::string': 'str', 'const char*': 'str',
        'std::pair<torch::Tensor, torch::Tensor>': 'tuple[torch.Tensor, torch.Tensor]',
        'std::tuple<torch::Tensor, torch::Tensor>': 'tuple[torch.Tensor, torch.Tensor]',
        'std::vector<int64_t>': 'list[int]',
        'std::optional<torch::Tensor>': 'Optional[torch.Tensor]',
        'std::optional<std::tuple<torch::Tensor, torch::Tensor>>':
            'Optional[tuple[torch.Tensor, torch.Tensor]]',
        'const std::shared_ptr<EpilogueClass>&': 'Any',
    }
    ok = True
    for k, v in cases.items():
        got = g.cpp_type_to_python_type(k)
        if got != v:
            ok = False
            print(f'    mismatch {k!r}: got {got!r} want {v!r}')
    check(f'F1 {len(cases)} 个映射全部正确', ok)
    check('F2 **过宽正则**：`std::variant<int, std::tuple<int, int>>` 落到 `int`',
          g.cpp_type_to_python_type('std::variant<int, std::tuple<int, int>>') == 'int',
          repr(g.cpp_type_to_python_type('std::variant<int, std::tuple<int, int>>')))

    # ---------------------------------------------------------------- G
    head('G. cpp_default_to_python_default（默认值映射）')
    dcases = {
        '"nk"': '"nk"', 'false': 'False', 'true': 'True',
        'std::nullopt': 'None', 'nullptr': 'None', 'NULL': 'None',
        'std::make_tuple(1, 1, 128)': '(1, 1, 128)',
        'std::tuple<int, int, int>({1, 1, 32})': '(1, 1, 32)',
        'std::vector<int>({1,2,3})': '[1,2,3]',
        '128': '128', '1.5': '1.5',
    }
    ok = True
    for k, v in dcases.items():
        got = g.cpp_default_to_python_default(k)
        if got != v:
            ok = False
            print(f'    mismatch {k!r}: got {got!r} want {v!r}')
    check(f'G1 {len(dcases)} 个默认值映射全部正确', ok)
    check('G2 未识别默认值回退 None（并告警）',
          g.cpp_default_to_python_default('SomeMacroX') == 'None')

    # ---------------------------------------------------------------- H
    head('H. parse_cpp_signature / parse_parameter_declaration')
    ps = g.parse_cpp_signature(idx['fp8_fp4_mega_moe'])
    check('H1 返回类型取 `(` 前除末 token 外全部：`static void`',
          ps['return_type'] == 'static void', repr(ps['return_type']))
    check('H2 16 个形参', ps['num_parameters'] == 16, f"got {ps['num_parameters']}")
    names = [p['name'] for p in ps['parameters']]
    types = [p['type'] for p in ps['parameters']]
    check('H3 形参名与源码一致（y, l1_weights_tuple, ...）',
          names[:3] == ['y', 'l1_weights_tuple', 'l2_weights_tuple'], f'{names[:4]}')
    check('H4 类型剥离 `const`/`&`：第 4 参 = const std::optional<...>',
          types[3] == 'const std::optional<std::tuple<torch::Tensor, torch::Tensor>>', repr(types[3]))
    check('H5 parse_parameter_declaration 拆 `const std::string& name`',
          g.parse_parameter_declaration('const std::string& name') ==
          {'type': 'const std::string', 'name': 'name'})
    check('H6 无参名返回 name=None',
          g.parse_parameter_declaration('std::vector<int>')['name'] is None)

    # ---------------------------------------------------------------- I
    head('I. 别名缺口：m.attr 造的公开名不在 .pyi 里')
    with tempfile.TemporaryDirectory() as td:
        g.generate_pyi_file(name='_C', root=str(DG_ROOT / 'csrc'), output_dir=td)
        content = Path(td, '_C.pyi').read_text()
    aliases_lhs = re.findall(r'm\.attr\("([^"]+)"\)\s*=', src_all)
    check('I1 gemm.hpp 里 10 个别名 LHS 名', len(set(aliases_lhs)) == 10, f'{sorted(set(aliases_lhs))}')
    missing = [a for a in set(aliases_lhs) if f'def {a}(' not in content]
    check('I2 所有 10 个别名都缺 stub（含最常用的 fp8_gemm_nt）',
          len(missing) == 10, f'missing {len(missing)}/{len(set(aliases_lhs))}')
    check('I3 但被别名的原名 `fp8_fp4_gemm_nt` 有 stub',
          'def fp8_fp4_gemm_nt(' in content)

    # ---------------------------------------------------------------- J
    head('J. default_args 的「py::arg 序位 vs C++ 形参序位」语义')
    check('J1 当前对齐：py_args 数 == C++ 形参数（fp8_fp4_gemm_nt 11==11）',
          fp['num_args'] == len(g.parse_cpp_signature(idx['fp8_fp4_gemm_nt'])['parameters']))
    # 反例：若 py::arg 少写前两个，默认值会错位挂到前两个 C++ 形参
    stmt = ('m.def("k", &foo, py::arg("c") = 1, py::arg("d") = std::nullopt)')
    p = g.parse_m_def_statement(stmt)
    check('J2 反例：默认值索引相对 py::arg 而非 C++ 形参（{0,1}）',
          sorted(p['default_args']) == [0, 1], f'{sorted(p["default_args"])}')

    # ---------------------------------------------------------------- K
    head('K. 死代码审计')
    gen_src = (DG_ROOT / 'scripts' / 'generate_pyi.py').read_text()
    n_calls = len(re.findall(r'\bextract_cpp_signature_from_content\s*\(', gen_src))
    check('K1 `extract_cpp_signature_from_content` 只有定义、无调用（== 1 次出现）',
          n_calls == 1, f'{n_calls} 次（含定义）')

    # ---------------------------------------------------------------- 汇总
    print('\n' + '=' * 60)
    if FAILS:
        print(f'RESULT: {len(FAILS)} FAIL')
        for f in FAILS:
            print('  -', f)
    else:
        print('RESULT: ALL PASS')
    print('=' * 60)
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
