#!/usr/bin/env python3
"""MPK task 接线可达性审计（静态、只读）。

用途：对 mirage `mpk` 分支快照做一次「四层可达性」交叉审计，产出
`docs/49-mpk-wiring-audit.md` 里的所有数字。纯静态 grep/解析，不构建、不运行。

四层：
  L1 登记：`src/kernel/graph.cc` 的 `register_task` 大表（name -> TaskType）
  L2 前端：`python/mirage/mpk/persistent_kernel.py` 的 `*_layer` 发射的 register 名
  L3 编译单元：`tasks/{ampere,hopper,blackwell}/task_header.cuh` 的传递 include 闭包
  L4 生产者：`models/*/builder.py` 与 `demo/**` 对 `*_layer` 的调用

用法：
  python3 mpk_wiring_audit.py /tmp/mirage

（快照：mirage-project/mirage 分支 mpk，commit f9eb70c。）
"""

import os
import re
import sys
import glob


def strip_comments(text):
    """去掉行内 # 注释（够用；不处理字符串里的 #）。"""
    return "\n".join(line.split("#")[0] for line in text.splitlines())


def parse_graph_register(root):
    """L1: name -> (emit_fn, TaskType)。"""
    txt = open(os.path.join(root, "src/kernel/graph.cc")).read()
    blocks = re.split(r'else if \(name == "', txt)
    out = {}
    # 第一个 if（embedding）不在 else-if 链里，单独抓
    m = re.search(r'if \(name == "embedding"\)(.*?)else if', txt, re.S)
    if m:
        f = re.search(r'task_register->(register_\w+)\(', m.group(1))
        t = re.search(r'task_config\[op\][^;]*?(TASK_[A-Z0-9_]+)', m.group(1), re.S)
        out["embedding"] = (f.group(1), t.group(1))
    for b in blocks[1:]:
        name = b.split('"', 1)[0]
        fn = re.search(r'task_register->(register_\w+)\(', b)
        t = re.search(r'task_config\[op\][^;]*?(TASK_[A-Z0-9_]+)', b, re.S)
        out[name] = (fn.group(1) if fn else "?", t.group(1) if t else "?")
    return out


def parse_emit_symbols(root):
    """register_* 函数体里 code.e("kernel::SYM<...") 发射的设备符号集。"""
    txt = open(os.path.join(root, "src/kernel/task_register.cc")).read()
    return sorted(set(re.findall(r'code\.e\("kernel::([A-Za-z0-9_]+)', txt)))


def parse_layers(root):
    """L2: layer 名 -> (起行, 止行, 发射的 register 名列表[含关键字 task_name=])。"""
    p = os.path.join(root, "python/mirage/mpk/persistent_kernel.py")
    lines = open(p).read().splitlines()
    defs = []
    for i, l in enumerate(lines):
        m = re.match(r'\s*def (\w+_layer)\s*\(', l)
        if m:
            defs.append((i, m.group(1)))
    defs.append((len(lines), None))
    out = {}
    for k in range(len(defs) - 1):
        s, name = defs[k]
        e = defs[k + 1][0]
        body = strip_comments("\n".join(lines[s:e]))
        regs = re.findall(r'register_task\([^,]+,\s*"([^"]+)"', body)
        regs += re.findall(r'task_name\s*=\s*"([^"]+)"', body)
        # 三元表达式（内联或 task_name= 赋值）两个分支都抓
        for m in re.finditer(r'("[^"]+"\s*if[^,;)]+else\s*"[^"]+")', body):
            regs += re.findall(r'"([^"]+)"', m.group(1))
        for m in re.finditer(r'register_task\([^,]+,\s*("[^"]+"\s*if[^,)]+else\s*"[^"]+")', body):
            regs += re.findall(r'"([^"]+)"', m.group(1))
        out[name] = (s + 1, e, sorted(set(regs)))
    return out


def header_closure(root, header):
    """L3: 一个 task_header 的传递 include 闭包（相对 tasks/ 目录）。"""
    ROOT = os.path.join(root, "include/mirage/persistent_kernel")

    def resolve(cur, inc):
        for cand in (os.path.normpath(os.path.join(os.path.dirname(cur), inc)),
                     os.path.normpath(os.path.join(ROOT, inc)),
                     os.path.normpath(os.path.join(root, "include", inc))):
            if os.path.exists(cand):
                return cand
        return None

    seen, st = set(), [os.path.join(ROOT, header)]
    while st:
        cur = st.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            t = open(cur, errors="ignore").read()
        except OSError:
            continue
        for m in re.finditer(r'#include\s+"([^"]+)"', t):
            r = resolve(cur, m.group(1))
            if r:
                st.append(r)
    return {os.path.relpath(p, ROOT) for p in seen}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mirage"
    reg = parse_graph_register(root)
    layers = parse_layers(root)
    syms = parse_emit_symbols(root)
    HDRS = ["tasks/ampere/task_header.cuh",
            "tasks/hopper/task_header.cuh",
            "tasks/blackwell/task_header.cuh"]
    closures = {h: header_closure(root, h) for h in HDRS}
    prod = set().union(*closures.values())

    print(f"[L1] register_task 名字数 = {len(reg)}")
    uniq_tt = sorted({v[1] for v in reg.values()})
    print(f"[L1] 涉及 TaskType = {len(uniq_tt)}")
    print(f"[L2] *_layer 方法数 = {len(layers)}")
    print(f"[L3] 生产闭包（并集）文件数 = {len(prod)}；"
          + "，".join(f"{h.split('/')[1]}={len(closures[h])}" for h in HDRS))

    print("\n== [L2] 登记名从未被任何 layer 活跃发射 ==")
    emitted = set()
    for _, _, regs in layers.values():
        emitted |= set(regs)
    # 动态多卡（multigpu.py）
    mg = open(os.path.join(root, "python/mirage/mpk/multigpu.py")).read()
    emitted |= set(re.findall(r'register_task\([^,]+,\s*"([^"]+)"', mg))
    for n in sorted(set(reg) - emitted):
        print("   ", n, "->", reg[n])

    print("\n== [L3] device 头不在任何生产 task_header 闭包内 ==")
    allh = set()
    for p in glob.glob(os.path.join(root, "include/mirage/persistent_kernel/tasks/**/*.cuh"),
                       recursive=True) + \
             glob.glob(os.path.join(root, "include/mirage/persistent_kernel/tasks/**/*.h"),
                       recursive=True):
        allh.add(os.path.relpath(p, os.path.join(root, "include/mirage/persistent_kernel")))
    for h in sorted(allh - prod):
        print("   ", h)

    print("\n== [L3] 发射符号 -> 定义头（不在生产闭包者标 ORPHAN） ==")
    contents = {}
    for p in glob.glob(os.path.join(root, "include/mirage/persistent_kernel/tasks/**/*.cuh"),
                       recursive=True):
        contents[os.path.relpath(p, os.path.join(root, "include/mirage/persistent_kernel"))] = \
            open(p, errors="ignore").read()
    for s in syms:
        hits = sorted({h for h, t in contents.items() if re.search(r'\b' + re.escape(s) + r'\b', t)})
        if hits and not any(h in prod for h in hits):
            print(f"   ORPHAN {s:45s} -> {hits}")

    print("\n== [L4] layer 无生产模型 builder 调用 ==")
    bfiles = [p for p in glob.glob(os.path.join(root, "python/mirage/mpk/models/**/builder.py"),
                                   recursive=True)]
    btxt = "\n".join(strip_comments(open(f).read()) for f in bfiles)
    called = set(re.findall(r'mpk\.(\w+_layer)\(', btxt))
    dfiles = glob.glob(os.path.join(root, "demo/**/*.py"), recursive=True)
    dtxt = "\n".join(strip_comments(open(f).read()) for f in dfiles)
    dcalled = set(re.findall(r'\.(\w+_layer)\(', dtxt))
    for name, (s, e, regs) in sorted(layers.items(), key=lambda kv: kv[1][0]):
        if name not in called and not name.startswith("_"):
            print(f"   {name:45s} L{s}-{e} regs={regs} demo={'Y' if name in dcalled else 'n'}")


if __name__ == "__main__":
    main()
