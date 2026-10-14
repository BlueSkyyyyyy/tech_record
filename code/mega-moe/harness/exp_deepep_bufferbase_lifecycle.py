#!/usr/bin/env python3
# P111-1 实验 driver：DeepEP 四个 Buffer 家族共享的生命周期契约 BufferBase。
#
# 为什么 H100(sm90) 可跑：
#   * BufferBase 的 Python 侧（deep_ep/buffers/base.py）只 import `abc`，不碰 CUDA/NCCL；
#   * C++ 侧（csrc/buffers/base.hpp）全是 host 代码，只用到 pybind11 的注册宏与 comm::Context
#     的六个只读字段，因此可以在临时目录搭一棵"影子 include 树"——真实 base.hpp 原样拷入，
#     用最小 stub 顶掉 pybind11/pybind11.h 和 csrc/kernels/comm/api.hpp——再 g++ 编译运行。
#   * `import deep_ep` 会触发 check_nccl_so()（本机 NCCL 2.29.7 与编译期不符 -> abort），
#     故 Python 侧按文件路径直接加载 base.py，绕过包引导；四个派生家族的 C++ 构造函数需要
#     真实 NCCL GIN（本机不可用），故 C++ 用一个同构的 DerivedBuffer 承接 base.hpp 的真实逻辑。
#
# 用法：bash harness/exp_deepep_bufferbase_lifecycle.sh > harness/exp_deepep_bufferbase_lifecycle.out.txt
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback

DEEPEP = os.environ.get("DEEPEP_ROOT", os.path.expanduser("~/github/DeepEP"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CXX_SRC = os.path.join(ROOT, "src", "bufferbase_lifecycle.cpp")

PASS = 0
FAIL = 0
RESULTS = []


def check(ok, name, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append(("PASS", name, detail))
        print(f"[PASS] {name}  {detail}")
    else:
        FAIL += 1
        RESULTS.append(("FAIL", name, detail))
        print(f"[FAIL] {name}  {detail}")


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Part 1: C++ 侧 —— 真实 csrc/buffers/base.hpp
# ---------------------------------------------------------------------------

PYBIND_STUB = r"""
#pragma once
// P111-1 影子 stub：只提供 base.hpp 的 register_apis 用到的最小 pybind11 面。
namespace pybind11 {
struct module_ {};
template <typename... Ts>
struct class_ {
    template <typename... A> explicit class_(A&&...) {}
    template <typename... A> class_& def_readonly(A&&...) { return *this; }
    template <typename... A> class_& def(A&&...) { return *this; }
};
}  // namespace pybind11
"""

COMM_API_STUB = r"""
#pragma once
// P111-1 影子 stub：真实 csrc/kernels/comm/api.hpp 会拉 NCCL/devcomm 头（本机版本不符），
// 这里只补 base.hpp 需要的 EP_HOST_ASSERT 宏与 comm::Context 的六个只读字段。
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#define EP_HOST_ASSERT(cond)                                                        \
    do {                                                                            \
        if (!(cond))                                                                \
            throw std::runtime_error(std::string("EP_HOST_ASSERT failed: ") + #cond); \
    } while (0)

namespace deep_ep::comm {
struct Context {
    int rank_idx = 0, num_ranks = 1;
    int64_t num_workspace_bytes = 0;
    int64_t num_gpu_buffer_bytes = 0;
    int64_t num_rdma_storage_bytes = 0;
    bool use_cpu_rdma_storage = false;
};
}  // namespace deep_ep::comm
"""


def part_cpp():
    print("\n===== Part 1: C++ 生命周期契约（真实 csrc/buffers/base.hpp）=====")
    real_base = os.path.join(DEEPEP, "csrc", "buffers", "base.hpp")
    print(f"[INFO] real base.hpp = {real_base}  sha256[:16]={sha16(real_base)}")
    print(f"[INFO] cpp harness   = {CXX_SRC}")

    shim = tempfile.mkdtemp(prefix="bb_shim_")
    os.makedirs(os.path.join(shim, "pybind11"), exist_ok=True)
    os.makedirs(os.path.join(shim, "csrc", "kernels", "comm"), exist_ok=True)
    os.makedirs(os.path.join(shim, "csrc", "buffers"), exist_ok=True)
    open(os.path.join(shim, "pybind11", "pybind11.h"), "w").write(PYBIND_STUB)
    open(os.path.join(shim, "csrc", "kernels", "comm", "api.hpp"), "w").write(COMM_API_STUB)
    shutil.copy2(real_base, os.path.join(shim, "csrc", "buffers", "base.hpp"))
    copied = os.path.join(shim, "csrc", "buffers", "base.hpp")
    check(sha16(copied) == sha16(real_base), "1.0 影子树里的 base.hpp 与真实文件逐字节相同",
          f"sha={sha16(copied)}")

    exe = os.path.join(shim, "harness")
    cmd = ["g++", "-std=c++17", "-O0", "-Wall", "-I", shim, CXX_SRC, "-o", exe]
    print("[INFO] " + " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        check(False, "1.1 g++ 编译通过", f"rc={cp.returncode}")
        return
    check(True, "1.1 g++ 编译通过（-Wall 无 error）")

    rp = subprocess.run([exe], capture_output=True, text=True)
    print("----- C++ harness stdout -----")
    print(rp.stdout, end="")
    if rp.stderr.strip():
        print("----- C++ harness stderr -----")
        print(rp.stderr, end="")
    print("------------------------------")

    for line in rp.stdout.splitlines():
        m = re.match(r"\[(PASS|FAIL)\] (.*)", line)
        if m:
            check(m.group(1) == "PASS", "1.cpp " + m.group(2))

    # 泄漏告警文本：只有场景 B（explicitly_destroy=True 且用户没 destroy）该出现一次。
    warn = "`destroy()` is not called before DeepEP Derived buffer destruction"
    n_warn = rp.stdout.count(warn)
    check(n_warn == 1, "1.2 泄漏告警恰好出现一次（仅场景 B）", f"count={n_warn}")
    check(rp.returncode == 0, "1.3 C++ harness 退出码为 0", f"rc={rp.returncode}")
    shutil.rmtree(shim, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 2: Python 侧 —— 真实 deep_ep/buffers/base.py
# ---------------------------------------------------------------------------

BASE_TESTS = r'''
import importlib.util
spec = importlib.util.spec_from_file_location("_deepep_base_real", BASE_PY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
BufferBase = mod.BufferBase


class FakeRuntime:
    def __init__(self):
        self.destroyed = False
        self.main_context = "CTX"
        self.pmu_calls = 0

    def destroy(self):
        self.destroyed = True

    def print_memory_usage(self):
        self.pmu_calls += 1


class FakeBuffer(BufferBase):
    # 与四个真实家族同构：__init__ 调 super().__init__，随后建 self.runtime；
    # destroy() 覆盖后调 super().destroy()，再清自己的字段。
    def __init__(self, explicitly_destroy):
        super().__init__(explicitly_destroy)
        self.runtime = FakeRuntime()

    def destroy(self):
        super().destroy()
        self.runtime = None


# A0: 抽象类不可直接实例化（destroy 是 abstractmethod）。
try:
    BufferBase(True)
    A0 = False
except TypeError:
    A0 = True
print(f"RESULT A0 {int(A0)} base.destroy_abstract={getattr(BufferBase.destroy, '__isabstractmethod__', None)}")

# A1: B main_context
b = FakeBuffer(True)
print(f"RESULT A1 {int(b.main_context == 'CTX')}")

# A2: 默认（False）调用 destroy() -> assert explicitly_destroy 失败。
b_false = FakeBuffer(False)
try:
    b_false.destroy()
    A2 = False
except AssertionError:
    A2 = True
print(f"RESULT A2 {int(A2)} runtime_destroyed={int(b_false.runtime.destroyed)}")

# A3: 显式模式 destroy() -> 委托 runtime.destroy()、runtime 置 None、main_context 属性变 None。
b.destroy()
print(f"RESULT A3 {int(b.runtime is None)} ctx_after={int(b.main_context is None)}")

# A4: 二次 destroy() -> Python 侧幂等（runtime 已 None，直接 no-op，不抛）。
double_ok = True
try:
    b.destroy()
except Exception:
    double_ok = False
print(f"RESULT A4 {int(double_ok)}")

# A5: print_memory_usage 在 runtime=None 时撞 assert（base.py:18）。
pmu_threw = False
try:
    b.print_memory_usage()
except AssertionError:
    pmu_threw = True
print(f"RESULT A5 {int(pmu_threw)}")

# A6: 清 runtime 之前 print_memory_usage 会转发到 runtime。
b2 = FakeBuffer(True)
b2.print_memory_usage()
print(f"RESULT A6 {int(b2.runtime.pmu_calls == 1)}")
'''


def part_py():
    print("\n===== Part 2: Python 生命周期契约（真实 deep_ep/buffers/base.py）=====")
    base_py = os.path.join(DEEPEP, "deep_ep", "buffers", "base.py")
    print(f"[INFO] real base.py = {base_py}  sha256[:16]={sha16(base_py)}")
    src = textwrap.dedent(BASE_TESTS).replace("BASE_PY", repr(base_py))
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr.strip():
        print(proc.stderr, end="")

    res = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"RESULT (\S+) (\d+)(.*)", line)
        if m:
            res[m.group(1)] = (m.group(2) == "1", m.group(3).strip())

    def expect(key, name):
        ok, detail = res.get(key, (False, "missing"))
        check(ok, name, detail)

    expect("A0", "2.0 抽象基类不可直接实例化（destroy 带 @abstractmethod）")
    expect("A1", "2.1 main_context 属性委托 self.runtime.main_context")
    expect("A2", "2.2 默认模式显式 destroy() 撞 assert self.explicitly_destroy（且未销毁 runtime）")
    expect("A3", "2.3 显式模式 destroy() 委托 runtime.destroy() 并把 runtime 置 None")
    expect("A4", "2.4 二次 destroy() 在 Python 侧幂等（与 C++ 侧抛异常形成对照）")
    expect("A5", "2.5 销毁后 print_memory_usage 撞 assert self.runtime is not None")
    expect("A6", "2.6 存活期 print_memory_usage 转发到 runtime")


# ---------------------------------------------------------------------------
# Part 3: 源解析 —— 四个家族的 pybind 继承链 + Python re-export + destroy 覆盖
# ---------------------------------------------------------------------------

def part_source():
    print("\n===== Part 3: 四个 Buffer 家族的契约接缝（源解析）=====")
    families = [
        ("EPBuffer", "csrc/buffers/ep.hpp", "deep_ep/buffers/ep.py"),
        ("EngramBuffer", "csrc/buffers/engram.hpp", "deep_ep/buffers/engram.py"),
        ("PPBuffer", "csrc/buffers/pp.hpp", "deep_ep/buffers/pp.py"),
        ("BucketBuffer", "csrc/buffers/bucket.hpp", "deep_ep/buffers/bucket.py"),
    ]
    chain_ok = True
    for cls, hpp, py in families:
        hpp_txt = open(os.path.join(DEEPEP, hpp)).read()
        py_txt = open(os.path.join(DEEPEP, py)).read()
        # C++：class X : public BufferBase  +  pybind11::class_<X, BufferBase>
        cpp_inherit = re.search(rf"class {cls}\s*:\s*public\s+BufferBase\b", hpp_txt) is not None
        cpp_pybind = re.search(rf'pybind11::class_<\s*{cls}\s*,\s*BufferBase\s*>', hpp_txt) is not None
        cpp_dtor = "destroy_on_destruction(" in hpp_txt
        # Python：class X(BufferBase) + 覆盖 destroy() 且调 super().destroy()
        py_inherit = re.search(rf"class {cls}\(BufferBase\)", py_txt) is not None
        py_super = "super().destroy()" in py_txt
        ok = all([cpp_inherit, cpp_pybind, cpp_dtor, py_inherit, py_super])
        chain_ok &= ok
        check(ok, f"3.{cls} 双重继承链 + destroy_on_destruction + super().destroy()",
              f"cpp_inherit={cpp_inherit} pybind={cpp_pybind} dtor={cpp_dtor} "
              f"py_inherit={py_inherit} py_super={py_super}")

    check(chain_ok, "3.5 四个家族接缝一致")

    # re-export
    init_txt = open(os.path.join(DEEPEP, "deep_ep", "__init__.py")).read()
    check("from .buffers.base import BufferBase" in init_txt,
          "3.6 deep_ep/__init__.py 顶层 re-export BufferBase")

    # pybind 注册函数：base.hpp 里 BufferBase 自身注册名
    base_txt = open(os.path.join(DEEPEP, "csrc", "buffers", "base.hpp")).read()
    check('class_<BufferBase>(m, "BufferBase")' in base_txt,
          "3.7 C++ 侧注册了独立的 BufferBase 类型（与 Python ABC 平行的第二条继承链）")

    # 显式 destroy 的派生字段清理（EPBuffer 是唯一在 destroy() 后清 self.context 的家族）
    ep_py = open(os.path.join(DEEPEP, "deep_ep", "buffers", "ep.py")).read()
    check("self.context = None" in ep_py and "self.nccl_comm_handle = None" in ep_py,
          "3.8 EPBuffer.destroy() 在 super().destroy() 后再清 self.context / nccl_comm_handle")


if __name__ == "__main__":
    print(f"DeepEP BufferBase 生命周期契约 —— H100 真机（DeepEP {DEEPEP}）")
    print(f"base.py   sha256[:16] = {sha16(os.path.join(DEEPEP, 'deep_ep', 'buffers', 'base.py'))}")
    print(f"base.hpp  sha256[:16] = {sha16(os.path.join(DEEPEP, 'csrc', 'buffers', 'base.hpp'))}")
    try:
        part_cpp()
    except Exception:
        traceback.print_exc()
        check(False, "Part 1 异常")
    try:
        part_py()
    except Exception:
        traceback.print_exc()
        check(False, "Part 2 异常")
    try:
        part_source()
    except Exception:
        traceback.print_exc()
        check(False, "Part 3 异常")

    print("\n===== 汇总 =====")
    for status, name, detail in RESULTS:
        print(f"  {status}  {name}")
    print(f"\n{PASS} PASS / {FAIL} FAIL")
    sys.exit(0 if FAIL == 0 else 1)
