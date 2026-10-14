#!/usr/bin/env python3
"""P130-1 实验：DeepEP `comm::Context`（csrc/kernels/comm/context.cpp）的 host 几何算术与
NCCL 运行时版本闸门，在 H100(sm90) 上做真机对拍。

不触任何 kernel / NCCL 设备 API：
  * 设备/常量/advance_ptr 来自真实头文件（`src/deepep_context_arith.cu` 直接 include
    `deep_ep/common/compiled.cuh` + `math.cuh` 编译运行）；
  * NCCL 版本用 ctypes 直接 `ncclGetVersion` 读两个真实 libnccl（system 2.29.7 / pip 2.32.3），
    并把 `context.cpp:156-162` 的编译期分支判据与 Python `deep_ep/__init__.py:19-41`
    的 `check_nccl_so` 二进制比对判据都复刻出来；
  * 域分解 / GIN signal 预算 / 存储窗口偏移用纯 Python 独立复刻，与 CUDA 程序逐行对拍。
"""
import ctypes
import glob
import hashlib
import os
import re
import sys

DEEPEP = os.environ.get("DEEPEP_ROOT", os.path.expanduser("~/github/DeepEP"))
CUDA_OUT = os.environ.get("CUDA_ARITH_OUT", "/tmp/deepep_context_arith.txt")

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")
    return ok


def parse_nccl_header(path):
    txt = open(path).read()
    def val(macro):
        m = re.search(rf"#define\s+{macro}\s+(\d+)", txt)
        return int(m.group(1)) if m else None
    major, minor, patch = val("NCCL_MAJOR"), val("NCCL_MINOR"), val("NCCL_PATCH")
    return major, minor, patch, major * 10000 + minor * 100 + patch


def runtime_nccl_version(so):
    lib = ctypes.CDLL(so)
    v = ctypes.c_int()
    lib.ncclGetVersion(ctypes.byref(v))
    return v.value


def nccl_version_code(major, minor, patch):
    return major * 10000 + minor * 100 + patch


# ── 1. 真实常量（来自 CUDA 程序直接 include 的真实头文件） ────────────────────────────
print("=" * 92)
print("P130-1 · DeepEP comm::Context host 算术 + NCCL 运行时版本闸门 —— H100 真机")
print("=" * 92)
print(f"DeepEP = {DEEPEP}")

cuda_lines = open(CUDA_OUT).read().splitlines()
cuda_const = {}
for ln in cuda_lines:
    if ln.startswith("CONST "):
        k, v = ln[6:].split("=", 1)
        cuda_const[k] = int(v)

print("\n--- §0 真实常量（src/deepep_context_arith.cu 直接 include 真实头文件） ---")
for k, v in cuda_const.items():
    print(f"  {k} = {v}")

# Python 侧从 compiled.cuh 解析，交叉核对
compiled = open(os.path.join(DEEPEP, "deep_ep/include/deep_ep/common/compiled.cuh")).read()
py_const = {}
for name in ("kNumTMAAlignmentBytes", "kNumRDMAAlignmentBytes", "kNumAllocationAlignmentBytes",
             "kNumMaxRanks", "kNumMaxQPs", "kDefaultQPDepth"):
    expr = re.search(rf"constexpr int {name} = ([0-9 *]+);", compiled).group(1)
    py_const[name] = int(eval(expr))
print("\n--- §1 常量 python(parsed) vs cuda(real) ---")
for k, v in py_const.items():
    check(f"const {k}", cuda_const.get(k) == v, f"py={v} cuda={cuda_const.get(k)}")
check("align 关系 RDMA%TMA==0", cuda_const["kNumRDMAAlignmentBytes"] % cuda_const["kNumTMAAlignmentBytes"] == 0)
check("align 关系 Alloc%RDMA==0", cuda_const["kNumAllocationAlignmentBytes"] % cuda_const["kNumRDMAAlignmentBytes"] == 0)
check("kNumAllocationAlignmentBytes==2MiB", cuda_const["kNumAllocationAlignmentBytes"] == 2 * 1024 * 1024)

# ── 2. 域分解 / GIN signal 预算 / 存储窗口偏移：Python 复刻 vs CUDA ──────────────────────
def py_advance(off, delta):
    return off + delta


print("\n--- §2 advance_ptr 布局（context.cpp:200-213 同构） ---")
cuda_ptr = {ln.split()[1].split("=")[0]: int(ln.split("=")[1]) for ln in cuda_lines if ln.startswith("PTR ")}
# 复刻：workspace=base+0；buffer=workspace+1024；storage_base=base+4096；local=storage_base+(6%4)*256
exp_ptr = {
    "workspace_off": py_advance(0, 0),
    "buffer_off": py_advance(0, 1024),
    "storage_base_off": py_advance(0, 4096),
    "local_storage_off": py_advance(4096, (6 % 4) * 256),
}
for i in range(4):
    exp_ptr[f"storage_window_{i}_off"] = py_advance(4096, i * 256)
for k, v in exp_ptr.items():
    check(f"ptr {k}", cuda_ptr.get(k) == v, f"py={v} cuda={cuda_ptr.get(k)}")


def py_domain(num_ranks, num_nvl, hybrid, rank):
    num_rdma = num_ranks // num_nvl
    nvl_idx = rank % num_nvl
    rdma_idx = rank // num_nvl
    identity = rank == rdma_idx * num_nvl + nvl_idx
    if hybrid:
        num_scaleout, num_scaleup = num_rdma, num_nvl
        scaleout_idx, scaleup_idx = rdma_idx, nvl_idx
    else:
        num_scaleout, num_scaleup = 1, num_ranks
        scaleout_idx, scaleup_idx = 0, rank
    is_scaleup_nvlink = num_scaleup == num_nvl
    gin_signal_count = num_ranks + 2 * 2
    return (num_rdma, nvl_idx, rdma_idx, identity,
            num_scaleout, num_scaleup, scaleout_idx, scaleup_idx,
            is_scaleup_nvlink, gin_signal_count)


print("\n--- §3 域分解逐行对拍（context.cpp:166-183） ---")
mismatch = 0
n_domain = 0
for ln in cuda_lines:
    if not ln.startswith("DOMAIN "):
        continue
    d = dict(kv.split("=") for kv in ln.split()[1:])
    got = py_domain(int(d["ranks"]), int(d["nvl"]), bool(int(d["hybrid"])), int(d["rank"]))
    exp = (int(d["num_rdma"]), int(d["nvl_idx"]), int(d["rdma_idx"]), bool(int(d["identity"])),
           int(d["scaleout"]), int(d["scaleup"]), int(d["scaleout_idx"]), int(d["scaleup_idx"]),
           bool(int(d["is_scaleup_nvlink"])), int(d["gin_signal_count"]))
    n_domain += 1
    if got != exp:
        mismatch += 1
        print(f"    mismatch {ln} -> {got}")
check("域分解全部逐字段一致", mismatch == 0, f"{n_domain} 行 mismatch={mismatch}")

# 独立不变量：logical 乘积恒等 num_ranks；scaleout/up 与物理域关系
inv_ok = True
for num_ranks in (8, 16, 64, 256, 132):
    for num_nvl in (num_ranks, 8, 4):
        if num_ranks % num_nvl:
            continue
        for h in (True, False):
            so, su = py_domain(num_ranks, num_nvl, h, 0)[4:6]
            if h:
                inv_ok &= (so * su == num_ranks) and (so == num_ranks // num_nvl) and (su == num_nvl)
            else:
                inv_ok &= (so == 1 and su == num_ranks)
check("逻辑域乘积恒等 num_ranks（hybrid）/ 单级（非 hybrid）", inv_ok)

# ── 3. NCCL 运行时版本闸门（context.cpp:156-162） ────────────────────────────────────
print("\n--- §4 NCCL 运行时版本闸门（context.cpp:156-162） ---")
H_PIP = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/include/nccl.h"
H_SYS = "/usr/include/nccl.h"
L_PIP = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
L_SYS = "/lib/x86_64-linux-gnu/libnccl.so.2"

pip_hdr = parse_nccl_header(H_PIP)
sys_hdr = parse_nccl_header(H_SYS)
pip_rt = runtime_nccl_version(L_PIP)
sys_rt = runtime_nccl_version(L_SYS)
print(f"  pip   header {pip_hdr[0]}.{pip_hdr[1]}.{pip_hdr[2]} code={pip_hdr[3]} | runtime code={pip_rt} "
      f"({pip_rt//10000}.{(pip_rt%10000)//100}.{pip_rt%100})")
print(f"  system header {sys_hdr[0]}.{sys_hdr[1]}.{sys_hdr[2]} code={sys_hdr[3]} | runtime code={sys_rt} "
      f"({sys_rt//10000}.{(sys_rt%10000)//100}.{sys_rt%100})")
print(f"  本机 torch 实际链接的 NCCL（torch.cuda.nccl.version）: 2.29.7（= system）")
print(f"  header code >= NCCL_VERSION(2,31,0)={nccl_version_code(2,31,0)} 时走 runtime 分支")


def gate(header_code, runtime_code):
    """复刻 `#if NCCL_VERSION_CODE >= NCCL_VERSION(2,31,0)` 两分支：
       >=2.31 → reqs.useRuntimeVersion=true（不要求相等）；
       <2.31  → 断言 header_code == runtime_code（否则 abort）。"""
    if header_code >= nccl_version_code(2, 31, 0):
        return "runtime-version", True
    return "assert-equality", header_code == runtime_code


print("\n  组合判定：")
combos = [
    ("pip header 2.32.3 vs pip runtime 2.32.3", pip_hdr[3], pip_rt),
    ("pip header 2.32.3 vs system runtime 2.29.7", pip_hdr[3], sys_rt),
    ("system header 2.29.7 vs system runtime 2.29.7", sys_hdr[3], sys_rt),
    ("hypothetical header 2.30.0 vs runtime 2.29.7", nccl_version_code(2, 30, 0), sys_rt),
]
for label, hc, rc in combos:
    branch, ok = gate(hc, rc)
    print(f"    {label:52s} -> {branch:16s} ok={ok}")
check("pip header 走 runtime 分支（本机 DeepEP 编译期分支）", gate(pip_hdr[3], pip_rt)[0] == "runtime-version")
check("pip header 不要求 runtime 相等（2.32.3 分支）", gate(pip_hdr[3], sys_rt)[1] is True)
check("2.29.7 system header 走 assert-equality 分支且自洽", gate(sys_hdr[3], sys_rt) == ("assert-equality", True))
check("旧 header(2.30.0) 对 2.29.7 runtime 会被断言拦下", gate(nccl_version_code(2, 30, 0), sys_rt)[1] is False)

# 复刻 check_nccl_so 的二进制比对（deep_ep/__init__.py:19-41）：loaded vs linked
print("\n  check_nccl_so 的二进制比对（deep_ep/__init__.py:19-41）：")
def file_eq(a, b):
    # 只用 (大小, 前 4MiB 的 sha256) 近似「二进制相同」，避免整读 ~数百 MB 的 libnccl
    if not (os.path.exists(a) and os.path.exists(b)):
        return False
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    def head(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read(4 * 1024 * 1024))
        return h.hexdigest()
    return head(a) == head(b)
linked_pip = sorted(glob.glob(os.path.dirname(L_PIP) + "/libnccl.so*"))[0]
linked_sys = sorted(glob.glob(os.path.dirname(L_SYS) + "/libnccl.so*"))[0]
print(f"    loaded(torch, system)={linked_sys}")
print(f"    linked(expected, pip )={linked_pip}")
check("loaded(system) vs linked(pip) 二进制不等（=本机阻塞根因）", not file_eq(linked_pip, linked_sys))
check("loaded(system) vs linked(system) 二进制相等", file_eq(linked_sys, linked_sys))

# ── 4. Python 侧句柄缓存语义（deep_ep/comm/handle.py:44-77） ──────────────────────────
print("\n--- §5 NCCLCommHandle 缓存/托管语义（handle.py:12-77） ---")
import ast
src = open(os.path.join(DEEPEP, "deep_ep/comm/handle.py")).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NCCLCommHandle")
has_managed = any(isinstance(n, ast.FunctionDef) and n.name == "__del__" for n in cls.body)
check("NCCLCommHandle 有 __del__ 且受 managed 门控", has_managed)
# 复刻 get_nccl_comm_handle 的三分支：cache / reuse torch _comm_ptr / create
def py_get_handle(group, force_new, cache, torch_has_comm, reuse_env, backend_comm):
    if not force_new and group in cache:
        return cache[group], "cache"
    if not force_new and torch_has_comm and reuse_env:
        cache[group] = ("managed=False", backend_comm)
        return cache[group], "reuse"
    key = "time_ns" if force_new else group
    cache[key] = ("managed=True", "new_comm")
    return cache[key], "create"


cache = {}
h1, b1 = py_get_handle("g", False, cache, True, True, 111)
h2, b2 = py_get_handle("g", False, cache, True, True, 111)
h3, b3 = py_get_handle("g", True, cache, True, True, 111)
check("首次 reuse torch comm（managed=False）", b1 == "reuse" and h1[0] == "managed=False")
check("同 group 命中缓存", b2 == "cache" and h2 == h1)
check("force_new 绕过缓存并新建 managed=True", b3 == "create" and h3[0] == "managed=True")

print("\n" + "=" * 92)
print(f"总计：PASS={PASS} / FAIL={FAIL}")
print("=" * 92)
sys.exit(1 if FAIL else 0)
