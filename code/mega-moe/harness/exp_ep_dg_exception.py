#!/usr/bin/env python3
# P116-1 实验驱动：验证 DeepEP / DeepGEMM 的「错误 / 断言层」。
#
# A. 真机跑 src/exception_layer.cu（host 宏 + 设备 trap，均已由 shell 编译为 EXC_BIN）
# B. pybind11 翻译：src/exception_bridge.cpp → Python RuntimeError
# C. 两库 EP_*/DG_* 宏用法普查（真实计数）
# D. DG_UNIFIED_ASSERT 在 nvcc host 段的负向编译（应报 no such instruction: trap）
#
# 直接使用真实头文件；不 import deep_ep（其 _C 在本机 NCCL 版本下受阻）。
import os
import re
import subprocess
import sys
from pathlib import Path

DE = Path(os.environ.get("DEEPEP_ROOT", os.path.expanduser("~/github/DeepEP")))
DG = Path(os.environ.get("DEEPGEMM_ROOT", os.path.expanduser("~/github/DeepGEMM")))
EXC_BIN = os.environ.get("EXC_BIN", "/tmp/exception_layer")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYBIND = subprocess.check_output(["python3", "-m", "pybind11", "--includes"]).decode().split()

pass_n = fail_n = 0


def check(name, ok, detail=""):
    global pass_n, fail_n
    tag = "PASS" if ok else "FAIL"
    pass_n += bool(ok)
    fail_n += (not ok)
    print(f"[{tag}] {name}" + (f"  :: {detail}" if detail else ""))


def run_bin(mode):
    p = subprocess.run([EXC_BIN, mode], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


print("=" * 78)
print("== 环境坐标 ==")
print("=" * 78)

def commit(repo):
    try:
        return subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(repo),
                                        "log", "--oneline", "-1"], text=True).strip()
    except Exception as e:
        return f"(git failed: {e})"

print("DeepEP  :", DE, commit(DE))
print("DeepGEMM:", DG, commit(DG))
for rel in ["deep_ep/include/deep_ep/common/exception.cuh",
            "deep_gemm/include/deep_gemm/common/exception.cuh",
            "csrc/utils/exception.hpp",
            "csrc/utils/compatibility.hpp",
            "deep_gemm/include/deep_gemm/common/compile.cuh"]:
    base = DE if rel.startswith("deep_ep/") else DG
    f = base / rel
    if f.exists():
        import hashlib
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        n = len(f.read_text().splitlines())
        print(f"  {h}  {n:4d}  {rel}  ({'DeepEP' if base == DE else 'DeepGEMM'})")

print()
print("=" * 78)
print("== A. 真机跑 src/exception_layer.cu（真实头文件）==")
print("=" * 78)
rc, out = run_bin("host")
print(out.rstrip())
m = re.search(r"HOST RESULT: PASS=(\d+) FAIL=(\d+)", out)
if m:
    hp, hf = int(m.group(1)), int(m.group(2))
    check("A0 host 模式全通过", hf == 0, f"PASS={hp} FAIL={hf}")
    pass_n += hp
    fail_n += hf
else:
    check("A0 host 模式产出 RESULT 行", False, "missing RESULT")

print()
print("-- 设备端 trap（每模式独立进程，避免 sticky 错误）--")
dev_expect = {
    "dev_ep_assert":   ("EP_DEVICE_ASSERT",   "__EP_TRAP__"),
    "dev_ep_unified":  ("EP_UNIFIED_ASSERT",  "__EP_TRAP__"),
    "dev_dg_assert":   ("DG_DEVICE_ASSERT",   "__EP_TRAP__"),
    "dev_dg_traponly": ("DG_TRAP_ONLY",       "__EP_TRAP__"),
    "dev_dg_unified":  ("DG_UNIFIED_ASSERT",  "__EP_TRAP__"),
}
for mode, (label, _) in dev_expect.items():
    rc, out = run_bin(mode)
    line = [l for l in out.splitlines() if l.startswith("DEVICE")] or [out.strip()]
    print("  " + line[0])
    # trap 后同步应报非 success（device-side assert / unspecified launch failure）
    ok = ("CUDA_ERROR" in line[0]) or ("unspecified launch failure" in line[0]) \
         or ("device-side assert" in line[0]) or ("assert" in line[0].lower() and "sync=cudaSuccess" not in line[0])
    check(f"A-dev {label} 触发设备端 trap（同步返回错误）", ok, line[0])

print()
print("=" * 78)
print("== B. pybind11 翻译：EPException/DGException → Python RuntimeError ==")
print("=" * 78)
try:
    from torch.utils.cpp_extension import load, CUDA_HOME
    incs = [str(DE / "deep_ep" / "include"), str(DG / "csrc"), str(DG / "deep_gemm" / "include")]
    if CUDA_HOME:
        incs.append(os.path.join(CUDA_HOME, "include"))
        # CUDA 13 moved libcu++ (cuda/std/*) under include/cccl
        if os.path.isdir(os.path.join(CUDA_HOME, "include", "cccl")):
            incs.append(os.path.join(CUDA_HOME, "include", "cccl"))
    mod = load(
        name="ep_dg_exc_bridge_p116",
        sources=[str(ROOT / "src" / "exception_bridge.cpp")],
        extra_include_paths=incs,
        extra_cflags=["-O2"],
        verbose=False,
    )
    check("B0 真实 EPException/DGException 头文件编译成 pybind 模块成功", True)

    cases = [
        ("throw_ep_assert",  "RuntimeError", "ep assert from bridge", "Assertion exception"),
        ("throw_ep_line",    "RuntimeError", "line-info payload",     "Bridge exception"),
        ("throw_ep_unreach", "RuntimeError", "ep unreachable from bridge", "Assertion exception"),
        ("throw_dg_assert",  "RuntimeError", "dg assert from bridge", "Assertion error"),
        ("throw_dg_unreach", "RuntimeError", "dg unreachable from bridge", "Assertion error"),
    ]
    for fn, want_type, want_msg, want_prefix in cases:
        try:
            getattr(mod, fn)()
            check(f"B {fn} 抛出异常", False, "no exception")
        except BaseException as e:
            tn = type(e).__name__
            msg = str(e)
            ok = (tn == want_type) and (want_msg in msg) and (want_prefix in msg)
            check(f"B {fn} → {want_type} 且消息含 {want_prefix!r}/{want_msg!r}", ok,
                  f"{tn}: {msg}")
except Exception as e:
    check("B0 编译 pybind 桥失败", False, repr(e))

print()
print("=" * 78)
print("== C. 两库 EP_*/DG_* 宏用法普查（真实计数）==")
print("=" * 78)


def census(root, macros, dirs):
    files = []
    for d in dirs:
        p = root / d
        if p.exists():
            files += [f for f in p.rglob("*") if f.suffix in (".hpp", ".cuh", ".h", ".cu", ".cpp")]
    total = {m: 0 for m in macros}
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        for mn in macros:
            total[mn] += len(re.findall(r"\b" + re.escape(mn) + r"\b", txt))
    return total


ep_macros = ["EP_HOST_ASSERT", "EP_DEVICE_ASSERT", "EP_STATIC_ASSERT", "EP_HOST_UNREACHABLE",
             "EP_UNIFIED_ASSERT", "CUDA_RUNTIME_CHECK", "CUDA_DRIVER_CHECK", "NCCL_CHECK",
             "EPExceptionWithLineInfo", "EPException"]
dg_macros = ["DG_HOST_ASSERT", "DG_DEVICE_ASSERT", "DG_STATIC_ASSERT", "DG_TRAP_ONLY_DEVICE_ASSERT",
             "DG_HOST_UNREACHABLE", "DG_UNIFIED_ASSERT", "DG_CUDA_RUNTIME_CHECK", "DG_CUBLASLT_CHECK",
             "DGException"]
ep_tot = census(DE, ep_macros, ["csrc", "deep_ep/include"])
dg_tot = census(DG, dg_macros, ["csrc", "deep_gemm/include"])
print("DeepEP  (csrc + deep_ep/include):")
for k, v in ep_tot.items():
    print(f"  {v:5d}  {k}")
print("DeepGEMM (csrc + deep_gemm/include):")
for k, v in dg_tot.items():
    print(f"  {v:5d}  {k}")
ep_sum = sum(ep_tot[m] for m in ep_macros[:5])  # 不计 CUDA/NCCL check
dg_sum = sum(dg_tot[m] for m in dg_macros[:6])
check("C1 DeepEP 断言宏总用例 > 0", ep_sum > 0, f"断言宏 = {ep_sum}")
check("C2 DeepGEMM 断言宏总用例 > 0", dg_sum > 0, f"断言宏 = {dg_sum}")
check("C3 DeepEP 同时有 NCCL_CHECK 与 CUDA_DRIVER_CHECK（DeepGEMM 没有）",
      ep_tot["NCCL_CHECK"] > 0 and ep_tot["CUDA_DRIVER_CHECK"] > 0
      and dg_tot["DG_CUBLASLT_CHECK"] > 0)
check("C4 DeepGEMM 独有 DG_TRAP_ONLY_DEVICE_ASSERT",
      dg_tot["DG_TRAP_ONLY_DEVICE_ASSERT"] > 0 and ep_tot.get("EP_TRAP_ONLY_DEVICE_ASSERT", 0) == 0)

print()
print("=" * 78)
print("== D. DG_UNIFIED_ASSERT 在 nvcc host 段的负向编译（真实陷阱）==")
print("=" * 78)
neg_src = Path("/tmp/_p116_neg_unified.cu")
neg_src.write_text(
    '#include <cstdio>\n#include <cuda_runtime.h>\n#include <deep_gemm/common/exception.cuh>\n'
    'int main() { DG_UNIFIED_ASSERT(1 == 2); return 0; }\n'
)
includes = [str(DE / "deep_ep" / "include"), str(DG / "deep_gemm" / "include"),
            str(DG / "csrc")]
cmd = ["nvcc", "-arch=sm_90", "-std=c++17", "-c", str(neg_src), "-o", "/tmp/_p116_neg.o"]
for inc in includes:
    cmd[1:1] = ["-I", inc]
p = subprocess.run(cmd, capture_output=True, text=True)
combined = p.stdout + p.stderr
print(combined.rstrip()[-500:])
check("D1 host 段 DG_UNIFIED_ASSERT 编译失败（展开成 asm trap）",
      p.returncode != 0 and "no such instruction" in combined)

# 对照：EP_UNIFIED_ASSERT 在同样 host 段正常编译（因为用 __CUDA_ARCH__ 分派）
ep_neg = Path("/tmp/_p116_ep_unified.cu")
ep_neg.write_text(
    '#include <cuda_runtime.h>\n#include <deep_ep/common/exception.cuh>\n'
    'int main() { try { EP_UNIFIED_ASSERT(1 == 2); } catch (...) {} return 0; }\n'
)
cmd2 = ["nvcc", "-arch=sm_90", "-std=c++17", "-c", str(ep_neg), "-o", "/tmp/_p116_ep.o",
        "-I", str(DE / "deep_ep" / "include")]
p2 = subprocess.run(cmd2, capture_output=True, text=True)
check("D2 host 段 EP_UNIFIED_ASSERT 正常编译（__CUDA_ARCH__ 未定义 → EP_HOST_ASSERT）",
      p2.returncode == 0, (p2.stdout + p2.stderr).strip()[-200:])

print()
print("=" * 78)
print(f"TOTAL: PASS={pass_n} FAIL={fail_n}")
print("=" * 78)
sys.exit(0 if fail_n == 0 else 1)
