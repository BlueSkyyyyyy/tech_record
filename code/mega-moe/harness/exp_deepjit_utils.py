#!/usr/bin/env python3
# P160-1：解析 C++ 程序输出 + 用 Python 独立复核 json/uuid/gil。见 exp_deepjit_utils.sh。
import json as jsonlib
import os
import re
import sys

RAW = "/tmp/deepjit_utils.raw.txt"
PASS = FAIL = 0


def ok(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[B] PASS {name}" + (f" : {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[B] FAIL {name}" + (f" : {detail}" if detail else ""))


lines = open(RAW).read().splitlines()


def one(pat):
    m = [l for l in lines if re.match(pat, l)]
    return m[-1] if m else None


# C++ 侧自检计数
m = re.search(r"CPP_SUMMARY pass=(\d+) fail=(\d+)", one(r"CPP_SUMMARY .*"))
cpp_pass, cpp_fail = int(m.group(1)), int(m.group(2))
print(f"C++ 段自检: pass={cpp_pass} fail={cpp_fail}")
ok(cpp_fail == 0, "cpp.no_fail")
ok(cpp_pass >= 30, "cpp.check_count", str(cpp_pass))

# --- json：用 Python 的 json 解析 C++ dump ---
basic = one(r"JSON_BASIC (.*)")
dumped = basic.split("JSON_BASIC ", 1)[1]
obj = jsonlib.loads(dumped)
ok(obj == {"a": 1, "b": "x", "c": [1, 2, 3], "d": None, "e": True}, "json.python_loads_basic", dumped)

ok(one(r"JSON_INT -5") is not None, "json.int64_literal")
ok(one(r"JSON_UINT 18446744073709551615") is not None, "json.uint64_literal")

esc = one(r'JSON_ESC (.*)').split("JSON_ESC ", 1)[1]
evil = '"\\\b\f\n\r\t\x01'
ok(jsonlib.loads(esc) == evil, "json.escape_roundtrip", repr(esc))
ok("\\u0001" in esc and "\\n" in esc, "json.escape_control_uses_u0001", esc)

dup = one(r"JSON_DUP (.*)").split("JSON_DUP ", 1)[1]
ok(dup == '{"k":1,"k":2}', "json.ordered_dup_keys", dup)
ok(jsonlib.loads(dup) == {"k": 2}, "json.python_keeps_last_dup", "Python 解析保留末值")

nest = one(r"JSON_NEST (.*)").split("JSON_NEST ", 1)[1]
ok(jsonlib.loads(nest) == {"x": [{"y": None}]}, "json.nested_loads", nest)

# --- str::join == Python " ".join ---
sj = one(r"STR_JOIN (.*)").split("STR_JOIN ", 1)[1]
ok(sj == "-O3 --use_fast_math -std=c++17", "str.join_matches_expected", sj)
ok(sj == " ".join(["-O3", "--use_fast_math", "-std=c++17"]), "str.join_equals_python_join")

# --- uuid ---
u = one(r"UUID (\S+)").split("UUID ", 1)[1]
ok(re.fullmatch(r"\d+-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}", u) is not None, "uuid.format", u)
# 注意：uuid 的 pid 前缀是 **C++ 程序**的 pid，Python 复核者本身是另一个进程，故只校验前缀是非零整数
# （与 `::getpid()` 的逐位相等由 C++ 侧断言负责）。
ok(u.split("-")[0].isdigit() and int(u.split("-")[0]) > 0, "uuid.pid_prefix_positive_int", u)

# --- command ---
ok(one(r"CMD_SUCCESS hi") is not None, "command.echo")
ok(one(r"CMD_STDERR to_stderr") is not None, "command.stderr_merged")
wn = one(r"CMD_WHICH_NVCC (.*)")
ok(wn is not None and "nvcc" in wn, "command.which_nvcc", wn)

# --- filesystem ---
fr = one(r"FS_ROUNDTRIP len=(\d+)")
ok(fr is not None and int(fr.split("len=")[1]) == 14, "fs.roundtrip_len_14", fr)
fn = one(r"FS_NORMALIZE (.*)").split("FS_NORMALIZE ", 1)[1]
ok(fn.startswith("/") and fn.endswith("/c"), "fs.normalize_absolute_c", fn)

# --- GilScopedRelease（pybind 探针，真机 import）---
sys.path.insert(0, "/tmp")
import deepjit_gil  # noqa: E402

before, inside, after = deepjit_gil.probe()
print(f"GIL probe (python 持 GIL): before={before} inside={inside} after={after}")
ok(before == 1 and inside == 0 and after == 1, "gil.released_inside_restored_after", f"{before}/{inside}/{after}")

nb, ni = deepjit_gil.probe_no_gil()
print(f"GIL probe (call_guard 预释放): before={nb} inside={ni}")
ok(nb == 0 and ni == 0, "gil.no_double_release_when_already_free", f"{nb}/{ni}")

print()
print(f"Python 侧: PASS={PASS} FAIL={FAIL}")
print("总判定: " + ("ALL PASS" if FAIL == 0 else "FAIL"))
sys.exit(0 if FAIL == 0 else 1)
