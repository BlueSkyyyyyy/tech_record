"""P7-87 timing: independently time the `BF16_VDW x TMA` cross (and its x BF16MMA stack) with
CUDA events, to confirm that the compile-time no-op is also a **zero timing delta** (the section 1.2
"one event pair wraps REPS back-to-back calls" rule).  Anchors: bf16tma, bf16mma_tma.

    python .../harness/check_bwd_b2_kv4_timing.py > .../check_bwd_b2_kv4_timing.out.txt
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)
from ncu_b2_kv2_driver import build_inputs, launch_reps  # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
# tag -> (extra nvcc flags, operand-dtype tag)
TAGS = {
    "bf16tma":          (["-DB2M_BF16=1", "-DB2M_TMA=1", "-DB2M_BVK=64"], "bf16tma"),
    "bf16_vdw_tma":     (["-DB2M_BF16=1", "-DB2M_BF16_VDW=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
                         "bf16tma"),
    "bf16mma_tma":      (["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
                         "bf16tma"),
    "bf16_vdw_mma_tma": (["-DB2M_BF16=1", "-DB2M_BF16_VDW=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1",
                          "-DB2M_BVK=64"], "bf16tma"),
}
# pair -> (numerator tag, denominator/twin tag)
PAIRS = [("bf16_vdw_tma", "bf16tma"), ("bf16_vdw_mma_tma", "bf16mma_tma")]


def build(tag, V):
    so = "/tmp/b2kv4t_%s_v%d.so" % (tag, V)
    flags, _ = TAGS[tag]
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-lcuda", "-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_VD=%d" % V] + flags + [CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed %s V=%d" % (tag, V))
    return so


def time_wall(fn, reps=100, rounds=3):
    best = float("inf")
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1e3 / reps)
    return best


def main():
    shape = "b8t2048h8"
    print("shape=%s  (K=128, BT=16, NS=2; event-pair mean, idle-card)" % shape)
    print("bf16 TMA builds use BVK=64 (section 76/88/89)")
    print("%-18s %4s %10s %10s" % ("tag", "V", "wall_us", "ratio"))
    for V in (64, 128, 256):
        libs, data = {}, {}
        for tag, (_, dtag) in TAGS.items():
            libs[tag] = ctypes.CDLL(build(tag, V))
            data[tag] = build_inputs(shape, V, dtag)
            launch_reps(libs[tag], data[tag], shape, 5)
        torch.cuda.synchronize()
        times = {tag: time_wall(lambda tg=tag: launch_reps(libs[tg], data[tg], shape, 100))
                 for tag in TAGS}
        for tag in TAGS:
            print("%-18s %4d %10.1f" % (tag, V, times[tag]))
        for num, den in PAIRS:
            print("    pair %-18s / %-14s = %.4f" % (num, den, times[num] / times[den]))


if __name__ == "__main__":
    main()
