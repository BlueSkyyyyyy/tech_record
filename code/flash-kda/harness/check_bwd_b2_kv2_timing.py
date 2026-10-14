"""P7-85 timing cross-check: independently time the four new K!=V production knobs (docs/07
section 88) with CUDA events, to validate the direction of the `ncu` `gpu__time_duration` window
(the P7-84 caveat: the ncu single-kernel window is not the same quantity as CUPTI/wall, section
63.5).  One event pair wraps `REPS` back-to-back calls and divides by `REPS` (the section 1.2
"mean, not median" rule).

    python .../harness/check_bwd_b2_kv2_timing.py > .../check_bwd_b2_kv2_timing.out.txt
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)
from ncu_b2_kv2_driver import build_inputs, launch_reps, SHAPES  # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
TAGS = {
    "prod":    [],
    "bf16":    ["-DB2M_BF16=1"],
    "bf16mma": ["-DB2M_BF16=1", "-DB2M_BF16MMA=1"],
    "damma":   ["-DB2M_DAMMA=1"],
    "vdw":     ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1"],
    "tma":     ["-DB2M_TMA=1"],
}


def build(tag, V):
    so = "/tmp/b2kv2_%s_v%d.so" % (tag, V)
    if os.path.exists(so):
        return so
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-lcuda", "-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_VD=%d" % V] + TAGS[tag] + \
          [CU, "-o", so]
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
        best = min(best, s.elapsed_time(e) * 1e3 / reps)  # us/call
    return best


def main():
    shape = "b8t2048h8"
    print("shape=%s  (K=128, BT=16, BVK=32, NS=2; event-pair mean, idle-card)" % shape)
    print("%-8s %4s %10s %8s" % ("tag", "V", "wall_us", "vs_prod"))
    for V in (64, 128, 256):
        libs, data = {}, {}
        for tag in TAGS:
            so = build(tag, V)
            libs[tag] = ctypes.CDLL(so)
            data[tag] = build_inputs(shape, V, tag)
            launch_reps(libs[tag], data[tag], shape, 5)  # warm
        torch.cuda.synchronize()
        base = None
        for tag in ("prod", "bf16", "bf16mma", "damma", "vdw", "tma"):
            t = time_wall(lambda tg=tag: launch_reps(libs[tg], data[tg], shape, 100))
            if tag == "prod":
                base = t
            print("%-8s %4d %10.1f %8s" % (tag, V, t, "" if base is None or tag == "prod"
                                           else "%.3f" % (t / base)))


if __name__ == "__main__":
    main()
