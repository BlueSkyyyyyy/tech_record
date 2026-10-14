"""P7-86 timing cross-check: independently time the K!=V **two-knob stacks** (docs/07 section 89)
with CUDA events, to validate the direction of the `ncu` `gpu__time_duration` window.  One event
pair wraps `REPS` back-to-back calls and divides by `REPS` (the section 1.2 "mean, not median"
rule).

Tags (K=128, V in {64,128,256}); a bf16 TMA box is 64 elements, so the bf16 TMA builds force
BVK=64 while the fp32 TMA builds keep BVK=32 (as in section 88).  `prod64`/`bf1664` are the
same-BVK anchors for the bf16-TMA deltas.

    python .../harness/check_bwd_b2_kv3_timing.py > .../check_bwd_b2_kv3_timing.out.txt
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)
from ncu_b2_kv2_driver import build_inputs, launch_reps  # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
# tag -> (extra nvcc flags, operand-dtype tag for build_inputs)
TAGS = {
    "prod":         ([], "prod"),
    "prod64":       (["-DB2M_BVK=64"], "prod"),
    "tma":          (["-DB2M_TMA=1"], "prod"),
    "damma_tma":    (["-DB2M_DAMMA=1", "-DB2M_TMA=1"], "prod"),
    "bf1664":       (["-DB2M_BF16=1", "-DB2M_BVK=64"], "bf16"),
    "bf16tma":      (["-DB2M_BF16=1", "-DB2M_TMA=1", "-DB2M_BVK=64"], "bf16tma"),
    "bf16mma_tma":  (["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
                     "bf16mma_tma"),
}


def build(tag, V):
    so = "/tmp/b2kv3t_%s_v%d.so" % (tag, V)
    if os.path.exists(so):
        return so
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
        best = min(best, s.elapsed_time(e) * 1e3 / reps)  # us/call
    return best


def main():
    shape = "b8t2048h8"
    print("shape=%s  (K=128, BT=16, NS=2; event-pair mean, idle-card)" % shape)
    print("bf16 TMA builds use BVK=64; fp32 TMA builds use BVK=32 (section 76/88)")
    print("%-12s %4s %10s %8s" % ("tag", "V", "wall_us", "vs_prod"))
    for V in (64, 128, 256):
        libs, data = {}, {}
        for tag, (_, dtag) in TAGS.items():
            libs[tag] = ctypes.CDLL(build(tag, V))
            data[tag] = build_inputs(shape, V, dtag)
            launch_reps(libs[tag], data[tag], shape, 5)  # warm
        torch.cuda.synchronize()
        base = None
        for tag in TAGS:
            t = time_wall(lambda tg=tag: launch_reps(libs[tg], data[tg], shape, 100))
            if tag == "prod":
                base = t
            print("%-12s %4d %10.1f %8s" % (tag, V, t, "" if base is None or tag == "prod"
                                            else "%.3f" % (t / base)))


if __name__ == "__main__":
    main()
