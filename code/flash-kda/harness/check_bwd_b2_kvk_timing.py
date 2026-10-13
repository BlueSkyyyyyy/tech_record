"""P7-104 timing cross-check: independently time the production `bf16mma_tma` (and its `BVK`
control) as **K** moves, with CUDA events and CUPTI, to validate the direction of the `ncu`
`gpu__time_duration` window collected by `probe_b2_kvk_ncu.sh`.

Tags (all `-DB2M_PROD -DB2M_HDW`; a bf16 TMA box is 64 elements -> bf16 TMA is hard-pinned to
`BVK in {64,128}`, so `bf16mma64` is the same-`BVK` cp.async anchor for the TMA delta):

    prod          fp32 / cp.async / BVK=32
    bf16mma       bf16 / native bf16 mma / BVK=32 / cp.async
    bf16mma64     bf16 / native bf16 mma / BVK=64 / cp.async
    bf16mma_tma   bf16 / native bf16 mma / BVK=64 / TMA

    python .../harness/check_bwd_b2_kvk_timing.py > .../check_bwd_b2_kvk_timing.out.txt
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)
from ncu_b2_k128_dtype_driver import build_inputs, launch_reps  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev, time_wall        # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
# tag -> (extra nvcc flags, operand-dtype tag for build_inputs)
TAGS = {
    "prod":         ([], "prod"),
    "bf16mma":      (["-DB2M_BF16=1", "-DB2M_BF16MMA=1"], "bf16mma"),
    "bf16mma64":    (["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_BVK=64"], "bf16mma64"),
    "bf16mma_tma":  (["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
                     "bf16mma_tma"),
}
PAIRS = [(64, 128), (128, 128), (256, 128), (128, 64), (128, 256)]


def build(tag, K, V):
    so = "/tmp/b2kvkt_%s_k%d_v%d.so" % (tag, K, V)
    if os.path.exists(so):
        return so
    flags, _ = TAGS[tag]
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-lcuda", "-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V]
    cmd += flags + [CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed %s K=%d V=%d" % (tag, K, V))
    return so


def main():
    shape = "b8t2048h8"
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  shape=%s (BT=16, NS=2, cv-mean idle-card)" % (torch.cuda.get_device_name(), shape))
    print("%-5s %-5s %-12s %10s %10s" % ("K", "V", "tag", "dev_us", "wall_us"))
    for (K, V) in PAIRS:
        libs, data = {}, {}
        for tag, (_, dtag) in TAGS.items():
            libs[tag] = ctypes.CDLL(build(tag, K, V))
            data[tag] = build_inputs(shape, K, V, dtag)
            launch_reps(libs[tag], data[tag], shape, 5)  # warm
        torch.cuda.synchronize()
        # NOTE: `profile_dev`/`time_wall` already repeat; pass a **single** launch so the result is
        # per-call (P7-104 first version passed reps=30 inside -> 30x inflated, noisy).
        dev, wall = {}, {}
        for tag in TAGS:
            dev[tag], _ = profile_dev(lambda tg=tag: launch_reps(libs[tg], data[tg], shape, 1))
            wall[tag] = time_wall(lambda tg=tag: launch_reps(libs[tg], data[tg], shape, 1),
                                  reps=30)
        for tag in TAGS:
            print("  %-5d %-5d %-12s %10.1f %10.1f" % (K, V, tag, dev[tag], wall[tag]))
        print("      => K=%d V=%d : bf16mma/prod=%.3f  bvk64/b32=%.3f  tma/bvk64=%.3f"
              % (K, V, dev["bf16mma"] / dev["prod"],
                 dev["bf16mma64"] / dev["bf16mma"], dev["bf16mma_tma"] / dev["bf16mma64"]))


if __name__ == "__main__":
    main()
