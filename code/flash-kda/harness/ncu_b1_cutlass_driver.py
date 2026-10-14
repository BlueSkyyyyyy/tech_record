"""P7-41 driver so ``ncu`` can compare the CUTLASS-pipeline B1 (`-DB1T_CUTLASS`) against the
hand-written-mbarrier B1 on the same operands.  nvcc hangs under ncu, so prebuild the .so,
then profile with --no-build.

    python ncu_b1_cutlass_driver.py --variant cut2 --B 8 --T 2048 --H 8 --bt 64 --ns 2
    ncu -k regex:b1_tma_kernel -c 1 --launch-skip 3 --set full \
        python ncu_b1_cutlass_driver.py --variant cut2 --B 8 --T 2048 --H 8 --bt 64 --ns 2 --no-build
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch                                                              # noqa: E402

D = 128
HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

from check_bwd_dh_roundtrip import build, load, run_b1                    # noqa: E402

CUTLASS_INC = "-I~/github/cutlass/include"
STORES = ["-DB1T_TMADH", "-DB1T_TMADV", "-DB1T_EPROXY", "-DB1T_SPROXY"]
VARIANT_FLAGS = {
    "base": ["-DB1T_ETMA"],
    "hnd":  ["-DB1T_ETMA"] + STORES,
    "cut":  ["-DB1T_ETMA", "-DB1T_CUTLASS", CUTLASS_INC],
    "cut2": ["-DB1T_ETMA", "-DB1T_CUTLASS", CUTLASS_INC] + STORES,
}
SO = {name: f"/tmp/b1c_{name}.so" for name in VARIANT_FLAGS}


def synth(B, T, H):
    torch.manual_seed(0)
    qg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device="cuda"), dim=-1).bfloat16()
    kg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device="cuda"), dim=-1).bfloat16()
    w = (torch.randn(B, T, H, D, device="cuda") * 0.1).bfloat16()
    g_cum = (-torch.rand(B, T, H, D, device="cuda") * 0.5).float()
    do = torch.randn(B, T, H, D, device="cuda").bfloat16()
    dv = torch.randn(B, T, H, D, device="cuda").bfloat16()
    # b1t_launch expects the FLA operand bundle: fi = {qg,kg,w,g_cum}, do, dv
    return dict(qg=qg, kg=kg, w=w, g_cum=g_cum), do, dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_FLAGS), default="cut2")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=64)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    scale = D ** -0.5
    fi, do, dv = synth(args.B, args.T, args.H)

    if not args.no_build:
        build("b1_tma_probe.cu", SO[args.variant], extra=["-lcuda"] + VARIANT_FLAGS[args.variant])
    lib = load(SO[args.variant])
    call = lambda: run_b1(lib, fi, do, dv, scale, args.bt, args.ns)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE variant={args.variant} B={args.B} T={args.T} H={args.H} bt={args.bt} ns={args.ns}")


if __name__ == "__main__":
    main()
