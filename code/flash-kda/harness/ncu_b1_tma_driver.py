"""P7-20: driver so ``ncu`` can profile the TMA + warp-specialized CUDA B1
(`harness/b1_tma_probe.cu`) against the same-shaped register-resident control
(`b1_reg_probe.cu` built with -DB1R_BV=64 -DB1R_NW=4).

nvcc hangs under ncu, so pre-compile first, then run with --no-build:
    nvcc ... b1_tma_probe.cu -o /tmp/b1_tma.so -lcuda
    nvcc ... -DB1R_BV=64 -DB1R_NW=4 b1_reg_probe.cu -o /tmp/b1_reg64.so
    ncu -k regex:b1_tma_kernel -c 1 --launch-skip 3 --set full \
        python ncu_b1_tma_driver.py --which tma --B 8 --T 2048 --H 8 --bt 64 --ns 2 --no-build
"""
import argparse
import os
import subprocess
import sys
import ctypes

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch                                                              # noqa: E402

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_b1_reg import build, load, run, REG_SO, BASE_SO            # noqa: E402

TMA_SO = "/tmp/b1_tma.so"
REG64_SO = "/tmp/b1_reg64.so"


def synth(B, T, H, device="cuda"):
    torch.manual_seed(0)
    qg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    kg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    w = (torch.randn(B, T, H, D, device=device) * 0.1).bfloat16()
    g_cum = (-torch.rand(B, T, H, D, device=device) * 0.5).float()
    do = torch.randn(B, T, H, D, device=device).bfloat16()
    dv = torch.randn(B, T, H, D, device=device).bfloat16()
    return dict(qg=qg, kg=kg, w=w, g_cum=g_cum), do, dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["tma", "reg64", "reg", "fla"], default="tma")
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
    B, T, H = args.B, args.T, args.H
    fi, do, dv = synth(B, T, H)

    if not args.no_build:
        if args.which == "tma":
            subprocess.run(["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
                            "-shared", HARNESS + "/b1_tma_probe.cu", "-o", TMA_SO, "-lcuda"],
                           check=True)
        elif args.which == "reg64":
            build("b1_reg_probe.cu", REG64_SO, extra=["-DB1R_BV=64", "-DB1R_NW=4"])

    if args.which == "tma":
        lib = load(TMA_SO, "b1t_launch", "b1t_smem_bytes")
        call = lambda: run(lib, fi, do, dv, scale, args.bt, args.ns, "b1t_launch")
    elif args.which == "reg64":
        lib = load(REG64_SO, "b1r_launch", "b1r_smem_bytes")
        call = lambda: run(lib, fi, do, dv, scale, args.bt, args.ns, "b1r_launch")
    elif args.which == "reg":
        lib = load(REG_SO, "b1r_launch", "b1r_smem_bytes")
        call = lambda: run(lib, fi, do, dv, scale, args.bt, args.ns, "b1r_launch")
    else:  # fla
        from check_bwd_dhu_dtype import make_inputs, as_dtype, fla_dhu
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)

        def call():
            chunk_gated_delta_rule_bwd_dhu(
                q=F["fi"]["qg"], k=F["fi"]["kg"], w=F["fi"]["w"], gk=F["fi"]["g_cum"],
                h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=64)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={B} T={T} H={H} bt={args.bt} ns={args.ns}")


if __name__ == "__main__":
    main()
