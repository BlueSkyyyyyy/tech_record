"""P7-44 `ncu` driver: attribute *why* the plain-SIMT CUDA B2 core (b2_core_probe.cu)
is 3x slower than the tensor-core Triton core.  Kept dependency-free (torch + ctypes
only, no Triton/FLA imports) so the profile contains exactly one kernel launch.

    python .../ncu_b2_core_driver.py --mode ours_core --shape b8t2048h8 --reps 1
    python .../ncu_b2_core_driver.py --mode trit_core --shape b8t2048h8 --reps 1

`trit_core` imports the identical-op Triton core from `check_bwd_b2_core` (so the
profile also pulls Triton's setup; use `-k regex` in ncu to keep only one kernel).
"""
import argparse
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

SHAPES = {"b8t2048h8": (8, 2048, 8), "t512h4": (1, 512, 4)}
SO = "/tmp/b2_core.so"
BT = 16


def make_core(B, T, H):
    dev = "cuda"
    K = V = 128
    g = torch.Generator(device=dev).manual_seed(0)
    NT = (T + BT - 1) // BT
    v = torch.randn(B, T, H, V, device=dev, generator=g)
    vn = torch.randn(B, T, H, V, device=dev, generator=g)
    do = torch.randn(B, T, H, V, device=dev, generator=g)
    dv = torch.randn(B, T, H, V, device=dev, generator=g)
    h = torch.randn(B, NT, H, K, V, device=dev, generator=g)
    dh = torch.randn(B, NT, H, K, V, device=dev, generator=g)
    return dict(v=v, vn=vn, do=do, dv=dv, h=h, dh=dh)


def run_ours(B, T, H, reps):
    lib = ctypes.CDLL(SO)
    lib.b2_core_launch.restype = ctypes.c_int
    lib.b2_core_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    x = make_core(B, T, H)
    V = 128
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dw = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")

    def p(t):
        return ctypes.c_void_p(t.data_ptr())
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = lib.b2_core_launch(p(x["v"]), p(x["vn"]), p(x["h"]), p(x["dh"]),
                                p(x["do"]), p(x["dv"]), p(dq), p(dk), p(dw), p(dgk),
                                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def run_trit(B, T, H, reps):
    from check_bwd_b2_core import build_core_triton, run_triton_core
    k = build_core_triton()
    x = make_core(B, T, H)
    x["q"] = torch.empty(B, T, H, 128, device="cuda")   # shape carrier only
    inp = dict(q=x["q"], v=x["v"], v_new=x["vn"], h=x["h"], dh=x["dh"], do=x["do"], dv=x["dv"])
    for _ in range(reps):
        run_triton_core(k, inp, BT, num_warps=8)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ours_core", "trit_core"], default="ours_core")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]
    if a.mode == "ours_core":
        run_ours(B, T, H, a.reps)
    else:
        run_trit(B, T, H, a.reps)


if __name__ == "__main__":
    main()
