"""P7-46 `ncu` driver: attribute *why* the K2-shaped mma CUDA B2 core
(b2_mma_probe.cu) lands near the DRAM floor while P7-44's SIMT core was L1-bound.
Kept dependency-free (torch + ctypes only) for the CUDA paths so the profile contains
exactly one kernel launch.

    python .../ncu_b2_mma_driver.py --mode ours_mma --shape b8t2048h8 --reps 1
    python .../ncu_b2_mma_driver.py --mode ours_simt --shape b8t2048h8 --reps 1
    python .../ncu_b2_mma_driver.py --mode trit_core --shape b8t2048h8 --reps 1
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
SO_MMA = "/tmp/b2_mma.so"
SO_SIMT = "/tmp/b2_core.so"
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


def _run(lib, sym, x, B, T, H, reps):
    V = 128
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dw = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")

    def p(t):
        return ctypes.c_void_p(t.data_ptr())
    fn = getattr(lib, sym)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(p(x["v"]), p(x["vn"]), p(x["h"]), p(x["dh"]),
                p(x["do"]), p(x["dv"]), p(dq), p(dk), p(dw), p(dgk),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def run_ours_mma(B, T, H, reps):
    _run(ctypes.CDLL(SO_MMA), "b2mma_launch", make_core(B, T, H), B, T, H, reps)


def run_ours_simt(B, T, H, reps):
    _run(ctypes.CDLL(SO_SIMT), "b2_core_launch", make_core(B, T, H), B, T, H, reps)


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
    ap.add_argument("--mode", choices=["ours_mma", "ours_simt", "trit_core"],
                    default="ours_mma")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]
    if a.mode == "ours_mma":
        run_ours_mma(B, T, H, a.reps)
    elif a.mode == "ours_simt":
        run_ours_simt(B, T, H, a.reps)
    else:
        run_trit(B, T, H, a.reps)


if __name__ == "__main__":
    main()
