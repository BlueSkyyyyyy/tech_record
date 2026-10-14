"""P7-52 `ncu` driver: profile the CUDA B2 **output-branch** segment (b2_mma_probe.cu with
`-DB2M_OUT`) and its GEMM-core baseline (`-DB2M_TMA`, no `-DB2M_OUT`) so the *delta* of the
non-GEMM segment can be read off one kernel launch each.  Dependency-free (torch + ctypes)
so the profile contains exactly one kernel.

The `.so` must be pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16).

    B2O_SO=/tmp/b2_out_tma.so python .../ncu_b2_out_driver.py --mode ours_out --shape b8t2048h8 --reps 5
    B2O_SO=/tmp/b2_core_tma.so python .../ncu_b2_out_driver.py --mode ours_core --shape b8t2048h8 --reps 5
"""
import argparse
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

SHAPES = {"b8t2048h8": (8, 2048, 8), "t512h4": (1, 512, 4)}
BT = 16


def make_operands(B, T, H):
    dev = "cuda"
    K = V = 128
    g = torch.Generator(device=dev).manual_seed(0)
    NT = (T + BT - 1) // BT
    x = dict(
        v=torch.randn(B, T, H, V, device=dev, generator=g),
        vn=torch.randn(B, T, H, V, device=dev, generator=g),
        do=torch.randn(B, T, H, V, device=dev, generator=g),
        dv=torch.randn(B, T, H, V, device=dev, generator=g),
        h=torch.randn(B, NT, H, K, V, device=dev, generator=g),
        dh=torch.randn(B, NT, H, K, V, device=dev, generator=g),
        A=torch.randn(B, T, H, BT, device=dev, generator=g),
        beta=torch.rand(B, T, H, device=dev, generator=g),
    )
    return x


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_core(lib, x, B, T, H, reps):
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dw = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")
    fn = lib.b2mma_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(_ptr(x["v"]), _ptr(x["vn"]), _ptr(x["h"]), _ptr(x["dh"]),
                _ptr(x["do"]), _ptr(x["dv"]), _ptr(dq), _ptr(dk), _ptr(dw), _ptr(dgk),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def run_out(lib, x, B, T, H, reps):
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dw = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")
    dv2 = torch.empty(B, T, H, 128, device="cuda")
    dvb = torch.empty(B, T, H, 128, device="cuda")
    fn = lib.b2out_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(_ptr(x["v"]), _ptr(x["vn"]), _ptr(x["h"]), _ptr(x["dh"]),
                _ptr(x["do"]), _ptr(x["dv"]), _ptr(x["A"]), _ptr(x["beta"]),
                _ptr(dq), _ptr(dk), _ptr(dw), _ptr(dgk), _ptr(dv2), _ptr(dvb),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ours_core", "ours_out"], default="ours_out")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    so = os.environ.get("B2O_SO", "/tmp/b2_out.so")
    lib = ctypes.CDLL(so)
    B, T, H = SHAPES[a.shape]
    x = make_operands(B, T, H)
    if a.mode == "ours_core":
        run_core(lib, x, B, T, H, a.reps)
    else:
        run_out(lib, x, B, T, H, a.reps)


if __name__ == "__main__":
    main()
