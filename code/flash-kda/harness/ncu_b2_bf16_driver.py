"""P7-73 `ncu` driver: attribute the bf16-native B2 K2-mma GEMM-core win (base vs `-DB2M_BF16`).
Kept dependency-free (torch + ctypes only) so the profile holds exactly one kernel launch.

    python .../ncu_b2_bf16_driver.py --mode base --shape b8t2048h8 --reps 3
    python .../ncu_b2_bf16_driver.py --mode bf16 --shape b8t2048h8 --reps 3
"""
import os
import argparse
import ctypes
import sys

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

SHAPES = {"b8t2048h8": (8, 2048, 8), "t512h4": (1, 512, 4)}
SO_BASE = "/tmp/b2_bf16_base.so"
SO_BF16 = "/tmp/b2_bf16_native.so"
BT = 16


def make_core(B, T, H, dt):
    dev = "cuda"
    K = V = 128
    g = torch.Generator(device=dev).manual_seed(0)
    NT = (T + BT - 1) // BT
    v = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    vn = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    do = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    dv = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    h = torch.randn(B, NT, H, K, V, device=dev, generator=g).to(dt)
    dh = torch.randn(B, NT, H, K, V, device=dev, generator=g).to(dt)
    return dict(v=v, vn=vn, do=do, dv=dv, h=h, dh=dh)


def _run(lib, x, B, T, H, reps):
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dw = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")

    def p(t):
        return ctypes.c_void_p(t.data_ptr())
    fn = lib.b2mma_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(p(x["v"]), p(x["vn"]), p(x["h"]), p(x["dh"]),
                p(x["do"]), p(x["dv"]), p(dq), p(dk), p(dw), p(dgk),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "bf16"], default="base")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]
    dt = torch.bfloat16 if a.mode == "bf16" else torch.float32
    so = SO_BF16 if a.mode == "bf16" else SO_BASE
    _run(ctypes.CDLL(so), make_core(B, T, H, dt), B, T, H, a.reps)


if __name__ == "__main__":
    main()
