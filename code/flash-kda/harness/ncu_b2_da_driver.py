"""P7-54 `ncu` driver: profile the CUDA B2 **dA-prewich segment** (b2_mma_probe.cu with
`-DB2M_DA`) and its GEMM-core baseline (`-DB2M_TMA`) so the *delta* of the added
`dApre = dv@v^T + dw@kg^T` segment is read off one kernel launch each.  Dependency-free
(torch + ctypes) so the profile contains exactly one kernel.

The `.so` must be pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16).

    B2D_SO=/tmp/b2_da_tma.so python .../ncu_b2_da_driver.py --mode ours_da --shape b8t2048h8
    B2D_SO=/tmp/b2_core_tma.so python .../ncu_b2_da_driver.py --mode ours_core --shape b8t2048h8
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
BT = 16


def make_operands(B, T, H):
    dev = "cuda"
    K = V = 128
    g = torch.Generator(device=dev).manual_seed(0)
    dtype = torch.float32
    x = dict(
        v=torch.randn(B, T, H, V, device=dev, generator=g, dtype=dtype),
        vn=torch.randn(B, T, H, V, device=dev, generator=g, dtype=dtype),
        do=torch.randn(B, T, H, V, device=dev, generator=g, dtype=dtype),
        dv=torch.randn(B, T, H, V, device=dev, generator=g, dtype=dtype),
        k=torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=g,
                                                    dtype=dtype), dim=-1),
        g=-torch.rand(B, T, H, K, device=dev, generator=g, dtype=dtype),
        h=torch.randn(B, (T + BT - 1) // BT, H, K, V, device=dev, generator=g, dtype=dtype),
        dh=torch.randn(B, (T + BT - 1) // BT, H, K, V, device=dev, generator=g, dtype=dtype),
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


def run_da(lib, x, B, T, H, reps):
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dwo = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")
    dApre = torch.empty(B, T, H, BT, device="cuda")
    fn = lib.b2da_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(_ptr(x["v"]), _ptr(x["vn"]), _ptr(x["h"]), _ptr(x["dh"]),
                _ptr(x["do"]), _ptr(x["dv"]), _ptr(x["k"]), _ptr(x["g"]),
                _ptr(dq), _ptr(dk), _ptr(dwo), _ptr(dgk), _ptr(dApre),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ours_core", "ours_da"], default="ours_da")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    so = os.environ.get("B2D_SO", "/tmp/b2_da.so")
    lib = ctypes.CDLL(so)
    B, T, H = SHAPES[a.shape]
    x = make_operands(B, T, H)
    if a.mode == "ours_core":
        run_core(lib, x, B, T, H, a.reps)
    else:
        run_da(lib, x, B, T, H, a.reps)


if __name__ == "__main__":
    main()
