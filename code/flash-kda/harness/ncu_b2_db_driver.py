"""P7-56 `ncu` driver: profile the CUDA B2 **`db` segment** (b2_mma_probe.cu with
`-DB2M_OUT -DB2M_DKGB -DB2M_DB`) against its `-DB2M_NODB` baseline so the *delta* of the added
per-token write-gate gradient (`db = sum_v dvb*v + sum_k dkgb*kg`, a cross-warp column-sum
reduction) is read off one kernel launch each.  Dependency-free (torch + ctypes), so the profile
contains exactly one kernel.

The `.so` must be pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16).

    B2DB_SO=/tmp/b2_db_tma.so      python .../ncu_b2_db_driver.py --mode ours_db   --shape b8t2048h8
    B2DB_SO=/tmp/b2_db_tma_nodb.so python .../ncu_b2_db_driver.py --mode ours_nodb --shape b8t2048h8
"""
import argparse
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4)}
BT = 16


def make_operands(B, T, H):
    dev = "cuda"
    K = V = 128
    g = torch.Generator(device=dev).manual_seed(0)
    return dict(
        v=torch.randn(B, T, H, V, device=dev, generator=g),
        vn=torch.randn(B, T, H, V, device=dev, generator=g),
        do=torch.randn(B, T, H, V, device=dev, generator=g),
        dv=torch.randn(B, T, H, V, device=dev, generator=g),
        dw=torch.randn(B, T, H, K, device=dev, generator=g),
        k=torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=g), dim=-1),
        g=-torch.rand(B, T, H, K, device=dev, generator=g),
        beta=torch.sigmoid(torch.randn(B, T, H, device=dev, generator=g)),
        A=torch.randn(B, T, H, BT, device=dev, generator=g) * 0.1,
        h=torch.randn(B, (T + BT - 1) // BT, H, K, V, device=dev, generator=g),
        dh=torch.randn(B, (T + BT - 1) // BT, H, K, V, device=dev, generator=g),
    )


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_db(lib, x, B, T, H, reps):
    NT = (T + BT - 1) // BT
    mk = lambda *s: torch.empty(*s, device="cuda")
    dq, dk, dwo = mk(B, T, H, 128), mk(B, T, H, 128), mk(B, T, H, 128)
    dgk = mk(B, NT, H, 128)
    dkgb = mk(B, T, H, 128)
    dv2, dvb = mk(B, T, H, 128), mk(B, T, H, 128)
    db = mk(B, T, H)
    fn = lib.b2db_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 19 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(_ptr(x["v"]), _ptr(x["vn"]), _ptr(x["h"]), _ptr(x["dh"]),
                _ptr(x["do"]), _ptr(x["dv"]), _ptr(x["dw"]),
                _ptr(x["k"]), _ptr(x["g"]), _ptr(x["A"]), _ptr(x["beta"]),
                _ptr(dq), _ptr(dk), _ptr(dwo), _ptr(dgk), _ptr(dkgb),
                _ptr(dv2), _ptr(dvb), _ptr(db),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ours_db", "ours_nodb"], default="ours_db")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    so = os.environ.get("B2DB_SO", "/tmp/b2_db.so")
    lib = ctypes.CDLL(so)
    B, T, H = SHAPES[a.shape]
    x = make_operands(B, T, H)
    run_db(lib, x, B, T, H, a.reps)   # same kernel for both modes (NODB only skips the reduction)


if __name__ == "__main__":
    main()
