"""P7-92/93 `ncu` driver: pipe attribution of the **`intra` fold x K!=128** -- profile the
six-segment production launch **without** (`prod`) vs **with** (`prod_intra`, `-DB2M_INTRA`)
the folded B3 diagonal at `K in {64,128}` (`V=K`).  The "other half" of section 94.7's blank (2):
section 95 (P7-92) bit-checked the fold at `K in {64,128}` but did not attribute where the fold's
extra cost lands as `K` moves.

The `.so` is pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16), so this
driver is dependency-free (torch + ctypes) and the profile holds exactly one `b2mma_kernel` launch.

    B2KI_SO=/tmp/b2ki_ncu_intra_k64_v64.so python .../ncu_b2_k128_intra_driver.py \
        --shape b8t2048h8 --K 64 --V 64 --which intra
"""
import argparse
import ctypes
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4)}
BT = 16
SCALE = 1.0 / (128 ** 0.5)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def build_inputs(shape, K, V, seed=0):
    B, T, H = SHAPES[shape]
    dev = "cuda"
    NT = (T + BT - 1) // BT
    gg = torch.Generator(device=dev).manual_seed(seed)
    mk = lambda *s: torch.empty(*s, device=dev)
    d = {}
    d["v"] = torch.randn(B, T, H, V, device=dev, generator=gg)
    d["vn"] = torch.randn(B, T, H, V, device=dev, generator=gg)
    d["do"] = torch.randn(B, T, H, V, device=dev, generator=gg)
    d["dv"] = torch.randn(B, T, H, V, device=dev, generator=gg)
    d["dw"] = torch.randn(B, T, H, K, device=dev, generator=gg)
    d["k"] = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    d["q"] = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    d["g"] = -torch.rand(B, T, H, K, device=dev, generator=gg)
    d["beta"] = torch.sigmoid(torch.randn(B, T, H, device=dev, generator=gg))
    d["A"] = torch.randn(B, T, H, BT, device=dev, generator=gg) * 0.1
    d["h"] = torch.randn(B, NT, H, K, V, device=dev, generator=gg)
    d["dh"] = torch.randn(B, NT, H, K, V, device=dev, generator=gg)
    d["dq"], d["dk"], d["dwo"] = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    d["dgk"] = mk(B, NT, H, K)
    d["dv2"], d["dvb"] = mk(B, T, H, V), mk(B, T, H, V)
    d["dkgb"] = mk(B, T, H, K)
    d["dqg"], d["dkg"], d["dg"] = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    d["db"] = torch.zeros(B, T, H, device=dev)
    d["dApre"], d["dA"] = mk(B, T, H, BT), mk(B, T, H, BT)
    o_t = torch.arange(BT, device=dev)
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device=dev, generator=gg) * low_incl
    d["dAqk"] = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    return d


def launch_reps(lib, d, shape, which, reps):
    B, T, H = SHAPES[shape]
    NT = (T + BT - 1) // BT
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    head = (ptr(d["v"]), ptr(d["vn"]), ptr(d["h"]), ptr(d["dh"]), ptr(d["do"]), ptr(d["dv"]),
            ptr(d["k"]), ptr(d["g"]), ptr(d["dw"]),
            ptr(d["dq"]), ptr(d["dk"]), ptr(d["dwo"]), ptr(d["dgk"]), ptr(d["A"]), ptr(d["beta"]),
            ptr(d["dv2"]), ptr(d["dvb"]), ptr(d["dkgb"]), ptr(d["q"]), ptr(d["k"]), ptr(d["g"]),
            ptr(d["dqg"]), ptr(d["dkg"]), ptr(d["dg"]), SCALE,
            ptr(d["db"]), ptr(d["dApre"]), ptr(d["A"]), ptr(d["dA"]))
    if which == "prod":
        fn = lib.b2prod_launch
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 4 + \
            [ctypes.c_int] * 4 + [ctypes.c_void_p]
        call = lambda: fn(*head, B, T, H, NT, st)
    else:
        fn = lib.b2prod_intra_launch
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 5 + \
            [ctypes.c_int] * 4 + [ctypes.c_void_p]
        call = lambda: fn(*head, ptr(d["dAqk"]), B, T, H, NT, st)
    rc = 0
    for _ in range(reps):
        rc = call()
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="b8t2048h8", choices=list(SHAPES))
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--which", default="intra", choices=["prod", "intra"])
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--so", default=os.environ.get("B2KI_SO", ""))
    args = ap.parse_args()
    d = build_inputs(args.shape, args.K, args.V)
    lib = ctypes.CDLL(args.so)
    rc = launch_reps(lib, d, args.shape, args.which, args.reps)
    if rc != 0:
        raise SystemExit("launch rc=%d" % rc)
    torch.cuda.synchronize()
    print("b2%s ok  shape=%s K=%d V=%d reps=%d"
          % (args.which, args.shape, args.K, args.V, args.reps))


if __name__ == "__main__":
    main()
