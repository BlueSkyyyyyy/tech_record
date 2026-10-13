"""P7-91 `ncu` driver: pipe attribution of the six-segment production kernel's bf16-native
entries (`prod` / `bf16` / `bf16mma` / `bf16_vdw`) as **K** moves in {64,128,256} -- the
"no pipe attribution" gap that section 93.6 (P7-90) named for the `K`-generalized production
kernel.

The `.so` is pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16), so this
driver is dependency-free (torch + ctypes) and the profile holds exactly one `b2mma_kernel`
launch.

    B2KDT_SO=/tmp/b2kdt_ncu_bf16_k64_v64.so B2KDT_TAG=bf16 B2KDT_K=64 \
        python .../ncu_b2_k128_dtype_driver.py --shape b8t2048h8
"""
import argparse
import ctypes
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4)}
BT = 16
SCALE = 1.0 / (128 ** 0.5)

# tags whose five streamed operands (`vn/do/dv/h/dh`) are bf16 on the wire
# P7-104: extended to the `BVK=64` / TMA bf16 builds (all tags whose name starts with "bf16").
FIVE_BF16 = {"bf16", "bf16mma", "bf16_vdw", "bf1664", "bf16mma64",
             "bf16tma", "bf16mma_tma"}
# tags whose raw `v` is bf16 on the wire (`BF16_VDW` non-TMA, or any bf16 **TMA** staging, §79)
V_BF16 = {"bf16_vdw", "bf16tma", "bf16mma_tma"}


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def build_inputs(shape, K, V, tag, seed=0):
    B, T, H = SHAPES[shape]
    dev = "cuda"
    NT = (T + BT - 1) // BT
    gg = torch.Generator(device=dev).manual_seed(seed)
    mk = lambda *s: torch.empty(*s, device=dev)
    st5 = dict(dtype=torch.bfloat16) if tag in FIVE_BF16 else dict()
    stv = dict(dtype=torch.bfloat16) if tag in V_BF16 else dict()
    d = {}
    d["v"] = torch.randn(B, T, H, V, device=dev, generator=gg, **stv)
    d["vn"] = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    d["do"] = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    d["dv"] = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    d["dw"] = torch.randn(B, T, H, K, device=dev, generator=gg)
    d["k"] = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    d["q"] = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    d["g"] = -torch.rand(B, T, H, K, device=dev, generator=gg)
    d["beta"] = torch.sigmoid(torch.randn(B, T, H, device=dev, generator=gg))
    d["A"] = torch.randn(B, T, H, BT, device=dev, generator=gg) * 0.1
    d["h"] = torch.randn(B, NT, H, K, V, device=dev, generator=gg, **st5)
    d["dh"] = torch.randn(B, NT, H, K, V, device=dev, generator=gg, **st5)
    d["dq"], d["dk"], d["dwo"] = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    d["dgk"] = mk(B, NT, H, K)
    d["dv2"], d["dvb"] = mk(B, T, H, V), mk(B, T, H, V)
    d["dkgb"] = mk(B, T, H, K)
    d["dqg"], d["dkg"], d["dg"] = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    d["db"] = torch.zeros(B, T, H, device=dev)
    d["dApre"], d["dA"] = mk(B, T, H, BT), mk(B, T, H, BT)
    return d


def launch_reps(lib, d, shape, reps):
    B, T, H = SHAPES[shape]
    NT = (T + BT - 1) // BT
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    fn = lib.b2prod_launch
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 4 +
                   [ctypes.c_int] * 4 + [ctypes.c_void_p])
    for _ in range(reps):
        rc = fn(ptr(d["v"]), ptr(d["vn"]), ptr(d["h"]), ptr(d["dh"]), ptr(d["do"]), ptr(d["dv"]),
                ptr(d["k"]), ptr(d["g"]), ptr(d["dw"]),
                ptr(d["dq"]), ptr(d["dk"]), ptr(d["dwo"]), ptr(d["dgk"]), ptr(d["A"]), ptr(d["beta"]),
                ptr(d["dv2"]), ptr(d["dvb"]), ptr(d["dkgb"]), ptr(d["q"]), ptr(d["k"]), ptr(d["g"]),
                ptr(d["dqg"]), ptr(d["dkg"]), ptr(d["dg"]), SCALE,
                ptr(d["db"]), ptr(d["dApre"]), ptr(d["A"]), ptr(d["dA"]), B, T, H, NT, st)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="b8t2048h8", choices=list(SHAPES))
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--tag", default=os.environ.get("B2KDT_TAG", "prod"))
    ap.add_argument("--K", type=int, default=int(os.environ.get("B2KDT_K", "128")))
    ap.add_argument("--V", type=int, default=int(os.environ.get("B2KDT_V", "128")))
    ap.add_argument("--so", default=os.environ.get("B2KDT_SO", ""))
    args = ap.parse_args()
    d = build_inputs(args.shape, args.K, args.V, args.tag)
    lib = ctypes.CDLL(args.so)
    rc = launch_reps(lib, d, args.shape, args.reps)
    if rc != 0:
        raise SystemExit("launch rc=%d" % rc)
    torch.cuda.synchronize()
    print("b2prod ok  shape=%s K=%d V=%d tag=%s reps=%d"
          % (args.shape, args.K, args.V, args.tag, args.reps))


if __name__ == "__main__":
    main()
