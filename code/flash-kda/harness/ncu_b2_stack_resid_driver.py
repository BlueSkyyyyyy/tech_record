"""P7-108 `ncu` driver for the **`(64,256)` bank-conflict secondary interaction** of the
`bf16mma_tma` multi-knob stack (closes the item flagged "机制待验证" in docs/07 §110.3/§110.6 and the
ROADMAP "当前进度 / 下一步").

§110 (P7-107) found the `bmt_da_intra` bank-conflict count is *sub-additive* only at `(K,V)=(64,256)`
(residual `-0.611 M`, ~2%, reproducible); the other 8 grid points are additive to within 0.18 M.
This driver launches exactly ONE `b2mma_kernel` (one `.so`, one tag) so `ncu` reads the pipe-level
mechanism.  Unlike `ncu_b2_stack_driver.py` it accepts an **arbitrary `(B,T,H)`** so we can sweep the
grid size at a fixed `(K,V)` (section 111.2), with the same operand construction (all `bf16` streamed
five + raw-`v`).  The `.so` is pre-built outside `ncu` (nvcc hangs under `ncu` here, P7-16).

    B2STK_SO=/tmp/b2res_bmt_da_intra_64_256.so \
        python .../ncu_b2_stack_resid_driver.py --B 8 --T 2048 --H 8 --K 64 --V 256 --tag bmt_da_intra
"""
import argparse
import ctypes
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

BT = 16
SCALE = 1.0 / (128 ** 0.5)
# all `bmt*` stack builds are `-DB2M_BF16` (five streamed operands bf16) and `-DB2M_TMA`
# (raw `v` staged bf16), see ncu_b2_stack_driver.py.
FIVE_BF16 = True
V_BF16 = True


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def build_inputs(B, T, H, K, V, seed=0):
    dev = "cuda"
    NT = (T + BT - 1) // BT
    gg = torch.Generator(device=dev).manual_seed(seed)
    mk = lambda *s: torch.empty(*s, device=dev)
    st5 = dict(dtype=torch.bfloat16) if FIVE_BF16 else dict()
    stv = dict(dtype=torch.bfloat16) if V_BF16 else dict()
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
    d["dAqk"] = torch.randn(B, T, H, BT, device=dev, generator=gg) * 0.1
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


def launch_reps(lib, d, B, T, H, reps, intra):
    NT = (T + BT - 1) // BT
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    name = "b2prod_intra_launch" if intra else "b2prod_launch"
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    npost = 5 if intra else 4
    fn.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * npost +
                   [ctypes.c_int] * 4 + [ctypes.c_void_p])
    args = [ptr(d["v"]), ptr(d["vn"]), ptr(d["h"]), ptr(d["dh"]), ptr(d["do"]), ptr(d["dv"]),
            ptr(d["k"]), ptr(d["g"]), ptr(d["dw"]),
            ptr(d["dq"]), ptr(d["dk"]), ptr(d["dwo"]), ptr(d["dgk"]), ptr(d["A"]), ptr(d["beta"]),
            ptr(d["dv2"]), ptr(d["dvb"]), ptr(d["dkgb"]), ptr(d["q"]), ptr(d["k"]), ptr(d["g"]),
            ptr(d["dqg"]), ptr(d["dkg"]), ptr(d["dg"]), SCALE,
            ptr(d["db"]), ptr(d["dApre"]), ptr(d["A"]), ptr(d["dA"])]
    if intra:
        args.append(ptr(d["dAqk"]))
    args += [B, T, H, NT, st]
    rc = 0
    for _ in range(reps):
        rc = fn(*args)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--K", type=int, default=int(os.environ.get("B2STK_K", "64")))
    ap.add_argument("--V", type=int, default=int(os.environ.get("B2STK_V", "256")))
    ap.add_argument("--tag", default=os.environ.get("B2STK_TAG", "bmt"))
    ap.add_argument("--so", default=os.environ.get("B2STK_SO", ""))
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    d = build_inputs(args.B, args.T, args.H, args.K, args.V)
    lib = ctypes.CDLL(args.so)
    rc = launch_reps(lib, d, args.B, args.T, args.H, args.reps, "intra" in args.tag)
    if rc != 0:
        raise SystemExit("launch rc=%d" % rc)
    torch.cuda.synchronize()
    print("b2prod ok B=%d T=%d H=%d K=%d V=%d tag=%s reps=%d"
          % (args.B, args.T, args.H, args.K, args.V, args.tag, args.reps))


if __name__ == "__main__":
    main()
