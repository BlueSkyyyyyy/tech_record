"""P7-106 `ncu` driver: pipe/resource attribution of the **`bf16mma_tma` multi-knob stack** on the
K2-shaped CUDA B2 production kernel -- the "no `ncu`" gap that section 108.6 (P7-105) names verbatim
("边界：... 未做 ncu（沿用 §107 读侧剖析）").

P7-105 proved the three knobs (`-DB2M_DAMMA` / `-DB2M_INTRA` / `BF16_VDW`) are **value-independent**
when stacked on the `bmt = PROD+HDW+BF16+BF16MMA+BVK=64+TMA` base (bit-identical twins); it never
looked at whether their **resource / pipe** effects are independent.  This driver runs one isolated
`b2mma_kernel` launch per build so `ncu` can read the pipe-level mechanism:

  bmt            bf16mma_tma                      (base)
  bmt_da         + -DB2M_DAMMA                     (DA -> mma.sync.tf32)
  bmt_intra      + -DB2M_INTRA                     (B3 diagonal folded; extra `dAqk` input)
  bmt_da_intra   + -DB2M_DAMMA -DB2M_INTRA         (both; the new corner)

The `.so` is pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16), so this
driver is dependency-free (torch + ctypes) and holds exactly one `b2mma_kernel` launch per rep.

    B2STK_SO=/tmp/b2stk_bmt_da_intra_128_128.so B2STK_TAG=bmt_da_intra B2STK_K=128 B2STK_V=128 \
        python .../ncu_b2_stack_driver.py --shape b8t2048h8
"""
import argparse
import ctypes
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4)}
BT = 16
SCALE = 1.0 / (128 ** 0.5)

# tags whose five streamed operands (`v_new/do/dv/h/dh`) are bf16 on the wire (all `bf16*` and the
# P7-105 `bmt*` stack builds are `-DB2M_BF16`).
FIVE_BF16 = {"bf16", "bf16mma", "bf16_vdw", "bf1664", "bf16mma64",
             "bf16tma", "bf16mma_tma",
             "bmt", "bmt_vdw", "bmt_da", "bmt_intra", "bmt_da_intra"}
# tags whose raw `v` is bf16 on the wire (any bf16 **TMA** staging, §79; the `bmt*` builds are TMA).
V_BF16 = {"bf16_vdw", "bf16tma", "bf16mma_tma",
          "bmt", "bmt_vdw", "bmt_da", "bmt_intra", "bmt_da_intra"}


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
    # `dAqk` is only read by the `-DB2M_INTRA` builds (the B3 diagonal fold).  Layout: the kernel
    # reads `[B, T, H, BT]` (row-rel `i_i*BC+o_i`, §63.6 pitfall) -- values are irrelevant for ncu.
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


def launch_reps(lib, d, shape, reps, intra):
    B, T, H = SHAPES[shape]
    NT = (T + BT - 1) // BT
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    name = "b2prod_intra_launch" if intra else "b2prod_launch"
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    npost = 5 if intra else 4        # intra appends one extra `dAqk` pointer right after `dA`
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
    ap.add_argument("--shape", default="b8t2048h8", choices=list(SHAPES))
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--tag", default=os.environ.get("B2STK_TAG", "bmt"))
    ap.add_argument("--K", type=int, default=int(os.environ.get("B2STK_K", "128")))
    ap.add_argument("--V", type=int, default=int(os.environ.get("B2STK_V", "128")))
    ap.add_argument("--so", default=os.environ.get("B2STK_SO", ""))
    args = ap.parse_args()
    intra = "intra" in args.tag
    d = build_inputs(args.shape, args.K, args.V, args.tag)
    lib = ctypes.CDLL(args.so)
    rc = launch_reps(lib, d, args.shape, args.reps, intra)
    if rc != 0:
        raise SystemExit("launch rc=%d" % rc)
    torch.cuda.synchronize()
    print("b2prod%s ok  shape=%s K=%d V=%d tag=%s reps=%d"
          % ("_intra" if intra else "", args.shape, args.K, args.V, args.tag, args.reps))


if __name__ == "__main__":
    main()
