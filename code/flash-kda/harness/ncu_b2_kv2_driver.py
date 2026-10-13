"""P7-85 `ncu` driver: extend P7-84's `K != V` pipe attribution (`ncu_b2_kv_driver.py`,
docs/07 section 87) to the **four production knobs it left unprofiled at K != V** -- the exact set
named in the P7-84 boundary note (docs/07 section 87.5):

  tag=bf16mma  `-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA`  (native bf16 mma, P7-74)
  tag=damma    `-DB2M_PROD -DB2M_HDW -DB2M_DAMMA`               (mma-ized dA, P7-66)
  tag=vdw      `-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16_VDW` (raw `v` also bf16-staged, P7-80)
  tag=tma      `-DB2M_PROD -DB2M_HDW -DB2M_TMA`                 (operands staged by TMA, P7-49)

P7-84 profiled `fp32`/`bf16`/`intra` at V in {64,128,256} and found the pipe **attribution
mechanism** invariant across V (bf16 = byte-halving + occupancy repair; intra = K-axis fixed tail).
This driver asks the same question for the four remaining knobs, and additionally collects the
`TMA` build (whose operand *wire* dtype is fp32 here, so `v`/`dw` stay fp32).

The `.so` is pre-built **outside** ncu (nvcc hangs under ncu in this container, P7-16), so this
driver is dependency-free (torch + ctypes) and the profile holds exactly one `b2mma_kernel` launch.

    B2KV2_SO=/tmp/b2kv2_vdw_v256.so B2KV2_TAG=vdw B2KV2_V=256 \
        python .../ncu_b2_kv2_driver.py --shape b8t2048h8

`build_inputs` / `launch_reps` are also imported by `check_bwd_b2_kv2_timing.py`.
"""
import argparse
import ctypes
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4)}
BT = 16
K = 128
SCALE = 1.0 / (128 ** 0.5)

# tags whose **five streamed operands** (`vn/do/dv/h/dh`) are bf16 on the wire
FIVE_BF16 = {"bf16", "bf16mma", "vdw", "bf16tma", "bf16mma_tma"}
# tags whose **raw `v`** is bf16 on the wire (BF16_VDW, or bf16 TMA staging both stage it bf16)
V_BF16 = {"vdw", "bf16tma", "bf16mma_tma"}


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def build_inputs(shape, V, tag, seed=0):
    """All tensors the production launcher consumes, with the tag's operand wire dtypes."""
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
    ap.add_argument("--tag", default=os.environ.get("B2KV2_TAG", "damma"))
    ap.add_argument("--V", type=int, default=int(os.environ.get("B2KV2_V", "128")))
    ap.add_argument("--so", default=os.environ.get("B2KV2_SO", ""))
    args = ap.parse_args()
    d = build_inputs(args.shape, args.V, args.tag)
    lib = ctypes.CDLL(args.so)
    rc = launch_reps(lib, d, args.shape, args.reps)
    if rc != 0:
        raise SystemExit("launch rc=%d" % rc)
    torch.cuda.synchronize()
    print("b2prod ok  shape=%s V=%d tag=%s reps=%d" % (args.shape, args.V, args.tag, args.reps))


if __name__ == "__main__":
    main()
