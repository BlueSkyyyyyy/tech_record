"""P7-84 `ncu` driver: profile the CUDA B2 **production launch** at `K != V` (K=128 fixed,
V in {64,128,256}), for the three production knobs whose *kernel-level* attribution had only ever
been measured at `K == V == 128`:

  tag=fp32   `-DB2M_PROD -DB2M_HDW`
  tag=bf16   `-DB2M_PROD -DB2M_HDW -DB2M_BF16`      (the five streamed operands are bf16)
  tag=intra  `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`     (`b2prod_intra_launch`, extra `dAqk` input)

Question (docs/07 section 86.5, the named next optional increment): does bf16's / intra-fold's
`ncu` pipe attribution stay put as `V` moves?  The `.so` is pre-built **outside** ncu (nvcc hangs
under ncu in this container, P7-16), so this driver is dependency-free (torch + ctypes) and the
profile holds exactly one `b2mma_kernel` launch.

    B2KV_SO=/tmp/b2kv_prod_bf16_v256.so B2KV_TAG=bf16 B2KV_V=256 \
        python .../ncu_b2_kv_driver.py --shape b8t2048h8
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="b8t2048h8", choices=list(SHAPES))
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    B, T, H = SHAPES[args.shape]
    dev = "cuda"
    K = 128
    V = int(os.environ.get("B2KV_V", "128"))
    tag = os.environ.get("B2KV_TAG", "fp32")
    intra = tag == "intra"
    bf16 = tag == "bf16"
    NT = (T + BT - 1) // BT
    gg = torch.Generator(device=dev).manual_seed(0)
    mk = lambda *s: torch.empty(*s, device=dev)
    st5 = dict(dtype=torch.bfloat16) if bf16 else dict()
    v = torch.randn(B, T, H, V, device=dev, generator=gg)
    vn = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    do = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    dv = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    dw = torch.randn(B, T, H, K, device=dev, generator=gg)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    g = -torch.rand(B, T, H, K, device=dev, generator=gg)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, generator=gg))
    A = torch.randn(B, T, H, BT, device=dev, generator=gg) * 0.1
    h = torch.randn(B, NT, H, K, V, device=dev, generator=gg, **st5)
    dh = torch.randn(B, NT, H, K, V, device=dev, generator=gg, **st5)
    # `intra` consumes dAqk = tril(randn) in the [B,T,H,BT] layout.
    o_t = torch.arange(BT, device=dev)
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device=dev, generator=gg) * low_incl
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    dq, dk, dwo = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    dgk = mk(B, NT, H, K)
    dv2, dvb = mk(B, T, H, V), mk(B, T, H, V)
    dkgb = mk(B, T, H, K)
    dqg, dkg, dg = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    db = torch.zeros(B, T, H, device=dev)
    dApre, dA = mk(B, T, H, BT), mk(B, T, H, BT)

    lib = ctypes.CDLL(os.environ["B2KV_SO"])
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if intra:
        fn = lib.b2prod_intra_launch
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 5 +
                       [ctypes.c_int] * 4 + [ctypes.c_void_p])
        for _ in range(args.reps):
            rc = fn(ptr(v), ptr(vn), ptr(h), ptr(dh), ptr(do), ptr(dv), ptr(k), ptr(g), ptr(dw),
                    ptr(dq), ptr(dk), ptr(dwo), ptr(dgk), ptr(A), ptr(beta), ptr(dv2), ptr(dvb),
                    ptr(dkgb), ptr(q), ptr(k), ptr(g), ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                    ptr(db), ptr(dApre), ptr(A), ptr(dA), ptr(dAqk), B, T, H, NT, st)
    else:
        fn = lib.b2prod_launch
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 4 +
                       [ctypes.c_int] * 4 + [ctypes.c_void_p])
        for _ in range(args.reps):
            rc = fn(ptr(v), ptr(vn), ptr(h), ptr(dh), ptr(do), ptr(dv), ptr(k), ptr(g), ptr(dw),
                    ptr(dq), ptr(dk), ptr(dwo), ptr(dgk), ptr(A), ptr(beta), ptr(dv2), ptr(dvb),
                    ptr(dkgb), ptr(q), ptr(k), ptr(g), ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                    ptr(db), ptr(dApre), ptr(A), ptr(dA), B, T, H, NT, st)
    if rc != 0:
        raise SystemExit("launch rc=%d" % rc)
    torch.cuda.synchronize()
    print("b2prod ok  shape=%s V=%d tag=%s reps=%d" % (args.shape, V, tag, args.reps))


if __name__ == "__main__":
    main()
