"""Probe: compile-time registers/spills/smem of the P7-27 fused B1+B2 kernel vs the
unfused B1 (`dhu_bf16`) and B2 (`wy_dqkg`) kernels, to attribute the fusion slowdown."""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/proj/tech_record/code/flash-kda/harness"))
sys.path.insert(0, os.path.expanduser("~/proj/tech_record/code/flash-kda/src"))
from check_bwd_b1_bf16_e2e import prep, build_dhu_bf16, dhu_bf16_triton       # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                        # noqa: E402
from check_bwd_fused import build_fused, fused_b12_triton                     # noqa: E402
from check_bwd_ws_reuse import D, CHUNK                                       # noqa: E402


def kinfos(jit):
    out = []
    for dev, tup in jit.device_caches.items():
        cache = tup[0] if isinstance(tup, tuple) else tup
        for k, v in cache.items():
            try:
                out.append((v.n_regs, v.n_spills, v.metadata.shared, v.metadata.num_warps,
                            v.metadata.num_ctas))
            except Exception as e:  # pragma: no cover
                out.append(("?", "?", "?", "?", str(e)))
    return out


def main():
    scale = 1.0 / math.sqrt(D)
    lb = -5.0
    for (name, B, T, H, seed, gs) in [("T512H4", 1, 512, 4, 0, 1.0),
                                      ("B8T2048H8", 8, 2048, 8, 2, 1.0)]:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lb, seed, gs)
        fi2 = dict(fi)
        fi2["v_raw"] = inp["v"].float()
        dav_k = build_dav_kernel()
        dAqk, dvin = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        NT = fi["h"].shape[1]
        print(f"\n=== {name}: grid fused=(B*H)={B*H}, unfused B1 grid=(cdiv(V,32),B*H)="
              f"({(D+31)//32},{B*H}), B2 grid=(NT,B*H)=({NT},{B*H}) ===")

        fk = build_fused()
        fused_b12_triton(fk, fi2, qn, kn, do, dvin, scale, CHUNK, dht=dfin)
        torch.cuda.synchronize()
        for r in kinfos(fk):
            print(f"  fused    regs={r[0]} spill={r[1]} smem={r[2]} nw={r[3]}")

        dhu_k = build_dhu_bf16()
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        dhu_bf16_triton(dhu_k, fi_b, do, dvin, scale, CHUNK, dht=dfin)
        torch.cuda.synchronize()
        for r in kinfos(dhu_k):
            print(f"  B1 bf16  regs={r[0]} spill={r[1]} smem={r[2]} nw={r[3]}")

        dqkg_k = build_dqkg()
        dh = torch.randn(B, NT, H, D, D, device="cuda")
        dv2 = torch.randn(B, T, H, D, device="cuda")
        inp2 = dict(q=qn, k=kn, v=inp["v"].float(), v_new=fi["v_new"], g=fi["g_cum"],
                    beta=fi["beta"], A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
        wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        torch.cuda.synchronize()
        for r in kinfos(dqkg_k):
            print(f"  B2       regs={r[0]} spill={r[1]} smem={r[2]} nw={r[3]}")
    print("\nDONE")


if __name__ == "__main__":
    main()
