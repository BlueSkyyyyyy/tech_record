# M13-3: per-kernel profile of the KDA-specific *fused pre-GDR* backward (native Triton).
#
# M13-2 (``src/kda_pregdr_bwd.py``) gave the fused pre-GDR a native Triton backward made of
# four kernels (L2-norm bwd, SiLU bwd, depthwise causal-conv bwd, sigmoid bwd) plus several
# fp32 [B,T,C] intermediates (``dqk_norm`` cat, ``dqk_in``, ``dconv``, ``da``).  Its Part D
# showed the native family **wins wall** (host launch tax) but **loses device time**
# (0.69-0.99x).  This harness attributes that regression: it prints the per-kernel
# self-device time of the native backward so we can see which of the four kernels / the
# elementwise materialisations dominate, before touching the implementation.
#
# Run (GPU, single process; pick an idle card):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && CUDA_VISIBLE_DEVICES=1 MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/profile_port_pregdr_bwd.py"

import argparse
import importlib.util
import os
import socket
import sys
import types

os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch


def _import_local(modname, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "src", filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_m13():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "check_port_pregdr_kernel.py")
    spec = importlib.util.spec_from_file_location("m13", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_grads(outs, seed=0):
    g = torch.Generator(device=outs[0].device).manual_seed(seed)
    return [torch.randn(o.shape, device=o.device, generator=g).float() for o in outs]


def profile_backward(name, fn, iters=50, warmup=20):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    print("  [%s] per-kernel self device time (us/iter):" % name)
    rows = []
    for e in prof.key_averages():
        if e.self_device_time_total <= 0:
            continue
        rows.append((e.self_device_time_total / iters, e.count / iters, e.key))
    rows.sort(reverse=True)
    for us, cnt, key in rows:
        print("     %8.2f us  x%-5.2f  %s" % (us, cnt, key[:80]))
    return sum(r[0] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    args = ap.parse_args()
    dtypes = [getattr(torch, t.strip()) for t in args.dtypes.split(",")]

    m13 = _import_m13()
    mega = m13._import_megatron()
    print("Megatron-LM:", mega)
    bwd = _import_local("kda_pregdr_bwd", "kda_pregdr_bwd.py")
    m13.init_dist()

    cases = [(64, 2), (256, 2), (1024, 2), (1024, 8)]
    for dtype in dtypes:
        print("=== dtype=%s ===" % str(dtype).split(".")[-1])
        m13.model_parallel_cuda_manual_seed(123)
        kda = m13.build_kda(m13.make_config(dtype))
        dims = bwd._dims(kda)
        for (T, B) in cases:
            cap = m13._capture(kda, dtype, T, B, seed=T * 5 + B)
            qkvfg, beta = cap["qkvfg"], cap["beta_in"]
            w = kda.conv1d.weight.detach()
            b = kda.conv1d.bias
            outs, saves = bwd._forward_raw_save(qkvfg, beta, w, b, dims)
            up = _mk_grads(outs, seed=T + B)

            tots = {}
            for tag, fused in (("unfused (M13-2)", False), ("fused (M13-3)", True)):
                def run(fused=fused):
                    bwd._backward_raw(qkvfg, beta, w, saves, up, dims, fused=fused)

                tots[tag] = profile_backward("T=%d B=%d %s" % (T, B, tag), run)
            print("     TOTAL self-dev: unfused=%.2f us  fused=%.2f us  (%.3fx)"
                  % (tots["unfused (M13-2)"], tots["fused (M13-3)"],
                     tots["fused (M13-3)"] / tots["unfused (M13-2)"]))

    m13.ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
