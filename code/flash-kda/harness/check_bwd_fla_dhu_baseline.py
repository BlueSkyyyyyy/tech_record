"""P7-34: reconcile the two FLA `dhu` baselines used across `docs/07`.

`docs/07` reports FLA's state-adjoint kernel `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`
with **two different magnitudes** for the same shape (`B2 T1024 H8`, D=128):

  * ~**23 us**  -- measured by *directly* calling `chunk_gated_delta_rule_bwd_dhu` with
                   **bf16** operands (P7-14/§17 .. P7-33/§36, the CUDA-probe sections);
  * ~**137 us** -- measured *inside* the full `torch.autograd` chain through
                   `chunk_kda` with **fp32** operands (P7-7/§10, P7-8/§11, P7-22/§25).

Both are used as the denominator of "xFLA" claims, so the ambiguity matters.  This script
isolates the two candidate confounds -- **dtype of the operands** and **direct call vs
in-graph call** -- and pins down what actually moves the number.

For each shape and dtype it reports:
  A. direct `chunk_gated_delta_rule_bwd_dhu` (bf16 / fp32 operands), kernel-only CUPTI;
  B. full FLA `chunk_kda` autograd chain (same dtype), the `dhu` kernel attributed per call;
  C. the autotune config FLA selected (BV / num_warps / num_stages) in each path.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_fla_dhu_baseline.py \
        > .../harness/check_bwd_fla_dhu_baseline.out.txt 2>&1
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dhu_kernel import fla_forward                              # noqa: E402
from check_bwd_dhu_dtype import cuda_kernel_us, make_inputs, as_dtype      # noqa: E402

D = 128
CHUNK = 64


def _configs(kernel):
    """Dump the autotuned Triton config FLA selected for this kernel (best-effort)."""
    out = []
    best = getattr(kernel, "best_config", None)
    if best is not None:
        out.append(f"      best_config -> BV={best.kwargs.get('BV')} "
                   f"nw={best.num_warps} ns={best.num_stages}")
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        for key, cfg in cache.items():
            out.append(f"      [{key}] -> {getattr(cfg, 'kwargs', cfg)}")
    for attr in ("device_caches", "device_cache"):
        dc = getattr(kernel, attr, None)
        if dc:
            try:
                for dev, entry in dc.items():
                    out.append(f"      {attr}[{dev}] -> {str(entry)[:200]}")
            except Exception as e:  # noqa: BLE001
                out.append(f"      {attr}: {e}")
    if not out:
        out.append("      (no cached config introspection available)")
    return out


def dhu_breakdown(label, fn, reps=50):
    """Profile one callable; return (dhu_us, {kernel_name: us})."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    seen = {}
    dhu = 0.0
    for ev in prof.key_averages():
        us = ev.self_device_time_total / reps
        if us <= 0.5:
            continue
        seen[ev.key] = seen.get(ev.key, 0.0) + us
        if "bwd_kernel_dhu" in ev.key:
            dhu += us
    return dhu, seen


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    scale = D ** -0.5
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv

    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]

    # ---------------------------------------------------------- A: direct call, dtype sweep
    print("\n" + "=" * 96)
    print("Part A: DIRECT chunk_gated_delta_rule_bwd_dhu (kernel-only CUPTI us/call)")
    print("  -> isolates operand dtype (bf16 vs fp32);  same arg shape for both")
    print("=" * 96)
    print(f"  {'shape':16s} | {'bf16':>9s} {'fp32':>9s} | {'fp32/bf16':>9s}")
    for (B, T, H) in shapes:
        res = {}
        for dt in (torch.bfloat16, torch.float32):
            q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
            q, k, v, g_log2, beta, h0 = (as_dtype(x, dt) for x in (q, k, v, g_log2, beta, h0))
            torch.manual_seed(99)
            do = as_dtype(torch.randn(B, T, H, D), dt)
            fi = fla_forward(q, k, v, g_log2, beta, h0, scale, CHUNK)
            _, dv = chunk_kda_bwd_dAv(q=q, k=k, v=fi["v_new"], do=do, A=fi["Aqk"],
                                      scale=scale, chunk_size=CHUNK)
            fn = lambda fi=fi, do=do, dv=dv, h0=h0: chunk_gated_delta_rule_bwd_dhu(
                q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"], h0=h0, dht=None,
                do=do, dv=dv, scale=scale, chunk_size=CHUNK)
            us, _ = dhu_breakdown(f"direct-{dt}", fn)
            res[str(dt).split(".")[-1]] = us
            del fi, dv
        print(f"  B{B} T{T} H{H:<8} | {res['bfloat16']:9.1f} {res['float32']:9.1f} | "
              f"{res['float32']/res['bfloat16']:9.2f}")

    # ---------------------------------------------------------- B: in-graph autograd call
    print("\n" + "=" * 96)
    print("Part B: FULL FLA chunk_kda autograd chain (as bench_bwd.py does it)")
    print("  -> the 'in-chain' FLA dhu baseline;  operand dtype follows the chain inputs")
    print("=" * 96)
    from fla.ops.kda import chunk_kda

    def make_chain(qn, kn, va, ba, ga, ha, do, dfin):
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = va.clone().requires_grad_(True)
        ba = ba.clone().requires_grad_(True)
        ga = ga.clone().requires_grad_(True)
        ha = ha.clone().requires_grad_(True)
        o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                         initial_state=ha.float().transpose(-1, -2),
                         output_final_state=True, use_qk_l2norm_in_kernel=False,
                         use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        loss = (o.float() * do).sum() + (s.float() * dfin).sum()
        holders = (qa, ka, va, ba, ga, ha)

        def run():
            loss.backward(retain_graph=True)
            for t in holders:
                t.grad = None
        return run

    print(f"  {'shape':16s} | {'chain dhu us':>12s} | {'direct bf16':>11s} | "
          f"{'chain/direct':>12s}")
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        # bench_bwd builds the chain in fp32 (normalized q/k, sigmoid beta, natural-log g)
        qn = torch.nn.functional.normalize(q.float(), dim=-1).cuda()
        kn = torch.nn.functional.normalize(k.float(), dim=-1).cuda()
        vv = v.float().cuda()
        ba = torch.sigmoid(beta.float()).cuda()
        ga = (g_log2 * math.log(2.0)).float().cuda()
        ha = h0.float().cuda()
        torch.manual_seed(99)
        do = torch.randn(B, T, H, D, device="cuda")
        dfin = torch.randn(B, H, D, D, device="cuda") * 0.01
        chain = make_chain(qn, kn, vv, ba, ga, ha, do, dfin)
        chain_us, seen = dhu_breakdown("chain", chain)
        # direct bf16 reference for the same shape
        q2, k2, v2, g2, b2, h2 = make_inputs(B, T, H)
        q2, k2, v2, g2, b2, h2 = (as_dtype(x, torch.bfloat16) for x in (q2, k2, v2, g2, b2, h2))
        torch.manual_seed(99)
        doB = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        fi = fla_forward(q2, k2, v2, g2, b2, h2, scale, CHUNK)
        _, dvB = chunk_kda_bwd_dAv(q=q2, k=k2, v=fi["v_new"], do=doB, A=fi["Aqk"],
                                   scale=scale, chunk_size=CHUNK)
        dirbf, _ = dhu_breakdown("direct-bf16", lambda fi=fi, do=doB, dv=dvB, h2=h2:
                                 chunk_gated_delta_rule_bwd_dhu(
                                     q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"],
                                     h0=h2, dht=None, do=do, dv=dv, scale=scale,
                                     chunk_size=CHUNK))
        print(f"  B{B} T{T} H{H:<8} | {chain_us:12.1f} | {dirbf:11.1f} | "
              f"{chain_us/dirbf:12.2f}")
        print("      chain kernels touching dhu:")
        for kname, us in sorted(seen.items(), key=lambda x: -x[1]):
            if "dhu" in kname or "recompute" in kname or "fwd_h" in kname:
                print(f"        {us:8.1f} us  {kname[:80]}")

    # ---------------------------------------------------------- D: full-chain dev, dtype sweep
    print("\n" + "=" * 96)
    print("Part D: FULL FLA chain end-to-end device time, fp32 vs bf16 (same graph)")
    print("  -> the honest denominator for chain-level 'dev/FLA' when our chain is bf16")
    print("=" * 96)
    from torch.profiler import ProfilerActivity, profile

    def chain_dev(qn, kn, va, ba, ga, ha, do, dfin, reps=50):
        chain = make_chain(qn, kn, va, ba, ga, ha, do, dfin)
        for _ in range(5):
            chain()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                chain()
            torch.cuda.synchronize()
        return sum(ev.self_device_time_total for ev in prof.key_averages()) / reps

    print(f"  {'shape':16s} | {'FLA fp32 dev':>13s} {'FLA bf16 dev':>13s} | {'bf16/fp32':>9s} | "
          f"{'dhu fp32':>9s} {'dhu bf16':>9s}")
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        torch.manual_seed(99)

        def build(dt):
            qn = torch.nn.functional.normalize(q.float(), dim=-1).to(dt).cuda()
            kn = torch.nn.functional.normalize(k.float(), dim=-1).to(dt).cuda()
            va = v.to(dt).cuda()
            ba = torch.sigmoid(beta.float()).to(dt).cuda()
            ga = (g_log2 * math.log(2.0)).to(dt).cuda()
            ha = h0.to(dt).cuda()
            do = torch.randn(B, T, H, D, device="cuda").to(dt)
            dfin = (torch.randn(B, H, D, D, device="cuda") * 0.01).float()
            return qn, kn, va, ba, ga, ha, do, dfin

        f32 = build(torch.float32)
        b16 = build(torch.bfloat16)
        dev32 = chain_dev(*f32)
        dev16 = chain_dev(*b16)
        _, s32 = dhu_breakdown("c32", make_chain(*f32))
        _, s16 = dhu_breakdown("c16", make_chain(*b16))
        d32 = sum(u for n, u in s32.items() if "bwd_kernel_dhu" in n)
        d16 = sum(u for n, u in s16.items() if "bwd_kernel_dhu" in n)
        print(f"  B{B} T{T} H{H:<8} | {dev32:13.1f} {dev16:13.1f} | {dev16/dev32:9.2f} | "
              f"{d32:9.1f} {d16:9.1f}")

    # ---------------------------------------------------------- C: selected autotune config
    print("\n" + "=" * 96)
    print("Part C: FLA autotune config selected for the dhu kernel (per call path)")
    print("=" * 96)
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 as K
    for line in _configs(K):
        print(line)

    print("\nDONE")


if __name__ == "__main__":
    main()
