# Copyright (c) 2026, tech_record contributors.
#
# megatron-port M9-1: align the *calling contract* of Megatron-LM
# ``megatron/core/ssm/gated_delta_net/kda.py`` with FLA ``chunk_kda`` and with the
# fp64 reference ``src/kda_ref.py``.
#
# This is an *interface* check, not another numerics sweep (that was M6/M7/M8):
# we replicate the exact tensor transforms ``kda.py`` applies before it hands tensors
# to FLA, run FLA the exact way ``_forward_compute`` does, and compare against the
# reference fed the same pre-processed operands.
#
# Parts:
#   A  SBHD -> BHSD transpose (`kda.py:533-535`) is exactly the layout the kernel wants.
#   B  `use_gate_in_kernel=True` + raw g (`kda.py:120-121,439-446`) == pre-activated g.
#   C  beta is a *separate* projection, sigmoided outside the kernel (`kda.py:181`).
#   D  A_log/dt_bias kept fp32 (`kda.py:117-118`) -- cast to bf16 and measure the damage.
#   E  l2norm outside the kernel (`common.py:431-432`, `use_qk_l2norm_in_kernel=False`).
#   F  GVA: Megatron `_compute_gates` emits H key-heads of g (`kda.py:180`), but FLA
#      requires `g` of shape `[B,T,HV,K]`; so KDA-in-Megatron does **not** support GVA
#      (docstring `kda.py:55-57`), unlike the FLA kernel / M7 reference.
#   G  training initial_state=None / output_final_state=False (`kda.py:441-442`), and the
#      FLA `initial_state must be float32` assertion (`chunk.py:597`).
#   H  static code audit of the contract lines + `gdn_pre_gated_delta_rule_fusion` gap.
#
# Usage (GPU, inside kernel_lab):
#   FLA_DISABLE_BACKEND_DISPATCH=1 python harness/check_port_contract.py
# Env:
#   MEGATRON_ROOT  default ~/proj/Megatron-LM

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from kda_ref import kda_gate, kda_l2norm, kda_chunked_fwd  # noqa: E402

MEGATRON_ROOT = os.environ.get("MEGATRON_ROOT", os.path.expanduser("~/proj/Megatron-LM"))
KDA_PY = os.path.join(MEGATRON_ROOT, "megatron/core/ssm/gated_delta_net/kda.py")
COMMON_PY = os.path.join(MEGATRON_ROOT, "megatron/core/ssm/gated_delta_net/common.py")
GDN_PY = os.path.join(MEGATRON_ROOT, "megatron/core/ssm/gated_delta_net/gdn.py")
FLA_CHUNK_PY = "/usr/local/lib/python3.12/dist-packages/fla/ops/kda/chunk.py"


# --------------------------------------------------------------------------------------
# The Megatron shim: the transforms kda.py/common.py apply before FLA (tp/cp size 1).
# --------------------------------------------------------------------------------------
def megatron_pre_gdr(qkvfg_sbhd, beta_logits_sbhd, *, num_key_heads, num_value_heads,
                     key_head_dim, value_head_dim, use_qk_l2norm=True):
    """Mirror ``kda.py::pre_gated_delta_rule`` (533-540) +
    ``common.py::_prepare_input_for_gated_delta_rule`` (421-443)."""
    qk = num_key_heads * key_head_dim
    vd = num_value_heads * value_head_dim
    B, S = qkvfg_sbhd.shape[1], qkvfg_sbhd.shape[0]

    # kda.py:533-535
    qkvfg = qkvfg_sbhd.transpose(0, 1)
    beta = beta_logits_sbhd.transpose(0, 1)
    qkv, raw_g, gate = torch.split(qkvfg, [2 * qk + vd, qk, vd], dim=-1)
    gate = gate.reshape(B, S, num_value_heads, value_head_dim)

    # common.py:421-428
    query_key, value = torch.split(qkv, [2 * qk, vd], dim=-1)
    query_key = query_key.reshape(B, S, -1, key_head_dim)
    value = value.reshape(B, S, -1, value_head_dim)

    # common.py:431-432
    if use_qk_l2norm:
        query_key = kda_l2norm(query_key.contiguous())
    # common.py:435-436
    query, key = torch.split(query_key, [num_key_heads, num_key_heads], dim=2)

    # common.py:438-442 (GVA)
    repeat_factor = num_value_heads // num_key_heads
    if repeat_factor > 1:
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    # kda.py:180-181
    raw_g = raw_g.reshape(B, S, num_key_heads, key_head_dim)
    beta = beta.reshape(B, S, num_key_heads).float().sigmoid()

    return {"q": query.contiguous(), "k": key.contiguous(), "v": value.contiguous(),
            "g": raw_g.contiguous(), "beta": beta.contiguous()}, gate


def raw_qk(qkvfg_sbhd, num_key_heads, key_head_dim, num_value_heads, value_head_dim):
    """The un-normalized q/k the shim sees (for the l2norm equivalence check)."""
    qk = num_key_heads * key_head_dim
    vd = num_value_heads * value_head_dim
    B, S = qkvfg_sbhd.shape[1], qkvfg_sbhd.shape[0]
    qkv = torch.split(qkvfg_sbhd.transpose(0, 1), [2 * qk + vd, qk, vd], dim=-1)[0]
    q = qkv[..., :qk].reshape(B, S, num_key_heads, key_head_dim)
    k = qkv[..., qk:2 * qk].reshape(B, S, num_key_heads, key_head_dim)
    return q, k


def make_case(B, S, H, HV, K, V, seed, dtype):
    torch.manual_seed(seed)
    dev = "cuda"
    qk, vd = H * K, HV * V
    qkvfg = torch.randn(S, B, 2 * qk + vd + qk + vd, device=dev, dtype=dtype)
    beta_logits = torch.randn(S, B, H, device=dev, dtype=dtype)
    A_log = torch.log(torch.empty(H, device=dev).uniform_(1.0, 16.0)).float()
    dt = torch.exp(torch.rand(H * K, device=dev)
                   * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
                   + torch.log(torch.tensor(0.001))).clamp(min=1e-4)
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).float()
    return qkvfg, beta_logits, A_log, dt_bias


def rel(a, b):
    return (a.float() - b.float()).abs().max().item() / (a.float().abs().max().item() + 1e-30)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="2,256,4,4,64,64")  # B,S,H,HV,K,V (HV==H default)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    B, S, H, HV, K, V = (int(x) for x in args.shape.split(","))

    assert torch.cuda.is_available(), "need GPU"
    from fla.ops.kda import chunk_kda

    print("=" * 78)
    print("megatron-port M9-1: calling-contract alignment")
    print(f"  shape B={B} S={S} H={H} HV={HV} K={K} V={V}  seed={args.seed}")
    print(f"  Megatron root: {MEGATRON_ROOT}")
    print(f"  FLA_DISABLE_BACKEND_DISPATCH={os.environ.get('FLA_DISABLE_BACKEND_DISPATCH')}")
    print("=" * 78)

    qkvfg, beta_logits, A_log, dt_bias = make_case(B, S, H, HV, K, V, args.seed, torch.float32)
    ki, gate = megatron_pre_gdr(qkvfg, beta_logits, num_key_heads=H, num_value_heads=HV,
                                key_head_dim=K, value_head_dim=V)

    # ---------------- Part A: SBHD transpose -----------------------------------------
    print("\n[Part A] SBHD -> BHSD transpose (kda.py:533-535)")
    qkvfg_bhsd = qkvfg.transpose(0, 1).contiguous()
    qkv1, _, _ = torch.split(qkvfg_bhsd, [2 * H * K + HV * V, H * K, HV * V], dim=-1)
    qk1 = kda_l2norm(qkv1[..., :2 * H * K].reshape(B, S, -1, K).contiguous())
    q1, k1 = torch.split(qk1, [H, H], dim=2)
    q1 = q1.repeat_interleave(HV // H, 2); k1 = k1.repeat_interleave(HV // H, 2)
    eA = max((q1 - ki["q"]).abs().max().item(), (k1 - ki["k"]).abs().max().item())
    print(f"  BHSD-built q/k vs SBHD-transposed q/k: max_abs={eA:.2e}"
          f"   {'OK' if eA == 0 else 'BAD'}")

    # ---------------- Part B: whole contract vs reference ----------------------------
    print("\n[Part B] whole contract vs fp64 reference (gate-in-kernel, l2norm outside)")
    o_fla, ht_fla = chunk_kda(
        ki["q"], ki["k"], ki["v"], ki["g"], ki["beta"],
        A_log=A_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=False,   # kda.py:443
        use_gate_in_kernel=True,         # kda.py:444 / :121
        safe_gate=False, lower_bound=None,
    )
    g_act = kda_gate(ki["g"], A_log, dt_bias)
    o_ref, ht_ref = kda_chunked_fwd(ki["q"], ki["k"], ki["v"], g_act, ki["beta"],
                                    initial_state=None, output_final_state=True, chunk_size=64)
    eo, eh = rel(o_fla, o_ref), rel(ht_fla, ht_ref)
    print(f"  FLA(kernel) vs ref: rel|do|={eo:.2e}  rel|dh|={eh:.2e}"
          f"   {'OK' if max(eo, eh) < 5e-3 else 'BAD'}")

    # ---------------- Part C: beta sigmoid outside -----------------------------------
    print("\n[Part C] beta sigmoid applied *outside* the kernel (kda.py:181)")
    b_manual = torch.sigmoid(beta_logits.transpose(0, 1).reshape(B, S, H).float())
    eb = (b_manual - ki["beta"]).abs().max().item()
    print(f"  sigmoid(raw_logits) vs shim beta max_abs={eb:.2e}   {'OK' if eb == 0 else 'BAD'}")

    # ---------------- Part D: fp32 gate params ---------------------------------------
    print("\n[Part D] A_log/dt_bias must stay fp32 (kda.py:117-118 mark_keep_in_fp32)")
    g_fp32 = kda_gate(ki["g"], A_log, dt_bias)
    g_bf16 = kda_gate(ki["g"], A_log.bfloat16().float(), dt_bias.bfloat16().float())
    gd = (g_fp32 - g_bf16).abs().max().item()
    print(f"  gate bf16 params vs fp32: max_abs={gd:.2e} "
          f"max_rel={gd / (g_fp32.abs().max().item() + 1e-30):.2e}"
          f"  (g range [{g_fp32.min().item():.3f}, {g_fp32.max().item():.3f}])")

    # ---------------- Part E: l2norm outside vs in-kernel ----------------------------
    print("\n[Part E] l2norm outside the kernel (common.py:431 / kda.py:443)")
    q_raw, k_raw = raw_qk(qkvfg, H, K, HV, V)
    q_raw = q_raw.repeat_interleave(HV // H, 2); k_raw = k_raw.repeat_interleave(HV // H, 2)
    o_kernnorm, _ = chunk_kda(q_raw.contiguous(), k_raw.contiguous(), ki["v"], ki["g"], ki["beta"],
                              A_log=A_log, dt_bias=dt_bias, use_gate_in_kernel=True,
                              use_qk_l2norm_in_kernel=True)
    eE = rel(o_fla, o_kernnorm)
    print(f"  l2norm(outside) vs l2norm(in-kernel): rel|do|={eE:.2e}"
          f"   {'OK' if eE < 5e-3 else 'BAD'}")

    # ---------------- Part F: GVA not wired into Megatron KDA ------------------------
    print("\n[Part F] GVA: Megatron KDA emits H-head g; FLA needs [B,T,HV,K]")
    H2, HV2 = 2, 4
    qkvfg2, beta2, A2, dt2 = make_case(B, S, H2, HV2, K, V, seed=1, dtype=torch.float32)
    ki2, _ = megatron_pre_gdr(qkvfg2, beta2, num_key_heads=H2, num_value_heads=HV2,
                              key_head_dim=K, value_head_dim=V)
    print(f"  Megatron _compute_gates -> g.shape={tuple(ki2['g'].shape)}  "
          f"v.shape={tuple(ki2['v'].shape)}")
    try:
        chunk_kda(ki2["q"], ki2["k"], ki2["v"], ki2["g"], ki2["beta"],
                  A_log=A2, dt_bias=dt2, use_gate_in_kernel=True)
        print("  FLA accepted H-head g with HV>H (unexpected!)  BAD")
    except AssertionError as e:
        print(f"  FLA rejects g.shape {tuple(ki2['g'].shape)} for HV={HV2}: {str(e)[:70]}"
              f"   OK  => KDA-in-Megatron requires num_value_heads == num_key_heads")

    # ---------------- Part G: initial_state ------------------------------------------
    print("\n[Part G] initial_state contract (kda.py:441 initial_state=None)")
    try:
        chunk_kda(ki["q"], ki["k"], ki["v"], ki["g"], ki["beta"], A_log=A_log,
                  dt_bias=dt_bias, use_gate_in_kernel=True,
                  initial_state=torch.zeros(B, H, K, V, device="cuda", dtype=torch.bfloat16))
        print("  bf16 initial_state accepted (unexpected!)  BAD")
    except AssertionError as e:
        print(f"  bf16 initial_state rejected: {str(e)[:46]}   OK")
    o_z, _ = chunk_kda(ki["q"], ki["k"], ki["v"], ki["g"], ki["beta"], A_log=A_log,
                       dt_bias=dt_bias, use_gate_in_kernel=True,
                       initial_state=torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32))
    eG = rel(o_fla, o_z)
    print(f"  zero fp32 initial_state vs None: rel|do|={eG:.2e}   {'OK' if eG == 0 else 'BAD'}")

    # ---------------- Part H: static code audit --------------------------------------
    print("\n[Part H] static contract audit + fusion gap")
    audit = [
        (KDA_PY, "self.use_gate_in_kernel = True", "raw g handed to kernel"),
        (KDA_PY, "mark_keep_in_fp32(self.dt_bias)", "dt_bias fp32"),
        (KDA_PY, "mark_keep_in_fp32(self.A_log)", "A_log fp32"),
        (KDA_PY, "use_qk_l2norm_in_kernel=False", "l2norm outside kernel"),
        (KDA_PY, "is not implemented for KDA yet", "fusion gap"),
        (KDA_PY, "self.gated_delta_rule = chunk_kda", "kernel entry"),
        (KDA_PY, "supports training with equal query/key and value", "GVA not supported"),
        (COMMON_PY, "query_key = l2norm(query_key.contiguous())", "l2norm applied"),
        (COMMON_PY, "query = query.repeat_interleave(repeat_factor, dim=2)", "GVA q"),
        (COMMON_PY, "key = key.repeat_interleave(repeat_factor, dim=2)", "GVA k"),
        (FLA_CHUNK_PY, "initial_state must be in float32", "FLA fp32 state assert"),
        (FLA_CHUNK_PY, "A_log must be provided when use_gate_in_kernel=True", "FLA gate assert"),
    ]
    ok = True
    for path, needle, why in audit:
        try:
            txt = open(path).read()
        except OSError:
            print(f"  [skip] {path}"); continue
        found = needle in txt
        ok &= found
        print(f"  {'OK ' if found else 'MISS'} {why:26s} <- {os.path.basename(path)}: {needle[:44]!r}")
    try:
        gdn = open(GDN_PY).read()
        print(f"  GDN fusion path present in gdn.py: "
              f"{'gdn_pre_gated_delta_rule_fusion' in gdn}")
    except OSError:
        pass

    print("\n" + "=" * 78)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
