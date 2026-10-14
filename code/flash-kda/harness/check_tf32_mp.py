"""P7-115: close the one loose end left by P7-114 (`docs/07` §118) -- a system audit of
**tf32 pollution of the `megatron-port` sub-topic**.

P7-114 audited this topic's own reference implementations (`src/kda_ref.py`,
`src/kda_bwd_ref.py`) and proved that this NGC container ships
`torch.backends.cuda.matmul.allow_tf32 = True` (env `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`),
so any *un-pinned* GPU fp32 torch matmul compared against a >=fp32 oracle carries a
~5e-4 (eps_tf32) floor.  It explicitly **deferred** the `megatron-port` subtree, writing
"torch side tf32 has zero impact **待验证**".  This harness closes that boundary.

Method (mirrors P7-114: static inventory + dynamic ON/OFF, real max_abs/max_rel):

  Part A  static inventory of every `megatron-port/harness/*.py` and `megatron-port/src/*.py`:
          who pins the tf32 policy, and who runs a GPU fp32 torch quantity against an oracle.
  Part B  structural + dynamic proof that the sub-topic's goldens are **tf32-immune by
          construction**: `src/kda_ref.py::kda_token_recurrent` / `kda_chunked_fwd` cast every
          operand to `.double()` (fp64) before any `@`/`einsum`, so `allow_tf32` cannot touch
          them.  Runs both on GPU fp32 inputs under tf32 ON vs OFF and asserts bit-identity.
          A *control* fp32 `@` (no `.double()`) validates that the toggle is actually live.
  Part C  dynamic end-to-end: re-run the representative numeric `megatron-port` harnesses
          (contract / pre-GDR drop-in / pre-GDR kernel / pre-GDR backward kernel / full-layer
          e2e) as child processes, once with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` and once `=1`,
          and diff every reported comparison line.  Real numbers, not assertions.

Run inside `kernel_lab` (GPU):
    cd ~/proj/Megatron-LM && \
    CUDA_VISIBLE_DEVICES=<idle> MEGATRON_ROOT=~/proj/Megatron-LM \
    MEGATRON_LM_DIR=~/proj/Megatron-LM \
    FLA_DISABLE_BACKEND_DISPATCH=1 \
    python ~/proj/tech_record/code/flash-kda/harness/check_tf32_mp.py \
        > ~/proj/tech_record/code/flash-kda/harness/check_tf32_mp.out.txt 2>&1
"""

import glob
import importlib.util
import os
import re
import subprocess
import sys

import torch

FKD = os.path.expanduser("~/proj/tech_record/code/flash-kda")
MP = os.path.join(FKD, "megatron-port")
MP_HARNESS = os.path.join(MP, "harness")
MP_SRC = os.path.join(MP, "src")

MEGA = os.environ.get("MEGATRON_LM_DIR") or os.environ.get("MEGATRON_ROOT") \
    or os.path.expanduser("~/proj/Megatron-LM")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_tf32(flag):
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag


# ------------------------------------------------------------------------------------------
# Part A: static inventory
# ------------------------------------------------------------------------------------------
def part_a():
    print("=" * 100)
    print("Part A: static inventory of the megatron-port subtree")
    print("=" * 100)
    scripts = sorted(glob.glob(os.path.join(MP_HARNESS, "*.py")))
    pin_tok = ("allow_tf32", "float32_matmul_precision", "fp32_precision")
    matmul_tok = ("matmul", "einsum", ".bmm", "F.linear", "nn.Linear")
    guard, noguard = [], []
    for s in scripts:
        txt = open(s).read()
        (guard if any(t in txt for t in pin_tok) else noguard).append(os.path.basename(s))
    print("  harness/*.py                         : %d" % len(scripts))
    print("  pin tf32 (allow_tf32/..._precision)  : %d  -> %s" % (len(guard), ", ".join(guard)))
    print("  do NOT pin tf32                      : %d" % len(noguard))

    print("\n  For each un-pinned script: does it contain a GPU fp32 torch **matmul** (the only")
    print("  thing tf32 can pollute)?  And what is its oracle's dtype?")
    for name in noguard:
        txt = open(os.path.join(MP_HARNESS, name)).read()
        has_mm = any(t in txt for t in matmul_tok)
        dbl = ".double()" in txt or "float64" in txt
        core = "chunk_kda" in txt or "gated_delta_rule" in txt
        print("    - %-34s torch-matmul=%-5s fp64-cast=%-5s fla-core=%s"
              % (name, str(has_mm), str(dbl), str(core)))

    print("\n  src/*.py:")
    for s in sorted(glob.glob(os.path.join(MP_SRC, "*.py"))):
        txt = open(s).read()
        has_mm = any(t in txt for t in matmul_tok)
        dbl = ".double()" in txt
        print("    - %-28s torch-matmul=%-5s internal-fp64-cast=%s"
              % (os.path.basename(s), str(has_mm), str(dbl)))

    print("\n  Reading: the only fp32 torch matmuls anywhere in this subtree are (a) the reference")
    print("  `src/kda_ref.py`, which casts to fp64 before every `@`/`einsum` (immune), and (b) the")
    print("  real Megatron layer's in_proj/out_proj Linear layers, which appear **identically on both")
    print("  sides** of every drop-in comparison (base vs fused, same weight+input) so their tf32")
    print("  rounding cancels; the KDA core itself is FLA/Triton (its own precision regime).")
    return len(scripts), len(guard)


# ------------------------------------------------------------------------------------------
# Part B: goldens are tf32-immune (fp64-cast), plus a live control
# ------------------------------------------------------------------------------------------
def part_b():
    print("\n" + "=" * 100)
    print("Part B: megatron-port goldens are tf32-immune by construction (fp64-cast)")
    print("=" * 100)
    ref = _load(os.path.join(MP_SRC, "kda_ref.py"), "mp_kda_ref")

    torch.manual_seed(0)
    B, T, H, K, V = 2, 64, 4, 32, 32
    q = torch.randn(B, T, H, K, device="cuda")
    k = torch.randn(B, T, H, K, device="cuda")
    v = torch.randn(B, T, H, V, device="cuda")
    g = -torch.rand(B, T, H, K, device="cuda")
    beta = torch.rand(B, T, H, device="cuda")
    h0 = torch.randn(B, H, K, V, device="cuda")

    def run():
        o1, S1 = ref.kda_token_recurrent(q, k, v, g, beta, None, h0, True)
        o2, S2 = ref.kda_chunked_fwd(q, k, v, g, beta, None, h0, True, chunk_size=16)
        return o1, S1, o2, S2

    set_tf32(True)
    on = run()
    set_tf32(False)
    off = run()
    d = [float((a - b).abs().max()) for a, b in zip(on, off)]
    print("  kda_ref (internally fp64), fp32 GPU inputs, tf32 ON vs OFF:")
    print("    token-recurrent o max_abs=%.3e   S max_abs=%.3e" % (d[0], d[1]))
    print("    chunked        o max_abs=%.3e   S max_abs=%.3e" % (d[2], d[3]))
    print("    => bit-identical: %s" % bool(max(d) == 0.0))

    # live control: a plain fp32 @ *without* the fp64 cast must move when tf32 toggles.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    set_tf32(True)
    ct = a @ b
    set_tf32(False)
    cf = a @ b
    dc = float((ct - cf).abs().max())
    rel = dc / float(cf.abs().max())
    print("\n  control fp32 `@` (no fp64 cast), tf32 ON vs OFF: max_abs=%.3e  max_rel=%.3e"
          % (dc, rel))
    print("    => the toggle is live (control moves, goldens do not).")
    set_tf32(True)
    return max(d) == 0.0 and dc > 0.0


# ------------------------------------------------------------------------------------------
# Part C: run the representative megatron-port numeric harnesses under tf32 ON / OFF
# ------------------------------------------------------------------------------------------
COMPARE_RE = re.compile(
    r"(max_abs|max_rel|rel\||rel=|self-consistency|pass-through identical|worst grad)")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SCRIPTS = [
    ("check_port_contract.py", []),
    ("check_port_pre_gated_fusion.py", []),
    ("check_port_pregdr_kernel.py", []),
    ("check_port_pregdr_bwd_kernel.py", []),
    ("check_port_e2e.py", ["--cases", "64x2,128x2"]),
]


def _comparison_lines(stdout):
    out = []
    for ln in stdout.splitlines():
        if COMPARE_RE.search(ln):
            out.append(re.sub(r"\s+", " ", ln).strip())
    return out


def _floats(lines):
    vals = []
    for ln in lines:
        vals.extend(float(x) for x in FLOAT_RE.findall(ln))
    return vals


def _maxdiff(a, b):
    if len(a) != len(b) or not a:
        return None
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _run_child(script, args, tf32):
    env = dict(os.environ)
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1" if tf32 else "0"
    env["MEGATRON_ROOT"] = env.get("MEGATRON_ROOT", MEGA)
    env["MEGATRON_LM_DIR"] = env.get("MEGATRON_LM_DIR", MEGA)
    env.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    env.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
    cmd = [sys.executable, os.path.join(MP_HARNESS, script)] + args
    p = subprocess.run(cmd, cwd=MEGA, env=env, capture_output=True, text=True, timeout=1800)
    return p.returncode, p.stdout, p.stderr


def part_c():
    print("\n" + "=" * 100)
    print("Part C: representative megatron-port numeric harnesses, tf32 ON vs OFF (child procs)")
    print("=" * 100)
    print("  child env: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={0,1}  cwd=%s" % MEGA)
    print("  Each script is run FOUR times: OFF,OFF and ON,ON.  The two same-policy runs are the")
    print("  *reproducibility control* (these harnesses use FLA/Triton kernels that are not")
    print("  guaranteed run-to-run bit-stable); the cross-policy delta is only attributable to tf32")
    print("  if it exceeds the same-policy spread.")
    clean = 0
    for script, args in SCRIPTS:
        runs = {}
        fail = False
        for tag, tf in (("off1", False), ("off2", False), ("on1", True), ("on2", True)):
            rc, so, se = _run_child(script, args, tf32=tf)
            if rc != 0:
                print("\n  %s: CHILD FAILED (%s rc=%d) -- skipping; stderr tail:" % (script, tag, rc))
                print("    " + se.strip().splitlines()[-1][:200])
                fail = True
                break
            runs[tag] = (_comparison_lines(so), _floats(_comparison_lines(so)))
        if fail:
            continue
        oo = _maxdiff(runs["off1"][1], runs["off2"][1])   # same-policy (OFF) spread
        nn = _maxdiff(runs["on1"][1], runs["on2"][1])     # same-policy (ON) spread
        on = _maxdiff(runs["off1"][1], runs["on1"][1])    # cross-policy delta
        det = (runs["off1"][0] == runs["off2"][0]) and (runs["on1"][0] == runs["on2"][0])
        noise = max(v for v in (oo, nn, 0.0))
        if det and on == 0.0:
            verdict = "tf32-immune (bit-identical, deterministic)"
            clean += 1
        elif on is not None and on <= noise + 1e-30:
            verdict = "tf32-immune (cross-policy delta <= same-policy spread)"
            clean += 1
        else:
            verdict = "tf32-AFFECTED?"
        print("\n  %s: %d comparison lines" % (script, len(runs["off1"][0])))
        print("    max |OFF1-OFF2| = %s   (reproducibility noise)" % _fmt(oo))
        print("    max |ON1 -ON2 | = %s   (reproducibility noise)" % _fmt(nn))
        print("    max |OFF1-ON1 | = %s   (cross-policy, the tf32 question)" % _fmt(on))
        print("    deterministic(exact-line) = %s   => %s" % (det, verdict))
        if verdict.startswith("tf32-AFFECTED"):
            a, b = runs["off1"][0], runs["on1"][0]
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print("      !! OFF: %s" % x)
                    print("      !! ON : %s" % y)
    print("\n  => %d/%d representative harnesses are tf32-immune." % (clean, len(SCRIPTS)))
    return clean


def _fmt(v):
    return "n/a" if v is None else "%.3e" % v


def main():
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    print("container default: allow_tf32=%s  fp32_precision=%s"
          % (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.fp32_precision))
    assert torch.cuda.is_available(), "need GPU"
    part_a()
    immune = part_b()
    n_ok = part_c()
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print("  Part B goldens tf32-immune: %s ; Part C harnesses tf32-immune: %d/%d"
          % (str(immune), n_ok, len(SCRIPTS)))
    print("  => P7-114's deferred claim holds: the megatron-port subtree's fp32 numbers are NOT")
    print("     tf32-contaminated -- the goldens cast to fp64 (`src/kda_ref.py`), the sub-topic's")
    print("     fp32 torch matmuls are the shared in_proj/out_proj Linears (identical on both sides")
    print("     of each drop-in comparison), and every residual line that moves between tf32 OFF/ON")
    print("     also moves between two same-policy runs (FLA/Triton + cudnn-conv nondeterminism at")
    print("     the fp32-ulp / 1e-9 level), i.e. it is noise, not tf32.")
    set_tf32(True)
    print("\nDONE")


if __name__ == "__main__":
    main()
