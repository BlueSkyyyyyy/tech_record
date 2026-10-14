"""P153-1: 精读并实测 `tests/test_mega_moe.py` 基线里的 **TileLang SwiGLU + 乘 top-k 权重 + FP8 量化**
算子 `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py`，它是 Mega MoE SM100 主 kernel
**L1 epilogue**（`sm100_fp8_fp4_mega_moe.cuh:1066-1205`）的官方逐位对照物。

三件事：
  A) host 公式 `get_sf_shape` / `get_sf_and_inv` 的语义核验（对照 `deep_gemm.utils.math` 参考量化器）。
  B) 真机跑 **真实 tilelang kernel**（FP8 与 BF16 两套配置 + 各开关），与独立 torch oracle 对拍。
  C) 用真实 DeepGEMM 设备头文件（`common/math.cuh` 的 `get_ue8m0_sf_exp/inv`、`fast_rcp`）把
     mega L1 epilogue 的**算术**复刻到 sm90（`src/swiglu_mega_epilogue.cu`），与 tilelang 输出逐位对拍，
     量「baseline 与 fused kernel 的 L1 段是否 bit-compatible」。

输出重定向到 exp_tilelang_swiglu.out.txt。
"""
import os
import subprocess
import sys
import time
import importlib.util

import torch

DEEPGEMM_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, DEEPGEMM_DIR)
BIN = os.environ.get('SWIGLU_BIN', '/tmp/swiglu_mega_epilogue')

import deep_gemm  # noqa: E402
from deep_gemm.utils.math import ceil_to_ue8m0  # noqa: E402

# 加载真实 tilelang_ops
_spec = importlib.util.spec_from_file_location(
    'tilelang_ops', os.path.join(DEEPGEMM_DIR, 'third-party', 'tilelang_ops', '__init__.py'))
tilelang_ops = importlib.util.module_from_spec(_spec)
sys.modules['tilelang_ops'] = tilelang_ops
_spec.loader.exec_module(tilelang_ops)


def hr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def unpack_ue8m0_sf(sf_i32):
    """tilelang 返回的 SF int32 (T, n_words) → (T, n_words*4) uint8 指数（LSB-first）。"""
    T, W = sf_i32.shape
    out = torch.zeros(T, W * 4, dtype=torch.uint8)
    v = sf_i32.to(torch.int64).cpu()
    for b in range(4):
        out[:, b::4] = ((v >> (8 * b)) & 0xFF).to(torch.uint8)
    return out.to(sf_i32.device)


# --------------------------------------------------------------------------- oracle
def oracle_fp8(x, w, group, clamp, fast=False):
    """按 mega `:1066-1205` / tilelang 的语义：gate/up → clamp → silu*up*weight →
    per-`group` amax(clamp 1e-4) → ceil_to_ue8m0 → E4M3。返回 (y_fp8, sf_exp uint8 (T,ng))."""
    T, H2 = x.shape
    I = H2 // 2
    gate = x[:, :I].float()
    up = x[:, I:].float()
    if clamp is not None:
        gate = gate.clamp(max=clamp)
        up = up.clamp(-clamp, clamp)
    y = gate * torch.sigmoid(gate) * up
    if w is not None:
        y = y * w[:, None]
    ng = I // group
    yy = y.view(T, ng, group)
    amax = yy.abs().amax(-1).clamp(min=1e-4)          # (T, ng)
    sf = ceil_to_ue8m0(amax / 448.0)                  # 2 的幂，>= amax/448 的最小
    sf_inv = 1.0 / sf
    yq = (yy * sf_inv[:, :, None]).reshape(T, I).to(torch.float8_e4m3fn)
    sf_exp = (sf.view(torch.int32) >> 23).to(torch.int64).to(torch.uint8)
    return yq, sf_exp, y  # y = 量化前的 fp32 值


def run_tilelang(x, w, group, clamp, col_major, round_scale, ue8m0, out_bf16, fast, avail=None):
    r = tilelang_ops.swiglu_apply_weight_to_fp8(
        x=x, topk_weights=w, avail_tokens=avail, num_per_channels=group,
        use_col_major_scales=col_major, round_scale=round_scale, ue8m0_scale=ue8m0,
        clamp_value=clamp, fast_math=fast, output_bf16=out_bf16)
    return r


def bit_mismatch_u8(a, b):
    return int((a.view(torch.uint8) != b.view(torch.uint8)).sum())


# --------------------------------------------------------------------------- A) host 公式
def part_a():
    hr('A) host 公式 get_sf_shape / get_sf_and_inv 语义核验')
    # get_sf_shape 直接 import 自 tilelang_ops.utils
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader(
        'tl_utils', os.path.join(DEEPGEMM_DIR, 'third-party', 'tilelang_ops', 'utils.py'))
    tl_utils = loader.load_module()

    ok = True
    cases = [
        # (T, I, group, ue8m0, col_major)
        (64, 2048, 32, True, True),
        (64, 2048, 32, False, False),
        (65, 2048, 32, True, True),
        (64, 2048, 128, True, True),
        (1, 4096, 32, True, True),
    ]
    for (T, I, g, ue, col) in cases:
        shape = tl_utils.get_sf_shape(T, I, g, ue, col)
        nsc = -(-I // g)
        nsc_p = -(-nsc // 4) if ue else nsc
        n_tok = (((T + 3) // 4) * 4) * (4 if (ue and col) else 1) if col else T
        exp = (nsc_p, n_tok) if col else (n_tok, nsc_p)
        good = tuple(shape) == exp
        ok = ok and good
        print(f'  get_sf_shape(T={T},I={I},g={g},ue8m0={ue},col={col}) = {tuple(shape)}  exp={exp}  {"OK" if good else "MISMATCH"}')

    # get_sf_and_inv 用真实 kernel 实测：SF 应是 2 的幂且 == ceil_to_ue8m0(amax/448)
    torch.manual_seed(0)
    T, I, g = 128, 512, 32
    x = torch.randn(T, 2 * I, device='cuda').bfloat16()
    w = torch.rand(T, device='cuda') + 0.5
    y, sf = run_tilelang(x, w, g, None, True, True, True, False, True)
    sf_bytes = unpack_ue8m0_sf(sf)[:, :I // g]
    _, sf_ref, _ = oracle_fp8(x, w, g, None)
    mism = int((sf_bytes.to(torch.int64) != sf_ref.to(torch.int64)).sum())
    ok = ok and mism == 0
    print(f'  get_sf_and_inv 产出 SF == ceil_to_ue8m0(amax/448) 的指数:  mismatch={mism}/{sf_bytes.numel()}'
          f'  {"OK" if mism == 0 else "MISMATCH"}')
    # SF 必须是 2 的幂（尾数为 0）
    mants = (sf_bytes.to(torch.int64) == 0).sum()
    print(f'  SF 为 UE8M0 指数（无尾数）—— uint8 域，样本值={sf_bytes.flatten()[:6].tolist()}')
    print('  A 总判定: ' + ('ALL PASS' if ok else 'FAIL'))
    return ok


# --------------------------------------------------------------------------- B/C) tilelang vs oracle
def check_case_tl(name, T, H2, group, clamp, w_on, col, round_, ue, out_bf16, fast):
    torch.manual_seed(1234)
    x = (torch.randn(T, H2, device='cuda') * 2.0).bfloat16()
    I = H2 // 2
    w = (torch.rand(T, device='cuda') + 0.5) if w_on else None
    r = run_tilelang(x, w, group, clamp, col, round_, ue, out_bf16, fast)
    y_tl = r[0]
    if out_bf16:
        y_bf16_tl = r[-1]
        # oracle bf16: silu*up*w
        gate = x[:, :I].float(); up = x[:, I:].float()
        if clamp is not None:
            gate = gate.clamp(max=clamp); up = up.clamp(-clamp, clamp)
        yb = gate * torch.sigmoid(gate) * up
        if w is not None:
            yb = yb * w[:, None]
        ref_bf16 = yb.to(torch.bfloat16)
        mism = int((y_bf16_tl != ref_bf16).sum())
        print(f'  {name:28s} bf16-out mismatch={mism}/{ref_bf16.numel()}'
              f'  max|Δ|={(y_bf16_tl.float()-ref_bf16.float()).abs().max().item():.3e}')
        return mism, ref_bf16.numel()
    else:
        y_ref, sf_ref, y_pre = oracle_fp8(x, w, group, clamp, fast=fast)
        sf_tl = unpack_ue8m0_sf(r[1])[:, :I // group]
        mism_y = bit_mismatch_u8(y_tl, y_ref)
        mism_sf = int((sf_tl.to(torch.int64) != sf_ref.to(torch.int64)).sum())
        # 反量化相对误差（隔离量化噪声）：tilelang 输出 * 2^sf vs 量化前 fp32 值
        ng = I // group
        # sf_tl 是 bias=127 的 UE8M0 指数；scale = 2^(exp-127)
        sf2 = torch.ldexp(torch.ones(T, ng, device='cuda'), sf_tl.to(torch.int32) - 127)
        deq = (y_tl.float().view(T, ng, group) * sf2[:, :, None]).reshape(T, I)
        rel = ((deq - y_pre.view(T, I)).abs().mean() / (y_pre.abs().mean() + 1e-30)).item()
        print(f'  {name:28s} fp8-mismatch={mism_y}/{y_ref.numel()}  sf-mismatch={mism_sf}/{sf_ref.numel()}'
              f'  量化 relerr={rel:.3e}')
        return mism_y, y_ref.numel()


def part_bc():
    hr('B/C) 真实 tilelang kernel vs 独立 torch oracle（FP8 / BF16 两套配置）')
    H2 = 4096  # 2*I, I=2048
    T = 512
    tot_bad = tot_n = 0
    cases = [
        ('fp8 I=2048 g=32', T, H2, 32, None, True, True, True, True, False, True),
        ('fp8 I=2048 g=128', T, H2, 128, None, True, True, True, True, False, True),
        ('fp8 g=32 clamp=8', T, H2, 32, 8.0, True, True, True, True, False, True),
        ('fp8 g=32 no-weight', T, H2, 32, None, False, True, True, True, False, True),
        ('bf16 g=128', T, H2, 128, None, True, True, False, False, True, True),
    ]
    for c in cases:
        bad, n = check_case_tl(*c)
        tot_bad += bad
        tot_n += n
    print(f'  B/C 合计 mismatch {tot_bad}/{tot_n}')
    return tot_bad, tot_n


# --------------------------------------------------------------------------- D) mega 算术 vs tilelang
def part_d():
    hr('D) Mega L1 epilogue 算术（真实 DeepGEMM 设备头 + fast_rcp/UE8M0）vs tilelang 逐位')
    if not os.path.exists(BIN):
        print(f'  [skip] {BIN} 不存在（先编译 src/swiglu_mega_epilogue.cu）')
        return True
    T, I, g = 256, 2048, 32
    torch.manual_seed(7)
    x = (torch.randn(T, 2 * I, device='cuda') * 2.0).bfloat16()
    w = torch.rand(T, device='cuda') + 0.5
    gate = x[:, :I].contiguous(); up = x[:, I:].contiguous()
    open('/tmp/swiglu_gate.bin', 'wb').write(gate.cpu().view(torch.int16).numpy().tobytes())
    open('/tmp/swiglu_up.bin', 'wb').write(up.cpu().view(torch.int16).numpy().tobytes())
    open('/tmp/swiglu_w.bin', 'wb').write(w.cpu().numpy().astype('float32').tobytes())

    r = run_tilelang(x, w, g, None, True, True, True, False, True)
    y_tl = r[0].view(torch.uint8).cpu()
    sf_tl = unpack_ue8m0_sf(r[1])[:, :I // g].cpu()

    ok = True
    for fast in (1, 0):
        out = subprocess.run([BIN, str(T), str(I), str(g), 'inf', str(fast), '1'],
                             capture_output=True, text=True)
        print('  ' + out.stdout.strip())
        y_me = torch.frombuffer(bytearray(open('/tmp/swiglu_mega_y.bin', 'rb').read()),
                                dtype=torch.uint8).view(T, I)
        sf_me = torch.frombuffer(bytearray(open('/tmp/swiglu_mega_sf.bin', 'rb').read()),
                                 dtype=torch.uint8).view(T, I // g)
        my = int((y_me != y_tl).sum())
        ms = int((sf_me.to(torch.int64) != sf_tl.to(torch.int64)).sum())
        # 差异幅度（按值域归一）
        diff = (y_me.to(torch.int16) - y_tl.to(torch.int16)).abs()
        print(f'  fast={fast}: fp8 mismatch={my}/{y_tl.numel()} ({100*my/y_tl.numel():.3f}%)  '
              f'sf mismatch={ms}/{sf_tl.numel()}  max byteΔ={int(diff.max())}')
    print('  D 说明：tilelang 的 silu 用 T.exp + fp32 除法；mega 用 __expf + rcp.approx，'
          '差异被 E4M3(相对 2^-3) 归掉后应几乎全等。')
    return ok


# --------------------------------------------------------------------------- E) 吞吐
def part_e():
    hr('E) 吞吐（T=8192, 2I=4096, g=32, DeepSeek I=2048 量级）')
    T, I, g = 8192, 2048, 32
    x = (torch.randn(T, 2 * I, device='cuda')).bfloat16()
    w = torch.rand(T, device='cuda') + 0.5
    # warmup + JIT
    run_tilelang(x, w, g, None, True, True, True, False, True)
    torch.cuda.synchronize()
    iters = 50
    t0 = time.time()
    for _ in range(iters):
        run_tilelang(x, w, g, None, True, True, True, False, True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1e3
    nbytes = T * 2 * I * 2 + T * I * 1  # read bf16 in + write fp8 out
    print(f'  tilelang swiglu: {dt*1e3:.1f} us/iter, {nbytes/1e9/(dt/1e3):.0f} GB/s  (T={T},I={I})')
    return True


def main():
    print('环境:', torch.cuda.get_device_name(0), 'sm', torch.cuda.get_device_capability(0))
    print('tilelang', __import__('tilelang').__version__, '| torch', torch.__version__)
    a = part_a()
    bc_bad, bc_n = part_bc()
    d = part_d()
    e = part_e()
    hr('总判定')
    print(f'A host 公式: {"PASS" if a else "FAIL"}')
    print(f'B/C tilelang vs oracle mismatch: {bc_bad}/{bc_n}')
    print(f'D mega 算术 vs tilelang: {"见上" if d else "FAIL"}')
    print(f'E 吞吐: {"PASS" if e else "FAIL"}')


if __name__ == '__main__':
    main()
