"""P136-1 实验：MoE wgrad 的 bf16 输出协议——2.2× 上界的落地与精度代价。

为什么做这篇（补的缺口）：
  第 145 篇（P135-1）把 wgrad 的「强制累积」代价量化清楚，结论是：真凶是**每专家一份 fp32 [M,N] 权重梯度**
  （读+写占 96% 流量），而 SM90 的 wgrad kernel `k_grouped_fp8_gemm_nt_contiguous` 被 API 钉死为
  **1D1D、D 必 fp32、必须累积**（`csrc/apis/gemm.hpp:373-427`，`:384` recipe、`:403` `c.has_value()`；
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`；kernel `SM90_TMA_REDUCE_ADD_2D`）。于是第 145 篇
  给了一个**等价但更快**的写法：`E` 个独立稠密 1D2D、**bf16 直写、无回读**（`sm90_fp8_gemm_1d2d.hpp:81`
  `not c.has_value() and d==bf16`），实测 **537 TFLOPS（2.23×）**——那是「wgrad 换 bf16 输出」的**性能上界**。

  但第 145 篇**没有回答代价面**：把参数梯度从 fp32 换成 bf16，**精度掉多少**？梯度累积 S 步后会不会爆？
  在训练里 wgrad 的输出要跨 micro-batch **累积**（gradient accumulation），bf16 每一步都要舍入一次，
  误差是否随 S 增长？本篇把这个「2.2× 上界的精度代价」用真实 kernel 落地并量化。

要建立的结论（待实测）：
  ① bf16 输出只是**末次舍入**，其误差 ~2^-9 ≈ 4e-3，比 wgrad **本来就有的 FP8 输入量化噪声**（~3.7e-2）
     还小一个量级 → 在精度上基本「免费」；
  ② 真正的风险在**累积**：把每步的结果先舍入到 bf16 再加，误差随 S 增长；而 fp32 单累加器（真实路径）
     几乎不涨；
  ③ 所以正确协议是「每步 fresh bf16 + **fp32** 归约」，其误差由 S 控制、可与 FP8 噪声同量级。

可跑性：全部 SM90 FP8 kernel（H100 可跑）。SM100 的 wgrad `k_grouped_tn_contiguous` 同为 fp32 累积
（`gemm.hpp:320-371`，仅分析）。本机 8 卡共享，本文在**空闲 GPU1** 单进程测，报数注明。

输出重定向：harness/exp_wgrad_bf16_output.out.txt
"""
import os
import sys
import traceback

import torch

import deep_gemm

DG_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, os.path.join(DG_DIR, 'tests'))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
)

GRAN = 128
DEV = 'cuda'


# --------------------------------------------------------------------------- helpers
def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def maxrel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs() / (b.abs() + 1e-6)).max().item()


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def gns(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _bench(fn, warm=10, iters=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 1e3 / iters


# --------------------------------------------------------------------------- dequant (fp64 exact)
def deq_kg(data, sf, host_ks, mn, gran=GRAN):
    """k_grouped 操作数的 fp32 反量化，返回 [mn, total_k]。

    核对过的布局：host 侧 `quant_kg` 把**每组**写成 `xf.T.reshape(-1)`（组内 m-major、k 最快），
    因此整块缓冲是 **group-major** 的 `[E, mn, k_g]` 行主序，而不是 `[mn, total_k]`。所以必须**逐组**
    切片再拼接到列上（错当成 `[mn, total_k]` 会让 fp64 参考 O(1) 崩——这是本轮实测踩到的坑）。
    """
    outs = []
    off = 0
    for hk in host_ks:
        x = data[off * mn:(off + hk) * mn].float().view(mn, hk)
        s = sf[:, off // gran:(off + hk) // gran].view(mn, hk // gran, 1)
        outs.append((x.view(mn, hk // gran, gran) * s).reshape(mn, hk))
        off += hk
    return torch.cat(outs, dim=1)


def deq_token(xf, sf, gran=GRAN):
    m, k = xf.shape
    return (xf.float().view(m, k // gran, gran) * sf.view(m, k // gran, 1)).reshape(m, k)


def deq_block(xf, sf, gran=GRAN):
    n, k = xf.shape
    x = xf.float().view(n // gran, gran, k // gran, gran)
    return (x * sf.view(n // gran, 1, k // gran, 1)).reshape(n, k)


# --------------------------------------------------------------------------- operands
def quant_kg(X, host_ks):
    """k_grouped 操作数（MN-major 逐 channel SF）。X: [total_k, mn] fp32。"""
    tk, mn = X.shape
    data = torch.zeros(tk * mn, dtype=torch.float8_e4m3fn, device=X.device)
    sfs = []
    off = 0
    for hk in host_ks:
        xg = X[off:off + hk].contiguous()
        xf, sf = per_channel_cast_to_fp8(xg, use_ue8m0=False, gran_k=GRAN)
        data[off * mn:(off + hk) * mn] = xf.T.reshape(-1)
        sfs.append(sf)
        off += hk
    sf_cat = torch.cat(sfs).T if sfs else X.new_zeros((mn, 0))
    return data, sf_cat


def kg_operands(Ag, Bg, host_ks):
    mn_a, mn_b = Ag.shape[1], Bg.shape[1]
    ad, asf = quant_kg(Ag, host_ks)
    bd, bsf = quant_kg(Bg, host_ks)
    return (ad, asf), (bd, bsf)


def dense_operands(Ag, Bg, host_ks):
    """返回每个专家的 1D2D 操作数 [(a[M,k_e],b[N,k_e])]。"""
    ops = []
    off = 0
    for k in host_ks:
        A = Ag[off:off + k].T.contiguous()          # [M, k]
        B = Bg[off:off + k].T.contiguous()          # [N, k]
        a = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN)
        b = per_block_cast_to_fp8(B, use_ue8m0=False, gran_k=GRAN)
        ops.append((a, b))
        off += k
    return ops


# --------------------------------------------------------------------------- parts
def part_a_precision():
    print('=' * 96)
    print('Part A · 单步精度分解：bf16 输出 vs 早已存在的 FP8 输入量化噪声')
    print('-' * 96)
    M, N, E, k_e = 2048, 7168, 8, 512
    host_ks = [align(k_e, GRAN)] * E
    sum_k = sum(host_ks)
    torch.manual_seed(0)
    Ag = torch.randn(sum_k, M, device=DEV)
    Bg = torch.randn(sum_k, N, device=DEV)

    (ad, asf), (bd, bsf) = kg_operands(Ag, Bg, host_ks)
    ops = dense_operands(Ag, Bg, host_ks)

    # 逐专家 fp64 参考（分别用两条路径各自的量化操作数）
    exact_kg_e = torch.empty(E, M, N, device=DEV, dtype=torch.float64)
    exact_raw_e = torch.empty(E, M, N, device=DEV, dtype=torch.float64)
    exact_dense_e = torch.empty(E, M, N, device=DEV, dtype=torch.float64)
    Aq = deq_kg(ad, asf, host_ks, M)   # [M, sum_k]
    Bq = deq_kg(bd, bsf, host_ks, N)   # [N, sum_k]
    off = 0
    for e in range(E):
        k = host_ks[e]
        exact_kg_e[e] = Aq[:, off:off + k].double() @ Bq[:, off:off + k].double().T
        exact_raw_e[e] = Ag[off:off + k].double().T @ Bg[off:off + k].double()
        aq = deq_token(ops[e][0][0], ops[e][0][1]).double()
        bq = deq_block(ops[e][1][0], ops[e][1][1]).double()
        exact_dense_e[e] = aq @ bq.T
        off += k

    # REF：真实 wgrad，k_grouped fp32 强制累积
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)
    d_ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (ad, asf), (bd, bsf), d_ref, host_ks, layout, d_ref, recipe=(1, 1, GRAN))
    torch.cuda.synchronize()

    # ALT：等价写法，E 个稠密 1D2D bf16 直写、无回读
    d_alt = torch.empty(E, M, N, device=DEV, dtype=torch.bfloat16)
    for e, (a, b) in enumerate(ops):
        deep_gemm.fp8_gemm_nt(a, b, d_alt[e])
    torch.cuda.synchronize()

    r_quant = rel(exact_kg_e, exact_raw_e)                 # FP8 输入量化噪声（wgrad 本来就有的）
    r_fp32 = rel(d_ref.double(), exact_kg_e)               # fp32 单累加器的误差
    r_bf16_round = rel(d_ref.bfloat16().double(), d_ref.double())   # 仅末次 bf16 舍入
    r_bf16_out = rel(d_alt.double(), exact_dense_e)        # bf16 输出的总误差（含累积）
    r_ref_alt = rel(d_alt.double(), d_ref.double())        # 两条路径总差（含两侧量化差异）
    c_ref_alt = cos(d_alt, d_ref)
    g = gns(d_alt, d_ref)

    print(f'  形状 dW2: M={M} N={N} E={E} 每专家 k_e={k_e}（sum_k={sum_k}）')
    print(f'  {"量":<42}{"relerr":>12}{"note":>34}')
    print(f'  {"FP8 输入量化噪声 (vs 未量化 fp64)":<42}{r_quant:>12.2e}{"wgrad 本来就有的地板":>34}')
    print(f'  {"REF k_grouped fp32 累积 vs fp64":<42}{r_fp32:>12.2e}{"单累加器几乎无损":>34}')
    print(f'  {"仅末次 bf16 舍入 (fp32->bf16)":<42}{r_bf16_round:>12.2e}{"2^-9 量级":>34}')
    print(f'  {"ALT 稠密 1D2D bf16  vs fp64":<42}{r_bf16_out:>12.2e}{"bf16 输出总误差":>34}')
    print(f'  {"REF vs ALT（含两侧量化差异）":<42}{r_ref_alt:>12.2e}{"≈ FP8 噪声":>34}')
    print(f'  -> 方向 cos(ALT, REF) = {c_ref_alt:.7f}   GNS=||ALT-REF||/||REF|| = {g:.3e}')
    print(f'  -> 结论：bf16 输出误差 {r_bf16_round:.1e} 比 FP8 输入噪声 {r_quant:.1e} 小 '
          f'{r_quant / r_bf16_round:.1f}× → 精度上基本「免费」')
    return r_quant, r_fp32, r_bf16_round, r_bf16_out


def part_b_accumulation(S_list=(1, 2, 4, 8)):
    print('=' * 96)
    print('Part B · 跨 micro-batch 累积 S 步：fp32 单累加器 vs「每步 fresh bf16 + fp32 归约」')
    print('-' * 96)
    M, N, E, k_e = 2048, 7168, 8, 512
    host_ks = [align(k_e, GRAN)] * E
    sum_k = sum(host_ks)
    Smax = max(S_list)
    torch.manual_seed(1)
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)

    steps = []
    for s in range(Smax):
        Ag = torch.randn(sum_k, M, device=DEV)
        Bg = torch.randn(sum_k, N, device=DEV)
        steps.append((kg_operands(Ag, Bg, host_ks), dense_operands(Ag, Bg, host_ks)))

    D_ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
    D_alt = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
    ex_ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float64)
    ex_alt = torch.zeros(E, M, N, device=DEV, dtype=torch.float64)
    d_s = torch.empty(E, M, N, device=DEV, dtype=torch.bfloat16)

    print(f'  {"S":>4}{"REF fp32 relerr":>18}{"ALT bf16 relerr":>18}{"cos(ALT,REF)":>16}'
          f'{"GNS":>12}{"REF/ALT 比值":>16}')
    out = {}
    for s in range(Smax):
        kg, dn = steps[s]
        (ad, asf), (bd, bsf) = kg
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (ad, asf), (bd, bsf), D_ref, host_ks, layout, D_ref, recipe=(1, 1, GRAN))
        for e, (a, b) in enumerate(dn):
            deep_gemm.fp8_gemm_nt(a, b, d_s[e])
        D_alt.add_(d_s)
        # 累积 fp64 参考
        Aq, Bq = deq_kg(ad, asf, host_ks, M), deq_kg(bd, bsf, host_ks, N)
        off = 0
        for e in range(E):
            k = host_ks[e]
            ex_ref[e] += Aq[:, off:off + k].double() @ Bq[:, off:off + k].double().T
            aq = deq_token(dn[e][0][0], dn[e][0][1]).double()
            bq = deq_block(dn[e][1][0], dn[e][1][1]).double()
            ex_alt[e] += aq @ bq.T
            off += k
        if (s + 1) in S_list:
            rr = rel(D_ref.double(), ex_ref)
            ra = rel(D_alt.double(), ex_alt)
            print(f'  {s + 1:>4}{rr:>18.2e}{ra:>18.2e}{cos(D_alt, D_ref):>16.7f}'
                  f'{gns(D_alt, D_ref):>12.2e}{ra / rr:>16.1f}')
            out[s + 1] = (rr, ra)
    print('  -> ALT 的误差随 S 增长、REF 几乎不涨；比值 = ALT/REF 精度代价（累积协议带来的）')
    return out


def part_c_timing():
    print('=' * 96)
    print('Part C · 2.2× 上界的落地计时（S 步累积 + 归约，含真实 kernel）')
    print('-' * 96)
    M, N, E, k_e = 2048, 7168, 8, 512
    host_ks = [align(k_e, GRAN)] * E
    sum_k = sum(host_ks)
    S_LIST = (1, 2, 4, 8, 16)
    Smax = max(S_LIST)
    torch.manual_seed(2)
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)

    # 每步操作数（量 REF/ALT 的单步时间用第 0 步；归约扫 S 用 stack 的前 S 片）
    kgs, dns = [], []
    for s in range(Smax):
        Ag = torch.randn(sum_k, M, device=DEV)
        Bg = torch.randn(sum_k, N, device=DEV)
        kgs.append(kg_operands(Ag, Bg, host_ks))
        dns.append(dense_operands(Ag, Bg, host_ks))

    D_ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
    scratch = torch.empty(E, M, N, device=DEV, dtype=torch.bfloat16)
    stack = torch.empty(Smax, E, M, N, device=DEV, dtype=torch.bfloat16)
    for s in range(Smax):
        for e, (a, b) in enumerate(dns[s]):
            deep_gemm.fp8_gemm_nt(a, b, stack[s, e])

    fl = 2 * M * N * sum_k

    def ref_step():
        (ad, asf), (bd, bsf) = kgs[0]
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (ad, asf), (bd, bsf), D_ref, host_ks, layout, D_ref, recipe=(1, 1, GRAN))

    def alt_step():
        for e, (a, b) in enumerate(dns[0]):
            deep_gemm.fp8_gemm_nt(a, b, scratch[e])

    t_ref = _bench(ref_step)
    t_alt = _bench(alt_step)

    print(f'  单步（墙体，各 30 次）：REF k_grouped fp32/acc {t_ref * 1e6:.1f}us '
          f'({fl / t_ref / 1e12:.0f} TFLOPS)、ALT E×稠密 1D2D bf16 {t_alt * 1e6:.1f}us '
          f'({fl / t_alt / 1e12:.0f} TFLOPS)、纯 GEMM 比 {t_ref / t_alt:.2f}×')
    print(f'  {"S":>4}{"归约 us":>12}{"REF 总 us":>12}{"ALT 总 us":>12}{"端到端比":>12}'
          f'{"归约占 ALT%":>14}')
    for S in S_LIST:
        t_red = _bench(lambda: torch.sum(stack[:S], dim=0, dtype=torch.float32))
        total_ref = t_ref * S
        total_alt = t_alt * S + t_red
        print(f'  {S:>4}{t_red * 1e6:>12.1f}{total_ref * 1e6:>12.1f}{total_alt * 1e6:>12.1f}'
              f'{total_ref / total_alt:>12.2f}{t_red / total_alt * 100:>14.0f}')
    unit_read = E * M * N * 2
    t_red_max = _bench(lambda: torch.sum(stack, dim=0, dtype=torch.float32))
    unit_bw = (Smax * unit_read + E * M * N * 4) / t_red_max   # B/s
    print(f'  -> 归约是带宽最优的：{unit_bw / 1e12:.2f} TB/s（≈H100 HBM 峰值 3.35 的 '
          f'{unit_bw / 3.35e12 * 100:.0f}%）')
    print(f'  -> 渐进（S→∞）每步归约只需读一份 bf16 输出 {unit_read / 1e6:.0f}MB；'
          f'端到端比上界 ≈ {t_ref / (t_alt + unit_read / unit_bw):.2f}×（非 2.2×）')
    return t_ref, t_alt


def main():
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, '
          f'sms={deep_gemm.get_num_sms()}')
    for name, fn in (('Part A', part_a_precision), ('Part B', part_b_accumulation),
                     ('Part C', part_c_timing)):
        try:
            fn()
        except Exception:
            print(f'{name} FAILED:\n' + traceback.format_exc())
    print('=' * 96)
    print('done')


if __name__ == '__main__':
    main()
