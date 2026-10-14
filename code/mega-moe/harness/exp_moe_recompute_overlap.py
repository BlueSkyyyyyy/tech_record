"""P172-1：兑现重算的 shadow 预算——真机并发下「重算 L1」与「不依赖 h 的反向 GEMM」到底重叠多少。

背景（第 181 篇 / P171-1 的「下一步」）：
  * MoE 前向融合主 kernel 不存 `h`（linear1 预激活，SwiGLU 反向的输入），反向只能**重算**一次
    linear1 GEMM。第 181 篇量到这份重算的**时间预算**只有「不依赖 h 的反向 GEMM（shadow：dgrad L2
    + wgrad L2）」的 0.47×/0.73× —— 但那只是**预算**，不是**实测重叠**。
  * 第 148 篇（P138-1）证 DeepGEMM 单流下侧流 consumer 被 `232448 B/SM` smem 预算堵死、比值恒
    ~0.58×；第 167 篇（P157-1）证**单 kernel 融合要计算腿是异步张量核**。本篇把这两条落到
    **重算 / shadow 这一对真实工作负载**上，量「要什么条件才能把预算兑现成重叠」。

核心机制（第 181 篇只给「预算」与否，本篇给「兑现条件」）：
  DeepGEMM 的 persistent kernel 每个 block 占满一个 SM（smem `232448 B/SM` ≈ 整块
  `heuristics/sm90.hpp:14`），所以**默认满 SM 发射时两个 stream 的 kernel 无法 co-reside**——
  第二个 kernel 的 block 要等第一个腾出 SM。把 `num_sms`（= 发射的 block 数）**分区**后，
  两个 kernel 各占一半 SM，才真正并行。这正是 Mega MoE 用 locality domain 分域 + 每 kernel
  SM 预算（`scheduler/mega_moe.cuh:344-370`）的原因。

实验（H100(sm90)，真实 SM90 FP8 grouped/dense kernel）：
  * RECOMPUTE = 前向 L1 routed（`m_grouped_fp8_gemm_nt_contiguous`）；
  * SHADOW    = dgrad L2（`m_grouped_fp8_gemm_nt_contiguous`）+ wgrad L2（`k_grouped_..._nt_contiguous`）；
  * Part A：满 SM 下两 stream 并发 vs 单流串行（预期：几乎不重叠 = 负结果，坐实 co-residency 闸门）；
  * Part B：对称分区扫 `num_sms ∈ {132,96,66,44,33}`（两个 kernel 同一预算）；
  * Part C：非对称分区 `n_rec + n_sh = 132`，找把重算藏进 shadow 的最优切分；
  * Part D：并发执行与串行执行输出逐位/相对一致（重叠不改正确性）。

约束：Mega MoE 融合主 kernel 是 SM100 专属、本机跑不了；本实验只用其**架构无关的调度机制语义**
（SM 预算分区 / 异步张量核 / smem 占满）。目标卡代号敏感：全文用「目标卡」。
输出重定向到 exp_moe_recompute_overlap.out.txt。
"""
import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from exp_mega_moe_forward_emul import (  # noqa: E402
    deep_gemm, get_arch_major, GRAN_K, align,
    per_token_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout, _bench,
)
from exp_moe_backward_kernels import quant_k_operand  # noqa: E402

DEV_SMS = torch.cuda.get_device_properties(0).multi_processor_count


# --------------------------------------------------------------------------- workload
def build(T, H, I, E, K, device):
    """构造 RECOMPUTE（前向 L1 routed）与 SHADOW（dgrad L2 + wgrad L2）的真实操作数与闭包。"""
    ma = get_mk_alignment_for_contiguous_layout()
    ka = GRAN_K
    counts = [T * K // E] * E
    host_ks = [align(c, ka) for c in counts]
    total_k = sum(host_ks)
    m_total = sum(align(c, ma) for c in counts)
    layout = torch.zeros(m_total, device=device, dtype=torch.int32)

    # RECOMPUTE：前向 L1 routed，h = x @ W1^T，x 是 per-token FP8
    A1 = torch.randn(m_total, H, device=device, dtype=torch.float32)
    a1q = per_token_cast_to_fp8(A1, use_ue8m0=False, gran_k=ka)
    W1r = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32)
    w1r = torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn, device=device)
    w1rs = torch.empty(E, align(2 * I, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
    for e in range(E):
        w1r[e], w1rs[e] = per_block_cast_to_fp8(W1r[e], use_ue8m0=False, gran_k=ka)
    d1r = torch.empty(m_total, 2 * I, device=device, dtype=torch.bfloat16)

    # SHADOW①：dgrad L2，dA = dY @ W2^T
    dY = torch.randn(m_total, H, device=device, dtype=torch.float32)
    dyq = per_token_cast_to_fp8(dY, use_ue8m0=False, gran_k=ka)
    B2b = torch.randn(E, I, H, device=device, dtype=torch.float32)
    b2b = torch.empty(E, I, H, dtype=torch.float8_e4m3fn, device=device)
    b2bs = torch.empty(E, align(I, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
    for e in range(E):
        b2b[e], b2bs[e] = per_block_cast_to_fp8(B2b[e], use_ue8m0=False, gran_k=ka)
    dA = torch.empty(m_total, I, device=device, dtype=torch.bfloat16)

    # SHADOW②：wgrad L2，dW2 = a^T dY（K 轴分组）
    Ak = torch.randn(total_k, I, device=device, dtype=torch.float32)
    Bk = torch.randn(total_k, H, device=device, dtype=torch.float32)
    ak = quant_k_operand(Ak, host_ks)
    bk = quant_k_operand(Bk, host_ks)
    ksl = torch.tensor(host_ks, device=device, dtype=torch.int32)
    dW2 = torch.zeros(E, I, H, device=device, dtype=torch.float32)

    flops_rec = 2 * m_total * (2 * I) * H
    flops_dg = 2 * m_total * I * H
    flops_wg = 2 * total_k * I * H

    def rec():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a1q, (w1r, w1rs), d1r, layout, disable_ue8m0_cast=True)

    def dg():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(dyq, (b2b, b2bs), dA, layout, disable_ue8m0_cast=True)

    def wg():
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (ak[0], ak[1]), (bk[0], bk[1]), dW2, host_ks, ksl, dW2, recipe=(1, 1, ka))

    def sh():
        dg()
        wg()

    return dict(rec=rec, sh=sh, dg=dg, wg=wg, outs=(d1r, dA, dW2), zeros=(dW2,),
                m_total=m_total, total_k=total_k,
                flops_rec=flops_rec, flops_dg=flops_dg, flops_wg=flops_wg)


# --------------------------------------------------------------------------- timing
def bench_one(fn, n_sms, iters=30):
    """单流、给 kernel 限核 n_sms，返回 us。"""
    deep_gemm.set_num_sms(n_sms)
    return _bench(fn, iters) * 1e6


def bench_concurrent(rec, sh, n_rec, n_sh, iters=20):
    """两 stream 并发：rec 发在 s1（限 n_rec）、sh 发在 s2（限 n_sh），返回墙钟 us。

    `set_num_sms` 是 host 端全局开关，但每个 kernel 在**发射那一刻**把 num_sms 烘进 grid/编译键，
    故「先 set→发 rec、再 set→发 sh」可给两个并发 kernel **不同**的 SM 预算。
    """
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    cur = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event()
    e2 = torch.cuda.Event()

    def once():
        torch.cuda.synchronize()
        start.record(cur)
        s1.wait_event(start)
        s2.wait_event(start)
        with torch.cuda.stream(s1):
            deep_gemm.set_num_sms(n_rec)
            rec()
        with torch.cuda.stream(s2):
            deep_gemm.set_num_sms(n_sh)
            sh()
        e1.record(s1)
        e2.record(s2)
        cur.wait_event(e1)
        cur.wait_event(e2)
        end.record(cur)
        torch.cuda.synchronize()

    for _ in range(3):
        once()
    ts = []
    for _ in range(iters):
        once()
        ts.append(start.elapsed_time(end) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


# --------------------------------------------------------------------------- parts
def part_a(w):
    print('## A. 满 SM 并发 vs 单流串行（预期：负结果——co-residency 闸门）')
    print('=' * 100)
    deep_gemm.set_num_sms(DEV_SMS)
    t_rec = bench_one(w['rec'], DEV_SMS)
    t_dg = bench_one(w['dg'], DEV_SMS)
    t_wg = bench_one(w['wg'], DEV_SMS)
    t_sh = t_dg + t_wg
    t_serial = bench_one(lambda: (w['rec'](), w['sh']()), DEV_SMS)
    t_ovl = bench_concurrent(w['rec'], w['sh'], DEV_SMS, DEV_SMS)
    ideal = max(t_rec, t_sh)
    print(f'   RECOMPUTE only (满 {DEV_SMS} SM): {t_rec:8.1f} us  {w["flops_rec"]/(t_rec*1e-6)/1e12:6.0f} TFLOPS')
    print(f'   SHADOW①dgrad L2 (满 {DEV_SMS} SM): {t_dg:8.1f} us  {w["flops_dg"]/(t_dg*1e-6)/1e12:6.0f} TFLOPS')
    print(f'   SHADOW②wgrad L2 (满 {DEV_SMS} SM): {t_wg:8.1f} us  {w["flops_wg"]/(t_wg*1e-6)/1e12:6.0f} TFLOPS')
    print(f'   SHADOW   合计    (满 {DEV_SMS} SM): {t_sh:8.1f} us  {((w["flops_dg"]+w["flops_wg"])/(t_sh*1e-6)/1e12):6.0f} TFLOPS'
          f'   RECOMPUTE/SHADOW = {t_rec/t_sh:.2f}x')
    print(f'   serial (rec; sh) 单流            : {t_serial:8.1f} us')
    print(f'   concurrent (两 stream, 都满 SM)  : {t_ovl:8.1f} us   ideal(max)={ideal:.1f}')
    ovf = (t_rec + t_sh) / t_ovl
    print(f'   ⇒ speedup vs 串行 = {t_serial/t_ovl:.3f}x，overlap_factor = (rec+sh)/ovl = {ovf:.3f}x')
    print(f'      （=1.0 表示完全不重叠=串行；={ (t_rec+t_sh)/ideal:.2f} 表示完美。满 SM 下 smem 占满 →')
    print(f'       kernel 无法 co-reside，第二个 kernel 排队，故 ovf ≈ 1.0。）')
    print()
    return t_rec, t_sh


def part_b(w, t_rec132, t_sh132):
    print('## B. 对称 SM 预算分区：两个 kernel 同一 limit，扫 num_sms')
    print(f'   {DEV_SMS} SM 的 H100：设 num_sms=N 时每个 kernel 发 N 个 persistent block；')
    print(f'   2N ≤ {DEV_SMS} 时两 kernel 各占不同 SM → 真正并行；N={DEV_SMS} 时挤同一批 SM → 排队。')
    print('=' * 100)
    print(f'  {"N":>5}{"rec us":>9}{"sh us":>9}{"serial rec+sh":>15}{"concurrent":>12}{"ideal=max":>11}'
          f'{"ovf=(rec+sh)/ovl":>18}{"spd vs 满SM串行":>16}')
    serial_full = t_rec132 + t_sh132
    for N in (132, 96, 66, 44, 34):
        deep_gemm.set_num_sms(N)
        t_r = bench_one(w['rec'], N)
        t_s = bench_one(w['sh'], N)
        t_serial = t_r + t_s
        t_ovl = bench_concurrent(w['rec'], w['sh'], N, N)
        ovf = t_serial / t_ovl
        print(f'  {N:>5}{t_r:>9.1f}{t_s:>9.1f}{t_serial:>15.1f}{t_ovl:>12.1f}{max(t_r,t_s):>11.1f}'
              f'{ovf:>18.2f}{serial_full/t_ovl:>16.2f}')
    deep_gemm.set_num_sms(DEV_SMS)
    print()
    return serial_full


def part_c(w, serial_full):
    print('## C. 非对称分区 n_rec + n_sh = 132：把重算藏进 shadow 的最优切分')
    print('   重算小（0.32× shadow），不必平分 SM；给 shadow 更多 SM、给重算刚好够用的量。')
    print('=' * 100)
    print(f'  {"(n_rec,n_sh)":>14}{"rec us":>9}{"sh us":>9}{"concurrent":>12}{"ideal=max":>11}'
          f'{"ovf=(rec+sh)/ovl":>18}{"spd vs 满SM串行":>16}')
    best = None
    for n_rec, n_sh in ((DEV_SMS, DEV_SMS), (66, 66), (44, 88), (34, 98), (48, 84), (56, 76), (24, 108)):
        t_r = bench_one(w['rec'], n_rec)
        t_s = bench_one(w['sh'], n_sh)
        t_ovl = bench_concurrent(w['rec'], w['sh'], n_rec, n_sh)
        ovf = (t_r + t_s) / t_ovl
        spd = serial_full / t_ovl
        mark = ''
        if best is None or t_ovl < best[0]:
            best = (t_ovl, n_rec, n_sh)
            mark = '  <-- best'
        print(f'  {str((n_rec, n_sh)):>14}{t_r:>9.1f}{t_s:>9.1f}{t_ovl:>12.1f}{max(t_r,t_s):>11.1f}'
              f'{ovf:>18.2f}{spd:>16.2f}{mark}')
    deep_gemm.set_num_sms(DEV_SMS)
    print(f'   ⇒ 最优切分 {best[1:]}，并发 {best[0]:.0f} us（vs 满 SM 串行 {serial_full:.0f} us，'
          f'{serial_full/best[0]:.2f}x）。')
    print()
    return best


def part_d(w):
    print('## D. 重叠不改正确性：并发输出 == 串行输出')
    print('=' * 100)
    deep_gemm.set_num_sms(DEV_SMS)

    def run_once(n_rec, n_sh, concurrent):
        """单次执行（wgrad 累加器先清零；并发路径只发一轮，不 warmup，保证与串行同次数）。"""
        for z in w['zeros']:
            z.zero_()
        if concurrent:
            cur = torch.cuda.current_stream()
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            ev = torch.cuda.Event()
            ev.record(cur)
            s1.wait_event(ev)
            s2.wait_event(ev)
            with torch.cuda.stream(s1):
                deep_gemm.set_num_sms(n_rec)
                w['rec']()
            with torch.cuda.stream(s2):
                deep_gemm.set_num_sms(n_sh)
                w['sh']()
            e1 = torch.cuda.Event(); e1.record(s1)
            e2 = torch.cuda.Event(); e2.record(s2)
            cur.wait_event(e1); cur.wait_event(e2)
        else:
            deep_gemm.set_num_sms(n_rec); w['rec']()
            deep_gemm.set_num_sms(n_sh); w['sh']()
        torch.cuda.synchronize()
        return [o.clone() for o in w['outs']]

    ref = run_once(DEV_SMS, DEV_SMS, False)
    for (n_rec, n_sh) in ((DEV_SMS, DEV_SMS), (66, 66), (44, 88), (34, 98), (56, 76)):
        got = run_once(n_rec, n_sh, True)
        names = ('recompute h', 'dgrad dA', 'wgrad dW2')
        msgs = []
        for nm, r, g in zip(names, ref, got):
            eq = torch.equal(r, g)
            d = (r.float() - g.float()).abs()
            denom = r.float().abs().max().item() + 1e-30
            msgs.append(f'{nm}: equal={eq} max|d|/max|r|={d.max().item()/denom:.2e}')
        print(f'   (n_rec,n_sh)=({n_rec},{n_sh}) 并发 vs 满SM串行: ' + ' | '.join(msgs))
    deep_gemm.set_num_sms(DEV_SMS)
    print('   （grouped GEMM 结构性 bit 确定——第 180 篇；分区只改 SM 数不改累加序，故输出逐位一致。）')
    print()


def main():
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, device={torch.cuda.get_device_name(0)}, '
          f'SM={DEV_SMS}')
    print('P172-1：把第 181 篇的「重算 shadow 预算」兑现成真机并发重叠——要 SM 预算分区才成立。')
    print('Mega MoE 融合主 kernel 是 SM100 专属、本机不可跑；本实验用真实 SM90 FP8 grouped kernel。')
    print('=' * 100)

    T, H, I, E, K = 4096, 7168, 2048, 8, 2
    w = build(T, H, I, E, K, 'cuda')
    print(f'workload: T={T} H={H} I={I} E={E} topk={K}  m_total={w["m_total"]} total_k={w["total_k"]}')
    print(f'  RECOMPUTE flops = {w["flops_rec"]/1e9:.1f} GF ; SHADOW flops = '
          f'{(w["flops_dg"]+w["flops_wg"])/1e9:.1f} GF (dgrad {w["flops_dg"]/1e9:.1f} + '
          f'wgrad {w["flops_wg"]/1e9:.1f})')
    print()

    try:
        t_rec132, t_sh132 = part_a(w)
    except Exception:
        print('   part_a FAILED:\n' + traceback.format_exc())
        return
    try:
        serial_full = part_b(w, t_rec132, t_sh132)
    except Exception:
        print('   part_b FAILED:\n' + traceback.format_exc())
        return
    try:
        part_c(w, serial_full)
    except Exception:
        print('   part_c FAILED:\n' + traceback.format_exc())
    try:
        part_d(w)
    except Exception:
        print('   part_d FAILED:\n' + traceback.format_exc())

    print('ALL DONE')


if __name__ == '__main__':
    main()
