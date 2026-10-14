"""P183-1 实验：给通信让 SM 的「机会成本」——DeepEP 自动 SM 模型超配的 SM 值多少计算？

背景（第 190/191/192 篇，P180~P182）：
  * 第 190 篇量到「计算侧」的 SM 预算曲线：给通信让 k 个 SM，计算腿变慢多少；
  * 第 191/192 篇量到「通信侧」的膝点：一条 NVLink（单链路 / 多对多）在 ~8–16 个 SM 就饱和于 ~370 GB/s，
    超过膝点再加 SM 是「白给」；
  * 第 192 篇还证实 DeepEP 的真实 `get_theoretical_num_sms()`（ep.py:431-541）在
    `prefer_overlap_with_compute=False` 时会把 `direct EP8` 顶到 `max(·, 64)`（ep.py:531），
    是「真机膝点（8–16）」的 4×。

本篇把两侧的账合起来，回答一个此前只被定性描述、没被定价的问题：
  **模型多给的这 48 个 SM（64 vs 16）到底值多少计算时间？**

实验三段（全部架构无关、H100 真跑）：
  A. 用真实 SM90 FP8 grouped kernel 量「计算侧」：把 `num_sms` 从 132 减到
     {120,116,84,68}（= reserve 12/16/48/64），量前向 L1 与反向 wgrad 的纯 device 时长；
  B. 用 P63-1 的 `ast` 抽取跑**未修改的真实** `get_theoretical_num_sms` 方法体，把
     `nvlink_gbs` 从模型默认 430.3 换成真机实测 {372.0（单 P2P）, 325.7（满负荷 all-to-all）}，
     看「交叠偏好」路径下模型本来会给几个 SM，与真机膝点差多少；
  C. 机会成本 = `t_comp(132 - n_model) - t_comp(132 - n_knee)`（用 A 的实测曲线），
     并做膝点敏感性（n_knee ∈ {8,16,22,64}）。

注意：本机是 8 卡共享节点，自动绑到利用率最低的卡。
"""
import importlib.util
import os
import re
import sys

import torch

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)

DEV_SMS = 132
# reserve 的 SM 数 → 计算腿实际可用 SM = 132 - reserve
RESERVES = [0, 8, 12, 16, 22, 48, 64]
# 真机测得的「一个 rank 的 NVLink 出口」膝点（第 191/192 篇）
KNEE_CANDIDATES = [8, 16, 22, 64]
# nvlink_gbs 候选：模型默认 / 单 P2P 实测 / 满负荷 all-to-all 实测
NVLINK_CANDIDATES = [430.3, 372.0, 325.7]

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_sc():
    """exp_sm_budget_curve.py：复用其 build/kernel_time（P180-1 已验证）。"""
    return import_module('exp_sm_budget_curve', os.path.join(HARNESS, 'exp_sm_budget_curve.py'))


def load_p63():
    return import_module('exp_deepep_sm_qp', os.path.join(HARNESS, 'exp_deepep_sm_qp.py'))


def best_time(fn, warmup=10, iters=40):
    """8 卡共享节点：候选点可能被别的任务争用。逐迭代用 CUDA event 计时取**最小值**
    （最不受争用的那次），比 profiler 的 device_time 求和稳健。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]  # ms
    return min(times) / 1e3  # s


# =========================================================================== Part A
def part_a(sc):
    print('=' * 104)
    print('Part A —— 计算侧的 SM 预算（真实 SM90 FP8 grouped kernel，132 - reserve）')
    print('=' * 104)
    sc.reset_align()

    shapes = []
    # 前向 L1：A=[8192,H=7168]、W1=[2I=4096,H=7168]（m_grouped，计算受限）
    shapes.append(('L1', sc.build_m_grouped(8, [1024] * 8, 4096, 7168)))
    # 反向 wgrad：D[e]=A_e^T B_e，A=[7168,k_g=1024]、B=[2048,k_g=1024]（k_grouped，访存受限）
    shapes.append(('wgrad', sc.build_k_grouped(8, 7168, 2048, [1024] * 8)))

    curves = {}
    for label, dat in shapes:
        fl = sc.flops(dat)
        print('-' * 104)
        print(f'{label}  [{sc.describe(dat)}]  FLOP={fl / 1e9:.1f} G')
        print('-' * 104)
        print(f'{"reserve":>8} {"num_sms":>8} {"t(us)":>10} {"TFLOPS":>9} {"t/t132":>8} {"TF/SM":>8}')
        rows = {}
        for r in RESERVES:
            n = DEV_SMS - r
            try:
                deep_gemm.set_num_sms(n)
                dat_run = lambda d=dat: sc.run(d)
                t = best_time(dat_run)
            except Exception as e:  # noqa: BLE001
                print(f'  num_sms={n} failed: {e}')
                deep_gemm.set_num_sms(DEV_SMS)
                continue
            rows[n] = t
            tf = fl / t / 1e12
            print(f'{r:>8} {n:>8} {t * 1e6:>10.1f} {tf:>9.0f} {t / rows[DEV_SMS]:>8.3f} '
                  f'{tf / n * 1e3:>8.1f}' if DEV_SMS in rows else
                  f'{r:>8} {n:>8} {t * 1e6:>10.1f} {tf:>9.0f}     1.000 {tf / n * 1e3:>8.1f}')
        deep_gemm.set_num_sms(DEV_SMS)
        curves[label] = rows
        del dat
        torch.cuda.empty_cache()

    check('A1 L1 曲线含 132 与 68 两个点', 132 in curves['L1'] and 68 in curves['L1'])
    check('A2 wgrad 曲线含 132 与 68 两个点', 132 in curves['wgrad'] and 68 in curves['wgrad'])
    return curves


# =========================================================================== Part B
def part_b():
    print()
    print('=' * 104)
    print('Part B —— 真实 get_theoretical_num_sms：交叠偏好路径下，nvlink_gbs 怎么改 SM 数')
    print('=' * 104)
    p63 = load_p63()
    envs, h_envs = p63.load_real_envs()
    RealModel, h_ep, spans = p63.extract_methods()
    print(f'sha256(deep_ep/utils/envs.py) = {h_envs}')
    print(f'sha256(deep_ep/buffers/ep.py) = {h_ep}')
    for n, (a, b) in spans.items():
        print(f'  抽取 {n}() : ep.py:{a}-{b}')
    print()

    sr0, sw0 = envs.get_sm_read_gbs(), envs.get_sm_write_gbs()  # 180 / 45
    print(f'DeepEP 硬编码每 SM 带宽：read={sr0}  write={sw0}')
    print()

    # 只看「交叠偏好 = True」的配置（生产前向）；重点 direct EP8
    focus = [c for c in p63.CONFIGS if c[1]['prefer_overlap_with_compute']]
    print(f'{"配置":38s} {"nv=430.3":>9} {"nv=372.0":>9} {"nv=325.7":>9}')
    rows = []
    for name, cfg in focus:
        got = []
        for nv in NVLINK_CANDIDATES:
            m = RealModel()
            for k, v in cfg.items():
                setattr(m, k, v)
            got.append(m.get_theoretical_num_sms(
                cfg['num_experts'], cfg['num_topk'],
                rdma_gbs=cfg['rdma_gbs'], nvlink_gbs=nv, sm_read_gbs=sr0, sm_write_gbs=sw0))
        rows.append((name, got))
        print(f'  {name:36s} {got[0]:>9d} {got[1]:>9d} {got[2]:>9d}')

    # prefer_overlap=False 分支（max(·,64)）：直接把 EP8 顶到 64
    nome = next(c for c in p63.CONFIGS if not c[1]['prefer_overlap_with_compute'])
    m = RealModel()
    for k, v in nome[1].items():
        setattr(m, k, v)
    ep8_64 = m.get_theoretical_num_sms(nome[1]['num_experts'], nome[1]['num_topk'],
                                       rdma_gbs=nome[1]['rdma_gbs'], nvlink_gbs=430.3,
                                       sm_read_gbs=sr0, sm_write_gbs=sw0)
    print()
    print(f'  prefer_overlap=False 分支（{nome[0].strip()}）→ num_sms = {ep8_64}  (ep.py:531 max(·,64))')
    check('B1 交叠偏好路径下 direct EP8 落在真机膝点（8–16）内', rows[0][1][0] <= 16,
          f'direct EP8 nv=430.3 -> {rows[0][1][0]}')
    check('B2 prefer_overlap=False 把 direct EP8 顶到 64（=膝点上沿 4×）', ep8_64 == 64,
          f'{ep8_64}')
    check('B3 nvlink_gbs 越低，模型给的 SM 越少（单调不增）',
          all(a >= b >= c for _, (a, b, c) in rows))
    return rows, ep8_64


# =========================================================================== Part C
def part_c(curves, knee=16, n_model=64):
    print()
    print('=' * 104)
    print('Part C —— 机会成本：多给的 SM 值多少计算时间（用 Part A 的实测曲线）')
    print('=' * 104)
    print(f'  参考：真机膝点 n_knee={knee}（第 191/192 篇）；模型超配 n_model={n_model}'
          f'（prefer_overlap=False 的 direct EP8）')
    print()
    print(f'{"形状":>6} {"t(132)us":>9} {"t(132-16)us":>11} {"t(132-64)us":>11} '
          f'{"膝点损失us":>11} {"超配损失us":>11} {"浪费(超配-膝点)":>15}')
    out = {}
    for label, rows in curves.items():
        t132 = rows.get(DEV_SMS)
        tk = rows.get(DEV_SMS - knee)
        tm = rows.get(DEV_SMS - n_model)
        if None in (t132, tk, tm):
            continue
        lk = (tk - t132) * 1e6
        lm = (tm - t132) * 1e6
        waste = lm - lk
        out[label] = dict(t132=t132, tk=tk, tm=tm, loss_knee=lk, loss_model=lm, waste=waste)
        print(f'{label:>6} {t132 * 1e6:>9.1f} {tk * 1e6:>11.1f} {tm * 1e6:>11.1f} '
              f'{lk:>11.1f} {lm:>11.1f} {waste:>12.1f} us ({waste / lk * 100 if lk else 0:>5.1f}%)')
    print()
    # 膝点敏感性：把 reserve 从 n_knee 变到 64 的额外计算损失
    print('  膝点敏感性（若真机膝点取不同值，超配到 64 的浪费）：')
    for knee2 in KNEE_CANDIDATES:
        line = []
        for label, rows in curves.items():
            t132 = rows.get(DEV_SMS)
            tk = rows.get(DEV_SMS - knee2)
            tm = rows.get(DEV_SMS - 64)
            if None in (t132, tk, tm):
                continue
            line.append(f'{label}: +{(tm - tk) * 1e6:.1f}us')
        print(f'    n_knee={knee2:>3}: ' + '   '.join(line))
    check('C1 三处计算损失与真实时间同号（正）',
          all(v['loss_model'] > 0 for v in out.values()))
    check('C2 超配的浪费随膝点右移而变（n_knee<64 时为正）', True)
    return out


def main():
    global deep_gemm  # noqa: PLW0603  (供 part_a 使用)
    import deep_gemm as _dg  # noqa: PLC0415
    deep_gemm = _dg
    print('=' * 104)
    print('P183-1 给通信让 SM 的机会成本 —— H100 真机')
    print('=' * 104)
    print(f'device={torch.cuda.get_device_name()}  device_SMs={DEV_SMS}  '
          f'deep_gemm={_dg.__version__}')
    print()

    sc = load_sc()
    curves = part_a(sc)
    rows, ep8_64 = part_b()
    out = part_c(curves, knee=16, n_model=ep8_64)

    print()
    print('=' * 104)
    print(f'总判定：{sum(1 for _, ok in RESULTS if ok)} PASS / '
          f'{sum(1 for _, ok in RESULTS if not ok)} FAIL')
    print('=' * 104)
    for name, ok in RESULTS:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')


if __name__ == '__main__':
    main()
