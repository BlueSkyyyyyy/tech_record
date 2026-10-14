"""P182-1 实验（Python 侧）：把「多对多（all-to-all）」下的 NVLink 膝点代回 DeepEP 的 SM 模型。

输入：src/nvlink_alltoall_budget.cu 的原始输出（环境变量 A2A_RAW）。
复用 P63-1 的 `exp_deepep_sm_qp.py`：它用 `sys.modules` 注入合成 `deep_ep._C`/`comm` stub
绕开 `check_nccl_so`、按文件路径 exec 真实 `deep_ep/utils/envs.py`、用 ast 从真实 `ep.py`
抽取 `get_theoretical_num_sms` 的方法体。这里 import 它拿 `load_real_envs` / `extract_methods`
/ `CONFIGS`，把真机量到的「每 rank 聚合 NVLink 带宽」和 DeepEP 的 `nvlink_gbs` 并排看。
"""
import importlib.util
import os
import re
import sys

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


def load_p63():
    path = os.path.join(HARNESS, 'exp_deepep_sm_qp.py')
    spec = importlib.util.spec_from_file_location('exp_deepep_sm_qp', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_raw(path):
    out = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}}
    for ln in open(path).read().splitlines():
        m = re.match(r'A_P1 N=(\d+) readGBps=([\d.]+) writeGBps=([\d.]+) perSMread=([\d.]+)', ln)
        if m:
            out['A'][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r'B_MP N=(\d+) readGBps=([\d.]+) writeGBps=([\d.]+) perSMread=([\d.]+)', ln)
        if m:
            out['B'][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r'C_STR N=(\d+) readGBps=([\d.]+) writeGBps=([\d.]+) perSMread=([\d.]+)', ln)
        if m:
            out['C'][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r'D_ALL N=(\d+) perRankWorstGBps=([\d.]+) perRankMin=([\d.]+) meanPerRank=([\d.]+) nodeAggGBps=([\d.]+)', ln)
        if m:
            out['D'][int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
        m = re.match(r'E_SWEEP npeer=(\d+) grid=64 readGBps=([\d.]+) perSMread=([\d.]+)', ln)
        if m:
            out['E'][int(m.group(1))] = float(m.group(2))
    return out


def knee(sweep, frac=0.95):
    """最小 N 使精度 >= frac*peak。"""
    peak = max(sweep.values())
    for N in sorted(sweep):
        if sweep[N] >= frac * peak:
            return N, peak
    return max(sweep), peak


def main():
    raw_path = os.environ.get('A2A_RAW', '/tmp/nvlink_alltoall_budget.raw.txt')
    r = parse_raw(raw_path)
    p63 = load_p63()
    envs, h_envs = p63.load_real_envs()
    bw = {'get_sm_read_gbs': envs.get_sm_read_gbs, 'get_sm_write_gbs': envs.get_sm_write_gbs,
          'get_rdma_gbs': envs.get_rdma_gbs, 'get_nvlink_gbs': envs.get_nvlink_gbs,
          'check_fast_rdma_atomic_support': envs.check_fast_rdma_atomic_support}
    RealModel, h_ep, spans = p63.extract_methods(bw)
    nvl = envs.get_nvlink_gbs()

    readA = {N: v[0] for N, v in r['A'].items()}
    readB = {N: v[0] for N, v in r['B'].items()}
    readC = {N: v[0] for N, v in r['C'].items()}
    writeA = {N: v[1] for N, v in r['A'].items()}
    writeB = {N: v[1] for N, v in r['B'].items()}

    print('=' * 96)
    print('P182-1 通信腿在「多对多（all-to-all）」下的每 SM 带宽预算 —— H100 真机')
    print('=' * 96)
    print(f'sha256(deep_ep/utils/envs.py) = {h_envs}')
    print(f'sha256(deep_ep/buffers/ep.py) = {h_ep}')
    for n, (a, b) in spans.items():
        print(f'  抽取 {n}()  : ep.py:{a}-{b}')
    print(f'DeepEP 认为的单 rank NVLink 总带宽：get_nvlink_gbs() = {nvl:.1f} GB/s')
    print()

    # ------------------------------------------------------------------ A1
    print('Part A1 —— 单 peer 基线 vs DeepEP 的 nvlink_gbs')
    nkA, pkA = knee(readA)
    print(f'  单 peer read 峰值 = {pkA:.1f} GB/s（{nkA} SM 处饱和）；write 峰值 = {max(writeA.values()):.1f}')
    check('A1a 真机单 peer read 峰值 ≈ DeepEP nvlink_gbs（0.7~1.1x）',
          0.7 < pkA / nvl < 1.1, f'{pkA:.1f}/{nvl:.1f} = {pkA / nvl:.3f}x')
    check('A1b 单 peer 在 ≤16 SM 内饱和', nkA <= 16, f'膝点 = {nkA} SM')

    # ------------------------------------------------------------------ A2
    print()
    print('Part A2 —— 多 peer 聚合：同样 SM 数下，peer 从 1 加到 6，带宽涨不涨？')
    nkB, pkB = knee(readB)
    print(f'  6 peer block-per-peer read 峰值 = {pkB:.1f} GB/s（{nkB} SM 处饱和）')
    for N in sorted(set(readA) & set(readB)):
        print(f'    N={N:4d}  1peer={readA[N]:7.1f}  6peer={readB[N]:7.1f}  比值={readB[N] / readA[N]:.3f}')
    check('A2a 6 peer 峰值 ≈ 1 peer 峰值（0.9~1.1x）—— 每 rank 聚合带宽是常数',
          0.9 < pkB / pkA < 1.1, f'{pkB:.1f}/{pkA:.1f} = {pkB / pkA:.3f}x')
    check('A2b 多 peer 膝点仍 ≤16 SM（不随 peer 数放大）', nkB <= 16, f'膝点 = {nkB} SM')

    # ------------------------------------------------------------------ A3
    print()
    print('Part A3 —— striped（每个 block 轮询所有 peer）把 MLP 摊到多链路')
    nkC, pkC = knee(readC)
    print(f'  striped read 峰值 = {pkC:.1f} GB/s（{nkC} SM 处饱和）')
    for N in sorted(readC):
        print(f'    N={N:4d}  striped={readC[N]:7.1f}  perSM={readC[N] / N:6.2f}')
    check('A3a striped 在 ≤16 SM 内饱和', nkC <= 16, f'膝点 = {nkC} SM')
    check('A3b striped 峰值 ≥ block-per-peer 峰值（0.95x）',
          pkC >= 0.95 * pkB, f'{pkC:.1f} vs {pkB:.1f}')

    # ------------------------------------------------------------------ A4
    print()
    print('Part A4 —— 固定 grid=64，扫 peer 数（1..6）：聚合带宽是否恒定？')
    E = r['E']
    for np_ in sorted(E):
        print(f'    npeer={np_}  grid=64  readGBps={E[np_]:.1f}')
    emax, emin = max(E.values()), min(E.values())
    check('A4 peer 数 1->6 聚合带宽变化 <1.25x（不随 peer 线性放大）',
          emax / emin < 1.25, f'max/min = {emax:.1f}/{emin:.1f} = {emax / emin:.3f}')

    # ------------------------------------------------------------------ A5
    print()
    print('Part D —— 全部 rank 同时 all-to-all（7 rank 全负荷）')
    for N in sorted(r['D']):
        worst, mn, mean, agg = r['D'][N]
        print(f'    N={N:4d}  perRank worst={worst:7.1f} min={mn:7.1f} mean={mean:7.1f}  nodeAgg={agg:7.1f}')
    dN = max(r['D'])
    check('A5a 满负荷下每 rank 仍 >= 0.8x 单租户峰值',
          r['D'][dN][0] >= 0.8 * pkA, f'{r["D"][dN][0]:.1f} vs 0.8*{pkA:.1f}={0.8 * pkA:.1f}')
    check('A5b 节点聚合带宽 > 单 rank 的 4 倍（全 NVSwitch 在动）',
          r['D'][dN][3] > 4 * pkA, f'{r["D"][dN][3]:.1f} GB/s')

    # ------------------------------------------------------------------ B
    print()
    print('Part B —— 代回 DeepEP 的真实 SM 模型（真实方法体 + 默认常量）')
    rows = []
    for name, cfg in p63.CONFIGS:
        m = RealModel()
        for k, v in cfg.items():
            setattr(m, k, v)
        g = m.get_theoretical_num_sms(cfg['num_experts'], cfg['num_topk'],
                                      rdma_gbs=cfg['rdma_gbs'], nvlink_gbs=cfg['nvlink_gbs'])
        rows.append((name, g))
        print(f'    {name:42s} num_sms={g}')
    ep8 = next(g for n, g in rows if n.startswith('direct EP8/LSA8'))
    ep8_noov = next(g for n, g in rows if n.startswith('direct EP8, not'))
    knee_hi = max(nkA, nkB, nkC)
    print(f'  真机测得「一条 NVLink（含多 peer 聚合）饱和膝点」= {min(nkA, nkB, nkC)}~{knee_hi} SM')
    check('B1 模型对 direct EP8 给出的 SM 数落在真机膝点的 1~4x 内',
          1 <= ep8 / knee_hi <= 4, f'model={ep8}  vs knee={knee_hi}')
    check('B2 prefer_overlap=False 时模型超配到真机膝点的 >=4x（白给的 SM）',
          ep8_noov >= 4 * knee_hi, f'model={ep8_noov}  vs 4x knee = {4 * knee_hi}')

    print()
    npass = sum(1 for _, ok in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print('=' * 96)
    print(f'总结：PASS={npass}  FAIL={nfail}  （共 {len(RESULTS)} 项）')
    print('=' * 96)
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
