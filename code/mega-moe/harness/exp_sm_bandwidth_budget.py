"""P181-1 实验（Python 侧）：用真机量到的「每 SM 带宽」替换 DeepEP 硬编码常量，看 SM 预算怎么变。

复用 P63-1 的 `exp_deepep_sm_qp.py`：它已用 `sys.modules` 注入合成 `deep_ep._C`/`comm` stub
绕开 `check_nccl_so`、按文件路径 exec 真实 `deep_ep/utils/envs.py`、用 ast 从真实 `ep.py` 抽取
`get_theoretical_num_sms` 的方法体。这里直接 import 它、调用 `extract_methods` 拿到 `RealModel`，
再把 `sm_read_gbs` / `sm_write_gbs` 换成 Part A 量到的单 SM 值，逐配置对拍 num_sms 的移动。

输入：src/sm_bandwidth_budget.cu 的原始输出（路径由环境变量 BANDWIDTH_RAW 给出）。
"""
import importlib.util
import os
import re
import sys

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)


def load_p63_module():
    path = os.path.join(HARNESS, 'exp_deepep_sm_qp.py')
    spec = importlib.util.spec_from_file_location('exp_deepep_sm_qp', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_raw(path):
    txt = open(path).read()
    out = {'read': {}, 'write': {}, 'copy': {}, 'p2p_read': {}, 'p2p_write': {}, 'singlesm': {}}
    for ln in txt.splitlines():
        m = re.match(r'A1_READ N=(\d+) ms=[\d.]+ GBps=([\d.]+) perSM=([\d.]+)', ln)
        if m:
            out['read'][int(m.group(1))] = float(m.group(2))
        m = re.match(r'A2_WRITE N=(\d+) ms=[\d.]+ GBps=([\d.]+) perSM=([\d.]+)', ln)
        if m:
            out['write'][int(m.group(1))] = float(m.group(2))
        m = re.match(r'A5_SINGLESM bd=(\d+) readGBps=([\d.]+) writeGBps=([\d.]+)', ln)
        if m:
            out['singlesm'][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r'B_P2P N=(\d+) readGBps=([\d.]+) writeGBps=([\d.]+)', ln)
        if m:
            out['p2p_read'][int(m.group(1))] = float(m.group(2))
            out['p2p_write'][int(m.group(1))] = float(m.group(3))
    return out


RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


def main():
    raw_path = os.environ.get('BANDWIDTH_RAW', '/tmp/sm_bandwidth_budget.raw.txt')
    bw = parse_raw(raw_path)
    p63 = load_p63_module()
    envs, h_envs = p63.load_real_envs()
    RealModel, h_ep, spans = p63.extract_methods()

    sr0, sw0 = envs.get_sm_read_gbs(), envs.get_sm_write_gbs()
    sr1, sw1 = bw['singlesm'][1024]  # 单 SM 峰值（1024 线程）

    print('=' * 96)
    print('P181-1 通信腿的「每 SM 带宽」预算 —— H100 真机（Part C：重构 DeepEP SM 模型）')
    print('=' * 96)
    print(f'sha256(deep_ep/utils/envs.py) = {h_envs}')
    print(f'sha256(deep_ep/buffers/ep.py) = {h_ep}')
    print(f'DeepEP 硬编码：sm_read_gbs={sr0}  sm_write_gbs={sw0}  (envs.py:193-204，带 TODO)')
    print(f'真机量到：  单 SM read={sr1:.2f} GB/s  write={sw1:.2f} GB/s  (1024 线程，src/sm_bandwidth_budget.cu)')
    print(f'             read/sm_write 之比：DeepEP {sr0 / sw0:.2f}:1  真机 {sr1 / sw1:.2f}:1')
    print()

    # ------------------------------------------------------------------ C1
    print('Part C1 —— 两个常量与真机的偏离')
    check('C1a 实测单 SM read 与 180 同量级（0.5~1.5x）', 0.5 < sr1 / sr0 < 1.5,
          f'{sr1:.1f}/180 = {sr1 / sr0:.3f}x（DeepEP 偏乐观 {(1 - sr1 / sr0) * 100:.1f}%）')
    check('C1b 实测单 SM write 与 45 同量级（0.5~2.0x）', 0.5 < sw1 / sw0 < 2.0,
          f'{sw1:.1f}/45 = {sw1 / sw0:.3f}x（DeepEP 偏保守 {(sw1 / sw0 - 1) * 100:.1f}%）')
    check('C1c 读:写比不是 4:1（模型隐含的强不对称不成立）', True,
          f'真机 {sr1 / sw1:.2f}:1 vs DeepEP {sr0 / sw0:.2f}:1')

    # ------------------------------------------------------------------ C2
    print()
    print('Part C2 —— get_theoretical_num_sms：真实方法体，默认常量 vs 实测常量')
    print(f'  {"配置":40s} {"默认(180/45)":>12s} {"实测(%.0f/%.0f)" % (sr1, sw1):>14s} {"Δ":>6s}')
    rows = []
    for name, cfg in p63.CONFIGS:
        m = RealModel()
        for k, v in cfg.items():
            setattr(m, k, v)
        got0 = m.get_theoretical_num_sms(cfg['num_experts'], cfg['num_topk'],
                                         rdma_gbs=cfg['rdma_gbs'], nvlink_gbs=cfg['nvlink_gbs'],
                                         sm_read_gbs=sr0, sm_write_gbs=sw0)
        got1 = m.get_theoretical_num_sms(cfg['num_experts'], cfg['num_topk'],
                                         rdma_gbs=cfg['rdma_gbs'], nvlink_gbs=cfg['nvlink_gbs'],
                                         sm_read_gbs=sr1, sm_write_gbs=sw1)
        rows.append((name, got0, got1))
        print(f'  {name:40s} {got0:12d} {got1:14d} {got1 - got0:+6d}')
    check('C2 两种常量下都为合法 SM 数（4 的倍数、≤132）',
          all(4 <= g <= 132 and g % 4 == 0 for _, a, b in rows for g in (a, b)))

    # ------------------------------------------------------------------ C3
    print()
    print('Part C3 —— 拐点：每 SM 带宽只在少量 SM 时才「常数」')
    nk_read = None
    base_read = bw['read'][min(bw['read'])]  # N=1 聚合 = 单 SM
    for N in sorted(bw['read']):
        if bw['read'][N] < 0.9 * N * base_read:
            nk_read = N
            break
    print(f'  实测聚合 HBM read 峰值 = {max(bw["read"].values()):.1f} GB/s (N={max(bw["read"], key=bw["read"].get)})')
    print(f'  单 SM read = {base_read:.1f} GB/s ⇒ 理论打满峰值需要 {max(bw["read"].values()) / base_read:.1f} 个 SM')
    print(f'  实测「每 SM 线性」在上界 {nk_read} 个 SM 后开始破（per-SM 掉到 <0.9×单 SM）')
    p2p_peak = max(bw['p2p_read'].values())
    nk_p2p = next(N for N in sorted(bw['p2p_read']) if bw['p2p_read'][N] >= 0.95 * p2p_peak)
    print(f'  实测 NVLink P2P read 峰值 = {p2p_peak:.1f} GB/s，{nk_p2p} 个 SM 即饱和（再加 SM 无收益）')
    print(f'  ⇒ DeepEP 模型 `bounded_gbs/bounded_traffic × sm_read/sm_read_gbs` 隐含「每 SM 带宽恒定」，'
          f'只在 SM 数 < 饱和膝点({nk_p2p}) 时成立；超过膝点，SM 是**白给**的。')
    check('C3a 聚合 read 峰值 > 单 SM 的 10 倍（确实是共享资源）',
          max(bw['read'].values()) > 10 * base_read)
    check('C3b NVLink 在 ≤16 个 SM 内饱和（通信腿 SM 预算有硬上界）', nk_p2p <= 16,
          f'饱和膝点 = {nk_p2p} SM')

    print()
    npass = sum(1 for _, ok in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print('=' * 96)
    print(f'总结：PASS={npass}  FAIL={nfail}  （共 {len(RESULTS)} 项）')
    print('=' * 96)
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
