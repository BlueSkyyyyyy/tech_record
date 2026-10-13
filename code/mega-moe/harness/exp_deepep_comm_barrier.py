#!/usr/bin/env python3
"""P53-1: DeepEP 多 rank 屏障与 symmetric-memory 句柄层 —— host 侧公式与设备结果交叉核对。

本文件不 import deep_ep（本机缺 `deep_ep._C`，见 ROADMAP 阻塞）：逐行等价复刻
  * csrc/kernels/comm/context.cpp:52-63   get_physical_domain_size / get_logical_domain_size
  * csrc/kernels/comm/context.cpp:96/100  workspace 对齐 / num_gpu_timeout_cycles
  * csrc/kernels/comm/context.cpp:140     ginSignalCount = num_ranks + 2*2
  * csrc/kernels/comm/context.cpp:174-183 scaleout/scaleup 域分配与 is_scaleup_nvlink
  * deep_ep/include/deep_ep/comm/barrier.cuh:56-86  get_qp_mode（SM/QP 映射）
  * deep_ep/include/deep_ep/comm/barrier.cuh:97-131 nvlink barrier 的四相位「免复位」循环
然后读入设备实验（src/deepep_comm_barrier.cu）写出的 /tmp dump，**逐字段交叉核对**。
"""
import sys


def physical_domain_size(num_ranks, num_nvl_ranks):          # context.cpp:52-57
    assert num_ranks % num_nvl_ranks == 0
    return num_ranks // num_nvl_ranks, num_nvl_ranks


def logical_domain_size(num_ranks, num_nvl_ranks, allow_hybrid):  # context.cpp:59-63
    num_rdma_ranks, num_nvl = physical_domain_size(num_ranks, num_nvl_ranks)
    return (num_rdma_ranks if allow_hybrid else 1,
            num_nvl if allow_hybrid else num_rdma_ranks * num_nvl)


def scaleout_scaleup(num_ranks, num_nvl_ranks, allow_hybrid):     # context.cpp:174-183
    num_rdma_ranks, num_nvl = physical_domain_size(num_ranks, num_nvl_ranks)
    if allow_hybrid:
        num_scaleout, num_scaleup = num_rdma_ranks, num_nvl
    else:
        num_scaleout, num_scaleup = 1, num_ranks
    is_scaleup_nvlink = (num_scaleup == num_nvl)
    return num_scaleout, num_scaleup, is_scaleup_nvlink


def gin_signal_count(num_ranks):                              # context.cpp:140
    return num_ranks + 2 * 2


def gpu_timeout_cycles(secs, clock_rate):                     # context.cpp:100
    return secs * clock_rate


def get_qp_mode(kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps,
                sm_idx, channel_in_sm_idx, is_notify_warp=False):  # barrier.cuh:56-86
    sharing_cta = 0
    sharing_grid = 0 if kNumSMs == 1 else 1
    if kNumQPs == 1:
        return 0, sharing_grid
    if is_notify_warp:
        return 0, sharing_cta
    qp_start = int(kWithNotifyWarps)
    num_avail = kNumQPs - qp_start
    if kNumSMs <= num_avail:
        num_qps_in_sm = (num_avail // kNumSMs) + (1 if sm_idx < (num_avail % kNumSMs) else 0)
        return qp_start + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, sharing_cta
    else:
        global_channel = sm_idx * kNumChannelsPerSM + channel_in_sm_idx
        return qp_start + (global_channel % num_avail), sharing_grid


def barrier_model(kNumRanks, num_iters):                      # barrier.cuh:97-131
    """四相位循环：counter&3 -> phase/sign；+1/-1 读写复用两个槽（免复位）。"""
    counter, s0, s1 = 0, 0, 0
    for _ in range(num_iters):
        status = counter & 3
        phase, sign = status & 1, status >> 1
        if phase == 0:
            s0 += -kNumRanks if sign else kNumRanks
        else:
            s1 += -kNumRanks if sign else kNumRanks
        counter += 1
    return counter, s0, s1


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/deepep_comm_barrier_dev.txt'
    lines = [ln.strip() for ln in open(dump_path).read().splitlines() if ln.strip()]

    passed = failed = 0

    def check(ok, what):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f'  [PASS] {what}')
        else:
            failed += 1
            print(f'  [FAIL] {what}')

    print('\n=== A. host 域分解（context.cpp:52-63/174-183）===')
    cases = [
        # (num_ranks, num_nvl_ranks, allow_hybrid)
        (8, 8, True),    # 单节点 8 卡：1 个 RDMA rank
        (16, 8, True),   # 2 节点 × 8 卡
        (64, 8, True),   # 8 节点 × 8 卡
        (8, 8, False),   # 关闭 hybrid：scaleup 覆盖全部
        (64, 8, False),
        (4, 4, True),
    ]
    for (nr, nn, hy) in cases:
        phys = physical_domain_size(nr, nn)
        logi = logical_domain_size(nr, nn, hy)
        so, su, nvl = scaleout_scaleup(nr, nn, hy)
        ok = (phys == (nr // nn, nn) and
              logi == ((nr // nn, nn) if hy else (1, nr)) and
              (so, su) == ((nr // nn, nn) if hy else (1, nr)) and
              nvl == (su == nn))
        print(f'    ranks={nr} nvl={nn} hybrid={int(hy)}: physical={phys} logical={logi} '
              f'scaleout/up=({so},{su}) is_scaleup_nvlink={int(nvl)} ginSignals={gin_signal_count(nr)}')
        check(ok, f'域分解 ranks={nr} nvl={nn} hybrid={int(hy)}')
    check(gin_signal_count(8) == 12 and gin_signal_count(64) == 68,
          'ginSignalCount = num_ranks + 4（barrier 需 num_ranks 个 scaleout 槽 + 2 相位 × 2 槽信号）')
    check(gpu_timeout_cycles(20, 1_980_000_000) == 39_600_000_000,
          'num_gpu_timeout_cycles = secs × device clock rate')

    print('\n=== B. get_qp_mode 与设备结果逐字段交叉核对（barrier.cuh:56-86）===')
    dev_qpmode = [ln for ln in lines if ln.startswith('qpmode ')]
    dev_maps = [ln for ln in lines if ln.startswith('qpmode_map ')]
    cfg_order = [(3, 10, 4, False), (8, 8, 1, False), (4, 2, 8, False),
                 (8, 8, 1, True), (1, 1, 4, False)]
    check(len(dev_qpmode) == len(cfg_order), f'设备 qpmode 用例数 = {len(dev_qpmode)}（期望 {len(cfg_order)}）')
    map_cursor = 0
    for (sms, qps, ch, notify) in cfg_order:
        dev_row = []
        for sm in range(sms):
            parts = dev_maps[map_cursor].split()[1:]
            dev_row.append([tuple(int(v) for v in p.split(',')) for p in parts])
            map_cursor += 1
        ref_row = [[get_qp_mode(sms, qps, ch, notify, sm, c, False) for c in range(ch)]
                   for sm in range(sms)]
        ok = True
        for sm in range(sms):
            for c in range(ch):
                # 设备 dump 的 mode：0=CTA/1=GPU，和 Python 一致
                if tuple(dev_row[sm][c]) != tuple(ref_row[sm][c]):
                    ok = False
        if notify:
            n = get_qp_mode(sms, qps, ch, notify, sms - 1, ch - 1, True)
            if n != (0, 0):
                ok = False
        print(f'    sms={sms} qps={qps} ch={ch} notify={int(notify)}: dev[SM0]={dev_row[0][:4]} '
              f'ref[SM0]={ref_row[0][:4]}')
        check(ok, f'get_qp_mode 逐字段一致（sms={sms} qps={qps} ch={ch} notify={int(notify)}）')

    print('\n=== C. NVLink barrier 四相位模型与设备结果交叉核对（barrier.cuh:88-131）===')
    dev_bar = [ln for ln in lines if ln.startswith('barrier ')]
    ok_all = len(dev_bar) > 0
    for ln in dev_bar:
        kv = dict(p.split('=') for p in ln.split()[1:])
        ranks, iters = int(kv['ranks']), int(kv['iters'])
        c, s0, s1 = barrier_model(ranks, iters)
        ok = (int(kv['counter']) == c and int(kv['slot0']) == s0 and int(kv['slot1']) == s1)
        ok_all = ok_all and ok
        print(f"    ranks={ranks} sms={kv['sms']}: dev counter={kv['counter']} slot0={kv['slot0']} "
              f"slot1={kv['slot1']} | ref counter={c} slot0={s0} slot1={s1}")
    check(ok_all, '所有 barrier 用例的 counter/双相位槽 与 host 模型逐字段一致（+1/-1 免复位）')

    # 免复位性：8 次调用后两槽恰好回到 (0,0)（因为 4 相位一循环 = +n,+n,-n,-n）。
    c4, a0, a1 = barrier_model(64, 4)
    c8, b0, b1 = barrier_model(64, 8)
    check((a0, a1) == (0, 0) and (b0, b1) == (0, 0),
          '4 相位一循环后两个信号槽回到 0（免复位、无需清零）')

    print('\n=== D. BarrierSignals 大小/对齐与设备一致 ===')
    dev_sig = [ln for ln in lines if ln.startswith('signals ')][0]
    kv = dict(p.split('=') for p in dev_sig.split()[1:])
    check(kv['sizeof'] == '64' and kv['align'] == '64',
          'BarrierSignals 64B（counter 8B + 双相位 8B，补齐到 RDMA 对齐）')

    print(f'\n=== 汇总：PASS={passed} FAIL={failed} ===')
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
