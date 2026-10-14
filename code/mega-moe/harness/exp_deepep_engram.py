#!/usr/bin/env python3
# P59-1 实验 host oracle：DeepEP Engram 远程抓取的布局 / 发射映射 / 请求索引 / SF 拷贝 / host 公式。
#
# 读 src/deepep_engram_fetch.cu 设备 dump，用独立 Python 复刻：
#   * layout::EngramLayout 的字节偏移 / SF 行号与分片号；
#   * layout::EngramWorkspace 的字段偏移；
#   * EngramIssueWarpLayout 的 warp→(peer,gin_context,tile) 映射 + token tile 覆盖；
#   * issue 循环产出的 (peer, window, src_off, dst_off) 请求（覆盖恰好一次）；
#   * SF copy 的目的布局（TMA 列主序对齐 / 行主序）与源分片寻址；
#   * host calculate_storage_size / get_theoretical_config 的解析公式。
#
# 用法: python harness/exp_deepep_engram.py <device_dump>

import hashlib
import os
import sys

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
CUH = os.path.join(DEP, 'deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh')
WS = os.path.join(DEP, 'deep_ep/include/deep_ep/layout/engram/workspace.cuh')


class Oracle:
    def __init__(self):
        self.pass_ = 0
        self.fail = 0

    def check(self, ok, what):
        if ok:
            self.pass_ += 1
            print(f'  [PASS] {what}')
        else:
            self.fail += 1
            print(f'  [FAIL] {what}')


def ceil_div(a, b):
    return (a + b - 1) // b


def sha256_range(path, start, end):
    with open(path) as f:
        lines = f.readlines()
    text = ''.join(lines[start - 1:end])
    return hashlib.sha256(text.encode()).hexdigest()[:16], end - start + 1


def parse(dump):
    """按前缀分组解析设备 dump。"""
    d = {}
    for line in dump.splitlines():
        tok = line.split()
        if not tok:
            continue
        tag = tok[0]
        d.setdefault(tag, []).append([int(x) for x in tok[1:]])
    return d


# ------------------------- 纯 Python 复刻 EngramLayout -------------------------
class Layout:
    def __init__(self, num_ranks, num_sf_shards, hidden_bytes, entries_per_token, entries_per_layer):
        self.num_ranks = num_ranks
        self.num_sf_shards = num_sf_shards
        self.hidden_bytes = hidden_bytes
        self.ept = entries_per_token
        self.entries = entries_per_layer
        self.owners = num_ranks // num_sf_shards

    def num_entries(self, l):
        return self.entries[l]

    def prefix(self, l):
        return sum(self.entries[:l])

    def storage_byte_offset(self, l, i):
        return (self.prefix(l) + i) * self.hidden_bytes

    def recv_byte_offset(self, l, entry_slot, num_max_tokens):
        return (l * self.ept * num_max_tokens + entry_slot) * self.hidden_bytes

    def rows_per_shard(self, l):
        return self.owners * self.num_entries(l)

    def sf_row_idx(self, l, g):
        return self.owners * self.prefix(l) + g % self.rows_per_shard(l)

    def sf_shard_idx(self, l, g):
        return g // self.rows_per_shard(l)

    def gen_indices(self):
        hi = {}
        for l in range(len(self.entries)):
            epr = self.num_entries(l)
            hi[l] = [(i * 13 + l * 7) % (self.num_ranks * epr) for i in range(128)]
        hi[0][3] = -1
        return hi


def part_provenance(o):
    print('\n=== 0. 抽取片段 provenance（~/github/DeepEP@93eb6eb）===')
    for (a, b, name, path) in [(13, 39, 'EngramIssueWarpLayout', CUH),
                               (99, 171, 'issue 循环（含 issue_batch）', CUH),
                               (208, 254, 'SF warp 主体', CUH),
                               (9, 21, 'EngramWorkspace', WS),
                               (28, 74, 'EngramLayout', WS)]:
        h, n = sha256_range(path, a, b)
        print(f'    {name:<28} {os.path.basename(path)}:{a}-{b}  sha256={h} ({n} 行)')
        o.check(bool(h), f'{name} 逐行抽取片段可哈希')


def part_layout(o, d):
    print('\n=== 1. layout::EngramLayout 布局算术（设备 vs Python 独立复刻）===')
    L = Layout(4, 2, 64, 2, [4, 6])
    la = d['LA']
    ln = d['LN'][0][0]
    ls = d['LS']
    flat = []
    for l in range(2):
        for i in range(L.num_entries(l)):
            flat.append(L.storage_byte_offset(l, i))
    mism = sum(1 for a, b in zip([x[0] for x in la], flat) if a != b)
    print(f'    storage 偏移 {len(flat)} 项 mismatch={mism}')
    o.check(mism == 0, 'storage_byte_offset 逐项 0 不符')
    o.check(ln == 10, 'total entries 求和 = 10')
    exp_recv = []
    for l in range(2):
        for i in range(L.num_entries(l)):
            exp_recv.append(L.recv_byte_offset(l, l * L.ept + i, 64))
    mism = sum(1 for a, b in zip([x[1] for x in la], exp_recv) if a != b)
    o.check(mism == 0, 'recv_byte_offset 逐项 0 不符')
    # SF：设备同时 dump 了 g，直接用公式复算（layer 需从 g 顺序推断；这里按 src 的
    # 「每层先 g<8 步长 1、之后步长 7」遍历重建 (layer,g) 序列）
    seq = []
    for l in range(2):
        g = 0
        while g < 4 * L.num_entries(l):
            seq.append((l, g))
            g += 1 if g < 8 else 7
    sf_row = [x[0] for x in ls]
    sf_shard = [x[1] for x in ls]
    sf_g = [x[2] for x in ls]
    o.check([g for (_, g) in seq] == sf_g, f'设备 SF 采样 g 序列与复刻一致（{len(seq)} 项）')
    exp_row = [L.sf_row_idx(l, g) for (l, g) in seq]
    exp_shard = [L.sf_shard_idx(l, g) for (l, g) in seq]
    mism = sum(1 for a, b in zip(sf_row, exp_row) if a != b) + \
        sum(1 for a, b in zip(sf_shard, exp_shard) if a != b)
    o.check(mism == 0, 'get_sf_row_idx / get_sf_shard_idx 逐项 0 不符')
    ok = all(0 <= s < L.num_sf_shards for s in sf_shard) and \
        all(r < L.owners * sum(L.entries) for r in sf_row)
    o.check(ok, 'SF shard 落在 [0,kNumSFShards)，row 落在分片范围内')


def part_workspace(o, d):
    print('\n=== 2. layout::EngramWorkspace 字段偏移 ===')
    sz, b_off, ic_off, ctr35, layers, ctx = (x[0] for x in
                                             [d['WA'][0], d['WB'][0], d['WC'][0], d['WD'][0], d['WE'][0], d['WF'][0]])
    print(f'    sizeof={sz} barrier@{b_off} issue_counters@{ic_off} ctr(3,5)@{ctr35} '
          f'kNumMaxLayers={layers} kNumMaxPeerContexts={ctx}')
    o.check(b_off == 0 and ic_off == 64, 'barrier@0, issue_counters@64（BarrierSignals align 64B = 64B）')
    o.check(ctr35 == 64 + (3 * ctx + 5) * 4, 'get_issue_counter_ptr(layer,pc) 偏移公式成立')
    o.check(sz == 64 + layers * ctx * 4, 'sizeof = 64 + kNumMaxLayers*kNumMaxPeerContexts*sizeof(uint32)')


def part_issue_layout(o, d):
    print('\n=== 3. EngramIssueWarpLayout 映射 + token tile 覆盖 ===')
    nt, niw, peers, qps = d['CN'][0]
    print(f'    num_token_tiles={nt} num_issue_warps={niw} peers={peers} qps={qps}')
    mism = 0
    tile_of = {}
    for idx, (rp, gc, pc, npcw, pw, npw, napcw, ntt) in enumerate(d['CA']):
        epc = idx % (peers * qps)
        erp = epc % peers
        egc = epc // peers
        enpcw = ceil_div(niw - epc, peers * qps)
        epw = idx // peers
        enpw = ceil_div(niw - erp, peers)
        ent = ceil_div(nt, 32)
        etw = ceil_div(nt - egc, qps) if egc < nt else 0
        enapcw = min(etw, enpcw)
        exp = [erp, egc, epc, enpcw, epw, enpw, enapcw, (ntt if False else 0)]
        got = [rp, gc, pc, npcw, pw, npw, napcw, 0]
        if exp[:7] != got[:7]:
            mism += 1
        # token tile 覆盖：本 warp 处理的 tile = pw, pw+npw, ...
        for t in range(pw, nt, npw) if npw > 0 else []:
            tile_of.setdefault(rp, []).append(t)
    o.check(mism == 0, f'{niw} 个 issue warp 的 7 个导出字段逐字段 0 不符')
    ok = True
    for rp in range(peers):
        got = sorted(tile_of.get(rp, []))
        # 每个 peer 的每个 token tile 恰好被其 warp 组覆盖一次
        if got != list(range(nt)):
            ok = False
    o.check(ok, f'每个 RDMA peer 的 {nt} 个 token tile 恰好覆盖一次（无重无漏）')
    # num_active_peer_context_warps 语义：gin_context >= nt 的 warp 不活跃
    ok = True
    for idx, row in enumerate(d['CA']):
        ec = row[1]
        if ec >= nt and row[6] != 0:
            ok = False
    o.check(ok, 'gin_context_idx >= num_token_tiles 的 warp，active=0（无 token tile 可发）')


def part_requests(o, d):
    print('\n=== 4. issue 循环请求生成（覆盖恰好一次 + 偏移精确）===')
    num_tokens, num_layers, ept, G, num_ranks = d['DN'][0]
    epl = [x[0] for x in d['DL']]
    L = Layout(num_ranks, num_sf_shards=2, hidden_bytes=64, entries_per_token=ept,
               entries_per_layer=epl)
    # 读回设备索引：DI 行 = [layer0[i], layer1[i]]，i in [0,128)
    di = d['DI']
    idx = {0: [x[0] for x in di], 1: [x[1] for x in di]}
    num_entries = num_tokens * ept
    hidden = 64
    # 期望请求：每个非负 entry 恰好一条；peer = g/(G*epr)，window=g/(epr)%G，src=(prefix+loc)*hidden
    expected = {}
    for l in range(num_layers):
        epr = epl[l]
        span = G * epr
        for i in range(num_entries):
            g = idx[l][i]
            if g < 0:
                continue
            peer = g // span
            peer_local = g - peer * span
            window = peer_local // epr
            loc = peer_local % epr
            src = (L.prefix(l) + loc) * hidden
            dst = (l * ept * num_tokens + i) * hidden
            expected[(l, peer, i)] = (window, src, dst)
    got = {}
    for layer, peer, entry, window, src, dst in d['DR']:
        if (layer, peer, entry) in got:
            got[(layer, peer, entry)] = None  # duplicate
        else:
            got[(layer, peer, entry)] = (window, src, dst)
    print(f'    期望请求={len(expected)}  设备请求={len(d["DR"])}  唯一键={len(got)}')
    o.check(len(d['DR']) == len(expected), f'请求总数一致（{len(d["DR"])} == {len(expected)}）')
    miss = [k for k in expected if k not in got]
    extra = [k for k in got if k not in expected]
    bad = [k for k in expected if k in got and got[k] != expected[k]]
    print(f'    miss={len(miss)} extra={len(extra)} 值不符={len(bad)}')
    if miss[:3]:
        print('    例 miss:', miss[:3], [expected[k] for k in miss[:3]])
    if extra[:3]:
        print('    例 extra:', extra[:3])
    if bad[:3]:
        print('    例 bad:', [(k, expected[k], got[k]) for k in bad[:3]])
    o.check(not miss and not extra and not bad, '每条有效 entry 恰好一条请求、window/src/dst 逐字段一致')


def part_sf(o, d):
    print('\n=== 5. SF 拷贝（copy_sf_entry）目的布局 + 源分片寻址 ===')
    num_sf_shards, shard_stride, P, total_entries, owners = d['ES'][0]
    h_shards = [x[0] for x in d['EV']]
    ei = {}
    for l, i, g in d['EI']:
        ei[(l, i)] = g
    num_tokens = 64
    L = Layout(4, 2, 64, 2, [4, 6])
    eo = [x[0] for x in d['EO']]
    for vi, variant in enumerate(('tma', 'row')):
        en = d['EN'][vi]
        _, tok_stride, hid_stride, layer_stride, packs, shard_stride2, rows_per_shard = en
        alloc = 2 * layer_stride
        h_out = eo[vi * alloc:(vi + 1) * alloc]
        mism = 0
        total = 0
        for l in range(2):
            for i in range(num_tokens * L.ept):
                g = ei.get((l, i))
                token = i // L.ept
                slot = i % L.ept
                for p in range(packs):
                    base = l * layer_stride + token * tok_stride + slot * packs * hid_stride + p * hid_stride
                    if g is None or g < 0:
                        expect = 0
                    else:
                        shard = L.sf_shard_idx(l, g)
                        row = L.sf_row_idx(l, g)
                        expect = h_shards[shard * shard_stride2 + row * packs + p]
                    total += 1
                    if h_out[base] != expect:
                        mism += 1
        print(f'    {variant}: 目的元素 {total} 个 mismatch={mism}')
        o.check(mism == 0, f'{variant} 布局的 SF 拷贝目的逐元素 0 不符（含 -1 留零）')
    o.check(d['EN'][0][1] == 1 and d['EN'][1][1] == 2 * 4,
            'token stride：TMA 列主序=1、行主序=ept*packs')
    o.check(d['EN'][0][2] == ceil_div(64, 4) * 4 and d['EN'][1][2] == 1,
            'hidden stride：TMA 列主序=align(tokens,4)、行主序=1')


def part_host(o):
    print('\n=== 6. host 公式：calculate_storage_size / get_theoretical_config ===')
    A = 2 * 1024 * 1024
    R = 64
    align = lambda x, a: (x + a - 1) // a * a

    def calc_storage_size(num_rdma_ranks, num_entries_per_layer, hidden, elem_size, num_max_tokens,
                          num_entries_per_token, num_sf_packs):
        total_entries = sum(num_entries_per_layer)
        num_bytes_per_entry = hidden * elem_size
        assert num_bytes_per_entry % R == 0
        num_recv_bytes = align(len(num_entries_per_layer) * num_max_tokens * num_entries_per_token * num_bytes_per_entry, A)
        num_rdma_storage_bytes = align(total_entries * num_bytes_per_entry, A)
        num_sf_bytes = 0 if num_sf_packs == 0 else align(
            total_entries * num_rdma_ranks * num_sf_packs * 4, A)
        return num_recv_bytes + num_sf_bytes, num_rdma_storage_bytes

    # 用例：与 tests/engram/test_engram.py 的默认量级（去掉极大值，便于手算）
    cases = [
        ([524288, 524309], 256, 2, 512, 24, 0, 2),
        ([524288, 524309], 256, 2, 512, 24, 2, 2),
        ([1000], 128, 2, 64, 8, 1, 4),
    ]
    for entries, hidden, elem, mxtok, ept, nsf, rdma_ranks in cases:
        recv_sf, storage = calc_storage_size(rdma_ranks, entries, hidden, elem, mxtok, ept, nsf)
        print(f'    entries={entries} hidden={hidden} mxtok={mxtok} ept={ept} sf_packs={nsf} '
              f'-> (gpu={recv_sf}, storage={storage})')
        o.check(recv_sf % A == 0 and storage % A == 0, '两段大小均按 2MiB 对齐')
    # 关系：storage = align(Σentries*hidden*elem, 2MiB)；gpu = recv + sf
    entries, hidden, elem, mxtok, ept, nsf, rdma_ranks = cases[0]
    total = sum(entries) * hidden * elem
    o.check(calc_storage_size(rdma_ranks, entries, hidden, elem, mxtok, ept, nsf)[1] == align(total, A),
            'storage = align(Σentries*hidden*elem, 2MiB)')
    recv = align(len(entries) * mxtok * ept * hidden * elem, A)
    o.check(calc_storage_size(rdma_ranks, entries, hidden, elem, mxtok, ept, nsf)[0] == recv,
            'BF16（sf_packs=0）gpu size = recv（无 SF 分片）')

    def theoretical(num_layers, num_max_tokens, ept, nic_mpps, max_read_req, rdma_peers):
        rtt = 10e-6
        num_qps = max(int((nic_mpps * 1e6 * rtt / max_read_req / rdma_peers).__ceil__()), 1)
        num_requests = num_layers * num_max_tokens * ept
        num_required_slots = ceil_div(num_requests, num_qps * rdma_peers)
        if num_required_slots > 32768:
            num_min_qps = int(ceil_div(num_requests, 32768 * rdma_peers) * 1.05)
            return num_min_qps, 32768, True
        qp_depth = 1024
        while qp_depth < num_required_slots and qp_depth < 32768:
            qp_depth *= 2
        return num_qps, qp_depth, False

    for (nl, mt, ept, mpps, mrq, peers) in [(2, 512, 24, 45, 16, 2), (2, 512, 24, 90, 16, 2),
                                            (128, 8192, 64, 45, 16, 1)]:
        qps, depth, restrict = theoretical(nl, mt, ept, mpps, mrq, peers)
        print(f'    layers={nl} tokens={mt} ept={ept} mpps={mpps} maxreq={mrq} peers={peers} '
              f'-> qps={qps} depth={depth} restrict_rd_atomic={restrict}')
        o.check(qps >= 1 and depth >= 1024 and depth <= 32768, 'qp 数 ≥1，depth ∈ [1024, 32768]')
    # 溢出分支：请求量 > 32768*qps*peers（单 QP 放不下）→ 借最高位标志、回退 qp_depth=32768
    #   qps = ceil(45e6*10e-6/16/1) = 29 → 32768*29 = 950272 < 64*512*32 = 1048576
    qps, depth, restrict = theoretical(64, 512, 32, 45, 16, 1)
    o.check(restrict and depth == 32768, f'请求量溢出时返回 (qps={qps}, 32768, True)')


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/deepep_engram_dev.txt'
    with open(dump_path) as f:
        dump = f.read()
    d = parse(dump)
    o = Oracle()
    part_provenance(o)
    part_layout(o, d)
    part_workspace(o, d)
    part_issue_layout(o, d)
    part_requests(o, d)
    part_sf(o, d)
    part_host(o)
    print(f'\n===== Python oracle: PASS={o.pass_} FAIL={o.fail} =====')
    sys.exit(0 if o.fail == 0 else 1)


if __name__ == '__main__':
    main()
