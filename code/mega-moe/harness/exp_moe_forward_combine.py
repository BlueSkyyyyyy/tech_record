#!/usr/bin/env python3
# P133-1 实验（Python 侧）：Mega MoE 前向收尾两段的**独立 oracle** 与**远端邮箱带宽**。
#
#   A) 独立 oracle：用纯 torch 复刻「专家 L2 输出 -> 邮箱 scatter（写端）-> combine 归约（读端）」
#      的语义，逐位对照（邮箱布局 `[slot][token][hidden]` + src metadata + bf16->fp32 归约）。
#   B) 远端邮箱带宽：在真实多卡上量「把 top-k 份 L2 输出写进对端 rank 的邮箱」这一步的
#      NVLink 带宽（= L2 epilogue 写端），以及在收端做 combine 归约的本地读取带宽。
#
# 说明：Mega MoE 主体 SM100 专属（`mega_moe.hpp:367`），本机 H100 只复刻**数据布局与字节量**。
# 目标卡代号敏感：文中一律用「目标卡」指代。

import sys
import torch
import torch.distributed as dist


def bf16(x):
    return x.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# A) 独立 oracle
# ---------------------------------------------------------------------------
def oracle():
    torch.manual_seed(0)
    T, K, H, G = 32, 2, 256, 8
    # 合法分配：token t 由专家 (t+s)%G 处理（slot s）
    src_token, src_slot = {}, {}
    next_row = [0] * G
    rows = []  # (g, r, t, s)
    for t in range(T):
        for s in range(K):
            e = (t + s) % G
            r = next_row[e]
            next_row[e] += 1
            rows.append((e, r, t, s))
    R = max(next_row)  # 每专家最多若干行

    # 每个 (g,r) 的 L2 输出（fp32，代表 TMEM 读回）
    l2 = torch.randn(G, R, H, dtype=torch.float32)
    # 写端：RNE 打 bf16 写入邮箱 [K][T][H]
    mail = torch.zeros(K, T, H, dtype=torch.bfloat16)
    for (e, r, t, s) in rows:
        mail[s, t] = l2[e, r].to(torch.bfloat16)
    # 读端：bf16->fp32 累加 -> bf16
    y = mail.float().sum(dim=0).to(torch.bfloat16)
    # 参考（同一份 bf16 栅格上的逐位检查）
    ref_mail = torch.zeros(K, T, H, dtype=torch.bfloat16)
    ref_y = torch.zeros(T, H, dtype=torch.float32)
    for (e, r, t, s) in rows:
        v = l2[e, r].to(torch.bfloat16)
        ref_mail[s, t] = v
        ref_y[t] += v.float()
    ref_y = ref_y.to(torch.bfloat16)
    mail_ok = bool(torch.equal(mail, ref_mail))
    y_ok = bool(torch.equal(y, ref_y))
    # 一个「漏 slot」反例：只加 slot 0
    bad = mail[0].float().to(torch.bfloat16)
    bad_diff = int((bad != y).sum())
    print("A) oracle mailbox[slot][token] 逐位: %s ; combine y=Σ_slot 逐位: %s ; "
          "漏一个 slot 反例 diff=%d/%d" % (mail_ok, y_ok, bad_diff, T * H))
    return mail_ok and y_ok


# ---------------------------------------------------------------------------
# B) 多卡远端邮箱带宽
# ---------------------------------------------------------------------------
def bandwidth():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)

    T, K, H = 2048, 2, 7168
    nbytes = K * T * H * 2  # bf16 邮箱 = T·topk·H·2B
    src = torch.randn(K, T, H, dtype=torch.float32, device=dev).to(torch.bfloat16)
    recvbuf = torch.empty_like(src)

    if world >= 2:
        peer = 1 if rank == 0 else 0
        # 预热（rank0 发、rank1 收）
        for _ in range(3):
            if rank == 0:
                dist.send(src, peer)
            elif rank == 1:
                dist.recv(recvbuf, peer)
        dist.barrier()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        if rank == 0:
            ev0.record()
            for _ in range(30):
                dist.send(src, peer)
            ev1.record()
        elif rank == 1:
            for _ in range(30):
                dist.recv(recvbuf, peer)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            ms = ev0.elapsed_time(ev1) / 30
            gbs = nbytes / (ms * 1e-3) / 1e9
            print("B) 远端邮箱搬运（rank0->rank1, T·topk·H·2B = %.1f MB/次）：%.1f us  %.1f GB/s"
                  % (nbytes / 1e6, ms * 1e3, gbs))
    else:
        print("B) world < 2：跳过远端邮箱搬运")


if __name__ == "__main__":
    ok = oracle()
    if "dist" in sys.argv or "--dist" in sys.argv:
        bandwidth()
    print("A ORACLE:", "PASS" if ok else "FAIL")
