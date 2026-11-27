"""Build results_moe_ep.md from the template + the benchmark CSVs.

    python3 build_results_doc.py
"""

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "results_moe_ep.template.md")
OUT = os.path.join(HERE, "results_moe_ep.md")


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def rows_of(note):
    m = re.search(r"(?:rows|expanded_rows|recv_rows)=(\d+)", note or "")
    return int(m.group(1)) if m else None


def get(rows, op, T, H):
    for r in rows:
        if r["op"] == op and int(r["tokens_per_rank"]) == T and int(r["hidden"]) == H:
            return r
    return None


def k(r, nd=1):
    return f"{float(r['kernel_us']):.{nd}f}" if r else "-"


def w(r, nd=1):
    return f"{float(r['wall_med_us']):.{nd}f}" if r else "-"


def configs(*row_sets):
    seen = []
    for rows in row_sets:
        for r in rows:
            key = (int(r["tokens_per_rank"]), int(r["hidden"]))
            if key not in seen:
                seen.append(key)
    return sorted(seen, key=lambda x: (-x[0], x[1]))


def tflops(flop, us):
    return flop / (us * 1e6) if us else 0.0


def table_gemm(uni, te):
    out = ["### 3.1 group GEMM（FC1 = X@W1ᵀ: N=2I, K=H；FC2 = A@W2ᵀ: N=H, K=I）", "",
           "| 配置 | 行数 uniEP / TE | FC1 uniEP (µs) | FC1 TE (µs) | FC2 uniEP (µs) | FC2 TE (µs) |",
           "|---|---|---:|---:|---:|---:|"]
    for T, H in configs(uni, te):
        u1, t1 = get(uni, "uniep_group_gemm_fc1", T, H), get(te, "te_grouped_gemm_fc1", T, H)
        u2, t2 = get(uni, "uniep_group_gemm_fc2", T, H), get(te, "te_grouped_gemm_fc2", T, H)
        ru = rows_of(u1["note"]) if u1 else None
        rt = rows_of(t1["note"]) if t1 else None
        out.append(f"| T{T}/rank H{H} | {ru} / {rt} | {k(u1)} | {k(t1)} | {k(u2)} | {k(t2)} |")
    return "\n".join(out)


def table_tflops(uni, te):
    out = []
    for T, H in configs(uni, te):
        I = H
        cells = []
        for op_u, op_t, nfac in (("uniep_group_gemm_fc1", "te_grouped_gemm_fc1", 2),
                                 ("uniep_group_gemm_fc2", "te_grouped_gemm_fc2", 1)):
            u, t = get(uni, op_u, T, H), get(te, op_t, T, H)
            vu = vt = "-"
            if u:
                r = rows_of(u["note"])
                vu = f"{tflops(2 * r * nfac * I * H, float(u['kernel_us'])):.0f}"
            if t:
                r = rows_of(t["note"])
                vt = f"{tflops(2 * r * nfac * I * H, float(t['kernel_us'])):.0f}"
            cells += [vu, vt]
        out.append(f"| T{T}/rank H{H} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    return "\n".join(out)


def table_comm(uni, dee):
    out = ["### 3.2 dispatch / combine（通信 op：上=wall 中位 µs，下=CUPTI kernel µs）", "",
           "| 配置 | dispatch uniEP | dispatch DeepEP(expanded) | dispatch DeepEP(expanded, async) | combine uniEP | combine DeepEP(expanded) |",
           "|---|---:|---:|---:|---:|---:|"]
    for T, H in configs(uni, dee):
        ud, udd = get(uni, "uniep_dispatch", T, H), get(uni, "uniep_combine", T, H)
        dd, dda = get(dee, "deepep_dispatch_expanded", T, H), get(dee, "deepep_dispatch_expanded_async", T, H)
        dc = get(dee, "deepep_combine_expanded", T, H)
        cells_w = [w(ud), w(dd), w(dda), w(udd), w(dc)]
        cells_k = [k(ud), k(dd), k(dda), k(udd), k(dc)]
        out.append(f"| T{T}/rank H{H} | " + " | ".join(
            f"{a}<br>{b}" for a, b in zip(cells_w, cells_k)) + " |")
    return "\n".join(out)


def _pipeline_sum(dee, te, uni, T, H, gemm_src):
    """Sum of measured components = what an unfused stack costs end to end."""
    parts = [get(dee, "deepep_dispatch_expanded_async", T, H),
             get(gemm_src, "te_grouped_gemm_fc1" if gemm_src is te else "uniep_group_gemm_fc1", T, H),
             get(uni, "uniep_swiglu", T, H),
             get(gemm_src, "te_grouped_gemm_fc2" if gemm_src is te else "uniep_group_gemm_fc2", T, H),
             get(dee, "deepep_combine_expanded", T, H)]
    if not all(parts):
        return None
    return sum(float(r["kernel_us"]) for r in parts)


def table_e2e(uni, dee, te):
    out = ["### 3.3 端到端一层 MoE（单位 µs；上=wall 中位，下=CUPTI kernel）", "",
           "> 「估算」两列是**逐项实测值相加**（DeepEP 通信 + 对应 GEMM + SwiGLU），不是一次调用实测："
           "本机的 DeepEP expand 布局无法可靠重建 `m_splits`（见 §8），故不报串行 pipeline 的整段实测值。", "",
           "| 配置 | uniEP 融合 (MK1+SwiGLU+MK2) | uniEP 非融合 (5 次 launch) | 估算: DeepEP 通信 + TE GEMM | 估算: DeepEP 通信 + uniEP GEMM | DeepEP 纯通信 (expanded) |",
           "|---|---:|---:|---:|---:|---:|"]
    for T, H in configs(uni, dee, te):
        f = get(uni, "uniep_e2e_fused", T, H)
        s = get(uni, "uniep_e2e_sequential", T, H)
        dc = get(dee, "deepep_e2e_expanded", T, H)
        hyb_te = _pipeline_sum(dee, te, uni, T, H, te)
        hyb_uni = _pipeline_sum(dee, te, uni, T, H, uni)
        row = f"| T{T}/rank H{H} | "
        for r in (f, s):
            row += f"{w(r)}<br>{k(r)} | "
        row += f"{hyb_te:.0f} | " if hyb_te else "- | "
        row += f"{hyb_uni:.0f} | " if hyb_uni else "- | "
        row += f"{w(dc)}<br>{k(dc)} |"
        out.append(row)
    return "\n".join(out)


def table_mk(uni):
    out = []
    for T, H in configs(uni):
        d, f1 = get(uni, "uniep_dispatch", T, H), get(uni, "uniep_group_gemm_fc1", T, H)
        mk1 = get(uni, "uniep_dispatch_group_gemm", T, H)
        f2, c = get(uni, "uniep_group_gemm_fc2", T, H), get(uni, "uniep_combine", T, H)
        mk2 = get(uni, "uniep_group_gemm_combine", T, H)
        if not all((d, f1, mk1, f2, c, mk2)):
            continue
        s1 = float(d["kernel_us"]) + float(f1["kernel_us"])
        s2 = float(f2["kernel_us"]) + float(c["kernel_us"])
        g1 = 100 * (s1 - float(mk1["kernel_us"])) / s1
        g2 = 100 * (s2 - float(mk2["kernel_us"])) / s2
        out.append(f"| T{T}/rank H{H} | {k(d)} | {k(f1)} | {s1:.0f} | {k(mk1)} | "
                   f"{'快' if g1 > 0 else '慢'}{abs(g1):.0f}% | {k(f2)} | {k(c)} | {s2:.0f} | "
                   f"{k(mk2)} | {'快' if g2 > 0 else '慢'}{abs(g2):.0f}% |")
    return "\n".join(out)


def table_swiglu(uni):
    out = []
    for T, H in configs(uni):
        s, seq = get(uni, "uniep_swiglu", T, H), get(uni, "uniep_e2e_sequential", T, H)
        if not s or not seq:
            continue
        out.append(f"| T{T}/rank H{H} | {k(s)} µs | {100 * float(s['kernel_us']) / float(seq['kernel_us']):.0f}% |")
    return "\n".join(out)


def table_raw(sets):
    out = ["### 3.4 全量明细（原始口径：`kernel耗时` = CUPTI 纯 device kernel；`wall中位` / `p90` 见 §0 提示 2）", "",
           "| op | dtype | shape | device | kernel耗时(µs) | wall中位(µs) | wall p90(µs) | 备注 |",
           "|---|---|---|---|---:|---:|---:|---|"]
    for name, (rows, order) in sets.items():
        if not rows:
            continue
        out.append(f"| **{name}** | | | | | | | |")
        for r in sorted(rows, key=lambda r: (order.get(r["op"], 99), -int(r["tokens_per_rank"]),
                                             int(r["hidden"]))):
            out.append(f"| {r['op']} | {r['dtype']} | {r['shape']} | {r['device']} | "
                       f"{k(r)} | {w(r)} | {float(r['wall_p90_us']):.1f} | {r['note']} |")
    return "\n".join(out)


def main():
    uni = load("moe_ep_uniep.csv")
    dee = load("moe_ep_deepep.csv")
    te = load("moe_ep_tegmm.csv")

    order_uni = {op: i for i, op in enumerate([
        "uniep_layout", "uniep_dispatch", "uniep_group_gemm_fc1", "uniep_swiglu",
        "uniep_group_gemm_fc2", "uniep_combine", "uniep_dispatch_group_gemm",
        "uniep_group_gemm_combine", "uniep_e2e_fused", "uniep_e2e_sequential"])}
    order_dee = {op: i for i, op in enumerate([
        "deepep_dispatch", "deepep_dispatch_expanded", "deepep_dispatch_async",
        "deepep_dispatch_expanded_async", "deepep_combine", "deepep_combine_expanded",
        "deepep_e2e", "deepep_e2e_expanded"])}
    order_te = {op: i for i, op in enumerate([
        "te_grouped_gemm_fc1", "te_grouped_gemm_fc2", "te_grouped_gemm_layer",
        "te_grouped_gemm_fc1_fwd_bwd", "te_grouped_gemm_fc2_fwd_bwd"])}

    doc = open(TEMPLATE).read()
    doc = doc.replace("{{TABLE_GEMM}}", table_gemm(uni, te))
    doc = doc.replace("{{TABLE_TFLOPS}}", table_tflops(uni, te))
    doc = doc.replace("{{TABLE_COMM}}", table_comm(uni, dee))
    doc = doc.replace("{{TABLE_E2E}}", table_e2e(uni, dee, te))
    doc = doc.replace("{{TABLE_MK}}", table_mk(uni))
    doc = doc.replace("{{TABLE_SWIGLU}}", table_swiglu(uni))
    doc = doc.replace("{{TABLE_RAW}}", table_raw({
        "uniEP": (uni, order_uni), "DeepEP": (dee, order_dee),
        "TE group-gemm": (te, order_te)}))
    left = re.findall(r"\{\{[A-Z_]+\}\}", doc)
    if left:
        print(f"WARNING: unfilled placeholders {set(left)}", file=sys.stderr)
    open(OUT, "w").write(doc)
    print(f"wrote {OUT} ({len(doc.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
