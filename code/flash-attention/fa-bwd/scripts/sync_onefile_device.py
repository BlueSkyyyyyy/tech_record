#!/usr/bin/env python3
"""把两文件版的 device 代码（.cuh）同步进单文件版（.cu），并校验二者逐字一致。

用法：scripts/sync_onefile_device.py <device.cuh> <onefile.cu> <marker>

<marker> 是 device 区起始行（两文件/单文件里以它开头的第一行）。device 区在单文件里
从 <marker> 延伸到 `struct NpyF32 {`（host 结构体）之前；脚本**只替换 device 区**，
不会碰宿主代码（历史上曾误删过 NpyF32，故显式以此为界）。
"""
import sys

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    cuh_path, one_path, marker = sys.argv[1], sys.argv[2], sys.argv[3]
    cuh = open(cuh_path).readlines()
    one = open(one_path).readlines()

    guard = next((l for l in cuh if l.startswith('#endif') and 'KERNELS_CUH_' in l), None)
    ci = next(i for i, l in enumerate(cuh) if l.startswith(marker))
    ce = next(i for i, l in enumerate(cuh) if guard and l.startswith(guard))
    dev = cuh[ci:ce]

    oi = next(i for i, l in enumerate(one) if l.startswith(marker))
    oe = next(i for i, l in enumerate(one) if l.startswith('struct NpyF32 {'))
    out = one[:oi] + dev + one[oe:]
    open(one_path, 'w').writelines(out)
    print(f"synced {len(dev)} device lines into {one_path}")

    # verify
    one2 = open(one_path).readlines()
    oi2 = next(i for i, l in enumerate(one2) if l.startswith(marker))
    same = one2[oi2:oi2 + len(dev)] == dev
    print("device region identical:", same)
    return 0 if same else 1

if __name__ == '__main__':
    sys.exit(main())
