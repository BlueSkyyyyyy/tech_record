"""1239. 串联字符串的最大长度（Maximum Length of a Concatenated String
with Unique Characters）

题目：给定字符串数组 arr，从中选出一个子序列拼起来，要求拼出的字符串里没有重复
字符，求能拼出的最大长度。

思路（掩码校验冲突 + 逐个"选/不选"）：
    题目的硬约束是"所有字符互不相同"。用 26 位掩码表示一个字符串用过哪些字母，两个
    字符串能拼在一起，当且仅当它们自身没有重复字符（内部合法），且两个掩码的交集为
    空（`m1 & m2 == 0`）。

    于是问题变成"从若干互不冲突的物品里选一组，使长度和最大"——一个 0/1 背包式枚举。
    用一个字典 dp 记录"当前用掉的字母集合 -> 已达成的最大长度"，每加入一个合法字符串
    s，就把它接到所有和它不相交的已有集合后面。

    `dp` 的键就是状态压缩后的集合；因为字母只有 26 种，掩码可以保证不冲突。

复杂度：时间 O(n * 2^n)（n = len(arr)，最多 n 个不同的可达集合），空间 O(2^n)。
"""


def max_length(arr):
    # 预先把每个字符串压成 (掩码, 长度)；自带重复字符的直接丢弃。
    items = []
    for s in arr:
        mask = 0
        ok = True
        for ch in s:
            bit = 1 << (ord(ch) - 97)
            if mask & bit:
                ok = False
                break
            mask |= bit
        if ok:
            items.append((mask, len(s)))

    dp = {0: 0}
    for mask, length in items:
        for cur, total in list(dp.items()):
            if cur & mask == 0:
                merged = cur | mask
                cand = total + length
                if dp.get(merged, -1) < cand:
                    dp[merged] = cand
    return max(dp.values())


if __name__ == "__main__":
    assert max_length(["un", "iq", "ue"]) == 4
    assert max_length(["cha", "r", "act", "ers"]) == 6
    assert max_length(["abcdefghijklmnopqrstuvwxyz"]) == 26
    assert max_length(["aa", "bb"]) == 0
    assert max_length(["a", "b", "c"]) == 3
    print("max_length: all tests passed")
