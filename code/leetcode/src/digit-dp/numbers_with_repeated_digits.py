"""1012. 至少有 1 位重复的数字（Numbers With Repeated Digits）

题目：给定正整数 n，返回 [1, n] 内「至少有一位数字重复」的整数个数。

思路（正难则反：总数减去「各位互不相同」的个数）：
    直接数「有重复」要在数位 DP 里跟踪「是否已经出现重复」，状态多；反过来，
    [1, n] 一共 n 个数，减去「各位互不相同」的个数即可。

    count_unique_upto(n)：用数位 DP 统计 [0, n] 内各位互不相同的整数个数。
    沿 n 的十进制串从左到右填位，状态只需要一个 10 位的二进制掩码 mask 记录
    「哪些数字已经用过」，外加 tight（是否贴着上界）和 started（是否已经开始，
    用于跳过前导零）。

    每个位置能填的数受上界限制：tight 时最多到 s[pos]，否则 0..9。枚举这一位
    填 d 时，若 d 已在 mask 中则该分支作废；否则把它并进 mask 继续。走完所有位
    就是一个合法的数（包括「一直没开始」对应的 0）。

    最后 [1, n] 的不同数 = count_unique_upto(n) - 1（去掉 0），有重复 = n 减去它。

复杂度：时间 O(位数 * 2^10 * 2 * 2)，空间同阶；实际位数不超过 10。
"""

from functools import lru_cache


def _count_unique_upto(n):
    s = str(n)
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, mask, tight, started):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and d == limit
            if not started and d == 0:
                total += dfs(pos + 1, mask, ntight, False)
            elif mask & (1 << d):
                continue
            else:
                total += dfs(pos + 1, mask | (1 << d), ntight, True)
        return total

    return dfs(0, 0, True, False)


def num_dup_digits_at_most_n(n):
    return n - (_count_unique_upto(n) - 1)


if __name__ == "__main__":
    assert num_dup_digits_at_most_n(1) == 0
    assert num_dup_digits_at_most_n(20) == 1
    assert num_dup_digits_at_most_n(100) == 10
    assert num_dup_digits_at_most_n(1000) == 262
    print("num_dup_digits_at_most_n: all tests passed")
