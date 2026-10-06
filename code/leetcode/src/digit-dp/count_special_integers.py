"""2376. 统计特殊整数（Count Special Integers）

题目：如果一个正整数的每一位数字都互不相同，就称它是「特殊整数」。给定 n，
返回 [1, n] 内特殊整数的个数。

思路（数位 DP 跟踪「已用数字的掩码」）：
    与 1012 是同一套逻辑：特殊整数就是「各位互不相同」的正整数。沿 n 的十进制
    串逐位填数，用 10 位二进制掩码 mask 记录用过的数字，tight 控制是否贴着上界，
    started 跳过前导零。

    填每一位时枚举 0..limit：
      - 还没开始且填 0：仍算「没开始」，掩码不变（这就是前导零，不算用掉数字 0）；
      - 否则该数字若已在 mask 中，分支作废；
      - 否则并入掩码继续。
    走到串尾返回 1，代表构成一个数（含全程未开始的 0）。

    最终 [1, n] 的个数 = [0, n] 的个数减 1（去掉 0）。

    与 357 的区别：357 的 n 是「位数上限」，答案有闭式；本题 n 是一个具体上界，
    必须带着 tight 逐位受限地数，所以用记忆化搜索。

复杂度：时间 O(位数 * 2^10)，空间同阶。
"""

from functools import lru_cache


def count_special_integers(n):
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

    return dfs(0, 0, True, False) - 1


if __name__ == "__main__":
    assert count_special_integers(5) == 5
    assert count_special_integers(20) == 19
    assert count_special_integers(135) == 110
    print("count_special_integers: all tests passed")
