"""600. 不含连续 1 的非负整数（Non-negative Integers without Consecutive Ones）

题目：给定正整数 n，返回 [0, n] 内二进制表示不含连续两个 1 的整数个数。

思路（二进制数位 DP，状态只记「上一位是不是 1」）：
    把 n 看成二进制串，从高位往低位填 bit。每个位置能不能填 1，只取决于上一位
    是不是 1：如果上一位是 1，这一位就只能填 0。于是状态只需 prev_one 一个布尔量，
    外加 tight（贴着上界）和位置 pos。

    决策：
      - 枚举这一位 b 取 0..limit（limit = s[pos] 或 1）；
      - 若 prev_one 且 b == 1，跳过；
      - 否则递归，下一状态的 prev_one = (b == 1)。

    走到串尾返回 1（包括全 0 这个数）。

    为什么不用像十进制那样额外处理前导零？因为二进制下「前导零」不影响是否出现
    「连续 1」，直接从头填 0/1 即可，0 本身天然合法。

复杂度：时间 O(位数)，空间 O(位数)。
"""

from functools import lru_cache


def find_integers(n):
    s = bin(n)[2:]
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, prev_one, tight):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 1
        total = 0
        for b in range(limit + 1):
            if prev_one and b == 1:
                continue
            total += dfs(pos + 1, b == 1, tight and b == limit)
        return total

    return dfs(0, False, True)


if __name__ == "__main__":
    assert find_integers(0) == 1
    assert find_integers(1) == 2
    assert find_integers(2) == 3
    assert find_integers(5) == 5
    assert find_integers(10) == 8
    print("find_integers: all tests passed")
