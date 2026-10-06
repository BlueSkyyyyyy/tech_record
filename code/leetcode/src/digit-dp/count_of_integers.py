"""2719. 统计整数数目（Count of Integers）

题目：给定两个数字字符串 num1、num2 和两个整数 min_sum、max_sum，统计
[num1, num2] 内「各位数字之和落在 [min_sum, max_sum]」的整数个数，对 1e9+7 取模。

思路（数位 DP + 窗口前缀相减）：
    答案 = count_upto(num2) - count_upto(num1 - 1)，其中 count_upto(s) 统计
    [1, s] 内数位和合格的数。因为 num 是大数字符串，减一要按字符串处理。

    数位 DP 的状态：
      - pos：当前处理到第几位；
      - sum_so_far：已经填的数字之和（可用它剪枝：超过 max_sum 直接不递归）；
      - tight：是否贴着上界；
      - started：是否已开始填（跳过前导零，保证 0 不被当成合法数）。

    每步枚举 0..limit 的数字 d：
      - 未开始且 d == 0：保持未开始；
      - 否则若 sum_so_far + d > max_sum 就跳过；再把 d 记入和继续。
    走到串尾：started 且和在 [min_sum, max_sum] 内才算 1。

    因为 min_sum >= 1，0 这种「数位和为 0」的数本来就不合格，所以无需特殊处理 0。

复杂度：时间 O(位数 * max_sum * 2 * 2)，空间同阶。
"""

from functools import lru_cache

MOD = 10 ** 9 + 7


def _decrement(s):
    arr = list(s)
    i = len(arr) - 1
    while arr[i] == "0":
        arr[i] = "9"
        i -= 1
    arr[i] = chr(ord(arr[i]) - 1)
    trimmed = "".join(arr).lstrip("0")
    return trimmed if trimmed else "0"


def count_of_integers(num1, num2, min_sum, max_sum):
    def count_upto(s):
        length = len(s)

        @lru_cache(maxsize=None)
        def dfs(pos, sum_so_far, tight, started):
            if pos == length:
                return 1 if started and min_sum <= sum_so_far <= max_sum else 0
            limit = int(s[pos]) if tight else 9
            total = 0
            for d in range(limit + 1):
                ntight = tight and d == limit
                if not started and d == 0:
                    total += dfs(pos + 1, sum_so_far, ntight, False)
                else:
                    if sum_so_far + d > max_sum:
                        continue
                    total += dfs(pos + 1, sum_so_far + d, ntight, True)
            return total % MOD

        return dfs(0, 0, True, False)

    return (count_upto(num2) - count_upto(_decrement(num1))) % MOD


if __name__ == "__main__":
    assert count_of_integers("1", "12", 1, 8) == 11
    assert count_of_integers("1", "5", 1, 5) == 5
    assert count_of_integers("1", "2026", 2, 5) == 95
    print("count_of_integers: all tests passed")
