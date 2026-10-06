"""902. 最大为 N 的数字组合（Numbers At Most N Given Digit Set）

题目：给定一个升序、不含重复元素且只含 '1'..'9' 的数字字符数组 digits，以及
整数 n，返回用 digits 里的数字（每个可重复使用）能拼出的、数值 <= n 的正整数
个数。

思路（按位数分两段，巧妙地绕开前导零）：
    因为 digits 里没有 0，用它们拼出的任何数都不会有前导零，也不会出现 0。

    设 n 的十进制长度为 L：
      1. 位数比 L 少的数一定 < n：长度为 len 时有 |digits|^len 种，
         长度 1..L-1 全部加起来。
      2. 位数恰好等于 L 的数需要逐位和 n 比较：用数位 DP 贴着上界枚举。
         在位置 pos，若 tight，这一位的上限是 s[pos]，否则是 9；从 digits 里
         依次取不超过上限的数字。由于 digits 有序，一旦某个数字超上限，后面的
         更大，直接 break。tight 时若选了等于上限的数字，下一位仍 tight。

    最后把两段相加。这里不需要 started/前导零状态，正是因为数字集排除了 0。

复杂度：时间 O(位数 * |digits|)，空间 O(位数 * |digits|)。
"""

from functools import lru_cache


def at_most_n_given_digit_set(digits, n):
    nums = [int(c) for c in digits]
    s = str(n)
    length = len(s)

    total = 0
    for size in range(1, length):
        total += len(nums) ** size

    @lru_cache(maxsize=None)
    def dfs(pos, tight):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 9
        count = 0
        for d in nums:
            if d > limit:
                break
            count += dfs(pos + 1, tight and d == limit)
        return count

    return total + dfs(0, True)


if __name__ == "__main__":
    assert at_most_n_given_digit_set(["1", "3", "5", "7"], 100) == 20
    assert at_most_n_given_digit_set(["1", "3", "5", "7"], 1) == 1
    assert at_most_n_given_digit_set(["1", "4", "9"], 1000000000) == 29523
    assert at_most_n_given_digit_set(["7"], 8) == 1
    print("at_most_n_given_digit_set: all tests passed")
