"""2801. 统计范围内的步进数字数目（Count Stepping Numbers in Range）

题目：步进数字指相邻两位数字都正好相差 1 的数字（如 10、12、321）。给定两个
字符串 low、high（长度可达 100），返回 [low, high] 内步进数字的个数，对
1e9+7 取模。

思路（十进制数位 DP + 大数字符串前缀相减）：
    答案 = count_upto(high) - count_upto(low-1)，其中 count_upto(x) 统计 [1, x]
    的步进数字。因为 low / high 是大数字符串，减法要对字符串做「减一」。

    count_upto 的 DP 状态：pos（当前位）、prev（上一位数字，-1 表示还没开始）、
    started（是否已填入首个非零数字，用来排除前导零）、tight（是否贴上限）。
      - 还没开始且填 0：保持未开始；
      - 已开始：这一位 d 必须满足 abs(d - prev) == 1，否则跳过；
      - 走到串尾：只有 started 才是合法数字，未开始代表全 0，不计入。
    步进数字按定义是正整数，所以 0 不算。

    由于长度可达 100，状态数约 100 * 11 * 2 * 2，DP 完全可行；每一步取模。

复杂度：时间 O(位数 * 10)，空间 O(位数 * 10)。
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


def _count_upto(s):
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, prev, started, tight):
        if pos == length:
            return 1 if started else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and d == limit
            if not started and d == 0:
                total += dfs(pos + 1, -1, False, ntight)
            elif started and abs(d - prev) != 1:
                continue
            else:
                total += dfs(pos + 1, d, True, ntight)
        return total % MOD

    return dfs(0, -1, False, True)


def count_stepping_numbers(low, high):
    return (_count_upto(high) - _count_upto(_decrement(low))) % MOD


if __name__ == "__main__":
    assert count_stepping_numbers("1", "11") == 10
    assert count_stepping_numbers("90", "101") == 2
    assert count_stepping_numbers("1", "10") == 10
    assert count_stepping_numbers("10", "10") == 1
    assert count_stepping_numbers("1", "100000000000000000000000000000") == 486147231
    print("count_stepping_numbers: all tests passed")
