"""1044. 最长重复子串（Longest Duplicate Substring）

题目：给定字符串 s，找出其中出现次数不少于两次的最长子串（出现可以有重叠）。返回任意
一个；不存在则返回空串。

思路（二分答案 + 滚动哈希去重）：
    直接枚举所有子串太慢。先用「二分答案」猜答案长度 L：如果存在长度为 L 的重复子串，
    那么一定也存在更短的重复子串（取它的前缀即可），答案长度关于 L 单调。于是二分 L，
    判定「是否存在长度为 L 的重复子串」。

    判定用滚动哈希：把窗口内字符按多项式编码成一个数
        h = s[i]·base^(L-1) + s[i+1]·base^(L-2) + ... + s[i+L-1]
    窗口右移一位时，不必重算，只需「减去最高位、整体乘 base、加上新字符」：
        h = (h - s[i-1]·base^(L-1)) · base + s[i+L-1]
    用哈希值当集合的键，若某个哈希已出现，再对两段真实子串比对一次以排除哈希碰撞。

复杂度：时间 O(n log n)（二分 log n 轮，每轮 O(n)），空间 O(n)。
"""


def longest_dup_substring(s):
    n = len(s)
    if n < 2:
        return ""
    base, mod = 131, (1 << 61) - 1

    def check(length):
        # 返回一个长度为 length 的重复子串；不存在返回 None
        if length == 0:
            return ""
        power = pow(base, length - 1, mod)
        h = 0
        for i in range(length):
            h = (h * base + ord(s[i])) % mod
        seen = {h: 0}
        for i in range(1, n - length + 1):
            h = ((h - ord(s[i - 1]) * power) % mod * base + ord(s[i + length - 1])) % mod
            if h in seen and s[seen[h] : seen[h] + length] == s[i : i + length]:
                return s[i : i + length]
            seen[h] = i
        return None

    lo, hi = 0, n - 1
    ans = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        found = check(mid)
        if found is not None:
            ans = found
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


if __name__ == "__main__":
    assert longest_dup_substring("banana") == "ana"
    assert longest_dup_substring("abcd") == ""
    assert longest_dup_substring("aa") == "a"
    assert longest_dup_substring("aaaa") == "aaa"
    assert longest_dup_substring("a") == ""
    assert longest_dup_substring("abcabcabcd") == "abcabc"
    print("longest_dup_substring: all tests passed")
