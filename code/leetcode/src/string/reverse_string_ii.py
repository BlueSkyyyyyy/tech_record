"""541. 反转字符串 II（Reverse String II）

题目：给定字符串 s 和整数 k，从开头起每 2k 个字符为一组，反转每组的前 k 个字符：
    - 若剩余字符不足 k 个，就把剩下的全部反转；
    - 若剩余字符在 [k, 2k) 之间，只反转前 k 个，其余保持原样。
返回处理后的字符串。

思路（按块跳跃 + 依次反转）：
    不要一个字符一个字符地数，而是以 2k 为步长直接跳到每组的开头 i。对每个 i，这一组
    需要反转的区间是 [i, min(i + k, n) - 1]：当 i + k <= n 时正好是 k 个字符；当剩余不
    足 k 个时右端被 n - 1 截住，等于「把剩下的全部反转」。区间内部仍然是 344 的对撞
    双指针交换。

    为什么用 min：它一次性覆盖了题目的两种边界情况，不需要单独写 if 分支。左端固定为 i
    （不会越界，因为 i 是步长 2k 的起点且 i < n），右端用 min 封顶。

复杂度：时间 O(n)（每个字符最多被交换一次），空间 O(n)（Python 需要把不可变字符串转成
字符列表）或 O(1)（C++ 直接原地改）。
"""


def reverse_str(s, k):
    chars = list(s)
    n = len(chars)
    for i in range(0, n, 2 * k):
        left = i
        right = min(i + k, n) - 1
        while left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
    return "".join(chars)


if __name__ == "__main__":
    assert reverse_str("abcdefg", 2) == "bacdfeg"
    assert reverse_str("abcd", 2) == "bacd"
    assert reverse_str("abcdefg", 8) == "gfedcba"
    assert reverse_str("a", 2) == "a"
    assert reverse_str("", 3) == ""
    assert reverse_str("abcd", 4) == "dcba"
    assert reverse_str("abcdef", 3) == "cbadef"
    assert reverse_str("ab", 1) == "ab"
    print("reverse_string_ii: all tests passed")
