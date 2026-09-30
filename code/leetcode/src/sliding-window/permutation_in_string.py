"""567. 字符串的排列（Permutation in String）

题目：给定字符串 s1 和 s2，判断 s2 中是否包含 s1 的某个排列（即长度相同、字母组成相同的
      连续子串）。例如 s1 = "ab"、s2 = "eidbaooo"，答案是 true（"ba" 是一个排列）。

思路：和 438 是同一个模型——「定长窗口 + 字母计数」。s1 的排列就是与 s1 字母组成一致的
      连续子串，窗口长度固定为 len(s1)。用两本长度 26 的计数表分别记录 s1 的需要和当前窗口，
      窗口右移时进一个字符、出一个字符，若两表相等就说明找到排列，直接返回 true。

      与 438 的唯一区别：438 要把所有匹配位置收集起来，本题找到一个就能提前返回，因此
      把「收集下标」换成「命中即真」，其余逻辑逐字相同。

复杂度：时间 O(|s2|)，空间 O(1)（字母表固定 26）。
"""


def check_inclusion(s1, s2):
    n1, n2 = len(s1), len(s2)
    if n1 > n2:
        return False
    need = [0] * 26
    window = [0] * 26
    for ch in s1:
        need[ord(ch) - ord("a")] += 1
    for i, ch in enumerate(s2):
        window[ord(ch) - ord("a")] += 1
        if i >= n1:
            window[ord(s2[i - n1]) - ord("a")] -= 1
        if i >= n1 - 1 and window == need:
            return True
    return False


if __name__ == "__main__":
    assert check_inclusion("ab", "eidbaooo") is True
    assert check_inclusion("ab", "eidboaoo") is False
    assert check_inclusion("abc", "bbbca") is True
    assert check_inclusion("hello", "ooolleoooleh") is False
    assert check_inclusion("a", "a") is True
    print("permutation_in_string: all tests passed")
