"""438. 找到字符串中所有字母异位词（Find All Anagrams in a String）

题目：给定字符串 s 和 p，找出 s 中所有是 p 的字母异位词的子串的起始下标。例如
      s = "cbaebabacd"、p = "abc"，答案是 [0, 6]（"cba" 与 "bac"）。

思路：定长滑动窗口。字母异位词的长度一定等于 p 的长度，所以窗口大小固定为 len(p)，
      只需判断每个长度为 len(p) 的窗口里各字母计数是否与 p 相同。

      用长度 26 的计数数组表示字母表：右端每进来一个字符就加一，窗口超过 len(p) 时
      把最左边离开的字符减一，然后比较两个计数数组是否相等。相等就记下窗口左端下标。

      为什么用定长窗而不是变长窗：判定条件「长度必须等于 len(p)」本身就是长度的约束，
      窗口右端每走一步，左端被动跟一步，不存在「收缩到何时」的判断，正是定长窗的标志。

      为什么比较整张表是对的：异位词只关心每个字母出现次数，不关心顺序；两个计数数组
      完全相同，就说明窗口里字符组成与 p 一致。

复杂度：时间 O(|s|)，空间 O(1)（字母表固定 26）。
"""


def find_anagrams(s, p):
    res = []
    n, m = len(s), len(p)
    if n < m:
        return res
    need = [0] * 26
    window = [0] * 26
    for ch in p:
        need[ord(ch) - ord("a")] += 1
    for i, ch in enumerate(s):
        window[ord(ch) - ord("a")] += 1
        if i >= m:
            window[ord(s[i - m]) - ord("a")] -= 1
        if i >= m - 1 and window == need:
            res.append(i - m + 1)
    return res


if __name__ == "__main__":
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert find_anagrams("abab", "ab") == [0, 1, 2]
    assert find_anagrams("a", "ab") == []
    assert find_anagrams("baa", "aa") == [1]
    assert find_anagrams("aaaa", "aa") == [0, 1, 2]
    print("find_all_anagrams: all tests passed")
