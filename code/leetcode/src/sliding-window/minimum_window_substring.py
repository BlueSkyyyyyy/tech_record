"""76. 最小覆盖子串（Minimum Window Substring）

题目：给定字符串 s 和 t，返回 s 中涵盖 t 所有字符（含重复次数）的最短子串；
      不存在则返回空串 ""。例如 s = "ADOBECODEBANC"、t = "ABC"，答案是 "BANC"。

思路：变长滑动窗口（计数型）。用 need 记录 t 中每个字符需要的次数，用 window 记录当前窗口
      里各字符的个数。关键是用 formed 表示「已经满足数量要求的字符种类数」：
      当 window[ch] == need[ch] 时该字符达标，formed 加一；当 formed == len(need) 时
      当前窗口已完整覆盖 t。

      此时窗口合法，题目要最短，于是尝试收缩左端：先记录当前长度更新答案，再吐出 s[left]，
      并检查吐出后该字符是否不再达标（window < need）来调整 formed，直到窗口不再合法，
      再继续扩右端。

      为什么用「达标字符种类数」而不是每次都逐字符比较两本字典：后者是 O(字符集) 的开销，
      而 formed 只在字符数量恰好跨过 need 阈值时增减，整体仍是 O(n)。

      为什么正确：右端只进不退；出现的第一个合法窗口可能很长，但随后每收缩一格都还在
      「尽量短」的方向上探索，直到不合法才停止，所以覆盖了所有可能的最短窗口。

复杂度：时间 O(|s| + |t|)，空间 O(字符集大小)。
"""


def min_window(s, t):
    if not s or not t:
        return ""
    need = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    window = {}
    formed = 0
    left = 0
    best_len = float("inf")
    best_left = 0
    for right, ch in enumerate(s):
        window[ch] = window.get(ch, 0) + 1
        if ch in need and window[ch] == need[ch]:
            formed += 1
        while formed == len(need):
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_left = left
            left_ch = s[left]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                formed -= 1
            left += 1
    return "" if best_len == float("inf") else s[best_left:best_left + best_len]


if __name__ == "__main__":
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""
    assert min_window("aa", "aa") == "aa"
    assert min_window("cabwefgewcwaefgcf", "cae") == "cwae"
    assert min_window("abc", "") == ""
    print("minimum_window_substring: all tests passed")
