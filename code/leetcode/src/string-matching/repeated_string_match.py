"""686. 重复叠加字符串匹配（Repeated String Match）

题目：给定字符串 a 和 b，返回 a 需要重复叠加的最小次数，使得 b 成为叠加结果的子串；
无法做到则返回 -1。

思路（先算够用的最少次数，再多试一次）：
    设 a 重复 t 次后长度达到 len(b)，即 t = ceil(len(b) / len(a))。答案只有两种可能：
    t 或 t + 1（b 的起点可能落在某一次重复的中间，所以长度刚好够时还差一点，需要多叠
    一个 a）。先试 t，再试 t + 1，哪个满足就返回哪个；都不满足返回 -1。
    为什么不试更多次：再加一个 a 只会让叠加串变长，若 b 在更长串里出现，它的某个窗口
    必然已经落在 t + 1 个 a 的范围内（b 长度固定）。

复杂度：时间 O((n + m)·m) 量级（子串查找），空间 O(n + m)；用 KMP 查找可到线性。
"""


def repeated_string_match(a, b):
    import math

    times = math.ceil(len(b) / len(a))
    for t in (times, times + 1):
        if b in a * t:
            return t
    return -1


if __name__ == "__main__":
    assert repeated_string_match("abcd", "cdabcdab") == 3
    assert repeated_string_match("a", "aa") == 2
    assert repeated_string_match("abc", "cabcabca") == 4
    assert repeated_string_match("abc", "wxyz") == -1
    assert repeated_string_match("aa", "a") == 1
    assert repeated_string_match("ab", "ba") == 2
    print("repeated_string_match: all tests passed")
