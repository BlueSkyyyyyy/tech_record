"""647. 回文子串（Palindromic Substrings）

题目：给定字符串 s，统计它有多少个不同的回文子串（按位置计数，相同内容出现在不同位置
算多个）。

思路（Manacher 一箭双雕）：
    上一题用 Manacher 求出了每个中心的最长回文半径，本题只需把「每个中心贡献多少个回文」
    加起来。在插了 '#' 的串 t 里，中心分两类：
      - 偶数下标是原串字符，对应奇数长度回文。半径 p[i] 中每 2 步多一个回文（长度
        1、3、5…），共 p[i] // 2 + 1 个。
      - 奇数下标是插入的 '#'，对应偶数长度回文。半径 p[i] 中每 2 步多一个回文（长度
        2、4、6…），共 (p[i] + 1) // 2 个。
    对所有中心求和即答案。这样把「每个中心往外扩」的 O(n²) 压成 O(n)。

复杂度：时间 O(n)，空间 O(n)。
"""


def count_substrings(s):
    if not s:
        return 0
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = right = 0
    for i in range(1, n - 1):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    total = 0
    for i in range(1, n - 1):
        if i % 2 == 0:
            total += p[i] // 2 + 1  # 原串字符为中心：奇数长度
        else:
            total += (p[i] + 1) // 2  # '#' 为中心：偶数长度
    return total


if __name__ == "__main__":
    assert count_substrings("abc") == 3
    assert count_substrings("aaa") == 6
    assert count_substrings("aba") == 4
    assert count_substrings("a") == 1
    assert count_substrings("") == 0
    assert count_substrings("aaaa") == 10
    print("count_substrings: all tests passed")
