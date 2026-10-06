"""72. 编辑距离（Edit Distance）

题目：给两个单词 word1 和 word2，返回将 word1 转换成 word2 所使用的最少操作数。
允许三种操作：插入一个字符、删除一个字符、替换一个字符。

思路（二维 DP：把两个前缀之间的距离表出来）：
    又是「双序列」，但这次每一步都要付出代价，适合用 DP 记录前缀之间的距离。设

        dp[i][j] = 把 word1 的前 i 个字符变成 word2 的前 j 个字符的最少操作数。

    考虑两个前缀的最后一个字符 word1[i-1]、word2[j-1]：

    - 若相等：这一步不用操作，问题缩小到去掉这两个字符：

        dp[i][j] = dp[i-1][j-1]

    - 若不等：三种操作各自对应一个「去掉一个结尾」的子问题，取最小再 +1：
        - 删除 word1[i-1]：剩下 word1[0..i-2] 对 word2[0..j-1]，即 dp[i-1][j]；
        - 插入 word2[j-1]：等于 word2 少一个字符去对齐，即 dp[i][j-1]；
        - 替换 word1[i-1] 为 word2[j-1]：两个结尾一起消掉，即 dp[i-1][j-1]。

        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    初始化是「边界语义」：dp[i][0] = i（把前 i 个字符删空要删 i 次）、
    dp[0][j] = j（从空串插入 j 个字符要插 j 次）。答案 dp[m][n]。

    为什么插入对应 dp[i][j-1]：把 word1 变成 word2 时，「插入 word2[j-1]」意味着
    word2 还剩前 j-1 个字符要匹配，而 word1 一个都没消耗，于是是同一行往左挪一格。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


if __name__ == "__main__":
    assert min_distance("horse", "ros") == 3
    assert min_distance("intention", "execution") == 5
    assert min_distance("", "") == 0
    assert min_distance("abc", "") == 3
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "abc") == 0
    print("edit_distance: all tests passed")
