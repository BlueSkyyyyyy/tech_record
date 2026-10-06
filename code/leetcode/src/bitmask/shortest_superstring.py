"""943. 最短超级串（Find the Shortest Superstring）

题目：给定字符串数组 words，返回一个最短的字符串，使 words 里每个词都是它的子串。

思路（先去掉冗余词，再对"排列"做 TSP 式 DP）：
    第一步做两件清理：去掉重复词；去掉那些本身是别的词子串的词——它们不需要单独拼接，
    只要那个更长的词被包含了就自动被包含。

    剩下的词两两之间会有"咬合"：把 j 接在 i 后面时，可以让 i 的一段后缀和 j 的一段
    前缀重合。预先算出 overlap[i][j] = 最长的重合长度。

    接下来要在所有排列里找总长度最小的拼接。这等价于旅行商问题（TSP）：把每个词当成
    一个城市，从 i 走到 j 的"路程"是 len(j) - overlap[i][j]。用状态压缩 DP：
    `dp[mask][last]` = 已经用过 mask 里的词、并且以 last 结尾时，最短超级串的长度。
    转移就是"往末尾再接一个没用过的词"。最后取 `dp[全集][*]` 的最小值，再顺着 parent
    数组回推出排列，拼成答案。

复杂度：时间 O(2^k * k^2)（k 是去重去子串后的词数），空间 O(2^k * k)。
"""


def shortest_superstring(words):
    # 去重 + 去掉是别人子串的词
    unique = []
    for w in words:
        if w not in unique:
            unique.append(w)
    words = [w for w in unique
             if not any(w != other and w in other for other in unique)]
    k = len(words)
    if k == 0:
        return ""

    overlap = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            a, b = words[i], words[j]
            for length in range(min(len(a), len(b)), 0, -1):
                if a[-length:] == b[:length]:
                    overlap[i][j] = length
                    break

    full = (1 << k) - 1
    INF = float("inf")
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]
    for i in range(k):
        dp[1 << i][i] = len(words[i])

    for mask in range(1 << k):
        for last in range(k):
            if dp[mask][last] == INF:
                continue
            for nxt in range(k):
                if mask >> nxt & 1:
                    continue
                nm = mask | (1 << nxt)
                cand = dp[mask][last] + len(words[nxt]) - overlap[last][nxt]
                if cand < dp[nm][nxt]:
                    dp[nm][nxt] = cand
                    parent[nm][nxt] = last

    best_len = INF
    best_last = -1
    for i in range(k):
        if dp[full][i] < best_len:
            best_len = dp[full][i]
            best_last = i

    # 回推排列
    order = []
    mask, last = full, best_last
    while last != -1:
        order.append(last)
        prev = parent[mask][last]
        mask ^= 1 << last
        last = prev
    order.reverse()

    ans = words[order[0]]
    for t in range(1, len(order)):
        i, j = order[t - 1], order[t]
        ans += words[j][overlap[i][j]:]
    return ans


if __name__ == "__main__":
    got = shortest_superstring(["alex", "loves", "leetcode"])
    assert all(w in got for w in ["alex", "loves", "leetcode"])
    assert len(got) == 17

    got = shortest_superstring(["catg", "ctaagt", "gcta", "ttca", "atgcatc"])
    assert all(w in got for w in ["catg", "ctaagt", "gcta", "ttca", "atgcatc"])
    assert len(got) == 16

    assert shortest_superstring(["abc"]) == "abc"
    assert shortest_superstring(["abc", "bcd"]) == "abcd"
    print("shortest_superstring: all tests passed")
