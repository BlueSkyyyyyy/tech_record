"""274. H 指数（H-Index）

题目：给你一个整数数组 citations，citations[i] 表示研究者的第 i 篇论文被引用的
      次数。H 指数定义为：满足「至少有 h 篇论文每篇被引用至少 h 次」的最大整数 h。

思路（排序后找到分界点）：
    把引用次数从大到小排序。排完后，第 i 个位置（从 1 数起）表示「前 i 篇里引用
    最少的那篇」。若这个位置的引用数 `citations[i-1] >= i`，说明前 i 篇每篇至少
    被引 i 次，于是 h 至少能取到 i；i 再往后走，直到条件第一次不成立就停。

    为什么排序后只需线性扫一遍：排序把「任意 h 篇」的问题变成了「引用最多的 h 篇」。
    要凑出 h 篇都至少 h 次，显然应该挑引用最多的前 h 篇；若连它们都不够，别的组合
    更不可能。所以最优的 h 一定对应排序后的某个前缀。

    也可以不排序：用计数数组统计「引用次数为 k 的论文有几篇」（k 超过 n 的按 n 记），
    再从高到低累加篇数，累计篇数首次达到当前引用次数时即得答案，时间 O(n)。

    本题只详展排序解，计数解一句话带过。
"""


def h_index(citations):
    citations.sort(reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


def h_index_counting(citations):
    n = len(citations)
    bucket = [0] * (n + 1)
    for c in citations:
        bucket[min(c, n)] += 1
    acc = 0
    for h in range(n, -1, -1):
        acc += bucket[h]
        if acc >= h:
            return h
    return 0


if __name__ == "__main__":
    assert h_index([3, 0, 6, 1, 5]) == 3
    assert h_index([1, 3, 1]) == 1
    assert h_index([1, 3, 5, 7, 9]) == 3
    assert h_index([0]) == 0
    assert h_index([]) == 0
    assert h_index_counting([3, 0, 6, 1, 5]) == 3
    assert h_index_counting([1, 3, 1]) == 1
    assert h_index_counting([0]) == 0
    print("h_index: all tests passed")
