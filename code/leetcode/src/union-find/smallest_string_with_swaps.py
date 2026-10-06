"""1202. 交换字符串中的元素（Smallest String With Swaps）

题目：给定字符串 s 和一组可交换的下标对 pairs。你可以任意多次交换 pairs 中任意一对
下标上的字符，求能得到字典序最小的字符串。

思路（可交换关系传递 → 连通分量内自由重排）：
    交换关系是可传递的：若下标 i 与 j 可换、j 与 k 可换，那么 i、j、k 上的字符可以
    在它们之间任意摆放（借助 j 中转）。把下标看成点、pairs 看成边，同一个连通分量内的
    字符可以随意排列。

    于是最优策略：每个连通分量内部，把字符按升序排序，再按下标升序依次填回——
    字典序最小就是「小的下标放小的字符」。

    为什么块间互不影响：不同连通分量之间没有任何可交换路径，字符被锁死在各自块内，
    逐块取最优即全局最优。

复杂度：时间 O(n log n)（排序），空间 O(n)。
"""
from collections import defaultdict

from dsu import DSU


def smallest_string_with_swaps(s, pairs):
    n = len(s)
    dsu = DSU(n)
    for a, b in pairs:
        dsu.union(a, b)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)

    result = list(s)
    for indices in groups.values():
        indices.sort()
        chars = sorted(s[i] for i in indices)
        for i, ch in zip(indices, chars):
            result[i] = ch
    return "".join(result)


if __name__ == "__main__":
    assert smallest_string_with_swaps("dcab", [[0, 3], [1, 2]]) == "bacd"
    assert smallest_string_with_swaps("dcab", [[0, 3], [1, 2], [0, 2]]) == "abcd"
    assert smallest_string_with_swaps("cba", []) == "cba"
    assert smallest_string_with_swaps("abcd", [[0, 1], [1, 2], [2, 3]]) == "abcd"
    print("smallest_string_with_swaps: all tests passed")
