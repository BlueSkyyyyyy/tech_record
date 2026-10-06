"""765. 情侣牵手（Couples Holding Hands）

题目：2n 个人坐成一排，第 i 对情侣的编号是 2i 与 2i+1。row 是当前座位安排（一个排列）。
每次可以交换任意两人，求最少交换几次，使每对情侣相邻而坐。

思路（把「坐错位的两对情侣」连边，答案是 n - 连通块数）：
    情侣 (2k, 2k+1) 看作一个「对」，编号 k。观察每一对相邻座位 (2i, 2i+1)：
    两个人分属的对 k1、k2，若 k1 != k2，就在 k1、k2 之间连一条边。

    这样得到的图里，一个含 m 个「对」的连通块，只需 m-1 次交换就能全部配对
    （每次交换把一个对归位并减少一个错位），所以总次数 = Σ(m_j - 1) = n - 连通块数。

    为什么用并查集而不是模拟交换：我们只关心「最少次数」，而次数只与连通块大小有关，
    并不需要真的执行交换。

复杂度：时间 O(n·α(n))，空间 O(n)。
"""
from dsu import DSU


def min_swaps_couples(row):
    n = len(row) // 2
    dsu = DSU(n)
    for i in range(0, len(row), 2):
        dsu.union(row[i] // 2, row[i + 1] // 2)
    return n - len({dsu.find(i) for i in range(n)})


if __name__ == "__main__":
    assert min_swaps_couples([0, 2, 1, 3]) == 1
    assert min_swaps_couples([3, 2, 0, 1]) == 0
    assert min_swaps_couples([0, 2, 1, 3, 4, 6, 5, 7]) == 2
    assert min_swaps_couples([2, 0, 1, 3]) == 1
    assert min_swaps_couples([0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11]) == 3
    print("couples_holding_hands: all tests passed")
