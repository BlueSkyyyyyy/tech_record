"""839. 相似字符串组（Similar String Groups）

题目：若两个字符串可以通过「交换其中恰好两个字符」变得相同（或本来就相同），
就称它们相似。给定一组互为字母异位词的字符串，相似关系可传递，求最终分成几组。

思路（「相似」即一条边，合并后数连通分量）：
    把每个字符串看成一个点。两两检查是否相似，相似就 union，最后数连通分量个数。

    相似判定：逐位比较，记录不同的位置。不同的位置数为 0（完全相等）或 2，
    且这两处字符正好互换（a[i]==b[j] 且 a[j]==b[i]）——因为题面保证都是字母异位词，
    只要「恰好两处不同」其实就已互换，但显式写出判断更严谨。

    为什么能两两枚举：n ≤ 300，且串长为字母异位数时相似判定是 O(L)，
    O(n²·L) 足够快；相比用哈希寻找邻居，直接枚举更简单不易错。

复杂度：时间 O(n²·L)，空间 O(n)。
"""
from dsu import DSU


def num_similar_groups(strs):
    n = len(strs)
    dsu = DSU(n)

    def similar(a, b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if not diff:
            return True
        if len(diff) != 2:
            return False
        i, j = diff
        return a[i] == b[j] and a[j] == b[i]

    for i in range(n):
        for j in range(i + 1, n):
            if similar(strs[i], strs[j]):
                dsu.union(i, j)
    return len({dsu.find(i) for i in range(n)})


if __name__ == "__main__":
    assert num_similar_groups(["tars", "rats", "arts", "star"]) == 2
    assert num_similar_groups(["omv", "ovm"]) == 1
    assert num_similar_groups(["abc"]) == 1
    print("similar_string_groups: all tests passed")
