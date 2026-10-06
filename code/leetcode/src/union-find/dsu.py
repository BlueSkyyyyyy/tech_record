"""并查集（Disjoint Set Union）模板。

本模板供 union-find 分类下的各题复用。两个优化同时上：
- 路径压缩：find 时把查询路径上的点直接挂到根，之后的查询一步到位；
- 按秩合并：union 时把矮树挂到高树下，避免树退化成一条链。

两者结合后，单次操作的**摊还**复杂度近似 O(α(n))，其中 α 是反阿克曼函数，
在 n 取到宇宙原子量级时也不超过 5，可当作「几乎 O(1)」。

用法：
    from dsu import DSU

    d = DSU(n)          # 元素编号为 0..n-1
    d.find(x)           # 返回 x 所在集合的代表元（根）
    d.union(a, b)       # 合并两个集合；返回 False 表示二者本就连通（可用来判环）
"""


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


if __name__ == "__main__":
    d = DSU(5)
    assert d.find(0) == 0
    assert d.find(4) == 4
    assert d.union(0, 1) is True
    assert d.union(1, 2) is True
    assert d.find(0) == d.find(2)
    assert d.union(2, 0) is False          # 已在同一集合，返回 False
    assert d.union(3, 4) is True
    assert d.find(0) != d.find(3)
    assert d.union(0, 4) is True
    assert d.find(2) == d.find(3)
    print("dsu: all tests passed")
