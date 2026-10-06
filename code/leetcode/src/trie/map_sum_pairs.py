"""677. 键值映射

题目：设计一个 map，支持两种操作：
        - insert(key, val)：插入键值对，若 key 已存在则覆盖旧值；
        - sum(prefix)：返回所有以 prefix 为前缀的键所对应的值之和。

思路：用前缀树存键，但每个节点额外维护一个 total，表示「经过这个节点的所有键的值之和」。
      插入 key 时，先算出这次值的增量 delta = val - 旧值（旧值不存在记为 0），
      再沿着 key 的字符往下走，每经过一个节点就把该节点的 total 加上 delta，
      最后在末节点记下这个 key 的最新值。

      为什么用增量 delta 而不是「重算一遍」：因为覆盖旧值时要先把旧贡献减掉、再加上
      新贡献。如果直接把整条路径的 total 加上 val，会把旧值重复累加。用 delta 一次
      修正，插入仍是 O(L)。这也是「覆盖型更新」的通用手法：先求差，再统一施加。

      为什么 sum 只需走到 prefix 末节点：那个节点的 total 恰好统计了所有经过它
      （即以该 prefix 开头）的键的值之和，直接返回即可。

复杂度：insert 和 sum 均为 O(L)，L 为键或前缀长度；空间 O(总字符数)。
"""


class MapSum:
    def __init__(self):
        self.children = {}
        self.total = 0
        self.value = 0

    def insert(self, key, val):
        delta = val - self._get(key)
        node = self
        for ch in key:
            if ch not in node.children:
                node.children[ch] = MapSum()
            node = node.children[ch]
            node.total += delta
        node.value = val

    def sum(self, prefix):
        node = self
        for ch in prefix:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.total

    def _get(self, key):
        node = self
        for ch in key:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.value


if __name__ == "__main__":
    m = MapSum()
    m.insert("apple", 3)
    assert m.sum("ap") == 3
    m.insert("app", 2)
    assert m.sum("ap") == 5
    m.insert("apple", 2)  # 覆盖：apple 由 3 变成 2
    assert m.sum("ap") == 4
    assert m.sum("app") == 4  # app(2) + apple(2)，因 apple 也以 app 为前缀
    assert m.sum("b") == 0
    print("map_sum_pairs: all tests passed")
