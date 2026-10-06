"""947. 移除最多的同行或同列石头（Most Stones Removed with Same Row or Column）

题目：二维平面上有若干石头，坐标为 [x, y]。若某块石头所在的行或列上还有别的石头，
就可以把它移走。求最多能移走多少块。

思路（把行、列都看成点，石头是连接行与列的边）：
    一块石头架在「行 x」和「列 y」之间。凡是能通过若干石头互相到达的行和列，
    就属于同一个连通块；在一个含 k 块石头的连通块里，总能只留下 1 块、移走 k-1 块。
    所以答案是 n -（连通块个数）。

    实现时把行坐标和列坐标分别映射成并查集的编号（列编号整体加一个偏移量避免与行冲突），
    每块石头 union(行, 列)。最后数不同根即可。

    为什么不同坐标要先离散化：坐标可能很大也可能为负，直接拿来当下标不安全。
    用集合去重 + 字典映射到 0..m-1 即可。

复杂度：时间 O(n·α(n))，空间 O(n)。
"""
from dsu import DSU


def remove_stones(stones):
    xs = {x for x, _ in stones}
    ys = {y for _, y in stones}
    x_id = {x: i for i, x in enumerate(xs)}
    y_id = {y: len(xs) + i for i, y in enumerate(ys)}

    dsu = DSU(len(xs) + len(ys))
    for x, y in stones:
        dsu.union(x_id[x], y_id[y])

    roots = {dsu.find(x_id[x]) for x, _ in stones}
    return len(stones) - len(roots)


if __name__ == "__main__":
    assert remove_stones([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]]) == 5
    assert remove_stones([[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]) == 3
    assert remove_stones([[0, 0]]) == 0
    print("most_stones_removed_with_same_row_or_column: all tests passed")
