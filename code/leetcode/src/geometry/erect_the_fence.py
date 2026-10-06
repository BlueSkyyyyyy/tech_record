"""587. 安装栅栏（Erect the Fence）

题目：给定平面上一组树的位置 points，用一圈栅栏把它们全部围起来（栅栏是
最短的凸多边形边界），返回**落在栅栏边界上**的所有树的坐标。

思路（Andrew 单调链求凸包）：
    先按 (x, y) 字典序排序并去重。最小凸包的边界可以分为「下链」和「上链」：
    - 从左到右扫描，维护下链：只要新点让最后三个点形成**右转**
      （叉积 < 0），说明中间那个点不在凸包边界上，弹出；否则压入。
    - 从右到左扫描，用同样的规则维护上链。
    两条链拼起来就是整圈边界。

    本题要求保留**边上共线的点**，所以判定用「叉积 < 0 才弹」而不是常见的
    「叉积 <= 0 才弹」：等于 0 表示三点共线，此时中间点也在栅栏上，必须留下。
    去重后的点数不超过 2 时，所有点都在边界上，直接返回。

复杂度：排序 O(n log n)，扫描 O(n)，总体 O(n log n)；空间 O(n)。
"""


def outer_trees(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)

    hull = set(lower) | set(upper)
    return [list(p) for p in sorted(hull)]


if __name__ == "__main__":
    got = outer_trees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]])
    want = [[1, 1], [2, 0], [2, 4], [3, 3], [4, 2]]
    assert sorted(got) == sorted(want)

    got = outer_trees([[1, 2], [2, 2], [4, 2]])
    want = [[1, 2], [2, 2], [4, 2]]
    assert sorted(got) == sorted(want)

    got = outer_trees([[0, 0], [0, 1], [0, 2], [1, 1]])
    want = [[0, 0], [0, 1], [0, 2], [1, 1]]
    assert sorted(got) == sorted(want)

    got = outer_trees([[0, 0], [1, 0], [0, 1]])
    want = [[0, 0], [1, 0], [0, 1]]
    assert sorted(got) == sorted(want)

    print("erect_the_fence: all tests passed")
