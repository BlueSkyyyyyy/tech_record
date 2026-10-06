"""75. 颜色分类（Sort Colors）

题目：给定一个包含红色、白色、蓝色（用 0、1、2 表示）的数组 nums，原地对它排序，
      使得相同颜色相邻，并按 0、1、2 的顺序排列。要求不使用库排序函数，且只允许
      常数级额外空间。

思路（荷兰国旗问题，三指针一趟扫描）：
    因为只有三种取值，我们不需要通用排序，可以一次扫描把它们分到「0 区」「1 区」
    「2 区」三块。用三个指针：
      - `p0`：0 区右边界之后的位置（下一个 0 该放的地方）；
      - `p2`：2 区左边界之前的位置（下一个 2 该放的地方）；
      - `cur`：当前考察的位置。

    从头往后扫描 `cur`：
      - 遇到 0：和 `p0` 交换，把 0 归入左侧 0 区，`p0`、`cur` 都右移；
      - 遇到 2：和 `p2` 交换，把 2 归入右侧 2 区，`p2` 左移，但 `cur` **不动**；
      - 遇到 1：它本就在中间，`cur` 右移。

    为什么遇到 0 交换后 cur 可以前进：换到 cur 位置的是 p0 处的元素，而 p0 <= cur，
    p0 走过的地方只可能是 0（已被归位），所以换来的其实是已经处理过的 0 或就是自己，
    前方安全，cur 前进。而遇到 2 时，从 p2 换过来的是「还没看过的元素」，它可能是
    0、1 或 2，必须留在原地下一轮再判断，所以 cur 不前进。

    为什么这样就是「一趟扫描原地排序」：`cur` 只在处理 0 和 1 时前进，处理 2 时靠
    `p2` 收缩，最终 `cur > p2` 时中间所有元素都归位，无需第二趟。
"""


def sort_colors(nums):
    p0, cur, p2 = 0, 0, len(nums) - 1
    while cur <= p2:
        if nums[cur] == 0:
            nums[p0], nums[cur] = nums[cur], nums[p0]
            p0 += 1
            cur += 1
        elif nums[cur] == 2:
            nums[cur], nums[p2] = nums[p2], nums[cur]
            p2 -= 1
        else:
            cur += 1


if __name__ == "__main__":
    a = [2, 0, 2, 1, 1, 0]
    sort_colors(a)
    assert a == [0, 0, 1, 1, 2, 2]

    b = [2, 0, 1]
    sort_colors(b)
    assert b == [0, 1, 2]

    c = [0]
    sort_colors(c)
    assert c == [0]

    d = [2, 2, 1, 0, 0, 1]
    sort_colors(d)
    assert d == [0, 0, 1, 1, 2, 2]

    e = []
    sort_colors(e)
    assert e == []

    print("sort_colors: all tests passed")
