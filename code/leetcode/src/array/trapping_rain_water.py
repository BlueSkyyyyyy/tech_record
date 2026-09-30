"""42. 接雨水（Trapping Rain Water）

题目：给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，
下雨之后能接多少雨水。

思路一（对撞双指针，推荐）：O(n) 时间、O(1) 空间。
    某个位置 i 能接的水 = min(它左边的最大高度, 它右边的最大高度) - height[i]。
    与其预先求出每个位置的左右最大值（需要额外空间），不如从两端向中间推进：
    维护 left_max、right_max 两个「已经扫过部分」的最大值。
    每轮让较矮的一端向中间走一步——因为这一端的水位由对面更高的柱子兜底，
    只取决于本侧的 left_max/right_max，于是可以立即结算它的储水量。

思路二（单调栈）：O(n) 时间、O(n) 空间。
    维护一个高度递减的下标栈。当遇到比栈顶更高的柱子时，说明栈顶那根柱子
    和当前柱子之间形成了一个「凹槽」，弹出栈顶作为槽底，横向按宽度累加水量。

复杂度：双指针 O(n)/O(1)；单调栈 O(n)/O(n)。
"""


def trap(height):
    if not height:
        return 0
    lo, hi = 0, len(height) - 1
    left_max, right_max = height[lo], height[hi]
    water = 0
    while lo < hi:
        if height[lo] < height[hi]:
            lo += 1
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]
        else:
            hi -= 1
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]
    return water


def trap_stack(height):
    stack = []
    water = 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            width = i - stack[-1] - 1
            bounded = min(height[stack[-1]], h) - height[bottom]
            water += width * bounded
        stack.append(i)
    return water


if __name__ == "__main__":
    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap_stack([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap([4, 2, 0, 3, 2, 5]) == 9
    assert trap_stack([4, 2, 0, 3, 2, 5]) == 9
    assert trap([]) == 0
    assert trap_stack([]) == 0
    assert trap([1, 2, 3, 4, 5]) == 0
    assert trap_stack([1, 2, 3, 4, 5]) == 0
    print("trapping_rain_water: all tests passed")
