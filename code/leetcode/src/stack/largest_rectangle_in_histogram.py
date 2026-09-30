"""84. 柱状图中最大的矩形（Largest Rectangle in Histogram）

题目：给定 n 个非负整数表示柱状图中各柱子的高度（宽度均为 1），
      求该柱状图内能勾勒出的最大矩形面积。

思路（单调递增栈，栈里存下标）：
    枚举「以某根柱子为矩形的高」。此时矩形的左右边界，是它左右两边
    第一个比它矮的柱子——因为一旦碰到更矮的柱子，高度就维持不住了。
    于是问题变成：对每根柱子，找它左边和右边第一个更矮的位置。
    这正是单调栈擅长的。

    从左到右扫描，维护一个高度单调递增的栈。遇到当前高度 h 比栈顶矮时，
    栈顶那根柱子「右边的第一个更矮者」就是当前 i，它左边第一个更矮者就是栈里它下面那根，
    两者之间的宽度 i - stack[-1] - 1（栈空则为 i）就是它能撑起的最大宽度，
    用它乘高度结算这块面积，然后弹出。重复到栈顶不再更高，再把 i 压栈。

    技巧：在数组末尾补一个高度 0 的哨兵。这样扫描结束时，栈里所有柱子
    都会遇到「更矮的 0」而被结算，不必再写一段收尾代码。

复杂度：时间 O(n)（每个下标进出栈一次），空间 O(n)（栈 + 一份拷贝）。
"""


def largest_rectangle_area(heights):
    heights = list(heights) + [0]
    stack = []
    best = 0
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            bar = stack.pop()
            height = heights[bar]
            left = stack[-1] if stack else -1
            width = i - left - 1
            best = max(best, height * width)
        stack.append(i)
    return best


if __name__ == "__main__":
    assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10
    assert largest_rectangle_area([2, 4]) == 4
    assert largest_rectangle_area([2, 1, 2]) == 3
    assert largest_rectangle_area([1]) == 1
    assert largest_rectangle_area([1, 1, 1, 1]) == 4
    print("largest_rectangle_area: all tests passed")
