"""11. 盛最多水的容器（Container With Most Water）

题目：给定一个长度为 n 的整数数组 height，第 i 条线的高度为 height[i]，
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

思路：「对撞双指针」。
    面积 = min(height[lo], height[hi]) * (hi - lo)。
    lo、hi 从两端出发，每次移动较矮的那一端：
      - 宽度在缩小，只有让「较矮的一端变高」才可能让面积变大；
      - 移动较高的一端没有意义：瓶口由矮端决定，宽变小、高不会变大。
    每一步都排除掉当前矮端作为答案的可能，因此能收敛到最优。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_area(height):
    lo, hi = 0, len(height) - 1
    best = 0
    while lo < hi:
        h = min(height[lo], height[hi])
        best = max(best, h * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best


if __name__ == "__main__":
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert max_area([1, 1]) == 1
    assert max_area([4, 3, 2, 1, 4]) == 16
    assert max_area([1, 2, 1]) == 2
    print("container_with_most_water: all tests passed")
