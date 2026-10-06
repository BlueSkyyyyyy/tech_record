"""391. 完美矩形（Perfect Rectangle）

题目：给定一组轴对齐矩形 rectangles[i] = [x1, y1, x2, y2]，判断它们能否**恰好**
（不重叠、无空隙）拼成一个大矩形。

思路（面积 + 角点奇偶性）：
    恰好铺满大矩形，需要且只需两个条件同时成立：
    1. **总面积 = 大矩形的面积**。大矩形由所有小矩形的边界决定：左下角是
       全局最小的 (min_x, min_y)，右上角是全局最大的 (max_x, max_y)；
    2. **角点出现次数的奇偶性正确**。把每个矩形的四个角点拿出来，用集合做「异或」
       （出现偶数次就抵消、奇数次留下）。在完美铺法中：
       - 内部的所有拼接点，都是偶数个矩形的角（成对重合），会互相抵消；
       - 大矩形的四个外角各只属于一个矩形，是奇点，必然留下；
       所以最终留下的点集**恰好等于大矩形的四个角**。
    若存在重叠，面积为满足条件但角点会多出奇点；若存在空缺，面积就不相等。
    两个条件一起就排除了所有非法情况。

复杂度：时间 O(n)，空间 O(n)。
"""


def is_rectangle_cover(rectangles):
    area = 0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    corners = set()

    for x1, y1, x2, y2 in rectangles:
        area += (x2 - x1) * (y2 - y1)
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
        for pt in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
            if pt in corners:
                corners.remove(pt)
            else:
                corners.add(pt)

    if area != (max_x - min_x) * (max_y - min_y):
        return False
    expected = {(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)}
    return corners == expected


if __name__ == "__main__":
    assert is_rectangle_cover(
        [[1, 1, 3, 3], [3, 1, 4, 2], [3, 2, 4, 3], [1, 3, 4, 4]]
    ) is True
    assert is_rectangle_cover(
        [[1, 1, 2, 3], [1, 3, 2, 4], [3, 1, 4, 2], [3, 2, 4, 4]]
    ) is False
    assert is_rectangle_cover(
        [[1, 1, 3, 3], [3, 1, 4, 2], [1, 3, 2, 4], [2, 2, 4, 4]]
    ) is False
    assert is_rectangle_cover(
        [[0, 0, 1, 1], [0, 1, 1, 2], [0, 2, 1, 3], [0, 3, 1, 4]]
    ) is True
    print("perfect_rectangle: all tests passed")
