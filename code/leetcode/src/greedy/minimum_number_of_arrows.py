"""452. 用最少数量的箭引爆气球（Minimum Number of Arrows to Burst Balloons）

题目：气球在一个水平面上，每个气球用区间 points[i] = [x_start, x_end] 表示其在 x 轴上的
直径范围。一支箭可以竖直向上射，若射击位置 x 落在某个气球的区间内，该气球就会被引爆。
可以无限次射箭，求引爆所有气球所需的最少箭数。

思路（贪心：按右端点排序，一支箭射在所有还能射中的气球最右端）：
    要让箭尽可能少，每支箭就应该尽量多穿几个气球。一个气球被引爆，等价于箭的 x 落在
    它的区间内，所以一支箭最多能引爆所有「共同覆盖某一点」的气球。
    把所有气球按右端点从小到大排序，先射一支箭在第一个气球的右端点，然后从左往右扫：
      - 若当前气球的左端点 <= 当前箭的位置，说明这支箭也能射中它，不用新箭；
      - 若当前气球的左端点 > 当前箭的位置，说明它完全在这支箭右侧、射不到，必须再射一支，
        并把箭挪到这个气球的右端点。
    把箭放在当前气球的右端点，是「既能射中当前气球、又尽量靠右」的位置中最靠右的一个，
    对后面气球的覆盖只会更多，不会更少。

    为什么贪心是对的（交换论证）：考虑按右端点排序后第一个气球，它的右端点是 e。任何
    最优方案里，射中这个气球的箭位置 p 一定满足 p <= e；把它平移到 e，这个气球照样被
    射中，而 e >= p 只会让箭更容易够到后面那些区间——后续气球只要和 p 有公共点，就也
    一定包含 e（它们在排序后右端点不比 e 小，而左端点又不大于 p）。所以第一支箭放在 e
    不影响最优性。之后对剩下「没被射中」的气球重复同样的论证，就得到整支贪心策略。

复杂度：时间 O(n log n)（排序），空间 O(1)（排序之外只用常数变量）。
"""


def find_min_arrow_shots(points):
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    end = points[0][1]
    for start, finish in points:
        if start > end:
            arrows += 1
            end = finish
    return arrows


if __name__ == "__main__":
    assert find_min_arrow_shots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
    assert find_min_arrow_shots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
    assert find_min_arrow_shots([[1, 2], [2, 3], [3, 4], [4, 5]]) == 2
    assert find_min_arrow_shots([]) == 0
    assert find_min_arrow_shots([[1, 2]]) == 1
    print("minimum_number_of_arrows: all tests passed")
