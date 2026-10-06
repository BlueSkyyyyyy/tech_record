"""455. 分发饼干（Assign Cookies）

题目：有一群孩子和一堆饼干，数组 g[i] 是第 i 个孩子的胃口值，数组 s[j] 是第 j 块
饼干的尺寸。只有当 s[j] >= g[i] 时，这块饼干才能喂饱这个孩子。求最多能满足多少
个孩子（每个孩子最多一块饼干，每块饼干最多给一个孩子）。

思路（贪心：小饼干优先喂最不挑食的孩子）：
    先把 g 和 s 都升序排序。然后用两个指针从前往后扫：
      - 若当前饼干 s[j] >= 当前孩子 g[i]，就把它分给这个孩子，i++、j++；
      - 否则这块饼干连当前胃口最小的孩子都喂不饱，对后面的孩子更没用，直接 j++ 丢掉。
    最后 i 就是被满足的孩子数。

    为什么贪心是对的（交换论证）：考虑任意一个最优分配方案。把所有人按胃口从小到大
    看，最优方案一定能让「胃口最小的若干个孩子」被满足，且分给他们的饼干也能按尺寸
    从小到大配对。假如有一块更小的饼干 s[j] 能喂饱当前最不挑的孩子 g[i]，而我们却
    把一块更大的饼干分给了他，那么把这两块饼干对调——小的那块照样喂饱这个孩子，
    大的那块留给后面胃口更大的孩子只会更宽裕，不会让答案变差。所以「能用小饼干满足
    就先用小饼干」不会错过最优解。

    为什么排序：排序后「当前最不挑的孩子」和「当前最小的饼干」都排在最前面，两个
    指针都只会单向移动，一趟扫描即可。

复杂度：时间 O(n log n + m log m)（两边排序），空间 O(1)（排序外只用常数变量，
    不计入输入）。
"""


def find_content_children(g, s):
    g.sort()
    s.sort()
    i = 0
    j = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            i += 1
            j += 1
        else:
            j += 1
    return i


if __name__ == "__main__":
    assert find_content_children([1, 2, 3], [1, 1]) == 1
    assert find_content_children([1, 2], [1, 2, 3]) == 2
    assert find_content_children([1, 2, 3], [3]) == 1
    assert find_content_children([10, 9, 8, 7], [5, 6, 7, 8]) == 2
    assert find_content_children([], [1, 2]) == 0
    assert find_content_children([1, 2], []) == 0
    print("assign_cookies: all tests passed")
