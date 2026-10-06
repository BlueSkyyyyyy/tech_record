"""950. 按递增顺序显示卡牌（Reveal Cards In Increasing Order）

题目：有一副牌，翻牌规则是「亮出牌堆顶、然后把下一张移到牌堆底、重复」。
已知最终亮出的顺序是递增的，求牌堆一开始的排列。

思路（把过程倒过来，用双端队列构造）：
    正过程是「弹出一张、把下一张搬到底」，很难从结果直接推初始。
    但我们知道**亮出的序列是排序后的牌**。于是倒着重建：
    从大到小处理每张牌 card，把 card 放到牌堆**最前面**；放之前，如果堆里已有牌，
    先把堆底那张搬到堆顶（这是正过程「把顶搬到底」的逆操作）。

    为什么倒着对：设正过程某一步弹出 X、把 Y 搬到底。倒着看，就是最后插入最前面的
    card 对应那次弹出，而堆底的 Y 要回到堆顶为更早的步骤做准备。按牌面从大到小插入，
    保证每次搬到底的正是正过程中会被亮出的下一张。

复杂度：时间 O(n log n)（排序），空间 O(n)。
"""

from collections import deque


def deck_revealed_increasing(deck):
    d = deque()
    for card in sorted(deck, reverse=True):
        if d:
            d.appendleft(d.pop())
        d.appendleft(card)
    return list(d)


if __name__ == "__main__":
    assert deck_revealed_increasing([17, 13, 11, 2, 3, 5, 7]) == [2, 13, 3, 11, 5, 17, 7]
    assert deck_revealed_increasing([1, 1000]) == [1, 1000]
    assert deck_revealed_increasing([1, 2, 3]) == [1, 3, 2]
    assert deck_revealed_increasing([1]) == [1]
    print("deck_revealed_increasing: all tests passed")
