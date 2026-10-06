"""860. 柠檬水找零（Lemonade Change）

题目：每杯柠檬水售价 5 元，顾客只会递给你 5、10 或 20 元的钞票，你必须当场找零。
一开始你手头没有零钱。给定顾客付款顺序 bills，判断能否给每位顾客正确找零。

思路（贪心：找零时优先用大面额）：
    维护手头 5 元、10 元钞票的数量。逐个顾客处理：
      - 收到 5 元：直接收下，five++；
      - 收到 10 元：必须找回 5 元，所以 five 至少要有 1，然后 five--、ten++；
      - 收到 20 元：需要找 15 元，有两种凑法——「一张 10 + 一张 5」或「三张 5」。
        贪心策略是**优先用 10 + 5**：因为 5 元更灵活（能找 10 元也能凑 15 元），
        而 10 元只能用来凑 15 元。所以先尝试 ten>=1 且 five>=1，不行再试 five>=3，
        都做不到就返回 False。

    为什么优先花掉 10 元是对的（交换论证）：面对一个 20 元顾客，如果同时存在
    「10+5」和「5+5+5」两种找法，两种都能让当前顾客过关。但用掉一张 5 元比用掉一张
    10 元更亏——5 元在后续既能单独应对 10 元顾客，也能参与 20 元顾客的找零；10 元
    却只对 20 元顾客有用。因此保留尽可能多的 5 元永远不吃亏，先花 10 元是安全的。

复杂度：时间 O(n)（一次遍历），空间 O(1)（只数 5 元和 10 元的张数）。
"""


def lemonade_change(bills):
    five = 0
    ten = 0
    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        else:
            if ten >= 1 and five >= 1:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    return True


if __name__ == "__main__":
    assert lemonade_change([5, 5, 5, 10, 20]) is True
    assert lemonade_change([5, 5, 10, 10, 20]) is False
    assert lemonade_change([5, 5, 5, 5, 10, 5, 10, 10, 10, 20]) is True
    assert lemonade_change([10]) is False
    assert lemonade_change([5, 5, 10]) is True
    print("lemonade_change: all tests passed")
