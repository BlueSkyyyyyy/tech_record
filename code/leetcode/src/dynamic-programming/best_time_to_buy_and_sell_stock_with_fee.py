"""714. 买卖股票的最佳时机含手续费（Best Time to Buy and Sell Stock with Transaction Fee）

题目：可以完成任意多笔交易，但每笔交易（一次买入 + 一次卖出）需支付固定手续费
`fee`，求最大利润。

思路（状态机 DP：把手续费塞进卖出的转移）：
    骨架与 122 完全一致，两个状态：

        hold = 目前持有股票的最大收益
        cash = 目前不持有股票的最大收益

    唯一的区别是：卖出时要扣掉手续费，所以卖出那一步从 `hold + p` 变成
    `hold + p - fee`：

        hold = max(hold, cash - p)
        cash = max(cash, hold + p - fee)

    手续费只在卖出时扣一次，天然对应「每笔完整交易收费一次」。因为已经扣过费，
    收益不再是「无脑吃下每段上涨」，价格涨幅大于手续费才值得交易，DP 会自动
    在「继续持有」和「落袋付小费」之间做取舍。最后返回 cash。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_profit_fee(prices, fee):
    neg = float("-inf")
    hold, cash = neg, 0
    for p in prices:
        hold = max(hold, cash - p)
        cash = max(cash, hold + p - fee)
    return cash


if __name__ == "__main__":
    assert max_profit_fee([1, 3, 2, 8, 4, 9], 2) == 8
    assert max_profit_fee([1, 3, 7, 5, 10, 3], 3) == 6
    assert max_profit_fee([1], 1) == 0
    assert max_profit_fee([5, 4, 3], 1) == 0
    assert max_profit_fee([1, 4, 6], 2) == 3
    print("best_time_to_buy_and_sell_stock_with_fee: all tests passed")
