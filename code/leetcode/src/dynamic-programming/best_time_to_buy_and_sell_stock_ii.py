"""122. 买卖股票的最佳时机 II（Best Time to Buy and Sell Stock II）

题目：与 121 相同的价格数组，但可以完成任意多笔交易（买入前必须先卖掉手上的
股票，同一天可以先卖后买），求最大利润。

思路（状态机 DP：把「买卖次数」放开）：
    状态定义与 121 完全一样：

        hold = 目前持有股票的最大收益
        cash = 目前不持有股票的最大收益

    区别只在买入时：既然允许反复交易，买入可以从**已经实现的收益** `cash`
    里扣钱，而不是像 121 那样从 0 起算：

        hold = max(hold, cash - p)
        cash = max(cash, hold + p)

    `cash - p` 表示「用之前几笔赚到的钱，今天再买一只」。这样每一段上涨都能
    被完整吃下，等价于「把所有相邻上涨差额相加」。最后返回 cash。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_profit_ii(prices):
    hold = float("-inf")
    cash = 0
    for p in prices:
        hold = max(hold, cash - p)
        cash = max(cash, hold + p)
    return cash


if __name__ == "__main__":
    assert max_profit_ii([7, 1, 5, 3, 6, 4]) == 7
    assert max_profit_ii([7, 6, 4, 3, 1]) == 0
    assert max_profit_ii([1, 2, 3, 4, 5]) == 4
    assert max_profit_ii([1]) == 0
    assert max_profit_ii([2, 4, 1]) == 2
    print("best_time_to_buy_and_sell_stock_ii: all tests passed")
