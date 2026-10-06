"""309. 买卖股票的最佳时机含冷冻期（Best Time to Buy and Sell Stock with Cooldown）

题目：可以完成任意多笔交易，但卖出股票的**第二天不能买入**（有一天冷冻期），
求最大利润。

思路（状态机 DP：多一个「刚卖出」的中间状态）：
    122 里只有「持仓 / 空仓」两态，因为卖出后可以立刻再买。冷却期把「卖出」
    拆成了一个必须停留一天的中间态，于是设三个状态：

        hold = 目前持有股票的最大收益
        sold = 今天刚卖出（明天必须冷冻，不能买）
        rest = 空仓且已经脱离冷冻期，随时可以买

    每读到一个价格 p，同时更新三态（都要用**旧值**，所以一起算再赋值）：

        hold = max(hold, rest - p)   # 只有 rest 状态才允许买入
        sold = hold + p              # 今天卖出，必然来自持仓
        rest = max(rest, sold)       # 昨天的 sold 今天解冻，并入 rest

    关键就是买入的来源从 122 的 `cash` 换成了 `rest`——`sold` 当天不能买，
    这一处改动就把冷冻期表达清楚了。答案是空仓态的最大值 `max(rest, sold)`。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_profit_cooldown(prices):
    neg = float("-inf")
    hold, sold, rest = neg, neg, 0
    for p in prices:
        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)
    return max(rest, sold)


if __name__ == "__main__":
    assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
    assert max_profit_cooldown([1]) == 0
    assert max_profit_cooldown([2, 1, 4]) == 3
    assert max_profit_cooldown([6, 1, 3, 2, 4, 7]) == 6
    print("best_time_to_buy_and_sell_stock_with_cooldown: all tests passed")
