"""188. 买卖股票的最佳时机 IV（Best Time to Buy and Sell Stock IV）

题目：给定价格数组 prices 和整数 k，最多完成 k 笔交易，求最大利润。

思路（状态机 DP：把 123 的四个状态推广成 2k 个）：
    123 里第一笔 / 第二笔分别有一对「买/卖」状态。把它推广到 k 笔：用两个长度为
    `k+1` 的数组维护每个阶段的最优收益。

        buy[j]  = 已完成 j-1 笔、且第 j 笔已买入时的最大收益
        sell[j] = 已完成 j 笔时的最大收益

    每读到一个价格 p，按 j 从小到大滚动更新这一天的所有阶段：

        buy[j]  = max(buy[j],  sell[j-1] - p)   # 第 j 笔买入，接在第 j-1 笔收益之后
        sell[j] = max(sell[j], buy[j] + p)      # 第 j 笔卖出

    因为 buy[j] 只用到 sell[j-1]、sell[j] 只用到 buy[j]，同一天先卖再买也合法，
    顺序从左到右滚动即可。`sell[0] = 0` 表示「一笔都不做」，所以 j=1 时
    `buy[1] = -p`，正是第一笔买入，边界自然。

    小技巧：当 `k >= 交易天数 // 2` 时交易次数已不受限，可退化成 122 的无限笔
    状态机；但通式 `O(nk)` 对 LeetCode 的数据范围已经足够。

复杂度：时间 O(n·k)，空间 O(k)。
"""


def max_profit_iv(k, prices):
    if k <= 0 or not prices:
        return 0
    neg = float("-inf")
    buy = [neg] * (k + 1)
    sell = [0] * (k + 1)
    for p in prices:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    return sell[k]


if __name__ == "__main__":
    assert max_profit_iv(2, [2, 4, 1]) == 2
    assert max_profit_iv(2, [3, 2, 6, 5, 0, 3]) == 7
    assert max_profit_iv(0, [1, 2, 3]) == 0
    assert max_profit_iv(1, [7, 1, 5, 3, 6, 4]) == 5
    assert max_profit_iv(100, [1, 2, 3, 4, 5]) == 4
    assert max_profit_iv(2, []) == 0
    print("best_time_to_buy_and_sell_stock_iv: all tests passed")
