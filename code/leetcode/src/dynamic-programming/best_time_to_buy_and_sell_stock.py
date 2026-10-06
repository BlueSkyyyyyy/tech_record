"""121. 买卖股票的最佳时机（Best Time to Buy and Sell Stock）

题目：给定数组 prices，prices[i] 表示第 i 天股票价格。你只能选择某一天买入、
并在未来某一天卖出，最多完成一笔交易，求最大利润；不能获利就返回 0。

思路（状态机 DP）：
    把「手上有没有股票」当成两个状态，边走边维护各自的「最大累计收益」：

        hold = 目前持有股票时的最大收益（收益为负表示花了多少钱买入）
        cash = 目前不持有股票时的最大收益

    每读到一个价格 p，两个状态各自可以「保持不动」或「切换」：

        hold = max(hold, -p)        # 要么之前就持有，要么今天买入（第一笔，扣 p）
        cash = max(cash, hold + p)  # 要么之前就空仓，要么今天把股票卖掉（加 p）

    为什么这样定义就能保证「只买一次」：`-p` 是「从 0 收益直接买入」，不会接在
    卖出收益之上，所以 hold 里始终只含一笔买入的成本；把 hold 卖掉得到 cash，
    就完成了一笔完整交易。最后空仓一定不比持仓差（卖掉落袋），返回 cash。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_profit(prices):
    hold = float("-inf")
    cash = 0
    for p in prices:
        hold = max(hold, -p)
        cash = max(cash, hold + p)
    return cash


if __name__ == "__main__":
    assert max_profit([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit([7, 6, 4, 3, 1]) == 0
    assert max_profit([1]) == 0
    assert max_profit([2, 4, 1]) == 2
    print("best_time_to_buy_and_sell_stock: all tests passed")
