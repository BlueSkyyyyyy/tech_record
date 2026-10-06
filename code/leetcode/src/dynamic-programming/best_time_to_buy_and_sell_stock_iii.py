"""123. 买卖股票的最佳时机 III（Best Time to Buy and Sell Stock III）

题目：最多完成**两笔**交易，求最大利润（买入前必须卖出，同一天可以先卖后买）。

思路（状态机 DP：把状态按「第几笔交易 + 持仓与否」摊开）：
    一笔交易由「买入」「卖出」两个动作组成。最多两笔，就有四个阶段：

        buy1  = 第一笔已买入、尚未卖出时的最大收益
        sell1 = 第一笔已完成后的最大收益
        buy2  = 第二笔已买入、尚未卖出时的最大收益
        sell2 = 第二笔已完成后的最大收益（答案）

    每读到一个价格 p，四个状态依次松弛（顺序天然满足「先买后卖、先卖后买」）：

        buy1  = max(buy1,  -p)          # 第一次买入
        sell1 = max(sell1, buy1 + p)    # 第一次卖出
        buy2  = max(buy2,  sell1 - p)   # 第二次买入（用第一笔的收益）
        sell2 = max(sell2, buy2 + p)    # 第二次卖出

    这样「最多两笔」自动成立：可以只用第一笔（buy2 永远不触发），也可以一笔不做
    （全初始化为 0 收益）。若把四个状态推广成 `2k` 个，就是下一题 188。

复杂度：时间 O(n)，空间 O(1)（状态数固定为 4）。
"""


def max_profit_iii(prices):
    neg = float("-inf")
    buy1 = buy2 = neg
    sell1 = sell2 = 0
    for p in prices:
        buy1 = max(buy1, -p)
        sell1 = max(sell1, buy1 + p)
        buy2 = max(buy2, sell1 - p)
        sell2 = max(sell2, buy2 + p)
    return sell2


if __name__ == "__main__":
    assert max_profit_iii([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert max_profit_iii([1, 2, 3, 4, 5]) == 4
    assert max_profit_iii([7, 6, 4, 3, 1]) == 0
    assert max_profit_iii([1]) == 0
    assert max_profit_iii([2, 1, 2, 0, 1]) == 2
    print("best_time_to_buy_and_sell_stock_iii: all tests passed")
