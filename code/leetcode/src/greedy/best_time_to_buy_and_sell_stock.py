"""121. 买卖股票的最佳时机（Best Time to Buy and Sell Stock）

题目：给定数组 prices，prices[i] 表示第 i 天的股票价格。你只能选择某一天买入、
并在未来某一天卖出，最多完成一笔交易，求能获得的最大利润；若无法获利则返回 0。

思路（贪心：边走边记「历史最低价」）：
    只允许「先买后卖」，那么在第 i 天卖出的最大利润，就是 prices[i] 减去**第 i 天
    之前出现过的最低价**。于是从左往右扫一遍，维护两个量：
      - min_price：到目前为止见过的最低价格（最佳买点）；
      - best：到目前为止的最大利润。
    每读到一天价格 p，先用 p - min_price 更新 best（今天卖出），再用 p 更新 min_price
    （今天也可以成为将来的买点）。顺序不能反：必须先算利润再更新最低价，否则会用
    今天买、今天卖得到 0。

    为什么贪心是对的：对每个「卖出日」i，能与它配对的最优买入日就是 i 之前的
    最低价那天——固定卖出价后，买入价越低利润越大。所以枚举卖出日、配上历史最低价，
    取最大，就是全局最优。这也等价于「把每个价格都当买点试一遍」，只是用 min_price
    把 O(n²) 压成了 O(n)。

    与动态规划的关系：状态机 DP（hold/cash）也能解本题，二者都是 O(n)/O(1)；贪心
    版本更直白，只维护一个最低价，是「一次遍历 + 维护极值」的典型模板。

复杂度：时间 O(n)（一次遍历），空间 O(1)。
"""


def max_profit(prices):
    min_price = float("inf")
    best = 0
    for p in prices:
        best = max(best, p - min_price)
        min_price = min(min_price, p)
    return best


if __name__ == "__main__":
    assert max_profit([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit([7, 6, 4, 3, 1]) == 0
    assert max_profit([1]) == 0
    assert max_profit([2, 4, 1]) == 2
    assert max_profit([3, 3, 5, 0, 0, 3, 1, 4]) == 4
    print("best_time_to_buy_and_sell_stock: all tests passed")
