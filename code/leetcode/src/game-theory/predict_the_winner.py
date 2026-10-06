"""486. 预测赢家（Predict the Winner）

题目：给一个整数数组 nums，两名玩家轮流从数组的**当前两端**取一个数并加入自己的总分，
取完为止。两人都最优策略。问先手玩家的总分能否大于等于后手。

思路（区间 DP 记录「分差」）：
    零和博弈里不必分别记两个人的总分，只记「当前行动者相对对手能领先多少分」就够了。
    设 dp[i][j] 表示在子数组 nums[i..j] 上，轮到行动的人能取得的**最大分差**（自己减
    对手）。若取左端 nums[i]，得到 nums[i]，但接下来对手在 [i+1, j] 上作为行动者能领先
    dp[i+1][j]，所以净分差是 nums[i] - dp[i+1][j]；取右端同理。取两者较大：

        dp[i][j] = max( nums[i] - dp[i+1][j], nums[j] - dp[i][j-1] )

    单元素区间 dp[i][i] = nums[i]（取走它，分差就是它）。最后 dp[0][n-1] 就是先手的分差，
    大于等于 0 说明先手不败（平局也算先手赢，题目如此规定）。

    为什么用「分差」：总分是固定的，谁领先在对局结束时只看差值。把「双方得分」压缩成
    一个差值，既少一维状态，转移也自然。

复杂度：时间 O(n^2)，空间 O(n^2)。
"""


def predict_the_winner(nums):
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = nums[i]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0


if __name__ == "__main__":
    assert predict_the_winner([1, 5, 2]) is False
    assert predict_the_winner([1, 5, 233, 7]) is True
    assert predict_the_winner([5]) is True
    assert predict_the_winner([1, 1]) is True
    assert predict_the_winner([2, 4, 1, 2, 7, 8]) is True
    print("predict_the_winner: all tests passed")
