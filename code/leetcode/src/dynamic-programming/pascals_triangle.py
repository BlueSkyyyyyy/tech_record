"""118. 杨辉三角（Pascal's Triangle）

题目：给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。每一行的首尾是 1，
中间每个数等于它上方两个数之和。

思路（动态规划：二维状态，每行由上一行推出）：
    状态是二维的：dp[i][j] 表示第 i 行第 j 个元素（行列都从 0 开始）。
    转移方程来自图形本身：除掉首尾的 1，
        dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
    也就是「正上方」和「左上方」两个数相加。逐行、逐列由小到大填表即可。

    初始化：每一行首尾都是 1。可以先把整行填 1，再只更新中间下标
    j = 1 .. i-1；这样首尾自然保持为 1，不用特判。

    为什么遍历顺序是「上行→下行、左→右」：算 dp[i][j] 时要用上一行的 j-1 和 j，
    只要保证上一行整行已经算完、且本行的 j-1 不影响本行 j（它们用的是不同行的数据），
    就行。从上往下逐行生成，天然满足。

    这题是「结果本身就是一整张表」的典型：dp 不是用来算某一个答案，而是题目
    要求输出的形状，所以直接构造每一行即可，不需要额外滚动压缩。

复杂度：时间 O(numRows^2)（每行长度线性递增），空间 O(numRows^2)（输出本身）。
"""


def generate(num_rows):
    triangle = []
    for i in range(num_rows):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(row)
    return triangle


if __name__ == "__main__":
    assert generate(0) == []
    assert generate(1) == [[1]]
    assert generate(2) == [[1], [1, 1]]
    assert generate(5) == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    print("pascals_triangle: all tests passed")
