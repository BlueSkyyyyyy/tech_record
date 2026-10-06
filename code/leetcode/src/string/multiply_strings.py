"""43. 字符串相乘（Multiply Strings）

题目：给定两个以字符串形式表示的非负整数 num1、num2，返回它们的乘积，也用字符串表示。
不能使用任何内置的大整数库，也不能直接把字符串转成整数。

思路（竖式乘法模拟）：
    回忆小学竖式：num1 的每一位与 num2 的每一位相乘，乘得的结果错位相加。关键观察是
    「错位相加」可以用一张定长数组一次装下——

    设 num1 长度 m、num2 长度 n，两数乘积最多 m+n 位。把结果数组 `res` 开成长度 m+n，
    从低位往高位遍历：num1[i] * num2[j] 的乘积，其低位落在下标 `i+j+1`、高位进位落在
    下标 `i+j`。于是：

        total = num1[i]*num2[j] + res[p2]
        res[p2] = total % 10          # 本位
        res[p1] += total // 10        # 进位

    这里不加进位的穿越处理也不出错，是因为：对固定 i，`p1 = i+j` 恰是下一列 j-1 要处理的
    乘积累加位（那时它变成那个位置的 `p2`），进位会在后续步骤里被继续向右传播。等所有对
    (i, j) 算完，res 就是答案的各位数字，只需去掉前导零。

    特判：任一乘数为 `"0"` 时直接返回 `"0"`，否则去前导零会把答案也去掉。

复杂度：时间 O(m·n)（两层循环），空间 O(m+n)（结果数组）。
"""


def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"

    m, n = len(num1), len(num2)
    res = [0] * (m + n)

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = (ord(num1[i]) - ord("0")) * (ord(num2[j]) - ord("0"))
            p1 = i + j
            p2 = i + j + 1
            total = mul + res[p2]
            res[p2] = total % 10
            res[p1] += total // 10

    start = 0
    while start < len(res) and res[start] == 0:
        start += 1
    return "".join(str(d) for d in res[start:])


if __name__ == "__main__":
    assert multiply("2", "3") == "6"
    assert multiply("123", "456") == "56088"
    assert multiply("0", "123") == "0"
    assert multiply("9", "9") == "81"
    assert multiply("999", "999") == "998001"
    assert multiply("123456789", "987654321") == "121932631112635269"
    assert multiply("100", "10") == "1000"
    print("multiply_strings: all tests passed")
