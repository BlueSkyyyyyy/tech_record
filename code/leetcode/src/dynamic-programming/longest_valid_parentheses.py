"""32. 最长有效括号（Longest Valid Parentheses）

题目：给定一个只含 '(' 和 ')' 的字符串 s，找出最长的、括号正确匹配的连续子串，
返回其长度。

思路（动态规划：以「我」结尾的最长有效括号）：
    要求「连续」，仍把状态钉在右端点：

        dp[i] = 以 s[i] 结尾的最长有效括号子串的长度（s[i] 必须是 ')'）。

    结尾是 '(' 时不可能配对，dp 直接为 0。结尾是 ')' 时，看它前面是谁：

    1. 若 s[i-1] == '('，即形成一对相邻的 "()"，长度为 2，再拼上前面（以
       s[i-2] 结尾）的有效段：

           dp[i] = dp[i-2] + 2

    2. 否则 s[i-1] == ')'，即 s[i-1] 处已有一段有效括号，长度为 dp[i-1]。
       那么能与 s[i] 配对的 '(' 必须正好在它前面一格：位置
       j = i - dp[i-1] - 1。若 j 合法且 s[j] == '('，则把「中间的整段」
       （dp[i-1]）、「这一对」（2）、以及 j 之前那段（dp[j-1]）拼起来：

           dp[i] = dp[i-1] + 2 + dp[j-1]（j >= 1 时）

    为什么必须跳过「中间整段」去找 j：一旦里面已经配对成功，外层的 '(' 只能来自
    更靠前的位置，中间那些 ')' 才不会和它冲突。这一步的推导最容易写错，写完务必
    用几个例子（"()(())"、"(()())"、")()())"、"()(()"）手推一遍验证。

    答案不是 dp[n-1]，而是所有 dp[i] 的最大值——最长有效段可以在任何位置结尾。

    本题也可用栈或左右计数两遍扫描，但 DP 版本最能体现「找对配对位置」的推理，
    这里只详展它。

复杂度：时间 O(n)，空间 O(n)（可用滚动变量或栈进一步优化，这里保留数组更清晰）。
"""


def longest_valid_parentheses(s):
    n = len(s)
    dp = [0] * n
    best = 0
    for i in range(1, n):
        if s[i] == ')':
            if s[i - 1] == '(':
                dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
            elif dp[i - 1] > 0:
                j = i - dp[i - 1] - 1
                if j >= 0 and s[j] == '(':
                    dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)
            best = max(best, dp[i])
    return best


if __name__ == "__main__":
    assert longest_valid_parentheses("(()") == 2
    assert longest_valid_parentheses(")()())") == 4
    assert longest_valid_parentheses("") == 0
    assert longest_valid_parentheses("()(())") == 6
    assert longest_valid_parentheses("(()())") == 6
    assert longest_valid_parentheses("()(()") == 2
    assert longest_valid_parentheses(")(") == 0
    print("longest_valid_parentheses: all tests passed")
