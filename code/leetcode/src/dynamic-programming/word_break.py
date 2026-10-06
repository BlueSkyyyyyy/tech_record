"""139. 单词拆分（Word Break）

题目：给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现
的一个或多个单词拼接出 s，则返回 true。字典中的单词可以重复使用。

思路（线性 DP·可行性，本质是「按前缀切割」）：
    设 dp[i] 表示「s 的前 i 个字符能否被字典拼出来」。考虑最后一段单词 s[j:i]
    （即从下标 j 切到 i 的这一段）：

        dp[i] = true，当存在某个 j 使得 dp[j] 为真且 s[j:i] 在字典中。

    换句话说，前缀 i 可拼 ⟺ 它能被切在某个「可拼的前缀 j」之后，且剩下的一截本身是
    字典词。初始化 dp[0] = true（空串当然可拼），答案 dp[n]。

    为什么用「以结尾为状态」：切割问题关心的是「前 i 个字符能否被完整覆盖」，末尾的
    位置天然是状态；每切一刀就把问题拆成「前缀」和「最后一截」。这与 53/674 那种
    「以 i 结尾」的连续性状态同源，只不过判断条件换成了「查字典」。

    为什么可以用集合加速：内层每次都要判断 s[j:i] 是否在字典里，把 wordDict 转成
    set 后，判断是 O(1)（不计子串哈希本身的开销）。

复杂度：时间 O(n²)（枚举切点，每次子串哈希 O(n) 时会更慢，可加长度上界优化），
空间 O(n)。
"""


def word_break(s, word_dict):
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]


if __name__ == "__main__":
    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("applepenapple", ["apple", "pen"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    assert word_break("", []) is True
    assert word_break("a", []) is False
    assert word_break("aaaaaaa", ["aaaa", "aaa"]) is True
    print("word_break: all tests passed")
