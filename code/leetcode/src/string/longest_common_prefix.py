"""14. 最长公共前缀（Longest Common Prefix）

题目：给定字符串数组 strs，找出所有字符串的最长公共前缀；若不存在公共前缀返回空串。

思路（纵向扫描）：
    以第一个字符串为「基准列」，按下标 i 一列一列地比较：取出 strs[0][i]，再看其余每个
    字符串的第 i 个字符是否都等于它。一旦某个字符串已经到头（i == len(s)）或字符不同，
    答案就是 strs[0][:i]；所有列都比完还没有差异，答案就是整个 strs[0]。

    为什么以第一串为基准：公共前缀必然也是第一串的前缀，所以它的长度不会超过第一串。沿
    第一串的每一列向外比对，找到第一处「不齐」即可，天然不用考虑前缀长度超过基准的情况。

    为什么选纵向扫描而不是两两求前缀：纵向扫描一旦发现某列不齐就能立刻返回，不需要等所有
    字符串都处理完，通常更早退出。两两归并（用前一结果和下一串取公共前缀）也是 O(总字符
    数)，可作为横向扫描的对照；分治版把数组二分后合并前缀，本质相同但常数更大。

复杂度：时间 O(所有字符总数)（最坏情况每列检查所有串，恰好等于总长度），空间 O(1)。
"""


def longest_common_prefix(strs):
    if not strs:
        return ""
    for i in range(len(strs[0])):
        ch = strs[0][i]
        for s in strs[1:]:
            if i == len(s) or s[i] != ch:
                return strs[0][:i]
    return strs[0]


if __name__ == "__main__":
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix(["abc"]) == "abc"
    assert longest_common_prefix([]) == ""
    assert longest_common_prefix(["", "b"]) == ""
    assert longest_common_prefix(["ab", "ab", "ab"]) == "ab"

    print("longest_common_prefix: all tests passed")
