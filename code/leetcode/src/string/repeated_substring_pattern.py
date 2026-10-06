"""459. 重复的子字符串（Repeated Substring Pattern）

题目：给定非空字符串 s，判断它能否由某个子串重复多次（至少两次）拼接而成。

思路（前缀函数 / 最小周期）：
    设 n = s 的长度，p = lps[n-1]，即 s 的「最长公共真前后缀」长度。那么 s 的最小周期
    长度是 n - p。如果这个周期能整除 n（n % (n - p) == 0），就说明整个串是由前 n - p 个
    字符重复 n / (n - p) 次组成的。

    为什么成立：lps[n-1] = p 意味着 s 的前 p 个字符与后 p 个字符完全相同，也就是说串尾部
    整整重叠了 p 个字符，相当于把整体「向左平移」了 n - p 位后仍自洽。若 n - p 能整除 n，
    这种平移不产生错位，串就由长度 n - p 的块循环拼成。

    注意必须同时满足 p > 0 和 n % (n - p) == 0，且要求重复至少两次。若 p == 0，说明没有
    任何公共前后缀，n - p = n，n % n == 0 会误判（自己重复一次），所以要先用 p > 0 挡掉。

    另一条等价思路：把 s 拼成 s + s，去掉首尾各一个字符后在其中查找 s。若还能找到，说明
    s 有一个非自身的循环位移，即由更小的块重复而成。这里用前缀函数实现，和上一题共用模板。

复杂度：时间 O(n)，空间 O(n)（前缀函数数组）。
"""


def repeated_substring_pattern(s):
    n = len(s)
    if n == 0:
        return False

    lps = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and s[i] != s[k]:
            k = lps[k - 1]
        if s[i] == s[k]:
            k += 1
        lps[i] = k

    p = lps[-1]
    return p > 0 and n % (n - p) == 0


if __name__ == "__main__":
    assert repeated_substring_pattern("abab") is True
    assert repeated_substring_pattern("aba") is False
    assert repeated_substring_pattern("abcabcabcabc") is True
    assert repeated_substring_pattern("a") is False
    assert repeated_substring_pattern("aa") is True
    assert repeated_substring_pattern("abc") is False
    assert repeated_substring_pattern("abcab") is False

    print("repeated_substring_pattern: all tests passed")
