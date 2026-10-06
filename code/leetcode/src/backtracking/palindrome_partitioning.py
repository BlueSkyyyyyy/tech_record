"""131. 分割回文串（Palindrome Partitioning）

题目：给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文串。
返回 s 所有可能的分割方案。

思路（回溯切割 + 回文判定）：
    把「分割」理解为在字符串上画竖线：第一个切线切出前缀 s[start:end]，
    如果这段是回文，就把它作为方案的第一块，然后对剩下的后缀从 end 继续切。
    这正是一棵决策树，每层决定「下一刀切在哪」。

    参数用 start 表示「当前待切部分的起点」而不是数组下标：枚举 end 从
    start+1 到 n，取出子串 s[start:end]，是回文才递归。当 start == n 时说明
    整串切完，收集 path。

    为什么用 start 而不是「剩余字符串」：用下标区间能避免反复切片拷贝，
    也让「只往后切」天然成立，不会出现顺序不同但内容相同的重复方案。

复杂度：时间 O(n·2^n)（最坏每处可切可不切，共 2^(n-1) 种切法，判定回文 O(n)），
    空间 O(n)（递归深度，不含答案本身）。
"""


def partition(s):
    res = []
    path = []

    def is_palindrome(lo, hi):
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo += 1
            hi -= 1
        return True

    def backtrack(start):
        if start == len(s):
            res.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            if not is_palindrome(start, end - 1):
                continue
            path.append(s[start:end])
            backtrack(end)
            path.pop()

    backtrack(0)
    return res


if __name__ == "__main__":
    assert partition("aab") == [["a", "a", "b"], ["aa", "b"]]
    assert partition("a") == [["a"]]
    assert partition("aba") == [["a", "b", "a"], ["aba"]]
    assert partition("") == [[]]
    print("partition: all tests passed")
