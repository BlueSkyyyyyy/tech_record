"""691. 贴纸拼词（Stickers to Spell Word）

题目：给定若干贴纸 stickers 和一个目标单词 target。每张贴纸上的字符可以被剪下来
使用（一张贴纸里的每个字符最多用一次），贴纸可以重复购买，求拼出 target 所需的最
少贴纸数；无法拼出返回 -1。

思路（状态 = "还差哪些位置的字符"）：
    直接枚举"用哪些贴纸、各用几张"会组合爆炸。换一个视角：target 每个位置只有
    "已拼好 / 还没拼好"两种状态，一共 2^|target| 种。用一个掩码 remaining 表示"还没
    拼好的位置"（或反过来用 covered 表示已拼好），状态空间就不超过 2^15。

    每一次操作是"再买一张贴纸"：拿这张贴纸的字母，从左到右尽量去补 remaining 里还没
    拼好的位置，得到一个新的 remaining，代价 +1。这就是在状态图上做搜索；用记忆化
    求最少步数。

    剪枝的关键：如果这张贴纸一个字符都补不上（新旧 remaining 相同），这次选择没有
    任何进展，直接跳过。

复杂度：时间 O(2^T * n * T)（T = len(target)，n = len(stickers)），空间 O(2^T)。
"""

from functools import lru_cache


def min_stickers(stickers, target):
    T = len(target)

    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return 0
        best = float("inf")
        for sticker in stickers:
            cnt = [0] * 26
            for ch in sticker:
                cnt[ord(ch) - 97] += 1
            nxt = remaining
            for i in range(T):
                if nxt >> i & 1:
                    idx = ord(target[i]) - 97
                    if cnt[idx] > 0:
                        cnt[idx] -= 1
                        nxt ^= 1 << i  # 这一位拼好了，清掉
            if nxt != remaining:  # 本贴纸至少有贡献
                best = min(best, 1 + solve(nxt))
        return best

    ans = solve((1 << T) - 1)
    return -1 if ans == float("inf") else ans


if __name__ == "__main__":
    assert min_stickers(["with", "example", "science"], "thehat") == 3
    assert min_stickers(["notice", "possible"], "basicbasic") == -1
    assert min_stickers(["a"], "a") == 1
    assert min_stickers(["ab", "bc", "cd"], "abcd") == 2
    print("min_stickers: all tests passed")
