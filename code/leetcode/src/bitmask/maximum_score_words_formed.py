"""1255. 得分最高的单词集合（Maximum Score Words Formed by Letters）

题目：给定单词列表 words、可用字母列表 letters（可重复）和每个字母的分值 score，
用 letters 里的字母拼出 words 的一个子集，每个字母使用次数不能超过它在 letters 里
出现的次数，求能得到的最大总分。

思路（字母计数 + 掩码枚举子集）：
    和 1239 一样是"选一个子集"，只是约束从"字符不重复"变成"不超出每个字母的供给"。
    先统计 letters 的字母库存 have[26]，把每个单词压成它的字母计数向量，并预先算好
    它的得分（如果某字母本身超出库存，这个单词永远不可用，得分为 0 且标记不可选）。

    然后枚举所有子集 mask，逐个子集累加字母用量并校验是否超过库存。n ≤ 14，2^n 枚举
    完全够用；用掩码枚举子集，每一格只做"选/不选"的判断。

复杂度：时间 O(2^n * n * 26)，空间 O(n * 26)。
"""


def max_score_words(words, letters, score):
    have = [0] * 26
    for ch in letters:
        have[ord(ch) - 97] += 1

    n = len(words)
    word_count = []
    word_score = []
    for w in words:
        cnt = [0] * 26
        ok = True
        for ch in w:
            idx = ord(ch) - 97
            cnt[idx] += 1
            if cnt[idx] > have[idx]:
                ok = False
        word_count.append(cnt)
        word_score.append(sum(score[ord(ch) - 97] for ch in w) if ok else 0)

    best = 0
    for mask in range(1 << n):
        used = [0] * 26
        total = 0
        ok = True
        for i in range(n):
            if mask >> i & 1:
                for k in range(26):
                    used[k] += word_count[i][k]
                    if used[k] > have[k]:
                        ok = False
                        break
                if not ok:
                    break
                total += word_score[i]
        if ok and total > best:
            best = total
    return best


if __name__ == "__main__":
    score1 = [1, 0, 9, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert max_score_words(["dog", "cat", "dad", "good"],
                           ["a", "a", "c", "d", "d", "d", "g", "o", "o"], score1) == 23

    score2 = [1, 10, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert max_score_words(["a", "b", "c"], ["a", "a", "b", "c"], score2) == 111
    assert max_score_words(["a", "aa", "aaa"], ["a", "a", "a", "a"], score2) == 4

    score3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert max_score_words(["leetcode"], ["l", "e", "t", "c", "o", "d"], score3) == 0
    print("max_score_words: all tests passed")
