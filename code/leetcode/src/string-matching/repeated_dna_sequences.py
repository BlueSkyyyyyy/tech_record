"""187. 重复的 DNA 序列（Repeated DNA Sequences）

题目：DNA 序列由 'A''C''G''T' 组成。给定字符串 s，找出所有「长度恰为 10、出现次数超过
一次」的子串，按任意顺序返回。

思路（固定长度窗口 + 哈希集合）：
    长度固定为 10，直接用一个窗口从左到右滑动，把每个窗口的切片丢进「见过」的集合。
    如果某个切片之前见过、且还没被收进答案，就加入答案。用一个 `added` 集合去重，
    避免同一个重复子串被加入多次。

    当窗口更长、字符集更大时，切片本身要 O(L) 复制，代价偏高；更省的做法是滚动哈希：
    把窗口内字符按多项式编码成一个数，滑动时「减去最高位、整体乘底数、加上新字符」，
    O(1) 更新哈希，用哈希值去重即可。本题 L=10 很小，直接切片最直观。

复杂度：时间 O(n·L)（Python 切片复制），空间 O(n·L)（存切片）；用滚动哈希可降为
时间 O(n)、空间 O(n)。
"""


def find_repeated_dna_sequences(s):
    length = 10
    if len(s) < length + 1:
        return []
    seen = set()
    added = set()
    res = []
    for i in range(len(s) - length + 1):
        sub = s[i : i + length]
        if sub in seen and sub not in added:
            res.append(sub)
            added.add(sub)
        seen.add(sub)
    return res


if __name__ == "__main__":
    got = find_repeated_dna_sequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")
    assert got == ["AAAAACCCCC", "CCCCCAAAAA"]
    assert find_repeated_dna_sequences("AAAAAAAAAAAAA") == ["AAAAAAAAAA"]
    assert find_repeated_dna_sequences("ACGT") == []
    assert find_repeated_dna_sequences("") == []
    print("find_repeated_dna_sequences: all tests passed")
