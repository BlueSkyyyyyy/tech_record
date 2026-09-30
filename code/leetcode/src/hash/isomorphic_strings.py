"""205. 同构字符串（Isomorphic Strings）

题目：给定两个字符串 s 和 t，判断它们是否同构。同构指 s 中的每个字符都能被
唯一替换成 t 中对应的字符，反之亦然，且字符的相对顺序不变。
例如 egg 与 add 同构（e->a, g->d），foo 与 bar 不同构。

思路：「同构」要求字符之间的映射是**双向唯一**的一一对应：
       - 同一个 s 字符不能映射到两个不同的 t 字符（正向唯一）；
       - 两个不同的 s 字符不能映射到同一个 t 字符（反向唯一）。
     所以维护两张哈希表：s->t 和 t->s。
     逐位检查：
       - 若 s[i] 已建立映射，必须等于 t[i]，否则不同构；
       - 若 t[i] 已被别的 s 字符占用，也不同构；
       - 否则建立双向映射。
     只维护一张表的反例：s="ab", t="aa" 中 a->a 合法，但 b 也想映射到 a，
     正向表发现不了冲突，必须靠反向表拦住。

复杂度：时间 O(n)，空间 O(字符集大小)。
"""


def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    forward = {}
    backward = {}
    for a, b in zip(s, t):
        if forward.get(a, b) != b or backward.get(b, a) != a:
            return False
        forward[a] = b
        backward[b] = a
    return True


if __name__ == "__main__":
    assert is_isomorphic("egg", "add") is True
    assert is_isomorphic("foo", "bar") is False
    assert is_isomorphic("paper", "title") is True
    assert is_isomorphic("ab", "aa") is False
    assert is_isomorphic("", "") is True
    print("isomorphic_strings: all tests passed")
