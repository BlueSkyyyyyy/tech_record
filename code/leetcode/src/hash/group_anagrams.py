"""49. 字母异位词分组（Group Anagrams）

题目：给定一个字符串数组 strs，把互为字母异位词（字母及数量完全相同、只是顺序不同）
的字符串分到同一组，返回所有分组（组内顺序与组间顺序均不限）。

思路：字母异位词是同一组「字母多重集合」的不同排列，所以只要给每个字符串找一个
    稳定且与顺序无关的「指纹」，指纹相同的就是一组。常用两种指纹：
      - 排序后的字符串：把字符排序，异位词必然得到同一个结果；
      - 字符计数数组（长度 26）：统计每个字母出现次数，拼成 key。
    这里用排序版，代码最短、可读性最好，代价是每个字符串排序 O(k log k)。

复杂度：设字符串长度 k、个数 n。时间 O(n * k log k)，空间 O(n * k)。
"""


def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = "".join(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())


if __name__ == "__main__":
    got = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    got = sorted(sorted(g) for g in got)
    assert got == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([""]) == [[""]]
    assert group_anagrams(["a"]) == [["a"]]
    print("group_anagrams: all tests passed")
