"""451. 根据字符出现频率排序（Sort Characters By Frequency）

题目：给你一个字符串 s，请按字符出现的频率降序重新排列，返回排序后的字符串。
      频率相同的字符，顺序不限（只要结果合法）。

思路（先计数，再按频率重新拼字符串）：
    目标是「频率高的字符排前面」，而字符本身要重复它出现的次数。所以先用计数器
    统计每个字符出现几次；再把「字符 → 次数」这些项按次数从大到小排序；最后把每项
    展开成「该字符重复次数遍」，拼起来即可。

    为什么不用对原字符串直接排序：原字符串里每个字符已经出现应有的次数，若排序键
    是「该字符的总频率」，所有相同字符的键一致，能保持连续，但稳定排序后同频字符
    的相对次序取决于输入，不便于确定结果；直接按 `(字符 → 次数)` 构造更清晰，也能
    顺带决定同频字符的顺序。

    更快的写法是桶排序：频率最大不超过字符串长度 n，开 n+1 个桶，把字符按键入频率
    放进对应桶，再从高频率往低频率输出，时间 O(n)。当字符集只是小写/大写字母时，
    计数 + 桶是最优解。

    本实现按 `(-频率, 字符)` 排序以保证结果确定。
"""


def frequency_sort_string(s):
    from collections import Counter

    cnt = Counter(s)
    parts = [ch * c for ch, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "".join(parts)


def frequency_sort_string_bucket(s):
    from collections import Counter

    if not s:
        return ""
    cnt = Counter(s)
    buckets = [[] for _ in range(len(s) + 1)]
    for ch, c in cnt.items():
        buckets[c].append(ch)
    res = []
    for c in range(len(s), 0, -1):
        for ch in sorted(buckets[c]):
            res.append(ch * c)
    return "".join(res)


if __name__ == "__main__":
    assert frequency_sort_string("tree") == "eert"
    assert frequency_sort_string("cccaaa") == "aaaccc"
    assert frequency_sort_string("Aabb") == "bbAa"
    assert frequency_sort_string("") == ""
    assert frequency_sort_string_bucket("tree") == "eert"
    assert frequency_sort_string_bucket("cccaaa") == "aaaccc"
    assert frequency_sort_string_bucket("Aabb") == "bbAa"
    print("frequency_sort_string: all tests passed")
