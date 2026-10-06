"""763. 划分字母区间（Partition Labels）

题目：给你一个字符串 s。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个
片段中。返回一个表示每个字符串片段长度的列表。划分得到的片段要能按顺序拼回原字符串。

思路（贪心：用「每个字母最后出现的位置」决定当前片段能切到哪）：
    同一个字母不能跨片段，所以一个片段一旦包含了某个字母，就必须一直延伸到该字母最后
    一次出现的位置。据此：
      1. 先扫一遍字符串，记录每个字母最后出现的下标 last[c]；
      2. 再从左往右扫，用 start 标记当前片段起点、end 标记当前片段必须延伸到的最远下标：
         - 每读到一个字母，就把它最后出现的下标并入 end（end = max(end, last[c])）；
         - 当扫描下标 i 正好等于 end，说明当前片段里所有字母都不会再往右出现了，可以在此
           切一刀，记录这一段的长度 end - start + 1，并把 start 移到 i + 1。

    为什么这样能切出「尽可能多」的片段：一个片段的结束位置存在硬性下界——必须覆盖其中所有
    字母的最后出现位置。我们每次都在「满足约束的前提下」最早的位置切分（当 i == end 就切），
    这样切出来的片段尽量短、数量尽量多，且不会让任何字母跨越切点。

复杂度：时间 O(n)（两趟线性扫描，字母表大小可视为常数），空间 O(1)
    （用大小为 26 的数组存最后位置，不计输入）。
"""


def partition_labels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = 0
    end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result


if __name__ == "__main__":
    assert partition_labels("ababcbacadefegdehijhklij") == [9, 7, 8]
    assert partition_labels("eccbbbbdec") == [10]
    assert partition_labels("a") == [1]
    assert partition_labels("abc") == [1, 1, 1]
    print("partition_labels: all tests passed")
