"""692. 前 K 个高频单词（Top K Frequent Words）

题目：给一个单词列表 words 和整数 k，返回出现频率前 k 高的单词。
      返回顺序按频率从高到低；频率相同时，按字典序从小到大排列。

思路（哈希计数 + 带自定义优先级的堆）：
    先用哈希表统计每个单词的出现次数，得到一批 (单词, 频率)。
    排序规则是「频率高的在前；频率相同则字典序小的在前」。
    把每个单词打包成 (-频率, 单词) 放进小顶堆，然后依次弹出 k 个：
      - 按 -频率 从小到大，正好是频率从大到小；
      - 频率相同时按单词从小到大，正好是字典序。
    所以堆的弹出顺序恰好就是题目要求的顺序，弹 k 次即得答案。

    为什么用 -频率 而不是频率：
    Python 的 heapq 是小顶堆，弹出的是最小元素。
    频率越大越靠前，就取相反数让「大频率」对应「更小的键」，
    弹出来自然是从高频到低频。

    顺带说明 C++ 的做法：C++ 的 priority_queue 可以直接传一个比较器，
    把「频率高优先、同频字典序小优先」写成偏序关系，堆顶就是最该输出的单词。

    另一种做法是「大小为 k 的小顶堆淘汰线」：维护 k 个最优，
    堆顶放「最差」的那个（频率最低，同频字典序最大）。Python 里要自定义
    比较器稍麻烦，本篇直接用全量堆 + 弹 k 次，代码更短且同样高效。

复杂度：时间 O(m + k log m)（m 为不同单词个数），空间 O(m)。
"""

import heapq


def top_k_frequent_words(words, k):
    count = {}
    for word in words:
        count[word] = count.get(word, 0) + 1

    heap = [(-freq, word) for word, freq in count.items()]
    heapq.heapify(heap)

    return [heapq.heappop(heap)[1] for _ in range(k)]


if __name__ == "__main__":
    words = ["i", "love", "leetcode", "i", "love", "coding"]
    assert top_k_frequent_words(words, 2) == ["i", "love"]
    assert top_k_frequent_words(words, 3) == ["i", "love", "coding"]

    words2 = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
    assert top_k_frequent_words(words2, 4) == ["the", "is", "sunny", "day"]

    assert top_k_frequent_words(["a"], 1) == ["a"]

    words3 = ["b", "a", "b", "a", "c"]
    assert top_k_frequent_words(words3, 2) == ["a", "b"]

    words4 = ["a", "aa", "aaa"]
    assert top_k_frequent_words(words4, 3) == ["a", "aa", "aaa"]
    print("top_k_frequent_words: all tests passed")
