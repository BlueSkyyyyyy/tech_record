// 692. 前 K 个高频单词
// 见 top_k_frequent_words.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::string> topKFrequent(std::vector<std::string> &words, int k) {
    std::unordered_map<std::string, int> count;
    for (const std::string &word : words) ++count[word];

    using P = std::pair<std::string, int>;  // (单词, 频率)
    auto better = [](const P &a, const P &b) {
        if (a.second != b.second) return a.second < b.second;  // 频率低的优先级低
        return a.first > b.first;                              // 同频字典序大的优先级低
    };
    std::priority_queue<P, std::vector<P>, decltype(better)> maxHeap(better);
    for (const auto &[word, freq] : count) maxHeap.push({word, freq});

    std::vector<std::string> result;
    for (int i = 0; i < k && !maxHeap.empty(); ++i) {
        result.push_back(maxHeap.top().first);
        maxHeap.pop();
    }
    return result;
}

int main() {
    std::vector<std::string> words = {"i", "love", "leetcode", "i", "love", "coding"};
    std::vector<std::string> want1 = {"i", "love"};
    assert(topKFrequent(words, 2) == want1);

    std::vector<std::string> want2 = {"i", "love", "coding"};
    assert(topKFrequent(words, 3) == want2);

    std::vector<std::string> words2 = {"the", "day", "is", "sunny", "the",
                                       "the", "the", "sunny", "is", "is"};
    std::vector<std::string> want3 = {"the", "is", "sunny", "day"};
    assert(topKFrequent(words2, 4) == want3);

    std::vector<std::string> words3 = {"b", "a", "b", "a", "c"};
    std::vector<std::string> want4 = {"a", "b"};
    assert(topKFrequent(words3, 2) == want4);
    std::cout << "top_k_frequent_words: all tests passed\n";
    return 0;
}
