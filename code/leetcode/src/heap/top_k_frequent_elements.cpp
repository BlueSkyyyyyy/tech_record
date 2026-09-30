// 347. 前 K 个高频元素
// 见 top_k_frequent_elements.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

std::vector<int> topKFrequent(const std::vector<int> &nums, int k) {
    std::unordered_map<int, int> count;
    for (int num : nums) ++count[num];

    using P = std::pair<int, int>;  // (频率, 元素)
    std::priority_queue<P, std::vector<P>, std::greater<P>> minHeap;
    for (const auto &[num, freq] : count) {
        minHeap.push({freq, num});
        if ((int)minHeap.size() > k) minHeap.pop();
    }

    std::vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().second);
        minHeap.pop();
    }
    return result;
}

static std::vector<int> sorted(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    return v;
}

int main() {
    std::vector<int> want1 = {1, 2};
    assert(sorted(topKFrequent({1, 1, 1, 2, 2, 3}, 2)) == want1);
    assert(topKFrequent({1}, 1) == std::vector<int>{1});
    assert(topKFrequent({4, 4, 4, 4}, 1) == std::vector<int>{4});
    assert(sorted(topKFrequent({1, 2, 1, 2, 1, 2, 3, 3, 3, 3}, 1)) == std::vector<int>{3});
    std::cout << "top_k_frequent_elements: all tests passed\n";
    return 0;
}
