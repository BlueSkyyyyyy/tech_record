// 373. 查找和最小的 K 对数字
// 见 find_k_pairs_with_smallest_sums.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

std::vector<std::vector<int>> kSmallestPairs(const std::vector<int> &nums1,
                                             const std::vector<int> &nums2, int k) {
    std::vector<std::vector<int>> result;
    if (nums1.empty() || nums2.empty() || k <= 0) return result;

    using Item = std::tuple<int, int, int>;  // (和, i, j)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> minHeap;
    int rows = std::min((int)nums1.size(), k);
    for (int i = 0; i < rows; ++i) {
        minHeap.push({nums1[i] + nums2[0], i, 0});
    }

    while (!minHeap.empty() && (int)result.size() < k) {
        auto [sum, i, j] = minHeap.top();
        minHeap.pop();
        result.push_back({nums1[i], nums2[j]});
        if (j + 1 < (int)nums2.size()) {
            minHeap.push({nums1[i] + nums2[j + 1], i, j + 1});
        }
    }
    return result;
}

int main() {
    std::vector<std::vector<int>> want1 = {{1, 2}, {1, 4}, {1, 6}};
    assert(kSmallestPairs({1, 7, 11}, {2, 4, 6}, 3) == want1);

    std::vector<std::vector<int>> want2 = {{1, 1}, {1, 1}};
    assert(kSmallestPairs({1, 1, 2}, {1, 2, 3}, 2) == want2);

    std::vector<std::vector<int>> want3 = {{1, 3}, {2, 3}};
    assert(kSmallestPairs({1, 2}, {3}, 3) == want3);

    assert(kSmallestPairs({}, {1}, 1).empty());
    assert(kSmallestPairs({1}, {}, 1).empty());
    std::cout << "find_k_pairs_with_smallest_sums: all tests passed\n";
    return 0;
}
