// 349. 两个数组的交集（哈希集合去重）
// 见 intersection_of_two_arrays.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>

std::vector<int> intersection(const std::vector<int> &nums1,
                              const std::vector<int> &nums2) {
    std::unordered_set<int> set1(nums1.begin(), nums1.end());
    std::vector<int> res;
    for (int x : nums2) {
        auto it = set1.find(x);
        if (it != set1.end()) {
            res.push_back(x);
            set1.erase(it);
        }
    }
    return res;
}

int main() {
    auto sorted = [](std::vector<int> v) {
        std::sort(v.begin(), v.end());
        return v;
    };
    assert(sorted(intersection({1, 2, 2, 1}, {2, 2})) == std::vector<int>{2});
    assert(sorted(intersection({4, 9, 5}, {9, 4, 9, 8, 4})) ==
           (std::vector<int>{4, 9}));
    assert(intersection({1, 2, 3}, {4, 5, 6}).empty());
    assert(sorted(intersection({1, 1, 1}, {1, 1})) == std::vector<int>{1});
    std::cout << "intersection_of_two_arrays: all tests passed\n";
    return 0;
}
