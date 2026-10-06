// 2407. 最长递增子序列 II
// 见 longest_increasing_subsequence_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int lengthOfLIS(const std::vector<int>& nums, int k) {
    int maxV = *std::max_element(nums.begin(), nums.end());
    int size = maxV + 1;
    std::vector<int> tree(2 * size, 0);
    auto update = [&](int pos, int val) {
        int i = pos + size;
        if (tree[i] >= val) return;
        tree[i] = val;
        for (i /= 2; i > 0; i /= 2) tree[i] = std::max(tree[2 * i], tree[2 * i + 1]);
    };
    auto query = [&](int lo, int hi) {
        int res = 0;
        for (int l = lo + size, r = hi + size + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) res = std::max(res, tree[l++]);
            if (r & 1) res = std::max(res, tree[--r]);
        }
        return res;
    };
    int ans = 0;
    for (int v : nums) {
        int lo = std::max(0, v - k), hi = v - 1;
        int best = (lo <= hi) ? query(lo, hi) : 0;
        int cur = best + 1;
        update(v, cur);
        ans = std::max(ans, cur);
    }
    return ans;
}

int main() {
    assert(lengthOfLIS(std::vector<int>{4, 2, 1, 4, 3, 4, 5, 8, 7}, 3) == 5);
    assert(lengthOfLIS(std::vector<int>{1, 5, 4, 2, 3}, 2) == 3);
    assert(lengthOfLIS(std::vector<int>{7, 7, 7, 7}, 1) == 1);
    assert(lengthOfLIS(std::vector<int>{1, 2, 3, 4, 5}, 0) == 1);
    std::cout << "longest_increasing_subsequence_ii: all tests passed\n";
    return 0;
}
