// 1438. 绝对差不超过限制的最长连续子数组
// 见 longest_continuous_subarray_with_absolute_diff.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

int longestSubarray(std::vector<int>& nums, int limit) {
    std::deque<int> maxq, minq;
    int left = 0, ans = 0;
    int n = static_cast<int>(nums.size());
    for (int right = 0; right < n; ++right) {
        int x = nums[right];
        while (!maxq.empty() && maxq.back() < x) {
            maxq.pop_back();
        }
        maxq.push_back(x);
        while (!minq.empty() && minq.back() > x) {
            minq.pop_back();
        }
        minq.push_back(x);

        while (maxq.front() - minq.front() > limit) {
            if (maxq.front() == nums[left]) {
                maxq.pop_front();
            }
            if (minq.front() == nums[left]) {
                minq.pop_front();
            }
            ++left;
        }
        ans = std::max(ans, right - left + 1);
    }
    return ans;
}

int main() {
    std::vector<int> a{8, 2, 4, 7};
    std::vector<int> b{10, 1, 2, 4, 7, 2};
    std::vector<int> c{4, 2, 2, 2, 4, 4, 2, 2};
    std::vector<int> d{1};
    std::vector<int> e{1, 5};

    assert(longestSubarray(a, 4) == 2);
    assert(longestSubarray(b, 5) == 4);
    assert(longestSubarray(c, 0) == 3);
    assert(longestSubarray(d, 0) == 1);
    assert(longestSubarray(e, 3) == 1);
    std::cout << "longest_subarray: all tests passed\n";
    return 0;
}
