// 862. 和至少为 K 的最短子数组
// 见 shortest_subarray_with_sum_at_least_k.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

int shortestSubarray(std::vector<int>& nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + nums[i];
    }

    int ans = n + 1;
    std::deque<int> dq;
    for (int j = 0; j <= n; ++j) {
        while (!dq.empty() && prefix[j] - prefix[dq.front()] >= k) {
            ans = std::min(ans, j - dq.front());
            dq.pop_front();
        }
        while (!dq.empty() && prefix[dq.back()] >= prefix[j]) {
            dq.pop_back();
        }
        dq.push_back(j);
    }
    return ans <= n ? ans : -1;
}

int main() {
    std::vector<int> a{1};
    std::vector<int> b{1, 2};
    std::vector<int> c{2, -1, 2};
    std::vector<int> d{1, 2};
    std::vector<int> e{2, 1, 2};
    std::vector<int> f{84, -37, 32, 40, 95};

    assert(shortestSubarray(a, 1) == 1);
    assert(shortestSubarray(b, 4) == -1);
    assert(shortestSubarray(c, 3) == 3);
    assert(shortestSubarray(d, 3) == 2);
    assert(shortestSubarray(e, 4) == 3);
    assert(shortestSubarray(f, 167) == 3);
    std::cout << "shortest_subarray: all tests passed\n";
    return 0;
}
