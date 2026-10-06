// 528. 按权重随机选择
// 见 random_pick_with_weight.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

class Solution {
public:
    Solution(std::vector<int> w) {
        int total = 0;
        for (int x : w) {
            total += x;
            prefix_.push_back(total);
        }
    }

    int pickIndex() {
        int target = std::rand() % prefix_.back() + 1;
        return static_cast<int>(
            std::lower_bound(prefix_.begin(), prefix_.end(), target) -
            prefix_.begin());
    }

private:
    std::vector<int> prefix_;
};

int main() {
    std::srand(12345);
    Solution s({1, 2, 3});
    int counts[3] = {0, 0, 0};
    const int N = 60000;
    for (int t = 0; t < N; ++t) {
        int i = s.pickIndex();
        assert(0 <= i && i < 3);
        ++counts[i];
    }
    assert(counts[0] < counts[1] && counts[1] < counts[2]);
    assert(counts[2] > N * 45 / 100);

    std::cout << "random_pick_with_weight: all tests passed\n";
    return 0;
}
