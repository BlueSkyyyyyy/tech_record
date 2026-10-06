// 384. 打乱数组
// 见 shuffle_an_array.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

class Solution {
public:
    Solution(std::vector<int> nums) : original_(nums), nums_(nums) {}

    std::vector<int> reset() {
        nums_ = original_;
        return nums_;
    }

    std::vector<int> shuffle() {
        for (int i = static_cast<int>(nums_.size()) - 1; i > 0; --i) {
            int j = std::rand() % (i + 1);
            std::swap(nums_[i], nums_[j]);
        }
        return nums_;
    }

private:
    std::vector<int> original_;
    std::vector<int> nums_;
};

int main() {
    std::srand(12345);
    std::vector<int> nums = {1, 2, 3, 4, 5};
    Solution s(nums);
    std::set<std::vector<int>> seen;
    for (int t = 0; t < 500; ++t) {
        std::vector<int> got = s.shuffle();
        std::vector<int> sorted = got;
        std::sort(sorted.begin(), sorted.end());
        assert(sorted == nums);
        seen.insert(got);
        assert(s.reset() == nums);
    }
    assert(seen.size() > 100);

    std::cout << "shuffle_an_array: all tests passed\n";
    return 0;
}
