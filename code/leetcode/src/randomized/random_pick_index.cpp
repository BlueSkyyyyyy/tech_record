// 398. 随机数索引
// 见 random_pick_index.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

class Solution {
public:
    Solution(std::vector<int> nums) : nums_(nums) {}

    int pick(int target) {
        int res = -1;
        int count = 0;
        for (int i = 0; i < static_cast<int>(nums_.size()); ++i) {
            if (nums_[i] == target) {
                ++count;
                if (std::rand() % count == 0) {
                    res = i;
                }
            }
        }
        return res;
    }

private:
    std::vector<int> nums_;
};

int main() {
    std::srand(12345);
    Solution s({1, 2, 3, 3, 3, 2});
    std::set<int> seen;
    for (int t = 0; t < 4000; ++t) {
        int i = s.pick(3);
        assert(2 <= i && i <= 4);
        seen.insert(i);
    }
    assert(seen.size() == 3);
    assert(*seen.begin() >= 2 && *seen.rbegin() <= 4);

    std::set<int> seen2;
    for (int t = 0; t < 500; ++t) {
        seen2.insert(s.pick(2));
    }
    assert(seen2.size() == 2);

    std::cout << "random_pick_index: all tests passed\n";
    return 0;
}
