// 497. 非重叠矩形中的随机点
// 见 random_point_in_non_overlapping_rectangles.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

class Solution {
public:
    Solution(std::vector<std::vector<int>> rects) : rects_(rects) {
        int total = 0;
        for (const auto& r : rects_) {
            total += (r[2] - r[0] + 1) * (r[3] - r[1] + 1);
            prefix_.push_back(total);
        }
    }

    std::vector<int> pick() {
        int target = std::rand() % prefix_.back() + 1;
        int i = static_cast<int>(
            std::lower_bound(prefix_.begin(), prefix_.end(), target) -
            prefix_.begin());
        const auto& r = rects_[i];
        int x = r[0] + std::rand() % (r[2] - r[0] + 1);
        int y = r[1] + std::rand() % (r[3] - r[1] + 1);
        return {x, y};
    }

private:
    std::vector<std::vector<int>> rects_;
    std::vector<int> prefix_;
};

int main() {
    std::srand(12345);
    std::vector<std::vector<int>> rects = {{1, 1, 5, 5}, {-2, -2, 0, 0}};
    Solution s(rects);
    std::set<std::pair<int, int>> covered;
    for (int t = 0; t < 20000; ++t) {
        std::vector<int> p = s.pick();
        bool ok = false;
        for (const auto& r : rects) {
            if (r[0] <= p[0] && p[0] <= r[2] && r[1] <= p[1] && p[1] <= r[3]) {
                ok = true;
            }
        }
        assert(ok);
        covered.insert({p[0], p[1]});
    }
    assert(covered.size() == 34);

    std::cout << "random_point_in_non_overlapping_rectangles: all tests passed\n";
    return 0;
}
