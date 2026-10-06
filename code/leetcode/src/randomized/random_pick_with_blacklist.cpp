// 710. 黑名单中的随机数
// 见 random_pick_with_blacklist.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Solution {
public:
    Solution(int n, std::vector<int> blacklist) {
        int m = static_cast<int>(blacklist.size());
        size_ = n - m;
        std::unordered_set<int> blocked(blacklist.begin(), blacklist.end());
        int last = n - 1;
        for (int b : blacklist) {
            if (b < size_) {
                while (blocked.count(last)) {
                    --last;
                }
                mapping_[b] = last;
                --last;
            }
        }
    }

    int pick() {
        int idx = std::rand() % size_;
        auto it = mapping_.find(idx);
        return it == mapping_.end() ? idx : it->second;
    }

private:
    int size_;
    std::unordered_map<int, int> mapping_;
};

int main() {
    std::srand(12345);
    Solution s(7, {2, 3, 5});
    std::unordered_set<int> blocked = {2, 3, 5};
    std::unordered_set<int> seen;
    for (int t = 0; t < 4000; ++t) {
        int x = s.pick();
        assert(0 <= x && x < 7 && !blocked.count(x));
        seen.insert(x);
    }
    assert(seen.size() == 4);

    Solution s2(5, {3, 4});
    std::unordered_set<int> seen2;
    for (int t = 0; t < 2000; ++t) {
        seen2.insert(s2.pick());
    }
    assert(seen2.size() == 3);

    std::cout << "random_pick_with_blacklist: all tests passed\n";
    return 0;
}
