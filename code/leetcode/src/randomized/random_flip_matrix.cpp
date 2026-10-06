// 519. 随机翻转矩阵
// 见 random_flip_matrix.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

class Solution {
public:
    Solution(int m, int n) : m_(m), n_(n), total_(m * n) {}

    std::vector<int> flip() {
        --total_;
        int idx = std::rand() % (total_ + 1);
        auto it = mapping_.find(idx);
        int chosen = it == mapping_.end() ? idx : it->second;
        auto it2 = mapping_.find(total_);
        mapping_[idx] = it2 == mapping_.end() ? total_ : it2->second;
        return {chosen / n_, chosen % n_};
    }

    void reset() {
        total_ = m_ * n_;
        mapping_.clear();
    }

private:
    int m_;
    int n_;
    int total_;
    std::unordered_map<int, int> mapping_;
};

int main() {
    std::srand(12345);
    Solution s(3, 4);
    std::set<std::pair<int, int>> got;
    for (int t = 0; t < 12; ++t) {
        std::vector<int> p = s.flip();
        assert(0 <= p[0] && p[0] < 3 && 0 <= p[1] && p[1] < 4);
        got.insert({p[0], p[1]});
    }
    assert(got.size() == 12);

    s.reset();
    std::set<std::pair<int, int>> got2;
    for (int t = 0; t < 12; ++t) {
        std::vector<int> p = s.flip();
        got2.insert({p[0], p[1]});
    }
    assert(got2.size() == 12);

    std::cout << "random_flip_matrix: all tests passed\n";
    return 0;
}
