// 1272. 删除区间
// 见 remove_interval.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> removeInterval(
    std::vector<std::vector<int>>& intervals, std::vector<int>& toBeRemoved) {
    int lo = toBeRemoved[0], hi = toBeRemoved[1];
    std::vector<std::vector<int>> res;
    for (const auto& iv : intervals) {
        int a = iv[0], b = iv[1];
        if (b <= lo || a >= hi) {
            res.push_back({a, b});
        } else {
            if (a < lo) {
                res.push_back({a, lo});
            }
            if (b > hi) {
                res.push_back({hi, b});
            }
        }
    }
    return res;
}

int main() {
    std::vector<std::vector<int>> a{{0, 2}, {3, 4}, {5, 7}};
    std::vector<int> ra{1, 6};
    std::vector<std::vector<int>> wa{{0, 1}, {6, 7}};

    std::vector<std::vector<int>> b{{0, 5}};
    std::vector<int> rb{1, 3};
    std::vector<std::vector<int>> wb{{0, 1}, {3, 5}};

    std::vector<std::vector<int>> c{{0, 5}};
    std::vector<int> rc{-5, -1};
    std::vector<std::vector<int>> wc{{0, 5}};

    assert(removeInterval(a, ra) == wa);
    assert(removeInterval(b, rb) == wb);
    assert(removeInterval(c, rc) == wc);

    std::cout << "remove_interval: all tests passed\n";
    return 0;
}
