// 455. 分发饼干
// 见 assign_cookies.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int findContentChildren(std::vector<int> g, std::vector<int> s) {
    std::sort(g.begin(), g.end());
    std::sort(s.begin(), s.end());
    int i = 0, j = 0;
    while (i < static_cast<int>(g.size()) && j < static_cast<int>(s.size())) {
        if (s[j] >= g[i]) {
            ++i;
            ++j;
        } else {
            ++j;
        }
    }
    return i;
}

int main() {
    std::vector<int> g1 = {1, 2, 3}, s1 = {1, 1};
    assert(findContentChildren(g1, s1) == 1);
    std::vector<int> g2 = {1, 2}, s2 = {1, 2, 3};
    assert(findContentChildren(g2, s2) == 2);
    std::vector<int> g3 = {1, 2, 3}, s3 = {3};
    assert(findContentChildren(g3, s3) == 1);
    std::vector<int> g4 = {10, 9, 8, 7}, s4 = {5, 6, 7, 8};
    assert(findContentChildren(g4, s4) == 2);
    std::vector<int> g5, s5 = {1, 2};
    assert(findContentChildren(g5, s5) == 0);
    std::vector<int> g6 = {1, 2}, s6;
    assert(findContentChildren(g6, s6) == 0);
    std::cout << "assign_cookies: all tests passed\n";
    return 0;
}
