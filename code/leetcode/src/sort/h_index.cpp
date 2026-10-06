// 274. H 指数
// 见 h_index.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int hIndex(std::vector<int> citations) {
    std::sort(citations.begin(), citations.end(), std::greater<int>());
    int h = 0;
    for (int i = 0; i < static_cast<int>(citations.size()); ++i) {
        if (citations[i] >= i + 1) {
            h = i + 1;
        } else {
            break;
        }
    }
    return h;
}

int hIndexCounting(const std::vector<int> &citations) {
    int n = static_cast<int>(citations.size());
    std::vector<int> bucket(n + 1, 0);
    for (int c : citations) {
        ++bucket[std::min(c, n)];
    }
    int acc = 0;
    for (int h = n; h >= 0; --h) {
        acc += bucket[h];
        if (acc >= h) {
            return h;
        }
    }
    return 0;
}

int main() {
    assert(hIndex({3, 0, 6, 1, 5}) == 3);
    assert(hIndex({1, 3, 1}) == 1);
    assert(hIndex({1, 3, 5, 7, 9}) == 3);
    assert(hIndex({0}) == 0);
    assert(hIndex({}) == 0);

    std::vector<int> c1 = {3, 0, 6, 1, 5};
    std::vector<int> c2 = {1, 3, 1};
    std::vector<int> c3 = {0};
    assert(hIndexCounting(c1) == 3);
    assert(hIndexCounting(c2) == 1);
    assert(hIndexCounting(c3) == 0);

    std::cout << "h_index: all tests passed\n";
    return 0;
}
