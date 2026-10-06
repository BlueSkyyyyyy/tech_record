// 1122. 数组的相对排序
// 见 relative_sort_array.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

std::vector<int> relativeSortArray(const std::vector<int> &arr1,
                                   const std::vector<int> &arr2) {
    std::unordered_map<int, int> rank;
    for (int i = 0; i < static_cast<int>(arr2.size()); ++i) {
        rank[arr2[i]] = i;
    }
    int fallback = static_cast<int>(arr2.size());
    std::vector<int> res = arr1;
    std::sort(res.begin(), res.end(), [&](int a, int b) {
        int ra = rank.count(a) ? rank[a] : fallback;
        int rb = rank.count(b) ? rank[b] : fallback;
        if (ra != rb) {
            return ra < rb;
        }
        return a < b;
    });
    return res;
}

int main() {
    std::vector<int> a1 = {2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19};
    std::vector<int> a2 = {2, 1, 4, 3, 9, 6};
    std::vector<int> want = {2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19};
    assert(relativeSortArray(a1, a2) == want);

    std::vector<int> b1 = {28, 6, 22, 8, 44, 17};
    std::vector<int> b2 = {22, 28, 8, 6};
    std::vector<int> want2 = {22, 28, 8, 6, 17, 44};
    assert(relativeSortArray(b1, b2) == want2);

    std::vector<int> c1 = {1, 2, 3};
    std::vector<int> c2 = {};
    std::vector<int> want3 = {1, 2, 3};
    assert(relativeSortArray(c1, c2) == want3);

    std::cout << "relative_sort_array: all tests passed\n";
    return 0;
}
