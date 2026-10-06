// 1636. 按照频率将数组升序排序
// 见 frequency_sort.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

std::vector<int> frequencySort(const std::vector<int> &nums) {
    std::unordered_map<int, int> cnt;
    for (int x : nums) {
        ++cnt[x];
    }
    std::vector<int> res = nums;
    std::sort(res.begin(), res.end(), [&](int a, int b) {
        if (cnt[a] != cnt[b]) {
            return cnt[a] < cnt[b];
        }
        return a > b;
    });
    return res;
}

int main() {
    std::vector<int> want1 = {3, 1, 1, 2, 2, 2};
    assert(frequencySort({1, 1, 2, 2, 2, 3}) == want1);

    std::vector<int> want2 = {1, 3, 3, 2, 2};
    assert(frequencySort({2, 3, 1, 3, 2}) == want2);

    std::vector<int> want3 = {5, -1, 4, 4, -6, -6, 1, 1, 1};
    assert(frequencySort({-1, 1, -6, 4, 5, -6, 1, 4, 1}) == want3);

    std::vector<int> want4 = {7};
    assert(frequencySort({7}) == want4);

    std::cout << "frequency_sort: all tests passed\n";
    return 0;
}
