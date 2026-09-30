// 1094. 拼车（差分数组）
// 见 car_pooling.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool carPooling(const std::vector<std::vector<int>> &trips, int capacity) {
    int size = 0;
    for (const auto &t : trips)
        if (t[2] > size) size = t[2];
    std::vector<long long> diff(size + 1, 0);
    for (const auto &t : trips) {
        int num = t[0], start = t[1], end = t[2];
        diff[start] += num;
        diff[end] -= num;
    }
    long long cur = 0;
    for (int i = 0; i < size; ++i) {
        cur += diff[i];
        if (cur > capacity) return false;
    }
    return true;
}

int main() {
    assert(carPooling({{2, 1, 5}, {3, 3, 7}}, 4) == false);
    assert(carPooling({{2, 1, 5}, {3, 3, 7}}, 5) == true);
    assert(carPooling({{2, 1, 5}, {3, 5, 7}}, 5) == true);
    assert(carPooling({}, 1) == true);
    assert(carPooling({{9, 0, 1}}, 8) == false);
    assert(carPooling({{3, 2, 4}, {2, 4, 6}}, 5) == true);
    std::cout << "car_pooling: all tests passed\n";
    return 0;
}
