// 1011. 在 D 天内送达包裹的能力（二分答案）
// 见 capacity_to_ship_packages_within_d_days.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool canShip(const std::vector<int> &weights, int days, int cap) {
    int used = 1, cur = 0;
    for (int w : weights) {
        if (cur + w > cap) {
            used += 1;
            cur = 0;
        }
        cur += w;
    }
    return used <= days;
}

int shipWithinDays(const std::vector<int> &weights, int days) {
    int left = 0, right = 0;
    for (int w : weights) {
        left = std::max(left, w);
        right += w;
    }
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canShip(weights, days, mid))
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

int main() {
    assert(shipWithinDays({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5) == 15);
    assert(shipWithinDays({3, 2, 2, 4, 1, 4}, 3) == 6);
    assert(shipWithinDays({1, 2, 3, 1, 1}, 4) == 3);
    assert(shipWithinDays({10}, 1) == 10);
    assert(shipWithinDays({5, 5, 5}, 3) == 5);
    std::cout << "capacity_to_ship_packages_within_d_days: all tests passed\n";
    return 0;
}
