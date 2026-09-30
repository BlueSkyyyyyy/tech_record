// 875. 爱吃香蕉的珂珂（二分答案）
// 见 koko_eating_bananas.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool canFinish(const std::vector<int> &piles, int h, int k) {
    long long hours = 0;
    for (int pile : piles) hours += (pile + k - 1) / k;
    return hours <= h;
}

int minEatingSpeed(const std::vector<int> &piles, int h) {
    int left = 1, right = 0;
    for (int pile : piles) right = std::max(right, pile);
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canFinish(piles, h, mid))
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

int main() {
    assert(minEatingSpeed({3, 6, 7, 11}, 8) == 4);
    assert(minEatingSpeed({30, 11, 23, 4, 20}, 5) == 30);
    assert(minEatingSpeed({30, 11, 23, 4, 20}, 6) == 23);
    assert(minEatingSpeed({1}, 1) == 1);
    assert(minEatingSpeed({312884470}, 312884469) == 2);
    assert(minEatingSpeed({1000000000}, 2) == 500000000);
    std::cout << "koko_eating_bananas: all tests passed\n";
    return 0;
}
