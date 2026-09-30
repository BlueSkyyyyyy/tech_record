// 1109. 航班预订统计（差分数组）
// 见 corporate_flight_bookings.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<long long> corpFlightBookings(const std::vector<std::vector<int>> &bookings,
                                          int n) {
    std::vector<long long> diff(n + 1, 0);
    for (const auto &b : bookings) {
        int first = b[0], last = b[1], seats = b[2];
        diff[first - 1] += seats;
        diff[last] -= seats;
    }
    std::vector<long long> res(n, 0);
    long long cur = 0;
    for (int i = 0; i < n; ++i) {
        cur += diff[i];
        res[i] = cur;
    }
    return res;
}

int main() {
    std::vector<long long> got =
        corpFlightBookings({{1, 2, 10}, {2, 3, 20}, {2, 5, 25}}, 5);
    std::vector<long long> want = {10, 55, 45, 25, 25};
    assert(got == want);
    std::vector<long long> empty_got = corpFlightBookings({}, 3);
    std::vector<long long> empty_want = {0, 0, 0};
    assert(empty_got == empty_want);
    std::vector<long long> one_got = corpFlightBookings({{1, 1, 7}}, 1);
    std::vector<long long> one_want = {7};
    assert(one_got == one_want);
    std::vector<long long> dup_got = corpFlightBookings({{2, 2, 5}, {2, 2, 3}}, 3);
    std::vector<long long> dup_want = {0, 8, 0};
    assert(dup_got == dup_want);
    std::cout << "corporate_flight_bookings: all tests passed\n";
    return 0;
}
