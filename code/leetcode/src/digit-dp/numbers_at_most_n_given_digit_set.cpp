// 902. 最大为 N 的数字组合
// 见 numbers_at_most_n_given_digit_set.py 的题目与思路说明。
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

std::string s;
int length;
std::vector<int> nums;
long long memo[12][2];

long long dfs(int pos, bool tight) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][tight];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long count = 0;
    for (int d : nums) {
        if (d > limit) {
            break;
        }
        count += dfs(pos + 1, tight && d == limit);
    }
    return res = count;
}

long long atMostNGivenDigitSet(std::vector<std::string> &digits, int n) {
    nums.clear();
    for (const std::string &c : digits) {
        nums.push_back(c[0] - '0');
    }
    s = std::to_string(n);
    length = static_cast<int>(s.size());
    long long total = 0;
    for (int size = 1; size < length; ++size) {
        long long power = 1;
        for (int i = 0; i < size; ++i) {
            power *= static_cast<long long>(nums.size());
        }
        total += power;
    }
    std::memset(memo, -1, sizeof(memo));
    return total + dfs(0, true);
}

int main() {
    std::vector<std::string> d1 = {"1", "3", "5", "7"};
    assert(atMostNGivenDigitSet(d1, 100) == 20);
    assert(atMostNGivenDigitSet(d1, 1) == 1);
    std::vector<std::string> d2 = {"1", "4", "9"};
    assert(atMostNGivenDigitSet(d2, 1000000000) == 29523);
    std::vector<std::string> d3 = {"7"};
    assert(atMostNGivenDigitSet(d3, 8) == 1);
    std::cout << "numbers_at_most_n_given_digit_set: all tests passed\n";
    return 0;
}
