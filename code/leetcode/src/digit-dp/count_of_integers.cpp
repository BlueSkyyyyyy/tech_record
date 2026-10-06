// 2719. 统计整数数目
// 见 count_of_integers.py 的题目与思路说明。
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

const long long MOD = 1000000007LL;
std::string s;
int length;
int minSum;
int maxSum;
long long memo[25][405][2][2];

long long dfs(int pos, int sum, bool tight, bool started) {
    if (pos == length) {
        return (started && sum >= minSum && sum <= maxSum) ? 1 : 0;
    }
    long long &res = memo[pos][sum][tight][started];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, sum, ntight, false);
        } else {
            if (sum + d > maxSum) {
                continue;
            }
            total += dfs(pos + 1, sum + d, ntight, true);
        }
    }
    return res = total % MOD;
}

long long countUpto(const std::string &x) {
    s = x;
    length = static_cast<int>(x.size());
    std::memset(memo, -1, sizeof(memo));
    return dfs(0, 0, true, false);
}

std::string decrement(std::string a) {
    int i = static_cast<int>(a.size()) - 1;
    while (a[i] == '0') {
        a[i] = '9';
        --i;
    }
    a[i] = static_cast<char>(a[i] - 1);
    std::size_t start = a.find_first_not_of('0');
    if (start == std::string::npos) {
        return "0";
    }
    return a.substr(start);
}

int countOfIntegers(std::string num1, std::string num2, int _minSum, int _maxSum) {
    minSum = _minSum;
    maxSum = _maxSum;
    long long r = (countUpto(num2) - countUpto(decrement(num1))) % MOD;
    if (r < 0) {
        r += MOD;
    }
    return static_cast<int>(r);
}

int main() {
    assert(countOfIntegers("1", "12", 1, 8) == 11);
    assert(countOfIntegers("1", "5", 1, 5) == 5);
    assert(countOfIntegers("1", "2026", 2, 5) == 95);
    std::cout << "count_of_integers: all tests passed\n";
    return 0;
}
