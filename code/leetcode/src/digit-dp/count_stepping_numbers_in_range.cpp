// 2801. 统计范围内的步进数字数目
// 见 count_stepping_numbers_in_range.py 的题目与思路说明。
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

const long long MOD = 1000000007LL;
std::string s;
int length;
long long memo[105][11][2][2];

long long dfs(int pos, int prev, bool started, bool tight) {
    if (pos == length) {
        return started ? 1 : 0;
    }
    long long &res = memo[pos][prev + 1][started][tight];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, -1, false, ntight);
        } else if (started && d - prev != 1 && prev - d != 1) {
            continue;
        } else {
            total += dfs(pos + 1, d, true, ntight);
        }
    }
    return res = total % MOD;
}

long long countUpto(const std::string &x) {
    s = x;
    length = static_cast<int>(x.size());
    std::memset(memo, -1, sizeof(memo));
    return dfs(0, -1, false, true);
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

int countSteppingNumbers(std::string low, std::string high) {
    long long r = (countUpto(high) - countUpto(decrement(low))) % MOD;
    if (r < 0) {
        r += MOD;
    }
    return static_cast<int>(r);
}

int main() {
    assert(countSteppingNumbers("1", "11") == 10);
    assert(countSteppingNumbers("90", "101") == 2);
    assert(countSteppingNumbers("1", "10") == 10);
    assert(countSteppingNumbers("10", "10") == 1);
    assert(countSteppingNumbers("1", "100000000000000000000000000000") == 486147231);
    std::cout << "count_stepping_numbers_in_range: all tests passed\n";
    return 0;
}
