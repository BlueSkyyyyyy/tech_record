// 1215. 步进数
// 见 stepping_numbers.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <vector>

std::vector<int> steppingNumbers(int low, int high) {
    std::vector<int> result;
    if (low <= 0 && 0 <= high) {
        result.push_back(0);
    }
    std::queue<long long> q;
    for (int i = 1; i <= 9; ++i) {
        q.push(i);
    }
    while (!q.empty()) {
        long long x = q.front();
        q.pop();
        if (x > high) {
            continue;
        }
        if (x >= low) {
            result.push_back(static_cast<int>(x));
        }
        int last = static_cast<int>(x % 10);
        if (last > 0) {
            q.push(x * 10 + last - 1);
        }
        if (last < 9) {
            q.push(x * 10 + last + 1);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

int main() {
    std::vector<int> want1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21};
    assert(steppingNumbers(0, 21) == want1);
    std::vector<int> want2 = {10, 12};
    assert(steppingNumbers(10, 15) == want2);
    std::vector<int> want3 = {};
    assert(steppingNumbers(100, 100) == want3);
    std::vector<int> want4 = {1};
    assert(steppingNumbers(1, 1) == want4);
    std::cout << "stepping_numbers: all tests passed\n";
    return 0;
}
