// 202. 快乐数（哈希集合判环）
// 见 happy_number.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_set>

int nextNumber(int n) {
    int total = 0;
    while (n > 0) {
        int d = n % 10;
        total += d * d;
        n /= 10;
    }
    return total;
}

bool isHappy(int n) {
    std::unordered_set<int> seen;
    while (n != 1 && seen.find(n) == seen.end()) {
        seen.insert(n);
        n = nextNumber(n);
    }
    return n == 1;
}

int main() {
    assert(isHappy(19) == true);
    assert(isHappy(1) == true);
    assert(isHappy(2) == false);
    assert(isHappy(7) == true);
    assert(isHappy(100) == true);
    std::cout << "happy_number: all tests passed\n";
    return 0;
}
