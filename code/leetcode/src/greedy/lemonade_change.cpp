// 860. 柠檬水找零
// 见 lemonade_change.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool lemonadeChange(const std::vector<int> &bills) {
    int five = 0, ten = 0;
    for (int bill : bills) {
        if (bill == 5) {
            ++five;
        } else if (bill == 10) {
            if (five == 0) return false;
            --five;
            ++ten;
        } else {
            if (ten >= 1 && five >= 1) {
                --ten;
                --five;
            } else if (five >= 3) {
                five -= 3;
            } else {
                return false;
            }
        }
    }
    return true;
}

int main() {
    std::vector<int> a = {5, 5, 5, 10, 20};
    assert(lemonadeChange(a) == true);
    std::vector<int> b = {5, 5, 10, 10, 20};
    assert(lemonadeChange(b) == false);
    std::vector<int> c = {5, 5, 5, 5, 10, 5, 10, 10, 10, 20};
    assert(lemonadeChange(c) == true);
    std::vector<int> d = {10};
    assert(lemonadeChange(d) == false);
    std::vector<int> e = {5, 5, 10};
    assert(lemonadeChange(e) == true);
    std::cout << "lemonade_change: all tests passed\n";
    return 0;
}
