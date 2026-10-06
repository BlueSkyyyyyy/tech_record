// 241. 为运算表达式设计优先级
// 见 different_ways_to_add_parentheses.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> diffWaysToCompute(const std::string &expression) {
    std::vector<int> results;
    for (int i = 0; i < static_cast<int>(expression.size()); ++i) {
        char c = expression[i];
        if (c == '+' || c == '-' || c == '*') {
            std::vector<int> left = diffWaysToCompute(expression.substr(0, i));
            std::vector<int> right = diffWaysToCompute(expression.substr(i + 1));
            for (int a : left) {
                for (int b : right) {
                    if (c == '+') results.push_back(a + b);
                    else if (c == '-') results.push_back(a - b);
                    else results.push_back(a * b);
                }
            }
        }
    }
    if (results.empty()) results.push_back(std::stoi(expression));
    return results;
}

int main() {
    std::vector<int> r1 = diffWaysToCompute("2-1-1");
    std::sort(r1.begin(), r1.end());
    std::vector<int> want1 = {0, 2};
    assert(r1 == want1);

    std::vector<int> r2 = diffWaysToCompute("2*3-4*5");
    std::sort(r2.begin(), r2.end());
    std::vector<int> want2 = {-34, -14, -10, -10, 10};
    assert(r2 == want2);

    std::vector<int> r3 = diffWaysToCompute("3");
    std::vector<int> want3 = {3};
    assert(r3 == want3);

    std::vector<int> r4 = diffWaysToCompute("1+2+3");
    std::sort(r4.begin(), r4.end());
    std::vector<int> want4 = {6, 6};
    assert(r4 == want4);
    std::cout << "diff_ways_to_compute: all tests passed\n";
    return 0;
}
