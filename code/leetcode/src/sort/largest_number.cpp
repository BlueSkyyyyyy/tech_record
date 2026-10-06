// 179. 最大数
// 见 largest_number.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::string largestNumber(const std::vector<int> &nums) {
    std::vector<std::string> strs;
    strs.reserve(nums.size());
    for (int x : nums) {
        strs.push_back(std::to_string(x));
    }
    std::sort(strs.begin(), strs.end(),
              [](const std::string &a, const std::string &b) {
                  return a + b > b + a;
              });
    std::string res;
    for (const auto &s : strs) {
        res += s;
    }
    if (!res.empty() && res[0] == '0') {
        return "0";
    }
    return res;
}

int main() {
    assert(largestNumber({10, 2}) == "210");
    assert(largestNumber({3, 30, 34, 5, 9}) == "9534330");
    assert(largestNumber({0, 0}) == "0");
    assert(largestNumber({1}) == "1");
    assert(largestNumber({3, 30}) == "330");

    std::cout << "largest_number: all tests passed\n";
    return 0;
}
