// 17. 电话号码的字母组合
// 见 letter_combinations.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void backtrack(const std::string &digits, int i,
               const std::vector<std::string> &table, std::string &path,
               std::vector<std::string> &res) {
    if (i == static_cast<int>(digits.size())) {
        res.push_back(path);
        return;
    }
    for (char ch : table[digits[i] - '0']) {
        path.push_back(ch);
        backtrack(digits, i + 1, table, path, res);
        path.pop_back();
    }
}

std::vector<std::string> letterCombinations(const std::string &digits) {
    if (digits.empty()) return {};
    std::vector<std::string> table = {"",    "",    "abc",  "def", "ghi",
                                      "jkl", "mno", "pqrs", "tuv", "wxyz"};
    std::vector<std::string> res;
    std::string path;
    backtrack(digits, 0, table, path, res);
    return res;
}

int main() {
    auto got = letterCombinations("23");
    assert(got.size() == 9);
    auto has = [&](const std::string &v) {
        for (auto &x : got)
            if (x == v) return true;
        return false;
    };
    assert(has("ad") && has("ae") && has("cf"));

    assert(letterCombinations("").empty());
    auto got2 = letterCombinations("2");
    std::vector<std::string> want2 = {"a", "b", "c"};
    assert(got2 == want2);

    std::cout << "letter_combinations: all tests passed\n";
    return 0;
}
