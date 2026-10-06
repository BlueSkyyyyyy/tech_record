// 784. 字母大小写全排列
// 见 letter_case_permutation.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> letterCasePermutation(std::string s) {
    std::vector<int> letters;
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        if (std::isalpha(static_cast<unsigned char>(s[i]))) {
            letters.push_back(i);
        }
    }
    int L = static_cast<int>(letters.size());
    std::vector<std::string> res;
    for (int mask = 0; mask < (1 << L); ++mask) {
        std::string cur = s;
        for (int j = 0; j < L; ++j) {
            if (mask >> j & 1) {
                cur[letters[j]] = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(cur[letters[j]])));
            } else {
                cur[letters[j]] = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(cur[letters[j]])));
            }
        }
        res.push_back(cur);
    }
    return res;
}

int main() {
    auto got = letterCasePermutation("a1b2");
    std::vector<std::string> want = {"A1B2", "A1b2", "a1B2", "a1b2"};
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    assert(got == want);

    auto got2 = letterCasePermutation("3z4");
    std::vector<std::string> want2 = {"3Z4", "3z4"};
    std::sort(got2.begin(), got2.end());
    std::sort(want2.begin(), want2.end());
    assert(got2 == want2);

    assert(letterCasePermutation("12345") == std::vector<std::string>{"12345"});
    assert(letterCasePermutation("ab").size() == 4);

    std::cout << "letter_case_permutation: all tests passed\n";
    return 0;
}
