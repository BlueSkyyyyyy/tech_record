// 557. 反转字符串中的单词 III
// 见 reverse_words_iii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

std::string reverseWordsIII(std::string s) {
    int n = static_cast<int>(s.size());
    int start = 0;
    for (int i = 0; i <= n; ++i) {
        if (i == n || s[i] == ' ') {
            std::reverse(s.begin() + start, s.begin() + i);
            start = i + 1;
        }
    }
    return s;
}

int main() {
    assert(reverseWordsIII("Let's take LeetCode contest") ==
           "s'teL ekat edoCteeL tsetnoc");
    assert(reverseWordsIII("Mr Ding") == "rM gniD");
    assert(reverseWordsIII("a") == "a");
    assert(reverseWordsIII("") == "");

    std::cout << "reverse_words_iii: all tests passed\n";
    return 0;
}
