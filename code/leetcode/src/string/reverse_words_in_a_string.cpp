// 151. 翻转字符串里的单词
// 见 reverse_words_in_a_string.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

std::string reverseWords(std::string s) {
    int n = static_cast<int>(s.size());
    std::reverse(s.begin(), s.end());

    int write = 0;
    for (int i = 0; i < n; ++i) {
        if (s[i] != ' ') {
            if (write != 0) s[write++] = ' ';
            int start = write;
            while (i < n && s[i] != ' ') {
                s[write++] = s[i++];
            }
            std::reverse(s.begin() + start, s.begin() + write);
        }
    }
    return s.substr(0, write);
}

int main() {
    assert(reverseWords("the sky is blue") == "blue is sky the");
    assert(reverseWords("  hello world  ") == "world hello");
    assert(reverseWords("a good   example") == "example good a");
    assert(reverseWords("") == "");
    assert(reverseWords("   ") == "");
    assert(reverseWords("word") == "word");
    assert(reverseWords("  a  ") == "a");
    assert(reverseWords("Epic   systems   rocks") == "rocks systems Epic");
    std::cout << "reverse_words_in_a_string: all tests passed\n";
    return 0;
}
