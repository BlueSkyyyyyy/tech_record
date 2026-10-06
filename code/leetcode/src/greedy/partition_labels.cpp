// 763. 划分字母区间
// 见 partition_labels.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> partitionLabels(const std::string &s) {
    int last[26];
    std::fill(last, last + 26, -1);
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        last[s[i] - 'a'] = i;
    }
    std::vector<int> result;
    int start = 0, end = 0;
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        end = std::max(end, last[s[i] - 'a']);
        if (i == end) {
            result.push_back(end - start + 1);
            start = i + 1;
        }
    }
    return result;
}

int main() {
    std::vector<int> a = partitionLabels("ababcbacadefegdehijhklij");
    std::vector<int> aWant = {9, 7, 8};
    assert(a == aWant);
    std::vector<int> b = partitionLabels("eccbbbbdec");
    std::vector<int> bWant = {10};
    assert(b == bWant);
    std::vector<int> c = partitionLabels("a");
    std::vector<int> cWant = {1};
    assert(c == cWant);
    std::vector<int> d = partitionLabels("abc");
    std::vector<int> dWant = {1, 1, 1};
    assert(d == dWant);
    std::cout << "partition_labels: all tests passed\n";
    return 0;
}
