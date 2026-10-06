// 1202. 交换字符串中的元素（Smallest String With Swaps）
// 见 smallest_string_with_swaps.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct DSU {
    std::vector<int> parent, rank_;
    explicit DSU(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank_[ra] == rank_[rb]) ++rank_[ra];
        return true;
    }
};

std::string smallestStringWithSwaps(std::string s,
                                    const std::vector<std::vector<int>> &pairs) {
    int n = s.size();
    DSU dsu(n);
    for (const auto &p : pairs) {
        dsu.unite(p[0], p[1]);
    }

    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[dsu.find(i)].push_back(i);

    for (auto &kv : groups) {
        std::vector<int> indices = kv.second;
        std::sort(indices.begin(), indices.end());
        std::string chars;
        for (int i : indices) chars.push_back(s[i]);
        std::sort(chars.begin(), chars.end());
        for (int k = 0; k < (int)indices.size(); ++k) s[indices[k]] = chars[k];
    }
    return s;
}

int main() {
    std::vector<std::vector<int>> p1 = {{0, 3}, {1, 2}};
    assert(smallestStringWithSwaps("dcab", p1) == "bacd");

    std::vector<std::vector<int>> p2 = {{0, 3}, {1, 2}, {0, 2}};
    assert(smallestStringWithSwaps("dcab", p2) == "abcd");

    std::vector<std::vector<int>> none = {};
    assert(smallestStringWithSwaps("cba", none) == "cba");

    std::vector<std::vector<int>> p3 = {{0, 1}, {1, 2}, {2, 3}};
    assert(smallestStringWithSwaps("abcd", p3) == "abcd");

    std::cout << "smallest_string_with_swaps: all tests passed\n";
    return 0;
}
