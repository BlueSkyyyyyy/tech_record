// 839. 相似字符串组（Similar String Groups）
// 见 similar_string_groups.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
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

int numSimilarGroups(const std::vector<std::string> &strs) {
    int n = strs.size();
    DSU dsu(n);

    auto similar = [](const std::string &a, const std::string &b) {
        std::vector<int> diff;
        for (int i = 0; i < (int)a.size(); ++i) {
            if (a[i] != b[i]) diff.push_back(i);
        }
        if (diff.empty()) return true;
        if (diff.size() != 2) return false;
        return a[diff[0]] == b[diff[1]] && a[diff[1]] == b[diff[0]];
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (similar(strs[i], strs[j])) dsu.unite(i, j);
        }
    }

    std::unordered_set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(dsu.find(i));
    return roots.size();
}

int main() {
    std::vector<std::string> s1 = {"tars", "rats", "arts", "star"};
    assert(numSimilarGroups(s1) == 2);

    std::vector<std::string> s2 = {"omv", "ovm"};
    assert(numSimilarGroups(s2) == 1);

    std::vector<std::string> s3 = {"abc"};
    assert(numSimilarGroups(s3) == 1);

    std::cout << "similar_string_groups: all tests passed\n";
    return 0;
}
