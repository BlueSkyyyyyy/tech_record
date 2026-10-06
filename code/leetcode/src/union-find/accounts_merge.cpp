// 721. 账户合并（Accounts Merge）
// 见 accounts_merge.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
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

std::vector<std::vector<std::string>> accountsMerge(
    const std::vector<std::vector<std::string>> &accounts) {
    int n = accounts.size();
    std::unordered_map<std::string, int> email_to_id;
    DSU dsu(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < (int)accounts[i].size(); ++j) {
            const std::string &email = accounts[i][j];
            auto it = email_to_id.find(email);
            if (it != email_to_id.end()) {
                dsu.unite(i, it->second);
            } else {
                email_to_id[email] = i;
            }
        }
    }

    std::map<int, std::vector<std::string>> root_to_emails;
    for (const auto &kv : email_to_id) {
        root_to_emails[dsu.find(kv.second)].push_back(kv.first);
    }

    std::vector<std::vector<std::string>> result;
    for (auto &kv : root_to_emails) {
        std::vector<std::string> emails = kv.second;
        std::sort(emails.begin(), emails.end());
        std::vector<std::string> row;
        row.push_back(accounts[kv.first][0]);
        row.insert(row.end(), emails.begin(), emails.end());
        result.push_back(row);
    }
    std::sort(result.begin(), result.end(),
              [](const std::vector<std::string> &a, const std::vector<std::string> &b) {
                  return a[1] < b[1];
              });
    return result;
}

int main() {
    std::vector<std::vector<std::string>> accounts = {
        {"John", "johnsmith@mail.com", "john_newyork@mail.com"},
        {"John", "johnsmith@mail.com", "john00@mail.com"},
        {"Mary", "mary@mail.com"},
        {"John", "johnnybravo@mail.com"},
    };
    std::vector<std::vector<std::string>> want = {
        {"John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"},
        {"John", "johnnybravo@mail.com"},
        {"Mary", "mary@mail.com"},
    };
    assert(accountsMerge(accounts) == want);

    std::vector<std::vector<std::string>> one = {{"Gabe", "g@m.com"}};
    std::vector<std::vector<std::string>> want_one = {{"Gabe", "g@m.com"}};
    assert(accountsMerge(one) == want_one);

    std::cout << "accounts_merge: all tests passed\n";
    return 0;
}
