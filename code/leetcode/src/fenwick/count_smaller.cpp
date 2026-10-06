// 315. 计算右侧小于当前元素的个数
// 见 count_smaller.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

struct Fenwick {
    int n;
    std::vector<long long> tree;
    explicit Fenwick(int n) : n(n), tree(n + 1, 0) {}
    void add(int i, long long delta) {
        for (; i <= n; i += i & -i) tree[i] += delta;
    }
    long long prefix(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += tree[i];
        return s;
    }
};

std::vector<int> countSmaller(std::vector<int> nums) {
    std::vector<int> vals = nums;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    int n = static_cast<int>(nums.size());
    std::vector<int> ans(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), nums[i]) - vals.begin()) +
                1;
        ans[i] = static_cast<int>(bit.prefix(r - 1));
        bit.add(r, 1);
    }
    return ans;
}

int main() {
    std::vector<int> got1 = countSmaller({5, 2, 6, 1});
    std::vector<int> want1 = {2, 1, 1, 0};
    assert(got1 == want1);

    std::vector<int> got2 = countSmaller({-1, -1});
    std::vector<int> want2 = {0, 0};
    assert(got2 == want2);

    std::vector<int> got3 = countSmaller({3, 4, 9, 6, 1});
    std::vector<int> want3 = {1, 1, 2, 1, 0};
    assert(got3 == want3);

    std::vector<int> got4 = countSmaller({2, 0, 1});
    std::vector<int> want4 = {2, 0, 0};
    assert(got4 == want4);

    std::cout << "count_smaller: all tests passed\n";
    return 0;
}
