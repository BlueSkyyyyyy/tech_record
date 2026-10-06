// 75. 颜色分类
// 见 sort_colors.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

void sortColors(std::vector<int> &nums) {
    int p0 = 0, cur = 0, p2 = static_cast<int>(nums.size()) - 1;
    while (cur <= p2) {
        if (nums[cur] == 0) {
            std::swap(nums[p0], nums[cur]);
            ++p0;
            ++cur;
        } else if (nums[cur] == 2) {
            std::swap(nums[cur], nums[p2]);
            --p2;
        } else {
            ++cur;
        }
    }
}

int main() {
    std::vector<int> a = {2, 0, 2, 1, 1, 0};
    std::vector<int> want1 = {0, 0, 1, 1, 2, 2};
    sortColors(a);
    assert(a == want1);

    std::vector<int> b = {2, 0, 1};
    std::vector<int> want2 = {0, 1, 2};
    sortColors(b);
    assert(b == want2);

    std::vector<int> c = {0};
    std::vector<int> want3 = {0};
    sortColors(c);
    assert(c == want3);

    std::vector<int> d = {2, 2, 1, 0, 0, 1};
    std::vector<int> want4 = {0, 0, 1, 1, 2, 2};
    sortColors(d);
    assert(d == want4);

    std::vector<int> e = {};
    std::vector<int> want5 = {};
    sortColors(e);
    assert(e == want5);

    std::cout << "sort_colors: all tests passed\n";
    return 0;
}
