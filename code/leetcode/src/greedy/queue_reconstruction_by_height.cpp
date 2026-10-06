// 406. 根据身高重建队列
// 见 queue_reconstruction_by_height.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> reconstructQueue(std::vector<std::vector<int>> people) {
    std::sort(people.begin(), people.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) {
                  if (a[0] != b[0]) return a[0] > b[0];
                  return a[1] < b[1];
              });
    std::vector<std::vector<int>> result;
    for (const auto &person : people) {
        result.insert(result.begin() + person[1], person);
    }
    return result;
}

int main() {
    std::vector<std::vector<int>> a = {{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}};
    std::vector<std::vector<int>> aWant = {{5, 0}, {7, 0}, {5, 2}, {6, 1}, {4, 4}, {7, 1}};
    assert(reconstructQueue(a) == aWant);
    std::vector<std::vector<int>> b = {{6, 0}, {5, 0}, {4, 0}, {3, 2}, {2, 2}, {1, 4}};
    std::vector<std::vector<int>> bWant = {{4, 0}, {5, 0}, {2, 2}, {3, 2}, {1, 4}, {6, 0}};
    assert(reconstructQueue(b) == bWant);
    std::vector<std::vector<int>> c;
    assert(reconstructQueue(c).empty());
    std::cout << "queue_reconstruction_by_height: all tests passed\n";
    return 0;
}
