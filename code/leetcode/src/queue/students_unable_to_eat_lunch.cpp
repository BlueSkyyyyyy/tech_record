// 1700. 无法吃午餐的学生数量
// 见 students_unable_to_eat_lunch.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int countStudents(std::vector<int>& students, std::vector<int>& sandwiches) {
    int zeros = 0;
    for (int s : students) {
        if (s == 0) {
            ++zeros;
        }
    }
    int ones = static_cast<int>(students.size()) - zeros;
    for (int s : sandwiches) {
        if (s == 0 && zeros > 0) {
            --zeros;
        } else if (s == 1 && ones > 0) {
            --ones;
        } else {
            break;
        }
    }
    return zeros + ones;
}

int main() {
    std::vector<int> s1{1, 1, 0, 0};
    std::vector<int> t1{0, 1, 0, 1};
    std::vector<int> s2{1, 1, 1, 0, 0, 1};
    std::vector<int> t2{1, 0, 0, 0, 1, 1};
    std::vector<int> s3{0};
    std::vector<int> t3{1};
    std::vector<int> s4{1};
    std::vector<int> t4{1};
    assert(countStudents(s1, t1) == 0);
    assert(countStudents(s2, t2) == 3);
    assert(countStudents(s3, t3) == 1);
    assert(countStudents(s4, t4) == 0);
    std::cout << "count_students: all tests passed\n";
    return 0;
}
