"""210. 课程表 II（Course Schedule II）

题目：与 207 相同，但要求返回一种可以修完所有课程的学习顺序；
如果不可能修完（有环），返回空数组。

思路（拓扑排序 / Kahn 算法，记录出队顺序）：
    和 207 完全同一套流程，只是这次的产物不是「计数是否等于总数」，
    而是「出队的顺序」。按出队顺序把课程依次记下来，就得到一组满足先修约束的课表。

    为什么出队顺序一定合法：只有当一门课的所有先修课都已出队（入度降为 0）时它才会出队，
    所以排在它前面的必然是它的先修课。环上的课永远不会入队，因此若最终顺序长度不足，
    就说明存在环，按题目要求返回空列表。

    若存在多组合法顺序，题目说返回任意一种即可；Kahn 算法返回哪一种取决于初始
    入度为 0 的课程顺序和队列的选择策略，都是正确的。

复杂度：时间 O(V + E)，空间 O(V + E)。
"""
from collections import deque


def find_order(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for course, pre in prerequisites:
        graph[pre].append(course)
        indegree[course] += 1

    queue = deque(c for c in range(num_courses) if indegree[c] == 0)
    order = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    return order if len(order) == num_courses else []


if __name__ == "__main__":
    assert find_order(2, [[1, 0]]) == [0, 1]
    assert find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]) == [0, 1, 2, 3]
    assert find_order(1, []) == [0]
    assert find_order(2, [[1, 0], [0, 1]]) == []

    # 校验顺序确实满足先修关系
    def valid(num_courses, prerequisites, order):
        if len(order) != num_courses:
            return False
        pos = {c: i for i, c in enumerate(order)}
        return all(pos[b] < pos[a] for a, b in prerequisites)

    assert valid(4, [[1, 0], [2, 0], [3, 1], [3, 2]],
                 find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
    print("course_schedule_ii: all tests passed")
