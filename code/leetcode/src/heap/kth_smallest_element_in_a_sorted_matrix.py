"""378. 有序矩阵中第 K 小的元素（Kth Smallest Element in a Sorted Matrix）

题目：给你一个 n x n 矩阵，每行、每列都按升序排列。
      返回矩阵中第 k 小的元素（按排序顺序，不是第 k 个不同元素）。

思路（多路归并：把每一行看成一条有序链，用堆取第 k 个）：
    每行都是升序的，所以「整个矩阵的第 k 小」等价于把 n 条有序行归并后取第 k 个。
    做法和 23 题合并 K 个有序链表一模一样：
    先把每一行的第一个元素 (值, 行, 列) 压入小顶堆，堆顶就是当前全局最小。
    每弹出一个，就把它所在行的下一个元素压回堆。弹出 k-1 次后，堆顶就是第 k 小。

    为什么不用把整个矩阵读出来排序：
    矩阵有 n^2 个元素，全部排序要 O(n^2 log n)。
    多路归并只需要维护 n 个候选（每行一个），每次 O(log n)，总共 O(k log n)，
    当 k 远小于 n^2 时快得多，也更省内存。

    为什么堆里要带 (行, 列) 下标：
    只存值无法知道它属于哪一行，也就无法取「该行的下一个」。带下标才能续推。

    另一种做法是二分答案：对值域二分，统计不超过 mid 的元素个数，
    也能做到 O(n log(max-min))，思路在二分篇里讲，这里详展更贴合堆主题的归并解。

复杂度：时间 O(k log n)，空间 O(n)。
"""

import heapq


def kth_smallest(matrix, k):
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)

    for _ in range(k - 1):
        value, i, j = heapq.heappop(heap)
        if j + 1 < len(matrix[i]):
            heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))

    return heap[0][0]


if __name__ == "__main__":
    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15],
    ]
    assert kth_smallest(matrix, 8) == 13
    assert kth_smallest(matrix, 1) == 1
    assert kth_smallest(matrix, 9) == 15
    assert kth_smallest([[-5]], 1) == -5
    assert kth_smallest([[1, 2], [1, 3]], 2) == 1
    assert kth_smallest([[1, 2], [3, 4]], 4) == 4
    print("kth_smallest_element_in_a_sorted_matrix: all tests passed")
