"""130. 被围绕的区域（Surrounded Regions）

题目：给你一个 m x n 的矩阵 board，由若干字符 'X' 和 'O' 组成。
捕获所有被 'X' 围绕的区域：把其中所有 'O' 用 'X' 填充。
注意：任何边界上的 'O'，以及与边界上的 'O' 相连的 'O'，都不会被填充。

思路（换个方向：先标记「安全的 O」，再翻转剩下的）：
    直接判断某个 'O' 是否被完全包围并不容易，但反过来想很轻松：
    一个 'O' 不会被填充，当且仅当它能通过四方向走到矩阵边界。
    于是：

    1. 从**四条边界**上的每个 'O' 出发做 DFS/BFS，把所有与边界相连的 'O'
       临时标记成另一个字符（比如 '#'），表示「安全、不能被翻转」；
    2. 遍历整个矩阵做收尾：仍是 'O' 的说明被 'X' 完全包围，改成 'X'；
       是 '#' 的说明安全，恢复成 'O'。

    为什么从边界出发：边界是「逃出去」的唯一出口，能从内部走到边界的 'O'
    一定会被步骤 1 标记到；反之走不到边界的 'O' 就被困住了，正是要翻转的对象。

    为什么用临时标记而不用 visited 表：标记字符既记录了「已访问」，又方便在收尾时
    区分「安全的 O」与「待翻转的 O」，一次遍历就能完成翻转与还原。

复杂度：时间 O(m·n)（每个格子至多访问一次），空间 O(m·n)（递归栈最坏情形）。
"""
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def solve(board):
    if not board or not board[0]:
        return
    rows, cols = len(board), len(board[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != "O":
            return
        board[r][c] = "#"
        for dr, dc in _DIRS:
            dfs(r + dr, c + dc)

    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)
    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == "#":
                board[r][c] = "O"


if __name__ == "__main__":
    board = [
        ["X", "X", "X", "X"],
        ["X", "O", "O", "X"],
        ["X", "X", "O", "X"],
        ["X", "O", "X", "X"],
    ]
    solve(board)
    assert board == [
        ["X", "X", "X", "X"],
        ["X", "X", "X", "X"],
        ["X", "X", "X", "X"],
        ["X", "O", "X", "X"],
    ]

    # 边界上的 O 及与其相连的 O 都保留
    b2 = [["O", "O"], ["O", "O"]]
    solve(b2)
    assert b2 == [["O", "O"], ["O", "O"]]

    # 空矩阵直接返回
    empty = []
    solve(empty)
    assert empty == []

    b3 = [["X"]]
    solve(b3)
    assert b3 == [["X"]]
    print("surrounded_regions: all tests passed")
