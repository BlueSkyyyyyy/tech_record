"""79. 单词搜索（Word Search）

题目：给定一个 m x n 的二维字符网格 board 和一个字符串单词 word。如果 word
存在于网格中，返回 true；否则返回 false。单词必须按照字母顺序，通过相邻的
单元格内的字母构成，其中「相邻」单元格是水平或垂直相邻的。同一个单元格内的
字母不允许被重复使用。

思路（网格 DFS 回溯）：
    这是把决策树铺在二维网格上：从任意格子出发，每一步向上下左右走，只要
    下一步的字母和单词下一位对得上，就继续深入；一旦走完整个单词就成功。

    和 78/46 的区别是「选择列表」不是数组下标，而是四个方向的邻居。状态撤销
    的方式也不同：这里用一个原地技巧——进入格子先把 board[r][c] 改成占位符
    （如 '#'）表示「已访问」，四个方向递归完再改回来。这样不需要额外的 visited
    数组，靠棋盘本身记录路径。

    为什么最后要还原：同一格可能在另一条搜索路径里被合法使用（不同出发点、
    不同的拐弯路线），不还原会误判为「已经用过」。

复杂度：时间 O(m·n·3^L)（每个起点最多向 3 个未回头方向扩展，L 为单词长度），
    空间 O(L)（递归深度）。
"""


def exist(board, word):
    rows, cols = len(board), len(board[0])

    def dfs(r, c, k):
        if k == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[k]:
            return False
        saved = board[r][c]
        board[r][c] = "#"
        found = (
            dfs(r + 1, c, k + 1)
            or dfs(r - 1, c, k + 1)
            or dfs(r, c + 1, k + 1)
            or dfs(r, c - 1, k + 1)
        )
        board[r][c] = saved
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False


if __name__ == "__main__":
    board = [
        ["A", "B", "C", "E"],
        ["S", "F", "C", "S"],
        ["A", "D", "E", "E"],
    ]
    assert exist(board, "ABCCED") is True
    assert exist(board, "SEE") is True
    assert exist(board, "ABCB") is False
    assert exist([["a"]], "a") is True
    assert exist([["a"]], "b") is False
    print("exist: all tests passed")
