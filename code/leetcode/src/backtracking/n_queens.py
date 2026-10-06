"""51. N 皇后（N-Queens）

题目：按照国际象棋的规则，皇后可以攻击与之处在同一行、同一列或同一斜线上的
棋子。n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此
之间不能互相攻击。给你一个整数 n，返回所有不同的 n 皇后问题的解决方案。
每一种解法包含一个不同的棋盘布局，其中 'Q' 表示皇后，'.' 表示空位。

思路（按行回溯 + 列 / 两条对角线冲突检测）：
    关键观察：每行恰好放一个皇后。因为一共 n 行、放 n 个皇后，若某行放两个
    必有另一行空着，那时两皇后同列或同行冲突无法避免。所以决策树按「行」展开：
    第 r 层决定这一行的皇后放在哪一列。

    每放一个皇后，要检查三件事：同列、主对角线（r - c 为常量）、副对角线
    （r + c 为常量）。用三个集合分别记录已占用的列、主对角线、副对角线，
    放子时加入、回溯时移除，实现 O(1) 冲突检测。

    为什么按行 + 三个集合就够：行冲突被「每行只放一个」天然排除；列、两斜线
    恰好覆盖国际象棋的全部攻击方式，所以只要这三样都不冲突，放置就合法。

复杂度：时间 O(n!)（每行可选列数急剧减少，实际远小于 n^n），
    空间 O(n)（递归深度 + 三个集合）。
"""


def solve_n_queens(n):
    res = []
    board = ["." * n for _ in range(n)]
    cols = set()
    diag_main = set()
    diag_anti = set()

    def backtrack(r):
        if r == n:
            res.append(board[:])
            return
        for c in range(n):
            if c in cols or (r - c) in diag_main or (r + c) in diag_anti:
                continue
            cols.add(c)
            diag_main.add(r - c)
            diag_anti.add(r + c)
            board[r] = board[r][:c] + "Q" + board[r][c + 1:]
            backtrack(r + 1)
            board[r] = board[r][:c] + "." + board[r][c + 1:]
            cols.remove(c)
            diag_main.remove(r - c)
            diag_anti.remove(r + c)

    backtrack(0)
    return res


if __name__ == "__main__":
    out4 = solve_n_queens(4)
    assert len(out4) == 2
    assert [".Q..", "...Q", "Q...", "..Q."] in out4
    assert ["..Q.", "Q...", "...Q", ".Q.."] in out4

    assert len(solve_n_queens(1)) == 1
    assert len(solve_n_queens(2)) == 0
    assert len(solve_n_queens(8)) == 92
    print("solve_n_queens: all tests passed")
