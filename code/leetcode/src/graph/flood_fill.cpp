// 733. 图像渲染
// 见 flood_fill.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void dfs(std::vector<std::vector<int>> &image, int r, int c, int start, int color) {
    int rows = image.size();
    int cols = image[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || image[r][c] != start) return;
    image[r][c] = color;
    dfs(image, r + 1, c, start, color);
    dfs(image, r - 1, c, start, color);
    dfs(image, r, c + 1, start, color);
    dfs(image, r, c - 1, start, color);
}

std::vector<std::vector<int>> floodFill(std::vector<std::vector<int>> image,
                                        int sr, int sc, int color) {
    int start = image[sr][sc];
    if (start == color) return image;
    dfs(image, sr, sc, start, color);
    return image;
}

int main() {
    std::vector<std::vector<int>> want1 = {
        {2, 2, 2},
        {2, 2, 0},
        {2, 0, 1},
    };
    std::vector<std::vector<int>> got1 = floodFill({{1, 1, 1}, {1, 1, 0}, {1, 0, 1}}, 1, 1, 2);
    assert(got1 == want1);

    std::vector<std::vector<int>> same = {{0, 0, 0}, {0, 0, 0}};
    std::vector<std::vector<int>> sameWant = {{0, 0, 0}, {0, 0, 0}};
    assert(floodFill(same, 0, 0, 0) == sameWant);

    std::vector<std::vector<int>> oneWant = {{9}};
    assert(floodFill({{5}}, 0, 0, 9) == oneWant);

    std::cout << "flood_fill: all tests passed\n";
    return 0;
}
