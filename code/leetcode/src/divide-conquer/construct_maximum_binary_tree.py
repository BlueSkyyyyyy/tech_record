"""654. 最大二叉树（Maximum Binary Tree）

题目：给定一个不重复的整数数组 nums，用下面的规则构造一棵最大二叉树：
      根是 nums 中的最大元素；左子树由最大值左边的那段元素递归构造，
      右子树由最大值右边的那段元素递归构造。返回构造出的根结点。

思路（分治：取最大值作根，左右两段各自递归）：
    题目已经把分治的三步写得清清楚楚：
      1. 分解：在当前区间里找到最大值，它的位置把区间分成左右两段；
      2. 解决：对左右两段分别递归构造最大二叉树；
      3. 合并：把递归得到的左右子树挂到当前根结点上。
    当区间为空时返回空结点，这是最小子问题。

    为什么最大值一定在根：规则要求如此；而左右两段在位置上天然被最大值隔开，
    且各自只含比它小的值，所以递归构造互不干扰，能拼出唯一确定的树。

    为什么用下标区间而不是切片：切片会复制数组、增加开销；用 [lo, hi) 的
    下标区间在原数组上操作，语义更清晰，也更容易看出每个元素只当一次根。

    这道题还可以用单调栈做到 O(n)：从右往左/从左往右扫，用递减栈直接确定
    每个结点的父结点。分治版是 O(n^2)（每次找最大值要扫一遍，最坏退化成
    单链），但分治结构最直观，适合用来理解「由序列递归建树」。

复杂度：时间 O(n^2)（最坏情况如严格递增数组，每层只缩小一个元素；
    平均 O(n log n)），空间 O(n)（递归栈，最坏退化成链表）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def construct_maximum_binary_tree(nums):
    if not nums:
        return None
    max_index = 0
    for i in range(1, len(nums)):
        if nums[i] > nums[max_index]:
            max_index = i
    node = TreeNode(nums[max_index])
    node.left = construct_maximum_binary_tree(nums[:max_index])
    node.right = construct_maximum_binary_tree(nums[max_index + 1:])
    return node


def preorder(node, out):
    if node is None:
        return
    out.append(node.val)
    preorder(node.left, out)
    preorder(node.right, out)


if __name__ == "__main__":
    out = []
    preorder(construct_maximum_binary_tree([3, 2, 1, 6, 0, 5]), out)
    assert out == [6, 3, 2, 1, 5, 0]

    out = []
    preorder(construct_maximum_binary_tree([3, 2, 1]), out)
    assert out == [3, 2, 1]

    out = []
    preorder(construct_maximum_binary_tree([1]), out)
    assert out == [1]

    assert construct_maximum_binary_tree([]) is None
    print("construct_maximum_binary_tree: all tests passed")
