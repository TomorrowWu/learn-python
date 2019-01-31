from mxnet import nd

# 广播（broadcasting）机制：先适当复制元素使这两个NDArray形状相同后再按元素运算
A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))

# 由于A和B分别是3行1列和1行2列的矩阵，如果要计算A + B，那么A中第一列的3个元素被广播（复制）到了第二列，而B中第一行的2个元素被广播（复制）到了第二行和第三行
print(A + B)
