from mxnet import nd

x = nd.arange(12)

print(x)
print(x.shape)
print(x.size)

X = x.reshape((3, 4))  # x.reshape((-1, 4))   x.reshape((3, -1))   由于x的元素个数是已知的，这里的-1是能够通过元素个数和其他维度的大小推断出来的。
print(X)

y = nd.zeros((2, 3, 4))
print(y)

z = nd.ones((3, 4))
print(z)

Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

normal = nd.random.normal(0, 1, shape=(3, 4))  # 均值为0、标准差为1的正态分布
print(normal)

# 按元素加法
print(X + Y)

# 按元素乘法
print(X * Y)

# 按元素除法
print(X / Y)

# 按元素做指数运算  exp，高等数学里以自然常数e为底的指数函数   返回e的n次方
print(Y.exp())

# 矩阵乘法   将X与Y的转置做矩阵乘法   ?????
print(nd.dot(X, Y.T))

# 连结   ??????
print(nd.concat(X, Y, dim=0))
print(nd.concat(X, Y, dim=1))

# 如果X和Y在相同位置的条件判断为真（值相等），那么新的NDArray在相同位置的值为1；反之为0
print(X == Y)

# 对NDArray中的所有元素求和得到只有一个元素的NDArray
print(X.sum())

#  norm函数: L2 范数结果 asscalar函数将结果变换为Python中的标量
print(X.norm().asscalar())
