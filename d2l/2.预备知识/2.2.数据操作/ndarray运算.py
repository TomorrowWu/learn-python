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

# 截取了矩阵X中行索引为1和2的两行
print(X[1:3])

# 访问的单个元素的位置，如矩阵中行和列的索引
X[1, 2] = 9

print(X)

# 截取一部分元素，并为它们重新赋值
X[1:2, :] = 12
print(X)

# 如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 指定结果到特定内存，我们可以使用前面介绍的索引来进行替换操作
# 为X + Y开了临时内存来存储计算结果，再复制到Z对应的内存
Z = Y.zeros_like()
before = id(Z)
Z[:] = X + Y
print(id(Z) == before)

# 避免这个临时内存开销，我们可以使用运算符全名函数中的out参数
nd.elemwise_add(X, Y, out=Z)
print(id(Z) == before)

# 如果X的值在之后的程序中不会复用,减少运算的内存开销
before = id(X)
X += Y
print(id(X) == before)
