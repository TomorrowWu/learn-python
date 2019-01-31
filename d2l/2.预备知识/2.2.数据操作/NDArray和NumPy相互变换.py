import numpy as np
from mxnet import nd

# 将NumPy实例变换成NDArray实例
P = np.ones((2, 3))
D = nd.array(P)
print(D)

# 将NDArray实例变换成NumPy实例
np = D.asnumpy()
print(np)
