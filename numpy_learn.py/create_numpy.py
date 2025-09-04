import numpy as np
#ndim参数指定数组的最小维度
a=np.array([1, 2, 3],ndmin=2)
print(a)
#dtype参数指定数组元素的数据类型 complex表示复数
b=np.array([1, 2, 3],dtype=complex)
print(b)