# import numpy as np 
# #获取数组中 (0,0)，(1,1) 和 (2,0) 位置处的元素。
# x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
# y = x[[0,1,2],  [0,1,0]]  
# print (y)

import numpy as np 
 
x = np.array([[ 0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：' )
print (x)
print ('\n')
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]  
print  ('这个数组的四个角元素是：')
print (y)



import numpy as np
 
a = np.array([[1,2,3], [4,5,6],[7,8,9]])
b = a[1:3, 1:3]
# 选择行：从索引 1 到 2（不包括 3），即第 2 和第 3 行。
# 选择列：[1,2]，即直接选取索引 1 和 2 列。
c = a[1:3,[1,2]]
#d = a[...,1:]
# ... 表示所有行。
# 1: 表示从索引 1 到最后，覆盖第 2 和第 3 列
d = a[...,1:]
print(b)
print(c)
print(d)

import numpy as np 
 
m = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：')
print (m)
print ('\n')
# 现在我们会打印出大于 5 的元素  
print  ('大于 5 的元素是：')
#m[m > 5] 会将数组中的大于 5 的元素按顺序返回，并形成一个一维数组。
print (m[m >  5])

#NaN表示“不是一个数字”（Not a Number）的缩写，通常用于表示缺失值或未定义的数值。
o = np.array([np.nan,  1,2,np.nan,3,4,5])  
print (o[~np.isnan(o)])
import numpy as np 
 

x=np.arange(32).reshape((8,4))
print(x)
# 二维数组读取指定下标对应的行
print("-------读取下标对应的行-------")
print (x[[4,2,1,7]])



import numpy as np 
#np.ix_ 函数就是输入两个数组，产生笛卡尔积的映射关系。 
x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])
#np,ix_([1,5,7,2],[0,3,1,2]) 生成了两个数组的笛卡尔积索引，第一个数组表示行索引，第二个数组表示列索引。 