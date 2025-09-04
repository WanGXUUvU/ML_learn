import numpy as np
dt=np.dtype(np.int32)
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
print(dt)
# ([<dtype>])表示接收一个列表,(<dtype1>,<dtype2>)表示一个元组,第一个元素表示元素的名字，第二个元素表示数据类型
dt1=np.dtype([('age',np.int8)])
student=np.array([(10,),(20,)],dtype=dt1)
print(student['age'])
print(student)

student1 = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student1) 
print(a['name'])