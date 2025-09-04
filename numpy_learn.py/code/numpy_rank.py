import numpy as np
a=np.arange(24)
print(a)
b=a.reshape(2,4,3)
print(b)
print(b.shape)
print(b.ndim)

 
c= np.array([[1,2,3],[4,5,6]]) 
c.shape =  (3,2)  
print (c)