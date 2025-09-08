import pandas as pd
#读取CSV文件
df=pd.read_csv('./power_predict/data/test.csv')
print(df.head(10))  #查看前十行数据
print(df.tail(10))  #查看后十行数据
print(df.info())  #查看数据的基本信息