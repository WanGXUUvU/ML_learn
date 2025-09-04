import pandas as pd
#创建DataFrame对象
data={'name':['Tom','Jack','Steve','Ricky'],'age':[28,34,29,42],
      'Age':[28,34,29,42],
      'city':['Beijing','Shanghai','Guangzhou','Shenzhen']}
df=pd.DataFrame(data)

#查看前两行数据
print(df.head(2))

#查看数据的基本信息
print(df.info())

#描述性统计
print(df.describe())

#按年龄排序
df_sorted=df.sort_values(by='age')

print(df_sorted)

#按索引选择行 
print(df.loc[0:1])  #选择前两行
print("按城市排序---------------")
#按城市分组，计算每个城市的平均年龄
print(df.groupby('city')['age'].mean())

df['Age']=df['Age'].fillna(df['Age'].mean())
print(df)

df.to_csv('output.csv',index=False)  #保存为CSV文件