import pandas as pd
df=pd.read_csv('./power_predict/data/test.csv')
df.dropna(inplace=True)  #删除缺失值 inplace=True表示在原数据上修改
df.to_csv('cleaned_output.csv',index=False)  #保存清洗后的数据
print(df.isnull().sum())  #查看每列缺失值数量
df1=pd.read_csv('./power_predict/data/test.csv')
print(df1.isnull().sum())  #查看每列缺失值数量
#按平均值填充
df1.fillna(df1.mean(numeric_only=True),inplace=True)
print(df1.isnull().sum())  #查看每列缺失值数量
df1.duplicated().sum()
df1.drop_duplicates(inplace=False)