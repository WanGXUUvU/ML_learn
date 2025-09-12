import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('./power_predict/data/test.csv')

print(df.isnull().sum())  #查看每列缺失值数量

df.ffill(inplace=True)  #前向填充

print(df.isnull().sum())  #查看每列缺失值数量
plt.plot(df['timestamp'], df['Active_Power'])
plt.title('Active_Power')
plt.xlabel('timestamp')
plt.ylabel('Active Power')
plt.legend()
plt.grid(True)
plt.show()
